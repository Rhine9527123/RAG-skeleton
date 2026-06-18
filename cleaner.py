"""
cleaner.py — LLM 驱动的内容清洗管线
====================================

三步清洗流水线：
  1. 去重：精确去重（content_hash）+ 近似去重（标题相似度）
  2. 质量过滤：长度、中文占比、噪声模式
  3. LLM 相关性评分：领域相关度 0-10，低于阈值丢弃

领域关键词和评分提示词由 config.py 统一管理，换领域时自动切换。

LLM 支持：
  - DeepSeek API（在线，质量高）
  - Ollama 本地（离线，免费）
  - 任一 OpenAI 兼容 API

依赖：pip install requests
"""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

import requests

# 导入中心化配置
from config import get_config

logger = logging.getLogger("cleaner")

# ============================================================
# LLM 客户端
# ============================================================

@dataclass
class LLMConfig:
    """LLM 配置"""
    provider: str = "deepseek"         # deepseek | ollama | openai_compatible
    api_key: str = ""
    api_base: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    temperature: float = 0.0           # 0 = 最确定，适合评分任务
    max_tokens: int = 10               # 评分只需要一个数字
    timeout: int = 30

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """从环境变量自动检测 LLM 配置"""
        # 检查 Ollama
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            r = requests.get(f"{ollama_url}/api/tags", timeout=3)
            if r.status_code == 200:
                return cls(
                    provider="ollama",
                    api_key="ollama",
                    api_base=f"{ollama_url}/v1",
                    model=os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"),
                )
        except Exception:
            pass

        # 默认 DeepSeek
        return cls(
            provider="deepseek",
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            api_base="https://api.deepseek.com/v1",
            model="deepseek-chat",
        )


class LLMClient:
    """轻量 LLM 调用客户端（OpenAI 兼容 API）"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()

    def chat(self, messages: list[dict], **overrides) -> str:
        """发送对话请求，返回文本回复"""
        cfg = self.config
        payload = {
            "model": cfg.model,
            "messages": messages,
            "temperature": overrides.get("temperature", cfg.temperature),
            "max_tokens": overrides.get("max_tokens", cfg.max_tokens),
        }

        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        }

        resp = requests.post(
            f"{cfg.api_base}/chat/completions",
            json=payload,
            headers=headers,
            timeout=cfg.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


# ============================================================
# 清洗器
# ============================================================

# 领域相关关键词（从配置加载，用于质量预检/LLM降级兜底）
_DOMAIN_KEYWORDS = None  # 延迟加载

def _get_keywords():
    global _DOMAIN_KEYWORDS
    if _DOMAIN_KEYWORDS is None:
        cfg = get_config()
        _DOMAIN_KEYWORDS = cfg.domain_keywords or []
    return _DOMAIN_KEYWORDS


class ArticleCleaner:
    """
    财经文章清洗器

    流水线：去重 → 质量 → LLM评分

    用法：
        cleaner = ArticleCleaner()
        clean_articles = cleaner.clean(raw_articles, min_llm_score=5)
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm = LLMClient(llm_config) if llm_config or True else None
        self._llm_config = llm_config

    # ── 第一步：去重 ─────────────────────────────────

    def deduplicate(self, articles: list) -> list:
        """
        两级去重：
          1. 精确去重（content_hash）
          2. 标题相似度去重（阈值 0.85）
        """
        if not articles:
            return []

        # 第一遍：精确去重
        seen_hashes = set()
        exact_deduped = []
        for a in articles:
            h = getattr(a, "content_hash", "") or hashlib.md5(
                (getattr(a, "title", "") + getattr(a, "content", "")[:200]).encode()
            ).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                exact_deduped.append(a)

        if len(exact_deduped) <= 1:
            return exact_deduped

        # 第二遍：标题相似度去重
        result = [exact_deduped[0]]
        for a in exact_deduped[1:]:
            title_a = getattr(a, "title", "")
            is_dup = False
            for kept in result:
                title_b = getattr(kept, "title", "")
                sim = SequenceMatcher(None, title_a, title_b).ratio()
                if sim > 0.85:
                    is_dup = True
                    break
            if not is_dup:
                result.append(a)

        dropped = len(exact_deduped) - len(result)
        logger.info(f"[清洗] 去重: {len(articles)} → {len(exact_deduped)} → {len(result)} (丢弃 {dropped} 近似重复)")
        return result

    # ── 第二步：质量预检 ──────────────────────────────

    def quality_check(self, article) -> tuple[bool, str]:
        """
        快速质量检查（不调 LLM，零延迟）

        返回 (通过?, 原因)
        """
        content = getattr(article, "content", "")
        title = getattr(article, "title", "")

        # 1. 长度检查
        if len(content) < 15:
            return False, "内容过短 (<15字)"

        # 2. 标题质量
        if len(title) < 2:
            return False, "标题过短"

        # 3. 中文占比（至少 30% 是中文字符）
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", content))
        total_chars = len(content.replace(" ", "").replace("\n", ""))
        if total_chars > 0 and chinese_chars / total_chars < 0.3:
            return False, f"中文占比过低 ({chinese_chars}/{total_chars})"

        # 4. 噪声模式：纯数字/纯符号
        meaningful = re.sub(r"[\d\s.,;:!?，。；：！？""''【】《》（）()\[\]]+", "", content)
        if len(meaningful) < 10:
            return False, "有效文字过少"

        return True, "OK"

    def filter_quality(self, articles: list) -> list:
        """批量质量过滤"""
        passed = []
        for a in articles:
            ok, reason = self.quality_check(a)
            if ok:
                passed.append(a)
            else:
                logger.debug(f"[清洗] 质量过滤: {reason} — {getattr(a, 'title', '')[:40]}")
        dropped = len(articles) - len(passed)
        logger.info(f"[清洗] 质量过滤: {len(articles)} → {len(passed)} (丢弃 {dropped})")
        return passed

    # ── 第三步：LLM 相关性评分 ────────────────────────

    SCORE_PROMPT = None  # 从 config 延迟加载

    def _get_score_prompt(self):
        if self.SCORE_PROMPT is None:
            cfg = get_config()
            # 将类属性设为字符串（避免每次查 config）
            type(self).SCORE_PROMPT = cfg.scoring_prompt
        return self.SCORE_PROMPT

    def _score_single(self, article) -> int:
        """单篇文章 LLM 评分"""
        content = getattr(article, "content", "")
        title = getattr(article, "title", "")

        # 截断过长内容（节省 token）
        content_snippet = content[:800]

        prompt = self._get_score_prompt().format(title=title, content=content_snippet)
        messages = [{"role": "user", "content": prompt}]

        try:
            reply = self.llm.chat(messages, temperature=0.0, max_tokens=10)
            # 提取数字
            match = re.search(r"\d+", reply)
            if match:
                score = int(match.group())
                return max(0, min(10, score))
            logger.warning(f"[清洗] LLM 返回非数字: {reply[:50]}")
            return 0
        except Exception as e:
            logger.error(f"[清洗] LLM 评分失败: {e}")
            # 失败时用关键词兜底
            return self._keyword_score(article)

    def _keyword_score(self, article) -> int:
        """关键词兜底评分（LLM 不可用时）"""
        content = getattr(article, "content", "")
        title = getattr(article, "title", "")
        text = title + content[:500]

        hits = sum(1 for kw in _get_keywords() if kw in text)
        # 每命中 1 个关键词 ≈ 0.5 分，上限 8 分
        score = min(8, hits // 2)
        return score

    def filter_llm(self, articles: list, min_score: int = 5) -> list:
        """
        LLM 相关性过滤

        Args:
            articles: 待评分文章
            min_score: 最低分数阈值（0-10），低于此分数的丢弃

        Returns:
            评分 >= min_score 的文章（附加 score 属性）
        """
        passed = []
        for i, a in enumerate(articles):
            title = getattr(a, "title", "")[:50]
            logger.info(f"[清洗] LLM评分 ({i+1}/{len(articles)}): {title}")

            score = self._score_single(a)
            # 把分数附加到对象上
            setattr(a, "relevance_score", score)

            if score >= min_score:
                passed.append(a)
                logger.info(f"        分数={score} ✓ 保留")
            else:
                logger.info(f"        分数={score} ✗ 丢弃")

            # 速率限制：避免 API 限流
            if i < len(articles) - 1:
                time.sleep(0.3)

        dropped = len(articles) - len(passed)
        logger.info(f"[清洗] LLM过滤: {len(articles)} → {len(passed)} (丢弃 {dropped}, 阈值≥{min_score})")
        return passed

    # ── 完整流水线 ───────────────────────────────────

    def clean(
        self,
        articles: list,
        min_llm_score: int = 5,
        skip_llm: bool = False,
    ) -> list:
        """
        完整清洗流水线

        Args:
            articles: 原始文章列表
            min_llm_score: LLM 最低分数
            skip_llm: 跳过 LLM 评分（仅做去重+质量过滤，快速模式）

        Returns:
            清洗后的文章列表
        """
        logger.info(f"[清洗] 开始: {len(articles)} 篇原始文章")

        # Step 1: 去重
        articles = self.deduplicate(articles)

        # Step 2: 质量预检
        articles = self.filter_quality(articles)

        # Step 3: LLM 评分过滤
        if not skip_llm:
            articles = self.filter_llm(articles, min_score=min_llm_score)
        else:
            # 快速模式：关键词兜底
            logger.info("[清洗] 跳过 LLM，使用关键词兜底")
            passed = []
            for a in articles:
                score = self._keyword_score(a)
                setattr(a, "relevance_score", score)
                if score >= min_llm_score:
                    passed.append(a)
            articles = passed
            logger.info(f"[清洗] 关键词过滤: {len(articles)} 篇通过 (阈值≥{min_llm_score})")

        logger.info(f"[清洗] 完成: {len(articles)} 篇通过")
        return articles


# ============================================================
# 便捷函数
# ============================================================

def clean_articles(articles: list, min_score: int = 5, skip_llm: bool = False) -> list:
    """快捷函数：清洗文章列表"""
    cleaner = ArticleCleaner()
    return cleaner.clean(articles, min_llm_score=min_score, skip_llm=skip_llm)


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    print("=" * 60)
    print("内容清洗器 — 测试运行")
    print("=" * 60)

    # 模拟数据
    from dataclasses import dataclass as dc

    @dc
    class MockArticle:
        title: str = ""
        content: str = ""
        content_hash: str = ""
        source: str = "test"

    test_articles = [
        MockArticle(
            title="央行宣布降准0.5个百分点，释放长期资金约1万亿",
            content="中国人民银行决定于2026年6月15日下调金融机构存款准备金率0.5个百分点..."
        ),
        MockArticle(
            title="央行宣布降准0.5个百分点，释放长期资金约1万亿",  # 近似重复
            content="中国人民银行决定下调存款准备金率0.5个百分点，释放资金约1万亿..."
        ),
        MockArticle(
            title="某明星参加综艺节目",
            content="近日，著名影星张某参加了某卫视的综艺节目录制，现场气氛热烈..."
        ),
        MockArticle(
            title="A股三大指数集体收涨，沪指重返3400点",
            content="今日A股三大指数集体收涨。截至收盘，沪指涨1.2%报3415点..."
        ),
        MockArticle(
            title="test",         # 质量过滤
            content="12",
        ),
    ]

    cleaner = ArticleCleaner()
    result = cleaner.clean(test_articles, min_llm_score=5)

    print(f"\n清洗结果: {len(result)}/{len(test_articles)} 篇通过\n")
    for a in result:
        score = getattr(a, "relevance_score", "?")
        print(f"  [{score}/10] {a.title[:50]}")
