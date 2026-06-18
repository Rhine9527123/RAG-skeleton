"""
crawler.py — 多源内容爬虫（多API灾冗设计）
=========================================

设计理念：
  1. 多个数据源按优先级排列，一个挂了自动切换下一个
  2. trafilatura 做正文提取（去广告、去导航、只留正文）
  3. 每次爬取产出标准化 Article 对象
  4. 内置速率限制，不当恶意爬虫

扩展方式：继承 SourceBase，实现 fetch() 即可新增数据源。

领域切换：数据源配置在 sources.json 中管理，可随领域更换。

依赖：pip install requests trafilatura
"""

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional

import requests
import trafilatura

logger = logging.getLogger("crawler")

# ============================================================
# 北京时间
# ============================================================
CST = timezone(timedelta(hours=8))

def now_cst() -> str:
    return datetime.now(CST).isoformat()


# ============================================================
# 数据模型
# ============================================================

@dataclass
class Article:
    """标准化的文章对象"""
    title: str
    content: str
    source: str
    url: str = ""
    published: Optional[str] = None
    crawled_at: str = field(default_factory=now_cst)
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash and self.content:
            self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        return hashlib.md5(
            (self.title + self.content[:500]).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict:
        d = {
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "url": self.url,
            "published": self.published,
            "crawled_at": self.crawled_at,
            "content_hash": self.content_hash,
        }
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ============================================================
# 数据源基类
# ============================================================

class SourceBase:
    """数据源基类 —— 子类只需实现 fetch()"""
    name: str = "base"
    timeout: int = 15
    max_retries: int = 2
    _session: Optional[requests.Session] = None

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            })
        return self._session

    def fetch(self) -> list[Article]:
        """子类实现：返回 Article 列表"""
        raise NotImplementedError(f"{self.name}.fetch() not implemented")

    def _get(self, url: str, **kwargs) -> requests.Response:
        """带重试的 GET 请求"""
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(url, timeout=self.timeout, **kwargs)
                resp.raise_for_status()
                return resp
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        f"[{self.name}] 请求失败，{wait}s 后重试 ({attempt+1}/{self.max_retries}): {url}"
                    )
                    time.sleep(wait)
        raise last_exc  # type: ignore

    def extract_text(self, html: str, url: str = "") -> str:
        """用 trafilatura 从 HTML 中提取正文"""
        text = trafilatura.extract(
            html,
            url=url,
            include_links=False,
            include_images=False,
            include_tables=False,
            favor_precision=True,
            output_format="txt",
        )
        return (text or "").strip()

    @staticmethod
    def clean_text(text: str) -> str:
        """清理多余空白"""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()


# ============================================================
# 具体数据源实现
# ============================================================

class ClsTelegraphSource(SourceBase):
    """
    财联社电报 — 实时财经快讯
    API: https://www.cls.cn/api/sw
    每条约 50-200 字，适合做实时知识补充
    """
    name = "财联社电报"

    def fetch(self) -> list[Article]:
        articles = []
        url = "https://www.cls.cn/api/sw"
        params = {
            "app": "CailianpressWeb",
            "os": "web",
            "sv": "8.4.6",
            "type": "telegram",
            "page": "1",
            "rn": "20",
        }
        resp = self._get(url, params=params)
        data = resp.json()

        items = data.get("data", {}).get("roll_data", []) or data.get("data", [])
        for item in items:
            title = (item.get("title") or item.get("brief") or "").strip()
            content = (item.get("content") or item.get("brief") or title).strip()
            if not title or len(content) < 10:
                continue

            # 清理 HTML 标签
            content = re.sub(r"<[^>]+>", "", content)
            content = self.clean_text(content)

            articles.append(Article(
                title=title,
                content=content,
                source=self.name,
                url=item.get("shareurl", ""),
                published=str(item.get("ctime", "")),
            ))

        logger.info(f"[{self.name}] 获取 {len(articles)} 条快讯")
        return articles


class WallstreetLiveSource(SourceBase):
    """
    华尔街见闻 — 实时直播
    API: https://api-one.wallstcn.com/apiv1/content/lives
    全球宏观 + A 股 + 美股快讯
    """
    name = "华尔街见闻"

    def fetch(self) -> list[Article]:
        articles = []
        url = "https://api-one.wallstcn.com/apiv1/content/lives"
        params = {
            "channel": "global-channel",
            "client": "pc",
            "limit": 20,
            "first_page": "true",
        }
        resp = self._get(url, params=params)
        data = resp.json()

        items = data.get("data", {}).get("items", [])
        for item in items:
            raw = item.get("resource", {})
            title = (raw.get("title") or "").strip()
            content = (raw.get("content_text") or raw.get("content") or "")
            content = re.sub(r"<[^>]+>", "", content).strip()

            if not content or len(content) < 10:
                continue
            if not title:
                title = content[:50]

            articles.append(Article(
                title=title,
                content=self.clean_text(content),
                source=self.name,
                url=raw.get("uri", ""),
                published=str(raw.get("display_time", "")),
            ))

        logger.info(f"[{self.name}] 获取 {len(articles)} 条快讯")
        return articles


class SinaFinanceSource(SourceBase):
    """
    新浪财经 — 滚动新闻
    使用 RSS 作为备选（更稳定），HTML 作为主方案
    """
    name = "新浪财经"

    def fetch(self) -> list[Article]:
        articles = []
        urls = [
            # 新浪财经 - 宏观新闻
            "https://finance.sina.com.cn/money/future/roll/index.d.json",
            # 新浪财经 - 股票新闻
            "https://finance.sina.com.cn/stock/roll/index.d.json",
        ]

        for target_url in urls:
            try:
                resp = self._get(target_url)
                data = resp.json()
                # 新浪滚动新闻 JSON 格式：{"result": {"data": [...]}}
                items = data.get("result", {}).get("data", [])
                for item in items:
                    title = (item.get("title") or "").strip()
                    article_url = item.get("url", "")
                    ctime = item.get("ctime", "")

                    if not title:
                        continue

                    # 尝试获取正文
                    content = ""
                    if article_url:
                        try:
                            page = self._get(article_url)
                            content = self.extract_text(page.text, article_url)
                        except Exception:
                            content = title  # 正文获取失败，用标题兜底

                    if not content:
                        content = title

                    articles.append(Article(
                        title=title,
                        content=self.clean_text(content),
                        source=self.name,
                        url=article_url,
                        published=ctime,
                    ))

                logger.info(f"[{self.name}] 从 {target_url} 获取 {len(articles)} 条")
            except Exception as e:
                logger.warning(f"[{self.name}] 接口失败: {target_url} — {e}")

        return articles


class EastMoneyKuaixunSource(SourceBase):
    """
    东方财富 — 7×24 快讯
    不需要 API Key，但接口偶尔会变参数要求
    """
    name = "东方财富快讯"

    def fetch(self) -> list[Article]:
        articles = []
        url = "https://push2ex.eastmoney.com/getAllIcsNews"
        params = {
            "type": "kuaixun",
            "page": "1",
            "pagesize": "20",
            "ut": "7eea3edcaed734be",
            "callback": "",
            "_": str(int(time.time() * 1000)),
        }
        try:
            resp = self._get(url, params=params)
            text = resp.text
            # 移除 JSONP callback 包装（如果有）
            if text.startswith("jQuery"):
                text = re.sub(r"^jQuery\d+_\d+\(|\);?$", "", text)
            data = json.loads(text)

            items = data.get("data", {}).get("list", []) or data.get("result", {}).get("data", [])
            for item in items:
                title = (item.get("title") or "").strip()
                content = (item.get("digest") or title).strip()
                if not title or len(content) < 5:
                    continue
                content = re.sub(r"<[^>]+>", "", content)
                articles.append(Article(
                    title=title,
                    content=self.clean_text(content),
                    source=self.name,
                    url=f"https://finance.eastmoney.com/a/{item.get('code', '')}.html",
                    published=str(item.get("showtime", "") or item.get("tdate", "")),
                ))

            logger.info(f"[{self.name}] 获取 {len(articles)} 条快讯")
        except Exception as e:
            logger.warning(f"[{self.name}] 接口失败: {e}")

        return articles


class NeteaseMoneySource(SourceBase):
    """
    网易财经 — 精选热点
    首页 HTML 抓取 + trafilatura 提取
    """
    name = "网易财经"

    def fetch(self) -> list[Article]:
        articles = []
        url = "https://money.163.com/"
        try:
            resp = self._get(url)
            # trafilatura 也能提取链接列表
            extracted = trafilatura.extract(
                resp.text,
                url=url,
                output_format="xml",
                include_links=True,
            )

            # 走回退：提取页面中的链接然后逐个抓
            # 网易首页动态加载较多，直接用 RSS 风格链接
            rss_url = "https://money.163.com/special/00251G8F/news_json.js"
            try:
                r = self._get(rss_url)
                text = r.text
                if text.startswith("var "):
                    text = text.split("=", 1)[1].strip().rstrip(";")
                data = json.loads(text)
                items = data.get("news", []) or data.get("list", [])
                for item in items[:20]:
                    title = item.get("title", "").strip()
                    article_url = item.get("docurl", "") or item.get("url", "")
                    digest = item.get("digest", "").strip()
                    if title and article_url:
                        content = digest
                        if not content:
                            try:
                                p = self._get(article_url)
                                content = self.extract_text(p.text, article_url)
                            except Exception:
                                content = title
                        articles.append(Article(
                            title=title,
                            content=self.clean_text(content),
                            source=self.name,
                            url=article_url,
                            published=str(item.get("time", "")),
                        ))
                logger.info(f"[{self.name}] 从 JS 接口获取 {len(articles)} 条")
            except Exception as e:
                logger.warning(f"[{self.name}] JS 接口失败: {e}")

        except Exception as e:
            logger.warning(f"[{self.name}] 抓取失败: {e}")

        return articles


# ============================================================
# 多源爬虫（灾冗核心）
# ============================================================

class MultiSourceCrawler:
    """
    多源财经爬虫 — 灾冗设计

    按优先级依次尝试各数据源，一个失败自动切换下一个。
    只要有一个源成功返回数据，爬取就视为成功。
    所有源都失败才报错。

    用法：
        crawler = MultiSourceCrawler()
        articles = crawler.crawl()  # -> list[Article]
    """

    def __init__(self, min_articles: int = 5):
        """
        Args:
            min_articles: 最少需要多少条才停止尝试后续源
                         设大一点可以获取更全面的数据
                         设 0 则每个源都试一遍（全面采集）
        """
        self.min_articles = min_articles
        # 按优先级排列：API 源优先（快 + 结构化），HTML 源兜底
        self.sources: list[SourceBase] = [
            ClsTelegraphSource(),       # 财联社 — 最稳定的实时快讯 API
            WallstreetLiveSource(),     # 华尔街见闻 — 全球宏观视
            EastMoneyKuaixunSource(),   # 东方财富 — A 股快讯
            SinaFinanceSource(),        # 新浪财经 — 滚动新闻
            NeteaseMoneySource(),       # 网易财经 — 精选热点（HTML 兜底）
        ]

    def crawl(self, source_names: Optional[list[str]] = None) -> list[Article]:
        """
        执行爬取，逐源尝试。

        Args:
            source_names: 指定要用的源名称列表（None = 全部）
                         如 ["财联社电报", "华尔街见闻"]

        Returns:
            去重后的 Article 列表
        """
        all_articles: list[Article] = []
        failed_sources: list[str] = []
        active_sources = self.sources
        if source_names:
            active_sources = [s for s in self.sources if s.name in source_names]

        for source in active_sources:
            try:
                logger.info(f"[多源爬虫] 尝试数据源: {source.name}")
                articles = source.fetch()

                if articles:
                    all_articles.extend(articles)
                    logger.info(
                        f"[多源爬虫] {source.name} 成功: +{len(articles)} 条 "
                        f"(累计 {len(all_articles)})"
                    )

                    # 够了就停（但仍然会把当前源的结果全收进来）
                    if self.min_articles > 0 and len(all_articles) >= self.min_articles:
                        break
                else:
                    logger.warning(f"[多源爬虫] {source.name} 返回 0 条，继续下一个源")

            except Exception as e:
                failed_sources.append(f"{source.name}: {e}")
                logger.error(f"[多源爬虫] {source.name} 失败 -> {e}")
                continue

        if not all_articles and failed_sources:
            raise RuntimeError(
                f"所有数据源均失败:\n" + "\n".join(f"  - {f}" for f in failed_sources)
            )

        # 去重（基于 content_hash）
        seen = set()
        unique = []
        for a in all_articles:
            if a.content_hash not in seen:
                seen.add(a.content_hash)
                unique.append(a)

        logger.info(
            f"[多源爬虫] 完成: {len(unique)} 条去重后 "
            f"(原始 {len(all_articles)}, 失败源 {len(failed_sources)})"
        )
        return unique

    def crawl_to_json(self, **kwargs) -> str:
        """爬取并返回 JSON 字符串"""
        articles = self.crawl(**kwargs)
        return json.dumps(
            [a.to_dict() for a in articles],
            ensure_ascii=False,
            indent=2,
        )

    def crawl_to_files(self, output_dir: str = "data", **kwargs) -> list[str]:
        """
        爬取并保存为 .txt 文件（供 RAG 索引）

        Returns:
            写入的文件路径列表
        """
        articles = self.crawl(**kwargs)
        os.makedirs(output_dir, exist_ok=True)
        saved = []

        for i, article in enumerate(articles):
            # 安全文件名
            safe_title = re.sub(r'[\\/*?:"<>|]', "_", article.title)[:60]
            filename = f"crawled_{article.source}_{i:03d}_{safe_title}.txt"
            filepath = os.path.join(output_dir, filename)

            content = (
                f"标题: {article.title}\n"
                f"来源: {article.source}\n"
                f"时间: {article.published or '未知'}\n"
                f"链接: {article.url}\n"
                f"{'─' * 40}\n"
                f"{article.content}\n"
            )

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            saved.append(filepath)

        logger.info(f"[多源爬虫] 保存 {len(saved)} 个文件到 {output_dir}/")
        return saved


# ============================================================
# 便捷函数
# ============================================================

_crawler: Optional[MultiSourceCrawler] = None


def get_crawler(min_articles: int = 5) -> MultiSourceCrawler:
    """获取全局爬虫实例（单例）"""
    global _crawler
    if _crawler is None:
        _crawler = MultiSourceCrawler(min_articles=min_articles)
    return _crawler


def crawl_news(min_articles: int = 10) -> list[Article]:
    """快捷函数：爬取最新财经新闻"""
    return get_crawler(min_articles).crawl()


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    print("=" * 60)
    print("多源财经爬虫 — 测试运行")
    print("=" * 60)

    crawler = MultiSourceCrawler(min_articles=10)
    articles = crawler.crawl()

    print(f"\n共获取 {len(articles)} 篇文章:\n")
    for i, a in enumerate(articles, 1):
        print(f"{i}. [{a.source}] {a.title[:60]}")
        print(f"   {a.content[:80]}...")
        print()
