"""
pipeline.py — 动态知识库更新流水线
===================================

爬取 → 清洗 → 去重 → LLM过滤 → 入库 → (可选)重建索引

集成现有模块：
  - crawler.py: 多源内容爬虫
  - cleaner.py: LLM 驱动的清洗管线
  - dedup.py:   URL + 内容双重去重（与 server.py 共享同一个 dedup.db）

用法：
  python pipeline.py              # 运行一次完整流水线
  python pipeline.py --quick      # 快速模式（跳过 LLM 过滤）
  python pipeline.py --sources 源A,源B  # 只爬指定源

定时运行（cron / Windows 任务计划程序）：
  每 30 分钟:  python pipeline.py
  每小时:      python pipeline.py --quick
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

# 将项目根目录加入 sys.path（兼容从任意目录运行）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from crawler import MultiSourceCrawler, Article
from cleaner import ArticleCleaner, clean_articles
from dedup import DedupManager

logger = logging.getLogger("pipeline")

CST = timezone(timedelta(hours=8))

# ============================================================
# 配置
# ============================================================

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CRAWLED_DIR = os.path.join(DATA_DIR, "crawled")       # 爬取文章存放子目录
PIPELINE_LOG_DIR = os.path.join(PROJECT_ROOT, ".pipeline_logs")

# 默认每次运行最多保留的文章数（避免 data/ 目录无限膨胀）
MAX_CRAWLED_FILES = 200


def load_pipeline_config() -> dict:
    """从 config.json 加载 pipeline 配置（可选）"""
    config_path = os.path.join(PROJECT_ROOT, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f).get("pipeline", {})
        except Exception:
            pass
    return {}


# ============================================================
# 流水线
# ============================================================

class KnowledgePipeline:
    """
    动态知识库更新流水线

    流程:
      1. 爬取:     多源内容（自动灾冗切换）
      2. 清洗:     去重 + 质量过滤 + LLM 相关性评分
      3. 入库:     保存到 data/crawled/ 目录
      4. 去重检查:  与已有文档对比（dedup.db）
      5. 报告:     输出本次更新摘要
    """

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.crawled_dir = CRAWLED_DIR
        self.dedup = DedupManager(os.path.join(PROJECT_ROOT, "dedup.db"))
        self.crawler = MultiSourceCrawler(min_articles=10)
        self.cleaner = ArticleCleaner()

        os.makedirs(self.crawled_dir, exist_ok=True)
        os.makedirs(PIPELINE_LOG_DIR, exist_ok=True)

    def run(
        self,
        sources: Optional[list[str]] = None,
        min_llm_score: int = 5,
        skip_llm: bool = False,
        rebuild: bool = False,
    ) -> dict:
        """
        执行一次完整流水线

        Args:
            sources:       指定数据源（None=全部）
            min_llm_score: LLM 最低分数阈值
            skip_llm:      跳过 LLM 过滤（快速模式）
            rebuild:       是否触发 RAG 索引重建（需 server.py 运行中）

        Returns:
            {"new": int, "duplicates": int, "errors": int, "articles": [...]}
        """
        start_time = time.time()
        report = {
            "pipeline_start": datetime.now(CST).isoformat(),
            "sources_used": sources or "全部",
            "skip_llm": skip_llm,
            "min_llm_score": min_llm_score,
            "raw_count": 0,
            "cleaned_count": 0,
            "new_files": 0,
            "duplicate_files": 0,
            "errors": 0,
            "articles": [],
        }

        logger.info("=" * 60)
        logger.info("[流水线] 开始执行")
        logger.info("=" * 60)

        # ── Step 1: 爬取 ───────────────────────────
        logger.info("[流水线] Step 1/4 — 多源爬取")
        try:
            raw_articles = self.crawler.crawl(source_names=sources)
            report["raw_count"] = len(raw_articles)
            logger.info(f"[流水线] 爬取完成: {len(raw_articles)} 篇原始文章")
        except Exception as e:
            logger.error(f"[流水线] 爬取失败: {e}")
            report["errors"] += 1
            return report

        if not raw_articles:
            logger.warning("[流水线] 未获取到任何文章，流水线终止")
            return report

        # ── Step 2: 清洗 ───────────────────────────
        logger.info("[流水线] Step 2/4 — 清洗过滤")
        try:
            clean_result = self.cleaner.clean(
                raw_articles,
                min_llm_score=min_llm_score,
                skip_llm=skip_llm,
            )
            report["cleaned_count"] = len(clean_result)
            logger.info(f"[流水线] 清洗完成: {len(clean_result)} 篇通过")
        except Exception as e:
            logger.error(f"[流水线] 清洗失败: {e}")
            report["errors"] += 1
            return report

        # ── Step 3: 入库 + 去重检查 ────────────────
        logger.info("[流水线] Step 3/4 — 入库去重")
        for article in clean_result:
            try:
                filename = self._save_article(article)

                # dedup.db 去重检查
                content = self._article_to_text(article)
                url = getattr(article, "url", "")
                is_dup, reason, _ = self.dedup.check(
                    content=content,
                    url=url,
                    filename=filename,
                )

                if is_dup:
                    # 已存在，删除刚保存的文件
                    os.remove(os.path.join(self.crawled_dir, filename))
                    report["duplicate_files"] += 1
                    logger.debug(f"[流水线] 重复跳过: {reason} — {article.title[:40]}")
                else:
                    # 新内容，写入去重记录
                    filepath = os.path.join(self.crawled_dir, filename)
                    file_size = os.path.getsize(filepath)
                    self.dedup.add(
                        content=content,
                        url=url,
                        filename=filename,
                        file_size=file_size,
                    )
                    report["new_files"] += 1
                    report["articles"].append({
                        "title": article.title,
                        "source": article.source,
                        "score": getattr(article, "relevance_score", None),
                        "file": filename,
                        "url": url,
                    })

            except Exception as e:
                logger.error(f"[流水线] 保存文章失败: {e}")
                report["errors"] += 1

        logger.info(
            f"[流水线] 入库完成: {report['new_files']} 新增, "
            f"{report['duplicate_files']} 重复跳过"
        )

        # ── Step 4: 可选 — 重建索引 ───────────────
        if rebuild and report["new_files"] > 0:
            logger.info("[流水线] Step 4/4 — 触发 RAG 索引重建")
            self._trigger_rebuild(report["new_files"])
        else:
            if report["new_files"] > 0:
                logger.info(
                    "[流水线] Step 4/4 — 跳过索引重建 "
                    "(新增文件已保存到 data/crawled/，重启 server.py 后自动索引)"
                )

        # ── 清理旧文件 ─────────────────────────────
        self._rotate_files()

        # ── 保存报告 ───────────────────────────────
        report["pipeline_end"] = datetime.now(CST).isoformat()
        report["elapsed_seconds"] = round(time.time() - start_time, 1)
        self._save_report(report)

        logger.info("=" * 60)
        logger.info(
            f"[流水线] 完成! 新增 {report['new_files']} 篇, "
            f"重复 {report['duplicate_files']} 篇, "
            f"耗时 {report['elapsed_seconds']}秒"
        )
        logger.info("=" * 60)

        return report

    # ── 内部方法 ───────────────────────────────────

    def _save_article(self, article: Article) -> str:
        """保存文章为 .txt 文件，返回文件名"""
        # 安全文件名：前缀 + 来源 + 标题截断
        safe_source = article.source.replace("/", "_")[:20]
        safe_title = article.title[:60]
        # 移除 Windows 非法字符
        safe_title = "".join(c for c in safe_title if c not in r'\/:*?"<>|')
        timestamp = datetime.now(CST).strftime("%Y%m%d_%H%M%S")
        filename = f"crawled_{timestamp}_{safe_source}_{safe_title}.txt"

        content = self._article_to_text(article)
        filepath = os.path.join(self.crawled_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return filename

    @staticmethod
    def _article_to_text(article: Article) -> str:
        """格式化文章为纯文本"""
        score = getattr(article, "relevance_score", None)
        score_line = f"相关度评分: {score}/10\n" if score is not None else ""
        return (
            f"标题: {article.title}\n"
            f"来源: {article.source}\n"
            f"时间: {article.published or '未知'}\n"
            f"链接: {article.url}\n"
            f"{score_line}"
            f"{'─' * 50}\n"
            f"{article.content}\n"
        )

    def _trigger_rebuild(self, new_count: int):
        """
        触发 RAG 索引重建。
        尝试调用 server.py 的 /upload 接口重建索引。
        如果 server 没运行，给出提示。
        """
        try:
            import urllib.request

            # server.py 没有专门的 /rebuild 端点，
            # 但 upload 一个空文件可以触发 _rebuild_index()
            # 更简单的方式：直接提示用户重启
            health_url = "http://localhost:8000/health"
            req = urllib.request.Request(health_url, method="GET")
            resp = urllib.request.urlopen(req, timeout=5)
            if resp.status == 200:
                logger.info(
                    f"[流水线] server.py 正在运行。"
                    f"新增 {new_count} 个文件已保存到 data/crawled/。"
                )
                logger.info(
                    "[流水线] 请重启 server.py 以重建索引，"
                    "或使用 API 重新上传文件触发重建。"
                )
            else:
                self._rebuild_hint()
        except Exception:
            self._rebuild_hint()

    @staticmethod
    def _rebuild_hint():
        logger.info("[流水线] server.py 未运行，文件已保存。")
        logger.info("[流水线] 下次启动 server.py 时会自动重建索引。")

    def _rotate_files(self):
        """
        清理旧爬取文件，保留最近 MAX_CRAWLED_FILES 个。
        避免 data/ 目录无限膨胀。
        """
        files = sorted(
            Path(self.crawled_dir).glob("crawled_*.txt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if len(files) > MAX_CRAWLED_FILES:
            for old_file in files[MAX_CRAWLED_FILES:]:
                try:
                    old_file.unlink()
                except Exception:
                    pass
            logger.info(
                f"[流水线] 清理旧文件: {len(files) - MAX_CRAWLED_FILES} 个"
            )

    def _save_report(self, report: dict):
        """保存流水线运行报告（JSON）"""
        timestamp = datetime.now(CST).strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(PIPELINE_LOG_DIR, f"pipeline_{timestamp}.json")
        try:
            # 精简 articles 列表（只保留摘要）
            slim_report = {**report}
            slim_report["articles"] = [
                {"title": a["title"][:60], "source": a["source"], "score": a["score"]}
                for a in report.get("articles", [])[:20]
            ]
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(slim_report, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[流水线] 保存报告失败: {e}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="动态知识库更新流水线 — 爬取财经新闻并清洗入库",
    )
    parser.add_argument(
        "--sources", type=str, default=None,
        help="指定数据源（逗号分隔），如: 财联社电报,华尔街见闻",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="快速模式：跳过 LLM 过滤，仅去重+质量检查",
    )
    parser.add_argument(
        "--min-score", type=int, default=5,
        help="LLM 最低相关性分数 (0-10)，默认 5",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="尝试触发 server.py 索引重建",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="详细日志",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 解析数据源
    sources = None
    if args.sources:
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    # 执行流水线
    pipeline = KnowledgePipeline()
    report = pipeline.run(
        sources=sources,
        min_llm_score=args.min_score,
        skip_llm=args.quick,
        rebuild=args.rebuild,
    )

    # 打印摘要
    print("\n" + "=" * 50)
    print("  动态知识库更新报告")
    print("=" * 50)
    print(f"  原始文章:    {report['raw_count']}")
    print(f"  清洗通过:    {report['cleaned_count']}")
    print(f"  新增入库:    {report['new_files']}")
    print(f"  重复跳过:    {report['duplicate_files']}")
    print(f"  错误:        {report['errors']}")
    print(f"  耗时:        {report.get('elapsed_seconds', '?')}秒")
    llm_status = "跳过" if report["skip_llm"] else f"阈值≥{report['min_llm_score']}"
    print(f"  LLM 过滤:    {llm_status}")
    print("=" * 50)

    if report["articles"]:
        print("\n  新增文章预览:")
        for i, a in enumerate(report["articles"][:10], 1):
            score_str = f"[{a['score']}]" if a.get("score") is not None else ""
            print(f"  {i:2d}. {score_str} [{a['source']}] {a['title'][:60]}")

    print(f"\n  文件保存位置: {CRAWLED_DIR}")
    print(f"  dedup.db 位置: {os.path.join(PROJECT_ROOT, 'dedup.db')}")
    print()


if __name__ == "__main__":
    main()
