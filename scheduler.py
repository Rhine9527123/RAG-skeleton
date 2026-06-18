"""
scheduler.py — APScheduler 定时任务模块
=======================================

两个定时任务：
  1. 爬虫调度 — 定时触发 pipeline 刷新知识库
  2. 旧知识清理 — 定时清理过期垃圾桶 + 旧爬取文件

集成方式：server.py 启动时调用 start_scheduler()，关闭时调用 stop_scheduler()

配置来源（优先级从高到低）：
  1. 环境变量（RAG_CRON_CRAWLER、RAG_CRON_CLEANUP）
  2. config.json 中的 scheduler 段
  3. 默认值

依赖：pip install apscheduler
"""

import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("scheduler")

# ── 常量 ────────────────────────────────────────────
CST = timezone(timedelta(hours=8))

# 默认调度配置
DEFAULT_CRAWLER_INTERVAL_HOURS = 6          # 爬虫间隔（小时）
DEFAULT_CLEANUP_HOUR = 3                     # 清理时间（凌晨3点）
DEFAULT_CRAWLED_RETENTION_DAYS = 7           # 爬取文件保留天数
DEFAULT_MAX_CRAWLED_FILES = 200              # 爬取文件最大数量
TRASH_DAYS = 30                              # 垃圾桶过期天数

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CRAWLED_DIR = os.path.join(DATA_DIR, "crawled")
TRASH_DIR = os.path.join(PROJECT_ROOT, ".trash")
TRASH_META_FILE = ".trash_meta.json"


def load_scheduler_config() -> dict:
    """加载 scheduler 配置，环境变量优先"""
    cfg = {
        "crawler_interval_hours": DEFAULT_CRAWLER_INTERVAL_HOURS,
        "cleanup_hour": DEFAULT_CLEANUP_HOUR,
        "crawled_retention_days": DEFAULT_CRAWLED_RETENTION_DAYS,
        "max_crawled_files": DEFAULT_MAX_CRAWLED_FILES,
        "enabled": True,
    }

    # 1. config.json
    config_path = os.path.join(PROJECT_ROOT, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            scheduler_cfg = raw.get("scheduler", {})
            for k in cfg:
                if k in scheduler_cfg:
                    cfg[k] = scheduler_cfg[k]
        except Exception:
            pass

    # 2. 环境变量覆盖
    env_overrides = {
        "RAG_CRON_CRAWLER": "crawler_interval_hours",
        "RAG_CRON_CLEANUP_HOUR": "cleanup_hour",
        "RAG_CRON_ENABLED": "enabled",
    }
    for env_key, cfg_key in env_overrides.items():
        val = os.environ.get(env_key)
        if val is not None:
            try:
                if cfg_key == "enabled":
                    cfg[cfg_key] = val.lower() in ("true", "1", "yes")
                else:
                    cfg[cfg_key] = int(val)
            except ValueError:
                pass

    return cfg


# ── 任务函数 ──────────────────────────────────────

def _run_crawler_pipeline():
    """
    爬虫定时任务：运行一次知识库更新流水线。
    快速模式：跳过 LLM 过滤，只做去重+质量检查。
    """
    from pipeline import KnowledgePipeline

    logger.info("[定时任务] 爬虫流水线开始...")
    start = time.time()

    try:
        pipeline = KnowledgePipeline()
        report = pipeline.run(
            skip_llm=True,        # 快速模式，省 LLM 费用
            min_llm_score=5,
            rebuild=False,        # 不触发 server 索引重建（文件已保存到 data/crawled/）
        )
        elapsed = round(time.time() - start, 1)
        logger.info(
            f"[定时任务] 爬虫流水线完成: "
            f"新增 {report['new_files']} 篇, "
            f"重复 {report['duplicate_files']} 篇, "
            f"耗时 {elapsed}秒"
        )
    except Exception as e:
        elapsed = round(time.time() - start, 1)
        logger.error(f"[定时任务] 爬虫流水线失败 (耗时{elapsed}秒): {e}")


def _run_cleanup():
    """
    旧知识清理任务：
      1. 清理垃圾桶中过期文件（>30天）
      2. 清理 data/crawled/ 中的旧爬取文件（>配置天数 或 超过数量上限）
    """
    logger.info("[定时任务] 旧知识清理开始...")
    start = time.time()

    cfg = load_scheduler_config()
    retention_days = cfg["crawled_retention_days"]
    max_files = cfg["max_crawled_files"]

    cleaned_trash = 0
    cleaned_crawled = 0
    errors = 0

    # ── 1. 清理过期垃圾桶 ──────────────────────
    try:
        meta_path = os.path.join(TRASH_DIR, TRASH_META_FILE)
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {}

        now = time.time()
        for filename, info in list(meta.items()):
            age_days = (now - info["deleted_at"]) / 86400
            if age_days >= TRASH_DAYS:
                trash_path = os.path.join(TRASH_DIR, filename)
                try:
                    if os.path.exists(trash_path):
                        os.remove(trash_path)
                    meta.pop(filename, None)
                    cleaned_trash += 1
                    logger.debug(f"[定时任务] 清理过期垃圾桶: {filename}（{int(age_days)}天）")
                except Exception as e:
                    logger.warning(f"[定时任务] 清理失败: {filename}: {e}")
                    errors += 1

        if cleaned_trash > 0:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[定时任务] 垃圾桶清理异常: {e}")
        errors += 1
    else:
        if cleaned_trash > 0:
            logger.info(f"[定时任务] 垃圾桶清理: {cleaned_trash} 个过期文件")

    # ── 2. 清理旧爬取文件 ──────────────────────
    if os.path.exists(CRAWLED_DIR):
        try:
            files = sorted(
                Path(CRAWLED_DIR).glob("crawled_*.txt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            cutoff = time.time() - retention_days * 86400

            for f in files:
                # 策略1: 超过保留天数的删除
                if f.stat().st_mtime < cutoff:
                    try:
                        f.unlink()
                        cleaned_crawled += 1
                    except Exception:
                        errors += 1
                # 策略2: 超过数量上限的删除（保留最新的 N 个）
                elif len(files) > max_files:
                    # 从末尾开始删（最旧的）
                    for old_file in files[max_files:]:
                        try:
                            if old_file.exists():
                                old_file.unlink()
                                cleaned_crawled += 1
                        except Exception:
                            errors += 1
                    break  # 已处理完数量超限的情况
        except Exception as e:
            logger.error(f"[定时任务] 爬取文件清理异常: {e}")
            errors += 1
        else:
            if cleaned_crawled > 0:
                logger.info(f"[定时任务] 爬取文件清理: {cleaned_crawled} 个旧文件")

    elapsed = round(time.time() - start, 1)
    logger.info(
        f"[定时任务] 清理完成: "
        f"垃圾桶 {cleaned_trash} + 爬取 {cleaned_crawled} = {cleaned_trash + cleaned_crawled} 个文件, "
        f"错误 {errors}, "
        f"耗时 {elapsed}秒"
    )


# ── 调度器管理 ──────────────────────────────────

_scheduler = None
_scheduler_started = False


def start_scheduler() -> bool:
    """
    启动 APScheduler 定时任务调度器。

    应在 FastAPI lifespan 启动阶段调用。
    重复调用安全（幂等）。
    """
    global _scheduler, _scheduler_started

    if _scheduler_started:
        logger.warning("[调度器] 已在运行，跳过重复启动")
        return True

    cfg = load_scheduler_config()

    if not cfg["enabled"]:
        logger.info("[调度器] 已禁用（scheduler.enabled=false 或 RAG_CRON_ENABLED=false），跳过启动")
        return False

    # 延迟导入（避免启动时依赖检查失败阻塞 server）
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error(
            "[调度器] APScheduler 未安装。请运行: pip install apscheduler"
        )
        return False

    # 确保日志目录存在
    os.makedirs(os.path.join(PROJECT_ROOT, ".pipeline_logs"), exist_ok=True)

    _scheduler = BackgroundScheduler(
        timezone=CST,
        job_defaults={
            "coalesce": True,          # 合并错过的任务（避免积压）
            "max_instances": 1,        # 同一任务最多 1 个并发实例
            "misfire_grace_time": 300, # 错过 5 分钟内仍执行
        },
    )

    interval_hours = cfg["crawler_interval_hours"]
    cleanup_hour = cfg["cleanup_hour"]

    # ── 任务 1: 爬虫调度 ──
    _scheduler.add_job(
        _run_crawler_pipeline,
        trigger=IntervalTrigger(hours=interval_hours),
        id="crawler_pipeline",
        name="财经新闻爬取流水线",
        replace_existing=True,
    )
    logger.info(
        f"[调度器] 爬虫任务已注册: 每 {interval_hours} 小时执行一次"
    )

    # ── 任务 2: 旧知识清理 ──
    _scheduler.add_job(
        _run_cleanup,
        trigger=CronTrigger(hour=cleanup_hour, minute=0),
        id="knowledge_cleanup",
        name="垃圾桶 + 旧爬取文件清理",
        replace_existing=True,
    )
    logger.info(
        f"[调度器] 清理任务已注册: 每天 {cleanup_hour}:00 执行"
    )

    _scheduler.start()
    _scheduler_started = True

    # 打印下次执行时间
    for job in _scheduler.get_jobs():
        logger.info(
            f"[调度器]   - {job.name}: 下次 {job.next_run_time.strftime('%Y-%m-%d %H:%M:%S') if job.next_run_time else 'N/A'}"
        )

    logger.info("[调度器] APScheduler 启动完成 ✓")
    return True


def stop_scheduler():
    """停止 APScheduler（应在 FastAPI lifespan 关闭阶段调用）"""
    global _scheduler, _scheduler_started

    if _scheduler and _scheduler.running:
        logger.info("[调度器] 正在停止...")
        _scheduler.shutdown(wait=False)
        _scheduler_started = False
        logger.info("[调度器] 已停止")


def get_scheduler_status() -> dict:
    """获取调度器运行状态（供 API 查询）"""
    if not _scheduler_started or not _scheduler:
        return {"status": "stopped", "jobs": []}

    jobs = []
    for job in _scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
        })

    return {
        "status": "running",
        "jobs": jobs,
    }


# ── 手动触发（供测试 / API 调用）──

def trigger_crawler_now():
    """手动触发一次爬虫流水线（在独立线程中执行，不阻塞）"""
    import threading
    if not _scheduler_started:
        logger.warning("[调度器] 未启动，无法手动触发")
        return {"status": "error", "message": "调度器未启动"}

    t = threading.Thread(target=_run_crawler_pipeline, daemon=True)
    t.start()
    return {"status": "ok", "message": "爬虫任务已在后台线程中启动"}


def trigger_cleanup_now():
    """手动触发一次旧知识清理（在独立线程中执行，不阻塞）"""
    import threading
    if not _scheduler_started:
        logger.warning("[调度器] 未启动，无法手动触发")
        return {"status": "error", "message": "调度器未启动"}

    t = threading.Thread(target=_run_cleanup, daemon=True)
    t.start()
    return {"status": "ok", "message": "清理任务已在后台线程中启动"}
