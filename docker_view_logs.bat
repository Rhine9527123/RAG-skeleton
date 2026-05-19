@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 按 Ctrl+C 退出日志查看
echo.
docker compose logs -f
