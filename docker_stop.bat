@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在停止所有服务...
docker compose down
echo.
echo 已停止。
pause
