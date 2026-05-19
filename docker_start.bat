@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo   RAG + Hermes Docker 一键部署
echo ============================================================
echo.

REM ---- 检查 Docker 是否运行 ----
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Docker Desktop 未启动！
    echo.
    echo   请先启动 Docker Desktop，等待左下角显示绿色 "Engine running" 后
    echo   再重新运行本脚本。
    echo.
    pause
    exit /b 1
)
echo [OK] Docker Desktop 已运行
echo.

REM ---- 检查 .env 是否存在 ----
if not exist .env (
    echo [提示] 未找到 .env 文件，正在从模板创建...
    copy .env.example .env >nul
    echo.
    echo   !!! 重要：请先编辑 .env 文件，填写你的 Kimi API Key !!!
    echo   获取地址：https://platform.moonshot.cn/console/api-keys
    echo.
    echo   用记事本打开 .env 文件，把 KIMI_API_KEY= 后面的值改成你的 Key
    echo   保存后再重新运行本脚本。
    echo.
    notepad .env
    pause
    exit /b 0
)

REM ---- 检查 API Key 是否已填写 ----
findstr /C:"KIMI_API_KEY=在这里填写" .env >nul 2>&1
if %errorlevel% equ 0 (
    echo [提示] 检测到 API Key 尚未填写！
    echo.
    echo   请先编辑 .env 文件，填写你的 Kimi API Key
    echo   获取地址：https://platform.moonshot.cn/console/api-keys
    echo.
    notepad .env
    pause
    exit /b 0
)

findstr /C:"KIMI_API_KEY=$" .env >nul 2>&1
if %errorlevel% equ 0 (
    echo [提示] 检测到 API Key 为空！
    echo.
    echo   请先编辑 .env 文件，填写你的 Kimi API Key
    echo   获取地址：https://platform.moonshot.cn/console/api-keys
    echo.
    notepad .env
    pause
    exit /b 0
)

echo [OK] API Key 已配置
echo.

echo [1/3] 构建 RAG 服务镜像（首次较慢，约 5-10 分钟）...
docker compose build rag-service
if %errorlevel% neq 0 (
    echo [错误] RAG 镜像构建失败！
    echo.
    echo   常见原因：
    echo   1. 网络问题 - 检查是否能正常访问 Docker Hub
    echo   2. 磁盘空间不足 - Docker 镜像需要约 4GB 空间
    echo   3. models/ 目录缺少模型文件 - 确保 models/bge-reranker-v2-m3/ 存在
    echo.
    pause
    exit /b 1
)

echo.
echo [2/3] 启动所有服务（RAG + Hermes）...
docker compose up -d
if %errorlevel% neq 0 (
    echo [错误] 服务启动失败！
    echo   运行 docker compose logs 查看详细日志
    pause
    exit /b 1
)

echo.
echo [3/3] 等待服务就绪...
timeout /t 8 /nobreak >nul
docker compose ps

echo.
echo ============================================================
echo   部署完成！
echo.
echo   RAG 接口文档:   http://localhost:8000/docs
echo   Hermes 管理面板: http://localhost:3000
echo ============================================================
echo.
echo   查看日志: 双击 docker_view_logs.bat
echo   停止服务: 双击 docker_stop.bat
echo.
pause
