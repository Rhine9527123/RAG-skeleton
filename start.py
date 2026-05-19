"""
一键启动：后端 + 前端
双击运行即可，不需要开两个终端
"""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Git Bash 没有 USERNAME 环境变量，torch 初始化时 getpass.getuser() 会报错
# 这里提前补上，子进程会继承
if "USERNAME" not in os.environ:
    os.environ["USERNAME"] = "User"

# 写死 Python 路径，避免双击 .py 时 Windows 用错 Python（如微软商店版缺依赖）
PYTHON = "python"
if not os.path.exists(PYTHON):
    # 回退到当前解释器
    PYTHON = sys.executable

print("=" * 40)
print("  RAG 财务助手 一键启动")
print("=" * 40)

# 启动后端（新窗口）
print("[1/2] 启动后端服务...")
backend = subprocess.Popen(
    [PYTHON, "server.py"],
    creationflags=subprocess.CREATE_NEW_CONSOLE,
)
print(f"       后端 PID: {backend.pid}")

# 启动前端（新窗口）
print("[2/2] 启动前端页面...")
frontend = subprocess.Popen(
    [PYTHON, "-m", "streamlit", "run", "web.py"],
    creationflags=subprocess.CREATE_NEW_CONSOLE,
)
print(f"       前端 PID: {frontend.pid}")

print()
print("✅ 启动完成！浏览器会自动打开聊天页面")
print("   关闭此窗口不会停止服务")
print("   要停止服务，请关闭弹出的两个黑色窗口")
print()

input("按回车键退出此启动器...")
