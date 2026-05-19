"""
RAG 一键启动 + cpolar 穿透 + 自动拼接公网链接
用法：python launch.py
"""
import os
import subprocess
import time
import re
import sys

BACKEND_PORT = 8000
FRONTEND_PORT = 8501
RAG_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = "python"


def start_process_hidden(title, cmd, output_file):
    """启动进程，输出重定向到文件，方便抓取"""
    subprocess.Popen(
        f'start "{title}" cmd /c "{cmd} > {output_file} 2>&1"',
        shell=True,
        cwd=RAG_DIR,
    )


def extract_url_from_file(filepath, port):
    """从 cpolar 输出文件里用正则提取公网 URL"""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        # 匹配 https://xxxx.cpolar.cn 或 https://xxxx.vip.cpolar.cn
        pattern = r"(https://[a-zA-Z0-9-]+\.(?:r\d+\.)?(?:vip\.)?cpolar\.(?:cn|top|io))"
        urls = re.findall(pattern, content)
        if urls:
            # 返回第一个 https 地址
            return urls[0]
    except Exception:
        pass
    return None


def main():
    print("=" * 50)
    print("  RAG - One Click Start + cpolar Tunnel")
    print("=" * 50)
    print()

    # 临时文件，用于捕获 cpolar 输出
    backend_log = "cpolar_backend.log"
    frontend_log = "cpolar_frontend.log"

    # 1. 启动后端
    print("[1/4] Starting backend (port 8000)...")
    subprocess.Popen(
        f'start "Backend" cmd /k "set USERNAME=User && cd /d {RAG_DIR} && {PYTHON} server.py"',
        shell=True,
    )

    # 2. 启动前端
    print("[2/4] Starting frontend (port 8501)...")
    time.sleep(2)
    subprocess.Popen(
        f'start "Frontend" cmd /k "set USERNAME=User && cd /d {RAG_DIR} && {PYTHON} -m streamlit run web.py --server.port {FRONTEND_PORT}"',
        shell=True,
    )

    # 3. 启动 cpolar，输出重定向到日志文件
    print("[3/4] Starting cpolar tunnels...")
    time.sleep(5)

    backend_log_path = f"{RAG_DIR}\\{backend_log}"
    frontend_log_path = f"{RAG_DIR}\\{frontend_log}"

    # cpolar 用 --log-to-stdout 参数确保输出可以被捕获
    subprocess.Popen(
        f'start "Tunnel-Backend" cmd /k "cd /d {RAG_DIR} && cpolar http {BACKEND_PORT} > {backend_log} 2>&1"',
        shell=True,
    )
    time.sleep(3)
    subprocess.Popen(
        f'start "Tunnel-Frontend" cmd /k "cd /d {RAG_DIR} && cpolar http {FRONTEND_PORT} > {frontend_log} 2>&1"',
        shell=True,
    )

    # 4. 等待并从日志文件抓取公网地址
    print("[4/4] Waiting for tunnel URLs...")
    backend_url = None
    frontend_url = None

    for attempt in range(15):
        time.sleep(2)
        backend_url = extract_url_from_file(backend_log_path, BACKEND_PORT)
        frontend_url = extract_url_from_file(frontend_log_path, FRONTEND_PORT)

        if backend_url and frontend_url:
            break
        print(f"  Waiting... (attempt {attempt + 1}/15)")

    # 清理日志文件
    for f in [backend_log_path, frontend_log_path]:
        try:
            os.remove(f)
        except Exception:
            pass

    # 显示结果
    print()
    print("=" * 50)
    print("  RESULT")
    print("=" * 50)
    print()
    print("  Local:")
    print(f"    Frontend: http://localhost:{FRONTEND_PORT}")
    print(f"    API Docs: http://localhost:{BACKEND_PORT}/docs")
    print()

    if backend_url and frontend_url:
        share_link = f"{frontend_url}?backend={backend_url}"
        print("  Public (share this link):")
        print()
        print(f"    {share_link}")
        print()
        # 复制到剪贴板
        try:
            process = subprocess.Popen(["clip"], stdin=subprocess.PIPE, shell=True)
            process.communicate(share_link.encode("utf-8"))
            print("  (Link copied to clipboard!)")
        except Exception:
            pass
    else:
        print("  Could not auto-detect URLs.")
        if backend_url:
            print(f"  Backend: {backend_url}")
        if frontend_url:
            print(f"  Frontend: {frontend_url}")
        print("  Check the 2 tunnel windows for Forwarding URLs.")

    print()
    print("=" * 50)
    print("  To stop: close all 4 black windows")
    print("=" * 50)
    print()
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
