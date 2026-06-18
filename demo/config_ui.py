"""
RAG 财务知识库服务 - 首次配置界面

客户第一次使用时双击运行，填写 API Key 后自动保存。
之后直接启动主程序即可。

配置保存位置：与 exe 同目录的 config.json
"""
import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import sys


def get_config_path():
    """
    获取 config.json 的路径
    如果是 PyInstaller 打包的 exe，放在 exe 同目录
    如果是开发模式，放在当前工作目录
    """
    if getattr(sys, "frozen", False):
        # PyInstaller 打包后，sys.executable 是 exe 路径
        exe_dir = os.path.dirname(sys.executable)
    else:
        exe_dir = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(exe_dir, "config.json")


def get_models_dir():
    """获取模型文件夹路径（与 exe 同目录的 models/）"""
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
    else:
        exe_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(exe_dir, "models")


def load_config():
    """加载已有配置，不存在则返回默认配置"""
    config_path = get_config_path()
    default = {
        "kimi_api_key": "",
        "kimi_api_base": "https://api.moonshot.cn/v1",
        "kimi_model": "moonshot-v1-8k",
        "reranker_model_path": "",
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                default.update(saved)
        except (json.JSONDecodeError, IOError):
            pass

    # 自动检测 Reranker 模型路径
    models_dir = get_models_dir()
    reranker_default = os.path.join(models_dir, "BAAI", "bge-reranker-v2-m3")
    if not default["reranker_model_path"] or not os.path.exists(default["reranker_model_path"]):
        if os.path.exists(reranker_default):
            default["reranker_model_path"] = reranker_default

    return default


def save_config(config):
    """保存配置到 config.json"""
    config_path = get_config_path()
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def create_config_ui():
    """创建配置界面"""
    config = load_config()

    root = tk.Tk()
    root.title("RAG 财务知识库 - 初始配置")
    root.geometry("480x380")
    root.resizable(False, False)

    # 窗口居中
    root.update_idletasks()
    x = (root.winfo_screenwidth() - 480) // 2
    y = (root.winfo_screenheight() - 380) // 2
    root.geometry(f"480x380+{x}+{y}")

    main_frame = ttk.Frame(root, padding=30)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # 标题
    ttk.Label(
        main_frame,
        text="RAG 财务知识库服务",
        font=("", 16, "bold"),
    ).pack(pady=(0, 5))

    ttk.Label(
        main_frame,
        text="首次使用，请填写以下配置信息",
        font=("", 10),
        foreground="gray",
    ).pack(pady=(0, 20))

    # API Key
    key_frame = ttk.Frame(main_frame)
    key_frame.pack(fill=tk.X, pady=(0, 12))

    ttk.Label(key_frame, text="Kimi API Key：", font=("", 10)).pack(anchor=tk.W)
    ttk.Label(
        key_frame,
        text="获取地址：platform.moonshot.cn",
        font=("", 9),
        foreground="gray",
    ).pack(anchor=tk.W)

    key_var = tk.StringVar(value=config.get("kimi_api_key", ""))
    key_entry = ttk.Entry(key_frame, textvariable=key_var, width=50, show="*")
    key_entry.pack(fill=tk.X, pady=(4, 0))

    # 模型选择
    model_frame = ttk.Frame(main_frame)
    model_frame.pack(fill=tk.X, pady=(0, 12))

    ttk.Label(model_frame, text="LLM 模型：", font=("", 10)).pack(anchor=tk.W)
    model_var = tk.StringVar(value=config.get("kimi_model", "moonshot-v1-8k"))
    model_combo = ttk.Combobox(
        model_frame,
        textvariable=model_var,
        values=["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
        state="readonly",
        width=30,
    )
    model_combo.pack(fill=tk.X, pady=(4, 0))

    # 状态标签
    status_var = tk.StringVar(value="")
    status_label = ttk.Label(main_frame, textvariable=status_var, font=("", 9), foreground="gray")
    status_label.pack(pady=(10, 5))

    # 检查 Reranker 模型
    reranker_path = config.get("reranker_model_path", "")
    if reranker_path and os.path.exists(reranker_path):
        status_var.set(f"Reranker 模型已就绪：{reranker_path}")
    else:
        status_var.set("未检测到 Reranker 模型，将使用在线 Reranker 或跳过精排")

    # 保存按钮
    def on_save():
        api_key = key_var.get().strip()
        if not api_key:
            messagebox.showwarning("提示", "请填写 Kimi API Key！")
            return

        new_config = {
            "kimi_api_key": api_key,
            "kimi_api_base": config.get("kimi_api_base", "https://api.moonshot.cn/v1"),
            "kimi_model": model_var.get(),
            "reranker_model_path": reranker_path,
        }
        save_config(new_config)
        status_var.set("配置已保存！")
        messagebox.showinfo("完成", "配置已保存，可以启动服务了！\n\n请关闭本窗口，双击启动程序。")
        root.destroy()

    save_btn = ttk.Button(main_frame, text="保存配置", command=on_save)
    save_btn.pack(pady=(15, 0))

    # 重新配置按钮（已有配置时显示）
    if config.get("kimi_api_key"):
        ttk.Label(
            main_frame,
            text="检测到已有配置，可直接修改后保存",
            font=("", 9),
            foreground="green",
        ).pack(pady=(10, 0))

    root.mainloop()


if __name__ == "__main__":
    create_config_ui()
