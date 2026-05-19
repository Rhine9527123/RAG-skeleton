"""测试 FastAPI 服务接口"""
import requests
import json

# 测试 health 接口
print("=== 测试 /health ===")
r = requests.get("http://localhost:8000/health")
print(f"状态码: {r.status_code}")
print(f"响应: {r.text[:500]}")

# 测试 chat 接口
print("\n=== 测试 /chat ===")
r = requests.post("http://localhost:8000/chat", json={"question": "小规模纳税人征收率是1%还是3%？"})
print(f"状态码: {r.status_code}")
print(f"响应: {r.text[:1000]}")
