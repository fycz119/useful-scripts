import os
from openai import OpenAI

# 1. 配置 API 密钥和基础 URL
# 如果你是本地部署（例如 vLLM），base_url 通常是 "http://localhost:8000/v1" 且不需要真实 key
# 如果你使用的是第三方 API 平台，请填入他们提供的 url 和 key
API_KEY = os.getenv("OPENAI_API_KEY", "your_api_key_here")  # 替换为你的 API Key
BASE_URL = "https://your_endpoint_url_here/v1"              # 替换为你的 API 服务地址

# 2. 目标模型名称
MODEL_NAME = "qwen3.6-35B-A3B"

# 3. 初始化 OpenAI 客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

def test_basic_chat():
    """测试基本的非流式对话"""
    print("--- 开始基础请求测试 ---")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个人工智能助手。"},
                {"role": "user", "content": "你好，请简要介绍一下你自己。"}
            ],
            temperature=0.7,
            max_tokens=512
        )
        print("回复内容:")
        print(response.choices[0].message.content)
        print("------------------------\n")
    except Exception as e:
        print(f"请求失败: {e}")

def test_stream_chat():
    """测试流式对话 (打字机效果)"""
    print("--- 开始流式请求测试 ---")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "请写一首关于春天的简短现代诗。"}
            ],
            temperature=0.7,
            max_tokens=512,
            stream=True  # 开启流式输出
        )
        
        print("回复内容: ", end="", flush=True)
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n------------------------\n")
    except Exception as e:
        print(f"请求失败: {e}")

if __name__ == "__main__":
    test_basic_chat()
    test_stream_chat()
