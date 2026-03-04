# -*- coding: utf-8 -*-
import os
import requests
import json
from datetime import datetime

# =============================================
#          OpenRouter 测试脚本
#   模型：stepfun/step-3.5-flash:free
# =============================================

# 方法1：直接填入你的 API Key（不推荐上传到 github）
# API_KEY = "sk-or-v1-***"

# 方法2：推荐从环境变量读取（更安全）
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    print("错误：未找到 OPENROUTER_API_KEY")
    print("请设置环境变量，或在代码中填写 API_KEY")
    print('  Windows CMD:   set OPENROUTER_API_KEY=sk-or-v1-xxx')
    print('  Linux/Mac:     export OPENROUTER_API_KEY=sk-or-v1-xxx')
    exit(1)


def call_stepfun_chat(prompt: str, temperature=0.7, max_tokens=1024):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-app-url.com",       # 可选，但建议填写
        "X-Title": "Stepfun Flash Test Script",           # 可选，显示在 openrouter 统计页面
    }

    payload = {
        "model": "stepfun/step-3.5-flash:free",
        "messages": [
            {"role": "system", "content": "你是一个非常聪明、回答简洁且有点幽默的AI助手。"},
            {"role": "user",   "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        # "stream": False,   # 如果想用流式可以改为 True
    }

    try:
        start_time = datetime.now()
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        elapsed = (datetime.now() - start_time).total_seconds()

        if response.status_code != 200:
            print(f"请求失败 {response.status_code}")
            print(response.text)
            return None

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        print("\n" + "="*60)
        print(f"用时: {elapsed:.2f} 秒")
        print(f"Tokens: {usage.get('total_tokens', '?'):,} "
              f"(prompt:{usage.get('prompt_tokens', '?'):,} "
              f"completion:{usage.get('completion_tokens', '?'):>4})")
        print("-"*60)
        print(content)
        print("="*60 + "\n")

        return content

    except Exception as e:
        print("请求发生异常:", str(e))
        return None


if __name__ == "__main__":

    test_cases = [
        "用三句话以内告诉我今天是星期几？",
        "写一段非常有画面感的恐怖小说开场（100字以内）",
        "请用古文回答：何为人生最大乐事？",
        "1+1=? （故意问弱智问题测试模型是否正常）",
    ]

    print("开始测试 stepfun/step-3.5-flash:free ...\n")

    for i, question in enumerate(test_cases, 1):
        print(f"\n测试 {i}/{len(test_cases)} ：{question}")
        call_stepfun_chat(question)

    print("\n测试结束。你也可以自己改 prompt 继续玩～")
