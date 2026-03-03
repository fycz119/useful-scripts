import requests

url = "https://api.openai.com/v1/chat/completions"

headers = {
    "Authorization": "Bearer sk-你的真实APIKey填这里",
    "Content-Type": "application/json"
}

data = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "user", "content": "测试是否成功"}
    ]
}

response = requests.post(url, headers=headers, json=data)

print(response.status_code)
print(response.json())
