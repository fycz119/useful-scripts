curl http://<你的-vllm-ip>:8000/v1/messages \
     -H "Content-Type: application/json" \
     -H "x-api-key: dummy-key" \
     -H "anthropic-version: 2023-06-01" \
     -d '{
         "model": "你的模型名称",
         "max_tokens": 1024,
         "messages": [
             {"role": "user", "content": "Hello, are you using the Anthropic API?"}
         ]
     }'
