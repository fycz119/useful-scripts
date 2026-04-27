import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "你的模型路径"   # 🔧 修改这里
DEVICE = "cuda:0"            # 🔧 强制单卡测试

print("=" * 50)
print("🔥 基础环境检查")
print("=" * 50)

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))

print("\n" + "=" * 50)
print("🔥 加载模型（单卡强制）")
print("=" * 50)

t0 = time.time()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
).to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"model load time: {time.time() - t0:.2f}s")

# 检查模型设备
print("\n模型参数设备:")
print(next(model.parameters()).device)

print("\n" + "=" * 50)
print("🔥 构造输入")
print("=" * 50)

prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt")

print("input device BEFORE:", inputs["input_ids"].device)

# 🔥 强制放 GPU
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

print("input device AFTER:", inputs["input_ids"].device)

print("\n" + "=" * 50)
print("🔥 Forward 单步性能测试")
print("=" * 50)

with torch.inference_mode():
    torch.cuda.synchronize()
    t0 = time.time()

    outputs = model(**inputs, use_cache=True)

    torch.cuda.synchronize()
    t1 = time.time()

print(f"forward time: {t1 - t0:.4f}s")

# KV cache 检查
print("\nKV cache 是否存在:")
print(outputs.past_key_values is not None)

print("\n" + "=" * 50)
print("🔥 连续 forward（模拟生成）")
print("=" * 50)

input_ids = inputs["input_ids"]

with torch.inference_mode():
    for i in range(5):
        torch.cuda.synchronize()
        t0 = time.time()

        outputs = model(input_ids=input_ids, use_cache=True)
        input_ids = outputs.logits[:, -1:].argmax(dim=-1)

        torch.cuda.synchronize()
        print(f"step {i}: {time.time() - t0:.4f}s")

print("\n" + "=" * 50)
print("🔥 generate 性能测试")
print("=" * 50)

with torch.inference_mode():
    torch.cuda.synchronize()
    t0 = time.time()

    out = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        use_cache=True,
    )

    torch.cuda.synchronize()
    t1 = time.time()

print(f"generate total time: {t1 - t0:.2f}s")
print(f"avg per token: {(t1 - t0)/20:.4f}s")

print("\n生成结果:")
print(tokenizer.decode(out[0]))

print("\n" + "=" * 50)
print("🔥 GPU 利用率提示")
print("=" * 50)

print("👉 运行时请同时开一个终端执行:")
print("watch -n 0.5 nvidia-smi")

print("\n" + "=" * 50)
print("🔥 结果判断指南")
print("=" * 50)

print("""
1️⃣ forward 很慢（>1s）：
   → attention / KV / 多卡问题

2️⃣ forward 很快，但 generate 很慢：
   → generate 循环 / sampling / python 阻塞

3️⃣ KV cache = False：
   → 你在重复全量计算（致命问题）

4️⃣ GPU 利用率 <20%：
   → 数据没在 GPU / 有同步阻塞

5️⃣ 单卡快，多卡慢：
   → device_map / 通信问题

6️⃣ step 时间越来越慢：
   → KV cache 没复用 / 内存问题
""")
