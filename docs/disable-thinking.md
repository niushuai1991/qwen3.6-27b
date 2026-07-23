# 关闭 Qwen3 Thinking 模式

## 背景

Qwen3.6-27B 是一个 reasoning/thinking 模型，默认会在回答前生成思考过程（`<｜end▁of▁thinking｜>` 标签内），导致额外 token 消耗并影响 benchmark 的可比性。

## 有效方式

经 2026-07-09 实测（vLLM 0.24.0），两种客户端参数均可关闭 thinking。**推荐使用 OpenAI 兼容的 `reasoning_effort`（方式 1）**；`chat_template_kwargs` 仍可用但不推荐。

### 方式 1: `reasoning_effort`（OpenAI 兼容，✅ 推荐）

OpenAI Chat Completions 标准参数，vLLM 原生支持。任意 OpenAI 兼容客户端/SDK 均可直接传，无需 `extra_body` 包装、也不依赖服务端 chat template 实现。

**curl:**

```bash
curl -s http://localhost:18001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 100,
    "reasoning_effort": "none"
  }'
```

**OpenAI SDK:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:18001/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="qwen3.6-27b",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    max_tokens=100,
    reasoning_effort="none",
)
print(response.choices[0].message.content)
```

### 方式 2: `chat_template_kwargs`（HuggingFace/vLLM 专有，⚠️ 不推荐）

> 仅作历史/兼容记录。它与 chat template 的 `enable_thinking` 变量强耦合，SDK 端需经 `extra_body` 透传，可移植性与健壮性均不如方式 1。新代码请用方式 1（`reasoning_effort`）。

**curl:**

```bash
curl -s http://localhost:18001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 100,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

**OpenAI SDK:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:18001/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="qwen3.6-27b",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    max_tokens=100,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.content)
```

## 误区

以下写法**无效**（实测确认）：

- `{"enable_thinking": false}` — vLLM 不识别顶层 `enable_thinking`
- `{"extra_body": {"enable_thinking": false}}` — `extra_body` 是 OpenAI SDK 客户端概念，SDK 会将其展开到请求顶层；直接在 HTTP body 里传不会生效

## benchmark.py 使用

```bash
# 默认 thinking 关闭
uv run python benchmark.py --label "baseline" --concurrency "1,5,10"

# 如果需要开启 thinking（调试/对比）
uv run python benchmark.py --label "baseline" --concurrency "1,5,10" --enable-thinking
```

## 效果验证

```
# 默认（thinking 关闭）
content: "4" | reasoning: (无)

# --enable-thinking
content: None | reasoning: "Here's a thinking process..."
```

## 参考

- [HuggingFace Qwen3.6-27B-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)
- [vLLM Issue #20976](https://github.com/vllm-project/vllm/issues/20976)
