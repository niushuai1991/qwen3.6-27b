# 关闭 Qwen3 Thinking 模式

## 背景

Qwen3.6-27B 是一个 reasoning/thinking 模型，默认会在回答前生成思考过程（`<｜end▁of▁thinking｜>` 标签内），导致额外 token 消耗并影响 benchmark 的可比性。

## 有效方式

经 2026-07-09 实测（vLLM 0.24.0），两种客户端参数可关闭 thinking：

| 方式 | curl 请求体 | OpenAI SDK |
|------|------------|------------|
| `chat_template_kwargs` | `{"chat_template_kwargs": {"enable_thinking": false}}` | `extra_body={"chat_template_kwargs": {"enable_thinking": false}}` |
| `reasoning_effort` | `{"reasoning_effort": "none"}` | `extra_body={"reasoning_effort": "none"}` |

本项目使用 `chat_template_kwargs` 方式（HuggingFace 官方推荐）。

## 误区

以下写法**无效**（实测确认）：

- `{"enable_thinking": false}` — vLLM 不识别顶层 `enable_thinking`
- `{"extra_body": {"enable_thinking": false}}` — `extra_body` 是 OpenAI SDK 客户端概念，SDK 会将其展开到请求顶层；直接在 HTTP body 里传不会生效

## benchmark.py 使用

```bash
# 默认 thinking 开启（兼容现有行为）
uv run python benchmark.py --label "baseline" --concurrency "1,5,10"

# 关闭 thinking
uv run python benchmark.py --label "baseline" --concurrency "1,5,10" --disable-thinking
```

## 效果验证

```
# 默认（thinking 开启）
content: None | reasoning: "Here's a thinking process..."

# --disable-thinking
content: "4" | reasoning: (无)
```

## 参考

- [HuggingFace Qwen3.6-27B-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)
- [vLLM Issue #20976](https://github.com/vllm-project/vllm/issues/20976)
