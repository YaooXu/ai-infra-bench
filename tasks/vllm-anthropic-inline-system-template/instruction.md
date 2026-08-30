I am using Claude Code through vLLM's Anthropic Messages API with Qwen3.6-27B. The following production conversation returns HTTP 400 even though its first message is already a system message:

```json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Please check the GPU status for vLLM."},
  {"role": "assistant", "content": "Sure, I will check it."},
  {"role": "user", "content": "Show me the nvidia-smi output."},
  {"role": "assistant", "content": "The GPU status looks normal, with utilization around 15%."},
  {"role": "user", "content": "Write up the results as a report."},
  {"role": "system", "content": "Task instruction: Based on the conversation above, generate a brief GPU status report."},
  {"role": "user", "content": "Please summarize the above."}
]
```

The response is `System message must be at the beginning.` The later system message is inserted by an auxiliary task after earlier conversation turns. Other model templates accept this ordering, where preserving the existing turn sequence avoids invalidating the reusable conversation prefix.

Fix the Anthropic request handling so this request works with model templates that require system-first ordering without unnecessarily changing the message order for templates that already accept it. Message generation and token counting must apply the same behavior.
