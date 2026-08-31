I’m using `MiniMaxAI/MiniMax-M3-MXFP8` behind vLLM’s OpenAI chat completions API. Our incident-response agent exposes an internal MCP runbook search as a function tool, and I sent this request:

```json
{
  "model": "MiniMaxAI/MiniMax-M3-MXFP8",
  "messages": [
    {
      "role": "user",
      "content": "Checkout API is returning elevated 502s. Search the current incident runbooks for mitigation steps before answering."
    }
  ],
  "stream": true,
  "tool_choice": "auto",
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search_incident_runbooks",
        "parameters": {
          "type": "object",
          "properties": {
            "service": {"type": "string"},
            "symptom": {"type": "string"}
          },
          "required": ["service", "symptom"]
        }
      }
    }
  ]
}
```

I notice that the beginning of the stream looks very strange:

```text
data: {"choices":[{"delta":{"reasoning":null,"content":"<mm:think>"}}]}
data: {"choices":[{"delta":{"reasoning":null,"content":"I should search the current checkout incident runbook first."}}]}
data: {"choices":[{"delta":{"reasoning":null,"content":"</mm:think>"}}]}
data: {"choices":[{"delta":{"reasoning":null,"content":null,"tool_calls":[{"type":"function","function":{"name":"search_incident_runbooks","arguments":"{\"service\":\"checkout-api\",\"symptom\":\"elevated 502s\"}"}}]}}]}
```

When I send the same request with `"stream":false`, the response is fine: the thinking text is in the reasoning field, the `<mm:think>` markers are hidden, and the same structured tool call is returned. Could you investigate why only the streaming response puts the thinking section in content and fix it without changing the non-streaming result or breaking tool calls?
