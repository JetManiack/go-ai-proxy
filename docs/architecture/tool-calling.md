# Tool Calling

Tool calling is the most complex part of the translation layer because OpenAI and Anthropic use fundamentally different conversation structures for multi-turn tool use.

## Format Differences

### OpenAI

Tool results are separate messages with `role: "tool"`:

```json
[
  { "role": "user", "content": "What's the weather in Paris?" },
  {
    "role": "assistant",
    "content": null,
    "tool_calls": [
      {
        "id": "call_abc",
        "type": "function",
        "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" }
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_abc",
    "content": "{\"temp\": 18, \"unit\": \"C\"}"
  },
  { "role": "assistant", "content": "It's 18°C in Paris." }
]
```

### Anthropic

Tool calls are content blocks inside assistant messages. Tool results are content blocks inside a **user** message:

```json
[
  { "role": "user", "content": "What's the weather in Paris?" },
  {
    "role": "assistant",
    "content": [
      {
        "type": "tool_use",
        "id": "toolu_abc",
        "name": "get_weather",
        "input": { "city": "Paris" }
      }
    ]
  },
  {
    "role": "user",
    "content": [
      {
        "type": "tool_result",
        "tool_use_id": "toolu_abc",
        "content": "{\"temp\": 18, \"unit\": \"C\"}"
      }
    ]
  },
  { "role": "assistant", "content": "It's 18°C in Paris." }
]
```

## Multiple Tool Results

When the assistant calls multiple tools in one turn, OpenAI sends multiple `role: "tool"` messages in sequence. Anthropic expects **a single user message** containing multiple `tool_result` content blocks.

The translator must group consecutive `role: "tool"` messages and collapse them into one Anthropic user message:

```
OpenAI:
  [tool result A]
  [tool result B]
  [tool result C]

Anthropic:
  [user: tool_result A, tool_result B, tool_result C]
```

## Streaming Tool Calls

Anthropic streams tool call arguments as `input_json_delta` events rather than emitting them all at once. The translator must:

1. Accumulate `input_json_delta` fragments per tool call index
2. Re-emit them as OpenAI `choices[0].delta.tool_calls[N].function.arguments` deltas

The tool call `id` and `name` arrive in the `content_block_start` event; subsequent `input_json_delta` events carry only the JSON fragment.

## Translation Responsibilities Summary

| Scenario | Translator Action |
|---|---|
| `role: "system"` message | Extract to Anthropic top-level `system` field |
| `role: "tool"` messages | Collect, group by position, emit as single Anthropic user message with `tool_result` blocks |
| Assistant `tool_calls` array | Convert each entry to `type: "tool_use"` content block |
| Anthropic `tool_use` response block | Convert to OpenAI `tool_calls` array on the assistant message |
| Streaming `input_json_delta` | Accumulate and forward as OpenAI argument deltas |
