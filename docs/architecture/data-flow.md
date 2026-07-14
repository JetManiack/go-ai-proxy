# Data Flow

## Non-Streaming Request

```
Client                   Server          Translator         Provider
  │                        │                 │                  │
  │── POST /v1/chat/comp ─►│                 │                  │
  │                        │── parse body   │                  │
  │                        │── FromOpenAI() ►│                  │
  │                        │                 │── Chat() ───────►│
  │                        │                 │                  │── Anthropic API
  │                        │                 │                  │◄─ response
  │                        │                 │◄── Response ─────│
  │                        │◄── ToOpenAI() ──│                  │
  │◄── 200 JSON ───────────│                 │                  │
```

**Steps:**
1. Client sends a standard OpenAI `POST /v1/chat/completions` JSON body
2. Server parses and calls `translator.FromOpenAI()` → `domain.Request`
3. Registry selects the appropriate provider by model name
4. Provider calls the upstream API and returns `domain.Response`
5. Server calls `translator.ToOpenAI()` → OpenAI JSON, responds with `200`

---

## Streaming Request

```
Client                   Server          Translator         Provider
  │                        │                 │                  │
  │── POST (stream:true) ─►│                 │                  │
  │                        │── FromOpenAI() ►│                  │
  │                        │                 │── ChatStream() ─►│
  │                        │                 │                  │── Anthropic SSE
  │◄── SSE headers ────────│                 │                  │
  │                        │                 │◄── Chunk ────────│  (per event)
  │◄── data: {...} ────────│◄── ToChunk() ───│                  │
  │◄── data: {...} ────────│                 │◄── Chunk ────────│
  │      ...               │                 │      ...         │
  │◄── data: [DONE] ───────│                 │◄── channel close │
```

**Steps:**
1. Client sends request with `stream: true`
2. Server opens SSE response (headers flushed immediately)
3. Provider returns `<-chan Chunk`; server reads from it in a loop
4. Each `Chunk` is translated to an OpenAI SSE frame and written to the response
5. When the channel closes, server writes `data: [DONE]\n\n` and closes the connection

---

## Model Passthrough and Provider Selection

The model name from the request is forwarded to the provider as-is. Callers use native model IDs directly (e.g. `claude-sonnet-4-6`, `claude-opus-4-6`).

The registry maintains a cached model→provider index, built at startup and refreshed in the background:

```
startup / background refresh
        │
        ▼
  for each registered provider:
    call Models(ctx) → []Model
    store model_id → provider mapping
        │
        ▼
  cache ready (refreshed every N minutes)


incoming request
        │
        ▼
  lookup model_id in cache
        │
  found? ──yes──► route to that provider
        │
        no
        │
        ▼
  400: model not available, known models: [...]
```
