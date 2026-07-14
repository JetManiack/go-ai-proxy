# go-ai-proxy

Universal OpenAI-compatible HTTP proxy for LLM providers.
Point any OpenAI client at it and transparently use Anthropic, local LM Studio, Ollama, or any OpenAI-compatible backend — without changing your application code.

Built in Go as a auditable, supply-chain-safe alternative to LiteLLM.

---

## Why not LiteLLM?

LiteLLM is the closest equivalent. It was rejected for two reasons:

- **Supply chain compromise** — a malicious package was published to PyPI under the LiteLLM name ([#24518](https://github.com/BerriAI/litellm/issues/24518)). Python's packaging model makes this class of attack structurally hard to prevent.
- **Auditability** — this proxy has a small, explicit dependency graph. Every line is readable by a single engineer in an afternoon.

---

## Features

- **Drop-in for any OpenAI client** — implements `/v1/chat/completions` and `/v1/models`; one config line to adopt
- **Multiple providers** — Anthropic and any OpenAI-compatible server (LM Studio, Ollama, OpenRouter, …)
- **Fallback chains** — register several providers for the same model; failed attempts transparently try the next
- **Load balancing** — least-connections routing across providers serving the same model
- **Concurrency limiting** — `max_concurrent` + `queue_size` per provider; overflow returns `503`
- **First-token timeout** — `request_timeout` triggers failover to the next provider if no token arrives in time; slow but running generation is never interrupted
- **Thinking / chain-of-thought** — extracts `<think>` tags (R1, QwQ) and `reasoning_content` (DeepSeek API) and forwards them as `reasoning_content` to clients; Anthropic extended thinking enabled via `budget_tokens`
- **Capability-based routing** — use `model: "auto:vision"` or `model: "auto:vision,reasoning"` instead of a concrete model name; the proxy picks the least-loaded matching model automatically
- **On-demand model refresh** — unknown model triggers an immediate re-fetch before returning `400`; no restart needed when a model is loaded into LM Studio mid-session
- **Rate limiting** — token-bucket rate limiter, global or per caller
- **Audit log** — structured request/response logging; full prompt and response content at `debug` level
- **Prometheus metrics** — request counts, latency, token usage per provider and model
- **Graceful shutdown** — drains in-flight requests before exiting
- **Single static binary** — no runtime, no virtualenv, ships as a Docker image or bare binary

---

## Quick start

```bash
# Build
go build -o gap ./cmd/gap

# Run with a local LM Studio
./gap --config config.yaml

# Run with Anthropic
ANTHROPIC_API_KEY=sk-ant-... ./gap
```

Then point any OpenAI client at the proxy:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8090/v1",
    api_key="any-string",          # not validated by the proxy
)
response = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "hello"}],
)
```

Or in a YAML config file:

```yaml
base_url: "http://127.0.0.1:8090/v1"
api_key: "any-string"
model: "claude-sonnet-4-6"
```

---

## Configuration

Minimal config for a local LM Studio instance:

```yaml
server:
  host: "127.0.0.1"
  port: 8090

providers:
  - name: lmstudio
    type: openai
    base_url: "http://localhost:1234"
```

Local with Anthropic fallback and concurrency control:

```yaml
server:
  host: "127.0.0.1"
  port: 8090

providers:
  - name: lmstudio
    type: openai
    base_url: "http://localhost:1234"
    max_concurrent: 2        # GPU can handle 2 parallel requests
    queue_size: 6            # up to 6 more wait in queue
    request_timeout: 20s     # fall back to Anthropic if no first token in 20s

  - name: anthropic
    type: anthropic
    auth:
      method: api_key
      api_key: "${ANTHROPIC_API_KEY}"
```

See [docs/config.md](docs/config.md) for the full configuration reference.

---

## Thinking / chain-of-thought

Reasoning models expose their thinking in different formats depending on the provider.
The proxy normalises all of them to `reasoning_content` — the field expected by
Continue.dev and similar tools.

| Source | Format |
|---|---|
| Local R1, QwQ (LM Studio / Ollama) | `<think>…</think>` tags in content stream |
| DeepSeek API | `reasoning_content` field in response / delta |
| Anthropic extended thinking | `thinking` content block |

For local models no configuration is required. For Anthropic extended thinking, pass
`budget_tokens` in the request body:

```python
response = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "..."}],
    extra_body={"budget_tokens": 8000},
)
print(response.choices[0].message.reasoning_content)
```

---

## Capability-based routing

Instead of specifying a model name, you can specify what the model must support.
The proxy resolves the selector to the least-loaded matching model at request time.

```yaml
model: "auto:vision"            # any model with vision capability
model: "auto:reasoning"         # any model with reasoning capability
model: "auto:vision,reasoning"  # both
```

Capabilities are populated from:
- `model_capabilities` in config (manual override — useful for LM Studio)
- `/model/info` response for `type: litellm` providers (automatic)
- OpenRouter-style `architecture` fields in `/v1/models` responses (automatic)

---

## Log levels

```bash
./gap --log-level info    # default: request metadata only
./gap --log-level debug   # also logs full prompt and response content
GAP_LOG_LEVEL=debug ./gap
```

---

## Docker

```bash
docker build -t go-ai-proxy .
docker run -e ANTHROPIC_API_KEY=sk-ant-... -p 8090:8090 go-ai-proxy
```

---

## Architecture

```
OpenAI client
     │  HTTP (OpenAI wire format)
     ▼
┌─────────────┐
│ HTTP Server │  /v1/chat/completions  /v1/models
└──────┬──────┘
       │  canonical Request
       ▼
┌────────────┐
│ Translator │  OpenAI ↔ canonical types
└──────┬─────┘
       │
       ▼
┌──────────────────┐
│ Provider Registry│  least-connections routing, fallback chain
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
Anthropic   OpenAI-compatible
provider    provider (LM Studio, Ollama, …)
```

Providers are isolated behind a canonical internal format (`domain.Request` / `domain.Response` / `domain.Chunk`). No component knows about another component's wire format.

See [docs/architecture/](docs/architecture/) for detailed design notes.

---

## Adding a provider

1. Create `internal/provider/<name>/provider.go`
2. Implement the `domain.Provider` interface (including `Name() string`)
3. Add `case "<name>":` to `buildProvider` in `cmd/gap/main.go`
4. Add a config section with `type: <name>` in `config.yaml`

---

## Testing

```bash
go test ./...
go test ./... -coverprofile=coverage.out && go tool cover -html=coverage.out
```

Tests use fake providers and `httptest` servers — no real network calls required.
End-to-end tests against a real LM Studio instance are gated behind `-tags e2e`.
