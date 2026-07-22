# Configuration reference

`gap` is configured via a YAML file (default: `config.yaml`).
CLI flags and environment variables override individual fields.

---

## Priority order

```
CLI flag  >  environment variable  >  config.yaml value  >  built-in default
```

---

## Top-level structure

```yaml
server:
  # ... server settings

providers:
  - name: my-provider
    # ... provider settings
```

---

## `server`

| Field | Type | Default | Description |
|---|---|---|---|
| `host` | string | `127.0.0.1` | Address to bind to. |
| `port` | int | `8090` | Port to listen on. |
| `refresh_interval` | duration | `1h` | How often to refresh the model list from all providers in the background. |
| `shutdown_timeout` | duration | `30s` | How long to wait for in-flight requests to finish on SIGINT/SIGTERM. |
| `max_request_body_bytes` | int | `0` (unlimited) | Reject request bodies larger than this. |
| `audit_log` | bool | `false` | Log every request/response at INFO level. Prompt and response content are logged at DEBUG level (see `--log-level`). |
| `metrics` | bool | `false` | Expose Prometheus-format metrics at `GET /metrics`. |
| `rate_limit` | object | — | Global or per-caller rate limit (see below). |

### `server.rate_limit`

| Field | Type | Default | Description |
|---|---|---|---|
| `rps` | float | — | Sustained requests per second (token bucket). |
| `burst` | int | — | Maximum burst size above `rps`. |
| `per_caller` | bool | `false` | `false` = global limit; `true` = separate bucket per `Authorization` header value (falls back to remote IP). |

### CLI flags / env vars

| Flag | Env var | Equivalent config field |
|---|---|---|
| `--config`, `-c` | `GAP_CONFIG` | — |
| `--host` | `GAP_HOST` | `server.host` |
| `--port` | `GAP_PORT` | `server.port` |
| `--log-level` | `GAP_LOG_LEVEL` | — |

`--log-level` accepts `debug`, `info`, `warn`, `error` (case-insensitive, default `info`).
Setting `debug` automatically enables the audit logger even without `audit_log: true`.

---

## `providers`

Each entry in the list describes one upstream provider instance.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string | — | **Required.** Identifier used in logs and metrics. |
| `type` | string | — | **Required.** `openai`, `litellm`, or `anthropic`. |
| `base_url` | string | — | Base URL of the upstream server (`openai` and `litellm` types). |
| `timeout` | duration | `0` | HTTP connection / response-header timeout. `0` = no limit. |
| `auth` | object | — | Authentication config (see below). |
| `token_budget` | object | — | Per-model max-token limits (see below). |
| `max_concurrent` | int | `0` | Max parallel requests to this provider. `0` = unlimited. |
| `queue_size` | int | `0` | Max requests waiting when `max_concurrent` is reached. `0` = unlimited queue. Requests beyond this limit receive `503` immediately. |
| `request_timeout` | duration | `0` | **Streaming**: max time to wait for the first token before failing over to the next provider. **Non-streaming**: total response timeout. `0` = no timeout. See notes below. |
| `model_capabilities` | map | — | Manual capability overrides: `model-id: [cap1, cap2]`. Useful when the provider does not report capabilities in `/v1/models` (e.g. LM Studio). |

### Provider types

**`type: openai`** — any OpenAI-compatible server (LM Studio, Ollama, OpenAI, OpenRouter, etc.)

**`type: litellm`** — LiteLLM instance. Identical to `openai` for request routing, but additionally calls `/model/info` at model-list refresh time to auto-discover `supports_vision` and `supports_reasoning` capabilities.

**`type: anthropic`** — Anthropic Messages API. Supports extended thinking via `budget_tokens` (see below).

### `providers[].auth`

| Field | Type | Description |
|---|---|---|
| `method` | string | `api_key` or `oauth`. |
| `api_key` | string | Static API key. Supports `${ENV_VAR}` expansion. |
| `api_keys` | list of strings | Multiple keys for round-robin rotation. Each entry supports `${ENV_VAR}`. |
| `token_file` | string | OAuth only. Path to the token cache file. Default: `~/.config/go-ai-proxy/anthropic.json`. |

### `providers[].token_budget`

Enforces a max-tokens cap before the request reaches the upstream.

| Field | Type | Description |
|---|---|---|
| `default` | int | Limit for any model from this provider not listed in `models`. `0` = no limit. |
| `models` | map[string]int | Per-model overrides, e.g. `claude-opus-4-6: 4096`. |

---

## Routing, load balancing, and failover

### Model discovery

At startup, `gap` calls `GET /v1/models` on every registered provider and builds
a routing index. The index is refreshed every `refresh_interval` in the background.

If a client requests a model that is not in the index, `gap` performs an immediate
on-demand refresh before returning `400`. This handles models that become available
in LM Studio or Ollama after the proxy started.

### Routing to providers

Multiple providers can serve the same model. `gap` keeps an ordered list of
candidates for each model ID.

On each request, `ProvidersFor` sorts the candidates by their current **active
request count** (least-connections, with registration order as the tiebreaker).
The server tries providers in that order and moves to the next on any error.

### Fallback and failover

If the primary (least-loaded) provider fails, returns `ErrQueueFull`, or is rate-limited
by its upstream (HTTP 429), the next candidate is tried automatically. This makes failover
transparent to clients.

A rate-limited provider is marked as cooling down for the duration of its `Retry-After`
value (default 60 s if the header is absent). While cooling, the provider is bypassed
without making an upstream call.

Final error codes:
- `400` — model not found after on-demand refresh
- `429` — proxy-level rate limit exceeded, **or** all upstream providers returned HTTP 429
  (response includes a `Retry-After` header with the maximum cooldown across all providers)
- `503` — all providers' queues full (`max_concurrent` + `queue_size` exhausted)
- `502` — all providers returned upstream errors

### `request_timeout` semantics

> **Streaming (time-to-first-token):** the timeout is only active until the first
> token arrives. Once streaming begins, the timeout is cancelled — slow but
> running generation is never interrupted mid-stream. If no first token arrives
> within the timeout, the stream attempt fails and the next provider is tried.
>
> **Non-streaming:** the timeout covers the entire response wait. If exceeded, the
> attempt fails and the next provider is tried.

This means `request_timeout` is safe to set even when only one provider is
configured — it only causes a `502` if the *only* provider times out, and it
never kills a generation that has already started.

---

## Concurrency limiting and load balancing across a provider group

A common setup is one cheap local provider and one paid cloud provider as fallback:

```yaml
providers:
  - name: lmstudio
    type: openai
    base_url: "http://localhost:1234"
    max_concurrent: 2      # at most 2 parallel generations (GPU limit)
    queue_size: 6          # up to 6 more requests wait in queue
    request_timeout: 20s   # fall back to cloud if no first token in 20s

  - name: anthropic
    type: anthropic
    auth:
      method: api_key
      api_key: "${ANTHROPIC_API_KEY}"
```

With this config:
- Requests go to `lmstudio` first (free, local).
- Up to 2 run in parallel; up to 6 more wait in queue.
- If `lmstudio` has no first token within 20 s → transparent failover to `anthropic`.
- If `lmstudio`'s queue is full (2 + 6 occupied) → immediate failover to `anthropic`.
- If both providers fail → `503`.

To distribute load across **multiple local instances** of the same model:

```yaml
providers:
  - name: lmstudio-1
    type: openai
    base_url: "http://localhost:1234"
    max_concurrent: 2
    queue_size: 4
    request_timeout: 20s

  - name: lmstudio-2
    type: openai
    base_url: "http://localhost:1235"
    max_concurrent: 2
    queue_size: 4
    request_timeout: 20s

  - name: anthropic
    type: anthropic
    auth:
      method: api_key
      api_key: "${ANTHROPIC_API_KEY}"
```

Both local instances expose the same models. On each request, `gap` picks the
one with fewer active requests (least-connections). When both are full, overflow
goes to `anthropic`.

---

## Capability-based routing

Instead of a concrete model name, clients can request a model by capability:

```yaml
model: "auto:vision"            # any model with vision capability
model: "auto:reasoning"         # any model with reasoning capability
model: "auto:vision,reasoning"  # both capabilities required
```

The proxy resolves the selector to the least-loaded model that satisfies all listed
capabilities. If no model matches, the request fails with `400` and a list of
available models.

Capabilities are populated (in priority order):

1. `model_capabilities` in config — manual override, highest priority
2. `/model/info` response — for `type: litellm` providers (automatic)
3. OpenRouter-style `architecture` fields in `/v1/models` — for providers that expose them

Example config:

```yaml
providers:
  - name: lmstudio
    type: openai
    base_url: "http://localhost:1234"
    model_capabilities:
      "llama-3.2-vision": ["vision"]
      "qwq-32b": ["reasoning"]

  - name: litellm
    type: litellm          # discovers capabilities automatically
    base_url: "https://litellm.example.com"
    auth:
      method: api_key
      api_key: "${LITELLM_API_KEY}"
```

---

## Embeddings

`POST /v1/embeddings` works exactly like `/v1/chat/completions`: same model
routing, same fallback/rate-limit/queue-full handling, same OpenAI wire format
(`input` as a string or array of strings; `encoding_format: "float" | "base64"`;
optional `dimensions`). There is no streaming variant.

Embedding models are tagged with the `"embeddings"` capability, the same way
`vision`/`reasoning` are tagged today — either automatically (LM Studio's
native `/api/v1/models` reports embedding-type models, and `gap` tags them
`["embeddings"]`) or manually via `model_capabilities`:

```yaml
providers:
  - name: openai
    type: openai
    base_url: "https://api.openai.com"
    auth:
      method: api_key
      api_key: "${OPENAI_API_KEY}"
    model_capabilities:
      "text-embedding-3-small": ["embeddings"]
```

This makes `auto:embeddings` work like any other capability selector. **Anthropic
has no native embeddings API** (Voyage AI is Anthropic's ecosystem answer, not
wired up here) — a request routed to a bounded Anthropic provider simply falls
through to the next candidate.

---

## Thinking / chain-of-thought

`gap` transparently extracts extended thinking content from providers that support
it and forwards it to clients as `reasoning_content` — the de-facto standard field
understood by Continue.dev and similar tools.

Supported upstream formats:

| Provider / runtime | Format |
|---|---|
| DeepSeek API, compatible servers | `reasoning_content` field in response / delta |
| Local R1, QwQ via LM Studio or Ollama | `<think>…</think>` tags embedded in `content` |
| Anthropic (extended thinking) | `thinking` content block *(Anthropic provider only)* |

For local models and DeepSeek API, no client-side configuration is required.

### Anthropic extended thinking

Pass `budget_tokens` in the request body to enable extended thinking on Claude:

```python
response = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "..."}],
    extra_body={"budget_tokens": 8000},
)
# thinking is in response.choices[0].message.reasoning_content
```

`budget_tokens` controls how many tokens Claude may spend on internal reasoning.
When set, `temperature` is ignored (Anthropic requires `temperature=1` for extended thinking).

---

## Minimal working example

```yaml
server:
  host: "127.0.0.1"
  port: 8090

providers:
  - name: lmstudio
    type: openai
    base_url: "http://localhost:1234"
```

```bash
./gap --config config.yaml --log-level debug
```

Then point any OpenAI-compatible client at `http://127.0.0.1:8090/v1`.

---

## Full example

```yaml
server:
  host: "0.0.0.0"
  port: 8090
  refresh_interval: 5m
  shutdown_timeout: 30s
  max_request_body_bytes: 10485760   # 10 MiB
  audit_log: true
  metrics: true
  rate_limit:
    rps: 10
    burst: 20
    per_caller: true

providers:
  - name: lmstudio
    type: openai
    base_url: "http://localhost:1234"
    max_concurrent: 2
    queue_size: 8
    request_timeout: 30s
    token_budget:
      default: 8192

  - name: anthropic
    type: anthropic
    auth:
      method: api_key
      api_key: "${ANTHROPIC_API_KEY}"
    token_budget:
      models:
        claude-opus-4-6: 4096
        claude-sonnet-4-6: 8192
```
