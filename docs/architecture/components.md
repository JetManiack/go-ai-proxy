# Components

## HTTP Server (`internal/server/server.go`)

Speaks the OpenAI HTTP wire format. Knows nothing about providers or translation.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Main route — streaming and non-streaming |
| `GET` | `/v1/models` | Returns models aggregated from all providers |

**Auth:** The proxy forwards `Authorization: Bearer <key>` to the upstream provider as-is. It does not enforce authentication itself — intended for local or trusted-network use.

**Streaming:** When `stream: true` is set in the request, the server opens a Server-Sent Events (SSE) connection and forwards chunks from the provider channel to the client as `data: {...}\n\n` frames, terminating with `data: [DONE]\n\n`.

---

## Translator (`internal/translator/openai.go`)

The most complex component. Converts between the OpenAI wire format and the canonical internal types in both directions.

**Key translation responsibilities:**

| Concern | OpenAI | Canonical / Anthropic |
|---|---|---|
| System messages | `messages[]` entry with `role: "system"` | Separate `system` field |
| Tool results | `role: "tool"` message with `tool_call_id` | `tool_result` content block inside a user message |
| Assistant tool calls | `tool_calls: [...]` array on assistant message | `type: "tool_use"` content blocks |
| Streaming chunks | `choices[].delta` SSE frames | `<-chan Chunk` |

---

## Provider Registry (`internal/provider/registry.go`)

Holds only the providers that are present in `config.yaml` **and** successfully initialized at startup (credentials present, auth bootstrapped). A provider that fails to initialize is logged and skipped — it does not enter the registry.

**Model cache:** at startup the registry calls `Models()` on each registered provider and stores the result. A background goroutine refreshes the cache on a configurable interval (default: 1 hour, configurable via `server.refresh_interval`). Routing and `GET /v1/models` both read from this cache — no upstream API call per request.

**Routing:** at request time the registry looks up the model ID from the incoming request:

1. **Exact match** — the model ID is in the index (e.g. `claude-sonnet-4-6`).
2. **Glob match** — the model ID matches a wildcard entry (e.g. `openai/*` matches `openai/gpt-4o`). Glob patterns use `path.Match` syntax.

If no match is found the server returns a `400` listing the models that are currently known.

**Conflict resolution:** when two providers expose the same model ID or a matching glob pattern, the **first provider listed in `config.yaml` wins**. The second is silently skipped and a `DEBUG` log entry is emitted. This makes priority explicit and deterministic — reorder entries in `config.yaml` to change routing preference.

---

## Authenticator (`internal/auth/`)

Abstracts how the proxy obtains a token for upstream API calls. The provider does not know which method is in use — it calls `GetToken()` and gets a string back.

```go
type Authenticator interface {
    GetToken(ctx context.Context) (string, error)
}
```

**Implementations:**

### `APIKeyAuthenticator` (`internal/auth/apikey.go`)

Returns a static API key from config or environment variable. Simplest case — works when the caller has a direct `sk-ant-...` key.

### `OAuthAuthenticator` (`internal/auth/oauth.go`)

Browser-based OAuth 2.0 flow, modeled after what Claude Code does internally. Required when only a claude.ai account is available (no direct API key).

**Flow:**
1. On first use: open the system browser to Anthropic's OAuth authorization URL
2. Start a local HTTP server on a loopback port to catch the OAuth callback
3. Exchange the authorization code for an access token + refresh token
4. Persist tokens to a local token file (e.g. `~/.config/gap/anthropic_tokens.json`)
5. On subsequent calls: return the cached access token; refresh transparently when expired

**Token storage:** tokens are written to disk with `0600` permissions. The token file path is configurable.

---

## Providers

### OpenAI-compatible (`internal/provider/openai/`)

Universal passthrough for any server that speaks the OpenAI HTTP API: LM Studio, Ollama, OpenAI itself, OpenRouter, LiteLLM, etc.

- Optional `Authorization: Bearer <key>` header via `WithAPIKey` — omit for local servers that don't require auth.
- No translation needed: requests and responses are forwarded as OpenAI JSON.

### Anthropic (`internal/provider/anthropic/`)

Implements `domain.Provider` using `github.com/anthropics/anthropic-sdk-go`.

- Obtains a token via the injected `Authenticator` before each request.
- Translates `domain.Request` → Anthropic Messages API: extracts `system` messages, maps `tool_calls` ↔ `tool_use` content blocks, groups consecutive `tool` role messages into a single user message with `tool_result` blocks.

---

## Configuration (`config.yaml`)

`providers` is a **list** — multiple instances of the same type are supported. The first entry that claims a model ID wins (see Registry conflict resolution above).

```yaml
server:
  host: "127.0.0.1"
  port: 8090
  refresh_interval: 1h   # how often to refresh the model list from all providers

providers:
  - name: lmstudio        # display name, used in logs
    type: openai          # "openai" or "anthropic"
    base_url: "http://localhost:1234"

  - name: litellm
    type: openai
    base_url: "https://litellm.example.com"
    auth:
      method: api_key
      api_key: "${LITELLM_API_KEY}"   # ${ENV_VAR} is expanded at startup

  - name: anthropic
    type: anthropic
    auth:
      method: api_key          # "api_key" or "oauth"
      api_key: "${ANTHROPIC_API_KEY}"
```

The model name from the request is passed through as-is to the upstream. Callers use native model IDs (e.g. `claude-sonnet-4-6`, `openai/gpt-4o`).

---

## Entrypoint (`cmd/gap/main.go`)

Built with `github.com/urfave/cli/v3`. Resolution order: CLI flag > env var > YAML config > default.

```
./gap [--config config.yaml] [--port 8090] [--host 127.0.0.1]
```

Env vars: `GAP_CONFIG`, `GAP_PORT`, `GAP_HOST`. The entrypoint iterates the `providers` list, constructs each provider via `buildProvider`, registers them in the registry, and starts the HTTP server.
