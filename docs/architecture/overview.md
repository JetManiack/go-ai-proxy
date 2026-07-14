# Architecture Overview

## Core Principle

Providers are isolated behind a canonical internal format. No component knows about another component's wire format.

```
OpenAI client
     │  HTTP (OpenAI wire format)
     ▼
┌─────────────┐
│ HTTP Server │  /v1/chat/completions, /v1/models
└──────┬──────┘
       │  canonical Request
       ▼
┌────────────┐
│ Translator │  OpenAI ↔ canonical
└──────┬─────┘
       │  canonical Request
       ▼
┌──────────────────┐
│ Provider Registry│  selects provider by model name
└────────┬─────────┘
         │  provider-specific call
         ▼
┌─────────────────────┐
│ Provider (Anthropic)│  canonical ↔ Anthropic API
└─────────────────────┘
```

## Directory Structure

```
cmd/proxy/main.go              # entrypoint — wires everything together
internal/
  domain/provider.go           # Provider interface + canonical types
  auth/
    authenticator.go           # Authenticator interface
    apikey.go                  # static API key implementation
    oauth.go                   # browser OAuth flow + token refresh
  provider/
    anthropic/provider.go      # Anthropic implementation
    registry.go                # provider name → Provider instance
  translator/
    openai.go                  # OpenAI wire format ↔ canonical types
  server/
    server.go                  # HTTP server
config.yaml                    # server and provider configuration
go.mod
```

## Canonical Types

All components communicate through types defined in `internal/domain/provider.go`. These types are deliberately provider-agnostic.

| Type | Purpose |
|---|---|
| `Request` | Chat request: messages, tools, temperature, max_tokens, stream flag |
| `Message` | Role + content + tool_calls + tool_call_id |
| `Tool` | Name + description + parameters (JSON Schema) |
| `Response` | Message + token usage (prompt/completion) |
| `Chunk` | Streaming delta: content string or tool call fragment |

## Provider Interface

```go
type Provider interface {
    Chat(ctx context.Context, req Request) (Response, error)
    ChatStream(ctx context.Context, req Request) (<-chan Chunk, error)
    Models(ctx context.Context) ([]Model, error)
}
```

`Models()` makes a real API call to the upstream provider. The registry calls it at startup and then on a background ticker to keep the list fresh. Routing and `GET /v1/models` both use the cached list — no network call per request.

Only providers that are configured in `config.yaml` and successfully initialized (auth bootstrapped) are registered. An unrecognized model name returns a `400` listing what is actually available.

## Adding a New Provider

1. Create `internal/provider/<name>/provider.go`
2. Implement `domain.Provider`
3. Register in `internal/provider/registry.go`
4. Add a config section in `config.yaml`
