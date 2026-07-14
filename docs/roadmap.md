# Roadmap

## Phase 1 — Core Proxy (MVP)

Goal: a working proxy validated end-to-end against a local LM Studio instance —
no auth complexity, no external API calls, instant feedback loop.

LM Studio and Ollama already speak the OpenAI wire format, so the provider
implementation is a thin passthrough. This lets us prove out the server,
translator, registry, and model cache before touching Anthropic.

- [x] `go.mod` — module init, add `github.com/urfave/cli/v3`
- [x] `internal/testutil/` — shared test fixtures and fake provider
- [x] `internal/domain/provider.go` — canonical types and `Provider` interface
- [x] `internal/translator/openai.go` + tests — OpenAI ↔ canonical translation (non-streaming)
- [x] `internal/provider/lmstudio/provider.go` + integration tests against fake upstream —
      OpenAI-compatible passthrough; configurable `base_url`; no auth
- [x] `internal/provider/registry.go` + tests — startup model fetch, background refresh,
      model→provider routing
- [x] `internal/server/server.go` + integration tests — HTTP server,
      `/v1/chat/completions`, `/v1/models`
- [x] `cmd/gap/main.go` — entrypoint, config loading, wiring
- [x] `config.yaml` — server + LM Studio provider config

**Exit criteria:** `go run ./cmd/gap` starts; LM Studio running locally; a curl to
`/v1/chat/completions` returns a valid response; `/v1/models` lists models fetched
from LM Studio; `go test ./...` passes with ≥ 80% coverage on translator and server.

---

## Phase 2 — Anthropic Provider

Goal: add Anthropic as a second provider, introducing the upstream auth layer.

- [x] `internal/auth/authenticator.go` — `Authenticator` interface
- [x] `internal/auth/apikey.go` — static API key authenticator
- [x] `internal/auth/oauth.go` — browser OAuth 2.0 PKCE flow: open browser →
      local callback server → token exchange → persist to disk → auto-refresh
- [x] `internal/provider/anthropic/provider.go` — Anthropic provider (non-streaming),
      accepts `Authenticator`
- [x] `go.mod` — add `github.com/anthropics/anthropic-sdk-go`
- [x] `config.yaml` — add Anthropic provider section; `auth.method: oauth | api_key`

**Exit criteria:** first run opens a browser for OAuth if no token file exists; both
LM Studio and Anthropic models appear in `/v1/models`; routing sends `claude-*`
requests to Anthropic and local model requests to LM Studio.

---

## Phase 3 — Streaming

Goal: support `stream: true` for real-time output.

- [x] Anthropic provider: `ChatStream()` via Anthropic's SSE event stream
- [x] Translator: `ChunkToOpenAI()` / `ChunkFromOpenAI()` — convert stream events
      to `domain.Chunk`
- [x] Server: SSE response loop — `data: {...}` frames, `data: [DONE]` termination
- [x] Graceful context cancellation when client disconnects mid-stream

**Exit criteria:** `stream: true` produces incremental output; client disconnect stops
upstream consumption without goroutine leak.

---

## Phase 4 — Tool Calling

Goal: multi-turn tool-calling conversations work correctly, including parallel tool calls.

Note: tool call translation lives in `internal/provider/anthropic/provider.go`, not in
the translator (which only handles OpenAI wire format ↔ canonical domain types).
OpenAI-to-OpenAI tool calling needs no translation — it is a passthrough.

- [x] Anthropic provider: `role: "system"` extracted to top-level `System` field
- [x] Anthropic provider: consecutive `role: "tool"` messages grouped into a single
      user message with `tool_result` content blocks
- [x] Anthropic provider: assistant `tool_calls` → `tool_use` content blocks
- [x] Anthropic provider: `tool_use` response blocks → OpenAI `tool_calls`
- [x] Tests: unit tests for tool calling paths in the Anthropic provider and the OpenAI
      translator (`tool_calls` round-trip in `RequestFromOpenAI`/`RequestToOpenAI`)
- [x] Streaming: accumulate `input_json_delta` events from Anthropic, emit as OpenAI
      `tool_calls[].function.arguments` deltas
- [ ] End-to-end test: multi-turn conversation with parallel tool calls round-trips correctly
      *(requires Anthropic API access; meaningful only for providers with translation)*

**Exit criteria:** an agentic loop with parallel tool calls completes without translation
errors.

---

## Phase 5 — Hardening

Goal: production-grade reliability, observability, and operational controls for
long-running and high-concurrency workloads.

- [x] Structured logging (slog) with request IDs — logging middleware sets `X-Request-ID`
      (validated, not echoed verbatim), logs method/path/status/duration per request
- [x] Per-provider timeout configuration — `WithTimeout` on both providers; `timeout:` in config
- [x] Graceful shutdown: drain in-flight requests before exit — SIGINT/SIGTERM triggers
      `http.Server.Shutdown` with configurable drain timeout (`server.shutdown_timeout`,
      default 30s)
- [x] Health check endpoint (`GET /healthz`) — returns `{"status":"ok"}`
- [x] Configurable request body size limit — `server.max_request_body_bytes` in config;
      returns 413 on exceed
- [x] Basic metrics (request count, latency, token usage per provider/model) —
      Prometheus text format, no external deps; `GET /metrics`;
      `server.metrics: true` in config
- [x] Request audit log — structured slog entries (`msg:"audit"`) for each chat
      request/response; `server.audit_log: true` in config; `WithAuditLog(*slog.Logger)`
      option; streaming emits `stream_start` + `stream_end` events
- [x] Token budget enforcement — 400 when `max_tokens` exceeds configured limit;
      per-model overrides + global default; checked before routing, no upstream call made;
      `provider.token_budget.default/models` in config
- [x] Rate limiting — token-bucket rate limiter (`golang.org/x/time/rate`); global or
      per-caller (keyed by SHA-256 of `Authorization` value, or IP with optional
      `X-Forwarded-For` trust); `server.rate_limit.rps/burst/per_caller/trust_proxy`
      in config; 429 on exceed
- [x] Multi-key round-robin — `auth.api_keys: [...]` alongside `auth.api_key`;
      `RoundRobinAuthenticator` uses `atomic.Uint64` to cycle keys without locks;
      single key falls back to `APIKeyAuthenticator`
- [x] Config schema validation at startup — `yaml.Decoder.KnownFields(true)` rejects
      unknown keys; required-field checks in `buildProvider`
- [x] Fallback chain — registry keeps ordered provider list per model (`ProvidersFor`);
      server tries each on upstream error; fallback order = provider order in config;
      works for both chat and streaming (pre-header fallback only)
- [x] On-demand model refresh — if a requested model is not in the registry cache,
      trigger an immediate refresh before returning 404; handles models loaded into
      LM Studio after the proxy started without requiring a restart
- [x] Upstream rate-limit tracking — providers return `RateLimitError{RetryAfter}` on
      HTTP 429; `BoundedProvider` marks the provider as cooling down for the indicated
      duration and short-circuits subsequent requests; `Retry-After` header is parsed
      (delay-seconds or HTTP-date, default 60s); when all candidates are rate-limited
      the server returns 429 with a `Retry-After` header to the client; a single
      rate-limited provider still triggers fallback to the next candidate

---

## Phase 6 — Static Proxy Auth

Goal: close threat-model findings S1 and D1 — any client on the network can use the
proxy as an open relay, exhausting upstream API quotas at the operator's expense.
A bearer-token check at the proxy boundary covers single-operator and small-team
deployments without requiring a database or identity provider.

Clients pass `api_key` (i.e. `Authorization: Bearer <token>`) as they already do
with the OpenAI SDK; the proxy validates it against a configured list before
forwarding the request. If `proxy_auth` is absent, the proxy remains auth-free,
preserving backwards compatibility for deployments that rely on network isolation.

- [ ] `internal/server/proxyauth.go` — `ProxyAuthEntry`, `ProxyAuthConfig`,
      `NewProxyAuthConfig` (hashes each token to `sha256 → [32]byte` at startup;
      plain-text not retained after init), `proxyAuthMiddleware` (validates `Bearer`
      header; 401 + `WWW-Authenticate: Bearer` on failure; `/healthz` bypassed)
- [ ] `internal/server/middleware.go` — `callerNameKey contextKey = 1`;
      `callerNameFromContext` helper
- [ ] `internal/server/server.go` — `proxyAuthCfg *ProxyAuthConfig` field;
      `WithProxyAuth` option; insert middleware before `rateLimitMiddleware` so
      unauthenticated requests don't consume rate-limit buckets
- [ ] `internal/server/ratelimit.go` — `callerKey` reads `callerNameFromContext` as
      first priority; per-token buckets with human-readable names instead of hashed values
- [ ] `internal/server/audit.go` — add `"caller"` field (token name) to INFO entries
      in `auditChat` and `auditStreamStart`; omit when empty so auth-free deployments
      produce clean logs
- [ ] `cmd/gap/main.go` — `proxy_auth.tokens[]{name, token}` config section;
      env-var expansion (`${VAR}`); startup validation: empty list, duplicate names,
      blank token after expansion are all fatal errors
- [ ] `internal/server/proxyauth_test.go` — valid/invalid/missing/malformed token
      → 200/401; `/healthz` always 200 regardless of token; `WWW-Authenticate` header
      present on 401; two named tokens share no rate-limit bucket; audit log carries
      `caller` field

Config example:
```yaml
server:
  proxy_auth:
    tokens:
      - name: "alice"
        token: "${PROXY_TOKEN_ALICE}"
      - name: "ci-bot"
        token: "${CI_BOT_TOKEN}"
```

**Exit criteria:** proxy with `proxy_auth` configured returns 401 for unknown tokens
and 200 for known ones; `/healthz` reachable without a token; `go test -race ./...`
passes; `caller` field appears in audit log entries.

---

## Phase 7 — Auth UI & Personal Access Tokens

Goal: multi-user deployments where individual users authenticate via an external
identity provider and self-manage long-lived tokens through a web UI.

Phase 6 provides the validation infrastructure (token hash lookup, `callerName` in
context, per-caller rate limit buckets). Phase 7 replaces the static token list with
a dynamic store and adds a login flow, making the proxy suitable for team or
organisation-wide deployments where tokens need to be issued and revoked without
editing config files.

Users open a browser, log in via OIDC (Google, Keycloak, etc.), and receive a
Personal Access Token (PAT) to paste as `api_key` in their client config.
The proxy validates PATs on every incoming request using the same middleware
pattern established in Phase 6.

- [ ] OIDC login flow — redirect to IdP, handle callback, verify ID token
      (`github.com/coreos/go-oidc/v3`); `server.oidc.issuer_url` + optional
      `audience` in config
- [ ] PAT storage — SQLite (embedded, no external service); table:
      `(id, user_sub, email, name, token_hash, created_at, last_used_at, expires_at)`;
      token is a random 32-byte base64 value, only the SHA-256 hash stored
- [ ] PAT management UI — minimal server-side HTML (`html/template`); pages: login,
      token list, create token, revoke token
- [ ] Inbound PAT validation middleware — replaces Phase 6 static-token middleware
      when OIDC is configured; `Authorization: Bearer <PAT>` → hash lookup →
      user identity (`sub`, `email`) into context
- [ ] Identity propagation — audit log gains `user` field (`sub`); rate-limit key
      uses `sub` when available, falls back to Phase 6 token name, then IP
- [ ] Config: `server.oidc` section; `server.db` for SQLite path (default `gap.db`);
      `server.proxy_auth` from Phase 6 remains valid and takes effect when `server.oidc`
      is absent
- [ ] mTLS — optional alternative to PAT for service-to-service deployments;
      `tls.Config{ClientAuth: RequireAndVerifyClientCert}`;
      `server.mtls.ca_cert` in config

---

## Phase 8 — More Providers

Goal: providers that genuinely require new code — different auth schemes or wire
formats not covered by the `type: openai` passthrough.

OpenAI-compatible providers (Groq, Together AI, OpenRouter, Ollama, Gemini via
OpenAI-compat endpoint, xAI, Perplexity, Fireworks, etc.) are already supported —
just add a `type: openai` entry with the right `base_url` in `config.yaml`.

- [ ] Azure OpenAI provider — same OpenAI wire format, no translation needed;
      adapter only: `api-key` header auth, deployment-based URLs
      (`/openai/deployments/{name}/chat/completions`), `api-version` query param;
      model name maps to deployment name
- [ ] Amazon Bedrock provider — AWS Signature V4 auth; translation layer
      (canonical ↔ Bedrock Converse API) inside `provider/bedrock/`, same pattern
      as the Anthropic provider; routes Claude, Titan, Mistral, Llama
- [ ] Cohere provider — translation layer (canonical ↔ Cohere `/v2/chat`) inside
      `provider/cohere/`; covers Cohere-specific message format and tool calling
      not available via their OpenAI-compat shim
- [ ] Document all OpenAI-compatible services in `config.yaml` examples
      (Groq, Together AI, OpenRouter, Ollama, Gemini, xAI, Perplexity, Fireworks)

---

## Future Ideas

Not scheduled, but worth tracking:

- **Semantic caching** — cache responses for identical or near-identical prompts
  to reduce API cost and latency
- **Admin API** — runtime inspection of registered providers, active request counts,
  and per-caller token usage; complements the Prometheus metrics endpoint
- **Streaming total timeout** — per-request deadline covering the full stream
  duration, not just first-token latency; guards against slow upstream responses
  holding goroutines indefinitely

### Deferred from vLLM provider design (2026-06-14, see specs/2026-06-14-vllm-provider-design.md)

- **`refusal` propagation** — surface OpenAI `message.refusal` to downstream
  clients with unified normalisation across providers (Anthropic
  `stop_reason: refusal`, OpenAI `message.refusal`, DeepSeek equivalents).
- **`routed_experts` MoE diagnostic** — vLLM-specific; optional Prometheus counter
  for MoE expert routing observability.
- **`audio` field handling** — gated on adding an audio-generating provider.
- **`annotations` field handling** — gated on adding a search/RAG provider that
  emits structured citations.
- **`system_fingerprint` propagation** — response-metadata pass-through to client.
- **Non-`"stop"` `finish_reason` handling** — proper streaming behaviour for
  `length`, `tool_calls`, `content_filter`. Requires adding `FinishReason` to
  `domain.Response`/`domain.Chunk` and propagating through translator + providers.
- **[optional]** `finish_reason: "length"` info-log when content is empty — spec
  §6 promised this for UX visibility, but the client already sees `finish_reason`
  in its own response and the audit log already captures it. Marginal operator
  value; reconsider if oncall actually grep logs for this pattern.
- **Context-window awareness** — parse `MaxModelLen` from vLLM `/v1/models`,
  optionally validate / clamp `prompt + max_tokens` server-side before forwarding.
- **[backlog] Legacy `function_call` parsing** — accept the deprecated single-call
  format on input and translate to canonical `tool_calls` on output, for
  downstream clients using pre-2024 OpenAI clients.
