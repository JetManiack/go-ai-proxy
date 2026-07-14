# Testing Strategy

## Approach: TDD

All production code is written test-first. The cycle is: write a failing test → implement the minimum code to pass it → refactor. No non-trivial logic ships without a test written before the implementation.

## Test Layers

### Unit tests — `internal/*/`

Each package has a `_test.go` file alongside the code. Tests are pure Go, no external processes.

**Translator** is the highest-priority unit test target — it has complex logic and zero I/O, so it is fully testable without any mocks:

```
TestFromOpenAI_SimpleMessage
TestFromOpenAI_SystemMessage
TestFromOpenAI_ToolCalls
TestFromOpenAI_ConsecutiveToolResults   ← multiple tool results → single Anthropic user message
TestToOpenAI_Response
TestToOpenAI_ToolUseBlock
TestToChunk_InputJsonDelta              ← streaming tool call accumulation
```

**Registry** — model cache, routing, provider selection:
```
TestRegistry_RoutesKnownModel
TestRegistry_UnknownModelReturns400
TestRegistry_CacheRefresh
TestRegistry_SkipsFailedProvider
```

**Auth** — authenticators:
```
TestAPIKeyAuthenticator_ReturnsKey
TestOAuthAuthenticator_RefreshesExpiredToken
TestOAuthAuthenticator_PersistsTokenToDisk
```

### Integration tests — `internal/*/integration_test.go` or `test/`

Use `httptest.NewServer` to spin up real HTTP servers in-process. No mocks for HTTP — real requests over loopback.

**Provider tests** run against a fake upstream (an `httptest.Server` that replays canned responses):
```
TestLMStudioProvider_Chat
TestLMStudioProvider_ChatStream
TestLMStudioProvider_Models
TestAnthropicProvider_Chat
TestAnthropicProvider_ChatStream
TestAnthropicProvider_ToolUse
```

**Server tests** run the full stack (server + translator + registry + fake provider):
```
TestServer_ChatCompletions_NonStreaming
TestServer_ChatCompletions_Streaming
TestServer_Models
TestServer_UnknownModel_Returns400
TestServer_ClientDisconnect_CancelsUpstream
```

### End-to-end tests — `test/e2e/`

Optional, skipped in CI by default (`-tags e2e`). Require a real LM Studio instance running locally. Validate the full path from HTTP request to real model response.

## Test Helpers

- `internal/testutil/` — shared fixtures: canned OpenAI request/response JSON, fake provider implementations, assertion helpers
- Fake provider implements `domain.Provider` and returns scripted responses — used in server integration tests without network

## Coverage

Target: **≥ 80% line coverage** on `internal/translator` and `internal/server`. Provider packages are covered by integration tests against fake upstreams.

Run coverage:
```bash
go test ./... -coverprofile=coverage.out
go tool cover -html=coverage.out
```

## What is NOT unit tested

- `cmd/proxy/main.go` — wiring only, covered by integration tests
- OAuth browser flow — the part that opens a browser is behind an interface; the token exchange and persistence logic is unit tested separately
