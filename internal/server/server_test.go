package server_test

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
	"github.com/JetManiack/go-ai-proxy/internal/metrics"
	"github.com/JetManiack/go-ai-proxy/internal/provider"
	"github.com/JetManiack/go-ai-proxy/internal/server"
	"github.com/JetManiack/go-ai-proxy/internal/testutil"
)

func newTestServer(t *testing.T, providers ...domain.Provider) *httptest.Server {
	t.Helper()
	return newTestServerOpts(t, nil, providers...)
}

func newTestServerOpts(t *testing.T, opts []server.Option, providers ...domain.Provider) *httptest.Server {
	t.Helper()
	reg := provider.NewRegistry(time.Hour)
	for _, p := range providers {
		reg.Register(p)
	}
	if err := reg.Start(context.Background()); err != nil {
		t.Fatalf("registry Start: %v", err)
	}
	srv := server.New(reg, opts...)
	return httptest.NewServer(srv)
}

func chatRequest(t *testing.T, srv *httptest.Server, body any) *http.Response {
	t.Helper()
	b, _ := json.Marshal(body)
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", bytes.NewReader(b))
	if err != nil {
		t.Fatalf("POST /v1/chat/completions: %v", err)
	}
	return resp
}

// --- /v1/chat/completions ---

func TestChatCompletions_OK(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "test-model"})
	fp.ChatFunc = func(_ context.Context, req domain.Request) (domain.Response, error) {
		return domain.Response{
			ID:      "resp-1",
			Model:   req.Model,
			Message: domain.Message{Role: "assistant", Content: "hello"},
			Usage:   domain.Usage{PromptTokens: 2, CompletionTokens: 1},
		}, nil
	}

	srv := newTestServer(t, fp)
	defer srv.Close()

	resp := chatRequest(t, srv, map[string]any{
		"model":    "test-model",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status: got %d, want 200", resp.StatusCode)
	}

	var body map[string]any
	json.NewDecoder(resp.Body).Decode(&body)

	choices := body["choices"].([]any)
	if len(choices) == 0 {
		t.Fatal("expected choices")
	}
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	if msg["content"] != "hello" {
		t.Errorf("content: got %v", msg["content"])
	}
}

func TestChatCompletions_UnknownModel(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "known-model"})
	srv := newTestServer(t, fp)
	defer srv.Close()

	resp := chatRequest(t, srv, map[string]any{
		"model":    "unknown-model",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status: got %d, want 400", resp.StatusCode)
	}

	var body map[string]any
	json.NewDecoder(resp.Body).Decode(&body)
	if _, ok := body["error"]; !ok {
		t.Error("expected error field in response")
	}
}

func TestChatCompletions_InvalidJSON(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	srv := newTestServer(t, fp)
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", bytes.NewBufferString("not json"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status: got %d, want 400", resp.StatusCode)
	}
}

func TestChatCompletions_ContentTypeIsJSON(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	srv := newTestServer(t, fp)
	defer srv.Close()

	resp := chatRequest(t, srv, map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	defer resp.Body.Close()

	ct := resp.Header.Get("Content-Type")
	if ct != "application/json" {
		t.Errorf("Content-Type: got %q, want application/json", ct)
	}
}

func TestChatCompletions_WrongMethod(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	srv := newTestServer(t, fp)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/v1/chat/completions")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Errorf("status: got %d, want 405", resp.StatusCode)
	}
}

// --- /v1/models ---

func TestModels_ReturnsAllModels(t *testing.T) {
	fp1 := testutil.NewFakeProvider(domain.Model{ID: "model-a", OwnedBy: "prov"})
	fp2 := testutil.NewFakeProvider(domain.Model{ID: "model-b", OwnedBy: "prov"})
	srv := newTestServer(t, fp1, fp2)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatalf("GET /v1/models: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status: got %d, want 200", resp.StatusCode)
	}

	var body map[string]any
	json.NewDecoder(resp.Body).Decode(&body)

	data := body["data"].([]any)
	if len(data) != 2 {
		t.Errorf("models count: got %d, want 2", len(data))
	}
}

func TestModels_WrongMethod(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	srv := newTestServer(t, fp)
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/v1/models", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Errorf("status: got %d, want 405", resp.StatusCode)
	}
}

// --- Streaming ---

func TestChatCompletions_StreamingContentType(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	fp.StreamFunc = func(_ context.Context, _ domain.Request) (<-chan domain.Chunk, error) {
		ch := make(chan domain.Chunk, 1)
		ch <- domain.Chunk{Done: true}
		close(ch)
		return ch, nil
	}
	srv := newTestServer(t, fp)
	defer srv.Close()

	resp := chatRequest(t, srv, map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
		"stream":   true,
	})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status: got %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type: got %q, want text/event-stream", ct)
	}
}

func TestChatCompletions_StreamingFrames(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	fp.StreamFunc = func(_ context.Context, _ domain.Request) (<-chan domain.Chunk, error) {
		ch := make(chan domain.Chunk, 3)
		ch <- domain.Chunk{ID: "c1", Model: "m", Delta: "Hello"}
		ch <- domain.Chunk{ID: "c1", Model: "m", Delta: " world"}
		ch <- domain.Chunk{ID: "c1", Model: "m", Done: true}
		close(ch)
		return ch, nil
	}
	srv := newTestServer(t, fp)
	defer srv.Close()

	resp := chatRequest(t, srv, map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
		"stream":   true,
	})
	defer resp.Body.Close()

	frames := readSSEFrames(t, resp.Body)

	// Last frame must be [DONE].
	if len(frames) == 0 || frames[len(frames)-1] != "[DONE]" {
		t.Fatalf("expected last frame to be [DONE], got frames: %v", frames)
	}

	// First frame must carry the first delta.
	var chunk map[string]any
	if err := json.Unmarshal([]byte(frames[0]), &chunk); err != nil {
		t.Fatalf("parse first frame: %v", err)
	}
	choices := chunk["choices"].([]any)
	delta := choices[0].(map[string]any)["delta"].(map[string]any)
	if delta["content"] != "Hello" {
		t.Errorf("first delta content: got %v, want Hello", delta["content"])
	}
}

func TestChatCompletions_StreamingContextCancel(t *testing.T) {
	released := make(chan struct{})
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	fp.StreamFunc = func(ctx context.Context, _ domain.Request) (<-chan domain.Chunk, error) {
		ch := make(chan domain.Chunk)
		go func() {
			defer close(ch)
			defer close(released)
			<-ctx.Done() // wait for cancellation
		}()
		return ch, nil
	}

	srv := newTestServer(t, fp)
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, srv.URL+"/v1/chat/completions",
		bytes.NewBufferString(`{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true}`))
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}

	// Cancel the client context — this closes the connection.
	cancel()
	resp.Body.Close()

	select {
	case <-released:
		// provider goroutine stopped
	case <-time.After(2 * time.Second):
		t.Error("provider goroutine did not stop after client disconnect")
	}
}

// readSSEFrames reads an SSE response body and returns the data payloads.
func readSSEFrames(t *testing.T, body io.Reader) []string {
	t.Helper()
	var frames []string
	scanner := bufio.NewScanner(body)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			frames = append(frames, strings.TrimPrefix(line, "data: "))
		}
	}
	return frames
}

// --- /healthz ---

func TestHealthz_OK(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	srv := newTestServer(t, fp)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status: got %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type: got %q", ct)
	}
	var body map[string]any
	json.NewDecoder(resp.Body).Decode(&body)
	if body["status"] != "ok" {
		t.Errorf("body: got %v", body)
	}
}

func TestHealthz_WrongMethod(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	srv := newTestServer(t, fp)
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/healthz", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Errorf("status: got %d, want 405", resp.StatusCode)
	}
}

// --- Body size limit ---

func TestChatCompletions_BodyTooLarge(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	srv := newTestServerOpts(t, []server.Option{server.WithMaxBodyBytes(100)}, fp)
	defer srv.Close()

	// Build a body larger than 100 bytes.
	big := make([]byte, 200)
	for i := range big {
		big[i] = 'x'
	}
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", bytes.NewReader(big))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Errorf("status: got %d, want 413", resp.StatusCode)
	}
}

// --- Request ID ---

func TestLogging_RequestIDHeader(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	srv := newTestServer(t, fp)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.Header.Get("X-Request-ID") == "" {
		t.Error("expected X-Request-ID header in response")
	}
}

// --- Fallback chain ---

func TestFallback_SecondProviderUsedOnError(t *testing.T) {
	primary := testutil.NewFakeProvider(domain.Model{ID: "m"})
	primary.ChatFunc = func(_ context.Context, _ domain.Request) (domain.Response, error) {
		return domain.Response{}, errors.New("primary down")
	}
	fallback := testutil.NewFakeProvider(domain.Model{ID: "m"})
	fallback.ChatFunc = func(_ context.Context, req domain.Request) (domain.Response, error) {
		return domain.Response{
			ID:      "fb",
			Model:   req.Model,
			Message: domain.Message{Role: "assistant", Content: "from fallback"},
		}, nil
	}

	srv := newTestServer(t, primary, fallback)
	defer srv.Close()

	resp := chatRequest(t, srv, map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status: got %d, want 200", resp.StatusCode)
	}
	var body map[string]any
	json.NewDecoder(resp.Body).Decode(&body)
	choices := body["choices"].([]any)
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	if msg["content"] != "from fallback" {
		t.Errorf("content: got %v, want 'from fallback'", msg["content"])
	}
}

func TestFallback_AllProvidersFail_Returns502(t *testing.T) {
	p1 := testutil.NewFakeProvider(domain.Model{ID: "m"})
	p1.ChatFunc = func(_ context.Context, _ domain.Request) (domain.Response, error) {
		return domain.Response{}, errors.New("p1 down")
	}
	p2 := testutil.NewFakeProvider(domain.Model{ID: "m"})
	p2.ChatFunc = func(_ context.Context, _ domain.Request) (domain.Response, error) {
		return domain.Response{}, errors.New("p2 down")
	}

	srv := newTestServer(t, p1, p2)
	defer srv.Close()

	resp := chatRequest(t, srv, map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadGateway {
		t.Errorf("status: got %d, want 502", resp.StatusCode)
	}
}

func TestFallback_Stream_SecondProviderUsedOnError(t *testing.T) {
	primary := testutil.NewFakeProvider(domain.Model{ID: "m"})
	primary.StreamFunc = func(_ context.Context, _ domain.Request) (<-chan domain.Chunk, error) {
		return nil, errors.New("primary stream down")
	}
	fallback := testutil.NewFakeProvider(domain.Model{ID: "m"})
	fallback.StreamFunc = func(_ context.Context, _ domain.Request) (<-chan domain.Chunk, error) {
		ch := make(chan domain.Chunk, 2)
		ch <- domain.Chunk{ID: "c1", Model: "m", Delta: "ok"}
		ch <- domain.Chunk{Done: true}
		close(ch)
		return ch, nil
	}

	srv := newTestServer(t, primary, fallback)
	defer srv.Close()

	resp := chatRequest(t, srv, map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
		"stream":   true,
	})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status: got %d, want 200", resp.StatusCode)
	}
	frames := readSSEFrames(t, resp.Body)
	if len(frames) == 0 || frames[len(frames)-1] != "[DONE]" {
		t.Errorf("expected [DONE] frame, got %v", frames)
	}
}

// --- Concurrency limit ---

func TestChatCompletions_QueueFull_Returns503(t *testing.T) {
	fp := &testutil.FakeProvider{
		ModelsFunc: func(_ context.Context) ([]domain.Model, error) {
			return []domain.Model{{ID: "m"}}, nil
		},
		ChatFunc: func(_ context.Context, _ domain.Request) (domain.Response, error) {
			return domain.Response{}, provider.ErrQueueFull
		},
	}

	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatalf("registry Start: %v", err)
	}
	srv := server.New(reg)
	ts := httptest.NewServer(srv)
	defer ts.Close()

	resp := chatRequest(t, ts, map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status: got %d, want 503", resp.StatusCode)
	}
}

// --- On-demand model refresh ---

func TestChatCompletions_OnDemandRefreshSucceeds(t *testing.T) {
	var mu sync.Mutex
	models := []domain.Model{{ID: "existing-model"}}

	fp := &testutil.FakeProvider{
		ModelsFunc: func(_ context.Context) ([]domain.Model, error) {
			mu.Lock()
			defer mu.Unlock()
			out := make([]domain.Model, len(models))
			copy(out, models)
			return out, nil
		},
	}
	fp.ChatFunc = func(_ context.Context, req domain.Request) (domain.Response, error) {
		return domain.Response{
			ID:      "resp",
			Model:   req.Model,
			Message: domain.Message{Role: "assistant", Content: "ok"},
		}, nil
	}

	srv := newTestServer(t, fp)
	defer srv.Close()

	// Simulate a model becoming available after the registry started.
	mu.Lock()
	models = append(models, domain.Model{ID: "new-model"})
	mu.Unlock()

	// Request for the new model should succeed via on-demand refresh.
	resp := chatRequest(t, srv, map[string]any{
		"model":    "new-model",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status: got %d, want 200; body: %s", resp.StatusCode, body)
	}
}

// --- Not Found ---

func TestUnknownPath_Returns404(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	srv := newTestServer(t, fp)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/v1/unknown")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("status: got %d, want 404", resp.StatusCode)
	}
}

// --- auto: cross-model fallback ---

func TestAutoSelector_CrossModelFallback_Chat(t *testing.T) {
	// fp1 has vision-a but always fails
	fp1 := testutil.NewFakeProvider(domain.Model{ID: "vision-a", Capabilities: []string{"vision"}})
	fp1.ChatFunc = func(_ context.Context, _ domain.Request) (domain.Response, error) {
		return domain.Response{}, errors.New("provider unavailable")
	}
	// fp2 has vision-b and succeeds
	fp2 := testutil.NewFakeProvider(domain.Model{ID: "vision-b", Capabilities: []string{"vision"}})
	fp2.ChatFunc = func(_ context.Context, req domain.Request) (domain.Response, error) {
		return domain.Response{
			ID:      "resp-2",
			Model:   req.Model,
			Message: domain.Message{Role: "assistant", Content: "from-" + req.Model},
		}, nil
	}

	srv := newTestServer(t, fp1, fp2)
	defer srv.Close()

	resp := chatRequest(t, srv, map[string]any{
		"model":    "auto:vision",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status: got %d, want 200; body: %s", resp.StatusCode, body)
	}
	var body map[string]any
	json.NewDecoder(resp.Body).Decode(&body)
	choices := body["choices"].([]any)
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	if msg["content"] != "from-vision-b" {
		t.Errorf("expected fallback to vision-b, got content %v", msg["content"])
	}
}

// --- upstream rate-limit passthrough ---

// TestChatCompletions_AllRateLimited_Returns429 verifies that when all providers
// return RateLimitError the server returns 429 with a Retry-After header.
func TestChatCompletions_AllRateLimited_Returns429(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	fp.ChatFunc = func(_ context.Context, _ domain.Request) (domain.Response, error) {
		return domain.Response{}, &provider.RateLimitError{RetryAfter: 30 * time.Second}
	}
	srv := newTestServer(t, fp)
	defer srv.Close()

	resp := chatRequest(t, srv, map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("status: got %d, want 429", resp.StatusCode)
	}
	if ra := resp.Header.Get("Retry-After"); ra != "30" {
		t.Errorf("Retry-After: got %q, want %q", ra, "30")
	}
}

// TestChatCompletions_FallbackOnRateLimit_Returns200 verifies that when the first
// provider is rate-limited but a second provider succeeds, the response is 200.
func TestChatCompletions_FallbackOnRateLimit_Returns200(t *testing.T) {
	fp1 := testutil.NewFakeProvider(domain.Model{ID: "m"})
	fp1.ChatFunc = func(_ context.Context, _ domain.Request) (domain.Response, error) {
		return domain.Response{}, &provider.RateLimitError{RetryAfter: 60 * time.Second}
	}
	fp2 := testutil.NewFakeProvider(domain.Model{ID: "m"})
	fp2.ChatFunc = func(_ context.Context, req domain.Request) (domain.Response, error) {
		return domain.Response{
			ID:      "ok",
			Model:   req.Model,
			Message: domain.Message{Role: "assistant", Content: "fallback ok"},
		}, nil
	}

	srv := newTestServer(t, fp1, fp2)
	defer srv.Close()

	resp := chatRequest(t, srv, map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status: got %d, want 200; body: %s", resp.StatusCode, body)
	}
}

// TestChatStream_AllRateLimited_Returns429 verifies the same 429 behaviour for
// streaming requests.
func TestChatStream_AllRateLimited_Returns429(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	fp.StreamFunc = func(_ context.Context, _ domain.Request) (<-chan domain.Chunk, error) {
		return nil, &provider.RateLimitError{RetryAfter: 45 * time.Second}
	}
	srv := newTestServer(t, fp)
	defer srv.Close()

	b, _ := json.Marshal(map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
		"stream":   true,
	})
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("status: got %d, want 429", resp.StatusCode)
	}
	if ra := resp.Header.Get("Retry-After"); ra != "45" {
		t.Errorf("Retry-After: got %q, want %q", ra, "45")
	}
}

func TestStreamingUsage_PropagatedToClientAndMetrics(t *testing.T) {
	// Fake provider that emits a content chunk, then a usage chunk, then a done chunk.
	fp := testutil.NewFakeProvider(domain.Model{ID: "m", OwnedBy: "fake"})
	fp.NameVal = "fake"
	fp.StreamFunc = func(_ context.Context, _ domain.Request) (<-chan domain.Chunk, error) {
		ch := make(chan domain.Chunk, 3)
		ch <- domain.Chunk{ID: "c", Model: "m", Delta: "hi"}
		ch <- domain.Chunk{ID: "c", Model: "m", Usage: &domain.Usage{PromptTokens: 10, CompletionTokens: 2, CachedTokens: 5}}
		ch <- domain.Chunk{ID: "c", Model: "m", Done: true}
		close(ch)
		return ch, nil
	}

	mtr := metrics.New()
	srv := newTestServerOpts(t, []server.Option{server.WithMetrics(mtr)}, fp)
	defer srv.Close()

	resp := chatRequest(t, srv, map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
		"stream":   true,
	})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status: %d", resp.StatusCode)
	}

	body := resp.Body
	frames := readSSEFrames(t, body)

	// The usage chunk should be forwarded to the client with cached_tokens in it.
	found := false
	for _, f := range frames {
		if strings.Contains(f, `"cached_tokens":5`) {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("response should contain cached_tokens=5; frames:\n%v", frames)
	}

	// Prometheus should have the cached counter.
	var promOut strings.Builder
	mtr.WritePrometheus(&promOut)
	if !strings.Contains(promOut.String(), `gap_tokens_total{provider="fake",model="m",type="cached"} 5`) {
		t.Errorf("metrics should contain cached counter; got:\n%s", promOut.String())
	}
}

func TestStreamingFinishReason_PropagatedToClientAndAudit(t *testing.T) {
	// Fake provider emits a content chunk then a terminal chunk with finish_reason=length.
	fp := testutil.NewFakeProvider(domain.Model{ID: "m", OwnedBy: "fake", Capabilities: []string{"reasoning"}})
	fp.NameVal = "fake"
	fp.StreamFunc = func(ctx context.Context, req domain.Request) (<-chan domain.Chunk, error) {
		ch := make(chan domain.Chunk, 2)
		ch <- domain.Chunk{ID: "c", Model: "m", Delta: "partial"}
		ch <- domain.Chunk{ID: "c", Model: "m", FinishReason: "length", Done: false}
		close(ch)
		return ch, nil
	}

	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatalf("registry.Start: %v", err)
	}

	var auditBuf bytes.Buffer
	auditLogger := slog.New(slog.NewTextHandler(&auditBuf, &slog.HandlerOptions{Level: slog.LevelInfo}))
	srv := server.New(reg, server.WithAuditLog(auditLogger))

	r := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)

	if w.Code != 200 {
		t.Fatalf("status: %d, body: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), `"finish_reason":"length"`) {
		t.Errorf("client SSE should contain finish_reason=length; got:\n%s", w.Body.String())
	}
	if !strings.Contains(auditBuf.String(), `finish_reason=length`) {
		t.Errorf("audit log should contain finish_reason=length; got:\n%s", auditBuf.String())
	}
}

func TestChat_FinishReason_PropagatedToClientAndAudit(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m", OwnedBy: "fake", Capabilities: []string{"reasoning"}})
	fp.NameVal = "fake"
	fp.ChatFunc = func(ctx context.Context, req domain.Request) (domain.Response, error) {
		return domain.Response{
			ID: "r", Model: "m",
			Message:      domain.Message{Role: "assistant", Content: ""},
			FinishReason: "length",
			Usage:        domain.Usage{PromptTokens: 100, CompletionTokens: 50},
		}, nil
	}

	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatalf("registry.Start: %v", err)
	}

	var auditBuf bytes.Buffer
	auditLogger := slog.New(slog.NewTextHandler(&auditBuf, &slog.HandlerOptions{Level: slog.LevelInfo}))
	srv := server.New(reg, server.WithAuditLog(auditLogger))

	r := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)

	if w.Code != 200 {
		t.Fatalf("status: %d, body: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), `"finish_reason":"length"`) {
		t.Errorf("client response should contain finish_reason=length; got:\n%s", w.Body.String())
	}
	if !strings.Contains(auditBuf.String(), `finish_reason=length`) {
		t.Errorf("audit log should contain finish_reason=length; got:\n%s", auditBuf.String())
	}
}

func TestStreamingAudit_TTFTAndContentLogged(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m", OwnedBy: "fake", Capabilities: []string{"reasoning"}})
	fp.StreamFunc = func(ctx context.Context, req domain.Request) (<-chan domain.Chunk, error) {
		ch := make(chan domain.Chunk, 4)
		go func() {
			defer close(ch)
			// 50ms delay before the first chunk to make TTFT measurable.
			time.Sleep(50 * time.Millisecond)
			ch <- domain.Chunk{ID: "c", Model: "m", Delta: "hello "}
			ch <- domain.Chunk{ID: "c", Model: "m", Delta: "world"}
			ch <- domain.Chunk{ID: "c", Model: "m", FinishReason: "stop", Done: true}
		}()
		return ch, nil
	}

	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatalf("registry.Start: %v", err)
	}

	var auditBuf bytes.Buffer
	auditLogger := slog.New(slog.NewTextHandler(&auditBuf, &slog.HandlerOptions{Level: slog.LevelDebug}))
	srv := server.New(reg, server.WithAuditLog(auditLogger))

	r := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)

	if w.Code != 200 {
		t.Fatalf("status: %d, body: %s", w.Code, w.Body.String())
	}

	auditOut := auditBuf.String()
	// INFO: ttft_ms key present in stream_end record.
	if !strings.Contains(auditOut, "ttft_ms=") {
		t.Errorf("audit should contain ttft_ms; got:\n%s", auditOut)
	}
	// DEBUG: response content assembled and logged.
	if !strings.Contains(auditOut, "response=\"hello world\"") && !strings.Contains(auditOut, `response="hello world"`) {
		t.Errorf("audit DEBUG should contain assembled response; got:\n%s", auditOut)
	}
}

func TestStreamingAudit_NoTTFTWhenStreamEmpty(t *testing.T) {
	// If upstream errors before any chunk, ttft_ms should not appear (and we don't log stream_end at all in the error path).
	// This test verifies that an empty stream — which closes without any content chunk — does NOT emit ttft_ms.
	fp := testutil.NewFakeProvider(domain.Model{ID: "m", OwnedBy: "fake", Capabilities: []string{"reasoning"}})
	fp.StreamFunc = func(ctx context.Context, req domain.Request) (<-chan domain.Chunk, error) {
		ch := make(chan domain.Chunk)
		close(ch) // immediately, no chunks
		return ch, nil
	}

	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatalf("registry.Start: %v", err)
	}

	var auditBuf bytes.Buffer
	auditLogger := slog.New(slog.NewTextHandler(&auditBuf, &slog.HandlerOptions{Level: slog.LevelInfo}))
	srv := server.New(reg, server.WithAuditLog(auditLogger))

	r := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)

	auditOut := auditBuf.String()
	if strings.Contains(auditOut, "ttft_ms=") {
		t.Errorf("audit should NOT contain ttft_ms when no chunks arrived; got:\n%s", auditOut)
	}
}

func TestAutoSelector_CrossModelFallback_AllFail_Returns502(t *testing.T) {
	fp1 := testutil.NewFakeProvider(domain.Model{ID: "vision-a", Capabilities: []string{"vision"}})
	fp1.ChatFunc = func(_ context.Context, _ domain.Request) (domain.Response, error) {
		return domain.Response{}, errors.New("unavailable")
	}
	fp2 := testutil.NewFakeProvider(domain.Model{ID: "vision-b", Capabilities: []string{"vision"}})
	fp2.ChatFunc = func(_ context.Context, _ domain.Request) (domain.Response, error) {
		return domain.Response{}, errors.New("unavailable")
	}

	srv := newTestServer(t, fp1, fp2)
	defer srv.Close()

	resp := chatRequest(t, srv, map[string]any{
		"model":    "auto:vision",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadGateway {
		t.Errorf("status: got %d, want 502", resp.StatusCode)
	}
}

func TestChat_ClientDisconnect_LoggedAsInfoNot502(t *testing.T) {
	// Fake provider that returns context.Canceled to simulate the client-disconnect
	// path (where the request context was canceled, our outgoing http.Client.Do
	// returns canceled, and our provider wraps it as "openai: do request: ... canceled").
	fp := testutil.NewFakeProvider(domain.Model{ID: "m", OwnedBy: "fake", Capabilities: []string{"reasoning"}})
	fp.ChatFunc = func(ctx context.Context, req domain.Request) (domain.Response, error) {
		// Wait for the caller to cancel us, then return canceled (mirrors what the
		// real openai provider does when the http.Client.Do gets ctx.Err()).
		<-ctx.Done()
		return domain.Response{}, fmt.Errorf("openai: do request: %w", ctx.Err())
	}

	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatalf("registry.Start: %v", err)
	}

	var auditBuf bytes.Buffer
	auditLogger := slog.New(slog.NewTextHandler(&auditBuf, &slog.HandlerOptions{Level: slog.LevelInfo}))
	srv := server.New(reg, server.WithAuditLog(auditLogger))

	// Build the request with a context we can cancel from the test.
	reqCtx, cancel := context.WithCancel(context.Background())
	r := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	r = r.WithContext(reqCtx)
	w := httptest.NewRecorder()

	// Run server in a goroutine so we can cancel from the test.
	done := make(chan struct{})
	go func() {
		srv.ServeHTTP(w, r)
		close(done)
	}()
	// Give the request a moment to dispatch to the fake provider, then cancel.
	time.Sleep(20 * time.Millisecond)
	cancel()
	<-done

	if w.Code != 499 {
		t.Errorf("status: got %d, want 499 (Client Closed Request)", w.Code)
	}
	auditOut := auditBuf.String()
	if !strings.Contains(auditOut, "event=client_disconnect") {
		t.Errorf("audit should contain event=client_disconnect; got:\n%s", auditOut)
	}
	if strings.Contains(auditOut, "level=ERROR") {
		t.Errorf("audit should NOT have ERROR level for client disconnect; got:\n%s", auditOut)
	}
}

func TestStream_ClientDisconnect_BeforeHeadersLoggedAsInfo(t *testing.T) {
	// Same scenario but for the streaming path, where the provider returns an
	// error before any chunks (so headers haven't been flushed yet).
	fp := testutil.NewFakeProvider(domain.Model{ID: "m", OwnedBy: "fake", Capabilities: []string{"reasoning"}})
	fp.StreamFunc = func(ctx context.Context, req domain.Request) (<-chan domain.Chunk, error) {
		<-ctx.Done()
		return nil, fmt.Errorf("openai: do request: %w", ctx.Err())
	}

	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatalf("registry.Start: %v", err)
	}

	var auditBuf bytes.Buffer
	auditLogger := slog.New(slog.NewTextHandler(&auditBuf, &slog.HandlerOptions{Level: slog.LevelInfo}))
	srv := server.New(reg, server.WithAuditLog(auditLogger))

	reqCtx, cancel := context.WithCancel(context.Background())
	r := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	r = r.WithContext(reqCtx)
	w := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		srv.ServeHTTP(w, r)
		close(done)
	}()
	time.Sleep(20 * time.Millisecond)
	cancel()
	<-done

	if w.Code != 499 {
		t.Errorf("status: got %d, want 499", w.Code)
	}
	auditOut := auditBuf.String()
	if !strings.Contains(auditOut, "event=client_disconnect") {
		t.Errorf("audit should contain event=client_disconnect; got:\n%s", auditOut)
	}
}

func TestChatCompletions_ResponseFormatReachesProviderAndWarns(t *testing.T) {
	// Capture the default slog output.
	var logBuf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&logBuf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() { slog.SetDefault(prev) })

	var gotReq domain.Request
	fp := testutil.NewFakeProvider(domain.Model{ID: "plain-model"}) // no structured_output capability
	fp.ChatFunc = func(_ context.Context, req domain.Request) (domain.Response, error) {
		gotReq = req
		return domain.Response{ID: "r", Model: req.Model, Message: domain.Message{Role: "assistant", Content: "{}"}}, nil
	}
	srv := newTestServer(t, fp)

	resp := chatRequest(t, srv, map[string]any{
		"model":    "plain-model",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
		"response_format": map[string]any{
			"type":        "json_schema",
			"json_schema": map[string]any{"name": "x", "schema": map[string]any{"type": "object"}},
		},
	})
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status: got %d, want 200", resp.StatusCode)
	}
	if gotReq.ResponseFormat == nil {
		t.Fatal("provider did not receive ResponseFormat")
	}
	if !strings.Contains(logBuf.String(), "structured_output") {
		t.Errorf("expected a structured_output warning, got logs: %s", logBuf.String())
	}
}

func TestChatCompletions_NoResponseFormatNoWarn(t *testing.T) {
	var logBuf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&logBuf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() { slog.SetDefault(prev) })

	fp := testutil.NewFakeProvider(domain.Model{ID: "plain-model"})
	srv := newTestServer(t, fp)

	resp := chatRequest(t, srv, map[string]any{
		"model":    "plain-model",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	defer resp.Body.Close()
	if strings.Contains(logBuf.String(), "structured_output") {
		t.Errorf("unexpected structured_output warning: %s", logBuf.String())
	}
}
