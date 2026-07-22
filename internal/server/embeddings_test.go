package server_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
	"github.com/JetManiack/go-ai-proxy/internal/provider"
	"github.com/JetManiack/go-ai-proxy/internal/testutil"
)

// chatOnlyProvider implements domain.Provider but deliberately does NOT
// implement domain.EmbeddingsProvider, modelling a provider like Anthropic
// that has no native embeddings support.
type chatOnlyProvider struct {
	model string
}

func (p *chatOnlyProvider) Name() string { return "chat-only" }
func (p *chatOnlyProvider) Chat(ctx context.Context, req domain.Request) (domain.Response, error) {
	return domain.Response{Message: domain.Message{Role: "assistant", Content: "ok"}}, nil
}
func (p *chatOnlyProvider) ChatStream(ctx context.Context, req domain.Request) (<-chan domain.Chunk, error) {
	ch := make(chan domain.Chunk, 1)
	ch <- domain.Chunk{Done: true}
	close(ch)
	return ch, nil
}
func (p *chatOnlyProvider) Models(ctx context.Context) ([]domain.Model, error) {
	return []domain.Model{{ID: p.model}}, nil
}

func embedRequest(t *testing.T, srv *httptest.Server, body any) *http.Response {
	t.Helper()
	b, _ := json.Marshal(body)
	resp, err := http.Post(srv.URL+"/v1/embeddings", "application/json", bytes.NewReader(b))
	if err != nil {
		t.Fatalf("POST /v1/embeddings: %v", err)
	}
	return resp
}

func TestEmbeddings_OK(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "text-embedding-3-small"})
	fp.EmbedFunc = func(_ context.Context, req domain.EmbedRequest) (domain.EmbedResponse, error) {
		return domain.EmbedResponse{
			Model:      req.Model,
			Embeddings: []domain.Embedding{{Index: 0, Values: []float64{0.1, 0.2}}},
			Usage:      domain.Usage{PromptTokens: 3},
		}, nil
	}

	srv := newTestServer(t, fp)
	defer srv.Close()

	resp := embedRequest(t, srv, map[string]any{
		"model": "text-embedding-3-small",
		"input": "hello",
	})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status: got %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type: got %q, want application/json", ct)
	}

	var body map[string]any
	json.NewDecoder(resp.Body).Decode(&body)
	data := body["data"].([]any)
	if len(data) != 1 {
		t.Fatalf("data len: got %d, want 1", len(data))
	}
	embedding := data[0].(map[string]any)["embedding"].([]any)
	if len(embedding) != 2 || embedding[1] != 0.2 {
		t.Errorf("embedding: got %v", embedding)
	}
}

func TestEmbeddings_UnknownModel(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "known-model"})
	srv := newTestServer(t, fp)
	defer srv.Close()

	resp := embedRequest(t, srv, map[string]any{"model": "unknown-model", "input": "hi"})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status: got %d, want 400", resp.StatusCode)
	}
}

func TestEmbeddings_WrongMethod(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	srv := newTestServer(t, fp)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/v1/embeddings")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Errorf("status: got %d, want 405", resp.StatusCode)
	}
}

func TestEmbeddings_InvalidJSON(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	srv := newTestServer(t, fp)
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/v1/embeddings", "application/json", bytes.NewBufferString("not json"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status: got %d, want 400", resp.StatusCode)
	}
}

// TestEmbeddings_ProviderDoesNotSupport_FallsBackToNext verifies that a
// candidate without EmbeddingsProvider support (e.g. Anthropic) is skipped
// in favour of a candidate that does support embeddings.
func TestEmbeddings_ProviderDoesNotSupport_FallsBackToNext(t *testing.T) {
	unsupported := &chatOnlyProvider{model: "m"}
	supported := testutil.NewFakeProvider(domain.Model{ID: "m"})
	supported.EmbedFunc = func(_ context.Context, req domain.EmbedRequest) (domain.EmbedResponse, error) {
		return domain.EmbedResponse{
			Model:      req.Model,
			Embeddings: []domain.Embedding{{Index: 0, Values: []float64{9}}},
		}, nil
	}

	srv := newTestServer(t, unsupported, supported)
	defer srv.Close()

	resp := embedRequest(t, srv, map[string]any{"model": "m", "input": "hi"})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status: got %d, want 200", resp.StatusCode)
	}
	var body map[string]any
	json.NewDecoder(resp.Body).Decode(&body)
	data := body["data"].([]any)
	embedding := data[0].(map[string]any)["embedding"].([]any)
	if len(embedding) != 1 || embedding[0] != float64(9) {
		t.Errorf("embedding: got %v, want fallback provider's [9]", embedding)
	}
}

// TestEmbeddings_AllUnsupported_Returns502 verifies that when every
// candidate lacks embeddings support, the request fails with 502 rather
// than a panic or a silent empty response.
func TestEmbeddings_AllUnsupported_Returns502(t *testing.T) {
	p1 := &chatOnlyProvider{model: "m"}
	srv := newTestServer(t, p1)
	defer srv.Close()

	resp := embedRequest(t, srv, map[string]any{"model": "m", "input": "hi"})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadGateway {
		t.Errorf("status: got %d, want 502", resp.StatusCode)
	}
}

func TestEmbeddings_AllRateLimited_Returns429(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	fp.EmbedFunc = func(_ context.Context, _ domain.EmbedRequest) (domain.EmbedResponse, error) {
		return domain.EmbedResponse{}, &provider.RateLimitError{RetryAfter: 30 * time.Second}
	}
	srv := newTestServer(t, fp)
	defer srv.Close()

	resp := embedRequest(t, srv, map[string]any{"model": "m", "input": "hi"})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("status: got %d, want 429", resp.StatusCode)
	}
	if ra := resp.Header.Get("Retry-After"); ra != "30" {
		t.Errorf("Retry-After: got %q, want %q", ra, "30")
	}
}

// TestEmbeddings_Base64EncodingFormat verifies the client-requested
// encoding_format threads through to the response shape (a base64 string
// per embedding rather than a float array).
func TestEmbeddings_Base64EncodingFormat(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	fp.EmbedFunc = func(_ context.Context, req domain.EmbedRequest) (domain.EmbedResponse, error) {
		return domain.EmbedResponse{
			Model:      req.Model,
			Embeddings: []domain.Embedding{{Index: 0, Values: []float64{1, 2, 3}}},
		}, nil
	}
	srv := newTestServer(t, fp)
	defer srv.Close()

	resp := embedRequest(t, srv, map[string]any{
		"model": "m", "input": "hi", "encoding_format": "base64",
	})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status: got %d, want 200", resp.StatusCode)
	}
	var body map[string]any
	json.NewDecoder(resp.Body).Decode(&body)
	data := body["data"].([]any)
	if _, ok := data[0].(map[string]any)["embedding"].(string); !ok {
		t.Errorf("embedding: got %T %v, want base64 string", data[0].(map[string]any)["embedding"], data[0])
	}
}
