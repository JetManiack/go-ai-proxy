package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
	"github.com/JetManiack/go-ai-proxy/internal/provider"
	"github.com/JetManiack/go-ai-proxy/internal/provider/openai"
)

func TestEmbeddings_BasicResponse(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"model":  "text-embedding-3-small",
			"data": []map[string]any{
				{"object": "embedding", "index": 0, "embedding": []float64{0.1, 0.2}},
			},
			"usage": map[string]any{"prompt_tokens": 4, "total_tokens": 4},
		})
	})

	p := openai.New(srv.URL)
	resp, err := p.Embeddings(context.Background(), domain.EmbedRequest{
		Model: "text-embedding-3-small",
		Input: []string{"hello"},
	})
	if err != nil {
		t.Fatalf("Embeddings error: %v", err)
	}
	if len(resp.Embeddings) != 1 || resp.Embeddings[0].Values[1] != 0.2 {
		t.Errorf("embeddings: got %+v", resp.Embeddings)
	}
	if resp.Usage.PromptTokens != 4 {
		t.Errorf("prompt_tokens: got %d, want 4", resp.Usage.PromptTokens)
	}
}

func TestEmbeddings_SendsAPIKeyAndForcesFloatUpstream(t *testing.T) {
	var gotAuth string
	var gotBody map[string]any
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"object": "list", "model": "m",
			"data":  []map[string]any{{"object": "embedding", "index": 0, "embedding": []float64{1}}},
			"usage": map[string]any{},
		})
	})

	p := openai.New(srv.URL, openai.WithAPIKey("sk-test-key"))
	_, err := p.Embeddings(context.Background(), domain.EmbedRequest{
		Model:          "m",
		Input:          []string{"a", "b"},
		EncodingFormat: "base64", // caller wants base64; upstream call must still force float
	})
	if err != nil {
		t.Fatalf("Embeddings error: %v", err)
	}
	if gotAuth != "Bearer sk-test-key" {
		t.Errorf("Authorization: got %q, want %q", gotAuth, "Bearer sk-test-key")
	}
	if gotBody["encoding_format"] != "float" {
		t.Errorf("encoding_format sent upstream: got %v, want %q", gotBody["encoding_format"], "float")
	}
	input, ok := gotBody["input"].([]any)
	if !ok || len(input) != 2 {
		t.Errorf("input sent upstream: got %#v, want 2-element array", gotBody["input"])
	}
}

func TestEmbeddings_UpstreamError(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":{"message":"boom"}}`))
	})

	p := openai.New(srv.URL)
	_, err := p.Embeddings(context.Background(), domain.EmbedRequest{Model: "m", Input: []string{"hi"}})
	var ue *provider.UpstreamError
	if !errors.As(err, &ue) {
		t.Fatalf("expected *provider.UpstreamError, got %T: %v", err, err)
	}
	if ue.StatusCode != http.StatusInternalServerError {
		t.Errorf("StatusCode: got %d, want 500", ue.StatusCode)
	}
	if !strings.Contains(ue.Body, "boom") {
		t.Errorf("Body should contain upstream message, got %q", ue.Body)
	}
}

// TestEmbeddings_UpstreamClientError_PreservesStatusAndBody verifies that a
// 4xx upstream response (e.g. an embeddings backend rejecting an input that
// exceeds its token limit) is captured verbatim, so the server can proxy it
// to the client instead of collapsing it into a generic 502.
func TestEmbeddings_UpstreamClientError_PreservesStatusAndBody(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error":{"message":"input exceeds model's maximum context length of 2048 tokens"}}`))
	})

	p := openai.New(srv.URL)
	_, err := p.Embeddings(context.Background(), domain.EmbedRequest{Model: "m", Input: []string{"hi"}})
	var ue *provider.UpstreamError
	if !errors.As(err, &ue) {
		t.Fatalf("expected *provider.UpstreamError, got %T: %v", err, err)
	}
	if ue.StatusCode != http.StatusBadRequest {
		t.Errorf("StatusCode: got %d, want 400", ue.StatusCode)
	}
	if !strings.Contains(ue.Body, "maximum context length") {
		t.Errorf("Body should preserve upstream message verbatim, got %q", ue.Body)
	}
}

func TestEmbeddings_429_RateLimitError(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "30")
		w.WriteHeader(http.StatusTooManyRequests)
	})

	p := openai.New(srv.URL)
	_, err := p.Embeddings(context.Background(), domain.EmbedRequest{Model: "m", Input: []string{"hi"}})
	var rl *provider.RateLimitError
	if !errors.As(err, &rl) {
		t.Fatalf("expected *provider.RateLimitError, got %T: %v", err, err)
	}
	if rl.RetryAfter != 30*time.Second {
		t.Errorf("RetryAfter: got %v, want 30s", rl.RetryAfter)
	}
}
