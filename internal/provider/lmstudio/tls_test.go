package lmstudio_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
	"github.com/JetManiack/go-ai-proxy/internal/provider/lmstudio"
)

func newTLSServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	srv := httptest.NewTLSServer(handler)
	t.Cleanup(srv.Close)
	return srv
}

// Without WithTLSInsecure, Models() should fail against a self-signed cert.
func TestLMStudio_TLSVerify_RejectsSelfSigned(t *testing.T) {
	srv := newTLSServer(t, func(w http.ResponseWriter, r *http.Request) {
		t.Fatalf("handler should not be reached when TLS verification is on")
	})

	p := lmstudio.New(srv.URL)
	_, err := p.Models(context.Background())
	if err == nil {
		t.Fatalf("expected TLS verification error, got nil")
	}
	if !strings.Contains(err.Error(), "x509") && !strings.Contains(err.Error(), "certificate") && !strings.Contains(err.Error(), "tls") {
		t.Fatalf("expected TLS-related error, got: %v", err)
	}
}

// Native /api/v1/models endpoint must work with WithTLSInsecure(true).
func TestLMStudio_TLSInsecure_NativeModels(t *testing.T) {
	srv := newTLSServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/models" {
			t.Fatalf("expected /api/v1/models, got %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"models": []map[string]any{
				{"type": "llm", "key": "m-1", "capabilities": map[string]any{"vision": true}},
			},
		})
	})

	p := lmstudio.New(srv.URL, lmstudio.WithTLSInsecure(true))
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}
	if len(models) != 1 || models[0].ID != "m-1" {
		t.Fatalf("unexpected models: %+v", models)
	}
}

// Chat is forwarded to the inner openai provider, which must also use insecure TLS.
func TestLMStudio_TLSInsecure_Chat(t *testing.T) {
	srv := newTLSServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("expected /v1/chat/completions, got %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id": "r", "model": "m",
			"choices": []map[string]any{
				{"index": 0, "message": map[string]any{"role": "assistant", "content": "ok"}, "finish_reason": "stop"},
			},
			"usage": map[string]any{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
		})
	})

	p := lmstudio.New(srv.URL, lmstudio.WithTLSInsecure(true))
	resp, err := p.Chat(context.Background(), domain.Request{
		Model:    "m",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.Message.Content != "ok" {
		t.Fatalf("content: got %q", resp.Message.Content)
	}
}
