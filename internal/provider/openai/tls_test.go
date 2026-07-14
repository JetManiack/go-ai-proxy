package openai_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
	"github.com/JetManiack/go-ai-proxy/internal/provider/openai"
)

// newFakeTLSServer returns an httptest.Server with a self-signed certificate.
func newFakeTLSServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	srv := httptest.NewTLSServer(handler)
	t.Cleanup(srv.Close)
	return srv
}

// Without WithTLSInsecure, the client must reject the self-signed cert.
func TestChat_TLSVerify_RejectsSelfSigned(t *testing.T) {
	srv := newFakeTLSServer(t, func(w http.ResponseWriter, r *http.Request) {
		t.Fatalf("handler should not be reached when TLS verification is on")
	})

	p := openai.New(srv.URL)
	_, err := p.Chat(context.Background(), domain.Request{
		Model:    "m",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatalf("expected TLS verification error, got nil")
	}
	if !strings.Contains(err.Error(), "x509") && !strings.Contains(err.Error(), "certificate") && !strings.Contains(err.Error(), "tls") {
		t.Fatalf("expected TLS-related error, got: %v", err)
	}
}

// With WithTLSInsecure(true), the same server must respond successfully.
func TestChat_TLSInsecure_AcceptsSelfSigned(t *testing.T) {
	srv := newFakeTLSServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":    "resp-1",
			"model": "m",
			"choices": []map[string]any{
				{"index": 0, "message": map[string]any{"role": "assistant", "content": "ok"}, "finish_reason": "stop"},
			},
			"usage": map[string]any{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
		})
	})

	p := openai.New(srv.URL, openai.WithTLSInsecure(true))
	resp, err := p.Chat(context.Background(), domain.Request{
		Model:    "m",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Chat with WithTLSInsecure: %v", err)
	}
	if resp.Message.Content != "ok" {
		t.Fatalf("content: got %q, want %q", resp.Message.Content, "ok")
	}
}

// Models() must use the same TLS configuration as Chat().
func TestModels_TLSInsecure_AcceptsSelfSigned(t *testing.T) {
	srv := newFakeTLSServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"id": "m-1", "owned_by": "test"}},
		})
	})

	p := openai.New(srv.URL, openai.WithTLSInsecure(true))
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}
	if len(models) != 1 || models[0].ID != "m-1" {
		t.Fatalf("unexpected models: %+v", models)
	}
}
