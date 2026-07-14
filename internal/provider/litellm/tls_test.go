package litellm_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
	"github.com/JetManiack/go-ai-proxy/internal/provider/litellm"
)

func newTLSServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	srv := httptest.NewTLSServer(handler)
	t.Cleanup(srv.Close)
	return srv
}

// Without WithTLSInsecure, Models() must reject the self-signed cert.
func TestLiteLLM_TLSVerify_RejectsSelfSigned(t *testing.T) {
	srv := newTLSServer(t, func(w http.ResponseWriter, r *http.Request) {
		t.Fatalf("handler should not be reached when TLS verification is on")
	})

	p := litellm.New(srv.URL)
	_, err := p.Models(context.Background())
	if err == nil {
		t.Fatalf("expected TLS verification error, got nil")
	}
	if !strings.Contains(err.Error(), "x509") && !strings.Contains(err.Error(), "certificate") && !strings.Contains(err.Error(), "tls") {
		t.Fatalf("expected TLS-related error, got: %v", err)
	}
}

// /v1/models and /model/info both must work with WithTLSInsecure(true).
func TestLiteLLM_TLSInsecure_Models(t *testing.T) {
	srv := newTLSServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/v1/models":
			json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{{"id": "m-1", "owned_by": "litellm"}},
			})
		case "/model/info":
			json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{{
					"model_name": "m-1",
					"model_info": map[string]any{"supports_vision": true},
				}},
			})
		default:
			http.NotFound(w, r)
		}
	})

	p := litellm.New(srv.URL, litellm.WithTLSInsecure(true))
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}
	if len(models) != 1 || models[0].ID != "m-1" {
		t.Fatalf("unexpected models: %+v", models)
	}
	if len(models[0].Capabilities) != 1 || models[0].Capabilities[0] != "vision" {
		t.Fatalf("expected vision capability via /model/info, got %+v", models[0].Capabilities)
	}
}

// Chat is forwarded to the inner openai provider, which must also use insecure TLS.
func TestLiteLLM_TLSInsecure_Chat(t *testing.T) {
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

	p := litellm.New(srv.URL, litellm.WithTLSInsecure(true))
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
