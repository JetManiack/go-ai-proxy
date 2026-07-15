package llamacpp_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
	llamacppprovider "github.com/JetManiack/go-ai-proxy/internal/provider/llamacpp"
)

func fakeChatServer(t *testing.T, capture *map[string]any) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": []any{map[string]any{"id": "gemma", "object": "model"}}})
			return
		}
		json.NewDecoder(r.Body).Decode(capture)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id": "x", "object": "chat.completion", "model": "gemma",
			"choices": []any{map[string]any{"index": 0, "message": map[string]any{"role": "assistant", "content": "{}"}, "finish_reason": "stop"}},
			"usage": map[string]any{"prompt_tokens": 1, "completion_tokens": 1},
		})
	}))
	t.Cleanup(srv.Close)
	return srv
}

func newProvider(t *testing.T, url string) *llamacppprovider.Provider {
	t.Helper()
	p, err := llamacppprovider.New(context.Background(), llamacppprovider.Config{
		Name: "llamacpp", BaseURL: url,
		ModelCapabilities: map[string][]string{"gemma": {"structured_output"}},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

func TestChat_ResponseFormatConvertedToGrammar(t *testing.T) {
	var body map[string]any
	srv := fakeChatServer(t, &body)
	p := newProvider(t, srv.URL)

	_, err := p.Chat(context.Background(), domain.Request{
		Model:    "gemma",
		Messages: []domain.Message{{Role: "user", Content: "classify"}},
		ResponseFormat: &domain.ResponseFormat{
			Schema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"category": map[string]any{"type": "string", "enum": []any{"a", "b"}}},
			},
		},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	g, ok := body["grammar"].(string)
	if !ok || g == "" {
		t.Fatalf("expected non-empty grammar in body, got: %v", body["grammar"])
	}
	if _, present := body["response_format"]; present {
		t.Errorf("response_format must be stripped, got: %v", body["response_format"])
	}
}

func TestChat_ExplicitGrammarPassedThrough(t *testing.T) {
	var body map[string]any
	srv := fakeChatServer(t, &body)
	p := newProvider(t, srv.URL)

	_, err := p.Chat(context.Background(), domain.Request{
		Model:    "gemma",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
		Grammar:  `root ::= "yes" | "no"`,
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if body["grammar"] != `root ::= "yes" | "no"` {
		t.Errorf("grammar: got %v, want the verbatim client grammar", body["grammar"])
	}
}

func TestChat_NoStructuredOutputBodyUnchanged(t *testing.T) {
	var body map[string]any
	srv := fakeChatServer(t, &body)
	p := newProvider(t, srv.URL)

	_, err := p.Chat(context.Background(), domain.Request{
		Model:    "gemma",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if _, present := body["grammar"]; present {
		t.Errorf("no grammar expected, got: %v", body["grammar"])
	}
}

func TestChat_ExplicitGrammarWinsAndStripsResponseFormat(t *testing.T) {
	var body map[string]any
	srv := fakeChatServer(t, &body)
	p := newProvider(t, srv.URL)

	_, err := p.Chat(context.Background(), domain.Request{
		Model:    "gemma",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
		Grammar:  `root ::= "yes" | "no"`,
		ResponseFormat: &domain.ResponseFormat{
			Schema: map[string]any{"type": "object", "properties": map[string]any{"x": map[string]any{"type": "string"}}},
		},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if body["grammar"] != `root ::= "yes" | "no"` {
		t.Errorf("explicit grammar must win, got: %v", body["grammar"])
	}
	if _, present := body["response_format"]; present {
		t.Errorf("response_format must be stripped, got: %v", body["response_format"])
	}
}

func TestChat_UnsupportedSchemaErrors(t *testing.T) {
	var body map[string]any
	srv := fakeChatServer(t, &body)
	p := newProvider(t, srv.URL)

	_, err := p.Chat(context.Background(), domain.Request{
		Model:    "gemma",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
		ResponseFormat: &domain.ResponseFormat{
			Schema: map[string]any{"type": "string", "pattern": "^x$"},
		},
	})
	if err == nil {
		t.Fatal("expected error for unsupported schema, got nil")
	}
}
