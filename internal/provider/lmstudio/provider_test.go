package lmstudio_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/JetManiack/go-ai-proxy/internal/provider/lmstudio"
)

func newFakeServer(t *testing.T, nativeResp, openaiResp any) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/api/v1/models":
			json.NewEncoder(w).Encode(nativeResp)
		case "/v1/models":
			if openaiResp == nil {
				http.NotFound(w, r)
				return
			}
			json.NewEncoder(w).Encode(openaiResp)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func nativeModel(key string, vision bool, hasReasoning bool) map[string]any {
	caps := map[string]any{
		"vision":              vision,
		"trained_for_tool_use": true,
	}
	if hasReasoning {
		caps["reasoning"] = map[string]any{
			"allowed_options": []string{"on"},
			"default":         "on",
		}
	}
	return map[string]any{
		"type":         "llm",
		"key":          key,
		"display_name": key,
		"capabilities": caps,
	}
}

func TestModels_MapsVisionCapability(t *testing.T) {
	srv := newFakeServer(t,
		map[string]any{"models": []any{nativeModel("vision-model", true, false)}},
		nil,
	)
	p := lmstudio.New(srv.URL)
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}
	if len(models) != 1 {
		t.Fatalf("len: got %d, want 1", len(models))
	}
	found := false
	for _, c := range models[0].Capabilities {
		if c == "vision" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected vision capability, got %v", models[0].Capabilities)
	}
}

func TestModels_MapsReasoningCapability(t *testing.T) {
	srv := newFakeServer(t,
		map[string]any{"models": []any{nativeModel("reasoning-model", false, true)}},
		nil,
	)
	p := lmstudio.New(srv.URL)
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}
	found := false
	for _, c := range models[0].Capabilities {
		if c == "reasoning" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected reasoning capability, got %v", models[0].Capabilities)
	}
}

func TestModels_FiltersEmbeddingModels(t *testing.T) {
	srv := newFakeServer(t,
		map[string]any{"models": []any{
			nativeModel("llm-model", false, false),
			map[string]any{"type": "embedding", "key": "embed-model", "display_name": "Embed"},
		}},
		nil,
	)
	p := lmstudio.New(srv.URL)
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}
	if len(models) != 1 || models[0].ID != "llm-model" {
		t.Errorf("expected only llm-model, got %v", models)
	}
}

func TestModels_MapsToolsCapability(t *testing.T) {
	srv := newFakeServer(t,
		map[string]any{"models": []any{nativeModel("tool-model", false, false)}},
		nil,
	)
	p := lmstudio.New(srv.URL)
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}
	found := false
	for _, c := range models[0].Capabilities {
		if c == "tools" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected tools capability, got %v", models[0].Capabilities)
	}
}

func TestModels_FallsBackToOpenAIWhenNativeUnavailable(t *testing.T) {
	srv2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/api/v1/models" {
			http.NotFound(w, r)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"data":   []map[string]any{{"id": "fallback-model", "object": "model", "owned_by": "local"}},
		})
	}))
	defer srv2.Close()

	p := lmstudio.New(srv2.URL)
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}
	if len(models) != 1 || models[0].ID != "fallback-model" {
		t.Errorf("expected fallback-model, got %v", models)
	}
}
