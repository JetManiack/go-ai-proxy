package litellm_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/JetManiack/go-ai-proxy/internal/provider/litellm"
)

func newFakeLiteLLMServer(t *testing.T, modelsResp, modelInfoResp any) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/v1/models":
			json.NewEncoder(w).Encode(modelsResp)
		case "/model/info":
			if modelInfoResp == nil {
				http.NotFound(w, r)
				return
			}
			json.NewEncoder(w).Encode(modelInfoResp)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestModels_EnrichesVisionCapability(t *testing.T) {
	srv := newFakeLiteLLMServer(t,
		map[string]any{
			"object": "list",
			"data": []map[string]any{
				{"id": "gpt-4-vision", "object": "model", "owned_by": "openai"},
			},
		},
		map[string]any{
			"data": []map[string]any{
				{
					"model_name": "gpt-4-vision",
					"model_info": map[string]any{
						"supports_vision":            true,
						"supports_function_calling":  true,
					},
				},
			},
		},
	)

	p := litellm.New(srv.URL)
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}
	if len(models) != 1 {
		t.Fatalf("models len: got %d, want 1", len(models))
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

func TestModels_FallsBackGracefullyWhenNoModelInfo(t *testing.T) {
	srv := newFakeLiteLLMServer(t,
		map[string]any{
			"object": "list",
			"data": []map[string]any{
				{"id": "text-model", "object": "model", "owned_by": "local"},
			},
		},
		nil, // /model/info returns 404
	)

	p := litellm.New(srv.URL)
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}
	if len(models) != 1 || models[0].ID != "text-model" {
		t.Errorf("models: got %+v", models)
	}
	if len(models[0].Capabilities) != 0 {
		t.Errorf("expected no capabilities, got %v", models[0].Capabilities)
	}
}

func TestModels_ParsesPricing(t *testing.T) {
	srv := newFakeLiteLLMServer(t,
		map[string]any{
			"object": "list",
			"data": []map[string]any{
				{"id": "gpt-4o", "object": "model", "owned_by": "openai"},
			},
		},
		map[string]any{
			"data": []map[string]any{
				{
					"model_name": "gpt-4o",
					"model_info": map[string]any{
						"input_cost_per_token":  0.000005,
						"output_cost_per_token": 0.000015,
					},
				},
			},
		},
	)

	p := litellm.New(srv.URL)
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}
	if models[0].InputCostPerToken == nil || *models[0].InputCostPerToken != 0.000005 {
		t.Errorf("InputCostPerToken: got %v, want 0.000005", models[0].InputCostPerToken)
	}
	if models[0].OutputCostPerToken == nil || *models[0].OutputCostPerToken != 0.000015 {
		t.Errorf("OutputCostPerToken: got %v, want 0.000015", models[0].OutputCostPerToken)
	}
}

func hasCap(caps []string, want string) bool {
	for _, c := range caps {
		if c == want {
			return true
		}
	}
	return false
}

func TestModels_MapsAllCapabilities(t *testing.T) {
	srv := newFakeLiteLLMServer(t,
		map[string]any{
			"object": "list",
			"data":   []map[string]any{{"id": "full-model", "object": "model", "owned_by": "test"}},
		},
		map[string]any{
			"data": []map[string]any{
				{
					"model_name": "full-model",
					"model_info": map[string]any{
						"supports_vision":           true,
						"supports_reasoning":        true,
						"supports_function_calling": true,
						"supports_pdf_input":        true,
						"supports_response_schema":  true,
						"supports_web_search":       true,
						"supports_audio_input":      true,
						"supports_audio_output":     true,
						"supports_computer_use":     true,
						"supports_prompt_caching":   true,
						"supports_url_context":      true,
					},
				},
			},
		},
	)

	p := litellm.New(srv.URL)
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}
	caps := models[0].Capabilities
	for _, want := range []string{"vision", "reasoning", "tools", "pdf", "structured_output", "web_search", "audio_input", "audio_output", "computer_use", "prompt_caching", "url_context"} {
		if !hasCap(caps, want) {
			t.Errorf("missing capability %q; got %v", want, caps)
		}
	}
}

func TestModels_MapsToolsFromToolChoice(t *testing.T) {
	srv := newFakeLiteLLMServer(t,
		map[string]any{"object": "list", "data": []map[string]any{{"id": "m", "object": "model", "owned_by": "test"}}},
		map[string]any{"data": []map[string]any{{"model_name": "m", "model_info": map[string]any{"supports_tool_choice": true}}}},
	)
	p := litellm.New(srv.URL)
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}
	if !hasCap(models[0].Capabilities, "tools") {
		t.Errorf("expected tools from supports_tool_choice, got %v", models[0].Capabilities)
	}
}

func TestModels_MapsReasoningCapability(t *testing.T) {
	srv := newFakeLiteLLMServer(t,
		map[string]any{
			"object": "list",
			"data": []map[string]any{
				{"id": "deepseek-r1", "object": "model", "owned_by": "deepseek"},
			},
		},
		map[string]any{
			"data": []map[string]any{
				{
					"model_name": "deepseek-r1",
					"model_info": map[string]any{
						"supports_reasoning": true,
					},
				},
			},
		},
	)

	p := litellm.New(srv.URL)
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
