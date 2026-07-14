package vllm_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
	"github.com/JetManiack/go-ai-proxy/internal/provider"
	"github.com/JetManiack/go-ai-proxy/internal/provider/vllm"
)

// newUpstream builds an httptest server with the given handler, ready to be
// passed as a vLLM base URL.
func newUpstream(t *testing.T, h http.HandlerFunc) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return srv
}

// modelsResponse returns a JSON body matching vLLM's /v1/models format.
func modelsResponse(ids ...string) string {
	data := []map[string]any{}
	for _, id := range ids {
		data = append(data, map[string]any{"id": id, "object": "model", "owned_by": "vllm"})
	}
	b, _ := json.Marshal(map[string]any{"object": "list", "data": data})
	return string(b)
}

func chatStopResponse(content, reasoning string) string {
	msg := map[string]any{"role": "assistant", "content": content}
	if reasoning != "" {
		msg["reasoning"] = reasoning
	}
	body := map[string]any{
		"id":      "r",
		"model":   "gemma-4-31b-it",
		"choices": []map[string]any{{"index": 0, "message": msg, "finish_reason": "stop"}},
		"usage":   map[string]any{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
	}
	b, _ := json.Marshal(body)
	return string(b)
}

func TestNew_FailsWhenDiscoverOffAndModelsEmpty(t *testing.T) {
	_, err := vllm.New(context.Background(), vllm.Config{
		Name:     "vt",
		BaseURL:  "http://example.invalid",
		Discover: false,
		Models:   nil,
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "no models") {
		t.Errorf("error should mention no models; got: %v", err)
	}
}

func TestNew_FailsWhenCapabilitiesMissing(t *testing.T) {
	srv := newUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, modelsResponse("gemma-4-31b-it"))
			return
		}
		http.NotFound(w, r)
	})

	_, err := vllm.New(context.Background(), vllm.Config{
		Name:               "vt",
		BaseURL:            srv.URL,
		Discover:           true,
		ModelCapabilities:  nil, // intentionally empty
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "gemma-4-31b-it") {
		t.Errorf("error should name the missing model; got: %v", err)
	}
	if !strings.Contains(err.Error(), "model_capabilities") {
		t.Errorf("error should mention model_capabilities; got: %v", err)
	}
}

func TestNew_DiscoverSuccessWithCaps(t *testing.T) {
	srv := newUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, modelsResponse("gemma-4-31b-it"))
			return
		}
		http.NotFound(w, r)
	})

	p, err := vllm.New(context.Background(), vllm.Config{
		Name:              "vt",
		BaseURL:           srv.URL,
		Discover:          true,
		ModelCapabilities: map[string][]string{"gemma-4-31b-it": {"vision", "reasoning"}},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if p.Name() != "vt" {
		t.Errorf("Name: got %q, want %q", p.Name(), "vt")
	}

	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}
	if len(models) != 1 || models[0].ID != "gemma-4-31b-it" {
		t.Fatalf("Models: got %+v", models)
	}
}

func TestNew_DiscoverFailsFallsBackToConfig(t *testing.T) {
	srv := newUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			http.Error(w, "boom", http.StatusInternalServerError)
			return
		}
		http.NotFound(w, r)
	})

	p, err := vllm.New(context.Background(), vllm.Config{
		Name:              "vt",
		BaseURL:           srv.URL,
		Discover:          true,
		Models:            []string{"gemma-4-31b-it"},
		ModelCapabilities: map[string][]string{"gemma-4-31b-it": {"reasoning"}},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}
	if len(models) != 1 || models[0].ID != "gemma-4-31b-it" {
		t.Fatalf("expected fallback models, got %+v", models)
	}
}

func TestChat_InjectsEnableThinking(t *testing.T) {
	var captured map[string]any
	srv := newUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, modelsResponse("gemma-4-31b-it"))
		case "/v1/chat/completions":
			b, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(b, &captured)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, chatStopResponse("answer", "let me think"))
		default:
			http.NotFound(w, r)
		}
	})

	p, err := vllm.New(context.Background(), vllm.Config{
		Name: "vt", BaseURL: srv.URL, Discover: true,
		ModelCapabilities: map[string][]string{"gemma-4-31b-it": {"reasoning"}},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	medium := "medium"
	resp, err := p.Chat(context.Background(), domain.Request{
		Model:           "gemma-4-31b-it",
		Messages:        []domain.Message{{Role: "user", Content: "hi"}},
		ReasoningEffort: &medium,
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	ctk, ok := captured["chat_template_kwargs"].(map[string]any)
	if !ok {
		t.Fatalf("upstream did not see chat_template_kwargs: %v", captured)
	}
	if ctk["enable_thinking"] != true {
		t.Errorf("enable_thinking: got %v, want true", ctk["enable_thinking"])
	}
	if resp.Thinking != "let me think" {
		t.Errorf("Thinking: got %q, want %q", resp.Thinking, "let me think")
	}
}

func TestChat_ExplicitDisableThinking_Minimal(t *testing.T) {
	var captured map[string]any
	srv := newUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, modelsResponse("gemma-4-31b-it"))
		case "/v1/chat/completions":
			b, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(b, &captured)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, chatStopResponse("answer", ""))
		}
	})

	p, err := vllm.New(context.Background(), vllm.Config{
		Name: "vt", BaseURL: srv.URL, Discover: true,
		ModelCapabilities: map[string][]string{"gemma-4-31b-it": {"reasoning"}},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	minimal := "minimal"
	if _, err := p.Chat(context.Background(), domain.Request{
		Model:           "gemma-4-31b-it",
		Messages:        []domain.Message{{Role: "user", Content: "hi"}},
		ReasoningEffort: &minimal,
	}); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	ctk, _ := captured["chat_template_kwargs"].(map[string]any)
	if ctk == nil || ctk["enable_thinking"] != false {
		t.Errorf("enable_thinking: got %v, want false", ctk)
	}
}

func TestChat_ExplicitDisableThinking_None(t *testing.T) {
	var captured map[string]any
	srv := newUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, modelsResponse("gemma-4-31b-it"))
		case "/v1/chat/completions":
			b, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(b, &captured)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, chatStopResponse("answer", ""))
		}
	})

	p, err := vllm.New(context.Background(), vllm.Config{
		Name: "vt", BaseURL: srv.URL, Discover: true,
		ModelCapabilities: map[string][]string{"gemma-4-31b-it": {"reasoning"}},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	none := "none"
	if _, err := p.Chat(context.Background(), domain.Request{
		Model:           "gemma-4-31b-it",
		Messages:        []domain.Message{{Role: "user", Content: "hi"}},
		ReasoningEffort: &none,
	}); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	ctk, _ := captured["chat_template_kwargs"].(map[string]any)
	if ctk == nil || ctk["enable_thinking"] != false {
		t.Errorf("enable_thinking: got %v, want false", ctk)
	}
}

func TestChat_ExplicitEnableThinking_Low(t *testing.T) {
	var captured map[string]any
	srv := newUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, modelsResponse("gemma-4-31b-it"))
		case "/v1/chat/completions":
			b, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(b, &captured)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, chatStopResponse("answer", ""))
		}
	})

	p, err := vllm.New(context.Background(), vllm.Config{
		Name: "vt", BaseURL: srv.URL, Discover: true,
		ModelCapabilities: map[string][]string{"gemma-4-31b-it": {"reasoning"}},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	low := "low"
	if _, err := p.Chat(context.Background(), domain.Request{
		Model:           "gemma-4-31b-it",
		Messages:        []domain.Message{{Role: "user", Content: "hi"}},
		ReasoningEffort: &low,
	}); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	ctk, _ := captured["chat_template_kwargs"].(map[string]any)
	if ctk == nil || ctk["enable_thinking"] != true {
		t.Errorf("enable_thinking: got %v, want true", ctk)
	}
}

func TestModels_DiscoverEmptyList_FallsBackToConfig(t *testing.T) {
	srv := newUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, modelsResponse()) // empty list
			return
		}
		http.NotFound(w, r)
	})

	p, err := vllm.New(context.Background(), vllm.Config{
		Name:              "vt",
		BaseURL:           srv.URL,
		Discover:          true,
		Models:            []string{"gemma-4-31b-it"},
		ModelCapabilities: map[string][]string{"gemma-4-31b-it": {"reasoning"}},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}
	if len(models) != 1 || models[0].ID != "gemma-4-31b-it" {
		t.Fatalf("expected fallback models on empty upstream list, got %+v", models)
	}
}

func TestChat_NoReasoningEffort_NoInjection(t *testing.T) {
	var captured map[string]any
	srv := newUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, modelsResponse("gemma-4-31b-it"))
		case "/v1/chat/completions":
			b, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(b, &captured)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, chatStopResponse("answer", ""))
		}
	})

	p, err := vllm.New(context.Background(), vllm.Config{
		Name: "vt", BaseURL: srv.URL, Discover: true,
		ModelCapabilities: map[string][]string{"gemma-4-31b-it": {"reasoning"}},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if _, err := p.Chat(context.Background(), domain.Request{
		Model:    "gemma-4-31b-it",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
	}); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if _, has := captured["chat_template_kwargs"]; has {
		t.Errorf("expected NO chat_template_kwargs when ReasoningEffort=nil, got %v", captured["chat_template_kwargs"])
	}
}

func TestChat_CustomThinkingKey(t *testing.T) {
	var captured map[string]any
	srv := newUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, modelsResponse("granite-3.2"))
		case "/v1/chat/completions":
			b, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(b, &captured)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, chatStopResponse("answer", ""))
		}
	})

	p, err := vllm.New(context.Background(), vllm.Config{
		Name: "vt", BaseURL: srv.URL, Discover: true,
		ThinkingKey:       "thinking",
		ModelCapabilities: map[string][]string{"granite-3.2": {"reasoning"}},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	high := "high"
	if _, err := p.Chat(context.Background(), domain.Request{
		Model:           "granite-3.2",
		Messages:        []domain.Message{{Role: "user", Content: "hi"}},
		ReasoningEffort: &high,
	}); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	ctk, _ := captured["chat_template_kwargs"].(map[string]any)
	if ctk == nil || ctk["thinking"] != true {
		t.Errorf("thinking key: got %v, want true", ctk)
	}
	if _, has := ctk["enable_thinking"]; has {
		t.Errorf("expected no enable_thinking key (custom key was set), got %v", ctk)
	}
}

func TestChatStream_ReasoningDeltas(t *testing.T) {
	srv := newUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, modelsResponse("gemma-4-31b-it"))
		case "/v1/chat/completions":
			w.Header().Set("Content-Type", "text/event-stream")
			frames := []string{
				`data: {"id":"c","model":"gemma-4-31b-it","choices":[{"index":0,"delta":{"reasoning":"step1"}}]}`,
				`data: {"id":"c","model":"gemma-4-31b-it","choices":[{"index":0,"delta":{"reasoning":"step2"}}]}`,
				`data: {"id":"c","model":"gemma-4-31b-it","choices":[{"index":0,"delta":{"content":"final"},"finish_reason":"stop"}]}`,
				`data: [DONE]`,
			}
			for _, f := range frames {
				_, _ = io.WriteString(w, f+"\n\n")
			}
		}
	})

	p, err := vllm.New(context.Background(), vllm.Config{
		Name: "vt", BaseURL: srv.URL, Discover: true,
		ModelCapabilities: map[string][]string{"gemma-4-31b-it": {"reasoning"}},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	high := "high"
	ch, err := p.ChatStream(context.Background(), domain.Request{
		Model:           "gemma-4-31b-it",
		Messages:        []domain.Message{{Role: "user", Content: "hi"}},
		ReasoningEffort: &high,
	})
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	var thinks, contents []string
	for chunk := range ch {
		if chunk.ThinkingDelta != "" {
			thinks = append(thinks, chunk.ThinkingDelta)
		}
		if chunk.Delta != "" {
			contents = append(contents, chunk.Delta)
		}
	}
	if got, want := strings.Join(thinks, ""), "step1step2"; got != want {
		t.Errorf("thinking: got %q, want %q", got, want)
	}
	if got, want := strings.Join(contents, ""), "final"; got != want {
		t.Errorf("content: got %q, want %q", got, want)
	}
}

func TestChat_CachedTokensPropagated(t *testing.T) {
	srv := newUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, modelsResponse("gemma-4-31b-it"))
		case "/v1/chat/completions":
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{
				"id":"r","model":"gemma-4-31b-it",
				"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],
				"usage":{"prompt_tokens":100,"completion_tokens":2,"total_tokens":102,
				         "prompt_tokens_details":{"cached_tokens":80}}
			}`)
		}
	})

	p, err := vllm.New(context.Background(), vllm.Config{
		Name: "vt", BaseURL: srv.URL, Discover: true,
		ModelCapabilities: map[string][]string{"gemma-4-31b-it": {"reasoning"}},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := p.Chat(context.Background(), domain.Request{
		Model:    "gemma-4-31b-it",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.Usage.CachedTokens != 80 {
		t.Errorf("CachedTokens: got %d, want 80", resp.Usage.CachedTokens)
	}
}

func TestChat_UpstreamRateLimit(t *testing.T) {
	srv := newUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, modelsResponse("gemma-4-31b-it"))
		case "/v1/chat/completions":
			w.Header().Set("Retry-After", "30")
			http.Error(w, "rate limited", http.StatusTooManyRequests)
		}
	})

	p, err := vllm.New(context.Background(), vllm.Config{
		Name: "vt", BaseURL: srv.URL, Discover: true,
		ModelCapabilities: map[string][]string{"gemma-4-31b-it": {"reasoning"}},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.Chat(context.Background(), domain.Request{
		Model:    "gemma-4-31b-it",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
	})
	var rle *provider.RateLimitError
	if !errors.As(err, &rle) {
		t.Fatalf("expected RateLimitError, got %T: %v", err, err)
	}
	if rle.RetryAfter != 30*time.Second {
		t.Errorf("RetryAfter: got %v, want 30s", rle.RetryAfter)
	}
}

func TestModels_PropagatesMaxModelLenFromUpstream(t *testing.T) {
	srv := newUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"object":"list","data":[
			{"id":"gemma-4-31b-it","object":"model","owned_by":"vllm","max_model_len":229376}
		]}`)
	})

	p, err := vllm.New(context.Background(), vllm.Config{
		Name: "vt", BaseURL: srv.URL, Discover: true,
		ModelCapabilities: map[string][]string{"gemma-4-31b-it": {"reasoning"}},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}
	if len(models) != 1 {
		t.Fatalf("got %d models, want 1", len(models))
	}
	if models[0].MaxModelLen != 229376 {
		t.Errorf("MaxModelLen: got %d, want 229376", models[0].MaxModelLen)
	}
	if models[0].OwnedBy != "vllm" {
		t.Errorf("OwnedBy: got %q, want %q", models[0].OwnedBy, "vllm")
	}
	if len(models[0].Capabilities) != 1 || models[0].Capabilities[0] != "reasoning" {
		t.Errorf("Capabilities: got %v, want [reasoning]", models[0].Capabilities)
	}
}

func TestModels_RefreshSkipsUnknownModelsButKeepsKnown(t *testing.T) {
	// Initial call returns only the known model.
	// Second call returns the known model + a new unknown one.
	// The new unknown one is excluded (not error), the known stays.
	calls := 0
	srv := newUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		calls++
		w.Header().Set("Content-Type", "application/json")
		if calls == 1 {
			_, _ = io.WriteString(w, modelsResponse("gemma-4-31b-it"))
		} else {
			_, _ = io.WriteString(w, modelsResponse("gemma-4-31b-it", "surprise-model"))
		}
	})

	p, err := vllm.New(context.Background(), vllm.Config{
		Name: "vt", BaseURL: srv.URL, Discover: true,
		ModelCapabilities: map[string][]string{"gemma-4-31b-it": {"reasoning"}},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models(refresh): %v", err)
	}
	if len(models) != 1 || models[0].ID != "gemma-4-31b-it" {
		t.Fatalf("expected only gemma-4-31b-it after refresh, got %+v", models)
	}
}
