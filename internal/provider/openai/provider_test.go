package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
	"github.com/JetManiack/go-ai-proxy/internal/provider"
	"github.com/JetManiack/go-ai-proxy/internal/provider/openai"
)

func newFakeServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return srv
}

func TestChat_BasicResponse(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":    "resp-1",
			"model": "local-model",
			"choices": []map[string]any{
				{"index": 0, "message": map[string]any{"role": "assistant", "content": "pong"}, "finish_reason": "stop"},
			},
			"usage": map[string]any{"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
		})
	})

	p := openai.New(srv.URL)
	resp, err := p.Chat(context.Background(), domain.Request{
		Model:    "local-model",
		Messages: []domain.Message{{Role: "user", Content: "ping"}},
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if resp.Message.Content != "pong" {
		t.Errorf("content: got %q, want %q", resp.Message.Content, "pong")
	}
	if resp.Usage.PromptTokens != 5 {
		t.Errorf("prompt_tokens: got %d", resp.Usage.PromptTokens)
	}
}

func TestChat_SendsAPIKey(t *testing.T) {
	var gotAuth string
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id": "r", "model": "m",
			"choices": []map[string]any{
				{"index": 0, "message": map[string]any{"role": "assistant", "content": "ok"}, "finish_reason": "stop"},
			},
			"usage": map[string]any{},
		})
	})

	p := openai.New(srv.URL, openai.WithAPIKey("sk-test-key"))
	p.Chat(context.Background(), domain.Request{
		Model:    "m",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
	})

	if gotAuth != "Bearer sk-test-key" {
		t.Errorf("Authorization: got %q, want %q", gotAuth, "Bearer sk-test-key")
	}
}

func TestChat_NoAPIKeyNoAuthHeader(t *testing.T) {
	var gotAuth string
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id": "r", "model": "m",
			"choices": []map[string]any{
				{"index": 0, "message": map[string]any{"role": "assistant", "content": "ok"}, "finish_reason": "stop"},
			},
			"usage": map[string]any{},
		})
	})

	p := openai.New(srv.URL) // no API key
	p.Chat(context.Background(), domain.Request{
		Model:    "m",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
	})

	if gotAuth != "" {
		t.Errorf("expected no Authorization header, got %q", gotAuth)
	}
}

func TestChat_UpstreamError(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":{"message":"boom"}}`))
	})

	p := openai.New(srv.URL)
	_, err := p.Chat(context.Background(), domain.Request{
		Model:    "m",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Error("expected error for 500 response")
	}
}

func TestModels_ReturnsList(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"data": []map[string]any{
				{"id": "model-a", "object": "model", "owned_by": "local"},
				{"id": "model-b", "object": "model", "owned_by": "local"},
			},
		})
	})

	p := openai.New(srv.URL)
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models error: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("models len: got %d, want 2", len(models))
	}
	if models[0].ID != "model-a" {
		t.Errorf("first model: got %q", models[0].ID)
	}
}

func TestModels_UpstreamError(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	})

	p := openai.New(srv.URL)
	_, err := p.Models(context.Background())
	if err == nil {
		t.Error("expected error for 503 response")
	}
}

func TestChatStream_ReturnsChannel(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		chunk := map[string]any{
			"id": "c1", "model": "m",
			"choices": []map[string]any{
				{"index": 0, "delta": map[string]any{"content": "hi"}, "finish_reason": nil},
			},
		}
		b, _ := json.Marshal(chunk)
		w.Write([]byte("data: " + string(b) + "\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
		if flusher != nil {
			flusher.Flush()
		}
	})

	p := openai.New(srv.URL)
	ch, err := p.ChatStream(context.Background(), domain.Request{
		Model:    "m",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("ChatStream error: %v", err)
	}
	if ch == nil {
		t.Error("expected non-nil channel")
	}
}

func TestChat_Timeout(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(200 * time.Millisecond)
	})

	p := openai.New(srv.URL, openai.WithTimeout(50*time.Millisecond))
	_, err := p.Chat(context.Background(), domain.Request{
		Model:    "m",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Error("expected timeout error")
	}
}

func TestChat_429_RateLimitError(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "30")
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte(`{"error":{"message":"rate limit exceeded"}}`))
	})

	p := openai.New(srv.URL)
	_, err := p.Chat(context.Background(), domain.Request{
		Model:    "m",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
	})
	var rl *provider.RateLimitError
	if !errors.As(err, &rl) {
		t.Fatalf("expected *provider.RateLimitError, got %T: %v", err, err)
	}
	if rl.RetryAfter != 30*time.Second {
		t.Errorf("RetryAfter: got %v, want 30s", rl.RetryAfter)
	}
}

func TestChat_429_DefaultRetryAfter(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
	})

	p := openai.New(srv.URL)
	_, err := p.Chat(context.Background(), domain.Request{
		Model:    "m",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
	})
	var rl *provider.RateLimitError
	if !errors.As(err, &rl) {
		t.Fatalf("expected *provider.RateLimitError, got %T: %v", err, err)
	}
	if rl.RetryAfter != 60*time.Second {
		t.Errorf("RetryAfter: got %v, want 60s (default)", rl.RetryAfter)
	}
}

func TestChatStream_429_RateLimitError(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "15")
		w.WriteHeader(http.StatusTooManyRequests)
	})

	p := openai.New(srv.URL)
	_, err := p.ChatStream(context.Background(), domain.Request{
		Model:    "m",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
	})
	var rl *provider.RateLimitError
	if !errors.As(err, &rl) {
		t.Fatalf("expected *provider.RateLimitError, got %T: %v", err, err)
	}
	if rl.RetryAfter != 15*time.Second {
		t.Errorf("RetryAfter: got %v, want 15s", rl.RetryAfter)
	}
}

func TestModels_Timeout(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(200 * time.Millisecond)
	})

	p := openai.New(srv.URL, openai.WithTimeout(50*time.Millisecond))
	_, err := p.Models(context.Background())
	if err == nil {
		t.Error("expected timeout error")
	}
}

// --- Thinking / CoT extraction ---

func TestChat_ThinkTagsExtracted(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id": "r1", "model": "qwq",
			"choices": []map[string]any{
				{"index": 0, "message": map[string]any{
					"role":    "assistant",
					"content": "<think>\nstep by step\n</think>\nfinal answer",
				}, "finish_reason": "stop"},
			},
			"usage": map[string]any{"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
		})
	})

	p := openai.New(srv.URL)
	resp, err := p.Chat(context.Background(), domain.Request{
		Model:    "qwq",
		Messages: []domain.Message{{Role: "user", Content: "think"}},
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if resp.Thinking != "\nstep by step\n" {
		t.Errorf("Thinking: got %q, want %q", resp.Thinking, "\nstep by step\n")
	}
	if resp.Message.Content != "final answer" {
		t.Errorf("Content: got %q, want %q", resp.Message.Content, "final answer")
	}
}

func TestChatStream_ThinkTagsExtracted(t *testing.T) {
	deltas := []string{"<think>", "\nreasoning\n", "</think>\n", "answer"}
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for _, d := range deltas {
			chunk := map[string]any{
				"id": "c1", "model": "qwq",
				"choices": []map[string]any{
					{"index": 0, "delta": map[string]any{"content": d}, "finish_reason": nil},
				},
			}
			b, _ := json.Marshal(chunk)
			w.Write([]byte("data: " + string(b) + "\n\n"))
		}
		w.Write([]byte("data: [DONE]\n\n"))
		if flusher != nil {
			flusher.Flush()
		}
	})

	p := openai.New(srv.URL)
	ch, err := p.ChatStream(context.Background(), domain.Request{
		Model:    "qwq",
		Messages: []domain.Message{{Role: "user", Content: "think"}},
	})
	if err != nil {
		t.Fatalf("ChatStream error: %v", err)
	}

	var thinking, content string
	for chunk := range ch {
		if chunk.Done {
			break
		}
		thinking += chunk.ThinkingDelta
		content += chunk.Delta
	}

	if thinking != "\nreasoning\n" {
		t.Errorf("thinking: got %q, want %q", thinking, "\nreasoning\n")
	}
	if content != "answer" {
		t.Errorf("content: got %q, want %q", content, "answer")
	}
}

func TestChat_RequestBodyMutator_InjectsField(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &got)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"r","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer srv.Close()

	mutator := func(body []byte, req domain.Request) ([]byte, error) {
		var obj map[string]any
		if err := json.Unmarshal(body, &obj); err != nil {
			return nil, err
		}
		obj["injected"] = "yes"
		return json.Marshal(obj)
	}

	p := openai.New(srv.URL, openai.WithRequestBodyMutator(mutator))
	if _, err := p.Chat(context.Background(), domain.Request{
		Model:    "m",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
	}); err != nil {
		t.Fatalf("Chat: %v", err)
	}

	if got["injected"] != "yes" {
		t.Errorf("upstream did not see mutator injection; body keys: %v", got)
	}
}

func TestChatStream_RequestBodyMutator_InjectsField(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &got)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
	}))
	defer srv.Close()

	mutator := func(body []byte, req domain.Request) ([]byte, error) {
		var obj map[string]any
		if err := json.Unmarshal(body, &obj); err != nil {
			return nil, err
		}
		obj["injected"] = "stream"
		return json.Marshal(obj)
	}

	p := openai.New(srv.URL, openai.WithRequestBodyMutator(mutator))
	ch, err := p.ChatStream(context.Background(), domain.Request{
		Model:    "m",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	for range ch {
		// drain
	}

	if got["injected"] != "stream" {
		t.Errorf("upstream did not see mutator injection in stream path; body keys: %v", got)
	}
}

func TestChat_NoMutator_BodyUnchanged(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &got)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"r","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer srv.Close()

	p := openai.New(srv.URL)
	if _, err := p.Chat(context.Background(), domain.Request{
		Model:    "m",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
	}); err != nil {
		t.Fatalf("Chat: %v", err)
	}

	if _, has := got["injected"]; has {
		t.Errorf("body should not have 'injected' key when no mutator is set; got: %v", got)
	}
}
