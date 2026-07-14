package anthropic_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/JetManiack/go-ai-proxy/internal/auth"
	"github.com/JetManiack/go-ai-proxy/internal/domain"
	anthropicprovider "github.com/JetManiack/go-ai-proxy/internal/provider/anthropic"
)

func anthropicTextResponse(id, model, text string, inputTokens, outputTokens int) map[string]any {
	return map[string]any{
		"id":   id,
		"type": "message",
		"role": "assistant",
		"content": []map[string]any{
			{"type": "text", "text": text},
		},
		"model":       model,
		"stop_reason": "end_turn",
		"usage": map[string]any{
			"input_tokens":  inputTokens,
			"output_tokens": outputTokens,
		},
	}
}

func newFakeAnthropicServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return srv
}

func TestChat_BasicTextResponse(t *testing.T) {
	srv := newFakeAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("unexpected method: %s", r.Method)
		}
		if r.Header.Get("x-api-key") == "" && r.Header.Get("Authorization") == "" {
			t.Error("expected auth header")
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(anthropicTextResponse("msg-1", "claude-sonnet-4-6", "Hello!", 10, 5))
	})

	p := anthropicprovider.New(auth.NewAPIKey("sk-test"), anthropicprovider.WithBaseURL(srv.URL))
	resp, err := p.Chat(context.Background(), domain.Request{
		Model:    "claude-sonnet-4-6",
		Messages: []domain.Message{{Role: "user", Content: "Hi"}},
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if resp.Message.Content != "Hello!" {
		t.Errorf("content: got %q, want %q", resp.Message.Content, "Hello!")
	}
	if resp.Usage.PromptTokens != 10 {
		t.Errorf("prompt tokens: got %d, want 10", resp.Usage.PromptTokens)
	}
	if resp.Usage.CompletionTokens != 5 {
		t.Errorf("completion tokens: got %d, want 5", resp.Usage.CompletionTokens)
	}
}

func TestChat_SystemMessageExtracted(t *testing.T) {
	var gotBody map[string]any
	srv := newFakeAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(anthropicTextResponse("m", "m", "ok", 1, 1))
	})

	p := anthropicprovider.New(auth.NewAPIKey("key"), anthropicprovider.WithBaseURL(srv.URL))
	p.Chat(context.Background(), domain.Request{
		Model: "claude-sonnet-4-6",
		Messages: []domain.Message{
			{Role: "system", Content: "You are helpful."},
			{Role: "user", Content: "Hi"},
		},
	})

	// Anthropic expects system as top-level field, not in messages array.
	system, ok := gotBody["system"]
	if !ok {
		t.Fatal("expected system field in request")
	}
	// system is a string or array; either way must contain the text.
	systemStr, _ := json.Marshal(system)
	if !contains(string(systemStr), "You are helpful.") {
		t.Errorf("system field does not contain expected text: %s", systemStr)
	}

	// messages array should not contain a system entry.
	msgs := gotBody["messages"].([]any)
	for _, m := range msgs {
		msg := m.(map[string]any)
		if msg["role"] == "system" {
			t.Error("messages array should not contain system message")
		}
	}
}

func TestChat_ToolCallResponse(t *testing.T) {
	srv := newFakeAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":   "msg-2",
			"type": "message",
			"role": "assistant",
			"content": []map[string]any{
				{
					"type":  "tool_use",
					"id":    "tool-1",
					"name":  "get_weather",
					"input": map[string]any{"city": "London"},
				},
			},
			"model":       "claude-sonnet-4-6",
			"stop_reason": "tool_use",
			"usage":       map[string]any{"input_tokens": 20, "output_tokens": 10},
		})
	})

	p := anthropicprovider.New(auth.NewAPIKey("key"), anthropicprovider.WithBaseURL(srv.URL))
	resp, err := p.Chat(context.Background(), domain.Request{
		Model:    "claude-sonnet-4-6",
		Messages: []domain.Message{{Role: "user", Content: "What's the weather?"}},
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(resp.Message.ToolCalls))
	}
	tc := resp.Message.ToolCalls[0]
	if tc.Name != "get_weather" {
		t.Errorf("tool name: got %q, want get_weather", tc.Name)
	}
	if tc.ID != "tool-1" {
		t.Errorf("tool id: got %q, want tool-1", tc.ID)
	}
}

func TestChat_ToolResultMessagesGrouped(t *testing.T) {
	var gotBody map[string]any
	srv := newFakeAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(anthropicTextResponse("m", "m", "done", 1, 1))
	})

	p := anthropicprovider.New(auth.NewAPIKey("key"), anthropicprovider.WithBaseURL(srv.URL))
	p.Chat(context.Background(), domain.Request{
		Model: "claude-sonnet-4-6",
		Messages: []domain.Message{
			{Role: "user", Content: "Use tools"},
			{Role: "assistant", ToolCalls: []domain.ToolCall{
				{ID: "t1", Name: "fn1", Arguments: `{}`},
				{ID: "t2", Name: "fn2", Arguments: `{}`},
			}},
			{Role: "tool", Content: "result1", ToolCallID: "t1"},
			{Role: "tool", Content: "result2", ToolCallID: "t2"},
		},
	})

	msgs := gotBody["messages"].([]any)
	// Expected: user, assistant, user (with 2 tool_result blocks)
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(msgs))
	}
	lastMsg := msgs[2].(map[string]any)
	if lastMsg["role"] != "user" {
		t.Errorf("last message role: got %q, want user", lastMsg["role"])
	}
	content := lastMsg["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("expected 2 tool_result blocks, got %d", len(content))
	}
	for i, block := range content {
		b := block.(map[string]any)
		if b["type"] != "tool_result" {
			t.Errorf("block[%d] type: got %q, want tool_result", i, b["type"])
		}
	}
}

func TestChat_UpstreamError(t *testing.T) {
	srv := newFakeAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    "authentication_error",
				"message": "invalid api key",
			},
		})
	})

	p := anthropicprovider.New(auth.NewAPIKey("bad-key"), anthropicprovider.WithBaseURL(srv.URL))
	_, err := p.Chat(context.Background(), domain.Request{
		Model:    "claude-sonnet-4-6",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Error("expected error for 401 response")
	}
}

func TestModels_ReturnsList(t *testing.T) {
	srv := newFakeAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"id": "claude-sonnet-4-6", "type": "model"},
				{"id": "claude-opus-4-6", "type": "model"},
			},
			"has_more":     false,
			"first_id":     "claude-sonnet-4-6",
			"last_id":      "claude-opus-4-6",
			"object":       "list",
		})
	})

	p := anthropicprovider.New(auth.NewAPIKey("key"), anthropicprovider.WithBaseURL(srv.URL))
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models error: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(models))
	}
	if models[0].ID != "claude-sonnet-4-6" {
		t.Errorf("first model: got %q", models[0].ID)
	}
}

// --- Streaming ---

type sseEvent struct{ typ, data string }

func writeSSEEvents(w http.ResponseWriter, events []sseEvent) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	flusher := w.(http.Flusher)
	for _, e := range events {
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", e.typ, e.data)
		flusher.Flush()
	}
}

func TestChatStream_TextChunks(t *testing.T) {
	srv := newFakeAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
		writeSSEEvents(w, []sseEvent{
			{"message_start", `{"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","content":[],"model":"m","stop_reason":null,"usage":{"input_tokens":5,"output_tokens":0}}}`},
			{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}`},
			{"content_block_stop", `{"type":"content_block_stop","index":0}`},
			{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`},
			{"message_stop", `{"type":"message_stop"}`},
		})
	})

	p := anthropicprovider.New(auth.NewAPIKey("key"), anthropicprovider.WithBaseURL(srv.URL))
	ch, err := p.ChatStream(context.Background(), domain.Request{
		Model:    "m",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("ChatStream error: %v", err)
	}

	var deltas []string
	for chunk := range ch {
		if chunk.Done {
			break
		}
		if chunk.Delta != "" {
			deltas = append(deltas, chunk.Delta)
		}
	}
	if len(deltas) != 2 {
		t.Fatalf("expected 2 text deltas, got %d: %v", len(deltas), deltas)
	}
	if deltas[0] != "Hello" || deltas[1] != " world" {
		t.Errorf("deltas: got %v", deltas)
	}
}

func TestChatStream_ToolCallChunks(t *testing.T) {
	srv := newFakeAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
		writeSSEEvents(w, []sseEvent{
			{"message_start", `{"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","content":[],"model":"m","stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}`},
			{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"London\"}"}}`},
			{"content_block_stop", `{"type":"content_block_stop","index":0}`},
			{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}`},
			{"message_stop", `{"type":"message_stop"}`},
		})
	})

	p := anthropicprovider.New(auth.NewAPIKey("key"), anthropicprovider.WithBaseURL(srv.URL))
	ch, err := p.ChatStream(context.Background(), domain.Request{
		Model:    "m",
		Messages: []domain.Message{{Role: "user", Content: "weather?"}},
	})
	if err != nil {
		t.Fatalf("ChatStream error: %v", err)
	}

	var chunks []domain.Chunk
	for chunk := range ch {
		if chunk.Done {
			break
		}
		chunks = append(chunks, chunk)
	}
	if len(chunks) == 0 {
		t.Fatal("expected tool call chunks, got none")
	}

	// First chunk: tool call start.
	start := chunks[0]
	if start.ToolCallID != "toolu_1" {
		t.Errorf("tool call id: got %q, want toolu_1", start.ToolCallID)
	}
	if start.ToolCallName != "get_weather" {
		t.Errorf("tool call name: got %q, want get_weather", start.ToolCallName)
	}
	if start.ToolCallIndex == nil || *start.ToolCallIndex != 0 {
		t.Errorf("tool call index: got %v, want 0", start.ToolCallIndex)
	}

	// Remaining chunks: accumulated args.
	var args string
	for _, c := range chunks[1:] {
		args += c.ToolCallArgs
	}
	if args != `{"city":"London"}` {
		t.Errorf("accumulated args: got %q, want %q", args, `{"city":"London"}`)
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || len(sub) == 0 ||
		func() bool {
			for i := 0; i <= len(s)-len(sub); i++ {
				if s[i:i+len(sub)] == sub {
					return true
				}
			}
			return false
		}())
}

func TestChat_ExtendedThinkingSendsConfig(t *testing.T) {
	var gotBody map[string]any
	srv := newFakeAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id": "msg-1", "type": "message", "role": "assistant",
			"content": []map[string]any{
				{"type": "thinking", "thinking": "let me think...", "signature": "sig"},
				{"type": "text", "text": "42"},
			},
			"model": "claude-sonnet-4-6", "stop_reason": "end_turn",
			"usage": map[string]any{"input_tokens": 10, "output_tokens": 30},
		})
	})

	budget := 8000
	p := anthropicprovider.New(auth.NewAPIKey("sk-test"), anthropicprovider.WithBaseURL(srv.URL))
	resp, err := p.Chat(context.Background(), domain.Request{
		Model:        "claude-sonnet-4-6",
		Messages:     []domain.Message{{Role: "user", Content: "think deeply"}},
		BudgetTokens: &budget,
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}

	thinking, ok := gotBody["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("thinking not in request body: %v", gotBody)
	}
	if thinking["type"] != "enabled" {
		t.Errorf("thinking.type: got %v, want enabled", thinking["type"])
	}
	if bt, _ := thinking["budget_tokens"].(float64); int(bt) != 8000 {
		t.Errorf("thinking.budget_tokens: got %v, want 8000", thinking["budget_tokens"])
	}
	if resp.Thinking != "let me think..." {
		t.Errorf("Thinking: got %q, want %q", resp.Thinking, "let me think...")
	}
	if resp.Message.Content != "42" {
		t.Errorf("Content: got %q, want %q", resp.Message.Content, "42")
	}
}

func TestChat_ExtendedThinkingIgnoresTemperature(t *testing.T) {
	var gotBody map[string]any
	srv := newFakeAnthropicServer(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(anthropicTextResponse("msg-1", "claude-sonnet-4-6", "hi", 5, 5))
	})

	budget := 5000
	temp := 0.5
	p := anthropicprovider.New(auth.NewAPIKey("sk-test"), anthropicprovider.WithBaseURL(srv.URL))
	_, err := p.Chat(context.Background(), domain.Request{
		Model:        "claude-sonnet-4-6",
		Messages:     []domain.Message{{Role: "user", Content: "hi"}},
		BudgetTokens: &budget,
		Temperature:  &temp,
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	// temperature must not be sent when thinking is enabled (Anthropic requires temp=1)
	if _, ok := gotBody["temperature"]; ok {
		t.Error("temperature should not be sent when extended thinking is enabled")
	}
}
