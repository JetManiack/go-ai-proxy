package translator_test

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
	"github.com/JetManiack/go-ai-proxy/internal/translator"
)

// --- RequestFromOpenAI ---

func TestRequestFromOpenAI_SimpleUserMessage(t *testing.T) {
	body := `{
		"model": "my-model",
		"messages": [{"role": "user", "content": "hello"}]
	}`
	req, err := translator.RequestFromOpenAI([]byte(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if req.Model != "my-model" {
		t.Errorf("model: got %q, want %q", req.Model, "my-model")
	}
	if len(req.Messages) != 1 {
		t.Fatalf("messages len: got %d, want 1", len(req.Messages))
	}
	if req.Messages[0].Role != "user" || req.Messages[0].Content != "hello" {
		t.Errorf("message: got %+v", req.Messages[0])
	}
}

func TestRequestFromOpenAI_SystemMessage(t *testing.T) {
	body := `{
		"model": "m",
		"messages": [
			{"role": "system", "content": "you are a bot"},
			{"role": "user", "content": "hi"}
		]
	}`
	req, err := translator.RequestFromOpenAI([]byte(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(req.Messages) != 2 {
		t.Fatalf("messages len: got %d, want 2", len(req.Messages))
	}
	if req.Messages[0].Role != "system" {
		t.Errorf("expected system role, got %q", req.Messages[0].Role)
	}
}

func TestRequestFromOpenAI_MultiTurn(t *testing.T) {
	body := `{
		"model": "m",
		"messages": [
			{"role": "user", "content": "what is 2+2?"},
			{"role": "assistant", "content": "4"},
			{"role": "user", "content": "and 3+3?"}
		]
	}`
	req, err := translator.RequestFromOpenAI([]byte(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(req.Messages) != 3 {
		t.Fatalf("messages len: got %d, want 3", len(req.Messages))
	}
}

func TestRequestFromOpenAI_TemperatureAndMaxTokens(t *testing.T) {
	temp := 0.7
	maxTok := 100
	body, _ := json.Marshal(map[string]any{
		"model":       "m",
		"messages":    []map[string]any{{"role": "user", "content": "hi"}},
		"temperature": temp,
		"max_tokens":  maxTok,
	})
	req, err := translator.RequestFromOpenAI(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if req.Temperature == nil || *req.Temperature != temp {
		t.Errorf("temperature: got %v, want %v", req.Temperature, temp)
	}
	if req.MaxTokens == nil || *req.MaxTokens != maxTok {
		t.Errorf("max_tokens: got %v, want %v", req.MaxTokens, maxTok)
	}
}

func TestRequestFromOpenAI_StreamFlag(t *testing.T) {
	body := `{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true}`
	req, err := translator.RequestFromOpenAI([]byte(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !req.Stream {
		t.Error("expected stream=true")
	}
}

func TestRequestFromOpenAI_InvalidJSON(t *testing.T) {
	_, err := translator.RequestFromOpenAI([]byte(`not json`))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

// --- RequestToOpenAI ---

func TestRequestToOpenAI_RoundTrip(t *testing.T) {
	original := `{"model":"m","messages":[{"role":"user","content":"hi"}]}`
	req, err := translator.RequestFromOpenAI([]byte(original))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	out, err := translator.RequestToOpenAI(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("output is not valid JSON: %v", err)
	}
	if m["model"] != "m" {
		t.Errorf("model: got %v", m["model"])
	}
}

func TestRequestToOpenAI_OmitsNilOptionals(t *testing.T) {
	req := domain.Request{
		Model:    "m",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
	}
	out, err := translator.RequestToOpenAI(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var m map[string]any
	json.Unmarshal(out, &m)
	if _, ok := m["temperature"]; ok {
		t.Error("temperature should be absent when nil")
	}
	if _, ok := m["max_tokens"]; ok {
		t.Error("max_tokens should be absent when nil")
	}
}

// --- ResponseToOpenAI ---

func TestResponseToOpenAI_Basic(t *testing.T) {
	resp := domain.Response{
		ID:    "resp-1",
		Model: "m",
		Message: domain.Message{
			Role:    "assistant",
			Content: "hello there",
		},
		Usage: domain.Usage{PromptTokens: 5, CompletionTokens: 3},
	}
	out, err := translator.ResponseToOpenAI(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("output is not valid JSON: %v", err)
	}
	choices, ok := m["choices"].([]any)
	if !ok || len(choices) == 0 {
		t.Fatal("expected choices array")
	}
	choice := choices[0].(map[string]any)
	msg := choice["message"].(map[string]any)
	if msg["content"] != "hello there" {
		t.Errorf("content: got %v", msg["content"])
	}
	if msg["role"] != "assistant" {
		t.Errorf("role: got %v", msg["role"])
	}
	usage := m["usage"].(map[string]any)
	if usage["prompt_tokens"].(float64) != 5 {
		t.Errorf("prompt_tokens: got %v", usage["prompt_tokens"])
	}
}

// --- ResponseFromOpenAI ---

func TestResponseFromOpenAI_Basic(t *testing.T) {
	body := `{
		"id": "resp-1",
		"object": "chat.completion",
		"model": "my-model",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "hello"},
			"finish_reason": "stop"
		}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
	}`
	resp, err := translator.ResponseFromOpenAI([]byte(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.ID != "resp-1" {
		t.Errorf("id: got %q", resp.ID)
	}
	if resp.Message.Content != "hello" {
		t.Errorf("content: got %q", resp.Message.Content)
	}
	if resp.Usage.PromptTokens != 10 {
		t.Errorf("prompt_tokens: got %d", resp.Usage.PromptTokens)
	}
}

func TestResponseFromOpenAI_EmptyChoices(t *testing.T) {
	body := `{"id":"r","model":"m","choices":[],"usage":{}}`
	_, err := translator.ResponseFromOpenAI([]byte(body))
	if err == nil {
		t.Error("expected error for empty choices")
	}
}

// --- Tool calling: request ---

func TestRequestFromOpenAI_ToolDefinitions(t *testing.T) {
	body := `{
		"model": "m",
		"messages": [{"role": "user", "content": "use a tool"}],
		"tools": [{
			"type": "function",
			"function": {
				"name": "get_weather",
				"description": "Gets weather",
				"parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
			}
		}]
	}`
	req, err := translator.RequestFromOpenAI([]byte(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(req.Tools) != 1 {
		t.Fatalf("tools len: got %d, want 1", len(req.Tools))
	}
	if req.Tools[0].Name != "get_weather" {
		t.Errorf("tool name: got %q", req.Tools[0].Name)
	}
	if req.Tools[0].Description != "Gets weather" {
		t.Errorf("tool description: got %q", req.Tools[0].Description)
	}
}

func TestRequestFromOpenAI_AssistantWithToolCalls(t *testing.T) {
	body := `{
		"model": "m",
		"messages": [{
			"role": "assistant",
			"content": null,
			"tool_calls": [{
				"id": "call-1",
				"type": "function",
				"function": {"name": "get_weather", "arguments": "{\"city\":\"London\"}"}
			}]
		}]
	}`
	req, err := translator.RequestFromOpenAI([]byte(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(req.Messages[0].ToolCalls) != 1 {
		t.Fatalf("tool_calls len: got %d, want 1", len(req.Messages[0].ToolCalls))
	}
	tc := req.Messages[0].ToolCalls[0]
	if tc.ID != "call-1" || tc.Name != "get_weather" {
		t.Errorf("tool call: id=%q name=%q", tc.ID, tc.Name)
	}
	if tc.Arguments != `{"city":"London"}` {
		t.Errorf("arguments: got %q", tc.Arguments)
	}
}

func TestRequestFromOpenAI_ToolResultMessage(t *testing.T) {
	body := `{
		"model": "m",
		"messages": [{"role": "tool", "content": "sunny", "tool_call_id": "call-1"}]
	}`
	req, err := translator.RequestFromOpenAI([]byte(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	msg := req.Messages[0]
	if msg.Role != "tool" || msg.Content != "sunny" || msg.ToolCallID != "call-1" {
		t.Errorf("tool result message: %+v", msg)
	}
}

func TestRequestToOpenAI_WithTools(t *testing.T) {
	req := domain.Request{
		Model:    "m",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
		Tools: []domain.Tool{{
			Name:        "fn",
			Description: "does fn",
			Parameters:  map[string]any{"type": "object"},
		}},
	}
	out, err := translator.RequestToOpenAI(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var m map[string]any
	json.Unmarshal(out, &m)
	tools, ok := m["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools: expected 1, got %v", m["tools"])
	}
	fn := tools[0].(map[string]any)["function"].(map[string]any)
	if fn["name"] != "fn" {
		t.Errorf("tool name: got %v", fn["name"])
	}
}

// --- Tool calling: response ---

func TestResponseToOpenAI_WithToolCalls(t *testing.T) {
	resp := domain.Response{
		ID:    "r1",
		Model: "m",
		Message: domain.Message{
			Role: "assistant",
			ToolCalls: []domain.ToolCall{
				{ID: "call-1", Name: "get_weather", Arguments: `{"city":"London"}`},
			},
		},
	}
	out, err := translator.ResponseToOpenAI(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var m map[string]any
	json.Unmarshal(out, &m)
	msg := m["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	toolCalls := msg["tool_calls"].([]any)
	if len(toolCalls) != 1 {
		t.Fatalf("tool_calls len: got %d, want 1", len(toolCalls))
	}
	tc := toolCalls[0].(map[string]any)
	if tc["id"] != "call-1" {
		t.Errorf("id: got %v", tc["id"])
	}
	fn := tc["function"].(map[string]any)
	if fn["name"] != "get_weather" {
		t.Errorf("name: got %v", fn["name"])
	}
	if fn["arguments"] != `{"city":"London"}` {
		t.Errorf("arguments: got %v", fn["arguments"])
	}
}

func TestResponseFromOpenAI_WithToolCalls(t *testing.T) {
	body := `{
		"id": "r1", "model": "m",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": null,
				"tool_calls": [{"id": "call-1", "type": "function",
					"function": {"name": "get_weather", "arguments": "{\"city\":\"London\"}"}}]
			},
			"finish_reason": "tool_calls"
		}],
		"usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
	}`
	resp, err := translator.ResponseFromOpenAI([]byte(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("tool_calls len: got %d, want 1", len(resp.Message.ToolCalls))
	}
	tc := resp.Message.ToolCalls[0]
	if tc.ID != "call-1" || tc.Name != "get_weather" {
		t.Errorf("tool call: id=%q name=%q", tc.ID, tc.Name)
	}
	if tc.Arguments != `{"city":"London"}` {
		t.Errorf("arguments: got %q", tc.Arguments)
	}
}

// --- Tool calling: streaming chunks ---

func TestChunkToOpenAI_ToolCallStart(t *testing.T) {
	idx := 0
	out, err := translator.ChunkToOpenAI(domain.Chunk{
		ID:           "c1",
		Model:        "m",
		ToolCallIndex: &idx,
		ToolCallID:   "call-1",
		ToolCallName: "get_weather",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var m map[string]any
	json.Unmarshal(out, &m)
	delta := m["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)
	toolCalls := delta["tool_calls"].([]any)
	if len(toolCalls) != 1 {
		t.Fatalf("tool_calls len: want 1, got %d", len(toolCalls))
	}
	tc := toolCalls[0].(map[string]any)
	if tc["id"] != "call-1" {
		t.Errorf("id: got %v", tc["id"])
	}
	fn := tc["function"].(map[string]any)
	if fn["name"] != "get_weather" {
		t.Errorf("name: got %v", fn["name"])
	}
}

func TestChunkToOpenAI_ToolCallArgsDelta(t *testing.T) {
	idx := 0
	out, err := translator.ChunkToOpenAI(domain.Chunk{
		ToolCallIndex: &idx,
		ToolCallArgs:  `{"city":`,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var m map[string]any
	json.Unmarshal(out, &m)
	delta := m["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)
	toolCalls := delta["tool_calls"].([]any)
	tc := toolCalls[0].(map[string]any)
	fn := tc["function"].(map[string]any)
	if fn["arguments"] != `{"city":` {
		t.Errorf("arguments: got %v", fn["arguments"])
	}
}

// --- ModelsToOpenAI / ModelsFromOpenAI ---

func TestModelsRoundTrip(t *testing.T) {
	models := []domain.Model{
		{ID: "model-a", OwnedBy: "provider-x"},
		{ID: "model-b", OwnedBy: "provider-x"},
	}
	out, err := translator.ModelsToOpenAI(models)
	if err != nil {
		t.Fatalf("ModelsToOpenAI error: %v", err)
	}
	parsed, err := translator.ModelsFromOpenAI(out)
	if err != nil {
		t.Fatalf("ModelsFromOpenAI error: %v", err)
	}
	if len(parsed) != 2 {
		t.Fatalf("models len: got %d, want 2", len(parsed))
	}
	if parsed[0].ID != "model-a" || parsed[1].ID != "model-b" {
		t.Errorf("models: got %+v", parsed)
	}
}

// --- Thinking / CoT ---

func TestResponseFromOpenAI_ReasoningContent(t *testing.T) {
	body := `{
		"id": "r1", "model": "deepseek-r1",
		"choices": [{"index":0,"message":{"role":"assistant","content":"answer","reasoning_content":"I thought about it"},"finish_reason":"stop"}],
		"usage": {"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}
	}`
	resp, err := translator.ResponseFromOpenAI([]byte(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Thinking != "I thought about it" {
		t.Errorf("Thinking: got %q, want %q", resp.Thinking, "I thought about it")
	}
	if resp.Message.Content != "answer" {
		t.Errorf("Content: got %q, want %q", resp.Message.Content, "answer")
	}
}

func TestResponseFromOpenAI_ThinkTags(t *testing.T) {
	body := `{
		"id": "r2", "model": "qwq",
		"choices": [{"index":0,"message":{"role":"assistant","content":"<think>\nhmm\n</think>\nthe answer"},"finish_reason":"stop"}],
		"usage": {}
	}`
	resp, err := translator.ResponseFromOpenAI([]byte(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Thinking != "\nhmm\n" {
		t.Errorf("Thinking: got %q, want %q", resp.Thinking, "\nhmm\n")
	}
	if resp.Message.Content != "the answer" {
		t.Errorf("Content: got %q, want %q", resp.Message.Content, "the answer")
	}
}

func TestResponseToOpenAI_WritesReasoningContent(t *testing.T) {
	resp := domain.Response{
		ID:    "r1",
		Model: "m",
		Message: domain.Message{
			Role:    "assistant",
			Content: "hello",
		},
		Thinking: "step by step",
	}
	out, err := translator.ResponseToOpenAI(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var body map[string]any
	json.Unmarshal(out, &body)
	msg := body["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	if msg["reasoning_content"] != "step by step" {
		t.Errorf("reasoning_content: got %v", msg["reasoning_content"])
	}
	if msg["content"] != "hello" {
		t.Errorf("content: got %v", msg["content"])
	}
}

func TestResponseToOpenAI_NoReasoningContentWhenEmpty(t *testing.T) {
	resp := domain.Response{
		ID:      "r1",
		Model:   "m",
		Message: domain.Message{Role: "assistant", Content: "hi"},
	}
	out, _ := translator.ResponseToOpenAI(resp)
	var body map[string]any
	json.Unmarshal(out, &body)
	msg := body["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	if _, ok := msg["reasoning_content"]; ok {
		t.Error("reasoning_content should be absent when Thinking is empty")
	}
}

func TestChunkFromOpenAI_ReasoningContentDelta(t *testing.T) {
	data := `{"id":"c1","model":"m","choices":[{"index":0,"delta":{"reasoning_content":"think..."},"finish_reason":null}]}`
	chunk, err := translator.ChunkFromOpenAI([]byte(data))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if chunk.ThinkingDelta != "think..." {
		t.Errorf("ThinkingDelta: got %q, want %q", chunk.ThinkingDelta, "think...")
	}
	if chunk.Delta != "" {
		t.Errorf("Delta should be empty when only reasoning_content is set, got %q", chunk.Delta)
	}
}

func TestChunkToOpenAI_WritesReasoningContentDelta(t *testing.T) {
	chunk := domain.Chunk{ID: "c1", Model: "m", ThinkingDelta: "thinking..."}
	out, err := translator.ChunkToOpenAI(chunk)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var body map[string]any
	json.Unmarshal(out, &body)
	delta := body["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)
	if delta["reasoning_content"] != "thinking..." {
		t.Errorf("reasoning_content: got %v", delta["reasoning_content"])
	}
	if _, ok := delta["content"]; ok {
		t.Error("content should be absent in pure thinking chunk")
	}
}

func TestRequestFromOpenAI_BudgetTokens(t *testing.T) {
	body := `{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"think"}],"budget_tokens":8000}`
	req, err := translator.RequestFromOpenAI([]byte(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if req.BudgetTokens == nil || *req.BudgetTokens != 8000 {
		t.Errorf("budget_tokens: got %v, want 8000", req.BudgetTokens)
	}
}

// --- Models capabilities ---

func TestModelsToOpenAI_IncludesCapabilities(t *testing.T) {
	models := []domain.Model{
		{ID: "m1", OwnedBy: "test", Capabilities: []string{"vision"}},
	}
	b, err := translator.ModelsToOpenAI(models)
	if err != nil {
		t.Fatalf("ModelsToOpenAI: %v", err)
	}
	var resp struct {
		Data []struct {
			ID           string   `json:"id"`
			Capabilities []string `json:"capabilities"`
		} `json:"data"`
	}
	if err := json.Unmarshal(b, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(resp.Data) != 1 || resp.Data[0].ID != "m1" {
		t.Fatalf("data: got %+v", resp.Data)
	}
	if len(resp.Data[0].Capabilities) != 1 || resp.Data[0].Capabilities[0] != "vision" {
		t.Errorf("capabilities: got %v", resp.Data[0].Capabilities)
	}
}

func TestModelsFromOpenAI_ParsesCapabilitiesField(t *testing.T) {
	body := `{"object":"list","data":[{"id":"m1","object":"model","owned_by":"test","capabilities":["vision","reasoning"]}]}`
	models, err := translator.ModelsFromOpenAI([]byte(body))
	if err != nil {
		t.Fatalf("ModelsFromOpenAI: %v", err)
	}
	if len(models) != 1 {
		t.Fatalf("models len: %d", len(models))
	}
	caps := models[0].Capabilities
	if len(caps) != 2 {
		t.Errorf("capabilities: got %v, want [vision reasoning]", caps)
	}
}

// --- ResponseFormat ---

func TestRequestFromOpenAI_ResponseFormatJSONSchema(t *testing.T) {
	body := `{
		"model": "m",
		"messages": [{"role": "user", "content": "hi"}],
		"response_format": {
			"type": "json_schema",
			"json_schema": {
				"name": "person",
				"strict": true,
				"schema": {"type": "object", "properties": {"name": {"type": "string"}}, "additionalProperties": false}
			}
		}
	}`
	req, err := translator.RequestFromOpenAI([]byte(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if req.ResponseFormat == nil {
		t.Fatal("ResponseFormat is nil, want non-nil")
	}
	if req.ResponseFormat.Name != "person" {
		t.Errorf("name: got %q, want %q", req.ResponseFormat.Name, "person")
	}
	if !req.ResponseFormat.Strict {
		t.Error("strict: got false, want true")
	}
	if req.ResponseFormat.Schema["type"] != "object" {
		t.Errorf("schema type: got %v, want object", req.ResponseFormat.Schema["type"])
	}
}

func TestRequestFromOpenAI_ResponseFormatNonSchemaIsNil(t *testing.T) {
	cases := map[string]string{
		"text":              `{"type": "text"}`,
		"json_object":       `{"type": "json_object"}`,
		"json_schema_empty": `{"type": "json_schema", "json_schema": {"name": "x"}}`,
	}
	for name, rf := range cases {
		t.Run(name, func(t *testing.T) {
			body := `{"model": "m", "messages": [{"role": "user", "content": "hi"}], "response_format": ` + rf + `}`
			req, err := translator.RequestFromOpenAI([]byte(body))
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if req.ResponseFormat != nil {
				t.Errorf("ResponseFormat: got %+v, want nil", req.ResponseFormat)
			}
		})
	}
}

func TestRequestFromOpenAI_NoResponseFormatIsNil(t *testing.T) {
	body := `{"model": "m", "messages": [{"role": "user", "content": "hi"}]}`
	req, err := translator.RequestFromOpenAI([]byte(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if req.ResponseFormat != nil {
		t.Errorf("ResponseFormat: got %+v, want nil", req.ResponseFormat)
	}
}

func TestModelsFromOpenAI_ParsesOpenRouterModality(t *testing.T) {
	body := `{"object":"list","data":[{"id":"m1","object":"model","owned_by":"openrouter","architecture":{"modality":"text+image->text"}}]}`
	models, err := translator.ModelsFromOpenAI([]byte(body))
	if err != nil {
		t.Fatalf("ModelsFromOpenAI: %v", err)
	}
	found := false
	for _, c := range models[0].Capabilities {
		if c == "vision" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected vision from modality string, got %v", models[0].Capabilities)
	}
}

func TestModelsFromOpenAI_ParsesOpenRouterInputModalities(t *testing.T) {
	body := `{"object":"list","data":[{"id":"m1","object":"model","owned_by":"openrouter","architecture":{"input_modalities":["text","image"]}}]}`
	models, err := translator.ModelsFromOpenAI([]byte(body))
	if err != nil {
		t.Fatalf("ModelsFromOpenAI: %v", err)
	}
	found := false
	for _, c := range models[0].Capabilities {
		if c == "vision" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected vision from input_modalities, got %v", models[0].Capabilities)
	}
}

func f64p(v float64) *float64 { return &v }

func TestModelsFromOpenAI_ParsesOpenRouterPricing(t *testing.T) {
	body := `{"object":"list","data":[{"id":"m1","object":"model","owned_by":"openrouter","pricing":{"prompt":"0.000005","completion":"0.000015"}}]}`
	models, err := translator.ModelsFromOpenAI([]byte(body))
	if err != nil {
		t.Fatalf("ModelsFromOpenAI: %v", err)
	}
	if models[0].InputCostPerToken == nil || *models[0].InputCostPerToken != 0.000005 {
		t.Errorf("InputCostPerToken: got %v, want 0.000005", models[0].InputCostPerToken)
	}
	if models[0].OutputCostPerToken == nil || *models[0].OutputCostPerToken != 0.000015 {
		t.Errorf("OutputCostPerToken: got %v, want 0.000015", models[0].OutputCostPerToken)
	}
}

func TestModelsFromOpenAI_ParsesDirectPricingFields(t *testing.T) {
	body := `{"object":"list","data":[{"id":"m1","object":"model","owned_by":"us","input_cost_per_token":0.000003,"output_cost_per_token":0.000009}]}`
	models, err := translator.ModelsFromOpenAI([]byte(body))
	if err != nil {
		t.Fatalf("ModelsFromOpenAI: %v", err)
	}
	if models[0].InputCostPerToken == nil || *models[0].InputCostPerToken != 0.000003 {
		t.Errorf("InputCostPerToken: got %v, want 0.000003", models[0].InputCostPerToken)
	}
	if models[0].OutputCostPerToken == nil || *models[0].OutputCostPerToken != 0.000009 {
		t.Errorf("OutputCostPerToken: got %v, want 0.000009", models[0].OutputCostPerToken)
	}
}

func TestModelsFromOpenAI_NoPricingReturnsNil(t *testing.T) {
	body := `{"object":"list","data":[{"id":"m1","object":"model","owned_by":"local"}]}`
	models, err := translator.ModelsFromOpenAI([]byte(body))
	if err != nil {
		t.Fatalf("ModelsFromOpenAI: %v", err)
	}
	if models[0].InputCostPerToken != nil {
		t.Errorf("InputCostPerToken: expected nil, got %v", *models[0].InputCostPerToken)
	}
	if models[0].OutputCostPerToken != nil {
		t.Errorf("OutputCostPerToken: expected nil, got %v", *models[0].OutputCostPerToken)
	}
}

func TestModelsToOpenAI_IncludesPricingWhenFinite(t *testing.T) {
	models := []domain.Model{
		{ID: "m1", OwnedBy: "test", InputCostPerToken: f64p(0.000005), OutputCostPerToken: f64p(0.000015)},
	}
	b, err := translator.ModelsToOpenAI(models)
	if err != nil {
		t.Fatalf("ModelsToOpenAI: %v", err)
	}
	var resp struct {
		Data []struct {
			InputCostPerToken  float64 `json:"input_cost_per_token"`
			OutputCostPerToken float64 `json:"output_cost_per_token"`
		} `json:"data"`
	}
	if err := json.Unmarshal(b, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if resp.Data[0].InputCostPerToken != 0.000005 {
		t.Errorf("input_cost_per_token: got %v", resp.Data[0].InputCostPerToken)
	}
	if resp.Data[0].OutputCostPerToken != 0.000015 {
		t.Errorf("output_cost_per_token: got %v", resp.Data[0].OutputCostPerToken)
	}
}

func TestModelsToOpenAI_OmitsPricingWhenInfinite(t *testing.T) {
	inf := math.Inf(1)
	models := []domain.Model{
		{ID: "m1", OwnedBy: "test", InputCostPerToken: &inf, OutputCostPerToken: &inf},
	}
	b, err := translator.ModelsToOpenAI(models)
	if err != nil {
		t.Fatalf("ModelsToOpenAI: %v", err)
	}
	var resp struct {
		Data []map[string]any `json:"data"`
	}
	if err := json.Unmarshal(b, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, ok := resp.Data[0]["input_cost_per_token"]; ok {
		t.Error("input_cost_per_token should be absent when +Inf")
	}
	if _, ok := resp.Data[0]["output_cost_per_token"]; ok {
		t.Error("output_cost_per_token should be absent when +Inf")
	}
}

func TestModelsToOpenAI_OmitsPricingWhenNil(t *testing.T) {
	models := []domain.Model{
		{ID: "m1", OwnedBy: "test"}, // no pricing set
	}
	b, err := translator.ModelsToOpenAI(models)
	if err != nil {
		t.Fatalf("ModelsToOpenAI: %v", err)
	}
	var resp struct {
		Data []map[string]any `json:"data"`
	}
	if err := json.Unmarshal(b, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, ok := resp.Data[0]["input_cost_per_token"]; ok {
		t.Error("input_cost_per_token should be absent when nil")
	}
}

// --- FinishReason propagation ---

func TestResponseFromOpenAI_CapturesFinishReason(t *testing.T) {
	body := []byte(`{
		"id":"r","model":"m",
		"choices":[{"index":0,"message":{"role":"assistant","content":"x"},"finish_reason":"length"}],
		"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
	}`)
	resp, err := translator.ResponseFromOpenAI(body)
	if err != nil {
		t.Fatalf("ResponseFromOpenAI: %v", err)
	}
	if resp.FinishReason != "length" {
		t.Errorf("FinishReason: got %q, want %q", resp.FinishReason, "length")
	}
}

func TestResponseFromOpenAI_DefaultFinishReason(t *testing.T) {
	body := []byte(`{
		"id":"r","model":"m",
		"choices":[{"index":0,"message":{"role":"assistant","content":"x"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
	}`)
	resp, err := translator.ResponseFromOpenAI(body)
	if err != nil {
		t.Fatalf("ResponseFromOpenAI: %v", err)
	}
	if resp.FinishReason != "stop" {
		t.Errorf("FinishReason: got %q, want %q", resp.FinishReason, "stop")
	}
}

func TestResponseToOpenAI_EmitsRealFinishReason(t *testing.T) {
	resp := domain.Response{
		ID: "r", Model: "m",
		Message:      domain.Message{Role: "assistant", Content: ""},
		FinishReason: "length",
		Usage:        domain.Usage{PromptTokens: 100, CompletionTokens: 50},
	}
	out, err := translator.ResponseToOpenAI(resp)
	if err != nil {
		t.Fatalf("ResponseToOpenAI: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	choices, _ := parsed["choices"].([]any)
	choice0, _ := choices[0].(map[string]any)
	if choice0["finish_reason"] != "length" {
		t.Errorf("finish_reason: got %v, want length", choice0["finish_reason"])
	}
}

func TestResponseToOpenAI_DefaultsToStopWhenEmpty(t *testing.T) {
	// Callers that don't set FinishReason should still get "stop" (back-compat).
	resp := domain.Response{
		ID: "r", Model: "m",
		Message: domain.Message{Role: "assistant", Content: "hi"},
		Usage:   domain.Usage{PromptTokens: 1, CompletionTokens: 1},
	}
	out, err := translator.ResponseToOpenAI(resp)
	if err != nil {
		t.Fatalf("ResponseToOpenAI: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	choices, _ := parsed["choices"].([]any)
	choice0, _ := choices[0].(map[string]any)
	if choice0["finish_reason"] != "stop" {
		t.Errorf("finish_reason: got %v, want stop", choice0["finish_reason"])
	}
}

func TestChunkFromOpenAI_CapturesFinishReason(t *testing.T) {
	data := []byte(`{
		"id":"c","model":"m",
		"choices":[{"index":0,"delta":{},"finish_reason":"length"}]
	}`)
	chunk, err := translator.ChunkFromOpenAI(data)
	if err != nil {
		t.Fatalf("ChunkFromOpenAI: %v", err)
	}
	if chunk.FinishReason != "length" {
		t.Errorf("FinishReason: got %q, want %q", chunk.FinishReason, "length")
	}
	if chunk.Done {
		t.Errorf("Done: got true, want false (Done is only for finish_reason='stop')")
	}
}

func TestChunkToOpenAI_EmitsFinishReasonWhenSet(t *testing.T) {
	chunk := domain.Chunk{ID: "c", Model: "m", FinishReason: "length"}
	out, err := translator.ChunkToOpenAI(chunk)
	if err != nil {
		t.Fatalf("ChunkToOpenAI: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	choices, _ := parsed["choices"].([]any)
	choice0, _ := choices[0].(map[string]any)
	if choice0["finish_reason"] != "length" {
		t.Errorf("finish_reason: got %v, want length", choice0["finish_reason"])
	}
}

func TestResponseFromOpenAI_ReadsCachedTokens(t *testing.T) {
	body := []byte(`{
		"id":"r1","model":"m",
		"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":100,"completion_tokens":5,"total_tokens":105,
		         "prompt_tokens_details":{"cached_tokens":80}}
	}`)
	resp, err := translator.ResponseFromOpenAI(body)
	if err != nil {
		t.Fatalf("ResponseFromOpenAI: %v", err)
	}
	if resp.Usage.CachedTokens != 80 {
		t.Errorf("CachedTokens: got %d, want 80", resp.Usage.CachedTokens)
	}
	if resp.Usage.PromptTokens != 100 {
		t.Errorf("PromptTokens: got %d, want 100", resp.Usage.PromptTokens)
	}
}

func TestResponseFromOpenAI_NoCachedTokensWhenAbsent(t *testing.T) {
	body := []byte(`{
		"id":"r2","model":"m",
		"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}
	}`)
	resp, err := translator.ResponseFromOpenAI(body)
	if err != nil {
		t.Fatalf("ResponseFromOpenAI: %v", err)
	}
	if resp.Usage.CachedTokens != 0 {
		t.Errorf("CachedTokens: got %d, want 0", resp.Usage.CachedTokens)
	}
}

func TestResponseToOpenAI_EmitsCachedTokensWhenNonZero(t *testing.T) {
	resp := domain.Response{
		ID: "r", Model: "m",
		Message: domain.Message{Role: "assistant", Content: "hi"},
		Usage:   domain.Usage{PromptTokens: 50, CompletionTokens: 3, CachedTokens: 40},
	}
	out, err := translator.ResponseToOpenAI(resp)
	if err != nil {
		t.Fatalf("ResponseToOpenAI: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("unmarshal output: %v", err)
	}
	usage, _ := parsed["usage"].(map[string]any)
	details, ok := usage["prompt_tokens_details"].(map[string]any)
	if !ok {
		t.Fatalf("usage.prompt_tokens_details missing: %v", usage)
	}
	if got := details["cached_tokens"]; got != float64(40) {
		t.Errorf("cached_tokens: got %v, want 40", got)
	}
}

func TestResponseToOpenAI_OmitsCachedTokensWhenZero(t *testing.T) {
	resp := domain.Response{
		ID: "r", Model: "m",
		Message: domain.Message{Role: "assistant", Content: "hi"},
		Usage:   domain.Usage{PromptTokens: 10, CompletionTokens: 2, CachedTokens: 0},
	}
	out, err := translator.ResponseToOpenAI(resp)
	if err != nil {
		t.Fatalf("ResponseToOpenAI: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("unmarshal output: %v", err)
	}
	usage, _ := parsed["usage"].(map[string]any)
	if _, has := usage["prompt_tokens_details"]; has {
		t.Errorf("prompt_tokens_details should be omitted when cached_tokens is 0; got %v", usage)
	}
}

func TestRequestFromOpenAI_ParsesTopLevelReasoningEffort(t *testing.T) {
	body := []byte(`{
		"model":"m","messages":[{"role":"user","content":"hi"}],
		"reasoning_effort":"medium"
	}`)
	req, err := translator.RequestFromOpenAI(body)
	if err != nil {
		t.Fatalf("RequestFromOpenAI: %v", err)
	}
	if req.ReasoningEffort == nil {
		t.Fatalf("ReasoningEffort: got nil, want *medium")
	}
	if *req.ReasoningEffort != "medium" {
		t.Errorf("ReasoningEffort: got %q, want %q", *req.ReasoningEffort, "medium")
	}
}

func TestRequestFromOpenAI_ParsesNestedReasoningEffort(t *testing.T) {
	body := []byte(`{
		"model":"m","messages":[{"role":"user","content":"hi"}],
		"reasoning":{"effort":"high"}
	}`)
	req, err := translator.RequestFromOpenAI(body)
	if err != nil {
		t.Fatalf("RequestFromOpenAI: %v", err)
	}
	if req.ReasoningEffort == nil || *req.ReasoningEffort != "high" {
		got := "nil"
		if req.ReasoningEffort != nil {
			got = *req.ReasoningEffort
		}
		t.Errorf("ReasoningEffort: got %q, want %q", got, "high")
	}
}

func TestRequestFromOpenAI_TopLevelEffortWinsOverNested(t *testing.T) {
	body := []byte(`{
		"model":"m","messages":[{"role":"user","content":"hi"}],
		"reasoning_effort":"low",
		"reasoning":{"effort":"high"}
	}`)
	req, err := translator.RequestFromOpenAI(body)
	if err != nil {
		t.Fatalf("RequestFromOpenAI: %v", err)
	}
	if req.ReasoningEffort == nil || *req.ReasoningEffort != "low" {
		got := "nil"
		if req.ReasoningEffort != nil {
			got = *req.ReasoningEffort
		}
		t.Errorf("ReasoningEffort: got %q, want %q", got, "low")
	}
}

func TestRequestFromOpenAI_NoReasoningEffort(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`)
	req, err := translator.RequestFromOpenAI(body)
	if err != nil {
		t.Fatalf("RequestFromOpenAI: %v", err)
	}
	if req.ReasoningEffort != nil {
		t.Errorf("ReasoningEffort: got %q, want nil", *req.ReasoningEffort)
	}
}

func TestResponseFromOpenAI_ReadsVLLMReasoningField(t *testing.T) {
	body := []byte(`{
		"id":"r","model":"m",
		"choices":[{"index":0,"message":{
			"role":"assistant","content":"final answer",
			"reasoning":"let me think..."
		},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
	}`)
	resp, err := translator.ResponseFromOpenAI(body)
	if err != nil {
		t.Fatalf("ResponseFromOpenAI: %v", err)
	}
	if resp.Thinking != "let me think..." {
		t.Errorf("Thinking: got %q, want %q", resp.Thinking, "let me think...")
	}
	if resp.Message.Content != "final answer" {
		t.Errorf("Content: got %q, want %q", resp.Message.Content, "final answer")
	}
}

func TestResponseFromOpenAI_ReasoningContentWinsOverReasoning(t *testing.T) {
	body := []byte(`{
		"id":"r","model":"m",
		"choices":[{"index":0,"message":{
			"role":"assistant","content":"ans",
			"reasoning_content":"explicit",
			"reasoning":"alternative"
		},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
	}`)
	resp, err := translator.ResponseFromOpenAI(body)
	if err != nil {
		t.Fatalf("ResponseFromOpenAI: %v", err)
	}
	if resp.Thinking != "explicit" {
		t.Errorf("Thinking: got %q, want %q (reasoning_content should win)", resp.Thinking, "explicit")
	}
}

func TestChunkFromOpenAI_ReadsVLLMReasoningDelta(t *testing.T) {
	data := []byte(`{
		"id":"c","model":"m",
		"choices":[{"index":0,"delta":{"reasoning":"think delta"}}]
	}`)
	chunk, err := translator.ChunkFromOpenAI(data)
	if err != nil {
		t.Fatalf("ChunkFromOpenAI: %v", err)
	}
	if chunk.ThinkingDelta != "think delta" {
		t.Errorf("ThinkingDelta: got %q, want %q", chunk.ThinkingDelta, "think delta")
	}
}

func TestChunkFromOpenAI_ReasoningContentWinsOverReasoning(t *testing.T) {
	data := []byte(`{
		"id":"c","model":"m",
		"choices":[{"index":0,"delta":{"reasoning_content":"explicit","reasoning":"alternative"}}]
	}`)
	chunk, err := translator.ChunkFromOpenAI(data)
	if err != nil {
		t.Fatalf("ChunkFromOpenAI: %v", err)
	}
	if chunk.ThinkingDelta != "explicit" {
		t.Errorf("ThinkingDelta: got %q, want %q (reasoning_content should win)", chunk.ThinkingDelta, "explicit")
	}
}

func TestChunkFromOpenAI_ParsesUsageOnTerminalChunk(t *testing.T) {
	data := []byte(`{
		"id":"c","model":"m",
		"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":42,"completion_tokens":7,"total_tokens":49,
		         "prompt_tokens_details":{"cached_tokens":30}}
	}`)
	chunk, err := translator.ChunkFromOpenAI(data)
	if err != nil {
		t.Fatalf("ChunkFromOpenAI: %v", err)
	}
	if chunk.Usage == nil {
		t.Fatalf("Usage: got nil, want populated")
	}
	if chunk.Usage.PromptTokens != 42 || chunk.Usage.CompletionTokens != 7 || chunk.Usage.CachedTokens != 30 {
		t.Errorf("Usage: got %+v, want {42 7 30}", *chunk.Usage)
	}
	if !chunk.Done {
		t.Errorf("Done: got false, want true")
	}
}

func TestChunkFromOpenAI_NilUsageWhenAbsent(t *testing.T) {
	data := []byte(`{"id":"c","model":"m","choices":[{"index":0,"delta":{"content":"hi"}}]}`)
	chunk, err := translator.ChunkFromOpenAI(data)
	if err != nil {
		t.Fatalf("ChunkFromOpenAI: %v", err)
	}
	if chunk.Usage != nil {
		t.Errorf("Usage: got non-nil, want nil")
	}
}

func TestChunkToOpenAI_EmitsUsageWhenPresent(t *testing.T) {
	chunk := domain.Chunk{
		ID:    "c",
		Model: "m",
		Done:  true,
		Usage: &domain.Usage{PromptTokens: 50, CompletionTokens: 3, CachedTokens: 40},
	}
	out, err := translator.ChunkToOpenAI(chunk)
	if err != nil {
		t.Fatalf("ChunkToOpenAI: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	usage, ok := parsed["usage"].(map[string]any)
	if !ok {
		t.Fatalf("usage missing: %v", parsed)
	}
	if usage["prompt_tokens"] != float64(50) {
		t.Errorf("prompt_tokens: got %v", usage["prompt_tokens"])
	}
	details, _ := usage["prompt_tokens_details"].(map[string]any)
	if details == nil || details["cached_tokens"] != float64(40) {
		t.Errorf("cached_tokens: got %v", details)
	}
}

func TestChunkToOpenAI_OmitsUsageWhenNil(t *testing.T) {
	chunk := domain.Chunk{ID: "c", Model: "m", Delta: "hi"}
	out, err := translator.ChunkToOpenAI(chunk)
	if err != nil {
		t.Fatalf("ChunkToOpenAI: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, has := parsed["usage"]; has {
		t.Errorf("usage should be omitted when nil; got: %v", parsed)
	}
}

func TestModelsFromOpenAI_ParsesMaxModelLen(t *testing.T) {
	body := []byte(`{"object":"list","data":[
		{"id":"gemma-4-31b-it","object":"model","owned_by":"vllm","max_model_len":229376},
		{"id":"other","object":"model","owned_by":"vllm"}
	]}`)
	models, err := translator.ModelsFromOpenAI(body)
	if err != nil {
		t.Fatalf("ModelsFromOpenAI: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("got %d models, want 2", len(models))
	}
	if models[0].MaxModelLen != 229376 {
		t.Errorf("[0].MaxModelLen: got %d, want 229376", models[0].MaxModelLen)
	}
	if models[1].MaxModelLen != 0 {
		t.Errorf("[1].MaxModelLen: got %d, want 0 (absent in upstream)", models[1].MaxModelLen)
	}
}

func TestModelsToOpenAI_EmitsMaxModelLenWhenNonZero(t *testing.T) {
	models := []domain.Model{
		{ID: "with-len", OwnedBy: "vllm", MaxModelLen: 229376},
		{ID: "without-len", OwnedBy: "vllm"},
	}
	out, err := translator.ModelsToOpenAI(models)
	if err != nil {
		t.Fatalf("ModelsToOpenAI: %v", err)
	}
	var parsed struct {
		Data []map[string]any `json:"data"`
	}
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got := parsed.Data[0]["max_model_len"]; got != float64(229376) {
		t.Errorf("[0].max_model_len: got %v, want 229376", got)
	}
	if _, has := parsed.Data[1]["max_model_len"]; has {
		t.Errorf("[1].max_model_len should be omitted when 0; got: %v", parsed.Data[1])
	}
}
