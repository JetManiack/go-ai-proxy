package vllm

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestInjectChatTemplateKwargs_EmptyObject(t *testing.T) {
	got, err := injectChatTemplateKwargs([]byte(`{}`), "enable_thinking", true)
	if err != nil {
		t.Fatalf("inject: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(got, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	ck, ok := parsed["chat_template_kwargs"].(map[string]any)
	if !ok {
		t.Fatalf("chat_template_kwargs missing or wrong type: %v", parsed)
	}
	if ck["enable_thinking"] != true {
		t.Errorf("enable_thinking: got %v, want true", ck["enable_thinking"])
	}
}

func TestInjectChatTemplateKwargs_PreservesOtherFields(t *testing.T) {
	got, err := injectChatTemplateKwargs(
		[]byte(`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`),
		"enable_thinking", true)
	if err != nil {
		t.Fatalf("inject: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(got, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if parsed["model"] != "m" {
		t.Errorf("model: got %v, want m", parsed["model"])
	}
	if parsed["stream"] != true {
		t.Errorf("stream: got %v, want true", parsed["stream"])
	}
	if _, ok := parsed["messages"]; !ok {
		t.Errorf("messages array dropped: %v", parsed)
	}
}

func TestInjectChatTemplateKwargs_MergesIntoExisting(t *testing.T) {
	got, err := injectChatTemplateKwargs(
		[]byte(`{"chat_template_kwargs":{"other_key":"keep"}}`),
		"enable_thinking", true)
	if err != nil {
		t.Fatalf("inject: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(got, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	ck, _ := parsed["chat_template_kwargs"].(map[string]any)
	if ck["other_key"] != "keep" {
		t.Errorf("other_key dropped: got %v, want keep", ck["other_key"])
	}
	if ck["enable_thinking"] != true {
		t.Errorf("enable_thinking: got %v, want true", ck["enable_thinking"])
	}
}

func TestInjectChatTemplateKwargs_OverwritesSameKey(t *testing.T) {
	got, err := injectChatTemplateKwargs(
		[]byte(`{"chat_template_kwargs":{"enable_thinking":false}}`),
		"enable_thinking", true)
	if err != nil {
		t.Fatalf("inject: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(got, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	ck, _ := parsed["chat_template_kwargs"].(map[string]any)
	if ck["enable_thinking"] != true {
		t.Errorf("enable_thinking: got %v, want true (our value should win)", ck["enable_thinking"])
	}
}

func TestInjectChatTemplateKwargs_FalseSerialisesCorrectly(t *testing.T) {
	got, err := injectChatTemplateKwargs([]byte(`{}`), "enable_thinking", false)
	if err != nil {
		t.Fatalf("inject: %v", err)
	}
	if !strings.Contains(string(got), `"enable_thinking":false`) {
		t.Errorf("output should contain explicit false: %s", got)
	}
}

func TestInjectChatTemplateKwargs_InvalidJSON(t *testing.T) {
	_, err := injectChatTemplateKwargs([]byte(`{not json`), "enable_thinking", true)
	if err == nil {
		t.Fatalf("expected error for invalid JSON, got nil")
	}
}

func TestInjectChatTemplateKwargs_NullInput(t *testing.T) {
	got, err := injectChatTemplateKwargs([]byte(`null`), "enable_thinking", true)
	if err != nil {
		t.Fatalf("inject: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(got, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	ck, ok := parsed["chat_template_kwargs"].(map[string]any)
	if !ok {
		t.Fatalf("chat_template_kwargs missing: %v", parsed)
	}
	if ck["enable_thinking"] != true {
		t.Errorf("enable_thinking: got %v, want true", ck["enable_thinking"])
	}
}
