// Package vllm implements a Provider for vLLM-served OpenAI-compatible upstreams.
//
// vLLM is OpenAI-compatible in shape but differs in three points addressed by
// this package:
//   - reasoning content arrives in a "reasoning" field (handled by translator)
//   - thinking mode is toggled via chat_template_kwargs.<key> (handled here)
//   - /v1/models reports no capability info (handled by capability validation)
package vllm

import (
	"encoding/json"
	"fmt"
)

// injectChatTemplateKwargs sets body["chat_template_kwargs"][key] = enabled,
// preserving any existing keys in chat_template_kwargs. On JSON decode failure
// the original body is not returned; an error is propagated instead.
func injectChatTemplateKwargs(body []byte, key string, enabled bool) ([]byte, error) {
	var root map[string]any
	if err := json.Unmarshal(body, &root); err != nil {
		return nil, fmt.Errorf("decode body: %w", err)
	}
	if root == nil {
		root = map[string]any{}
	}
	ctk, _ := root["chat_template_kwargs"].(map[string]any)
	if ctk == nil {
		ctk = map[string]any{}
	}
	ctk[key] = enabled
	root["chat_template_kwargs"] = ctk
	return json.Marshal(root)
}
