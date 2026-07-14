// Package translator converts between the OpenAI wire format and canonical domain types.
package translator

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
)

// --- OpenAI wire format types (unexported) ---

type oaRequest struct {
	Model           string       `json:"model"`
	Messages        []oaMessage  `json:"messages"`
	Temperature     *float64     `json:"temperature,omitempty"`
	MaxTokens       *int         `json:"max_tokens,omitempty"`
	Stream          bool         `json:"stream,omitempty"`
	Tools           []oaTool     `json:"tools,omitempty"`
	BudgetTokens    *int         `json:"budget_tokens,omitempty"`
	ReasoningEffort *string      `json:"reasoning_effort,omitempty"`
	Reasoning       *oaReasoning `json:"reasoning,omitempty"`
}

type oaReasoning struct {
	Effort *string `json:"effort,omitempty"`
}

type oaMessage struct {
	Role             string          `json:"role"`
	Content          json.RawMessage `json:"content,omitempty"` // string, array, or null
	ReasoningContent string          `json:"reasoning_content,omitempty"`
	Reasoning        string          `json:"reasoning,omitempty"` // vLLM-style alias for reasoning_content
	ToolCalls        []oaToolCall    `json:"tool_calls,omitempty"`
	ToolCallID       string          `json:"tool_call_id,omitempty"`
}

type oaTool struct {
	Type     string         `json:"type"`
	Function oaToolFunction `json:"function"`
}

type oaToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type oaToolCall struct {
	Index    *int               `json:"index,omitempty"` // present in streaming deltas only
	ID       string             `json:"id,omitempty"`
	Type     string             `json:"type,omitempty"`
	Function oaToolCallFunction `json:"function"`
}

type oaToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type oaResponse struct {
	ID      string     `json:"id"`
	Object  string     `json:"object"`
	Created int64      `json:"created"`
	Model   string     `json:"model"`
	Choices []oaChoice `json:"choices"`
	Usage   oaUsage    `json:"usage"`
}

type oaChoice struct {
	Index        int       `json:"index"`
	Message      oaMessage `json:"message"`
	FinishReason string    `json:"finish_reason"`
}

type oaUsage struct {
	PromptTokens        int                    `json:"prompt_tokens"`
	CompletionTokens    int                    `json:"completion_tokens"`
	TotalTokens         int                    `json:"total_tokens"`
	PromptTokensDetails *oaPromptTokensDetails `json:"prompt_tokens_details,omitempty"`
}

type oaPromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type oaChunk struct {
	ID      string    `json:"id"`
	Object  string    `json:"object"`
	Created int64     `json:"created"`
	Model   string    `json:"model"`
	Choices []oaDelta `json:"choices"`
	Usage   *oaUsage  `json:"usage,omitempty"`
}

type oaDelta struct {
	Index        int        `json:"index"`
	Delta        oaDeltaMsg `json:"delta"`
	FinishReason *string    `json:"finish_reason"`
}

type oaDeltaMsg struct {
	Role             string       `json:"role,omitempty"`
	Content          string       `json:"content,omitempty"`
	ReasoningContent string       `json:"reasoning_content,omitempty"`
	Reasoning        string       `json:"reasoning,omitempty"` // vLLM-style alias
	ToolCalls        []oaToolCall `json:"tool_calls,omitempty"`
}

type oaModelsResponse struct {
	Object string    `json:"object"`
	Data   []oaModel `json:"data"`
}

type oaModel struct {
	ID                 string               `json:"id"`
	Object             string               `json:"object"`
	Created            int64                `json:"created"`
	OwnedBy            string               `json:"owned_by"`
	Architecture       *oaModelArchitecture `json:"architecture,omitempty"`
	Capabilities       []string             `json:"capabilities,omitempty"`
	Pricing            *oaModelPricing      `json:"pricing,omitempty"`              // OpenRouter input format
	InputCostPerToken  *float64             `json:"input_cost_per_token,omitempty"` // our own format (in/out)
	OutputCostPerToken *float64             `json:"output_cost_per_token,omitempty"`
	MaxModelLen        int                  `json:"max_model_len,omitempty"`
}

type oaModelArchitecture struct {
	Modality        string   `json:"modality,omitempty"`
	InputModalities []string `json:"input_modalities,omitempty"`
}

// oaModelPricing mirrors the OpenRouter "pricing" field (string USD values per token).
type oaModelPricing struct {
	Prompt     string `json:"prompt,omitempty"`
	Completion string `json:"completion,omitempty"`
}

// --- Helpers ---

// extractThinkTags pulls all <think>…</think> blocks out of content.
// Returns the concatenated thinking text and the remaining content (trimmed).
// Unclosed <think> at the end is treated as all-thinking.
func extractThinkTags(content string) (thinking, remaining string) {
	var thinkParts, otherParts []string
	for {
		start := strings.Index(content, "<think>")
		if start < 0 {
			otherParts = append(otherParts, content)
			break
		}
		otherParts = append(otherParts, content[:start])
		content = content[start+len("<think>"):]
		end := strings.Index(content, "</think>")
		if end < 0 {
			thinkParts = append(thinkParts, content) // unclosed — rest is thinking
			content = ""
			break
		}
		thinkParts = append(thinkParts, content[:end])
		content = content[end+len("</think>"):]
	}
	return strings.Join(thinkParts, ""), strings.TrimSpace(strings.Join(otherParts, ""))
}

// firstNonEmpty returns the first non-empty string from the list, "" if all are empty.
func firstNonEmpty(s ...string) string {
	for _, v := range s {
		if v != "" {
			return v
		}
	}
	return ""
}

// finishReasonOrDefault returns reason when non-empty, else fallback.
// Preserves backward compatibility with callers that don't yet set FinishReason.
func finishReasonOrDefault(reason, fallback string) string {
	if reason != "" {
		return reason
	}
	return fallback
}

// rawToString extracts a string value from a json.RawMessage.
// Returns empty string if the value is null, empty, or not a string.
func rawToString(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	return ""
}

func stringToRaw(s string) json.RawMessage {
	b, _ := json.Marshal(s)
	return b
}

// --- Request ---

// RequestFromOpenAI parses an OpenAI-format request body into a canonical Request.
func RequestFromOpenAI(body []byte) (domain.Request, error) {
	var oar oaRequest
	if err := json.Unmarshal(body, &oar); err != nil {
		return domain.Request{}, fmt.Errorf("parse request: %w", err)
	}

	req := domain.Request{
		Model:        oar.Model,
		Temperature:  oar.Temperature,
		MaxTokens:    oar.MaxTokens,
		Stream:       oar.Stream,
		BudgetTokens: oar.BudgetTokens,
	}

	switch {
	case oar.ReasoningEffort != nil:
		req.ReasoningEffort = oar.ReasoningEffort
	case oar.Reasoning != nil && oar.Reasoning.Effort != nil:
		req.ReasoningEffort = oar.Reasoning.Effort
	}

	for _, m := range oar.Messages {
		text := rawToString(m.Content)
		msg := domain.Message{
			Role:       m.Role,
			Content:    text,
			ToolCallID: m.ToolCallID,
		}
		if text == "" && len(m.Content) > 0 {
			msg.RawContent = m.Content
		}
		for _, tc := range m.ToolCalls {
			msg.ToolCalls = append(msg.ToolCalls, domain.ToolCall{
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			})
		}
		req.Messages = append(req.Messages, msg)
	}

	for _, t := range oar.Tools {
		req.Tools = append(req.Tools, domain.Tool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			Parameters:  t.Function.Parameters,
		})
	}

	return req, nil
}

// RequestToOpenAI converts a canonical Request into OpenAI-format JSON.
func RequestToOpenAI(req domain.Request) ([]byte, error) {
	oar := oaRequest{
		Model:       req.Model,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		Stream:      req.Stream,
	}

	for _, m := range req.Messages {
		content := stringToRaw(m.Content)
		if len(m.RawContent) > 0 {
			content = m.RawContent
		}
		om := oaMessage{
			Role:       m.Role,
			Content:    content,
			ToolCallID: m.ToolCallID,
		}
		for _, tc := range m.ToolCalls {
			om.ToolCalls = append(om.ToolCalls, oaToolCall{
				ID:   tc.ID,
				Type: "function",
				Function: oaToolCallFunction{
					Name:      tc.Name,
					Arguments: tc.Arguments,
				},
			})
		}
		oar.Messages = append(oar.Messages, om)
	}

	for _, t := range req.Tools {
		oar.Tools = append(oar.Tools, oaTool{
			Type: "function",
			Function: oaToolFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
			},
		})
	}

	return json.Marshal(oar)
}

// --- Response ---

// ResponseToOpenAI converts a canonical Response to OpenAI-format JSON.
func ResponseToOpenAI(resp domain.Response) ([]byte, error) {
	om := oaMessage{
		Role:             resp.Message.Role,
		Content:          stringToRaw(resp.Message.Content),
		ReasoningContent: resp.Thinking,
	}
	for _, tc := range resp.Message.ToolCalls {
		om.ToolCalls = append(om.ToolCalls, oaToolCall{
			ID:   tc.ID,
			Type: "function",
			Function: oaToolCallFunction{
				Name:      tc.Name,
				Arguments: tc.Arguments,
			},
		})
	}

	oar := oaResponse{
		ID:      resp.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   resp.Model,
		Choices: []oaChoice{
			{
				Index:        0,
				Message:      om,
				FinishReason: finishReasonOrDefault(resp.FinishReason, "stop"),
			},
		},
		Usage: usageToOpenAI(resp.Usage),
	}

	return json.Marshal(oar)
}

// usageToOpenAI converts canonical usage to wire format, emitting prompt_tokens_details only when cached > 0.
func usageToOpenAI(u domain.Usage) oaUsage {
	out := oaUsage{
		PromptTokens:     u.PromptTokens,
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.PromptTokens + u.CompletionTokens,
	}
	if u.CachedTokens > 0 {
		out.PromptTokensDetails = &oaPromptTokensDetails{CachedTokens: u.CachedTokens}
	}
	return out
}

// ResponseFromOpenAI parses an OpenAI-format response body into a canonical Response.
func ResponseFromOpenAI(body []byte) (domain.Response, error) {
	var oar oaResponse
	if err := json.Unmarshal(body, &oar); err != nil {
		return domain.Response{}, fmt.Errorf("parse response: %w", err)
	}
	if len(oar.Choices) == 0 {
		return domain.Response{}, fmt.Errorf("response has no choices")
	}

	m := oar.Choices[0].Message
	content := rawToString(m.Content)

	// Extract thinking: prefer explicit reasoning_content field (DeepSeek API style),
	// fall back to reasoning field (vLLM-style alias), then <think>…</think> tags
	// embedded in content (local R1/QwQ style).
	thinking := m.ReasoningContent
	if thinking == "" {
		thinking = m.Reasoning
	}
	if thinking == "" && strings.Contains(content, "<think>") {
		thinking, content = extractThinkTags(content)
	}

	msg := domain.Message{
		Role:       m.Role,
		Content:    content,
		ToolCallID: m.ToolCallID,
	}
	for _, tc := range m.ToolCalls {
		msg.ToolCalls = append(msg.ToolCalls, domain.ToolCall{
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Arguments: tc.Function.Arguments,
		})
	}

	return domain.Response{
		ID:           oar.ID,
		Model:        oar.Model,
		Message:      msg,
		Thinking:     thinking,
		Usage:        usageFromOpenAI(oar.Usage),
		FinishReason: oar.Choices[0].FinishReason,
	}, nil
}

// usageFromOpenAI converts wire usage to canonical, honouring optional prompt_tokens_details.
func usageFromOpenAI(u oaUsage) domain.Usage {
	out := domain.Usage{
		PromptTokens:     u.PromptTokens,
		CompletionTokens: u.CompletionTokens,
	}
	if u.PromptTokensDetails != nil {
		out.CachedTokens = u.PromptTokensDetails.CachedTokens
	}
	return out
}

// --- Streaming ---

// ChunkToOpenAI converts a canonical Chunk to OpenAI SSE data payload JSON.
func ChunkToOpenAI(chunk domain.Chunk) ([]byte, error) {
	var finishReason *string
	switch {
	case chunk.FinishReason != "":
		fr := chunk.FinishReason
		finishReason = &fr
	case chunk.Done:
		s := "stop"
		finishReason = &s
	}

	var delta oaDeltaMsg
	if chunk.ToolCallIndex != nil {
		tc := oaToolCall{
			Index: chunk.ToolCallIndex,
			Function: oaToolCallFunction{
				Name:      chunk.ToolCallName,
				Arguments: chunk.ToolCallArgs,
			},
		}
		if chunk.ToolCallID != "" {
			tc.ID = chunk.ToolCallID
			tc.Type = "function"
		}
		delta.ToolCalls = []oaToolCall{tc}
	} else {
		delta.Content = chunk.Delta
		delta.ReasoningContent = chunk.ThinkingDelta
	}

	oac := oaChunk{
		ID:      chunk.ID,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   chunk.Model,
		Choices: []oaDelta{
			{
				Index:        0,
				Delta:        delta,
				FinishReason: finishReason,
			},
		},
	}
	if chunk.Usage != nil {
		u := usageToOpenAI(*chunk.Usage)
		oac.Usage = &u
	}

	return json.Marshal(oac)
}

// ChunkFromOpenAI parses an OpenAI SSE data payload (the part after "data: ")
// into a canonical Chunk.
func ChunkFromOpenAI(data []byte) (domain.Chunk, error) {
	var oac oaChunk
	if err := json.Unmarshal(data, &oac); err != nil {
		return domain.Chunk{}, fmt.Errorf("parse chunk: %w", err)
	}
	chunk := domain.Chunk{ID: oac.ID, Model: oac.Model}
	if oac.Usage != nil {
		u := usageFromOpenAI(*oac.Usage)
		chunk.Usage = &u
	}
	if len(oac.Choices) == 0 {
		return chunk, nil
	}
	choice := oac.Choices[0]
	chunk.Delta = choice.Delta.Content
	chunk.ThinkingDelta = firstNonEmpty(choice.Delta.ReasoningContent, choice.Delta.Reasoning)
	if choice.FinishReason != nil {
		chunk.FinishReason = *choice.FinishReason
		chunk.Done = *choice.FinishReason == "stop"
	}
	return chunk, nil
}

// --- Models ---

// ModelsToOpenAI converts a list of canonical Models to OpenAI-format JSON.
func ModelsToOpenAI(models []domain.Model) ([]byte, error) {
	resp := oaModelsResponse{Object: "list"}
	for _, m := range models {
		om := oaModel{
			ID:           m.ID,
			Object:       "model",
			OwnedBy:      m.OwnedBy,
			Capabilities: m.Capabilities,
			MaxModelLen:  m.MaxModelLen,
		}
		if m.InputCostPerToken != nil && !math.IsInf(*m.InputCostPerToken, 0) {
			om.InputCostPerToken = m.InputCostPerToken
		}
		if m.OutputCostPerToken != nil && !math.IsInf(*m.OutputCostPerToken, 0) {
			om.OutputCostPerToken = m.OutputCostPerToken
		}
		resp.Data = append(resp.Data, om)
	}
	return json.Marshal(resp)
}

// ModelsFromOpenAI parses an OpenAI-format models response into canonical Models.
// Capabilities are read from our own "capabilities" field or from OpenRouter-style
// "architecture.modality" / "architecture.input_modalities" fields.
func ModelsFromOpenAI(body []byte) ([]domain.Model, error) {
	var resp oaModelsResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("parse models: %w", err)
	}
	models := make([]domain.Model, 0, len(resp.Data))
	for _, m := range resp.Data {
		inCost, outCost := modelPricing(m)
		models = append(models, domain.Model{
			ID:                 m.ID,
			OwnedBy:            m.OwnedBy,
			Capabilities:       modelCapabilities(m),
			InputCostPerToken:  inCost,
			OutputCostPerToken: outCost, // nil when provider doesn't report pricing
			MaxModelLen:        m.MaxModelLen,
		})
	}
	return models, nil
}

// modelPricing extracts per-token cost from an oaModel.
// Prefers our own float64 fields; falls back to OpenRouter string "pricing" fields.
// Returns (nil, nil) when pricing is absent — registry normalises nil to +Inf.
func modelPricing(m oaModel) (input, output *float64) {
	if m.InputCostPerToken != nil || m.OutputCostPerToken != nil {
		return m.InputCostPerToken, m.OutputCostPerToken
	}
	if m.Pricing != nil {
		in, errIn := strconv.ParseFloat(m.Pricing.Prompt, 64)
		out, errOut := strconv.ParseFloat(m.Pricing.Completion, 64)
		if errIn == nil && errOut == nil {
			return &in, &out
		}
	}
	return nil, nil
}

// modelCapabilities derives capability strings from an oaModel.
// Prefers our own "capabilities" field; falls back to OpenRouter architecture fields.
func modelCapabilities(m oaModel) []string {
	if len(m.Capabilities) > 0 {
		return m.Capabilities
	}
	if m.Architecture == nil {
		return nil
	}
	seen := map[string]bool{}
	for _, mod := range m.Architecture.InputModalities {
		if mod == "image" {
			seen["vision"] = true
		}
	}
	if strings.Contains(m.Architecture.Modality, "image") {
		seen["vision"] = true
	}
	if len(seen) == 0 {
		return nil
	}
	caps := make([]string, 0, len(seen))
	for c := range seen {
		caps = append(caps, c)
	}
	sort.Strings(caps)
	return caps
}
