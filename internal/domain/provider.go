package domain

import (
	"context"
	"encoding/json"
)

// Message is a single turn in a conversation.
type Message struct {
	Role       string          // "system", "user", "assistant", "tool"
	Content    string          // plain text; empty when RawContent is set
	RawContent json.RawMessage // original content verbatim (multimodal array or null)
	ToolCalls  []ToolCall
	ToolCallID string // non-empty for role=="tool" messages
}

// ToolCall is a function call made by the assistant.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string // JSON string
}

// Tool describes a function the model can invoke.
type Tool struct {
	Name        string
	Description string
	Parameters  map[string]any // JSON Schema object
}

// ResponseFormat constrains the model's output to a JSON schema. A nil
// *ResponseFormat on Request means no constraint (plain text). Only the
// json_schema form is represented; response_format "text" and "json_object"
// parse to nil.
type ResponseFormat struct {
	Name   string         // json_schema.name (OpenAI); ignored by Anthropic
	Schema map[string]any // the JSON Schema object
	Strict bool           // json_schema.strict (OpenAI); implicit on Anthropic
}

// Request is the provider-agnostic chat request.
type Request struct {
	Model           string
	Messages        []Message
	Tools           []Tool
	Temperature     *float64
	MaxTokens       *int
	Stream          bool
	BudgetTokens    *int    // extended thinking budget; non-nil enables reasoning on supporting providers
	ReasoningEffort *string // OpenAI-standard effort tier: "none" | "minimal" | "low" | "medium" | "high"
	ResponseFormat  *ResponseFormat // non-nil enables structured output
}

// Model describes a single model available from a provider.
type Model struct {
	ID                 string
	OwnedBy            string
	Capabilities       []string // e.g. ["vision", "reasoning"]
	InputCostPerToken  *float64 // USD per input token; nil = not reported by provider
	OutputCostPerToken *float64 // USD per output token; nil = not reported by provider
	MaxModelLen        int      // context window in tokens (prompt + completion); 0 = unknown/not reported
}

// Usage holds token consumption for a single request.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	CachedTokens     int // prompt tokens served from prefix cache (0 = no cache hit or not reported)
}

// Response is the provider-agnostic chat response.
type Response struct {
	ID           string
	Model        string
	Message      Message
	Usage        Usage
	Thinking     string // extended thinking / CoT content, if any
	FinishReason string // "stop" | "length" | "tool_calls" | "content_filter" | "" (unknown)
}

// Chunk is a single delta in a streaming response.
type Chunk struct {
	ID            string
	Model         string
	Delta         string // text delta
	ThinkingDelta string // thinking/CoT delta, if any

	// Tool call streaming fields.
	ToolCallIndex *int
	ToolCallID    string
	ToolCallName  string
	ToolCallArgs  string // JSON fragment

	Done         bool
	Usage        *Usage // non-nil on the terminal chunk when upstream emits usage (stream_options.include_usage)
	FinishReason string // when upstream sets one (typically only on the terminating chunk)
}

// Provider is the interface all LLM providers must implement.
type Provider interface {
	Name() string
	Chat(ctx context.Context, req Request) (Response, error)
	ChatStream(ctx context.Context, req Request) (<-chan Chunk, error)
	Models(ctx context.Context) ([]Model, error)
}
