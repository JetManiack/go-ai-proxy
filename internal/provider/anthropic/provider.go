// Package anthropic implements a Provider for Anthropic's Claude models.
package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"

	"github.com/JetManiack/go-ai-proxy/internal/auth"
	"github.com/JetManiack/go-ai-proxy/internal/domain"
	"github.com/JetManiack/go-ai-proxy/internal/provider"
)

// Provider calls the Anthropic Messages API.
type Provider struct {
	name    string
	auth    auth.Authenticator
	baseURL string
	timeout time.Duration // 0 = no timeout
}

// Name returns the provider's configured name.
func (p *Provider) Name() string { return p.name }

// Option configures a Provider.
type Option func(*Provider)

// WithName sets the provider's name used in logs and metrics.
func WithName(name string) Option {
	return func(p *Provider) { p.name = name }
}

// WithBaseURL overrides the default Anthropic API base URL. Primarily used in tests.
func WithBaseURL(u string) Option {
	return func(p *Provider) { p.baseURL = u }
}

// WithTimeout sets a per-request timeout passed to the Anthropic SDK.
func WithTimeout(d time.Duration) Option {
	return func(p *Provider) { p.timeout = d }
}

// New returns a Provider that authenticates with a.
func New(a auth.Authenticator, opts ...Option) *Provider {
	p := &Provider{auth: a}
	for _, o := range opts {
		o(p)
	}
	return p
}

func (p *Provider) newClient(ctx context.Context) (anthropic.Client, error) {
	token, err := p.auth.GetToken(ctx)
	if err != nil {
		return anthropic.Client{}, fmt.Errorf("anthropic: get token: %w", err)
	}
	reqOpts := []option.RequestOption{option.WithAPIKey(token)}
	if p.baseURL != "" {
		reqOpts = append(reqOpts, option.WithBaseURL(p.baseURL))
	}
	if p.timeout > 0 {
		reqOpts = append(reqOpts, option.WithRequestTimeout(p.timeout))
	}
	return anthropic.NewClient(reqOpts...), nil
}

// Chat sends a non-streaming chat request.
func (p *Provider) Chat(ctx context.Context, req domain.Request) (domain.Response, error) {
	client, err := p.newClient(ctx)
	if err != nil {
		return domain.Response{}, err
	}

	params, err := toMessageParams(req)
	if err != nil {
		return domain.Response{}, fmt.Errorf("anthropic: build params: %w", err)
	}

	msg, err := client.Messages.New(ctx, params)
	if err != nil {
		var apiErr *anthropic.Error
		if errors.As(err, &apiErr) && apiErr.StatusCode == http.StatusTooManyRequests {
			d := provider.ParseRetryAfter(apiErr.Response.Header.Get("Retry-After"))
			return domain.Response{}, &provider.RateLimitError{RetryAfter: d}
		}
		return domain.Response{}, fmt.Errorf("anthropic: messages.new: %w", err)
	}

	return fromMessage(msg), nil
}

// ChatStream initiates a streaming chat request and returns a channel of chunks.
func (p *Provider) ChatStream(ctx context.Context, req domain.Request) (<-chan domain.Chunk, error) {
	client, err := p.newClient(ctx)
	if err != nil {
		return nil, err
	}

	params, err := toMessageParams(req)
	if err != nil {
		return nil, fmt.Errorf("anthropic: build params: %w", err)
	}

	stream := client.Messages.NewStreaming(ctx, params)

	// Call Next() once synchronously to detect early HTTP errors (e.g. 429) before
	// returning the channel, so callers can fall back to the next provider.
	if !stream.Next() {
		stream.Close()
		if streamErr := stream.Err(); streamErr != nil {
			var apiErr *anthropic.Error
			if errors.As(streamErr, &apiErr) && apiErr.StatusCode == http.StatusTooManyRequests {
				d := provider.ParseRetryAfter(apiErr.Response.Header.Get("Retry-After"))
				return nil, &provider.RateLimitError{RetryAfter: d}
			}
			return nil, fmt.Errorf("anthropic: stream error: %w", streamErr)
		}
		// Empty stream (message_stop as first event or no events).
		ch := make(chan domain.Chunk, 1)
		ch <- domain.Chunk{Done: true}
		close(ch)
		return ch, nil
	}
	firstEvent := stream.Current()

	ch := make(chan domain.Chunk)
	go func() {
		defer close(ch)
		defer stream.Close()

		// toolCallIdx maps content block index → tool call index (0-based among tool_use blocks).
		toolCallIdx := map[int64]int{}
		nextToolCall := 0

		send := func(c domain.Chunk) bool {
			select {
			case ch <- c:
				return true
			case <-ctx.Done():
				return false
			}
		}

		processEvent := func(event anthropic.MessageStreamEventUnion) bool {
			switch event.Type {
			case "content_block_start":
				cb := event.AsContentBlockStart()
				if cb.ContentBlock.Type == "tool_use" {
					idx := nextToolCall
					nextToolCall++
					toolCallIdx[cb.Index] = idx
					return send(domain.Chunk{
						ToolCallIndex: &idx,
						ToolCallID:    cb.ContentBlock.ID,
						ToolCallName:  cb.ContentBlock.Name,
					})
				}
			case "content_block_delta":
				delta := event.AsContentBlockDelta()
				switch delta.Delta.Type {
				case "text_delta":
					return send(domain.Chunk{Delta: delta.Delta.Text})
				case "thinking_delta":
					return send(domain.Chunk{ThinkingDelta: delta.Delta.Thinking})
				case "input_json_delta":
					if idx, ok := toolCallIdx[delta.Index]; ok {
						idxCopy := idx
						return send(domain.Chunk{ToolCallIndex: &idxCopy, ToolCallArgs: delta.Delta.PartialJSON})
					}
				}
			case "message_stop":
				send(domain.Chunk{Done: true})
				return false
			}
			return true
		}

		if !processEvent(firstEvent) {
			return
		}
		for stream.Next() {
			if !processEvent(stream.Current()) {
				return
			}
		}

		if err := stream.Err(); err != nil && ctx.Err() == nil {
			slog.Error("anthropic: stream error", "error", err)
		}
	}()

	return ch, nil
}

// Models returns the list of available Anthropic models.
func (p *Provider) Models(ctx context.Context) ([]domain.Model, error) {
	client, err := p.newClient(ctx)
	if err != nil {
		return nil, err
	}

	page, err := client.Models.List(ctx, anthropic.ModelListParams{})
	if err != nil {
		return nil, fmt.Errorf("anthropic: models.list: %w", err)
	}

	models := make([]domain.Model, 0, len(page.Data))
	for _, m := range page.Data {
		models = append(models, domain.Model{
			ID:      m.ID,
			OwnedBy: "anthropic",
		})
	}
	return models, nil
}

// --- Translation: canonical → Anthropic ---

func toMessageParams(req domain.Request) (anthropic.MessageNewParams, error) {
	maxTokens := int64(4096)
	if req.MaxTokens != nil {
		maxTokens = int64(*req.MaxTokens)
	}

	var systemText string
	var rawMsgs []domain.Message
	for _, m := range req.Messages {
		if m.Role == "system" {
			if systemText != "" {
				systemText += "\n"
			}
			systemText += m.Content
		} else {
			rawMsgs = append(rawMsgs, m)
		}
	}

	msgs, err := toAnthropicMessages(rawMsgs)
	if err != nil {
		return anthropic.MessageNewParams{}, err
	}

	p := anthropic.MessageNewParams{
		Model:     anthropic.Model(req.Model),
		MaxTokens: maxTokens,
		Messages:  msgs,
	}

	if req.BudgetTokens != nil {
		// Extended thinking requires temperature=1; ignore any client-supplied value.
		p.Thinking = anthropic.ThinkingConfigParamOfEnabled(int64(*req.BudgetTokens))
	} else if req.Temperature != nil {
		p.Temperature = param.NewOpt(*req.Temperature)
	}

	if systemText != "" {
		p.System = []anthropic.TextBlockParam{
			{Text: systemText},
		}
	}

	if len(req.Tools) > 0 {
		tools, err := toAnthropicTools(req.Tools)
		if err != nil {
			return anthropic.MessageNewParams{}, err
		}
		p.Tools = tools
	}

	return p, nil
}

func toAnthropicMessages(messages []domain.Message) ([]anthropic.MessageParam, error) {
	var result []anthropic.MessageParam

	i := 0
	for i < len(messages) {
		m := messages[i]
		switch m.Role {
		case "user":
			result = append(result, anthropic.NewUserMessage(anthropic.NewTextBlock(m.Content)))
			i++

		case "assistant":
			blocks, err := assistantContentBlocks(m)
			if err != nil {
				return nil, err
			}
			result = append(result, anthropic.NewAssistantMessage(blocks...))
			i++

		case "tool":
			// Group consecutive tool-result messages into a single user message.
			var toolBlocks []anthropic.ContentBlockParamUnion
			for i < len(messages) && messages[i].Role == "tool" {
				t := messages[i]
				toolBlocks = append(toolBlocks, anthropic.ContentBlockParamUnion{
					OfToolResult: &anthropic.ToolResultBlockParam{
						ToolUseID: t.ToolCallID,
						Content: []anthropic.ToolResultBlockParamContentUnion{
							{OfText: &anthropic.TextBlockParam{Text: t.Content}},
						},
					},
				})
				i++
			}
			result = append(result, anthropic.NewUserMessage(toolBlocks...))

		default:
			i++
		}
	}

	return result, nil
}

func assistantContentBlocks(m domain.Message) ([]anthropic.ContentBlockParamUnion, error) {
	var blocks []anthropic.ContentBlockParamUnion
	if m.Content != "" {
		blocks = append(blocks, anthropic.NewTextBlock(m.Content))
	}
	for _, tc := range m.ToolCalls {
		var input any
		if err := json.Unmarshal([]byte(tc.Arguments), &input); err != nil {
			slog.Warn("anthropic: invalid tool call arguments, sending empty input", "id", tc.ID, "name", tc.Name, "error", err)
			input = map[string]any{}
		}
		blocks = append(blocks, anthropic.ContentBlockParamUnion{
			OfToolUse: &anthropic.ToolUseBlockParam{
				ID:    tc.ID,
				Name:  tc.Name,
				Input: input,
			},
		})
	}
	return blocks, nil
}

func toAnthropicTools(tools []domain.Tool) ([]anthropic.ToolUnionParam, error) {
	result := make([]anthropic.ToolUnionParam, 0, len(tools))
	for _, t := range tools {
		tp := anthropic.ToolParam{
			Name:        t.Name,
			Description: param.NewOpt(t.Description),
			InputSchema: anthropic.ToolInputSchemaParam{
				Properties: t.Parameters,
			},
		}
		result = append(result, anthropic.ToolUnionParam{OfTool: &tp})
	}
	return result, nil
}

// --- Translation: Anthropic → canonical ---

func fromMessage(msg *anthropic.Message) domain.Response {
	var textContent, thinkingContent string
	var toolCalls []domain.ToolCall

	for _, block := range msg.Content {
		switch block.Type {
		case "text":
			textContent += block.Text
		case "thinking":
			thinkingContent += block.Thinking
		case "tool_use":
			args, err := json.Marshal(block.Input)
			if err != nil {
				slog.Error("anthropic: failed to marshal tool use input", "id", block.ID, "name", block.Name, "error", err)
				args = []byte("{}")
			}
			toolCalls = append(toolCalls, domain.ToolCall{
				ID:        block.ID,
				Name:      block.Name,
				Arguments: string(args),
			})
		}
	}

	return domain.Response{
		ID:      msg.ID,
		Model:   string(msg.Model),
		Thinking: thinkingContent,
		Message: domain.Message{
			Role:      "assistant",
			Content:   textContent,
			ToolCalls: toolCalls,
		},
		Usage: domain.Usage{
			PromptTokens:     int(msg.Usage.InputTokens),
			CompletionTokens: int(msg.Usage.OutputTokens),
		},
	}
}
