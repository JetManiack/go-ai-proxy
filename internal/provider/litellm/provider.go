// Package litellm implements a Provider for LiteLLM-hosted endpoints.
// It proxies all chat/stream requests via the embedded OpenAI-compatible provider
// and enriches model listings with capability metadata from LiteLLM's /model/info.
package litellm

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
	openaiprovider "github.com/JetManiack/go-ai-proxy/internal/provider/openai"
)

// Provider forwards requests to a LiteLLM instance and auto-discovers model capabilities.
type Provider struct {
	inner       *openaiprovider.Provider
	name        string
	baseURL     string
	apiKey      string
	timeout     time.Duration
	tlsInsecure bool
	client      *http.Client
}

// Option configures a Provider.
type Option func(*Provider)

// WithName sets the provider's name.
func WithName(name string) Option {
	return func(p *Provider) { p.name = name }
}

// WithAPIKey sets a Bearer token sent on every request.
func WithAPIKey(key string) Option {
	return func(p *Provider) { p.apiKey = key }
}

// WithTimeout sets a response-header timeout for all upstream requests.
func WithTimeout(d time.Duration) Option {
	return func(p *Provider) { p.timeout = d }
}

// WithTLSInsecure disables TLS certificate verification for upstream HTTPS calls.
// Intended for self-hosted endpoints with self-signed certificates.
func WithTLSInsecure(insecure bool) Option {
	return func(p *Provider) { p.tlsInsecure = insecure }
}

// New returns a Provider pointing at baseURL (e.g. "https://litellm.example.com").
func New(baseURL string, opts ...Option) *Provider {
	p := &Provider{baseURL: baseURL}
	for _, o := range opts {
		o(p)
	}

	innerOpts := []openaiprovider.Option{openaiprovider.WithName(p.name)}
	if p.apiKey != "" {
		innerOpts = append(innerOpts, openaiprovider.WithAPIKey(p.apiKey))
	}
	if p.timeout > 0 {
		innerOpts = append(innerOpts, openaiprovider.WithTimeout(p.timeout))
	}
	if p.tlsInsecure {
		innerOpts = append(innerOpts, openaiprovider.WithTLSInsecure(true))
	}
	p.inner = openaiprovider.New(baseURL, innerOpts...)

	if p.timeout == 0 && !p.tlsInsecure {
		p.client = &http.Client{}
	} else {
		tr := &http.Transport{ResponseHeaderTimeout: p.timeout}
		if p.tlsInsecure {
			tr.TLSClientConfig = &tls.Config{InsecureSkipVerify: true}
		}
		p.client = &http.Client{Transport: tr}
	}
	return p
}

// Name returns the configured provider name.
func (p *Provider) Name() string { return p.name }

// Chat delegates to the inner OpenAI-compatible provider.
func (p *Provider) Chat(ctx context.Context, req domain.Request) (domain.Response, error) {
	return p.inner.Chat(ctx, req)
}

// ChatStream delegates to the inner OpenAI-compatible provider.
func (p *Provider) ChatStream(ctx context.Context, req domain.Request) (<-chan domain.Chunk, error) {
	return p.inner.ChatStream(ctx, req)
}

// Embeddings delegates to the inner OpenAI-compatible provider.
func (p *Provider) Embeddings(ctx context.Context, req domain.EmbedRequest) (domain.EmbedResponse, error) {
	return p.inner.Embeddings(ctx, req)
}

// Models fetches the model list and enriches it with capability metadata from /model/info.
// If /model/info is unavailable (e.g. non-LiteLLM server), capabilities are omitted silently.
func (p *Provider) Models(ctx context.Context) ([]domain.Model, error) {
	models, err := p.inner.Models(ctx)
	if err != nil {
		return nil, err
	}

	meta, err := p.fetchModelInfo(ctx)
	if err != nil {
		slog.Debug("litellm: /model/info unavailable, skipping capability enrichment",
			"provider", p.name, "error", err)
		return models, nil
	}

	for i, m := range models {
		if md, ok := meta[m.ID]; ok {
			if len(md.Capabilities) > 0 {
				models[i].Capabilities = md.Capabilities
			}
			models[i].InputCostPerToken = md.InputCostPerToken
			models[i].OutputCostPerToken = md.OutputCostPerToken
		}
	}
	return models, nil
}

// litellmModelInfo mirrors the relevant fields of LiteLLM's /model/info response.
type litellmModelInfo struct {
	Data []struct {
		ModelName string `json:"model_name"`
		ModelInfo struct {
			SupportsVision          bool     `json:"supports_vision"`
			SupportsReasoning       bool     `json:"supports_reasoning"`
			SupportsFunctionCalling bool     `json:"supports_function_calling"`
			SupportsToolChoice      bool     `json:"supports_tool_choice"`
			SupportsPdfInput        bool     `json:"supports_pdf_input"`
			SupportsResponseSchema  bool     `json:"supports_response_schema"`
			SupportsWebSearch       bool     `json:"supports_web_search"`
			SupportsAudioInput      bool     `json:"supports_audio_input"`
			SupportsAudioOutput     bool     `json:"supports_audio_output"`
			SupportsComputerUse     bool     `json:"supports_computer_use"`
			SupportsPromptCaching   bool     `json:"supports_prompt_caching"`
			SupportsUrlContext      bool     `json:"supports_url_context"`
			InputCostPerToken       *float64 `json:"input_cost_per_token"`
			OutputCostPerToken      *float64 `json:"output_cost_per_token"`
		} `json:"model_info"`
	} `json:"data"`
}

type litellmModelMeta struct {
	Capabilities       []string
	InputCostPerToken  *float64 // nil when /model/info doesn't include this model
	OutputCostPerToken *float64
}

// fetchModelInfo calls /model/info and returns a map of model name → metadata.
func (p *Provider) fetchModelInfo(ctx context.Context) (map[string]litellmModelMeta, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.baseURL+"/model/info", nil)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("upstream returned %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}

	var info litellmModelInfo
	if err := json.Unmarshal(body, &info); err != nil {
		return nil, fmt.Errorf("parse /model/info: %w", err)
	}

	result := make(map[string]litellmModelMeta, len(info.Data))
	for _, entry := range info.Data {
		mi := entry.ModelInfo
		var caps []string
		if mi.SupportsVision {
			caps = append(caps, "vision")
		}
		if mi.SupportsReasoning {
			caps = append(caps, "reasoning")
		}
		if mi.SupportsFunctionCalling || mi.SupportsToolChoice {
			caps = append(caps, "tools")
		}
		if mi.SupportsPdfInput {
			caps = append(caps, "pdf")
		}
		if mi.SupportsResponseSchema {
			caps = append(caps, "structured_output")
		}
		if mi.SupportsWebSearch {
			caps = append(caps, "web_search")
		}
		if mi.SupportsAudioInput {
			caps = append(caps, "audio_input")
		}
		if mi.SupportsAudioOutput {
			caps = append(caps, "audio_output")
		}
		if mi.SupportsComputerUse {
			caps = append(caps, "computer_use")
		}
		if mi.SupportsPromptCaching {
			caps = append(caps, "prompt_caching")
		}
		if mi.SupportsUrlContext {
			caps = append(caps, "url_context")
		}
		result[entry.ModelName] = litellmModelMeta{
			Capabilities:       caps,
			InputCostPerToken:  mi.InputCostPerToken,
			OutputCostPerToken: mi.OutputCostPerToken,
		}
	}
	return result, nil
}
