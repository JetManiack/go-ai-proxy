package vllm

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"sync/atomic"
	"time"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
	openaiprovider "github.com/JetManiack/go-ai-proxy/internal/provider/openai"
)

// DefaultThinkingKey is the chat_template_kwargs key used when none is configured.
// Matches Qwen3 / Gemma 4 conventions. For Granite / DeepSeek-V3.1, set
// Config.ThinkingKey to "thinking" instead.
const DefaultThinkingKey = "enable_thinking"

// Config holds the construction parameters for a vLLM Provider.
type Config struct {
	Name              string              // provider name shown in logs and audit entries
	BaseURL           string              // upstream vLLM URL, e.g. https://vllm.local
	APIKey            string              // optional; sent as Authorization: Bearer
	Timeout           time.Duration       // response-header timeout for upstream calls
	TLSInsecure       bool                // disable TLS certificate verification (self-signed CAs)
	ThinkingKey       string              // chat_template_kwargs key for thinking control; default "enable_thinking"
	Models            []string            // fallback model IDs when discover is off or upstream /v1/models fails
	Discover          bool                // call upstream /v1/models; falls back to Models[] on failure
	ModelCapabilities map[string][]string // required: model ID → capabilities; missing entries cause New() to fail
}

// Provider wraps an openai.Provider with vLLM-specific request mutation.
type Provider struct {
	cfg    Config
	inner  *openaiprovider.Provider
	primed atomic.Bool // true after first successful Models() validation
}

// New constructs a Provider. It performs an initial Models() round and
// validates that every discovered model has a capability entry in
// cfg.ModelCapabilities. Missing entries cause New() to return an error so
// gap can fail-fast at startup; subsequent refresh-cycle Models() calls are
// lenient (excluded, not fatal).
func New(ctx context.Context, cfg Config) (*Provider, error) {
	if cfg.BaseURL == "" {
		return nil, fmt.Errorf("vllm[%s]: base_url is required", cfg.Name)
	}
	if cfg.ThinkingKey == "" {
		cfg.ThinkingKey = DefaultThinkingKey
	}
	if !cfg.Discover && len(cfg.Models) == 0 {
		return nil, fmt.Errorf("vllm[%s]: discover disabled and no models configured", cfg.Name)
	}

	innerOpts := []openaiprovider.Option{openaiprovider.WithName(cfg.Name)}
	if cfg.APIKey != "" {
		innerOpts = append(innerOpts, openaiprovider.WithAPIKey(cfg.APIKey))
	}
	if cfg.Timeout > 0 {
		innerOpts = append(innerOpts, openaiprovider.WithTimeout(cfg.Timeout))
	}
	if cfg.TLSInsecure {
		innerOpts = append(innerOpts, openaiprovider.WithTLSInsecure(true))
	}

	p := &Provider{cfg: cfg}
	innerOpts = append(innerOpts, openaiprovider.WithRequestBodyMutator(p.mutator))
	p.inner = openaiprovider.New(cfg.BaseURL, innerOpts...)

	// Strict first-call validation — Models() will return error if any
	// discovered ID lacks a capability entry.
	if _, err := p.Models(ctx); err != nil {
		return nil, err
	}
	p.primed.Store(true)
	return p, nil
}

// Name returns the configured provider name.
func (p *Provider) Name() string { return p.cfg.Name }

// Chat delegates to the inner openai provider, which invokes the mutator.
func (p *Provider) Chat(ctx context.Context, req domain.Request) (domain.Response, error) {
	return p.inner.Chat(ctx, req)
}

// ChatStream delegates to the inner openai provider, which invokes the mutator.
func (p *Provider) ChatStream(ctx context.Context, req domain.Request) (<-chan domain.Chunk, error) {
	return p.inner.ChatStream(ctx, req)
}

// Embeddings delegates to the inner openai provider. The mutator is not
// invoked — chat_template_kwargs injection has no meaning for embeddings.
func (p *Provider) Embeddings(ctx context.Context, req domain.EmbedRequest) (domain.EmbedResponse, error) {
	return p.inner.Embeddings(ctx, req)
}

// Models discovers (or falls back to configured) model IDs, validates them
// against ModelCapabilities, and returns the validated set.
//
// First call (primed == false): missing capability entry is a hard error.
// Subsequent calls (primed == true): missing capability entry is logged and
// the model is excluded — keeps the proxy running through upstream surprises.
func (p *Provider) Models(ctx context.Context) ([]domain.Model, error) {
	discovered, discoverErr := p.discoverModels(ctx)
	if (discoverErr != nil || len(discovered) == 0) && len(p.cfg.Models) > 0 {
		slog.Warn("vllm: upstream model discovery failed, using configured fallback",
			"provider", p.cfg.Name, "error", discoverErr, "fallback", p.cfg.Models)
		discovered = make([]domain.Model, 0, len(p.cfg.Models))
		for _, id := range p.cfg.Models {
			discovered = append(discovered, domain.Model{ID: id})
		}
		discoverErr = nil
	}
	if discoverErr != nil {
		return nil, fmt.Errorf("vllm[%s]: model discovery failed and no fallback configured: %w", p.cfg.Name, discoverErr)
	}
	if len(discovered) == 0 {
		return nil, fmt.Errorf("vllm[%s]: no models discovered and none configured", p.cfg.Name)
	}

	primed := p.primed.Load()
	var out []domain.Model
	for _, m := range discovered {
		caps := p.cfg.ModelCapabilities[m.ID]
		if len(caps) == 0 {
			msg := fmt.Sprintf(
				"vllm[%s]: model %q has no capabilities — vLLM upstream does not "+
					"report capabilities, so model_capabilities[%q] must be set "+
					"explicitly in config for capability-based routing (auto:*) to work",
				p.cfg.Name, m.ID, m.ID)
			if !primed {
				return nil, fmt.Errorf("%s", msg)
			}
			slog.Error("vllm: skipping model on refresh: missing capabilities", "provider", p.cfg.Name, "model", m.ID)
			continue
		}
		m.Capabilities = caps
		m.OwnedBy = "vllm"
		out = append(out, m)
	}
	return out, nil
}

// discoverModels returns the list of upstream models with any metadata the
// inner provider/translator could extract (id, max_model_len, etc.).
// If cfg.Discover is true, it calls the inner openai provider's Models();
// otherwise it wraps the configured static list into bare domain.Model values.
func (p *Provider) discoverModels(ctx context.Context) ([]domain.Model, error) {
	if !p.cfg.Discover {
		out := make([]domain.Model, 0, len(p.cfg.Models))
		for _, id := range p.cfg.Models {
			out = append(out, domain.Model{ID: id})
		}
		return out, nil
	}
	return p.inner.Models(ctx)
}

// mutator is the per-request closure installed on the inner openai provider.
// It injects chat_template_kwargs.<thinking_key> based on the canonical
// request's ReasoningEffort. If ReasoningEffort is nil, the body is returned
// unchanged (defer to upstream's server-side default).
func (p *Provider) mutator(body []byte, req domain.Request) ([]byte, error) {
	if req.ReasoningEffort == nil {
		return body, nil
	}
	effort := strings.ToLower(strings.TrimSpace(*req.ReasoningEffort))
	enabled := effort != "" && effort != "none" && effort != "minimal"
	out, err := injectChatTemplateKwargs(body, p.cfg.ThinkingKey, enabled)
	if err != nil {
		return nil, fmt.Errorf("vllm[%s]: inject chat_template_kwargs: %w", p.cfg.Name, err)
	}
	return out, nil
}
