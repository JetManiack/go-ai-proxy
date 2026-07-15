package llamacpp

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
	openaiprovider "github.com/JetManiack/go-ai-proxy/internal/provider/openai"
)

// Config holds construction parameters for a llama.cpp Provider.
type Config struct {
	Name              string
	BaseURL           string
	APIKey            string
	Timeout           time.Duration
	TLSInsecure       bool
	ModelCapabilities map[string][]string // required: model ID → capabilities
}

// Provider wraps an openai.Provider, converting response_format json_schema to
// a GBNF grammar (or forwarding a client-supplied grammar) on each request.
type Provider struct {
	cfg   Config
	inner *openaiprovider.Provider
}

func (p *Provider) Name() string { return p.cfg.Name }

// New constructs a Provider. model_capabilities is required — llama.cpp does
// not report capabilities, and structured_output must be declared for
// capability-based routing.
func New(_ context.Context, cfg Config) (*Provider, error) {
	if cfg.BaseURL == "" {
		return nil, fmt.Errorf("llamacpp[%s]: base_url is required", cfg.Name)
	}
	if len(cfg.ModelCapabilities) == 0 {
		return nil, fmt.Errorf("llamacpp[%s]: model_capabilities is required — llama.cpp upstreams do not report capabilities", cfg.Name)
	}
	p := &Provider{cfg: cfg}
	opts := []openaiprovider.Option{
		openaiprovider.WithName(cfg.Name),
		openaiprovider.WithRequestBodyMutator(p.mutator),
	}
	if cfg.APIKey != "" {
		opts = append(opts, openaiprovider.WithAPIKey(cfg.APIKey))
	}
	if cfg.Timeout > 0 {
		opts = append(opts, openaiprovider.WithTimeout(cfg.Timeout))
	}
	if cfg.TLSInsecure {
		opts = append(opts, openaiprovider.WithTLSInsecure(true))
	}
	p.inner = openaiprovider.New(cfg.BaseURL, opts...)
	return p, nil
}

func (p *Provider) Chat(ctx context.Context, req domain.Request) (domain.Response, error) {
	return p.inner.Chat(ctx, req)
}

func (p *Provider) ChatStream(ctx context.Context, req domain.Request) (<-chan domain.Chunk, error) {
	return p.inner.ChatStream(ctx, req)
}

// Models returns upstream models, attaching configured capabilities.
func (p *Provider) Models(ctx context.Context) ([]domain.Model, error) {
	models, err := p.inner.Models(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]domain.Model, 0, len(models))
	for _, m := range models {
		caps := p.cfg.ModelCapabilities[m.ID]
		if len(caps) == 0 {
			slog.Warn("llamacpp: model has no configured capabilities", "provider", p.cfg.Name, "model", m.ID)
		}
		m.Capabilities = caps
		m.OwnedBy = "llamacpp"
		out = append(out, m)
	}
	return out, nil
}

// mutator converts structured-output intent into a GBNF grammar field:
//   - explicit client grammar → forwarded verbatim (response_format stripped)
//   - response_format json_schema → converted to GBNF (response_format stripped)
//   - neither → body unchanged
func (p *Provider) mutator(body []byte, req domain.Request) ([]byte, error) {
	if req.Grammar != "" {
		return injectGrammar(body, req.Grammar)
	}
	if req.ResponseFormat != nil {
		grammar, err := schemaToGBNF(req.ResponseFormat.Schema)
		if err != nil {
			return nil, err
		}
		return injectGrammar(body, grammar)
	}
	return body, nil
}

// injectGrammar sets body["grammar"] and removes body["response_format"].
func injectGrammar(body []byte, grammar string) ([]byte, error) {
	var root map[string]any
	if err := json.Unmarshal(body, &root); err != nil {
		return nil, fmt.Errorf("llamacpp: decode body: %w", err)
	}
	if root == nil {
		root = map[string]any{}
	}
	root["grammar"] = grammar
	delete(root, "response_format")
	return json.Marshal(root)
}
