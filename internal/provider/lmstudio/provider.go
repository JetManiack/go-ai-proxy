// Package lmstudio implements a Provider for LM Studio's local server.
// Chat and streaming are forwarded via the embedded OpenAI-compatible provider.
// Models() uses LM Studio's native /api/v1/models to auto-discover vision and
// reasoning capabilities without requiring manual model_capabilities config.
package lmstudio

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

// Provider forwards requests to LM Studio and auto-discovers model capabilities.
type Provider struct {
	inner       *openaiprovider.Provider
	name        string
	baseURL     string
	timeout     time.Duration
	tlsInsecure bool
	client      *http.Client
}

// Option configures a Provider.
type Option func(*Provider)

// WithName sets the provider's name used in logs and metrics.
func WithName(name string) Option {
	return func(p *Provider) { p.name = name }
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

// New returns a Provider pointing at baseURL (e.g. "http://localhost:1234").
func New(baseURL string, opts ...Option) *Provider {
	p := &Provider{baseURL: baseURL}
	for _, o := range opts {
		o(p)
	}

	innerOpts := []openaiprovider.Option{openaiprovider.WithName(p.name)}
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

// Models fetches model metadata from LM Studio's native /api/v1/models endpoint
// and maps capabilities automatically. Falls back to /v1/models without capabilities
// if the native endpoint is unavailable (older LM Studio versions).
func (p *Provider) Models(ctx context.Context) ([]domain.Model, error) {
	models, err := p.fetchNativeModels(ctx)
	if err != nil {
		slog.Debug("lmstudio: /api/v1/models unavailable, falling back to /v1/models",
			"provider", p.name, "error", err)
		return p.inner.Models(ctx)
	}
	return models, nil
}

// lmStudioModelsResponse mirrors the /api/v1/models response.
type lmStudioModelsResponse struct {
	Models []lmStudioModel `json:"models"`
}

type lmStudioModel struct {
	Type         string              `json:"type"`
	Key          string              `json:"key"`
	Capabilities *lmStudioCapabilities `json:"capabilities"`
}

type lmStudioCapabilities struct {
	Vision            bool               `json:"vision"`
	TrainedForToolUse bool               `json:"trained_for_tool_use"`
	Reasoning         *lmStudioReasoning `json:"reasoning"`
}

type lmStudioReasoning struct {
	AllowedOptions []string `json:"allowed_options"`
}

func (p *Provider) fetchNativeModels(ctx context.Context) ([]domain.Model, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.baseURL+"/api/v1/models", nil)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
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

	var native lmStudioModelsResponse
	if err := json.Unmarshal(body, &native); err != nil {
		return nil, fmt.Errorf("parse /api/v1/models: %w", err)
	}

	var models []domain.Model
	for _, m := range native.Models {
		if m.Type != "llm" {
			continue
		}
		models = append(models, domain.Model{
			ID:           m.Key,
			OwnedBy:      "lmstudio",
			Capabilities: mapCapabilities(m.Capabilities),
		})
	}
	return models, nil
}

func mapCapabilities(caps *lmStudioCapabilities) []string {
	if caps == nil {
		return nil
	}
	var result []string
	if caps.Vision {
		result = append(result, "vision")
	}
	if caps.TrainedForToolUse {
		result = append(result, "tools")
	}
	if caps.Reasoning != nil {
		result = append(result, "reasoning")
	}
	return result
}
