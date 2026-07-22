// Package openai implements a Provider for any OpenAI-compatible HTTP server
// (LM Studio, Ollama, OpenAI, OpenRouter, etc.).
package openai

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
	"github.com/JetManiack/go-ai-proxy/internal/provider"
	"github.com/JetManiack/go-ai-proxy/internal/translator"
)

// RequestBodyMutator transforms the marshalled OpenAI request body before it is
// sent upstream. It receives both the JSON-encoded body and the canonical
// domain.Request that produced it, so closures can branch on per-request
// state without sharing mutable provider state.
type RequestBodyMutator func(body []byte, req domain.Request) ([]byte, error)

// Provider forwards requests to an OpenAI-compatible server.
type Provider struct {
	name        string
	baseURL     string
	apiKey      string // optional; if set, sent as "Authorization: Bearer <key>"
	timeout     time.Duration
	tlsInsecure bool
	mutator     RequestBodyMutator // optional; mutates the marshalled body before sending
	client      *http.Client
}

// Name returns the provider's configured name.
func (p *Provider) Name() string { return p.name }

// Option configures a Provider.
type Option func(*Provider)

// WithName sets the provider's name used in logs and metrics.
func WithName(name string) Option {
	return func(p *Provider) { p.name = name }
}

// WithAPIKey sets an API key sent as a Bearer token on every request.
// Leave unset for local servers that don't require auth (LM Studio, Ollama).
func WithAPIKey(key string) Option {
	return func(p *Provider) { p.apiKey = key }
}

// WithTimeout sets a timeout on the initial server response (headers).
// It does not affect ongoing streaming bodies, so long SSE sessions continue uninterrupted.
func WithTimeout(d time.Duration) Option {
	return func(p *Provider) { p.timeout = d }
}

// WithTLSInsecure disables TLS certificate verification for upstream HTTPS calls.
// Intended for self-hosted endpoints with self-signed certificates; never use
// against public providers.
func WithTLSInsecure(insecure bool) Option {
	return func(p *Provider) { p.tlsInsecure = insecure }
}

// WithRequestBodyMutator installs a hook invoked after the canonical request
// has been marshalled to OpenAI wire format but before it is sent upstream.
// The hook may return a transformed body or an error. nil mutator is a no-op.
func WithRequestBodyMutator(fn RequestBodyMutator) Option {
	return func(p *Provider) { p.mutator = fn }
}

// New returns a Provider pointing at baseURL (e.g. "http://localhost:1234").
func New(baseURL string, opts ...Option) *Provider {
	p := &Provider{baseURL: baseURL}
	for _, o := range opts {
		o(p)
	}
	p.client = buildHTTPClient(p.timeout, p.tlsInsecure)
	return p
}

// buildHTTPClient assembles a *http.Client honouring both the response-header
// timeout and the TLS-insecure flag. A nil Transport on http.Client falls back
// to http.DefaultTransport, so when neither option is set we leave Transport
// unset to keep connection pooling defaults.
func buildHTTPClient(timeout time.Duration, tlsInsecure bool) *http.Client {
	if timeout == 0 && !tlsInsecure {
		return &http.Client{}
	}
	tr := &http.Transport{ResponseHeaderTimeout: timeout}
	if tlsInsecure {
		tr.TLSClientConfig = &tls.Config{InsecureSkipVerify: true}
	}
	return &http.Client{Transport: tr}
}

// Chat sends a non-streaming chat request and returns the response.
func (p *Provider) Chat(ctx context.Context, req domain.Request) (domain.Response, error) {
	body, err := translator.RequestToOpenAI(req)
	if err != nil {
		return domain.Response{}, fmt.Errorf("openai: encode request: %w", err)
	}

	if p.mutator != nil {
		body, err = p.mutator(body, req)
		if err != nil {
			return domain.Response{}, fmt.Errorf("openai: mutate request body: %w", err)
		}
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		p.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return domain.Response{}, fmt.Errorf("openai: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	httpResp, err := p.client.Do(httpReq)
	if err != nil {
		return domain.Response{}, fmt.Errorf("openai: do request: %w", err)
	}
	defer httpResp.Body.Close()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return domain.Response{}, fmt.Errorf("openai: read response: %w", err)
	}
	if httpResp.StatusCode == http.StatusTooManyRequests {
		retryAfter := provider.ParseRetryAfter(httpResp.Header.Get("Retry-After"))
		return domain.Response{}, &provider.RateLimitError{RetryAfter: retryAfter}
	}
	if httpResp.StatusCode != http.StatusOK {
		return domain.Response{}, fmt.Errorf("openai: upstream returned %d: %s", httpResp.StatusCode, respBody)
	}

	return translator.ResponseFromOpenAI(respBody)
}

// Embeddings sends a request to the upstream /v1/embeddings endpoint.
func (p *Provider) Embeddings(ctx context.Context, req domain.EmbedRequest) (domain.EmbedResponse, error) {
	body, err := translator.EmbedRequestToOpenAI(req)
	if err != nil {
		return domain.EmbedResponse{}, fmt.Errorf("openai: encode embeddings request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		p.baseURL+"/v1/embeddings", bytes.NewReader(body))
	if err != nil {
		return domain.EmbedResponse{}, fmt.Errorf("openai: build embeddings request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	httpResp, err := p.client.Do(httpReq)
	if err != nil {
		return domain.EmbedResponse{}, fmt.Errorf("openai: do embeddings request: %w", err)
	}
	defer httpResp.Body.Close()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return domain.EmbedResponse{}, fmt.Errorf("openai: read embeddings response: %w", err)
	}
	if httpResp.StatusCode == http.StatusTooManyRequests {
		retryAfter := provider.ParseRetryAfter(httpResp.Header.Get("Retry-After"))
		return domain.EmbedResponse{}, &provider.RateLimitError{RetryAfter: retryAfter}
	}
	if httpResp.StatusCode != http.StatusOK {
		return domain.EmbedResponse{}, &provider.UpstreamError{StatusCode: httpResp.StatusCode, Body: string(respBody)}
	}

	return translator.EmbedResponseFromOpenAI(respBody)
}

// ChatStream initiates a streaming chat request and returns a channel of chunks.
func (p *Provider) ChatStream(ctx context.Context, req domain.Request) (<-chan domain.Chunk, error) {
	req.Stream = true
	body, err := translator.RequestToOpenAI(req)
	if err != nil {
		return nil, fmt.Errorf("openai: encode request: %w", err)
	}

	if p.mutator != nil {
		body, err = p.mutator(body, req)
		if err != nil {
			return nil, fmt.Errorf("openai: mutate request body: %w", err)
		}
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		p.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	httpResp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai: do request: %w", err)
	}
	if httpResp.StatusCode != http.StatusOK {
		retryAfterHdr := httpResp.Header.Get("Retry-After")
		b, _ := io.ReadAll(httpResp.Body)
		httpResp.Body.Close()
		if httpResp.StatusCode == http.StatusTooManyRequests {
			return nil, &provider.RateLimitError{RetryAfter: provider.ParseRetryAfter(retryAfterHdr)}
		}
		return nil, fmt.Errorf("openai: upstream returned %d: %s", httpResp.StatusCode, b)
	}

	ch := make(chan domain.Chunk)
	go func() {
		defer close(ch)
		defer httpResp.Body.Close()

		parser := &thinkParser{}
		scanner := bufio.NewScanner(httpResp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				ch <- domain.Chunk{Done: true}
				return
			}
			chunk, err := translator.ChunkFromOpenAI([]byte(data))
			if err != nil {
				slog.Warn("openai: failed to parse stream chunk", "error", err)
				continue
			}
			select {
			case ch <- parser.process(chunk):
			case <-ctx.Done():
				return
			}
		}
	}()

	return ch, nil
}

// Models fetches the list of available models from the upstream server.
func (p *Provider) Models(ctx context.Context) ([]domain.Model, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, p.baseURL+"/v1/models", nil)
	if err != nil {
		return nil, fmt.Errorf("openai: build models request: %w", err)
	}
	if p.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	httpResp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai: do models request: %w", err)
	}
	defer httpResp.Body.Close()

	body, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("openai: read models response: %w", err)
	}
	if httpResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("openai: upstream returned %d: %s", httpResp.StatusCode, body)
	}

	return translator.ModelsFromOpenAI(body)
}
