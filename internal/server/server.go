// Package server implements the OpenAI-compatible HTTP API.
package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
	"github.com/JetManiack/go-ai-proxy/internal/metrics"
	"github.com/JetManiack/go-ai-proxy/internal/provider"
	"github.com/JetManiack/go-ai-proxy/internal/translator"
)

// Option configures a Server.
type Option func(*Server)

// WithMaxBodyBytes sets a maximum allowed size for request bodies.
// Requests exceeding the limit are rejected with 413. Zero means no limit.
func WithMaxBodyBytes(n int64) Option {
	return func(s *Server) { s.maxBodyBytes = n }
}

// WithAuditLog enables structured audit logging of every chat request/response
// using the provided logger. Pass nil to disable (default).
// In production, pass slog.Default() or a child logger.
func WithAuditLog(logger *slog.Logger) Option {
	return func(s *Server) { s.auditLogger = logger }
}

// WithMetrics enables Prometheus-format metrics collection.
// The /metrics endpoint is always registered; without this option it returns 204.
func WithMetrics(m *metrics.Metrics) Option {
	return func(s *Server) { s.metrics = m }
}

// Server is an http.Handler that exposes the OpenAI-compatible API.
type Server struct {
	registry     *provider.Registry
	handler      http.Handler
	maxBodyBytes int64
	auditLogger  *slog.Logger
	rateLimitCfg *RateLimitConfig
	tokenBudget  *TokenBudgetConfig
	metrics      *metrics.Metrics
}

// New creates a Server backed by the given registry.
func New(reg *provider.Registry, opts ...Option) *Server {
	s := &Server{registry: reg}
	for _, o := range opts {
		o(s)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/healthz", s.handleHealthz)
	mux.HandleFunc("/metrics", s.handleMetrics)

	var h http.Handler = mux
	h = bodyLimitMiddleware(h, s.maxBodyBytes)
	if s.rateLimitCfg != nil {
		h = rateLimitMiddleware(h, *s.rateLimitCfg)
	}
	h = loggingMiddleware(h)
	s.handler = h
	return s
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.handler.ServeHTTP(w, r)
}

// logStructuredOutputSupport emits a routing-time warning when a structured
// output request targets a model that is unlikely to support it. It never
// blocks the request — the upstream provider is the source of truth.
func (s *Server) logStructuredOutputSupport(modelID string) {
	caps, known := s.registry.CapabilitiesFor(modelID)
	switch {
	case !known || len(caps) == 0:
		slog.Warn("response_format requested; structured_output capability unknown for model",
			"model", modelID)
	case !slices.Contains(caps, "structured_output"):
		slog.Warn("response_format requested; model does not report structured_output capability",
			"model", modelID)
	}
}

// handleChatCompletions handles POST /v1/chat/completions.
func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "only POST is supported")
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			writeError(w, http.StatusRequestEntityTooLarge, "request_too_large",
				fmt.Sprintf("request body exceeds limit of %d bytes", maxErr.Limit))
			return
		}
		writeError(w, http.StatusBadRequest, "bad_request", "failed to read request body")
		return
	}

	req, err := translator.RequestFromOpenAI(body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "bad_request", fmt.Sprintf("invalid request: %v", err))
		return
	}

	originalModel := req.Model
	candidates := s.registry.CandidatesFor(originalModel)
	if len(candidates) == 0 {
		// Model not in cache — trigger an on-demand refresh and retry once.
		// This handles models that became available after the proxy started
		// (e.g. a model loaded into LM Studio mid-session).
		slog.Info("model not in cache, triggering on-demand refresh", "model", originalModel)
		s.registry.Refresh(r.Context())
		candidates = s.registry.CandidatesFor(originalModel)
	}
	if len(candidates) == 0 {
		models := s.registry.Models()
		ids := make([]string, len(models))
		for i, m := range models {
			ids[i] = m.ID
		}
		writeError(w, http.StatusBadRequest, "model_not_found",
			fmt.Sprintf("model %q not available; known models: %v", originalModel, ids))
		return
	}

	// Use first (least-loaded) candidate's model ID for token budget check.
	req.Model = candidates[0].ModelID
	if limit := s.tokenBudget.limitFor(req.Model); limit > 0 && req.MaxTokens != nil && *req.MaxTokens > limit {
		writeError(w, http.StatusBadRequest, "token_budget_exceeded",
			fmt.Sprintf("max_tokens %d exceeds budget of %d for model %q", *req.MaxTokens, limit, req.Model))
		return
	}

	if req.ResponseFormat != nil {
		s.logStructuredOutputSupport(req.Model)
	}

	if req.Stream {
		s.handleStream(w, r, candidates, req)
		return
	}

	start := time.Now()
	var resp domain.Response
	var lastErr error
	allRateLimited := len(candidates) > 0
	var maxRetryAfter time.Duration
	for _, c := range candidates {
		req.Model = c.ModelID
		var err error
		resp, err = c.Provider.Chat(r.Context(), req)
		durationMs := time.Since(start).Milliseconds()
		if err == nil {
			lastErr = nil
			allRateLimited = false
			if s.metrics != nil {
				s.metrics.RecordRequest(req.Model, c.Provider.Name(), "success", durationMs)
				s.metrics.RecordTokens(c.Provider.Name(), req.Model, resp.Usage.PromptTokens, resp.Usage.CompletionTokens, resp.Usage.CachedTokens)
			}
			break
		}
		if s.metrics != nil {
			s.metrics.RecordRequest(req.Model, c.Provider.Name(), "error", durationMs)
		}
		slog.Warn("provider chat error, trying next", "model", req.Model, "provider", c.Provider.Name(), "error", err)
		lastErr = err
		var rl *provider.RateLimitError
		if errors.As(err, &rl) {
			if rl.RetryAfter > maxRetryAfter {
				maxRetryAfter = rl.RetryAfter
			}
		} else {
			allRateLimited = false
		}
	}
	if lastErr != nil {
		if r.Context().Err() != nil {
			// Client disconnected (or server shutdown) before any candidate
			// returned. Don't log as ERROR / write 502 to a dead socket.
			auditClientDisconnect(s.auditLogger, r.Context(), req.Model, start)
			w.WriteHeader(499) // Nginx convention: Client Closed Request
			return
		}
		slog.Error("all providers failed", "model", req.Model, "error", lastErr)
		auditChat(s.auditLogger, r.Context(), req, nil, lastErr, start)
		if allRateLimited {
			w.Header().Set("Retry-After", strconv.Itoa(int(maxRetryAfter.Seconds())))
			writeError(w, http.StatusTooManyRequests, "rate_limit_exceeded", fmt.Sprintf("all providers are rate limited, retry after %s", maxRetryAfter.Round(time.Second)))
			return
		}
		if errors.Is(lastErr, provider.ErrQueueFull) {
			writeError(w, http.StatusServiceUnavailable, "provider_busy", "all providers are at capacity, retry later")
			return
		}
		writeError(w, http.StatusBadGateway, "upstream_error", fmt.Sprintf("all providers failed: %v", lastErr))
		return
	}
	auditChat(s.auditLogger, r.Context(), req, &resp, nil, start)

	out, err := translator.ResponseToOpenAI(resp)
	if err != nil {
		slog.Error("failed to encode response", "error", err)
		writeError(w, http.StatusInternalServerError, "internal_error", "failed to encode response")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(out)
}

// handleStream writes a Server-Sent Events response, forwarding chunks from the provider.
// Tries each candidate in order until one succeeds (before headers are committed).
func (s *Server) handleStream(w http.ResponseWriter, r *http.Request, candidates []provider.Candidate, req domain.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "internal_error", "streaming not supported by server")
		return
	}

	streamStart := time.Now()
	var ch <-chan domain.Chunk
	var lastErr error
	var successProviderName string
	allRateLimited := len(candidates) > 0
	var maxRetryAfter time.Duration
	for _, c := range candidates {
		req.Model = c.ModelID
		var err error
		ch, err = c.Provider.ChatStream(r.Context(), req)
		if err == nil {
			allRateLimited = false
			successProviderName = c.Provider.Name()
			break
		}
		if s.metrics != nil {
			s.metrics.RecordRequest(req.Model, c.Provider.Name(), "error", time.Since(streamStart).Milliseconds())
		}
		slog.Warn("provider stream error, trying next", "model", req.Model, "provider", c.Provider.Name(), "error", err)
		lastErr = err
		ch = nil
		var rl *provider.RateLimitError
		if errors.As(err, &rl) {
			if rl.RetryAfter > maxRetryAfter {
				maxRetryAfter = rl.RetryAfter
			}
		} else {
			allRateLimited = false
		}
	}
	if ch == nil {
		if r.Context().Err() != nil {
			// Client disconnected before any candidate accepted the stream.
			auditClientDisconnect(s.auditLogger, r.Context(), req.Model, streamStart)
			w.WriteHeader(499)
			return
		}
		slog.Error("all providers failed for stream", "model", req.Model, "error", lastErr)
		if allRateLimited {
			w.Header().Set("Retry-After", strconv.Itoa(int(maxRetryAfter.Seconds())))
			writeError(w, http.StatusTooManyRequests, "rate_limit_exceeded", fmt.Sprintf("all providers are rate limited, retry after %s", maxRetryAfter.Round(time.Second)))
			return
		}
		if errors.Is(lastErr, provider.ErrQueueFull) {
			writeError(w, http.StatusServiceUnavailable, "provider_busy", "all providers are at capacity, retry later")
			return
		}
		writeError(w, http.StatusBadGateway, "upstream_error", fmt.Sprintf("all providers failed: %v", lastErr))
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher.Flush() // send headers to client immediately so Do() can return
	auditStreamStart(s.auditLogger, r.Context(), req)

	start := time.Now()
	var (
		lastUsage        *domain.Usage
		lastFinishReason string
		firstTokenAt     time.Time
		contentBuf       strings.Builder
	)
	recordStreamEnd := func() {
		durationMs := time.Since(streamStart).Milliseconds()
		if s.metrics != nil {
			s.metrics.RecordRequest(req.Model, successProviderName, "success", durationMs)
			if lastUsage != nil {
				s.metrics.RecordTokens(successProviderName, req.Model,
					lastUsage.PromptTokens, lastUsage.CompletionTokens, lastUsage.CachedTokens)
			}
		}
		var ttftMs int64
		if !firstTokenAt.IsZero() {
			ttftMs = firstTokenAt.Sub(start).Milliseconds()
		}
		auditStreamEnd(s.auditLogger, r.Context(), req.Model, start, lastUsage, lastFinishReason, ttftMs, contentBuf.String())
	}

	for chunk := range ch {
		if chunk.Usage != nil {
			lastUsage = chunk.Usage
		}
		if chunk.FinishReason != "" {
			lastFinishReason = chunk.FinishReason
		}
		if firstTokenAt.IsZero() && (chunk.Delta != "" || chunk.ThinkingDelta != "" || chunk.ToolCallName != "") {
			firstTokenAt = time.Now()
		}
		if chunk.Delta != "" {
			contentBuf.WriteString(chunk.Delta)
		}
		if chunk.Done {
			fmt.Fprint(w, "data: [DONE]\n\n")
			flusher.Flush()
			recordStreamEnd()
			return
		}
		data, err := translator.ChunkToOpenAI(chunk)
		if err != nil {
			slog.Error("failed to encode chunk", "error", err)
			continue
		}
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
	recordStreamEnd()
}

// handleModels handles GET /v1/models.
func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "only GET is supported")
		return
	}

	out, err := translator.ModelsToOpenAI(s.registry.Models())
	if err != nil {
		slog.Error("failed to encode models", "error", err)
		writeError(w, http.StatusInternalServerError, "internal_error", "failed to encode models")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(out)
}

// handleHealthz handles GET /healthz.
func (s *Server) handleHealthz(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "only GET is supported")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`{"status":"ok"}`))
}

// handleMetrics handles GET /metrics — Prometheus text exposition format.
func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "only GET is supported")
		return
	}
	if s.metrics == nil {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	s.metrics.WritePrometheus(w)
}

// writeError writes an OpenAI-format error response.
func writeError(w http.ResponseWriter, status int, errType, message string) {
	body, _ := json.Marshal(map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    errType,
		},
	})
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	w.Write(body)
}
