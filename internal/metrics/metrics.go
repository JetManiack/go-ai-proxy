// Package metrics provides in-process counters and a Prometheus text format exporter.
// No external dependencies — counters use sync/atomic, output is hand-formatted.
package metrics

import (
	"fmt"
	"io"
	"sync"
	"sync/atomic"
)

type requestKey struct{ model, provider, status string }
type durationKey struct{ model, provider string }
type tokenKey struct{ provider, model, tokenType string }

// Metrics holds thread-safe counters for request throughput, latency, and token usage.
type Metrics struct {
	mu       sync.Mutex
	requests map[requestKey]*atomic.Int64
	duration map[durationKey]*atomic.Int64
	tokens   map[tokenKey]*atomic.Int64
}

// New returns an empty Metrics instance.
func New() *Metrics {
	return &Metrics{
		requests: map[requestKey]*atomic.Int64{},
		duration: map[durationKey]*atomic.Int64{},
		tokens:   map[tokenKey]*atomic.Int64{},
	}
}

func (m *Metrics) reqCounter(k requestKey) *atomic.Int64 {
	m.mu.Lock()
	c, ok := m.requests[k]
	if !ok {
		c = new(atomic.Int64)
		m.requests[k] = c
	}
	m.mu.Unlock()
	return c
}

func (m *Metrics) durCounter(k durationKey) *atomic.Int64 {
	m.mu.Lock()
	c, ok := m.duration[k]
	if !ok {
		c = new(atomic.Int64)
		m.duration[k] = c
	}
	m.mu.Unlock()
	return c
}

func (m *Metrics) tokCounter(k tokenKey) *atomic.Int64 {
	m.mu.Lock()
	c, ok := m.tokens[k]
	if !ok {
		c = new(atomic.Int64)
		m.tokens[k] = c
	}
	m.mu.Unlock()
	return c
}

// RecordRequest records a completed request (success or error) with its duration.
func (m *Metrics) RecordRequest(model, provider, status string, durationMs int64) {
	m.reqCounter(requestKey{model, provider, status}).Add(1)
	m.durCounter(durationKey{model, provider}).Add(durationMs)
}

// RecordTokens records token usage for a successful request.
// cachedTokens is the prefix-cache hit count; 0 means no cache hit or not reported.
func (m *Metrics) RecordTokens(provider, model string, promptTokens, completionTokens, cachedTokens int) {
	m.tokCounter(tokenKey{provider, model, "prompt"}).Add(int64(promptTokens))
	m.tokCounter(tokenKey{provider, model, "completion"}).Add(int64(completionTokens))
	if cachedTokens > 0 {
		m.tokCounter(tokenKey{provider, model, "cached"}).Add(int64(cachedTokens))
	}
}

// WritePrometheus writes all metrics in Prometheus text exposition format to w.
func (m *Metrics) WritePrometheus(w io.Writer) {
	m.mu.Lock()
	// snapshot maps to release lock quickly
	reqs := make(map[requestKey]int64, len(m.requests))
	for k, c := range m.requests {
		reqs[k] = c.Load()
	}
	durs := make(map[durationKey]int64, len(m.duration))
	for k, c := range m.duration {
		durs[k] = c.Load()
	}
	toks := make(map[tokenKey]int64, len(m.tokens))
	for k, c := range m.tokens {
		toks[k] = c.Load()
	}
	m.mu.Unlock()

	if len(reqs) > 0 {
		fmt.Fprintln(w, "# HELP gap_requests_total Total number of requests handled")
		fmt.Fprintln(w, "# TYPE gap_requests_total counter")
		for k, v := range reqs {
			fmt.Fprintf(w, "gap_requests_total{model=%q,provider=%q,status=%q} %d\n", k.model, k.provider, k.status, v)
		}
	}

	if len(durs) > 0 {
		fmt.Fprintln(w, "# HELP gap_request_duration_ms_total Sum of request durations in milliseconds")
		fmt.Fprintln(w, "# TYPE gap_request_duration_ms_total counter")
		for k, v := range durs {
			fmt.Fprintf(w, "gap_request_duration_ms_total{model=%q,provider=%q} %d\n", k.model, k.provider, v)
		}
	}

	if len(toks) > 0 {
		fmt.Fprintln(w, "# HELP gap_tokens_total Total tokens processed (billing)")
		fmt.Fprintln(w, "# TYPE gap_tokens_total counter")
		for k, v := range toks {
			fmt.Fprintf(w, "gap_tokens_total{provider=%q,model=%q,type=%q} %d\n", k.provider, k.model, k.tokenType, v)
		}
	}
}
