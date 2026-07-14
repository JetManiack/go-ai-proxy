package server_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
	"github.com/JetManiack/go-ai-proxy/internal/metrics"
	"github.com/JetManiack/go-ai-proxy/internal/server"
	"github.com/JetManiack/go-ai-proxy/internal/testutil"
)

func TestMetrics_RecordedOnSuccess(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	fp.NameVal = "prov"
	fp.ChatFunc = func(_ context.Context, _ domain.Request) (domain.Response, error) {
		return domain.Response{
			Message: domain.Message{Role: "assistant", Content: "ok"},
			Usage:   domain.Usage{PromptTokens: 10, CompletionTokens: 5},
		}, nil
	}

	m := metrics.New()
	ts := newTestServerOpts(t, []server.Option{server.WithMetrics(m)}, fp)
	defer ts.Close()

	resp := chatRequest(t, ts, map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	out := metricsOutput(t, ts)
	if !strings.Contains(out, `gap_requests_total{model="m",provider="prov",status="success"}`) {
		t.Errorf("missing success counter in metrics:\n%s", out)
	}
	if !strings.Contains(out, `gap_tokens_total{provider="prov",model="m",type="prompt"} 10`) {
		t.Errorf("missing prompt token counter in metrics:\n%s", out)
	}
	if !strings.Contains(out, `gap_tokens_total{provider="prov",model="m",type="completion"} 5`) {
		t.Errorf("missing completion token counter in metrics:\n%s", out)
	}
}

func TestMetrics_RecordedOnProviderError(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	fp.NameVal = "prov"
	fp.ChatFunc = func(_ context.Context, _ domain.Request) (domain.Response, error) {
		return domain.Response{}, errors.New("upstream error")
	}

	m := metrics.New()
	ts := newTestServerOpts(t, []server.Option{server.WithMetrics(m)}, fp)
	defer ts.Close()

	resp := chatRequest(t, ts, map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("expected 502, got %d", resp.StatusCode)
	}

	out := metricsOutput(t, ts)
	if !strings.Contains(out, `gap_requests_total{model="m",provider="prov",status="error"}`) {
		t.Errorf("missing error counter in metrics:\n%s", out)
	}
}

func TestMetrics_EndpointNoContent_WhenDisabled(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	ts := newTestServerOpts(t, nil, fp)
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/metrics")
	if err != nil {
		t.Fatalf("GET /metrics: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent {
		t.Errorf("expected 204 when metrics disabled, got %d", resp.StatusCode)
	}
}

func metricsOutput(t *testing.T, ts *httptest.Server) string {
	t.Helper()
	resp, err := http.Get(ts.URL + "/metrics")
	if err != nil {
		t.Fatalf("GET /metrics: %v", err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return string(b)
}
