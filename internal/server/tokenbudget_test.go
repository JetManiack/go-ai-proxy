package server_test

import (
	"context"
	"testing"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
	"github.com/JetManiack/go-ai-proxy/internal/server"
	"github.com/JetManiack/go-ai-proxy/internal/testutil"
)

func ptr(n int) *int { return &n }

func TestTokenBudget_UnderLimit_Allowed(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	cfg := server.TokenBudgetConfig{Default: 1000}
	srv := newTestServerOpts(t, []server.Option{server.WithTokenBudget(cfg)}, fp)
	defer srv.Close()

	resp := chatRequest(t, srv, map[string]any{
		"model":      "m",
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
		"max_tokens": 500,
	})
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Errorf("status: got %d, want 200", resp.StatusCode)
	}
}

func TestTokenBudget_OverLimit_Rejected(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	cfg := server.TokenBudgetConfig{Default: 1000}
	srv := newTestServerOpts(t, []server.Option{server.WithTokenBudget(cfg)}, fp)
	defer srv.Close()

	resp := chatRequest(t, srv, map[string]any{
		"model":      "m",
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
		"max_tokens": 2000,
	})
	resp.Body.Close()
	if resp.StatusCode != 400 {
		t.Errorf("status: got %d, want 400", resp.StatusCode)
	}
}

func TestTokenBudget_AtLimit_Allowed(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	cfg := server.TokenBudgetConfig{Default: 1000}
	srv := newTestServerOpts(t, []server.Option{server.WithTokenBudget(cfg)}, fp)
	defer srv.Close()

	resp := chatRequest(t, srv, map[string]any{
		"model":      "m",
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
		"max_tokens": 1000,
	})
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Errorf("status: got %d, want 200", resp.StatusCode)
	}
}

func TestTokenBudget_NoMaxTokens_Allowed(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	cfg := server.TokenBudgetConfig{Default: 100}
	srv := newTestServerOpts(t, []server.Option{server.WithTokenBudget(cfg)}, fp)
	defer srv.Close()

	// request without max_tokens — must pass through regardless of budget
	resp := chatRequest(t, srv, map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Errorf("status: got %d, want 200", resp.StatusCode)
	}
}

func TestTokenBudget_ModelOverride_TakesPrecedence(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "big-model"})
	cfg := server.TokenBudgetConfig{
		Default: 500,
		Models:  map[string]int{"big-model": 8000},
	}
	srv := newTestServerOpts(t, []server.Option{server.WithTokenBudget(cfg)}, fp)
	defer srv.Close()

	// 2000 > default(500) but ≤ model override(8000) → allowed
	resp := chatRequest(t, srv, map[string]any{
		"model":      "big-model",
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
		"max_tokens": 2000,
	})
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Errorf("status: got %d, want 200", resp.StatusCode)
	}
}

func TestTokenBudget_ModelOverride_StillRejectsOverOverride(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "small-model"})
	cfg := server.TokenBudgetConfig{
		Default: 5000,
		Models:  map[string]int{"small-model": 256},
	}
	srv := newTestServerOpts(t, []server.Option{server.WithTokenBudget(cfg)}, fp)
	defer srv.Close()

	// 1000 ≤ default(5000) but > model override(256) → rejected
	resp := chatRequest(t, srv, map[string]any{
		"model":      "small-model",
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
		"max_tokens": 1000,
	})
	resp.Body.Close()
	if resp.StatusCode != 400 {
		t.Errorf("status: got %d, want 400", resp.StatusCode)
	}
}

func TestTokenBudget_NoConfig_Passthrough(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	// no WithTokenBudget — all requests pass
	srv := newTestServer(t, fp)
	defer srv.Close()

	resp := chatRequest(t, srv, map[string]any{
		"model":      "m",
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
		"max_tokens": 999999,
	})
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Errorf("status: got %d, want 200", resp.StatusCode)
	}
}

func TestTokenBudget_NoUpstreamCallOnReject(t *testing.T) {
	called := false
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	fp.ChatFunc = func(_ context.Context, _ domain.Request) (domain.Response, error) {
		called = true
		return domain.Response{}, nil
	}
	cfg := server.TokenBudgetConfig{Default: 100}
	srv := newTestServerOpts(t, []server.Option{server.WithTokenBudget(cfg)}, fp)
	defer srv.Close()

	resp := chatRequest(t, srv, map[string]any{
		"model":      "m",
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
		"max_tokens": 9999,
	})
	resp.Body.Close()

	if called {
		t.Error("provider must not be called when token budget is exceeded")
	}
}
