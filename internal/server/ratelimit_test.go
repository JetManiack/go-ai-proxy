package server_test

import (
	"fmt"
	"net/http"
	"sync"
	"testing"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
	"github.com/JetManiack/go-ai-proxy/internal/server"
	"github.com/JetManiack/go-ai-proxy/internal/testutil"
)

func TestRateLimit_Global_AllowsUnderBurst(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	cfg := server.RateLimitConfig{RPS: 100, Burst: 5}
	srv := newTestServerOpts(t, []server.Option{server.WithRateLimit(cfg)}, fp)
	defer srv.Close()

	for i := range 5 {
		resp, err := http.Get(srv.URL + "/healthz")
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Errorf("request %d: got %d, want 200", i+1, resp.StatusCode)
		}
	}
}

func TestRateLimit_Global_BlocksOverBurst(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	// burst=3, rps very low so tokens don't refill during the test
	cfg := server.RateLimitConfig{RPS: 0.001, Burst: 3}
	srv := newTestServerOpts(t, []server.Option{server.WithRateLimit(cfg)}, fp)
	defer srv.Close()

	var got429 bool
	for range 10 {
		resp, err := http.Get(srv.URL + "/healthz")
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode == http.StatusTooManyRequests {
			got429 = true
			break
		}
	}
	if !got429 {
		t.Error("expected 429 after burst exhausted")
	}
}

func TestRateLimit_PerCaller_SeparateLimitsPerKey(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	// burst=3 per caller — two callers together make 6 requests but each stays under limit
	cfg := server.RateLimitConfig{RPS: 0.001, Burst: 3, PerCaller: true}
	srv := newTestServerOpts(t, []server.Option{server.WithRateLimit(cfg)}, fp)
	defer srv.Close()

	sendWith := func(key string) int {
		req, _ := http.NewRequest(http.MethodGet, srv.URL+"/healthz", nil)
		if key != "" {
			req.Header.Set("Authorization", "Bearer "+key)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		return resp.StatusCode
	}

	// 3 requests each from two different callers — all should pass
	for i := range 3 {
		if sc := sendWith("caller-a"); sc != http.StatusOK {
			t.Errorf("caller-a request %d: got %d, want 200", i+1, sc)
		}
		if sc := sendWith("caller-b"); sc != http.StatusOK {
			t.Errorf("caller-b request %d: got %d, want 200", i+1, sc)
		}
	}
}

func TestRateLimit_PerCaller_SameCallerBlocked(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	cfg := server.RateLimitConfig{RPS: 0.001, Burst: 2, PerCaller: true}
	srv := newTestServerOpts(t, []server.Option{server.WithRateLimit(cfg)}, fp)
	defer srv.Close()

	sendWith := func(key string) int {
		req, _ := http.NewRequest(http.MethodGet, srv.URL+"/healthz", nil)
		req.Header.Set("Authorization", "Bearer "+key)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		return resp.StatusCode
	}

	var got429 bool
	for range 10 {
		if sendWith("same-key") == http.StatusTooManyRequests {
			got429 = true
			break
		}
	}
	if !got429 {
		t.Error("expected 429 after per-caller burst exhausted")
	}
}

func TestRateLimit_Returns429WithJSON(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	cfg := server.RateLimitConfig{RPS: 0.001, Burst: 0}
	srv := newTestServerOpts(t, []server.Option{server.WithRateLimit(cfg)}, fp)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("status: got %d, want 429", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type: got %q, want application/json", ct)
	}
}

func TestRateLimit_Concurrent_NoPanic(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	cfg := server.RateLimitConfig{RPS: 1000, Burst: 100, PerCaller: true}
	srv := newTestServerOpts(t, []server.Option{server.WithRateLimit(cfg)}, fp)
	defer srv.Close()

	var wg sync.WaitGroup
	for i := range 20 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			req, _ := http.NewRequest(http.MethodGet, srv.URL+"/healthz", nil)
			req.Header.Set("Authorization", fmt.Sprintf("Bearer key-%d", i%5))
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				return
			}
			resp.Body.Close()
		}(i)
	}
	wg.Wait()
}
