package server_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"testing"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
	"github.com/JetManiack/go-ai-proxy/internal/server"
	"github.com/JetManiack/go-ai-proxy/internal/testutil"
)

// captureLogger returns a *slog.Logger that writes JSON into buf.
func captureLogger(buf *bytes.Buffer) *slog.Logger {
	return slog.New(slog.NewJSONHandler(buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
}

// parseAuditEntries parses the JSON-lines written by captureLogger.
// If level is non-empty (e.g. "INFO", "DEBUG"), only entries at that level are returned.
func parseAuditEntries(t *testing.T, buf *bytes.Buffer, level string) []map[string]any {
	t.Helper()
	var entries []map[string]any
	dec := json.NewDecoder(buf)
	for dec.More() {
		var e map[string]any
		if err := dec.Decode(&e); err != nil {
			t.Fatalf("parse log entry: %v", err)
		}
		if e["msg"] != "audit" {
			continue
		}
		if level == "" || e["level"] == level {
			entries = append(entries, e)
		}
	}
	return entries
}

func TestAuditLog_Chat_Success(t *testing.T) {
	var buf bytes.Buffer
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	fp.ChatFunc = func(_ context.Context, req domain.Request) (domain.Response, error) {
		return domain.Response{
			ID:      "r1",
			Model:   req.Model,
			Message: domain.Message{Role: "assistant", Content: "hello"},
			Usage:   domain.Usage{PromptTokens: 2, CompletionTokens: 1},
		}, nil
	}

	srv := newTestServerOpts(t, []server.Option{server.WithAuditLog(captureLogger(&buf))}, fp)
	defer srv.Close()

	resp := chatRequest(t, srv, map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "ping"}},
	})
	resp.Body.Close()

	entries := parseAuditEntries(t, &buf, "INFO")
	if len(entries) != 1 {
		t.Fatalf("audit entries: got %d, want 1", len(entries))
	}
	e := entries[0]
	if e["model"] != "m" {
		t.Errorf("model: got %v", e["model"])
	}
	if e["error"] != nil {
		t.Errorf("expected no error field, got %v", e["error"])
	}
	if e["request_id"] == "" {
		t.Error("expected non-empty request_id")
	}
	if e["duration_ms"] == nil {
		t.Error("expected duration_ms field")
	}
}

func TestAuditLog_Chat_UpstreamError(t *testing.T) {
	var buf bytes.Buffer
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	fp.ChatFunc = func(_ context.Context, _ domain.Request) (domain.Response, error) {
		return domain.Response{}, errors.New("boom")
	}

	srv := newTestServerOpts(t, []server.Option{server.WithAuditLog(captureLogger(&buf))}, fp)
	defer srv.Close()

	resp := chatRequest(t, srv, map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "ping"}},
	})
	resp.Body.Close()

	entries := parseAuditEntries(t, &buf, "ERROR")
	if len(entries) != 1 {
		t.Fatalf("audit error entries: got %d, want 1", len(entries))
	}
	if entries[0]["error"] == nil {
		t.Error("expected error field in audit entry")
	}
}

func TestAuditLog_Streaming_Logged(t *testing.T) {
	var buf bytes.Buffer
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	fp.StreamFunc = func(_ context.Context, _ domain.Request) (<-chan domain.Chunk, error) {
		ch := make(chan domain.Chunk, 1)
		ch <- domain.Chunk{Done: true}
		close(ch)
		return ch, nil
	}

	srv := newTestServerOpts(t, []server.Option{server.WithAuditLog(captureLogger(&buf))}, fp)
	defer srv.Close()

	resp := chatRequest(t, srv, map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
		"stream":   true,
	})
	// drain body so server handler finishes before we check the log
	var drain bytes.Buffer
	drain.ReadFrom(resp.Body)
	resp.Body.Close()

	// expect two audit entries: stream_start and stream_end
	entries := parseAuditEntries(t, &buf, "INFO")
	if len(entries) != 2 {
		t.Fatalf("audit entries: got %d, want 2 (stream_start + stream_end)", len(entries))
	}

	start := entries[0]
	if start["model"] != "m" {
		t.Errorf("start model: got %v", start["model"])
	}
	if start["event"] != "stream_start" {
		t.Errorf("start event: got %v, want stream_start", start["event"])
	}

	end := entries[1]
	if end["event"] != "stream_end" {
		t.Errorf("end event: got %v, want stream_end", end["event"])
	}
	if end["duration_ms"] == nil {
		t.Error("expected duration_ms in stream_end entry")
	}
}

func TestAuditLog_Nil_NoOp(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	// nil logger = audit disabled, must not panic
	srv := newTestServerOpts(t, []server.Option{server.WithAuditLog(nil)}, fp)
	defer srv.Close()

	resp := chatRequest(t, srv, map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	resp.Body.Close()
}

func TestAuditLog_MultipleRequests_AllLogged(t *testing.T) {
	var buf bytes.Buffer
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})

	srv := newTestServerOpts(t, []server.Option{server.WithAuditLog(captureLogger(&buf))}, fp)
	defer srv.Close()

	for range 3 {
		resp := chatRequest(t, srv, map[string]any{
			"model":    "m",
			"messages": []map[string]any{{"role": "user", "content": "hi"}},
		})
		resp.Body.Close()
	}

	entries := parseAuditEntries(t, &buf, "INFO")
	if len(entries) != 3 {
		t.Fatalf("audit entries: got %d, want 3", len(entries))
	}
}

func TestAuditLog_Debug_ContainsPromptAndResponse(t *testing.T) {
	var buf bytes.Buffer
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	fp.ChatFunc = func(_ context.Context, req domain.Request) (domain.Response, error) {
		return domain.Response{
			ID:      "r1",
			Model:   req.Model,
			Message: domain.Message{Role: "assistant", Content: "pong"},
		}, nil
	}

	srv := newTestServerOpts(t, []server.Option{server.WithAuditLog(captureLogger(&buf))}, fp)
	defer srv.Close()

	resp := chatRequest(t, srv, map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "ping"}},
	})
	resp.Body.Close()

	dbg := parseAuditEntries(t, &buf, "DEBUG")
	if len(dbg) != 1 {
		t.Fatalf("debug audit entries: got %d, want 1", len(dbg))
	}
	d := dbg[0]
	// messages field should be present
	if d["messages"] == nil {
		t.Error("expected messages field in DEBUG entry")
	}
	// response content should be present
	if d["response"] == nil {
		t.Error("expected response field in DEBUG entry")
	}
}

func TestAuditLog_Debug_Stream_ContainsPrompt(t *testing.T) {
	var buf bytes.Buffer
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	fp.StreamFunc = func(_ context.Context, _ domain.Request) (<-chan domain.Chunk, error) {
		ch := make(chan domain.Chunk, 1)
		ch <- domain.Chunk{Done: true}
		close(ch)
		return ch, nil
	}

	srv := newTestServerOpts(t, []server.Option{server.WithAuditLog(captureLogger(&buf))}, fp)
	defer srv.Close()

	resp := chatRequest(t, srv, map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
		"stream":   true,
	})
	var drain bytes.Buffer
	drain.ReadFrom(resp.Body)
	resp.Body.Close()

	// one DEBUG entry at stream_start with request messages
	dbg := parseAuditEntries(t, &buf, "DEBUG")
	if len(dbg) != 1 {
		t.Fatalf("debug audit entries: got %d, want 1", len(dbg))
	}
	if dbg[0]["messages"] == nil {
		t.Error("expected messages field in DEBUG stream entry")
	}
}
