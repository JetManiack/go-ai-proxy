package provider_test

import (
	"net/http"
	"testing"
	"time"

	"github.com/JetManiack/go-ai-proxy/internal/provider"
)

func TestParseRetryAfter_Empty(t *testing.T) {
	if d := provider.ParseRetryAfter(""); d != 60*time.Second {
		t.Errorf("empty: got %v, want 60s", d)
	}
}

func TestParseRetryAfter_Seconds(t *testing.T) {
	if d := provider.ParseRetryAfter("30"); d != 30*time.Second {
		t.Errorf("seconds: got %v, want 30s", d)
	}
}

func TestParseRetryAfter_HTTPDate(t *testing.T) {
	future := time.Now().Add(45 * time.Second)
	d := provider.ParseRetryAfter(future.UTC().Format(http.TimeFormat))
	if d < 43*time.Second || d > 47*time.Second {
		t.Errorf("http-date: got %v, want ~45s", d)
	}
}

func TestParseRetryAfter_PastDate_UsesDefault(t *testing.T) {
	past := time.Now().Add(-10 * time.Second)
	if d := provider.ParseRetryAfter(past.UTC().Format(http.TimeFormat)); d != 60*time.Second {
		t.Errorf("past date: got %v, want 60s (default)", d)
	}
}

func TestParseRetryAfter_Invalid_UsesDefault(t *testing.T) {
	if d := provider.ParseRetryAfter("not-a-value"); d != 60*time.Second {
		t.Errorf("invalid: got %v, want 60s", d)
	}
}
