package provider_test

import (
	"errors"
	"fmt"
	"testing"

	"github.com/JetManiack/go-ai-proxy/internal/provider"
)

func TestUpstreamError_Error(t *testing.T) {
	err := &provider.UpstreamError{StatusCode: 400, Body: "input exceeds 2048 tokens"}
	got := err.Error()
	if got != "upstream returned 400: input exceeds 2048 tokens" {
		t.Errorf("Error(): got %q", got)
	}
}

func TestUpstreamError_ErrorsAs(t *testing.T) {
	var wrapped error = fmt.Errorf("wrapping: %w", &provider.UpstreamError{StatusCode: 413, Body: "too big"})
	var ue *provider.UpstreamError
	if !errors.As(wrapped, &ue) {
		t.Fatal("expected errors.As to unwrap *provider.UpstreamError")
	}
	if ue.StatusCode != 413 {
		t.Errorf("StatusCode: got %d, want 413", ue.StatusCode)
	}
}
