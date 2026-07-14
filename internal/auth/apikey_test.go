package auth_test

import (
	"context"
	"testing"

	"github.com/JetManiack/go-ai-proxy/internal/auth"
)

func TestAPIKeyAuthenticator_ReturnsKey(t *testing.T) {
	a := auth.NewAPIKey("sk-test-123")
	token, err := a.GetToken(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if token != "sk-test-123" {
		t.Errorf("got %q, want %q", token, "sk-test-123")
	}
}

func TestAPIKeyAuthenticator_EmptyKey(t *testing.T) {
	a := auth.NewAPIKey("")
	token, err := a.GetToken(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if token != "" {
		t.Errorf("expected empty token, got %q", token)
	}
}
