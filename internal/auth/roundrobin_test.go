package auth_test

import (
	"context"
	"sync"
	"testing"

	"github.com/JetManiack/go-ai-proxy/internal/auth"
)

func TestRoundRobin_CyclesInOrder(t *testing.T) {
	keys := []string{"key-a", "key-b", "key-c"}
	a := auth.NewRoundRobin(keys)

	for cycle := range 3 {
		for i, want := range keys {
			got, err := a.GetToken(context.Background())
			if err != nil {
				t.Fatalf("cycle %d call %d: unexpected error: %v", cycle, i, err)
			}
			if got != want {
				t.Errorf("cycle %d call %d: got %q, want %q", cycle, i, got, want)
			}
		}
	}
}

func TestRoundRobin_SingleKey_AlwaysReturnsSame(t *testing.T) {
	a := auth.NewRoundRobin([]string{"only-key"})
	for range 5 {
		got, err := a.GetToken(context.Background())
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != "only-key" {
			t.Errorf("got %q, want only-key", got)
		}
	}
}

func TestRoundRobin_EmptyKeys_ReturnsError(t *testing.T) {
	a := auth.NewRoundRobin(nil)
	_, err := a.GetToken(context.Background())
	if err == nil {
		t.Error("expected error for empty key list")
	}
}

func TestRoundRobin_Concurrent_NoRace(t *testing.T) {
	keys := []string{"k1", "k2", "k3"}
	a := auth.NewRoundRobin(keys)

	var wg sync.WaitGroup
	for range 100 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			tok, err := a.GetToken(context.Background())
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			found := false
			for _, k := range keys {
				if tok == k {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("unexpected token: %q", tok)
			}
		}()
	}
	wg.Wait()
}
