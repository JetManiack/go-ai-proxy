package auth_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/JetManiack/go-ai-proxy/internal/auth"
)

// writeTokenFile writes a tokenData-compatible JSON file at path.
func writeTokenFile(t *testing.T, path string, accessToken, refreshToken string, expiresAt time.Time) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		t.Fatal(err)
	}
	data, _ := json.Marshal(map[string]any{
		"access_token":  accessToken,
		"refresh_token": refreshToken,
		"expires_at":    expiresAt,
	})
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
}

func TestOAuth_ValidTokenLoadedFromDisk(t *testing.T) {
	dir := t.TempDir()
	tokenFile := filepath.Join(dir, "token.json")
	expiresAt := time.Now().Add(time.Hour)
	writeTokenFile(t, tokenFile, "valid-access-token", "refresh-token", expiresAt)

	a := auth.NewOAuth(auth.OAuthConfig{
		TokenFile: tokenFile,
		AuthURL:   "http://example.com/auth",
		TokenURL:  "http://example.com/token",
		ClientID:  "client-id",
	})

	token, err := a.GetToken(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if token != "valid-access-token" {
		t.Errorf("got %q, want %q", token, "valid-access-token")
	}
}

func TestOAuth_ExpiredTokenTriggersRefresh(t *testing.T) {
	dir := t.TempDir()
	tokenFile := filepath.Join(dir, "token.json")
	expiresAt := time.Now().Add(-time.Minute) // expired
	writeTokenFile(t, tokenFile, "old-token", "my-refresh-token", expiresAt)

	refreshCalled := false
	tokenSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.FormValue("grant_type") != "refresh_token" {
			t.Errorf("expected refresh_token grant, got %q", r.FormValue("grant_type"))
		}
		if r.FormValue("refresh_token") != "my-refresh-token" {
			t.Errorf("unexpected refresh_token: %q", r.FormValue("refresh_token"))
		}
		refreshCalled = true
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"access_token":  "new-access-token",
			"refresh_token": "new-refresh-token",
			"expires_in":    3600,
		})
	}))
	defer tokenSrv.Close()

	a := auth.NewOAuth(auth.OAuthConfig{
		TokenFile: tokenFile,
		AuthURL:   "http://example.com/auth",
		TokenURL:  tokenSrv.URL,
		ClientID:  "client-id",
	})

	token, err := a.GetToken(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !refreshCalled {
		t.Error("expected refresh to be called")
	}
	if token != "new-access-token" {
		t.Errorf("got %q, want new-access-token", token)
	}

	// Persisted token should also be updated.
	data, _ := os.ReadFile(tokenFile)
	var saved map[string]any
	json.Unmarshal(data, &saved)
	if saved["access_token"] != "new-access-token" {
		t.Errorf("saved token: got %v", saved["access_token"])
	}
}

func TestOAuth_RefreshedTokenCachedInMemory(t *testing.T) {
	dir := t.TempDir()
	tokenFile := filepath.Join(dir, "token.json")
	expiresAt := time.Now().Add(-time.Minute)
	writeTokenFile(t, tokenFile, "old", "refresh-tok", expiresAt)

	callCount := 0
	tokenSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"access_token":  "fresh-token",
			"refresh_token": "r2",
			"expires_in":    3600,
		})
	}))
	defer tokenSrv.Close()

	a := auth.NewOAuth(auth.OAuthConfig{
		TokenFile: tokenFile,
		AuthURL:   "http://example.com/auth",
		TokenURL:  tokenSrv.URL,
		ClientID:  "cid",
	})

	a.GetToken(context.Background())
	a.GetToken(context.Background()) // second call should use memory cache

	if callCount != 1 {
		t.Errorf("expected 1 refresh call, got %d", callCount)
	}
}
