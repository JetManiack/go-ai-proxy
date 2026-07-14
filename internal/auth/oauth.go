package auth

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// OAuthConfig holds the endpoints and credentials for the OAuth 2.0 PKCE flow.
type OAuthConfig struct {
	AuthURL   string
	TokenURL  string
	ClientID  string
	TokenFile string
	Scopes    []string
}

// tokenData is the on-disk representation of persisted OAuth tokens.
type tokenData struct {
	AccessToken  string    `json:"access_token"`
	RefreshToken string    `json:"refresh_token"`
	ExpiresAt    time.Time `json:"expires_at"`
}

// OAuthAuthenticator performs a browser-based OAuth 2.0 PKCE flow and caches tokens.
// On first use, it opens the system browser for authorization.
// Subsequent calls return the cached token, refreshing it transparently when expired.
type OAuthAuthenticator struct {
	cfg   OAuthConfig
	mu    sync.Mutex
	token *tokenData
}

// NewOAuth returns an OAuthAuthenticator backed by cfg.
func NewOAuth(cfg OAuthConfig) *OAuthAuthenticator {
	return &OAuthAuthenticator{cfg: cfg}
}

// GetToken returns a valid access token, refreshing or re-authorizing as needed.
func (o *OAuthAuthenticator) GetToken(ctx context.Context) (string, error) {
	o.mu.Lock()
	defer o.mu.Unlock()

	if o.token == nil {
		if t, err := o.loadToken(); err == nil {
			o.token = t
		}
	}

	if o.token != nil && time.Now().Before(o.token.ExpiresAt) {
		return o.token.AccessToken, nil
	}
	if o.token != nil && o.token.RefreshToken != "" {
		if err := o.refresh(ctx); err == nil {
			return o.token.AccessToken, nil
		}
	}

	if err := o.authorize(ctx); err != nil {
		return "", err
	}
	return o.token.AccessToken, nil
}

func (o *OAuthAuthenticator) loadToken() (*tokenData, error) {
	data, err := os.ReadFile(expandPath(o.cfg.TokenFile))
	if err != nil {
		return nil, err
	}
	var t tokenData
	if err := json.Unmarshal(data, &t); err != nil {
		return nil, err
	}
	return &t, nil
}

func (o *OAuthAuthenticator) saveToken(t *tokenData) error {
	path := expandPath(o.cfg.TokenFile)
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}
	data, err := json.Marshal(t)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0600)
}

func (o *OAuthAuthenticator) refresh(ctx context.Context) error {
	return o.exchangeToken(ctx, url.Values{
		"grant_type":    {"refresh_token"},
		"refresh_token": {o.token.RefreshToken},
		"client_id":     {o.cfg.ClientID},
	})
}

func (o *OAuthAuthenticator) authorize(ctx context.Context) error {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return fmt.Errorf("oauth: start callback listener: %w", err)
	}
	callbackURL := fmt.Sprintf("http://127.0.0.1:%d/callback", ln.Addr().(*net.TCPAddr).Port)

	verifier, challenge, err := generatePKCE()
	if err != nil {
		return fmt.Errorf("oauth: generate pkce: %w", err)
	}
	state, err := randomHex(16)
	if err != nil {
		return fmt.Errorf("oauth: generate state: %w", err)
	}

	params := url.Values{
		"response_type":         {"code"},
		"client_id":             {o.cfg.ClientID},
		"redirect_uri":          {callbackURL},
		"code_challenge":        {challenge},
		"code_challenge_method": {"S256"},
		"state":                 {state},
	}
	if len(o.cfg.Scopes) > 0 {
		params.Set("scope", strings.Join(o.cfg.Scopes, " "))
	}
	authURL := o.cfg.AuthURL + "?" + params.Encode()

	fmt.Printf("Opening browser for authorization:\n%s\n", authURL)
	_ = openBrowser(authURL)

	codeCh := make(chan string, 1)
	errCh := make(chan error, 1)

	cbSrv := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			q := r.URL.Query()
			if q.Get("state") != state {
				http.Error(w, "state mismatch", http.StatusBadRequest)
				errCh <- fmt.Errorf("oauth: state mismatch in callback")
				return
			}
			code := q.Get("code")
			if code == "" {
				http.Error(w, "missing code", http.StatusBadRequest)
				errCh <- fmt.Errorf("oauth: no code in callback")
				return
			}
			fmt.Fprintln(w, "Authorization complete. You may close this tab.")
			codeCh <- code
		}),
	}
	go cbSrv.Serve(ln) //nolint:errcheck
	defer func() {
		shutCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		cbSrv.Shutdown(shutCtx) //nolint:errcheck
	}()

	var code string
	select {
	case code = <-codeCh:
	case err = <-errCh:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}

	return o.exchangeToken(ctx, url.Values{
		"grant_type":    {"authorization_code"},
		"code":          {code},
		"redirect_uri":  {callbackURL},
		"client_id":     {o.cfg.ClientID},
		"code_verifier": {verifier},
	})
}

func (o *OAuthAuthenticator) exchangeToken(ctx context.Context, vals url.Values) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.cfg.TokenURL,
		strings.NewReader(vals.Encode()))
	if err != nil {
		return fmt.Errorf("oauth: build token request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("oauth: token request: %w", err)
	}
	defer resp.Body.Close()

	var result struct {
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
		ExpiresIn    int    `json:"expires_in"`
		Error        string `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return fmt.Errorf("oauth: decode token response: %w", err)
	}
	if result.Error != "" {
		return fmt.Errorf("oauth: token error: %s", result.Error)
	}

	t := &tokenData{
		AccessToken:  result.AccessToken,
		RefreshToken: result.RefreshToken,
		ExpiresAt:    time.Now().Add(time.Duration(result.ExpiresIn) * time.Second),
	}
	o.token = t
	return o.saveToken(t)
}

// --- Helpers ---

func generatePKCE() (verifier, challenge string, err error) {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		return "", "", err
	}
	verifier = base64.RawURLEncoding.EncodeToString(b)
	sum := sha256.Sum256([]byte(verifier))
	challenge = base64.RawURLEncoding.EncodeToString(sum[:])
	return verifier, challenge, nil
}

func randomHex(n int) (string, error) {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", b), nil
}

func expandPath(p string) string {
	if strings.HasPrefix(p, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, p[2:])
		}
	}
	return p
}

func openBrowser(u string) error {
	var cmd string
	var args []string
	switch runtime.GOOS {
	case "darwin":
		cmd = "open"
	case "windows":
		cmd = "rundll32"
		args = []string{"url.dll,FileProtocolHandler"}
	default:
		cmd = "xdg-open"
	}
	return exec.Command(cmd, append(args, u)...).Start()
}
