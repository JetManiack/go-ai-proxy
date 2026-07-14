package auth

import "context"

// APIKeyAuthenticator returns a static API key on every call.
type APIKeyAuthenticator struct {
	key string
}

// NewAPIKey returns an Authenticator that always returns key.
func NewAPIKey(key string) *APIKeyAuthenticator {
	return &APIKeyAuthenticator{key: key}
}

func (a *APIKeyAuthenticator) GetToken(_ context.Context) (string, error) {
	return a.key, nil
}
