// Package auth provides token-based authenticator implementations.
package auth

import "context"

// Authenticator provides a bearer token for upstream API calls.
type Authenticator interface {
	GetToken(ctx context.Context) (string, error)
}
