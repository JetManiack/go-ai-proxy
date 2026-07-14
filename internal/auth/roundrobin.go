package auth

import (
	"context"
	"errors"
	"sync/atomic"
)

// RoundRobinAuthenticator cycles through a list of API keys on each GetToken call.
// The counter is atomic so concurrent calls are safe without locks.
type RoundRobinAuthenticator struct {
	keys    []string
	counter atomic.Uint64
}

// NewRoundRobin returns an Authenticator that cycles through keys in order.
// Passing nil or an empty slice causes every GetToken call to return an error.
func NewRoundRobin(keys []string) *RoundRobinAuthenticator {
	return &RoundRobinAuthenticator{keys: keys}
}

func (r *RoundRobinAuthenticator) GetToken(_ context.Context) (string, error) {
	if len(r.keys) == 0 {
		return "", errors.New("auth: round-robin authenticator has no keys configured")
	}
	idx := r.counter.Add(1) - 1
	return r.keys[idx%uint64(len(r.keys))], nil
}
