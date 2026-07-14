package provider

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

const defaultRetryAfter = 60 * time.Second

// RateLimitError is returned when an upstream provider responds with HTTP 429.
// BoundedProvider catches it, marks the provider as cooling down, and re-returns it.
// The server returns 429 to the client when all candidates are rate-limited.
type RateLimitError struct {
	RetryAfter time.Duration
}

func (e *RateLimitError) Error() string {
	return fmt.Sprintf("upstream rate limit: retry after %s", e.RetryAfter.Round(time.Second))
}

// ParseRetryAfter parses an HTTP Retry-After header value (delay-seconds or HTTP-date).
// Returns defaultRetryAfter (60s) if the header is absent or unparseable.
func ParseRetryAfter(header string) time.Duration {
	if header == "" {
		return defaultRetryAfter
	}
	if n, err := strconv.Atoi(strings.TrimSpace(header)); err == nil && n > 0 {
		return time.Duration(n) * time.Second
	}
	if t, err := http.ParseTime(header); err == nil {
		if d := time.Until(t); d > 0 {
			return d
		}
	}
	return defaultRetryAfter
}
