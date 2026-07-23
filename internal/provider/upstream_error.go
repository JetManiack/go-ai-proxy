package provider

import "fmt"

// UpstreamError carries the exact HTTP status code and body an upstream
// provider returned for a non-200, non-429 response. Callers can type-assert
// via errors.As to proxy a client-caused error (4xx) verbatim instead of
// folding it into gap's own generic "all providers failed" framing — without
// gap needing to understand the upstream's request semantics itself (e.g. an
// embeddings backend rejecting an input that exceeds its token limit).
type UpstreamError struct {
	StatusCode int
	Body       string
}

func (e *UpstreamError) Error() string {
	return fmt.Sprintf("upstream returned %d: %s", e.StatusCode, e.Body)
}
