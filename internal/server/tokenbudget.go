package server

// TokenBudgetConfig sets per-model and global max_tokens limits.
// Requests whose max_tokens field exceeds the applicable limit are rejected
// with 400 before any upstream call is made.
// Requests that do not set max_tokens are always allowed through.
type TokenBudgetConfig struct {
	Default int            // applies when no model-specific limit is configured; 0 = no limit
	Models  map[string]int // model-specific overrides; take precedence over Default
}

// limitFor returns the effective max_tokens limit for model.
// 0 means no limit is configured.
func (c *TokenBudgetConfig) limitFor(model string) int {
	if c == nil {
		return 0
	}
	if limit, ok := c.Models[model]; ok {
		return limit
	}
	return c.Default
}

// WithTokenBudget rejects requests whose max_tokens exceeds the configured
// limit before any upstream call is made.
func WithTokenBudget(cfg TokenBudgetConfig) Option {
	return func(s *Server) { s.tokenBudget = &cfg }
}
