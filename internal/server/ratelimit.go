package server

import (
	"crypto/sha256"
	"encoding/hex"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"golang.org/x/time/rate"
)

// RateLimitConfig configures the token-bucket rate limiter.
type RateLimitConfig struct {
	RPS        float64 // sustained request rate (tokens per second)
	Burst      int     // bucket capacity (max burst)
	PerCaller  bool    // false = one global limiter; true = per Authorization value or IP
	TrustProxy bool    // honour X-Forwarded-For / X-Real-IP for IP-based per-caller limiting
}

// WithRateLimit adds a token-bucket rate limiter middleware.
// Requests that exceed the limit receive 429 Too Many Requests.
func WithRateLimit(cfg RateLimitConfig) Option {
	return func(s *Server) { s.rateLimitCfg = &cfg }
}

// rateLimitMiddleware wraps next with token-bucket rate limiting.
func rateLimitMiddleware(next http.Handler, cfg RateLimitConfig) http.Handler {
	if cfg.PerCaller {
		return perCallerLimiter(next, cfg)
	}
	return globalLimiter(next, cfg)
}

func globalLimiter(next http.Handler, cfg RateLimitConfig) http.Handler {
	lim := rate.NewLimiter(rate.Limit(cfg.RPS), cfg.Burst)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !lim.Allow() {
			writeError(w, http.StatusTooManyRequests, "rate_limit_exceeded", "too many requests")
			return
		}
		next.ServeHTTP(w, r)
	})
}

type callerEntry struct {
	lim     *rate.Limiter
	lastUse time.Time
}

// maxCallerEntries is the soft cap on tracked callers. When exceeded, stale
// entries (idle > 10 min) are evicted before inserting a new one.
const maxCallerEntries = 10_000

func perCallerLimiter(next http.Handler, cfg RateLimitConfig) http.Handler {
	var mu sync.Mutex
	limiters := map[string]*callerEntry{}

	getLimiter := func(key string) *rate.Limiter {
		mu.Lock()
		defer mu.Unlock()

		if e, ok := limiters[key]; ok {
			e.lastUse = time.Now()
			return e.lim
		}

		// Evict idle entries when approaching the cap to prevent unbounded growth.
		if len(limiters) >= maxCallerEntries {
			cutoff := time.Now().Add(-10 * time.Minute)
			for k, e := range limiters {
				if e.lastUse.Before(cutoff) {
					delete(limiters, k)
				}
			}
			// If still at cap, return a fresh limiter without caching it so the
			// existing buckets for known callers remain intact.
			if len(limiters) >= maxCallerEntries {
				return rate.NewLimiter(rate.Limit(cfg.RPS), cfg.Burst)
			}
		}

		e := &callerEntry{
			lim:     rate.NewLimiter(rate.Limit(cfg.RPS), cfg.Burst),
			lastUse: time.Now(),
		}
		limiters[key] = e
		return e.lim
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := callerKey(r, cfg.TrustProxy)
		if !getLimiter(key).Allow() {
			writeError(w, http.StatusTooManyRequests, "rate_limit_exceeded", "too many requests")
			return
		}
		next.ServeHTTP(w, r)
	})
}

// callerKey derives a rate-limit bucket key for the request.
// When an Authorization header is present its SHA-256 is used so the raw token
// is never stored in the limiters map.
// When TrustProxy is true, X-Forwarded-For / X-Real-IP are consulted for the
// client IP before falling back to RemoteAddr.
func callerKey(r *http.Request, trustProxy bool) string {
	if auth := r.Header.Get("Authorization"); auth != "" {
		h := sha256.Sum256([]byte(auth))
		return hex.EncodeToString(h[:16])
	}
	if trustProxy {
		if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
			if ip := strings.TrimSpace(strings.SplitN(xff, ",", 2)[0]); ip != "" {
				return ip
			}
		}
		if xri := r.Header.Get("X-Real-IP"); xri != "" {
			return strings.TrimSpace(xri)
		}
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return strings.TrimSpace(r.RemoteAddr)
	}
	return host
}
