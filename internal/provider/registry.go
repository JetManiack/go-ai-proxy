// Package provider contains the provider registry and related utilities.
package provider

import (
	"context"
	"log/slog"
	"math"
	"path"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
	"golang.org/x/sync/singleflight"
)

// providerEntry pairs a provider with its optional capability and context-length overrides.
type providerEntry struct {
	provider              domain.Provider
	capabilityOverrides   map[string][]string // model ID → capabilities; nil = no overrides
	contextLengthOverrides map[string]int     // model ID → context window tokens; nil = none
}

// RegisterOption configures how a provider is registered.
type RegisterOption func(*providerEntry)

// WithCapabilities sets manual capability overrides for specific model IDs.
// Useful for providers (e.g. LM Studio) that do not advertise capabilities in /v1/models.
func WithCapabilities(caps map[string][]string) RegisterOption {
	return func(e *providerEntry) { e.capabilityOverrides = caps }
}

// WithContextLengths sets manual context-window (max_model_len) overrides for
// specific model IDs. Useful for providers that do not report max_model_len in
// /v1/models (Anthropic, LiteLLM, LM Studio, llama.cpp). The override wins over
// any provider-reported value.
func WithContextLengths(lengths map[string]int) RegisterOption {
	return func(e *providerEntry) { e.contextLengthOverrides = lengths }
}

// Registry maintains a set of registered providers and a model→provider index
// that is refreshed at startup and periodically in the background.
//
// When two providers expose the same model ID (or a matching glob pattern),
// the first registered provider wins for ProviderFor — but ProvidersFor returns
// all providers in registration order, enabling fallback chains.
type Registry struct {
	providers []providerEntry
	index     map[string]domain.Provider   // model ID → primary provider
	fallbacks map[string][]domain.Provider // model ID → ordered provider list
	globs     []string                     // glob patterns in registration order
	allModels []domain.Model
	mu        sync.RWMutex
	interval  time.Duration
	sfg       singleflight.Group // coalesces concurrent on-demand refreshes
}

// NewRegistry creates a Registry with the given model list refresh interval.
func NewRegistry(interval time.Duration) *Registry {
	return &Registry{
		interval:  interval,
		index:     make(map[string]domain.Provider),
		fallbacks: make(map[string][]domain.Provider),
	}
}


// Register adds a provider to the registry. Must be called before Start.
func (r *Registry) Register(p domain.Provider, opts ...RegisterOption) {
	entry := providerEntry{provider: p}
	for _, o := range opts {
		o(&entry)
	}
	r.providers = append(r.providers, entry)
}

// Start fetches models from all registered providers, builds the routing index,
// and launches a background goroutine that refreshes the index every r.interval.
// The background goroutine stops when ctx is cancelled.
func (r *Registry) Start(ctx context.Context) error {
	r.refresh(ctx)

	go func() {
		ticker := time.NewTicker(r.interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				r.refresh(ctx)
			}
		}
	}()

	return nil
}

// Refresh triggers an immediate model list refresh from all providers.
// Concurrent calls coalesce into a single upstream fetch via singleflight,
// so a burst of cache misses causes only one round of upstream requests.
func (r *Registry) Refresh(ctx context.Context) {
	r.sfg.Do("refresh", func() (any, error) { //nolint:errcheck
		r.refresh(ctx)
		return nil, nil
	})
}

// refresh queries all providers and rebuilds the index. Providers that fail are
// logged and skipped; the rest remain available.
// First registered provider wins on model ID conflicts.
func (r *Registry) refresh(ctx context.Context) {
	newIndex := make(map[string]domain.Provider)
	newFallbacks := make(map[string][]domain.Provider)
	var newModels []domain.Model
	var newGlobs []string

	for _, entry := range r.providers {
		models, err := entry.provider.Models(ctx)
		if err != nil {
			slog.Warn("registry: failed to fetch models from provider", "error", err)
			continue
		}
		// Merge manual capability overrides (override wins over provider-reported caps).
		if entry.capabilityOverrides != nil {
			for i, m := range models {
				if caps, ok := entry.capabilityOverrides[m.ID]; ok {
					models[i].Capabilities = caps
				}
			}
		}
		// Merge manual context-length overrides (override wins over provider-reported value).
		if entry.contextLengthOverrides != nil {
			for i, m := range models {
				if n, ok := entry.contextLengthOverrides[m.ID]; ok {
					models[i].MaxModelLen = n
				}
			}
		}
		for _, m := range models {
			// Glob patterns are routing wildcards, not real models — pricing doesn't apply.
			if isGlob(m.ID) {
				m.InputCostPerToken = nil
				m.OutputCostPerToken = nil
			}
			// Providers that don't report pricing leave the fields nil.
			// Normalise to +Inf so routing policies can treat unknown cost as worst-case.
			if m.InputCostPerToken == nil {
				inf := math.Inf(1)
				m.InputCostPerToken = &inf
			}
			if m.OutputCostPerToken == nil {
				inf := math.Inf(1)
				m.OutputCostPerToken = &inf
			}
			if _, exists := newIndex[m.ID]; !exists {
				newIndex[m.ID] = entry.provider // first provider wins for primary routing
				newModels = append(newModels, m)
				if isGlob(m.ID) {
					newGlobs = append(newGlobs, m.ID)
				}
			}
			newFallbacks[m.ID] = append(newFallbacks[m.ID], entry.provider)
		}
	}

	r.mu.Lock()
	r.index = newIndex
	r.fallbacks = newFallbacks
	r.globs = newGlobs
	r.allModels = newModels
	r.mu.Unlock()
}

// ResolveModel resolves a virtual capability selector to a concrete model ID.
// For "auto:cap1,cap2" it returns the ID of the least-loaded model that has all
// listed capabilities. If no model matches, the original selector is returned unchanged
// so the caller receives a clear "model not found" error.
// For any other ID it returns the ID unchanged.
func (r *Registry) ResolveModel(id string) string {
	if !strings.HasPrefix(id, "auto:") {
		return id
	}
	required := parseAutoSelector(id)
	if len(required) == 0 {
		return id
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	type candidate struct {
		modelID  string
		provider domain.Provider
	}
	var candidates []candidate
	for _, m := range r.allModels {
		if hasAllCaps(m.Capabilities, required) {
			if p, ok := r.index[m.ID]; ok {
				candidates = append(candidates, candidate{m.ID, p})
			}
		}
	}
	if len(candidates) == 0 {
		return id
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		return activeCount(candidates[i].provider) < activeCount(candidates[j].provider)
	})
	return candidates[0].modelID
}

func parseAutoSelector(id string) []string {
	s := strings.TrimPrefix(id, "auto:")
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	caps := make([]string, 0, len(parts))
	for _, p := range parts {
		if t := strings.TrimSpace(p); t != "" {
			caps = append(caps, t)
		}
	}
	return caps
}

func hasAllCaps(have, want []string) bool {
	set := make(map[string]bool, len(have))
	for _, c := range have {
		set[c] = true
	}
	for _, c := range want {
		if !set[c] {
			return false
		}
	}
	return true
}

// Candidate pairs a resolved model ID with the provider that serves it.
// Used by CandidatesFor to enable cross-model fallback for auto: selectors.
type Candidate struct {
	ModelID  string
	Provider domain.Provider
}

// CandidatesFor returns all (modelID, provider) pairs that can handle id,
// sorted by active request count (least-loaded first).
//
// For concrete model IDs it wraps ProvidersFor in Candidates.
// For "auto:cap1,cap2" selectors it expands to all matching models across all
// providers — this enables cross-model fallback if the first choice fails.
func (r *Registry) CandidatesFor(id string) []Candidate {
	if !strings.HasPrefix(id, "auto:") {
		providers := r.ProvidersFor(id)
		candidates := make([]Candidate, len(providers))
		for i, p := range providers {
			candidates[i] = Candidate{ModelID: id, Provider: p}
		}
		return candidates
	}
	return r.candidatesForAuto(id)
}

func (r *Registry) candidatesForAuto(id string) []Candidate {
	required := parseAutoSelector(id)
	if len(required) == 0 {
		return nil
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	type key struct {
		modelID  string
		provider domain.Provider
	}
	seen := map[key]bool{}
	var candidates []Candidate
	for _, m := range r.allModels {
		if !hasAllCaps(m.Capabilities, required) {
			continue
		}
		for _, p := range r.fallbacks[m.ID] {
			k := key{m.ID, p}
			if !seen[k] {
				seen[k] = true
				candidates = append(candidates, Candidate{ModelID: m.ID, Provider: p})
			}
		}
	}

	sort.SliceStable(candidates, func(i, j int) bool {
		return activeCount(candidates[i].Provider) < activeCount(candidates[j].Provider)
	})
	return candidates
}

// ProviderFor returns the provider responsible for the given model ID.
// Exact matches take priority over glob patterns.
// When multiple glob patterns match, the first registered provider wins.
// Returns false if no registered provider serves that model.
func (r *Registry) ProviderFor(modelID string) (domain.Provider, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if p, ok := r.index[modelID]; ok {
		return p, true
	}

	for _, pattern := range r.globs {
		if matched, _ := path.Match(pattern, modelID); matched {
			return r.index[pattern], true
		}
	}

	return nil, false
}

// ProvidersFor returns all providers that can serve modelID, sorted by active
// request count (least loaded first). Registration order is the tiebreaker.
// Returns an empty slice if no provider serves the model.
func (r *Registry) ProvidersFor(modelID string) []domain.Provider {
	r.mu.RLock()

	var out []domain.Provider
	if ps, ok := r.fallbacks[modelID]; ok {
		out = make([]domain.Provider, len(ps))
		copy(out, ps)
	} else {
		// Glob fallback: collect all providers whose pattern matches modelID.
		seen := map[domain.Provider]bool{}
		for _, pattern := range r.globs {
			if matched, _ := path.Match(pattern, modelID); matched {
				for _, p := range r.fallbacks[pattern] {
					if !seen[p] {
						out = append(out, p)
						seen[p] = true
					}
				}
			}
		}
	}

	r.mu.RUnlock()

	// Sort by active request count so the least-loaded provider is tried first.
	// Stable sort preserves registration order as the tiebreaker.
	sort.SliceStable(out, func(i, j int) bool {
		return activeCount(out[i]) < activeCount(out[j])
	})
	return out
}

// activeCount returns the number of in-flight requests for p, or 0 if p does
// not implement the optional Active() method.
func activeCount(p domain.Provider) int {
	type activeCounter interface{ Active() int }
	if ac, ok := p.(activeCounter); ok {
		return ac.Active()
	}
	return 0
}

// isGlob reports whether a model ID contains glob metacharacters.
func isGlob(id string) bool {
	for _, c := range id {
		if c == '*' || c == '?' || c == '[' {
			return true
		}
	}
	return false
}

// Models returns all models currently known across all registered providers.
func (r *Registry) Models() []domain.Model {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]domain.Model, len(r.allModels))
	copy(out, r.allModels)
	return out
}

// CapabilitiesFor returns the capability list for a concrete model ID and
// whether the model is known to the registry. Capabilities may be empty for a
// known model that does not report them. Intended for observability, not routing.
func (r *Registry) CapabilitiesFor(modelID string) ([]string, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, m := range r.allModels {
		if m.ID == modelID {
			return m.Capabilities, true
		}
	}
	return nil, false
}
