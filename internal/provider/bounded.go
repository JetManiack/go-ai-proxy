package provider

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
)

// ErrQueueFull is returned when a provider's request queue has reached its
// capacity and cannot accept new requests.
var ErrQueueFull = errors.New("provider queue full")

// BoundedProvider wraps a domain.Provider with:
//   - a concurrency limit (semaphore): at most maxConcurrent requests run simultaneously
//   - an optional queue depth: requests beyond maxConcurrent wait; if maxQueueSize > 0
//     and the queue is full, new requests receive ErrQueueFull immediately
//   - an optional request timeout: for non-streaming requests, the total response time;
//     for streaming requests, the time allowed to receive the first token — after
//     the first token arrives the timeout is cancelled so slow generation is not killed
//   - upstream rate-limit tracking: when the inner provider returns RateLimitError the
//     provider is marked as cooling down until the Retry-After deadline, and all
//     incoming requests short-circuit with RateLimitError for that duration
//
// Providers are sorted by Active() in the registry (least-connections routing).
type BoundedProvider struct {
	inner          domain.Provider
	sem            chan struct{} // nil when maxConcurrent == 0
	maxWaiting     int
	waitingCount   atomic.Int32
	activeCount    atomic.Int32
	requestTimeout time.Duration
	coolUntil      atomic.Int64 // unix nanoseconds; zero means not cooling
}

// NewBounded wraps inner with the given limits.
//   - maxConcurrent: max parallel requests; 0 = unlimited (only timeout applies)
//   - maxQueueSize:  max waiting requests; 0 = unlimited queue
//   - requestTimeout: 0 = no timeout
func NewBounded(inner domain.Provider, maxConcurrent, maxQueueSize int, requestTimeout time.Duration) *BoundedProvider {
	var sem chan struct{}
	if maxConcurrent > 0 {
		sem = make(chan struct{}, maxConcurrent)
	}
	return &BoundedProvider{
		inner:          inner,
		sem:            sem,
		maxWaiting:     maxQueueSize,
		requestTimeout: requestTimeout,
	}
}

// Name delegates to the inner provider.
func (p *BoundedProvider) Name() string { return p.inner.Name() }

// Active returns the number of requests currently executing (not queued).
// Used by the registry for least-connections routing.
func (p *BoundedProvider) Active() int { return int(p.activeCount.Load()) }

// cooling returns the remaining cooldown duration and true if the provider is
// currently rate-limited by its upstream. Returns 0, false otherwise.
func (p *BoundedProvider) cooling() (time.Duration, bool) {
	nano := p.coolUntil.Load()
	if nano == 0 {
		return 0, false
	}
	remaining := time.Until(time.Unix(0, nano))
	if remaining <= 0 {
		return 0, false
	}
	return remaining, true
}

// setCooldown records a rate-limit cooldown using the RetryAfter from err if it
// is a *RateLimitError, extending any existing cooldown if the new deadline is later.
func (p *BoundedProvider) setCooldown(err error) {
	var rl *RateLimitError
	if !errors.As(err, &rl) {
		return
	}
	deadline := time.Now().Add(rl.RetryAfter).UnixNano()
	for {
		current := p.coolUntil.Load()
		if deadline <= current {
			return
		}
		if p.coolUntil.CompareAndSwap(current, deadline) {
			return
		}
	}
}

// acquire blocks until a concurrency slot is available or ctx expires.
// Returns ErrQueueFull immediately if the waiting queue is at capacity.
func (p *BoundedProvider) acquire(ctx context.Context) error {
	if p.sem == nil {
		p.activeCount.Add(1)
		return nil
	}

	if p.maxWaiting > 0 {
		n := p.waitingCount.Add(1)
		if int(n) > p.maxWaiting {
			p.waitingCount.Add(-1)
			return ErrQueueFull
		}
		defer p.waitingCount.Add(-1)
	}

	select {
	case p.sem <- struct{}{}:
		p.activeCount.Add(1)
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (p *BoundedProvider) release() {
	p.activeCount.Add(-1)
	if p.sem != nil {
		<-p.sem
	}
}

// Chat acquires a slot, applies request_timeout to the whole call, then releases.
func (p *BoundedProvider) Chat(ctx context.Context, req domain.Request) (domain.Response, error) {
	if remaining, ok := p.cooling(); ok {
		return domain.Response{}, &RateLimitError{RetryAfter: remaining}
	}
	if err := p.acquire(ctx); err != nil {
		return domain.Response{}, err
	}
	defer p.release()

	callCtx := ctx
	if p.requestTimeout > 0 {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, p.requestTimeout)
		defer cancel()
	}
	resp, err := p.inner.Chat(callCtx, req)
	if err != nil {
		p.setCooldown(err)
		return domain.Response{}, err
	}
	return resp, nil
}

// ChatStream acquires a slot and starts the inner stream.
//
// When request_timeout > 0, ChatStream blocks until the first token arrives or
// the timeout fires. On timeout it returns an error so the server can fall back
// to the next provider. After the first token the timeout is cancelled — slow
// but running generation is never interrupted.
//
// The concurrency slot is held for the full stream duration and released when
// the returned channel is drained or ctx is cancelled.
func (p *BoundedProvider) ChatStream(ctx context.Context, req domain.Request) (<-chan domain.Chunk, error) {
	if remaining, ok := p.cooling(); ok {
		return nil, &RateLimitError{RetryAfter: remaining}
	}
	if err := p.acquire(ctx); err != nil {
		return nil, err
	}

	// innerCtx lets us cancel the inner stream independently of the caller's ctx
	// (e.g. on first-token timeout) without propagating cancellation upward.
	innerCtx, cancelInner := context.WithCancel(ctx)

	inner, err := p.inner.ChatStream(innerCtx, req)
	if err != nil {
		p.setCooldown(err)
		cancelInner()
		p.release()
		return nil, err
	}

	// When request_timeout is set, wait synchronously for the first token.
	// This is the only window where the timeout is active.
	var prefetched *domain.Chunk
	if p.requestTimeout > 0 {
		timer := time.NewTimer(p.requestTimeout)
		select {
		case chunk, ok := <-inner:
			timer.Stop()
			if !ok {
				cancelInner()
				p.release()
				return nil, fmt.Errorf("provider stream closed before first token")
			}
			prefetched = &chunk
		case <-timer.C:
			cancelInner()
			p.release()
			return nil, fmt.Errorf("no first token within %s", p.requestTimeout)
		case <-ctx.Done():
			timer.Stop()
			cancelInner()
			p.release()
			return nil, ctx.Err()
		}
	}

	// Buffer size 1 so the pre-fetched chunk can be written without blocking.
	out := make(chan domain.Chunk, 1)
	if prefetched != nil {
		out <- *prefetched
	}

	go func() {
		defer close(out)
		defer p.release()
		defer cancelInner()
		for {
			select {
			case chunk, ok := <-inner:
				if !ok {
					return
				}
				select {
				case out <- chunk:
				case <-ctx.Done():
					return
				}
			case <-ctx.Done():
				return
			}
		}
	}()
	return out, nil
}

// Models delegates to the inner provider (model listing is not rate-limited).
func (p *BoundedProvider) Models(ctx context.Context) ([]domain.Model, error) {
	return p.inner.Models(ctx)
}

// Embeddings applies the same concurrency/queue/rate-limit-cooldown/timeout
// semantics as Chat, delegating to the inner provider if it supports
// embeddings. *BoundedProvider always satisfies domain.EmbeddingsProvider so
// callers can type-assert regardless of whether a provider ended up
// bounded — it fails clearly here instead if the wrapped provider doesn't
// implement it (e.g. a bounded Anthropic provider).
func (p *BoundedProvider) Embeddings(ctx context.Context, req domain.EmbedRequest) (domain.EmbedResponse, error) {
	ep, ok := p.inner.(domain.EmbeddingsProvider)
	if !ok {
		return domain.EmbedResponse{}, fmt.Errorf("%s: embeddings not supported", p.inner.Name())
	}

	if remaining, cooling := p.cooling(); cooling {
		return domain.EmbedResponse{}, &RateLimitError{RetryAfter: remaining}
	}
	if err := p.acquire(ctx); err != nil {
		return domain.EmbedResponse{}, err
	}
	defer p.release()

	callCtx := ctx
	if p.requestTimeout > 0 {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, p.requestTimeout)
		defer cancel()
	}
	resp, err := ep.Embeddings(callCtx, req)
	if err != nil {
		p.setCooldown(err)
		return domain.EmbedResponse{}, err
	}
	return resp, nil
}
