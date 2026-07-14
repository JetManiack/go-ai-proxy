package provider_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
	"github.com/JetManiack/go-ai-proxy/internal/provider"
	"github.com/JetManiack/go-ai-proxy/internal/testutil"
)

// slowProvider blocks until released, simulating a slow upstream.
type slowProvider struct {
	testutil.FakeProvider
	block chan struct{} // close to unblock
}

func newSlowProvider(model string) *slowProvider {
	sp := &slowProvider{block: make(chan struct{})}
	sp.ModelsFunc = func(_ context.Context) ([]domain.Model, error) {
		return []domain.Model{{ID: model}}, nil
	}
	sp.ChatFunc = func(ctx context.Context, req domain.Request) (domain.Response, error) {
		select {
		case <-sp.block:
		case <-ctx.Done():
			return domain.Response{}, ctx.Err()
		}
		return domain.Response{Message: domain.Message{Role: "assistant", Content: "ok"}}, nil
	}
	return sp
}

// TestBounded_AllowsUpToMaxConcurrent verifies that max_concurrent requests
// proceed immediately without blocking.
func TestBounded_AllowsUpToMaxConcurrent(t *testing.T) {
	sp := newSlowProvider("m")
	bp := provider.NewBounded(sp, 3, 0, 0)

	started := make(chan struct{}, 3)
	var wg sync.WaitGroup
	for range 3 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			started <- struct{}{}
			bp.Chat(context.Background(), domain.Request{Model: "m"}) //nolint:errcheck
		}()
	}

	for range 3 {
		select {
		case <-started:
		case <-time.After(500 * time.Millisecond):
			t.Fatal("request did not start within timeout")
		}
	}
	close(sp.block)
	wg.Wait()
}

// TestBounded_BlocksWhenFull verifies that a request beyond max_concurrent
// waits until a slot is released.
func TestBounded_BlocksWhenFull(t *testing.T) {
	sp := newSlowProvider("m")
	bp := provider.NewBounded(sp, 1, 10, 0)

	go bp.Chat(context.Background(), domain.Request{Model: "m"}) //nolint:errcheck
	time.Sleep(20 * time.Millisecond)

	done := make(chan struct{})
	go func() {
		defer close(done)
		bp.Chat(context.Background(), domain.Request{Model: "m"}) //nolint:errcheck
	}()

	select {
	case <-done:
		t.Error("second request should still be queued")
	case <-time.After(50 * time.Millisecond):
	}

	close(sp.block)

	select {
	case <-done:
	case <-time.After(500 * time.Millisecond):
		t.Error("second request did not complete after slot was freed")
	}
}

// TestBounded_RejectsWhenQueueFull verifies that requests beyond
// max_concurrent+queue_size receive ErrQueueFull immediately.
func TestBounded_RejectsWhenQueueFull(t *testing.T) {
	sp := newSlowProvider("m")
	bp := provider.NewBounded(sp, 1, 1, 0) // 1 active + 1 queued; 3rd is rejected

	go bp.Chat(context.Background(), domain.Request{Model: "m"}) //nolint:errcheck
	time.Sleep(20 * time.Millisecond)

	go bp.Chat(context.Background(), domain.Request{Model: "m"}) //nolint:errcheck
	time.Sleep(20 * time.Millisecond)

	_, err := bp.Chat(context.Background(), domain.Request{Model: "m"})
	if !errors.Is(err, provider.ErrQueueFull) {
		t.Errorf("expected ErrQueueFull, got %v", err)
	}

	close(sp.block)
}

// TestBounded_ContextCancelWhileWaiting verifies that a queued request
// unblocks and returns an error when its context is cancelled.
func TestBounded_ContextCancelWhileWaiting(t *testing.T) {
	sp := newSlowProvider("m")
	bp := provider.NewBounded(sp, 1, 10, 0)

	go bp.Chat(context.Background(), domain.Request{Model: "m"}) //nolint:errcheck
	time.Sleep(20 * time.Millisecond)

	ctx, cancel := context.WithCancel(context.Background())
	errc := make(chan error, 1)
	go func() {
		_, err := bp.Chat(ctx, domain.Request{Model: "m"})
		errc <- err
	}()
	time.Sleep(20 * time.Millisecond)
	cancel()

	select {
	case err := <-errc:
		if err == nil {
			t.Error("expected error on context cancel")
		}
	case <-time.After(500 * time.Millisecond):
		t.Error("request did not unblock after context cancel")
	}

	close(sp.block)
}

// TestBounded_ActiveCount verifies that Active() reflects in-flight requests.
func TestBounded_ActiveCount(t *testing.T) {
	sp := newSlowProvider("m")
	bp := provider.NewBounded(sp, 5, 0, 0)

	if bp.Active() != 0 {
		t.Fatalf("initial active count: got %d, want 0", bp.Active())
	}

	ready := make(chan struct{})
	var wg sync.WaitGroup
	for range 3 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			ready <- struct{}{}
			bp.Chat(ctx, domain.Request{Model: "m"}) //nolint:errcheck
		}()
	}

	for range 3 {
		<-ready
	}
	time.Sleep(20 * time.Millisecond)

	if got := bp.Active(); got != 3 {
		t.Errorf("active count: got %d, want 3", got)
	}

	close(sp.block)
	wg.Wait()

	if bp.Active() != 0 {
		t.Errorf("active count after completion: got %d, want 0", bp.Active())
	}
}

// TestBounded_Stream_HoldsSlotForDuration verifies that a streaming request
// holds its concurrency slot for the full stream duration and releases on close.
func TestBounded_Stream_HoldsSlotForDuration(t *testing.T) {
	streamDone := make(chan struct{})
	fp := &testutil.FakeProvider{
		ModelsFunc: func(_ context.Context) ([]domain.Model, error) {
			return []domain.Model{{ID: "m"}}, nil
		},
		StreamFunc: func(ctx context.Context, _ domain.Request) (<-chan domain.Chunk, error) {
			ch := make(chan domain.Chunk)
			go func() {
				defer close(ch)
				<-streamDone
				ch <- domain.Chunk{Done: true}
			}()
			return ch, nil
		},
	}
	bp := provider.NewBounded(fp, 1, 0, 0)

	ch, err := bp.ChatStream(context.Background(), domain.Request{Model: "m"})
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}

	if bp.Active() != 1 {
		t.Errorf("active during stream: got %d, want 1", bp.Active())
	}

	close(streamDone)
	for range ch {
	}

	time.Sleep(20 * time.Millisecond)
	if bp.Active() != 0 {
		t.Errorf("active after stream close: got %d, want 0", bp.Active())
	}
}

// TestBounded_Stream_BlocksSecondWhenFull verifies that a second streaming
// request blocks while the first holds the only concurrency slot.
func TestBounded_Stream_BlocksSecondWhenFull(t *testing.T) {
	firstDone := make(chan struct{})
	fp := &testutil.FakeProvider{
		ModelsFunc: func(_ context.Context) ([]domain.Model, error) {
			return []domain.Model{{ID: "m"}}, nil
		},
		StreamFunc: func(ctx context.Context, _ domain.Request) (<-chan domain.Chunk, error) {
			ch := make(chan domain.Chunk)
			go func() {
				defer close(ch)
				<-firstDone
				ch <- domain.Chunk{Done: true}
			}()
			return ch, nil
		},
	}
	bp := provider.NewBounded(fp, 1, 10, 0)

	first, err := bp.ChatStream(context.Background(), domain.Request{Model: "m"})
	if err != nil {
		t.Fatalf("first ChatStream: %v", err)
	}

	secondStarted := make(chan struct{})
	go func() {
		ch, err := bp.ChatStream(context.Background(), domain.Request{Model: "m"})
		if err == nil {
			close(secondStarted)
			for range ch {
			}
		}
	}()

	select {
	case <-secondStarted:
		t.Error("second stream should be blocked while first holds the slot")
	case <-time.After(50 * time.Millisecond):
	}

	close(firstDone)
	for range first {
	}

	select {
	case <-secondStarted:
	case <-time.After(500 * time.Millisecond):
		t.Error("second stream did not start after first slot was freed")
	}
}

// --- request_timeout ---

// TestBounded_Chat_RequestTimeout verifies that Chat returns an error when
// the provider does not respond within request_timeout.
func TestBounded_Chat_RequestTimeout(t *testing.T) {
	sp := newSlowProvider("m") // blocks until sp.block is closed
	bp := provider.NewBounded(sp, 5, 0, 50*time.Millisecond)

	_, err := bp.Chat(context.Background(), domain.Request{Model: "m"})
	if err == nil {
		t.Fatal("expected timeout error, got nil")
	}
	// Unblock the inner goroutine to avoid goroutine leak in test.
	close(sp.block)
}

// TestBounded_Stream_RequestTimeoutBeforeFirstToken verifies that ChatStream
// returns an error (not a channel) when no first token arrives within request_timeout.
// This lets the server fall back to the next provider.
func TestBounded_Stream_RequestTimeoutBeforeFirstToken(t *testing.T) {
	block := make(chan struct{})
	fp := &testutil.FakeProvider{
		ModelsFunc: func(_ context.Context) ([]domain.Model, error) {
			return []domain.Model{{ID: "m"}}, nil
		},
		StreamFunc: func(ctx context.Context, _ domain.Request) (<-chan domain.Chunk, error) {
			ch := make(chan domain.Chunk)
			go func() {
				defer close(ch)
				select {
				case <-block:
					ch <- domain.Chunk{Done: true}
				case <-ctx.Done():
				}
			}()
			return ch, nil
		},
	}
	bp := provider.NewBounded(fp, 5, 0, 50*time.Millisecond)

	_, err := bp.ChatStream(context.Background(), domain.Request{Model: "m"})
	if err == nil {
		t.Fatal("expected timeout error, got nil channel")
	}
	close(block)
}

// TestBounded_Stream_TimeoutCancelledAfterFirstToken verifies that a slow
// generation is NOT killed once the first token has arrived — the timeout
// only applies to the wait for the first token.
func TestBounded_Stream_TimeoutCancelledAfterFirstToken(t *testing.T) {
	firstSent := make(chan struct{})
	slowDone := make(chan struct{})
	fp := &testutil.FakeProvider{
		ModelsFunc: func(_ context.Context) ([]domain.Model, error) {
			return []domain.Model{{ID: "m"}}, nil
		},
		StreamFunc: func(ctx context.Context, _ domain.Request) (<-chan domain.Chunk, error) {
			ch := make(chan domain.Chunk, 2)
			go func() {
				defer close(ch)
				ch <- domain.Chunk{ID: "c", Model: "m", Delta: "first"}
				close(firstSent)
				// Simulate slow generation: wait longer than request_timeout.
				select {
				case <-slowDone:
				case <-ctx.Done():
					return
				}
				ch <- domain.Chunk{Done: true}
			}()
			return ch, nil
		},
	}
	bp := provider.NewBounded(fp, 5, 0, 50*time.Millisecond)

	ch, err := bp.ChatStream(context.Background(), domain.Request{Model: "m"})
	if err != nil {
		t.Fatalf("ChatStream: unexpected error %v", err)
	}

	// Read first chunk.
	chunk, ok := <-ch
	if !ok || chunk.Delta != "first" {
		t.Fatalf("expected first chunk, got ok=%v chunk=%+v", ok, chunk)
	}

	// Wait longer than request_timeout — stream must still be alive.
	time.Sleep(100 * time.Millisecond)
	close(slowDone)

	// Drain remaining chunks — must complete without error.
	for range ch {
	}
}

// --- rate-limit cooldown ---

// TestBounded_Chat_CooldownShortCircuits verifies that after an inner provider
// returns RateLimitError, subsequent Chat calls return RateLimitError immediately
// without forwarding to the inner provider.
func TestBounded_Chat_CooldownShortCircuits(t *testing.T) {
	calls := 0
	fp := &testutil.FakeProvider{
		ModelsFunc: func(_ context.Context) ([]domain.Model, error) {
			return []domain.Model{{ID: "m"}}, nil
		},
		ChatFunc: func(_ context.Context, _ domain.Request) (domain.Response, error) {
			calls++
			return domain.Response{}, &provider.RateLimitError{RetryAfter: 100 * time.Millisecond}
		},
	}
	bp := provider.NewBounded(fp, 5, 0, 0)

	// First call hits inner; inner sets the cooldown.
	_, err := bp.Chat(context.Background(), domain.Request{Model: "m"})
	var rl *provider.RateLimitError
	if !errors.As(err, &rl) {
		t.Fatalf("first call: expected RateLimitError, got %v", err)
	}

	// Second call is short-circuited; inner is NOT called again.
	_, err = bp.Chat(context.Background(), domain.Request{Model: "m"})
	if !errors.As(err, &rl) {
		t.Fatalf("second call: expected RateLimitError, got %v", err)
	}
	if calls != 1 {
		t.Errorf("inner called %d times; want 1 (cooldown should prevent second call)", calls)
	}

	// After cooldown expires, inner is reachable again.
	time.Sleep(150 * time.Millisecond)
	_, _ = bp.Chat(context.Background(), domain.Request{Model: "m"})
	if calls != 2 {
		t.Errorf("inner called %d times after cooldown; want 2", calls)
	}
}

// TestBounded_Stream_CooldownShortCircuits verifies the same cooldown behaviour
// for ChatStream.
func TestBounded_Stream_CooldownShortCircuits(t *testing.T) {
	calls := 0
	fp := &testutil.FakeProvider{
		ModelsFunc: func(_ context.Context) ([]domain.Model, error) {
			return []domain.Model{{ID: "m"}}, nil
		},
		StreamFunc: func(_ context.Context, _ domain.Request) (<-chan domain.Chunk, error) {
			calls++
			return nil, &provider.RateLimitError{RetryAfter: 100 * time.Millisecond}
		},
	}
	bp := provider.NewBounded(fp, 5, 0, 0)

	_, err := bp.ChatStream(context.Background(), domain.Request{Model: "m"})
	var rl *provider.RateLimitError
	if !errors.As(err, &rl) {
		t.Fatalf("first call: expected RateLimitError, got %v", err)
	}

	_, err = bp.ChatStream(context.Background(), domain.Request{Model: "m"})
	if !errors.As(err, &rl) {
		t.Fatalf("second call: expected RateLimitError, got %v", err)
	}
	if calls != 1 {
		t.Errorf("inner called %d times; want 1 (cooldown should prevent second call)", calls)
	}
}

// TestRegistry_ProvidersFor_LeastLoadedFirst verifies that ProvidersFor
// returns BoundedProviders sorted by Active() count (least loaded first).
func TestRegistry_ProvidersFor_LeastLoadedFirst(t *testing.T) {
	sp1 := newSlowProvider("m")
	sp2 := newSlowProvider("m")
	bp1 := provider.NewBounded(sp1, 5, 0, 0)
	bp2 := provider.NewBounded(sp2, 5, 0, 0)

	reg := provider.NewRegistry(time.Hour)
	reg.Register(bp1)
	reg.Register(bp2)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatal(err)
	}

	for range 2 {
		go bp1.Chat(context.Background(), domain.Request{Model: "m"}) //nolint:errcheck
	}
	time.Sleep(20 * time.Millisecond)

	ps := reg.ProvidersFor("m")
	if len(ps) != 2 {
		t.Fatalf("expected 2 providers, got %d", len(ps))
	}
	if ps[0] != bp2 {
		t.Error("least loaded provider (bp2, active=0) should be first")
	}
	if ps[1] != bp1 {
		t.Error("busier provider (bp1, active=2) should be second")
	}

	close(sp1.block)
	close(sp2.block)
}
