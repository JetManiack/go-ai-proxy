package provider_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
	"github.com/JetManiack/go-ai-proxy/internal/provider"
	"github.com/JetManiack/go-ai-proxy/internal/testutil"
)

// newChatOnlyProvider returns a provider that implements domain.Provider but
// deliberately does NOT implement domain.EmbeddingsProvider, modelling a
// provider like Anthropic that has no native embeddings support.
func newChatOnlyProvider(model string) domain.Provider {
	return &chatOnlyProviderNoEmbed{
		modelsFunc: func(_ context.Context) ([]domain.Model, error) {
			return []domain.Model{{ID: model}}, nil
		},
	}
}

// chatOnlyProviderNoEmbed implements exactly domain.Provider's four methods
// and nothing else, so it never satisfies domain.EmbeddingsProvider.
type chatOnlyProviderNoEmbed struct {
	modelsFunc func(ctx context.Context) ([]domain.Model, error)
}

func (p *chatOnlyProviderNoEmbed) Name() string { return "chat-only" }
func (p *chatOnlyProviderNoEmbed) Chat(ctx context.Context, req domain.Request) (domain.Response, error) {
	return domain.Response{Message: domain.Message{Role: "assistant", Content: "ok"}}, nil
}
func (p *chatOnlyProviderNoEmbed) ChatStream(ctx context.Context, req domain.Request) (<-chan domain.Chunk, error) {
	ch := make(chan domain.Chunk, 1)
	ch <- domain.Chunk{Done: true}
	close(ch)
	return ch, nil
}
func (p *chatOnlyProviderNoEmbed) Models(ctx context.Context) ([]domain.Model, error) {
	return p.modelsFunc(ctx)
}

func TestBounded_Embeddings_DelegatesWhenSupported(t *testing.T) {
	fp := &testutil.FakeProvider{
		ModelsFunc: func(_ context.Context) ([]domain.Model, error) {
			return []domain.Model{{ID: "m"}}, nil
		},
		EmbedFunc: func(_ context.Context, req domain.EmbedRequest) (domain.EmbedResponse, error) {
			return domain.EmbedResponse{
				Model:      req.Model,
				Embeddings: []domain.Embedding{{Index: 0, Values: []float64{1, 2, 3}}},
			}, nil
		},
	}
	bp := provider.NewBounded(fp, 5, 0, 0)

	resp, err := bp.Embeddings(context.Background(), domain.EmbedRequest{Model: "m", Input: []string{"hi"}})
	if err != nil {
		t.Fatalf("Embeddings: unexpected error %v", err)
	}
	if len(resp.Embeddings) != 1 || resp.Embeddings[0].Values[2] != 3 {
		t.Errorf("Embeddings: got %+v", resp.Embeddings)
	}
}

func TestBounded_Embeddings_UnsupportedInner(t *testing.T) {
	inner := newChatOnlyProvider("m")
	bp := provider.NewBounded(inner, 5, 0, 0)

	_, err := bp.Embeddings(context.Background(), domain.EmbedRequest{Model: "m", Input: []string{"hi"}})
	if err == nil {
		t.Fatal("expected error for provider without embeddings support, got nil")
	}
}

func TestBounded_Embeddings_RequestTimeout(t *testing.T) {
	block := make(chan struct{})
	fp := &testutil.FakeProvider{
		ModelsFunc: func(_ context.Context) ([]domain.Model, error) {
			return []domain.Model{{ID: "m"}}, nil
		},
		EmbedFunc: func(ctx context.Context, _ domain.EmbedRequest) (domain.EmbedResponse, error) {
			select {
			case <-block:
			case <-ctx.Done():
				return domain.EmbedResponse{}, ctx.Err()
			}
			return domain.EmbedResponse{}, nil
		},
	}
	bp := provider.NewBounded(fp, 5, 0, 50*time.Millisecond)

	_, err := bp.Embeddings(context.Background(), domain.EmbedRequest{Model: "m"})
	if err == nil {
		t.Fatal("expected timeout error, got nil")
	}
	close(block) // avoid goroutine leak
}

func TestBounded_Embeddings_CooldownShortCircuits(t *testing.T) {
	calls := 0
	fp := &testutil.FakeProvider{
		ModelsFunc: func(_ context.Context) ([]domain.Model, error) {
			return []domain.Model{{ID: "m"}}, nil
		},
		EmbedFunc: func(_ context.Context, _ domain.EmbedRequest) (domain.EmbedResponse, error) {
			calls++
			return domain.EmbedResponse{}, &provider.RateLimitError{RetryAfter: 100 * time.Millisecond}
		},
	}
	bp := provider.NewBounded(fp, 5, 0, 0)

	_, err := bp.Embeddings(context.Background(), domain.EmbedRequest{Model: "m"})
	var rl *provider.RateLimitError
	if !errors.As(err, &rl) {
		t.Fatalf("first call: expected RateLimitError, got %v", err)
	}

	_, err = bp.Embeddings(context.Background(), domain.EmbedRequest{Model: "m"})
	if !errors.As(err, &rl) {
		t.Fatalf("second call: expected RateLimitError (cooldown), got %v", err)
	}
	if calls != 1 {
		t.Errorf("inner Embeddings should be called once (cooldown short-circuits the second call), got %d calls", calls)
	}
}
