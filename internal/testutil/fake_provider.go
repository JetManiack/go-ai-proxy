package testutil

import (
	"context"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
)

// FakeProvider is a configurable test double for domain.Provider.
type FakeProvider struct {
	NameVal    string
	ChatFunc   func(ctx context.Context, req domain.Request) (domain.Response, error)
	StreamFunc func(ctx context.Context, req domain.Request) (<-chan domain.Chunk, error)
	ModelsFunc func(ctx context.Context) ([]domain.Model, error)
}

func (f *FakeProvider) Name() string {
	if f.NameVal != "" {
		return f.NameVal
	}
	return "fake"
}

func (f *FakeProvider) Chat(ctx context.Context, req domain.Request) (domain.Response, error) {
	if f.ChatFunc != nil {
		return f.ChatFunc(ctx, req)
	}
	return domain.Response{ID: "fake-id", Model: req.Model, Message: domain.Message{Role: "assistant", Content: "fake response"}}, nil
}

func (f *FakeProvider) ChatStream(ctx context.Context, req domain.Request) (<-chan domain.Chunk, error) {
	if f.StreamFunc != nil {
		return f.StreamFunc(ctx, req)
	}
	ch := make(chan domain.Chunk, 1)
	ch <- domain.Chunk{Done: true}
	close(ch)
	return ch, nil
}

func (f *FakeProvider) Models(ctx context.Context) ([]domain.Model, error) {
	if f.ModelsFunc != nil {
		return f.ModelsFunc(ctx)
	}
	return nil, nil
}

// NewFakeProvider returns a FakeProvider that serves the given models.
func NewFakeProvider(models ...domain.Model) *FakeProvider {
	return &FakeProvider{
		ModelsFunc: func(_ context.Context) ([]domain.Model, error) {
			return models, nil
		},
	}
}
