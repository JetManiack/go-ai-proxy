package openai

import (
	"testing"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
)

func TestThinkParser_NormalDelta(t *testing.T) {
	p := &thinkParser{}
	chunk := p.process(domain.Chunk{ID: "c", Delta: "hello"})
	if chunk.Delta != "hello" || chunk.ThinkingDelta != "" {
		t.Errorf("got Delta=%q ThinkingDelta=%q", chunk.Delta, chunk.ThinkingDelta)
	}
}

func TestThinkParser_OpenTag(t *testing.T) {
	p := &thinkParser{}
	chunk := p.process(domain.Chunk{ID: "c", Delta: "<think>"})
	if chunk.Delta != "" {
		t.Errorf("Delta should be empty after <think>, got %q", chunk.Delta)
	}
	if chunk.ThinkingDelta != "" {
		t.Errorf("ThinkingDelta should be empty for bare open tag, got %q", chunk.ThinkingDelta)
	}
	if !p.inThink {
		t.Error("should be in think state after <think>")
	}
}

func TestThinkParser_ThinkingContent(t *testing.T) {
	p := &thinkParser{inThink: true}
	chunk := p.process(domain.Chunk{ID: "c", Delta: "reasoning here"})
	if chunk.ThinkingDelta != "reasoning here" {
		t.Errorf("ThinkingDelta: got %q", chunk.ThinkingDelta)
	}
	if chunk.Delta != "" {
		t.Errorf("Delta should be empty inside think block, got %q", chunk.Delta)
	}
}

func TestThinkParser_CloseTag(t *testing.T) {
	p := &thinkParser{inThink: true}
	chunk := p.process(domain.Chunk{ID: "c", Delta: "</think>\nactual answer"})
	if chunk.ThinkingDelta != "" {
		t.Errorf("ThinkingDelta should be empty at close tag, got %q", chunk.ThinkingDelta)
	}
	if chunk.Delta != "actual answer" {
		t.Errorf("Delta: got %q, want %q", chunk.Delta, "actual answer")
	}
	if p.inThink {
		t.Error("should not be in think state after </think>")
	}
}

func TestThinkParser_OpenTagWithContentAfter(t *testing.T) {
	p := &thinkParser{}
	chunk := p.process(domain.Chunk{ID: "c", Delta: "<think>\nstep one"})
	if chunk.ThinkingDelta != "step one" {
		t.Errorf("ThinkingDelta: got %q, want %q", chunk.ThinkingDelta, "step one")
	}
	if chunk.Delta != "" {
		t.Errorf("Delta should be empty, got %q", chunk.Delta)
	}
}

func TestThinkParser_ContentBeforeOpenTag(t *testing.T) {
	p := &thinkParser{}
	chunk := p.process(domain.Chunk{ID: "c", Delta: "prefix<think>"})
	if chunk.Delta != "prefix" {
		t.Errorf("Delta: got %q, want %q", chunk.Delta, "prefix")
	}
}

func TestThinkParser_DonePassThrough(t *testing.T) {
	p := &thinkParser{inThink: true}
	chunk := p.process(domain.Chunk{Done: true})
	if chunk.Done != true {
		t.Error("Done chunk should pass through unchanged")
	}
}

func TestThinkParser_AlreadyExtractedPassThrough(t *testing.T) {
	p := &thinkParser{}
	chunk := p.process(domain.Chunk{ID: "c", ThinkingDelta: "already set", Delta: ""})
	if chunk.ThinkingDelta != "already set" {
		t.Errorf("pre-set ThinkingDelta should pass through, got %q", chunk.ThinkingDelta)
	}
}

func TestThinkParser_FullSequence(t *testing.T) {
	p := &thinkParser{}
	cases := []struct {
		delta        string
		wantThinking string
		wantContent  string
	}{
		{"<think>", "", ""},
		{"\nstep one\n", "\nstep one\n", ""},
		{"\nstep two\n", "\nstep two\n", ""},
		{"</think>\n", "", ""},
		{"final answer", "", "final answer"},
	}

	for i, tc := range cases {
		chunk := p.process(domain.Chunk{ID: "c", Delta: tc.delta})
		if chunk.ThinkingDelta != tc.wantThinking {
			t.Errorf("chunk[%d] ThinkingDelta: got %q, want %q", i, chunk.ThinkingDelta, tc.wantThinking)
		}
		if chunk.Delta != tc.wantContent {
			t.Errorf("chunk[%d] Delta: got %q, want %q", i, chunk.Delta, tc.wantContent)
		}
	}
}
