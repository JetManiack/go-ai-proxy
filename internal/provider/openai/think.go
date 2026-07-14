package openai

import (
	"strings"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
)

// thinkParser is a stateful parser that extracts <think>…</think> content
// from streaming deltas and routes it to Chunk.ThinkingDelta.
//
// Local reasoning models (DeepSeek-R1, QwQ, etc.) embed their chain-of-thought
// inside the content stream using these tags. The parser assumes tags arrive as
// complete tokens, which is universally true for llama.cpp-based runtimes.
type thinkParser struct {
	inThink bool
}

// process rewrites a chunk's Delta/ThinkingDelta based on current think state.
// Chunks with tool calls or the Done sentinel are passed through unchanged.
func (p *thinkParser) process(chunk domain.Chunk) domain.Chunk {
	if chunk.Done || chunk.ToolCallIndex != nil {
		return chunk
	}
	// reasoning_content already extracted upstream (e.g. DeepSeek API) — pass through.
	if chunk.ThinkingDelta != "" {
		return chunk
	}

	delta := chunk.Delta

	if !p.inThink {
		idx := strings.Index(delta, "<think>")
		if idx < 0 {
			return chunk
		}
		p.inThink = true
		after := delta[idx+len("<think>"):]
		chunk.Delta = delta[:idx]
		chunk.ThinkingDelta = strings.TrimPrefix(after, "\n")
		return chunk
	}

	// Inside <think> block.
	idx := strings.Index(delta, "</think>")
	if idx < 0 {
		chunk.ThinkingDelta = delta
		chunk.Delta = ""
		return chunk
	}
	p.inThink = false
	chunk.ThinkingDelta = delta[:idx]
	chunk.Delta = strings.TrimPrefix(delta[idx+len("</think>"):], "\n")
	return chunk
}
