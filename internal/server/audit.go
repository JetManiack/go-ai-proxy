package server

import (
	"context"
	"log/slog"
	"time"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
)

// auditChat logs a completed non-streaming chat request.
// INFO: metadata (model, request_id, duration, tokens/error).
// DEBUG: full prompt messages and response content.
// logger nil = disabled.
func auditChat(
	logger *slog.Logger,
	ctx context.Context,
	req domain.Request,
	resp *domain.Response,
	err error,
	start time.Time,
) {
	if logger == nil {
		return
	}

	// DEBUG: full content — only emitted when handler is configured at DEBUG level.
	if logger.Enabled(ctx, slog.LevelDebug) {
		dbgArgs := []any{
			"request_id", requestIDFromContext(ctx),
			"model", req.Model,
			"messages", req.Messages,
		}
		if err == nil && resp != nil {
			dbgArgs = append(dbgArgs, "response", resp.Message.Content)
		}
		logger.DebugContext(ctx, "audit", dbgArgs...)
	}

	// INFO/ERROR: metadata only.
	args := []any{
		"request_id", requestIDFromContext(ctx),
		"model", req.Model,
		"duration_ms", time.Since(start).Milliseconds(),
	}
	if err != nil {
		args = append(args, "error", err.Error())
		logger.ErrorContext(ctx, "audit", args...)
		return
	}
	if resp != nil {
		args = append(args,
			"prompt_tokens", resp.Usage.PromptTokens,
			"completion_tokens", resp.Usage.CompletionTokens,
		)
		if resp.Usage.CachedTokens > 0 {
			args = append(args, "cached_tokens", resp.Usage.CachedTokens)
		}
		if resp.FinishReason != "" {
			args = append(args, "finish_reason", resp.FinishReason)
		}
	}
	logger.InfoContext(ctx, "audit", args...)
}

// auditEmbeddings logs a completed embeddings request.
// INFO: metadata (model, request_id, duration, input_count, tokens/error).
// DEBUG: the input strings themselves.
// logger nil = disabled.
func auditEmbeddings(
	logger *slog.Logger,
	ctx context.Context,
	req domain.EmbedRequest,
	resp *domain.EmbedResponse,
	err error,
	start time.Time,
) {
	if logger == nil {
		return
	}

	if logger.Enabled(ctx, slog.LevelDebug) {
		logger.DebugContext(ctx, "audit",
			"request_id", requestIDFromContext(ctx),
			"model", req.Model,
			"input", req.Input,
		)
	}

	args := []any{
		"request_id", requestIDFromContext(ctx),
		"model", req.Model,
		"input_count", len(req.Input),
		"duration_ms", time.Since(start).Milliseconds(),
	}
	if err != nil {
		args = append(args, "error", err.Error())
		logger.ErrorContext(ctx, "audit", args...)
		return
	}
	if resp != nil {
		args = append(args, "prompt_tokens", resp.Usage.PromptTokens)
	}
	logger.InfoContext(ctx, "audit", args...)
}

// auditStreamStart logs the moment a streaming response begins (headers flushed).
// INFO: model + request_id.
// DEBUG: full prompt messages.
// logger nil = disabled.
func auditStreamStart(logger *slog.Logger, ctx context.Context, req domain.Request) {
	if logger == nil {
		return
	}
	if logger.Enabled(ctx, slog.LevelDebug) {
		logger.DebugContext(ctx, "audit",
			"request_id", requestIDFromContext(ctx),
			"model", req.Model,
			"messages", req.Messages,
		)
	}
	logger.InfoContext(ctx, "audit",
		"event", "stream_start",
		"request_id", requestIDFromContext(ctx),
		"model", req.Model,
	)
}

// auditClientDisconnect logs an INFO record for a client-disconnect — the
// client closed the connection before we could return a response. Operationally
// normal (often a client-side timeout); not an error.
// logger nil = disabled.
func auditClientDisconnect(logger *slog.Logger, ctx context.Context, model string, start time.Time) {
	if logger == nil {
		return
	}
	logger.InfoContext(ctx, "audit",
		"event", "client_disconnect",
		"request_id", requestIDFromContext(ctx),
		"model", model,
		"duration_ms", time.Since(start).Milliseconds(),
	)
}

// auditStreamEnd logs when the last chunk has been sent to the client.
//
// content is the assembled delta content (text the client received); logged at
// DEBUG when the handler is configured at debug level (symmetric with
// auditChat's response field).
// ttftMs is the time from stream start to the first content/reasoning chunk;
// 0 means no first-token signal was observed (empty stream, error path).
// usage is non-nil only when the upstream emitted token counts.
// finishReason is the last finish_reason seen on a chunk.
// logger nil = disabled.
func auditStreamEnd(
	logger *slog.Logger,
	ctx context.Context,
	model string,
	start time.Time,
	usage *domain.Usage,
	finishReason string,
	ttftMs int64,
	content string,
) {
	if logger == nil {
		return
	}

	// DEBUG: assembled stream content — only emitted when handler is at DEBUG level.
	if logger.Enabled(ctx, slog.LevelDebug) && content != "" {
		logger.DebugContext(ctx, "audit",
			"event", "stream_end",
			"request_id", requestIDFromContext(ctx),
			"model", model,
			"response", content,
		)
	}

	args := []any{
		"event", "stream_end",
		"request_id", requestIDFromContext(ctx),
		"model", model,
		"duration_ms", time.Since(start).Milliseconds(),
	}
	if ttftMs > 0 {
		args = append(args, "ttft_ms", ttftMs)
	}
	if usage != nil {
		args = append(args,
			"prompt_tokens", usage.PromptTokens,
			"completion_tokens", usage.CompletionTokens,
		)
		if usage.CachedTokens > 0 {
			args = append(args, "cached_tokens", usage.CachedTokens)
		}
	}
	if finishReason != "" {
		args = append(args, "finish_reason", finishReason)
	}
	logger.InfoContext(ctx, "audit", args...)
}
