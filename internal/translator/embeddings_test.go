package translator_test

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
	"github.com/JetManiack/go-ai-proxy/internal/translator"
)

// --- EmbedRequestFromOpenAI ---

func TestEmbedRequestFromOpenAI_StringInput(t *testing.T) {
	body := `{"model": "text-embedding-3-small", "input": "hello world"}`
	req, err := translator.EmbedRequestFromOpenAI([]byte(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if req.Model != "text-embedding-3-small" {
		t.Errorf("model: got %q", req.Model)
	}
	if len(req.Input) != 1 || req.Input[0] != "hello world" {
		t.Errorf("input: got %#v, want [\"hello world\"]", req.Input)
	}
	if req.EncodingFormat != "float" {
		t.Errorf("encoding_format: got %q, want default %q", req.EncodingFormat, "float")
	}
	if req.Dimensions != nil {
		t.Errorf("dimensions: got %v, want nil", req.Dimensions)
	}
}

func TestEmbedRequestFromOpenAI_ArrayInput(t *testing.T) {
	body := `{"model": "m", "input": ["a", "b", "c"]}`
	req, err := translator.EmbedRequestFromOpenAI([]byte(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := []string{"a", "b", "c"}
	if len(req.Input) != len(want) {
		t.Fatalf("input len: got %d, want %d", len(req.Input), len(want))
	}
	for i := range want {
		if req.Input[i] != want[i] {
			t.Errorf("input[%d]: got %q, want %q", i, req.Input[i], want[i])
		}
	}
}

func TestEmbedRequestFromOpenAI_EncodingFormatAndDimensions(t *testing.T) {
	body := `{"model": "m", "input": "x", "encoding_format": "base64", "dimensions": 256}`
	req, err := translator.EmbedRequestFromOpenAI([]byte(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if req.EncodingFormat != "base64" {
		t.Errorf("encoding_format: got %q, want %q", req.EncodingFormat, "base64")
	}
	if req.Dimensions == nil || *req.Dimensions != 256 {
		t.Errorf("dimensions: got %v, want 256", req.Dimensions)
	}
}

func TestEmbedRequestFromOpenAI_UnsupportedInputType(t *testing.T) {
	body := `{"model": "m", "input": [1, 2, 3]}`
	if _, err := translator.EmbedRequestFromOpenAI([]byte(body)); err == nil {
		t.Fatal("expected error for token-ID array input, got nil")
	}
}

// --- EmbedRequestToOpenAI ---

func TestEmbedRequestToOpenAI_AlwaysArrayAndFloat(t *testing.T) {
	req := domain.EmbedRequest{
		Model:          "m",
		Input:          []string{"hello"},
		EncodingFormat: "base64", // caller wanted base64; upstream call must still force float
	}
	body, err := translator.EmbedRequestToOpenAI(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var wire map[string]any
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if wire["encoding_format"] != "float" {
		t.Errorf("encoding_format: got %v, want %q", wire["encoding_format"], "float")
	}
	input, ok := wire["input"].([]any)
	if !ok || len(input) != 1 || input[0] != "hello" {
		t.Errorf("input: got %#v, want array [\"hello\"]", wire["input"])
	}
	if _, present := wire["dimensions"]; present {
		t.Errorf("dimensions should be omitted when nil, got %v", wire["dimensions"])
	}
}

func TestEmbedRequestToOpenAI_DimensionsIncludedWhenSet(t *testing.T) {
	dims := 512
	req := domain.EmbedRequest{Model: "m", Input: []string{"x"}, Dimensions: &dims}
	body, err := translator.EmbedRequestToOpenAI(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var wire map[string]any
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if wire["dimensions"] != float64(512) {
		t.Errorf("dimensions: got %v, want 512", wire["dimensions"])
	}
}

// --- EmbedResponseFromOpenAI ---

func TestEmbedResponseFromOpenAI_ParsesFloats(t *testing.T) {
	body := `{
		"object": "list",
		"model": "text-embedding-3-small",
		"data": [
			{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
			{"object": "embedding", "index": 1, "embedding": [-0.5, 1.0]}
		],
		"usage": {"prompt_tokens": 7, "total_tokens": 7}
	}`
	resp, err := translator.EmbedResponseFromOpenAI([]byte(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Model != "text-embedding-3-small" {
		t.Errorf("model: got %q", resp.Model)
	}
	if resp.Usage.PromptTokens != 7 {
		t.Errorf("prompt tokens: got %d, want 7", resp.Usage.PromptTokens)
	}
	if len(resp.Embeddings) != 2 {
		t.Fatalf("embeddings len: got %d, want 2", len(resp.Embeddings))
	}
	if resp.Embeddings[0].Index != 0 || len(resp.Embeddings[0].Values) != 3 || resp.Embeddings[0].Values[1] != 0.2 {
		t.Errorf("embedding[0]: got %+v", resp.Embeddings[0])
	}
	if resp.Embeddings[1].Index != 1 || resp.Embeddings[1].Values[0] != -0.5 {
		t.Errorf("embedding[1]: got %+v", resp.Embeddings[1])
	}
}

// --- EmbedResponseToOpenAI ---

func TestEmbedResponseToOpenAI_FloatFormat(t *testing.T) {
	resp := domain.EmbedResponse{
		Model:      "m",
		Embeddings: []domain.Embedding{{Index: 0, Values: []float64{0.25, -0.75}}},
		Usage:      domain.Usage{PromptTokens: 3},
	}
	body, err := translator.EmbedResponseToOpenAI(resp, "float")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var wire struct {
		Object string `json:"object"`
		Data   []struct {
			Embedding []float64 `json:"embedding"`
			Index     int       `json:"index"`
		} `json:"data"`
		Model string `json:"model"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if wire.Object != "list" {
		t.Errorf("object: got %q, want %q", wire.Object, "list")
	}
	if len(wire.Data) != 1 || len(wire.Data[0].Embedding) != 2 || wire.Data[0].Embedding[0] != 0.25 {
		t.Errorf("data: got %+v", wire.Data)
	}
	if wire.Usage.PromptTokens != 3 || wire.Usage.TotalTokens != 3 {
		t.Errorf("usage: got %+v", wire.Usage)
	}
}

func TestEmbedResponseToOpenAI_Base64Format(t *testing.T) {
	original := []float64{0.1, -0.2, 3.5}
	resp := domain.EmbedResponse{
		Model:      "m",
		Embeddings: []domain.Embedding{{Index: 0, Values: original}},
	}
	body, err := translator.EmbedResponseToOpenAI(resp, "base64")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var wire struct {
		Data []struct {
			Embedding string `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if len(wire.Data) != 1 {
		t.Fatalf("data len: got %d, want 1", len(wire.Data))
	}

	raw, err := base64.StdEncoding.DecodeString(wire.Data[0].Embedding)
	if err != nil {
		t.Fatalf("invalid base64: %v", err)
	}
	if len(raw) != len(original)*4 {
		t.Fatalf("decoded byte len: got %d, want %d", len(raw), len(original)*4)
	}
	for i, want := range original {
		bits := binary.LittleEndian.Uint32(raw[i*4 : i*4+4])
		got := float64(math.Float32frombits(bits))
		if math.Abs(got-want) > 1e-6 {
			t.Errorf("value[%d]: got %v, want ~%v (float32 precision)", i, got, want)
		}
	}
}

func TestEmbedResponseToOpenAI_DefaultsToFloatWhenEncodingFormatEmpty(t *testing.T) {
	resp := domain.EmbedResponse{
		Model:      "m",
		Embeddings: []domain.Embedding{{Index: 0, Values: []float64{1, 2}}},
	}
	body, err := translator.EmbedResponseToOpenAI(resp, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var wire struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("invalid JSON (expected plain float array, not base64 string): %v", err)
	}
	if len(wire.Data) != 1 || len(wire.Data[0].Embedding) != 2 {
		t.Errorf("data: got %+v", wire.Data)
	}
}
