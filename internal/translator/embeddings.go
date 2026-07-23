package translator

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
)

// --- Embeddings wire format types (unexported) ---

type oaEmbedRequest struct {
	Model          string          `json:"model"`
	Input          json.RawMessage `json:"input"`
	EncodingFormat string          `json:"encoding_format,omitempty"`
	Dimensions     *int            `json:"dimensions,omitempty"`
}

type oaEmbedResponse struct {
	Object string        `json:"object"`
	Data   []oaEmbedData `json:"data"`
	Model  string        `json:"model"`
	Usage  oaEmbedUsage  `json:"usage"`
}

type oaEmbedData struct {
	Object    string `json:"object"`
	Embedding any    `json:"embedding"`
	Index     int    `json:"index"`
}

type oaEmbedUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// --- Embeddings ---

// EmbedRequestFromOpenAI parses an OpenAI-compatible /v1/embeddings request body.
func EmbedRequestFromOpenAI(body []byte) (domain.EmbedRequest, error) {
	var oa oaEmbedRequest
	if err := json.Unmarshal(body, &oa); err != nil {
		return domain.EmbedRequest{}, fmt.Errorf("decode embeddings request: %w", err)
	}

	input, err := decodeEmbedInput(oa.Input)
	if err != nil {
		return domain.EmbedRequest{}, err
	}

	return domain.EmbedRequest{
		Model:          oa.Model,
		Input:          input,
		EncodingFormat: firstNonEmpty(oa.EncodingFormat, "float"),
		Dimensions:     oa.Dimensions,
	}, nil
}

// decodeEmbedInput normalizes the OpenAI "input" field, which may be a single
// string or an array of strings. Token-ID array input (an older, rarer
// OpenAI variant) is not supported.
func decodeEmbedInput(raw json.RawMessage) ([]string, error) {
	var arr []string
	if err := json.Unmarshal(raw, &arr); err == nil {
		return arr, nil
	}
	var single string
	if err := json.Unmarshal(raw, &single); err == nil {
		return []string{single}, nil
	}
	return nil, fmt.Errorf("embeddings request: unsupported input type (expected string or array of strings)")
}

// EmbedRequestToOpenAI encodes a canonical EmbedRequest for an upstream
// OpenAI-compatible /v1/embeddings call. It always sends "input" as an array
// and forces encoding_format to "float" upstream — gap always wants raw
// floats back so it can re-encode per the original caller's requested format.
func EmbedRequestToOpenAI(req domain.EmbedRequest) ([]byte, error) {
	inputJSON, err := json.Marshal(req.Input)
	if err != nil {
		return nil, fmt.Errorf("encode embeddings input: %w", err)
	}
	oa := oaEmbedRequest{
		Model:          req.Model,
		Input:          inputJSON,
		EncodingFormat: "float",
		Dimensions:     req.Dimensions,
	}
	return json.Marshal(oa)
}

// EmbedResponseFromOpenAI parses an OpenAI-compatible /v1/embeddings response.
func EmbedResponseFromOpenAI(body []byte) (domain.EmbedResponse, error) {
	var oa oaEmbedResponse
	if err := json.Unmarshal(body, &oa); err != nil {
		return domain.EmbedResponse{}, fmt.Errorf("decode embeddings response: %w", err)
	}

	embeddings := make([]domain.Embedding, 0, len(oa.Data))
	for _, d := range oa.Data {
		values, err := decodeEmbeddingValues(d.Embedding)
		if err != nil {
			return domain.EmbedResponse{}, err
		}
		embeddings = append(embeddings, domain.Embedding{Index: d.Index, Values: values})
	}

	return domain.EmbedResponse{
		Model:      oa.Model,
		Embeddings: embeddings,
		Usage:      domain.Usage{PromptTokens: oa.Usage.PromptTokens},
	}, nil
}

// decodeEmbeddingValues accepts either a JSON float array (the shape gap
// always requests upstream via EmbedRequestToOpenAI) or a base64 string, in
// case an upstream ignores the encoding_format override.
func decodeEmbeddingValues(embedding any) ([]float64, error) {
	switch v := embedding.(type) {
	case []any:
		values := make([]float64, len(v))
		for i, item := range v {
			f, ok := item.(float64)
			if !ok {
				return nil, fmt.Errorf("embeddings response: non-numeric value in embedding array")
			}
			values[i] = f
		}
		return values, nil
	case string:
		return decodeBase64Float32(v)
	default:
		return nil, fmt.Errorf("embeddings response: unsupported embedding value type %T", embedding)
	}
}

// EmbedResponseToOpenAI encodes a canonical EmbedResponse in the shape the
// original caller requested via encodingFormat ("float" or "base64"; empty
// defaults to "float", matching OpenAI's own default).
func EmbedResponseToOpenAI(resp domain.EmbedResponse, encodingFormat string) ([]byte, error) {
	base64Encoding := encodingFormat == "base64"

	data := make([]oaEmbedData, len(resp.Embeddings))
	for i, e := range resp.Embeddings {
		var embedding any
		if base64Encoding {
			embedding = encodeBase64Float32(e.Values)
		} else {
			embedding = e.Values
		}
		data[i] = oaEmbedData{Object: "embedding", Embedding: embedding, Index: e.Index}
	}

	oa := oaEmbedResponse{
		Object: "list",
		Data:   data,
		Model:  resp.Model,
		Usage: oaEmbedUsage{
			PromptTokens: resp.Usage.PromptTokens,
			TotalTokens:  resp.Usage.PromptTokens,
		},
	}
	return json.Marshal(oa)
}

// decodeBase64Float32 decodes a base64 string of little-endian float32
// values, matching how OpenAI/openai-python encode embeddings when
// encoding_format is "base64".
func decodeBase64Float32(s string) ([]float64, error) {
	raw, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return nil, fmt.Errorf("embeddings response: decode base64 embedding: %w", err)
	}
	if len(raw)%4 != 0 {
		return nil, fmt.Errorf("embeddings response: base64 embedding length %d not a multiple of 4", len(raw))
	}
	values := make([]float64, len(raw)/4)
	for i := range values {
		bits := binary.LittleEndian.Uint32(raw[i*4 : i*4+4])
		values[i] = float64(math.Float32frombits(bits))
	}
	return values, nil
}

// encodeBase64Float32 encodes values as little-endian float32 bytes, base64.
// float64→float32 narrowing matches OpenAI's own encoding (embedding models
// don't carry float64 precision, and clients decode assuming float32).
func encodeBase64Float32(values []float64) string {
	raw := make([]byte, len(values)*4)
	for i, v := range values {
		bits := math.Float32bits(float32(v))
		binary.LittleEndian.PutUint32(raw[i*4:i*4+4], bits)
	}
	return base64.StdEncoding.EncodeToString(raw)
}
