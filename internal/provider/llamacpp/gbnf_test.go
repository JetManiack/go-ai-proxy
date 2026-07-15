package llamacpp

import (
	"strings"
	"testing"
)

func TestSchemaToGBNF_ClassificationEnum(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"category": map[string]any{
				"type": "string",
				"enum": []any{"spam", "billing", "support"},
			},
		},
		"required": []any{"category"},
	}
	got, err := schemaToGBNF(schema)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := "root ::= obj0\n" +
		"obj0 ::= \"{\" space \"\\\"category\\\"\" space \":\" space (\"\\\"spam\\\"\" | \"\\\"billing\\\"\" | \"\\\"support\\\"\") space \"}\"\n" +
		"space ::= | \" \" | \"\\n\"{1,2} [ \\t]{0,20}\n"
	if got != want {
		t.Errorf("grammar mismatch:\n got: %q\nwant: %q", got, want)
	}
}

func TestSchemaToGBNF_PrimitiveProperties(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
			"age":  map[string]any{"type": "integer"},
		},
	}
	got, err := schemaToGBNF(schema)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// properties emitted sorted by name: age (integer), then name (string)
	want := "root ::= obj0\n" +
		"obj0 ::= \"{\" space \"\\\"age\\\"\" space \":\" space integer \",\" space \"\\\"name\\\"\" space \":\" space string space \"}\"\n" +
		"space ::= | \" \" | \"\\n\"{1,2} [ \\t]{0,20}\n" +
		"char ::= [^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})\n" +
		"string ::= \"\\\"\" char* \"\\\"\"\n" +
		"integral-part ::= [0] | [1-9] [0-9]{0,15}\n" +
		"integer ::= (\"-\"? integral-part)\n"
	if got != want {
		t.Errorf("grammar mismatch:\n got: %q\nwant: %q", got, want)
	}
}

func TestSchemaToGBNF_UnsupportedKeyword(t *testing.T) {
	schema := map[string]any{
		"type":    "string",
		"pattern": "^[a-z]+$",
	}
	_, err := schemaToGBNF(schema)
	if err == nil {
		t.Fatal("expected error for unsupported keyword 'pattern', got nil")
	}
	if !strings.Contains(err.Error(), "pattern") {
		t.Errorf("error should name the unsupported keyword, got: %v", err)
	}
}

func TestSchemaToGBNF_MissingType(t *testing.T) {
	if _, err := schemaToGBNF(map[string]any{}); err == nil {
		t.Fatal("expected error for missing type, got nil")
	}
}
