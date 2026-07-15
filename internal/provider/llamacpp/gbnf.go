// Package llamacpp implements a Provider for llama.cpp servers, converting
// OpenAI response_format json_schema into a GBNF grammar (llama.cpp's schema
// path crashes the Gemma-3 sampler; a raw grammar field works).
package llamacpp

import (
	"fmt"
	"sort"
	"strings"
)

// primDefs are llama.cpp's own JSON building blocks, matching the reference
// converter examples/json_schema_to_grammar.py: primitives carry NO trailing
// space; whitespace is added at composition points. Only referenced blocks are
// emitted (see usePrim / primOrder).
var primDefs = map[string]string{
	"space":         `space ::= | " " | "\n"{1,2} [ \t]{0,20}`,
	"char":          `char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})`,
	"string":        `string ::= "\"" char* "\""`,
	"integral-part": `integral-part ::= [0] | [1-9] [0-9]{0,15}`,
	"decimal-part":  `decimal-part ::= [0-9]{1,16}`,
	"number":        `number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?`,
	"integer":       `integer ::= ("-"? integral-part)`,
	"boolean":       `boolean ::= ("true" | "false")`,
}

// primDeps lists the building blocks each block references. Primitives do NOT
// depend on space (space is added by object/array composition, not primitives).
var primDeps = map[string][]string{
	"string":  {"char"},
	"number":  {"integral-part", "decimal-part"},
	"integer": {"integral-part"},
}

// primOrder fixes emission order for deterministic output.
var primOrder = []string{"space", "char", "string", "integral-part", "decimal-part", "number", "integer", "boolean"}

// unsupportedKeywords are JSON Schema constructs the v1 converter rejects.
var unsupportedKeywords = []string{
	"$ref", "$defs", "definitions", "allOf", "pattern", "format",
	"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
	"multipleOf", "minItems", "maxItems", "minLength", "maxLength",
}

type gbnfBuilder struct {
	rules    []string        // generated named rules (objN, arrN) in emission order
	prims    map[string]bool // referenced building blocks
	objCount int
	arrCount int
}

// schemaToGBNF converts a supported JSON Schema subset into a llama.cpp GBNF
// grammar. Unsupported constructs return an error naming the offending keyword.
func schemaToGBNF(schema map[string]any) (string, error) {
	b := &gbnfBuilder{prims: map[string]bool{}}
	rootExpr, err := b.value(schema)
	if err != nil {
		return "", err
	}
	var sb strings.Builder
	fmt.Fprintf(&sb, "root ::= %s\n", rootExpr)
	for _, r := range b.rules {
		sb.WriteString(r)
		sb.WriteByte('\n')
	}
	for _, name := range primOrder {
		if b.prims[name] {
			sb.WriteString(primDefs[name])
			sb.WriteByte('\n')
		}
	}
	return sb.String(), nil
}

// usePrim marks a building block (and its dependencies) as referenced.
func (b *gbnfBuilder) usePrim(name string) {
	if b.prims[name] {
		return
	}
	b.prims[name] = true
	for _, d := range primDeps[name] {
		b.usePrim(d)
	}
}

// value returns a GBNF expression matching one schema node.
func (b *gbnfBuilder) value(schema map[string]any) (string, error) {
	for _, k := range unsupportedKeywords {
		if _, present := schema[k]; present {
			return "", fmt.Errorf("llamacpp: unsupported JSON Schema keyword %q; supply an explicit grammar field for this schema", k)
		}
	}

	if enumRaw, ok := schema["enum"]; ok {
		return b.enum(enumRaw)
	}

	typ, _ := schema["type"].(string)
	switch typ {
	case "string":
		b.usePrim("string")
		return "string", nil
	case "integer":
		b.usePrim("integer")
		return "integer", nil
	case "number":
		b.usePrim("number")
		return "number", nil
	case "boolean":
		b.usePrim("boolean")
		return "boolean", nil
	case "object":
		return b.object(schema)
	default:
		return "", fmt.Errorf("llamacpp: unsupported or missing schema type %q; supply an explicit grammar field for this schema", typ)
	}
}

// enum builds an alternation of string-literal terminals.
func (b *gbnfBuilder) enum(enumRaw any) (string, error) {
	list, ok := enumRaw.([]any)
	if !ok || len(list) == 0 {
		return "", fmt.Errorf("llamacpp: enum must be a non-empty array")
	}
	alts := make([]string, 0, len(list))
	for _, v := range list {
		s, ok := v.(string)
		if !ok {
			return "", fmt.Errorf("llamacpp: non-string enum value %v not supported; supply an explicit grammar", v)
		}
		alts = append(alts, gbnfJSONString(s))
	}
	return "(" + strings.Join(alts, " | ") + ")", nil
}

// object builds a named rule for a JSON object; properties are emitted sorted
// by name and all treated as required (v1 simplification).
func (b *gbnfBuilder) object(schema map[string]any) (string, error) {
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		return "", fmt.Errorf("llamacpp: object schema without a 'properties' map is not supported; supply an explicit grammar")
	}
	name := fmt.Sprintf("obj%d", b.objCount)
	b.objCount++

	keys := make([]string, 0, len(props))
	for k := range props {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	b.usePrim("space")
	var parts []string
	parts = append(parts, `"{" space`)
	for i, k := range keys {
		propSchema, ok := props[k].(map[string]any)
		if !ok {
			return "", fmt.Errorf("llamacpp: property %q has a non-object schema", k)
		}
		valExpr, err := b.value(propSchema)
		if err != nil {
			return "", err
		}
		kv := fmt.Sprintf(`%s space ":" space %s`, gbnfJSONString(k), valExpr)
		if i > 0 {
			parts = append(parts, `"," space`)
		}
		parts = append(parts, kv)
	}
	parts = append(parts, `space "}"`)
	rule := name + " ::= " + strings.Join(parts, " ")
	b.rules = append(b.rules, rule)
	return name, nil
}

// gbnfJSONString returns a GBNF terminal matching the JSON string form of s
// (s wrapped in double quotes). Escaping mirrors the reference converter's
// _format_literal (GRAMMAR_LITERAL_ESCAPES): backslash, quote, and the \n \r \t
// control characters.
func gbnfJSONString(s string) string {
	e := strings.ReplaceAll(s, `\`, `\\`)
	e = strings.ReplaceAll(e, `"`, `\"`)
	e = strings.ReplaceAll(e, "\n", `\n`)
	e = strings.ReplaceAll(e, "\r", `\r`)
	e = strings.ReplaceAll(e, "\t", `\t`)
	return `"\"` + e + `\""`
}
