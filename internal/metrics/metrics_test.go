package metrics_test

import (
	"strings"
	"testing"

	"github.com/JetManiack/go-ai-proxy/internal/metrics"
)

func TestMetrics_RecordRequest_IncrementsCounter(t *testing.T) {
	m := metrics.New()
	m.RecordRequest("claude-3", "anthropic", "success", 42)
	m.RecordRequest("claude-3", "anthropic", "success", 10)
	m.RecordRequest("claude-3", "anthropic", "error", 5)

	out := prometheusOutput(m)

	if !containsLine(out, `gap_requests_total{model="claude-3",provider="anthropic",status="success"} 2`) {
		t.Errorf("expected success counter=2, output:\n%s", out)
	}
	if !containsLine(out, `gap_requests_total{model="claude-3",provider="anthropic",status="error"} 1`) {
		t.Errorf("expected error counter=1, output:\n%s", out)
	}
}

func TestMetrics_RecordRequest_AccumulatesDuration(t *testing.T) {
	m := metrics.New()
	m.RecordRequest("m", "p", "success", 100)
	m.RecordRequest("m", "p", "success", 200)

	out := prometheusOutput(m)
	if !containsLine(out, `gap_request_duration_ms_total{model="m",provider="p"} 300`) {
		t.Errorf("expected duration sum=300, output:\n%s", out)
	}
}

func TestMetrics_RecordTokens_PerProvider(t *testing.T) {
	m := metrics.New()
	m.RecordTokens("anthropic", "claude-3", 100, 50, 0)
	m.RecordTokens("anthropic", "claude-3", 200, 30, 0)
	m.RecordTokens("openai", "gpt-4", 50, 10, 0)

	out := prometheusOutput(m)
	if !containsLine(out, `gap_tokens_total{provider="anthropic",model="claude-3",type="prompt"} 300`) {
		t.Errorf("expected anthropic/claude-3 prompt=300, output:\n%s", out)
	}
	if !containsLine(out, `gap_tokens_total{provider="anthropic",model="claude-3",type="completion"} 80`) {
		t.Errorf("expected anthropic/claude-3 completion=80, output:\n%s", out)
	}
	if !containsLine(out, `gap_tokens_total{provider="openai",model="gpt-4",type="prompt"} 50`) {
		t.Errorf("expected openai/gpt-4 prompt=50, output:\n%s", out)
	}
}

func TestMetrics_WritePrometheus_ContainsHelpAndType(t *testing.T) {
	m := metrics.New()
	m.RecordRequest("m", "p", "success", 1)
	m.RecordTokens("p", "m", 10, 5, 0)

	out := prometheusOutput(m)
	for _, want := range []string{
		"# HELP gap_requests_total",
		"# TYPE gap_requests_total counter",
		"# HELP gap_request_duration_ms_total",
		"# TYPE gap_request_duration_ms_total counter",
		"# HELP gap_tokens_total",
		"# TYPE gap_tokens_total counter",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in output:\n%s", want, out)
		}
	}
}

func TestMetrics_ZeroValues_NotEmitted(t *testing.T) {
	m := metrics.New()
	out := prometheusOutput(m)
	// No recorded data → no metric lines (only HELP/TYPE headers are optional too)
	if strings.Contains(out, "gap_requests_total{") {
		t.Errorf("unexpected metric lines with zero data:\n%s", out)
	}
}

func TestMetrics_Concurrent_NoRace(t *testing.T) {
	m := metrics.New()
	done := make(chan struct{})
	for range 50 {
		go func() {
			m.RecordRequest("m", "p", "success", 10)
			m.RecordTokens("p", "m", 5, 3, 0)
			done <- struct{}{}
		}()
	}
	for range 50 {
		<-done
	}
	// just verify no panic/race
	prometheusOutput(m)
}

func TestRecordTokens_IncludesCached(t *testing.T) {
	m := metrics.New()
	m.RecordTokens("p", "m", 100, 50, 80)
	out := prometheusOutput(m)
	if !containsLine(out, `gap_tokens_total{provider="p",model="m",type="prompt"} 100`) {
		t.Errorf("missing prompt counter; got:\n%s", out)
	}
	if !containsLine(out, `gap_tokens_total{provider="p",model="m",type="completion"} 50`) {
		t.Errorf("missing completion counter; got:\n%s", out)
	}
	if !containsLine(out, `gap_tokens_total{provider="p",model="m",type="cached"} 80`) {
		t.Errorf("missing cached counter; got:\n%s", out)
	}
}

func TestRecordTokens_OmitsCachedWhenZero(t *testing.T) {
	m := metrics.New()
	m.RecordTokens("p", "m", 10, 5, 0)
	out := prometheusOutput(m)
	if strings.Contains(out, `type="cached"`) {
		t.Errorf("should not emit cached counter when 0; got:\n%s", out)
	}
}

func prometheusOutput(m *metrics.Metrics) string {
	var sb strings.Builder
	m.WritePrometheus(&sb)
	return sb.String()
}

func containsLine(s, line string) bool {
	for _, l := range strings.Split(s, "\n") {
		if l == line {
			return true
		}
	}
	return false
}
