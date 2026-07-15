package provider_test

import (
	"context"
	"errors"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/JetManiack/go-ai-proxy/internal/domain"
	"github.com/JetManiack/go-ai-proxy/internal/provider"
	"github.com/JetManiack/go-ai-proxy/internal/testutil"
)

func TestRegistry_RoutesKnownModel(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "model-a"})
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)

	ctx := context.Background()
	if err := reg.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	p, ok := reg.ProviderFor("model-a")
	if !ok {
		t.Fatal("expected provider for model-a")
	}
	if p != fp {
		t.Error("returned wrong provider")
	}
}

func TestRegistry_UnknownModelReturnsFalse(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "model-a"})
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)

	if err := reg.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}

	_, ok := reg.ProviderFor("unknown-model")
	if ok {
		t.Error("expected no provider for unknown model")
	}
}

func TestRegistry_MultipleProviders(t *testing.T) {
	fp1 := testutil.NewFakeProvider(domain.Model{ID: "model-a"})
	fp2 := testutil.NewFakeProvider(domain.Model{ID: "model-b"})
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp1)
	reg.Register(fp2)

	if err := reg.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}

	p1, ok1 := reg.ProviderFor("model-a")
	p2, ok2 := reg.ProviderFor("model-b")

	if !ok1 || p1 != fp1 {
		t.Error("model-a should route to fp1")
	}
	if !ok2 || p2 != fp2 {
		t.Error("model-b should route to fp2")
	}
}

func TestRegistry_SkipsFailedProvider(t *testing.T) {
	good := testutil.NewFakeProvider(domain.Model{ID: "model-good"})
	bad := &testutil.FakeProvider{
		ModelsFunc: func(_ context.Context) ([]domain.Model, error) {
			return nil, errors.New("connection refused")
		},
	}

	reg := provider.NewRegistry(time.Hour)
	reg.Register(bad)
	reg.Register(good)

	if err := reg.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}

	_, ok := reg.ProviderFor("model-good")
	if !ok {
		t.Error("good provider should still be reachable")
	}
}

func TestRegistry_Models_ReturnsAllCachedModels(t *testing.T) {
	fp1 := testutil.NewFakeProvider(domain.Model{ID: "a"}, domain.Model{ID: "b"})
	fp2 := testutil.NewFakeProvider(domain.Model{ID: "c"})
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp1)
	reg.Register(fp2)

	if err := reg.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}

	models := reg.Models()
	if len(models) != 3 {
		t.Errorf("models len: got %d, want 3", len(models))
	}
}

func TestRegistry_GlobPatternMatchesModel(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "openai/*"})
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)

	if err := reg.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}

	// Concrete model ID should match the glob.
	p, ok := reg.ProviderFor("openai/gpt-4o")
	if !ok {
		t.Fatal("expected glob match for openai/gpt-4o")
	}
	if p != fp {
		t.Error("returned wrong provider")
	}

	// Exact pattern itself should also match.
	_, ok = reg.ProviderFor("openai/*")
	if !ok {
		t.Error("expected exact match for openai/*")
	}
}

func TestRegistry_GlobFirstProviderWinsOnConflict(t *testing.T) {
	first := testutil.NewFakeProvider(domain.Model{ID: "openai/*"})
	second := testutil.NewFakeProvider(domain.Model{ID: "openai/*"})
	reg := provider.NewRegistry(time.Hour)
	reg.Register(first)
	reg.Register(second)

	if err := reg.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}

	p, ok := reg.ProviderFor("openai/gpt-4o")
	if !ok {
		t.Fatal("expected a provider")
	}
	if p != first {
		t.Error("first registered provider should win on conflict")
	}
}

func TestRegistry_ExactFirstProviderWinsOnConflict(t *testing.T) {
	first := testutil.NewFakeProvider(domain.Model{ID: "gpt-4o"})
	second := testutil.NewFakeProvider(domain.Model{ID: "gpt-4o"})
	reg := provider.NewRegistry(time.Hour)
	reg.Register(first)
	reg.Register(second)

	if err := reg.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}

	p, ok := reg.ProviderFor("gpt-4o")
	if !ok {
		t.Fatal("expected a provider")
	}
	if p != first {
		t.Error("first registered provider should win on conflict")
	}
}

func TestRegistry_GlobExactMatchTakesPriority(t *testing.T) {
	exact := testutil.NewFakeProvider(domain.Model{ID: "openai/gpt-4o"})
	glob := testutil.NewFakeProvider(domain.Model{ID: "openai/*"})
	reg := provider.NewRegistry(time.Hour)
	reg.Register(exact)
	reg.Register(glob)

	if err := reg.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}

	p, ok := reg.ProviderFor("openai/gpt-4o")
	if !ok {
		t.Fatal("expected provider for openai/gpt-4o")
	}
	if p != exact {
		t.Error("exact match should take priority over glob")
	}
}

func TestRegistry_ProvidersFor_ReturnsSingleProvider(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatal(err)
	}

	ps := reg.ProvidersFor("m")
	if len(ps) != 1 || ps[0] != fp {
		t.Errorf("ProvidersFor: got %v, want [fp]", ps)
	}
}

func TestRegistry_ProvidersFor_ReturnsAllInRegistrationOrder(t *testing.T) {
	fp1 := testutil.NewFakeProvider(domain.Model{ID: "shared-model"})
	fp2 := testutil.NewFakeProvider(domain.Model{ID: "shared-model"})
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp1)
	reg.Register(fp2)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatal(err)
	}

	ps := reg.ProvidersFor("shared-model")
	if len(ps) != 2 {
		t.Fatalf("ProvidersFor: got %d providers, want 2", len(ps))
	}
	if ps[0] != fp1 || ps[1] != fp2 {
		t.Error("providers not in registration order")
	}
}

func TestRegistry_ProvidersFor_UnknownModelReturnsEmpty(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatal(err)
	}

	ps := reg.ProvidersFor("unknown")
	if len(ps) != 0 {
		t.Errorf("expected empty slice, got %v", ps)
	}
}

func TestRegistry_ProvidersFor_GlobReturnsAllMatching(t *testing.T) {
	fp1 := testutil.NewFakeProvider(domain.Model{ID: "openai/*"})
	fp2 := testutil.NewFakeProvider(domain.Model{ID: "openai/*"})
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp1)
	reg.Register(fp2)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatal(err)
	}

	ps := reg.ProvidersFor("openai/gpt-4o")
	if len(ps) != 2 {
		t.Fatalf("ProvidersFor glob: got %d providers, want 2", len(ps))
	}
	if ps[0] != fp1 || ps[1] != fp2 {
		t.Error("glob providers not in registration order")
	}
}

func TestRegistry_Refresh_MakesNewModelAvailable(t *testing.T) {
	models := []domain.Model{{ID: "model-v1"}}
	fp := &testutil.FakeProvider{
		ModelsFunc: func(_ context.Context) ([]domain.Model, error) {
			return models, nil
		},
	}

	reg := provider.NewRegistry(time.Hour) // long interval — no background refresh
	reg.Register(fp)

	if err := reg.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}

	// model-v2 is not visible yet.
	if _, ok := reg.ProviderFor("model-v2"); ok {
		t.Fatal("model-v2 should not exist before provider update")
	}

	// Simulate the provider gaining a new model.
	models = []domain.Model{{ID: "model-v1"}, {ID: "model-v2"}}

	// Explicit on-demand refresh.
	reg.Refresh(context.Background())

	if _, ok := reg.ProviderFor("model-v2"); !ok {
		t.Error("model-v2 should be visible after Refresh()")
	}
}

func TestRegistry_WithCapabilities_OverridesModelCaps(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "my-model"})
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp, provider.WithCapabilities(map[string][]string{
		"my-model": {"vision", "reasoning"},
	}))
	if err := reg.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	got := reg.ResolveModel("auto:vision,reasoning")
	if got != "my-model" {
		t.Errorf("WithCapabilities: got %q, want %q", got, "my-model")
	}
}

func TestRegistry_ResolveModel_PassthroughForRealID(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "gpt-4"})
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := reg.ResolveModel("gpt-4"); got != "gpt-4" {
		t.Errorf("ResolveModel passthrough: got %q, want %q", got, "gpt-4")
	}
}

func TestRegistry_ResolveModel_AutoSingleCap(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "vision-model", Capabilities: []string{"vision"}})
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	got := reg.ResolveModel("auto:vision")
	if got != "vision-model" {
		t.Errorf("ResolveModel auto: got %q, want %q", got, "vision-model")
	}
}

func TestRegistry_ResolveModel_AutoMultiCap(t *testing.T) {
	fp := testutil.NewFakeProvider(
		domain.Model{ID: "basic", Capabilities: []string{"vision"}},
		domain.Model{ID: "full", Capabilities: []string{"vision", "reasoning"}},
	)
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	got := reg.ResolveModel("auto:vision,reasoning")
	if got != "full" {
		t.Errorf("ResolveModel multi-cap: got %q, want %q", got, "full")
	}
}

func TestRegistry_ResolveModel_AutoNoMatch(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "text-model"})
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	got := reg.ResolveModel("auto:vision")
	if got != "auto:vision" {
		t.Errorf("ResolveModel no match: got %q, want unchanged %q", got, "auto:vision")
	}
}

func regF64p(v float64) *float64 { return &v }

func TestRegistry_DefaultsPricingToInf(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"}) // no pricing set → nil
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	models := reg.Models()
	if models[0].InputCostPerToken == nil || !math.IsInf(*models[0].InputCostPerToken, 1) {
		t.Errorf("InputCostPerToken: got %v, want +Inf", models[0].InputCostPerToken)
	}
	if models[0].OutputCostPerToken == nil || !math.IsInf(*models[0].OutputCostPerToken, 1) {
		t.Errorf("OutputCostPerToken: got %v, want +Inf", models[0].OutputCostPerToken)
	}
}

func TestRegistry_PreservesKnownPricing(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m", InputCostPerToken: regF64p(0.000005), OutputCostPerToken: regF64p(0.000015)})
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	models := reg.Models()
	if models[0].InputCostPerToken == nil || *models[0].InputCostPerToken != 0.000005 {
		t.Errorf("InputCostPerToken: got %v, want 0.000005", models[0].InputCostPerToken)
	}
	if models[0].OutputCostPerToken == nil || *models[0].OutputCostPerToken != 0.000015 {
		t.Errorf("OutputCostPerToken: got %v, want 0.000015", models[0].OutputCostPerToken)
	}
}

func TestRegistry_PreservesZeroPricing(t *testing.T) {
	zero := 0.0
	fp := testutil.NewFakeProvider(domain.Model{ID: "m", InputCostPerToken: &zero, OutputCostPerToken: &zero})
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	models := reg.Models()
	if models[0].InputCostPerToken == nil || *models[0].InputCostPerToken != 0 {
		t.Errorf("InputCostPerToken: got %v, want 0 (free model)", models[0].InputCostPerToken)
	}
}

func TestRegistry_GlobPricingIsInf(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "openai/*", InputCostPerToken: regF64p(0), OutputCostPerToken: regF64p(0)})
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	models := reg.Models()
	if models[0].InputCostPerToken == nil || !math.IsInf(*models[0].InputCostPerToken, 1) {
		t.Errorf("glob model should have +Inf pricing, got %v", models[0].InputCostPerToken)
	}
}

func TestCandidatesFor_ConcreteID(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	cs := reg.CandidatesFor("m")
	if len(cs) != 1 || cs[0].ModelID != "m" || cs[0].Provider != fp {
		t.Errorf("CandidatesFor concrete: got %v", cs)
	}
}

func TestCandidatesFor_UnknownID_ReturnsEmpty(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "m"})
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	if cs := reg.CandidatesFor("unknown"); len(cs) != 0 {
		t.Errorf("expected empty, got %v", cs)
	}
}

func TestCandidatesFor_AutoSelector_SingleModel(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "vision-model", Capabilities: []string{"vision"}})
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	cs := reg.CandidatesFor("auto:vision")
	if len(cs) != 1 || cs[0].ModelID != "vision-model" || cs[0].Provider != fp {
		t.Errorf("CandidatesFor auto single: got %v", cs)
	}
}

func TestCandidatesFor_AutoSelector_MultipleModels(t *testing.T) {
	fp1 := testutil.NewFakeProvider(domain.Model{ID: "vision-a", Capabilities: []string{"vision"}})
	fp2 := testutil.NewFakeProvider(domain.Model{ID: "vision-b", Capabilities: []string{"vision"}})
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp1)
	reg.Register(fp2)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	cs := reg.CandidatesFor("auto:vision")
	if len(cs) != 2 {
		t.Fatalf("CandidatesFor multi-model: got %d candidates, want 2", len(cs))
	}
	ids := map[string]bool{}
	for _, c := range cs {
		ids[c.ModelID] = true
	}
	if !ids["vision-a"] || !ids["vision-b"] {
		t.Errorf("expected both vision-a and vision-b, got %v", cs)
	}
}

func TestCandidatesFor_AutoSelector_NoMatch_ReturnsEmpty(t *testing.T) {
	fp := testutil.NewFakeProvider(domain.Model{ID: "text-model"})
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	if cs := reg.CandidatesFor("auto:vision"); len(cs) != 0 {
		t.Errorf("expected empty, got %v", cs)
	}
}

func TestRegistry_RefreshUpdatesIndex(t *testing.T) {
	var mu sync.Mutex
	models := []domain.Model{{ID: "model-v1"}}
	fp := &testutil.FakeProvider{
		ModelsFunc: func(_ context.Context) ([]domain.Model, error) {
			mu.Lock()
			defer mu.Unlock()
			return models, nil
		},
	}

	reg := provider.NewRegistry(50 * time.Millisecond)
	reg.Register(fp)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := reg.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	// Update the model list on the provider side.
	mu.Lock()
	models = []domain.Model{{ID: "model-v2"}}
	mu.Unlock()

	// Wait for the background refresh.
	time.Sleep(200 * time.Millisecond)

	_, oldOk := reg.ProviderFor("model-v1")
	_, newOk := reg.ProviderFor("model-v2")

	if oldOk {
		t.Error("model-v1 should no longer be in the index after refresh")
	}
	if !newOk {
		t.Error("model-v2 should be in the index after refresh")
	}
}

func TestCapabilitiesFor(t *testing.T) {
	fp := testutil.NewFakeProvider(
		domain.Model{ID: "with-caps", Capabilities: []string{"structured_output", "vision"}},
		domain.Model{ID: "no-caps"},
	)
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp)
	if err := reg.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}

	caps, known := reg.CapabilitiesFor("with-caps")
	if !known {
		t.Fatal("with-caps: known=false, want true")
	}
	if len(caps) != 2 || caps[0] != "structured_output" {
		t.Errorf("with-caps caps: got %v", caps)
	}

	caps, known = reg.CapabilitiesFor("no-caps")
	if !known {
		t.Fatal("no-caps: known=false, want true")
	}
	if len(caps) != 0 {
		t.Errorf("no-caps caps: got %v, want empty", caps)
	}

	if _, known := reg.CapabilitiesFor("absent"); known {
		t.Error("absent: known=true, want false")
	}
}

func TestWithContextLengths_ConfigWins(t *testing.T) {
	fp := testutil.NewFakeProvider(
		domain.Model{ID: "reported", MaxModelLen: 4096},
		domain.Model{ID: "unreported"}, // MaxModelLen 0
		domain.Model{ID: "untouched", MaxModelLen: 8192},
	)
	reg := provider.NewRegistry(time.Hour)
	reg.Register(fp, provider.WithContextLengths(map[string]int{
		"reported":   131072, // overrides the provider-reported 4096
		"unreported": 32768,  // fills in where provider reported nothing
	}))
	if err := reg.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}

	byID := map[string]domain.Model{}
	for _, m := range reg.Models() {
		byID[m.ID] = m
	}
	if got := byID["reported"].MaxModelLen; got != 131072 {
		t.Errorf("reported: config should win, got %d want 131072", got)
	}
	if got := byID["unreported"].MaxModelLen; got != 32768 {
		t.Errorf("unreported: override should fill in, got %d want 32768", got)
	}
	if got := byID["untouched"].MaxModelLen; got != 8192 {
		t.Errorf("untouched: should keep provider value, got %d want 8192", got)
	}
}
