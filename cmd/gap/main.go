// Command gap is the Go AI Proxy — a universal OpenAI-compatible HTTP proxy.
package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/urfave/cli/v3"
	"gopkg.in/yaml.v3"

	"github.com/JetManiack/go-ai-proxy/internal/auth"
	"github.com/JetManiack/go-ai-proxy/internal/domain"
	"github.com/JetManiack/go-ai-proxy/internal/metrics"
	"github.com/JetManiack/go-ai-proxy/internal/provider"
	anthropicprovider "github.com/JetManiack/go-ai-proxy/internal/provider/anthropic"
	litellmprovider  "github.com/JetManiack/go-ai-proxy/internal/provider/litellm"
	lmstudioprovider "github.com/JetManiack/go-ai-proxy/internal/provider/lmstudio"
	openaiprovider   "github.com/JetManiack/go-ai-proxy/internal/provider/openai"
	vllmprovider     "github.com/JetManiack/go-ai-proxy/internal/provider/vllm"
	"github.com/JetManiack/go-ai-proxy/internal/server"
)

// config mirrors the structure of config.yaml.
type config struct {
	Server struct {
		Host            string        `yaml:"host"`
		Port            int           `yaml:"port"`
		RefreshInterval time.Duration `yaml:"refresh_interval"`
		MaxBodyBytes    int64         `yaml:"max_request_body_bytes"` // 0 = no limit
		ShutdownTimeout time.Duration `yaml:"shutdown_timeout"`       // default 30s
		AuditLog  bool `yaml:"audit_log"`  // emit audit entries via slog
		Metrics   bool `yaml:"metrics"`    // expose /metrics in Prometheus text format
		RateLimit *struct {
			RPS        float64 `yaml:"rps"`         // sustained requests per second
			Burst      int     `yaml:"burst"`       // max burst size
			PerCaller  bool    `yaml:"per_caller"`  // false = global, true = per Authorization/IP
			TrustProxy bool    `yaml:"trust_proxy"` // honour X-Forwarded-For for IP-based limiting
		} `yaml:"rate_limit"`
	} `yaml:"server"`
	Providers []providerConfig `yaml:"providers"`
}

// providerConfig describes one provider instance.
// Type selects the implementation; other fields are type-specific.
type providerConfig struct {
	Name    string        `yaml:"name"`
	Type    string        `yaml:"type"`     // "openai" or "anthropic"
	BaseURL string        `yaml:"base_url"` // openai only
	Timeout time.Duration `yaml:"timeout"`  // optional; 0 = no timeout
	Auth    *struct {
		Method    string   `yaml:"method"`     // "api_key" or "oauth"
		APIKey    string   `yaml:"api_key"`    // single key; supports ${ENV_VAR}
		APIKeys   []string `yaml:"api_keys"`   // multiple keys for round-robin; supports ${ENV_VAR} per entry
		TokenFile string   `yaml:"token_file"` // oauth: path to token cache
	} `yaml:"auth"`
	TokenBudget *struct {
		Default int            `yaml:"default"` // max_tokens limit for any model from this provider; 0 = no limit
		Models  map[string]int `yaml:"models"`  // per-model overrides
	} `yaml:"token_budget"`
	MaxConcurrent      int                 `yaml:"max_concurrent"`       // max parallel requests to this provider; 0 = unlimited
	QueueSize          int                 `yaml:"queue_size"`           // max queued requests when at max_concurrent; 0 = unlimited
	RequestTimeout     time.Duration       `yaml:"request_timeout"`      // streaming: max wait for first token; non-streaming: total response timeout
	ModelCapabilities  map[string][]string `yaml:"model_capabilities"`   // manual capability override: model ID → ["vision", "reasoning", ...]

	// TLS controls outgoing HTTPS behaviour. Currently honoured only by
	// openai, lmstudio, and litellm providers.
	TLS *struct {
		Insecure bool `yaml:"insecure"` // skip TLS certificate verification — use only with self-hosted endpoints
	} `yaml:"tls"`

	// vllm-specific fields (see provider/vllm for details).
	VLLM *struct {
		ThinkingKey string   `yaml:"thinking_key"` // chat_template_kwargs key; default "enable_thinking"
		Models      []string `yaml:"models"`       // fallback model IDs when discover is off or upstream fails
		Discover    *bool    `yaml:"discover"`     // call upstream /v1/models (default true)
	} `yaml:"vllm"`
}

func loadConfig(path string) (config, error) {
	var cfg config
	cfg.Server.Host = "127.0.0.1"
	cfg.Server.Port = 8090
	cfg.Server.RefreshInterval = time.Hour
	cfg.Server.ShutdownTimeout = 30 * time.Second

	if path == "" {
		return cfg, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return cfg, fmt.Errorf("open config: %w", err)
	}
	defer f.Close()

	dec := yaml.NewDecoder(f)
	dec.KnownFields(true) // reject unrecognised keys
	if err := dec.Decode(&cfg); err != nil {
		return cfg, fmt.Errorf("parse config: %w", err)
	}
	if cfg.Server.RefreshInterval == 0 {
		cfg.Server.RefreshInterval = time.Hour
	}
	if cfg.Server.ShutdownTimeout == 0 {
		cfg.Server.ShutdownTimeout = 30 * time.Second
	}
	return cfg, nil
}

func main() {
	app := &cli.Command{
		Name:  "gap",
		Usage: "Go AI Proxy — universal OpenAI-compatible LLM proxy",
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:    "config",
				Aliases: []string{"c"},
				Usage:   "path to config.yaml",
				Sources: cli.EnvVars("GAP_CONFIG"),
			},
			&cli.StringFlag{
				Name:    "host",
				Usage:   "host to listen on",
				Sources: cli.EnvVars("GAP_HOST"),
			},
			&cli.IntFlag{
				Name:    "port",
				Usage:   "port to listen on",
				Sources: cli.EnvVars("GAP_PORT"),
			},
			&cli.StringFlag{
				Name:    "log-level",
				Usage:   "log level: debug, info, warn, error",
				Value:   "info",
				Sources: cli.EnvVars("GAP_LOG_LEVEL"),
			},
		},
		Action: run,
	}

	if err := app.Run(context.Background(), os.Args); err != nil {
		slog.Error("fatal", "error", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, cmd *cli.Command) error {
	var level slog.Level
	if err := level.UnmarshalText([]byte(cmd.String("log-level"))); err != nil {
		return fmt.Errorf("invalid --log-level %q: use debug, info, warn, or error", cmd.String("log-level"))
	}
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level})))

	cfg, err := loadConfig(cmd.String("config"))
	if err != nil {
		return err
	}
	if cmd.IsSet("host") {
		cfg.Server.Host = cmd.String("host")
	}
	if cmd.IsSet("port") {
		cfg.Server.Port = cmd.Int("port")
	}

	reg := provider.NewRegistry(cfg.Server.RefreshInterval)

	// providerCfgs maps a registered provider instance back to its config (for budget lookup).
	providerCfgs := map[domain.Provider]providerConfig{}
	for _, pc := range cfg.Providers {
		p, err := buildProvider(ctx, pc)
		if err != nil {
			return fmt.Errorf("provider %q: %w", pc.Name, err)
		}
		if pc.MaxConcurrent > 0 || pc.RequestTimeout > 0 {
			p = provider.NewBounded(p, pc.MaxConcurrent, pc.QueueSize, pc.RequestTimeout)
			slog.Info("registering provider", "name", pc.Name, "type", pc.Type,
				"max_concurrent", pc.MaxConcurrent, "queue_size", pc.QueueSize,
				"request_timeout", pc.RequestTimeout)
		} else {
			slog.Info("registering provider", "name", pc.Name, "type", pc.Type)
		}
		var regOpts []provider.RegisterOption
		if len(pc.ModelCapabilities) > 0 {
			regOpts = append(regOpts, provider.WithCapabilities(pc.ModelCapabilities))
		}
		reg.Register(p, regOpts...)
		providerCfgs[p] = pc
	}

	if err := reg.Start(ctx); err != nil {
		return fmt.Errorf("start registry: %w", err)
	}

	var srvOpts []server.Option
	if cfg.Server.MaxBodyBytes > 0 {
		srvOpts = append(srvOpts, server.WithMaxBodyBytes(cfg.Server.MaxBodyBytes))
	}
	// Build token budget: for each known model, look up the provider's budget config.
	budgetModels := map[string]int{}
	for _, m := range reg.Models() {
		p, ok := reg.ProviderFor(m.ID)
		if !ok {
			continue
		}
		pc, ok := providerCfgs[p]
		if !ok || pc.TokenBudget == nil {
			continue
		}
		if limit, ok := pc.TokenBudget.Models[m.ID]; ok {
			budgetModels[m.ID] = limit
		} else if pc.TokenBudget.Default > 0 {
			budgetModels[m.ID] = pc.TokenBudget.Default
		}
	}
	if len(budgetModels) > 0 {
		srvOpts = append(srvOpts, server.WithTokenBudget(server.TokenBudgetConfig{
			Models: budgetModels,
		}))
	}
	if cfg.Server.AuditLog || level <= slog.LevelDebug {
		srvOpts = append(srvOpts, server.WithAuditLog(slog.Default()))
	}
	if cfg.Server.Metrics {
		srvOpts = append(srvOpts, server.WithMetrics(metrics.New()))
	}
	if rl := cfg.Server.RateLimit; rl != nil {
		srvOpts = append(srvOpts, server.WithRateLimit(server.RateLimitConfig{
			RPS:        rl.RPS,
			Burst:      rl.Burst,
			PerCaller:  rl.PerCaller,
			TrustProxy: rl.TrustProxy,
		}))
	}
	srv := server.New(reg, srvOpts...)

	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	slog.Info("gap listening", "addr", addr)

	httpSrv := &http.Server{Addr: addr, Handler: srv}

	// Graceful shutdown: wait for SIGINT/SIGTERM, then drain in-flight requests.
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		select {
		case <-stop:
		case <-ctx.Done():
		}
		slog.Info("shutting down", "drain_timeout", cfg.Server.ShutdownTimeout)
		shutCtx, cancel := context.WithTimeout(context.Background(), cfg.Server.ShutdownTimeout)
		defer cancel()
		if err := httpSrv.Shutdown(shutCtx); err != nil {
			slog.Error("shutdown error", "error", err)
		}
	}()

	if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		return fmt.Errorf("listen: %w", err)
	}
	return nil
}

func buildProvider(ctx context.Context, pc providerConfig) (domain.Provider, error) {
	if pc.TLS != nil && pc.TLS.Insecure {
		switch pc.Type {
		case "openai", "lmstudio", "litellm", "vllm":
			// supported below
		default:
			return nil, fmt.Errorf("tls.insecure is not supported for provider type %q (only openai, lmstudio, litellm, vllm)", pc.Type)
		}
	}
	switch pc.Type {
	case "openai":
		if pc.BaseURL == "" {
			return nil, fmt.Errorf("base_url is required for openai provider")
		}
		opts := []openaiprovider.Option{openaiprovider.WithName(pc.Name)}
		if pc.Auth != nil {
			if key := os.ExpandEnv(pc.Auth.APIKey); key != "" {
				opts = append(opts, openaiprovider.WithAPIKey(key))
			}
		}
		if pc.Timeout > 0 {
			opts = append(opts, openaiprovider.WithTimeout(pc.Timeout))
		}
		if pc.TLS != nil && pc.TLS.Insecure {
			slog.Warn("TLS certificate verification disabled for provider — accepts any upstream cert",
				"provider", pc.Name, "type", pc.Type, "base_url", pc.BaseURL)
			opts = append(opts, openaiprovider.WithTLSInsecure(true))
		}
		return openaiprovider.New(pc.BaseURL, opts...), nil

	case "lmstudio":
		if pc.BaseURL == "" {
			return nil, fmt.Errorf("base_url is required for lmstudio provider")
		}
		opts := []lmstudioprovider.Option{lmstudioprovider.WithName(pc.Name)}
		if pc.Timeout > 0 {
			opts = append(opts, lmstudioprovider.WithTimeout(pc.Timeout))
		}
		if pc.TLS != nil && pc.TLS.Insecure {
			slog.Warn("TLS certificate verification disabled for provider — accepts any upstream cert",
				"provider", pc.Name, "type", pc.Type, "base_url", pc.BaseURL)
			opts = append(opts, lmstudioprovider.WithTLSInsecure(true))
		}
		return lmstudioprovider.New(pc.BaseURL, opts...), nil

	case "litellm":
		if pc.BaseURL == "" {
			return nil, fmt.Errorf("base_url is required for litellm provider")
		}
		opts := []litellmprovider.Option{litellmprovider.WithName(pc.Name)}
		if pc.Auth != nil {
			if key := os.ExpandEnv(pc.Auth.APIKey); key != "" {
				opts = append(opts, litellmprovider.WithAPIKey(key))
			}
		}
		if pc.Timeout > 0 {
			opts = append(opts, litellmprovider.WithTimeout(pc.Timeout))
		}
		if pc.TLS != nil && pc.TLS.Insecure {
			slog.Warn("TLS certificate verification disabled for provider — accepts any upstream cert",
				"provider", pc.Name, "type", pc.Type, "base_url", pc.BaseURL)
			opts = append(opts, litellmprovider.WithTLSInsecure(true))
		}
		return litellmprovider.New(pc.BaseURL, opts...), nil

	case "anthropic":
		a, err := buildAuthenticator(pc)
		if err != nil {
			return nil, err
		}
		opts := []anthropicprovider.Option{anthropicprovider.WithName(pc.Name)}
		if pc.Timeout > 0 {
			opts = append(opts, anthropicprovider.WithTimeout(pc.Timeout))
		}
		return anthropicprovider.New(a, opts...), nil

	case "vllm":
		if pc.BaseURL == "" {
			return nil, fmt.Errorf("base_url is required for vllm provider")
		}
		if len(pc.ModelCapabilities) == 0 {
			return nil, fmt.Errorf("model_capabilities is required for vllm provider — vLLM upstreams do not report capabilities")
		}
		cfg := vllmprovider.Config{
			Name:              pc.Name,
			BaseURL:           pc.BaseURL,
			Timeout:           pc.Timeout,
			Discover:          true, // default
			ModelCapabilities: pc.ModelCapabilities,
		}
		if pc.Auth != nil {
			if key := os.ExpandEnv(pc.Auth.APIKey); key != "" {
				cfg.APIKey = key
			}
		}
		if pc.TLS != nil && pc.TLS.Insecure {
			slog.Warn("TLS certificate verification disabled for provider — accepts any upstream cert",
				"provider", pc.Name, "type", pc.Type, "base_url", pc.BaseURL)
			cfg.TLSInsecure = true
		}
		if v := pc.VLLM; v != nil {
			if v.ThinkingKey != "" {
				cfg.ThinkingKey = v.ThinkingKey
			}
			if v.Models != nil {
				cfg.Models = v.Models
			}
			if v.Discover != nil {
				cfg.Discover = *v.Discover
			}
		}
		return vllmprovider.New(ctx, cfg)

	default:
		return nil, fmt.Errorf("unknown provider type %q (use \"openai\", \"lmstudio\", \"litellm\", \"vllm\", or \"anthropic\")", pc.Type)
	}
}

func buildAuthenticator(pc providerConfig) (auth.Authenticator, error) {
	if pc.Auth == nil {
		return nil, fmt.Errorf("auth section is required for anthropic provider")
	}
	switch pc.Auth.Method {
	case "api_key":
		// Collect all keys: api_keys list takes precedence; api_key is appended if set.
		var keys []string
		for _, k := range pc.Auth.APIKeys {
			if expanded := os.ExpandEnv(k); expanded != "" {
				keys = append(keys, expanded)
			}
		}
		if single := os.ExpandEnv(pc.Auth.APIKey); single != "" {
			keys = append(keys, single)
		}
		if len(keys) == 0 {
			return nil, fmt.Errorf("api_key is empty (set auth.api_key, auth.api_keys, or the referenced env vars)")
		}
		if len(keys) == 1 {
			return auth.NewAPIKey(keys[0]), nil
		}
		return auth.NewRoundRobin(keys), nil

	case "oauth", "":
		tokenFile := pc.Auth.TokenFile
		if tokenFile == "" {
			tokenFile = "~/.config/go-ai-proxy/anthropic.json"
		}
		return auth.NewOAuth(auth.OAuthConfig{
			AuthURL:   "https://claude.ai/oauth/authorize",
			TokenURL:  "https://claude.ai/oauth/token",
			ClientID:  "gap-proxy",
			TokenFile: tokenFile,
			Scopes:    []string{"openid", "email"},
		}), nil

	default:
		return nil, fmt.Errorf("unknown auth method %q (use \"api_key\" or \"oauth\")", pc.Auth.Method)
	}
}
