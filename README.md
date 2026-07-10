# opencode-models-discovery

Local opencode plugin that refreshes model metadata from OpenAI-compatible `/models` endpoints and applies it to opencode provider config.

This is for OpenAI-compatible wrappers/proxies that set `provider.<name>.options.baseURL`. Native OpenAI OAuth is skipped because it does not use a custom `baseURL` and opencode already handles Codex OAuth model limits internally.

It targets custom-baseURL providers using the OpenAI SDK paths:

- Provider id `openai`
- `npm: "@ai-sdk/openai"`
- `npm: "@ai-sdk/openai-compatible"`

All matching providers are included by default. Use `providers.include` as an optional allowlist; when it is empty or omitted, all matching providers are included. Use `providers.exclude` to skip a provider.

For every discovered model, it maps endpoint metadata into opencode model config:

- `metadata.context_window` -> `limit.context`
- `metadata.context_length` or top-level `context_length` -> `limit.context`
- `metadata.input_context_window` -> `limit.input`
- `metadata.max_output_tokens` -> `limit.output`
- `metadata.display_name` -> `name`
- `metadata.input_modalities` -> `modalities.input`
- `metadata.output_modalities` -> `modalities.output`
- `architecture.input_modalities` -> `modalities.input`
- `architecture.output_modalities` -> `modalities.output`

Existing models are overridden by default so the wrapper/proxy `/models` metadata wins over models.dev metadata. With the default `overrideExisting: true`, providers expose only models returned by the `/models` endpoint. This is especially useful for custom `openai` base URLs because opencode otherwise starts from its built-in OpenAI model catalog.

Unknown limits are left unchanged for existing models. Newly discovered OpenAI models use the provider template's limits unless `fallbackContextTokens` or `fallbackOutputTokens` is configured.

## Usage

```json
{
  "plugin": ["@abyssal-labs/opencode-models-discovery"],
  "provider": {
    "openai": {
      "options": {
        "modelsDiscovery": {
          "refreshIntervalHours": 24
        }
      }
    },
    "my-openai-compatible-provider": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "https://example.com/v1",
        "apiKey": "{env:MY_PROVIDER_API_KEY}"
      }
    }
  }
}
```

## Options

Default refresh interval is 24 hours. Cached values are still applied on startup when the cache is fresh; the endpoint is only rechecked after the interval.

```json
{
  "plugin": [
    [
      "@abyssal-labs/opencode-models-discovery",
      {
        "refreshIntervalHours": 12,
        "fallbackContextTokens": 128000,
        "fallbackOutputTokens": 16384,
        "maxResponseBytes": 5242880,
        "overrideExisting": true,
        "providers": {
          "include": [],
          "exclude": ["provider-to-skip"]
        }
      }
    ]
  ]
}
```

Provider-level overrides can live under `provider.<name>.options.modelsDiscovery`:

```json
{
  "provider": {
    "my-openai-compatible-provider": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "https://example.com/v1",
        "apiKey": "{env:MY_PROVIDER_API_KEY}",
        "modelsDiscovery": {
          "refreshIntervalHours": 6,
          "overrideExisting": true
        }
      }
    }
  }
}
```

Supported options:

- `enabled`: set `false` to disable globally.
- `refreshIntervalHours`: defaults to `24`.
- `refreshIntervalMs`: millisecond form, takes precedence over hours.
- `cachePath`: global option for cache file location.
- `fallbackContextTokens`: optional context limit for newly discovered OpenAI models without context metadata.
- `fallbackOutputTokens`: optional output limit for newly discovered OpenAI models without output metadata.
- `maxResponseBytes`: maximum discovery response size in bytes; defaults to 5 MiB.
- `providers.include`: optional allowlist; empty or omitted means include all matching providers.
- `providers.exclude`: skip these provider ids.
- `overrideExisting`: defaults to `true`; when `true`, providers expose only discovered models. When `false`, discovered models are merged into the existing model catalog without replacing existing definitions.

Default cache path:

```text
~/.cache/opencode-models-discovery/models-cache.json
```

Restart opencode after changing plugin or config files.
