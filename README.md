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

Existing models are overridden by default so the wrapper/proxy `/models` metadata wins over models.dev metadata.

If the model list does not expose max output tokens, the plugin uses `128000` as `limit.output`.

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
- `providers.include`: optional allowlist; empty or omitted means include all matching providers.
- `providers.exclude`: skip these provider ids.
- `overrideExisting`: defaults to `true`.

Default cache path:

```text
~/.cache/opencode-models-discovery/models-cache.json
```

Restart opencode after changing plugin or config files.
