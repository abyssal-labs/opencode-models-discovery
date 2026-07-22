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
- `max_model_len` and `max_context_length` -> `limit.context`
- `metadata.input_context_window` -> `limit.input`
- `metadata.max_output_tokens` -> `limit.output`
- `max_completion_tokens` and `max_tokens` -> `limit.output`
- `metadata.display_name` -> `name`
- `metadata.input_modalities` -> `modalities.input`
- `metadata.output_modalities` -> `modalities.output`
- `architecture.input_modalities` -> `modalities.input`
- `architecture.output_modalities` -> `modalities.output`

Modality names are normalized to lowercase, and `vision` is normalized to `image` for opencode compatibility.

Existing models are overridden by default so the wrapper/proxy `/models` metadata wins over models.dev metadata. With the default `overrideExisting: true`, providers expose only models returned by the `/models` endpoint. This is especially useful for custom `openai` base URLs because opencode otherwise starts from its built-in OpenAI model catalog.

Unknown limits are left unchanged for existing models. Newly discovered OpenAI models use the provider template's limits unless `fallbackContextTokens` or `fallbackOutputTokens` is configured.

## Modes and service tiers

The plugin creates additional model entries when a response provides `experimental.modes`, `modes`, or `additional_speed_tiers`. A mode named `fast` for model `example` becomes `example-fast`.

Mode metadata can provide:

- `provider.body`, converted from snake_case to opencode option names.
- `provider.headers`, restricted to string values.
- `cost`, including input, output, cache-read, and cache-write prices.
- Service tiers through `additional_speed_tiers` and `service_tiers`; these set the OpenAI `serviceTier` request option.

Paginated responses can use `next_page`, `next`, or `has_more` with `last_id`. Continuation URLs must remain on the original origin.

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

For the built-in `openai` provider, discovery uses API credentials configured through OpenCode's `/connect` flow when `options.apiKey` is not set. Credentials are supplied by OpenCode's plugin runtime and are never read from its on-disk auth store. An explicit `options.apiKey` takes precedence.

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
        "maxPages": 10,
        "timeoutMs": 10000,
        "headers": {
          "x-api-key": "{env:DISCOVERY_API_KEY}"
        },
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
- `maxPages`: maximum number of same-origin discovery pages; defaults to `10`.
- `timeoutMs`: timeout for each discovery page request; defaults to 10 seconds.
- `headers`: additional headers for `/models` discovery; values support `{env:NAME}` expansion.
- `providers.include`: optional allowlist; empty or omitted means include all matching providers.
- `providers.exclude`: skip these provider ids.
- `overrideExisting`: defaults to `true`; when `true`, providers expose only discovered models. When `false`, discovered models are merged into the existing model catalog without replacing existing definitions.

Invalid values and unknown option names are rejected during plugin initialization so configuration mistakes fail visibly.

TypeScript consumers can import `PluginOptions` and `ProviderDiscoveryOptions` from the package.

Default cache path:

```text
~/.cache/opencode-models-discovery/models-cache.json
```

`XDG_CACHE_HOME` is honored when set. On Windows, `LOCALAPPDATA` is used before the home-directory fallback.

## Compatibility

- Tested against `@opencode-ai/plugin` 1.17.x.
- Published output is standard ESM and requires Node.js 20 or newer when imported outside opencode.
- Discovery endpoints must use HTTP or HTTPS and return an array, `{ "data": [] }`, or `{ "models": [] }`.

## Troubleshooting

Discovery failures are written to opencode logs under the `opencode-models-discovery` service. Logs include the provider, sanitized endpoint, HTTP status when available, and whether stale cache data was used. Credentials and URL query strings are not logged.

Set `refreshIntervalMs` to `0` to force discovery during each startup while debugging. Delete the cache file to discard all previously discovered metadata.

Release history and upgrade notes are available on the [GitHub Releases](https://github.com/abyssal-labs/opencode-models-discovery/releases) page.

Restart opencode after changing plugin or config files.
