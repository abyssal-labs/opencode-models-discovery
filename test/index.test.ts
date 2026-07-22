import assert from "node:assert/strict"
import { access, mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { createServer } from "node:http"
import { tmpdir } from "node:os"
import { join } from "node:path"
import test from "node:test"
import plugin from "../src/index.ts"

test("discovers models for an OpenAI-compatible provider", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))

  const originalFetch = globalThis.fetch
  globalThis.fetch = async () =>
    Response.json({
      data: [
        {
          id: "example-model",
          name: "Example Model",
          metadata: {
            context_window: 32_000,
            max_output_tokens: 8_000,
            input_modalities: ["text", "image"],
            output_modalities: ["text"],
          },
        },
      ],
    })
  t.after(() => {
    globalThis.fetch = originalFetch
  })

  const hooks = await plugin({} as never, { cachePath: join(directory, "cache.json") })
  const config: {
    provider: Record<
      string,
      {
        npm: string
        options: { baseURL: string }
        models?: Record<string, unknown>
      }
    >
  } = {
    provider: {
      proxy: {
        npm: "@ai-sdk/openai-compatible",
        options: { baseURL: "https://proxy.example/v1" },
      },
    },
  }

  await hooks.config?.(config as never)

  assert.deepEqual(config.provider.proxy.models, {
    "example-model": {
      id: "example-model",
      name: "Example Model",
      limit: { context: 32_000, output: 8_000 },
      modalities: { input: ["text", "image"], output: ["text"] },
    },
  })
})

test("uses the discovered ID for OpenAI API requests", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))

  const originalFetch = globalThis.fetch
  globalThis.fetch = async () =>
    Response.json({
      data: [
        {
          id: "discovered-model",
          modes: { fast: { provider: { body: { service_tier: "priority" } } } },
        },
      ],
    })
  t.after(() => {
    globalThis.fetch = originalFetch
  })

  const hooks = await plugin({} as never, {
    cachePath: join(directory, "cache.json"),
    fallbackContextTokens: 16_000,
    fallbackOutputTokens: 2_000,
  })
  await hooks.config?.({ provider: { openai: { options: { baseURL: "https://proxy.example/v1" } } } } as never)

  const template = providerModel("template-model")
  const models = await hooks.provider?.models?.(
    {
      id: "openai",
      name: "OpenAI",
      source: "config",
      env: [],
      options: {},
      models: { "template-model": template },
    },
    {},
  )

  assert.equal(models?.["discovered-model"].id, "discovered-model")
  assert.equal(models?.["discovered-model"].api.id, "discovered-model")
  assert.deepEqual(models?.["discovered-model"].limit, { context: 16_000, output: 2_000 })
  assert.equal(models?.["discovered-model-fast"].id, "discovered-model-fast")
  assert.equal(models?.["discovered-model-fast"].api.id, "discovered-model")
  assert.equal(models?.["discovered-model-fast"].options.serviceTier, "priority")
})

test("uses OpenCode connect credentials for OpenAI discovery", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))

  const originalFetch = globalThis.fetch
  let authorization = ""
  globalThis.fetch = async (_input, init) => {
    authorization = new Headers(init?.headers).get("authorization") ?? ""
    return Response.json({ data: [{ id: "connected-model" }] })
  }
  t.after(() => {
    globalThis.fetch = originalFetch
  })

  const hooks = await plugin({} as never, { cachePath: join(directory, "cache.json") })
  await hooks.config?.({ provider: { openai: { options: { baseURL: "https://proxy.example/v1" } } } } as never)
  const models = await hooks.provider?.models?.(
    {
      id: "openai",
      name: "OpenAI",
      source: "config",
      env: [],
      options: {},
      models: { "template-model": providerModel("template-model") },
    },
    { auth: { type: "api", key: "connected-secret" } },
  )

  assert.equal(authorization, "Bearer connected-secret")
  assert.ok(models?.["connected-model"])
})

test("prefers a configured OpenAI key over connect credentials", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))

  const originalFetch = globalThis.fetch
  let authorization = ""
  globalThis.fetch = async (_input, init) => {
    authorization = new Headers(init?.headers).get("authorization") ?? ""
    return Response.json({ data: [] })
  }
  t.after(() => {
    globalThis.fetch = originalFetch
  })

  const hooks = await plugin({} as never, { cachePath: join(directory, "cache.json") })
  await hooks.config?.({
    provider: { openai: { options: { baseURL: "https://proxy.example/v1", apiKey: "configured-secret" } } },
  } as never)
  await hooks.provider?.models?.(
    {
      id: "openai",
      name: "OpenAI",
      source: "config",
      env: [],
      options: {},
      models: { "template-model": providerModel("template-model") },
    },
    { auth: { type: "api", key: "connected-secret" } },
  )

  assert.equal(authorization, "Bearer configured-secret")
})

test("does not copy model-specific metadata into newly discovered OpenAI models", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))

  const originalFetch = globalThis.fetch
  globalThis.fetch = async () =>
    Response.json({ data: [{ id: "discovered-model", metadata: { context_window: 8_000, max_output_tokens: 1_000 } }] })
  t.after(() => {
    globalThis.fetch = originalFetch
  })

  const hooks = await plugin({} as never, { cachePath: join(directory, "cache.json") })
  await hooks.config?.({ provider: { openai: { options: { baseURL: "https://proxy.example/v1" } } } } as never)
  const models = await hooks.provider?.models?.(
    {
      id: "openai",
      name: "OpenAI",
      source: "config",
      env: [],
      options: {},
      models: { "template-model": providerModel("template-model") },
    },
    {},
  )

  const discovered = models?.["discovered-model"]
  assert.deepEqual(discovered?.capabilities, {
    temperature: false,
    reasoning: false,
    attachment: false,
    toolcall: false,
    input: { text: true, audio: false, image: false, video: false, pdf: false },
    output: { text: true, audio: false, image: false, video: false, pdf: false },
    interleaved: false,
  })
  assert.deepEqual(discovered?.cost, { input: 0, output: 0, cache: { read: 0, write: 0 } })
  assert.deepEqual(discovered?.limit, { context: 8_000, output: 1_000 })
  assert.equal(discovered?.status, "active")
  assert.equal(discovered?.release_date, "")
  assert.deepEqual(discovered?.options, {})
  assert.deepEqual(discovered?.headers, {})
})

test("preserves existing OpenAI models when overriding is disabled", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))

  const originalFetch = globalThis.fetch
  globalThis.fetch = async () =>
    Response.json({
      data: [
        { id: "existing-model", name: "Remote Name", metadata: { context_window: 8_000 } },
        { id: "new-model", name: "New Model", metadata: { context_window: 4_000 } },
      ],
    })
  t.after(() => {
    globalThis.fetch = originalFetch
  })

  const hooks = await plugin({} as never, {
    cachePath: join(directory, "cache.json"),
    overrideExisting: false,
  })
  await hooks.config?.({ provider: { openai: { options: { baseURL: "https://proxy.example/v1" } } } } as never)

  const existing = providerModel("existing-model")
  const models = await hooks.provider?.models?.(
    {
      id: "openai",
      name: "OpenAI",
      source: "config",
      env: [],
      options: {},
      models: { "existing-model": existing },
    },
    {},
  )

  assert.strictEqual(models?.["existing-model"], existing)
  assert.equal(models?.["new-model"].name, "New Model")
})

test("retries discovery after a transient failure", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))
  const cachePath = join(directory, "cache.json")

  const responses = [
    Response.json({ data: [{ id: "stale-model" }] }),
    new Response("unavailable", { status: 503 }),
    Response.json({ data: [{ id: "fresh-model" }] }),
  ]
  const originalFetch = globalThis.fetch
  let fetchCount = 0
  globalThis.fetch = async () => responses[fetchCount++]
  t.after(() => {
    globalThis.fetch = originalFetch
  })

  const staleHooks = await plugin({} as never, { cachePath, refreshIntervalMs: 0 })
  await staleHooks.config?.(compatibleConfig() as never)

  const cache = JSON.parse(await readFile(cachePath, "utf8"))
  const cacheKey = Object.keys(cache.providers)[0]
  cache.providers[cacheKey].checkedAt = 0
  await writeFile(cachePath, JSON.stringify(cache))

  const failedHooks = await plugin({} as never, { cachePath })
  const failedConfig = compatibleConfig()
  await failedHooks.config?.(failedConfig as never)
  assert.ok(failedConfig.provider.proxy.models?.["stale-model"])

  const retryHooks = await plugin({} as never, { cachePath })
  const retryConfig = compatibleConfig()
  await retryHooks.config?.(retryConfig as never)
  assert.equal(fetchCount, 3)
  assert.ok(retryConfig.provider.proxy.models?.["fresh-model"])
})

test("maps output limits without requiring context metadata", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))

  const originalFetch = globalThis.fetch
  globalThis.fetch = async () =>
    Response.json({ data: [{ id: "output-only", metadata: { max_output_tokens: 4_096 } }] })
  t.after(() => {
    globalThis.fetch = originalFetch
  })

  const hooks = await plugin({} as never, { cachePath: join(directory, "cache.json") })
  const config = compatibleConfig()
  await hooks.config?.(config as never)

  assert.deepEqual(config.provider.proxy.models?.["output-only"], {
    id: "output-only",
    name: "output-only",
    limit: { output: 4_096 },
  })
})

test("does not invent output limits for custom providers", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))

  const originalFetch = globalThis.fetch
  globalThis.fetch = async () => Response.json({ data: [{ id: "context-only", metadata: { context_window: 8_000 } }] })
  t.after(() => {
    globalThis.fetch = originalFetch
  })

  const hooks = await plugin({} as never, { cachePath: join(directory, "cache.json") })
  const config = compatibleConfig()
  await hooks.config?.(config as never)

  assert.deepEqual(config.provider.proxy.models?.["context-only"], {
    id: "context-only",
    name: "context-only",
    limit: { context: 8_000 },
  })
})

test("ignores malformed cache data", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))
  const cachePath = join(directory, "cache.json")
  await writeFile(cachePath, JSON.stringify({ providers: "invalid" }))

  const originalFetch = globalThis.fetch
  globalThis.fetch = async () => Response.json({ data: [{ id: "valid-model" }] })
  t.after(() => {
    globalThis.fetch = originalFetch
  })

  const hooks = await plugin({} as never, { cachePath })
  const config = compatibleConfig()
  await hooks.config?.(config as never)

  assert.ok(config.provider.proxy.models?.["valid-model"])
})

test("treats a successful empty response as authoritative", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))

  const originalFetch = globalThis.fetch
  globalThis.fetch = async () => Response.json({ data: [] })
  t.after(() => {
    globalThis.fetch = originalFetch
  })

  const hooks = await plugin({} as never, { cachePath: join(directory, "cache.json") })
  const config = compatibleConfig()
  config.provider.proxy.models = { existing: { id: "existing" } }
  await hooks.config?.(config as never)

  assert.deepEqual(config.provider.proxy.models, {})
})

test("gives provider exclusions precedence over inclusions", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))

  const originalFetch = globalThis.fetch
  let fetched = false
  globalThis.fetch = async () => {
    fetched = true
    return Response.json({ data: [{ id: "model" }] })
  }
  t.after(() => {
    globalThis.fetch = originalFetch
  })

  const hooks = await plugin({} as never, {
    cachePath: join(directory, "cache.json"),
    providers: { include: ["proxy"], exclude: ["proxy"] },
  })
  const config = compatibleConfig()
  await hooks.config?.(config as never)

  assert.equal(fetched, false)
  assert.equal(config.provider.proxy.models, undefined)
})

test("constructs the models URL without corrupting query parameters", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))

  const originalFetch = globalThis.fetch
  let requestedURL = ""
  globalThis.fetch = async (input) => {
    requestedURL = input.toString()
    return Response.json({ data: [] })
  }
  t.after(() => {
    globalThis.fetch = originalFetch
  })

  const hooks = await plugin({} as never, { cachePath: join(directory, "cache.json") })
  const config = compatibleConfig()
  config.provider.proxy.options.baseURL = "https://proxy.example/v1/?api-version=2026-01-01#ignored"
  await hooks.config?.(config as never)

  assert.equal(requestedURL, "https://proxy.example/v1/models?api-version=2026-01-01")
})

test("ignores malformed model entries without discarding valid models", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))

  const originalFetch = globalThis.fetch
  globalThis.fetch = async () => Response.json({ data: [null, 42, {}, { id: "valid-model" }] })
  t.after(() => {
    globalThis.fetch = originalFetch
  })

  const hooks = await plugin({} as never, { cachePath: join(directory, "cache.json") })
  const config = compatibleConfig()
  await hooks.config?.(config as never)

  assert.deepEqual(Object.keys(config.provider.proxy.models ?? {}), ["valid-model"])
})

test("rejects discovery responses larger than the configured limit", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))

  const originalFetch = globalThis.fetch
  globalThis.fetch = async () =>
    new Response(JSON.stringify({ data: [{ id: "model" }] }), {
      headers: { "content-length": "100" },
    })
  t.after(() => {
    globalThis.fetch = originalFetch
  })

  const hooks = await plugin({} as never, {
    cachePath: join(directory, "cache.json"),
    maxResponseBytes: 50,
  })
  const config = compatibleConfig()
  config.provider.proxy.models = { existing: { id: "existing" } }
  await hooks.config?.(config as never)

  assert.deepEqual(config.provider.proxy.models, { existing: { id: "existing" } })
})

test("sends configured discovery headers", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))

  const originalFetch = globalThis.fetch
  const originalKey = process.env.DISCOVERY_TEST_KEY
  process.env.DISCOVERY_TEST_KEY = "secret"
  let requestHeaders = new Headers()
  globalThis.fetch = async (_input, init) => {
    requestHeaders = new Headers(init?.headers)
    return Response.json({ data: [] })
  }
  t.after(() => {
    globalThis.fetch = originalFetch
    if (originalKey === undefined) delete process.env.DISCOVERY_TEST_KEY
    else process.env.DISCOVERY_TEST_KEY = originalKey
  })

  const hooks = await plugin({} as never, {
    cachePath: join(directory, "cache.json"),
    headers: {
      "x-api-key": "{env:DISCOVERY_TEST_KEY}",
      authorization: "Basic custom",
    },
  })
  await hooks.config?.(compatibleConfig() as never)

  assert.equal(requestHeaders.get("x-api-key"), "secret")
  assert.equal(requestHeaders.get("authorization"), "Basic custom")
  assert.equal(requestHeaders.get("accept"), "application/json")
})

test("scopes cached models to discovery credentials", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))
  const cachePath = join(directory, "cache.json")

  const originalFetch = globalThis.fetch
  let fetchCount = 0
  globalThis.fetch = async () => Response.json({ data: [{ id: `model-${++fetchCount}` }] })
  t.after(() => {
    globalThis.fetch = originalFetch
  })

  const firstHooks = await plugin({} as never, { cachePath })
  const firstConfig = compatibleConfig()
  firstConfig.provider.proxy.options.apiKey = "first-key"
  await firstHooks.config?.(firstConfig as never)

  const secondHooks = await plugin({} as never, { cachePath })
  const secondConfig = compatibleConfig()
  secondConfig.provider.proxy.options.apiKey = "second-key"
  await secondHooks.config?.(secondConfig as never)

  assert.equal(fetchCount, 2)
  assert.ok(secondConfig.provider.proxy.models?.["model-2"])
  assert.doesNotMatch(await readFile(cachePath, "utf8"), /first-key|second-key/)
})

test("honors XDG_CACHE_HOME for the default cache", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))

  const originalCacheHome = process.env.XDG_CACHE_HOME
  const originalFetch = globalThis.fetch
  process.env.XDG_CACHE_HOME = directory
  globalThis.fetch = async () => Response.json({ data: [] })
  t.after(() => {
    globalThis.fetch = originalFetch
    if (originalCacheHome === undefined) delete process.env.XDG_CACHE_HOME
    else process.env.XDG_CACHE_HOME = originalCacheHome
  })

  const hooks = await plugin({} as never)
  await hooks.config?.(compatibleConfig() as never)

  await access(join(directory, "opencode-models-discovery", "models-cache.json"))
})

test("rejects invalid and unknown plugin options", async () => {
  await assert.rejects(plugin({} as never, { maxResponseBytes: 0 }), /maxResponseBytes has an invalid value/)
  await assert.rejects(plugin({} as never, { refreshIntervlMs: 1 }), /refreshIntervlMs is not supported/)
})

test("logs discovery failures without exposing credentials", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))

  const originalFetch = globalThis.fetch
  globalThis.fetch = async () => new Response("unavailable", { status: 503 })
  t.after(() => {
    globalThis.fetch = originalFetch
  })

  const logs: unknown[] = []
  const hooks = await plugin(
    {
      client: {
        app: {
          log: async (entry: unknown) => {
            logs.push(entry)
          },
        },
      },
    } as never,
    { cachePath: join(directory, "cache.json") },
  )
  const config = compatibleConfig()
  config.provider.proxy.options.apiKey = "must-not-be-logged"
  config.provider.proxy.options.baseURL = "https://user:password@proxy.example/v1?token=query-secret"
  await hooks.config?.(config as never)

  assert.equal(logs.length, 1)
  assert.match(JSON.stringify(logs[0]), /503/)
  assert.doesNotMatch(JSON.stringify(logs[0]), /must-not-be-logged/)
  assert.doesNotMatch(JSON.stringify(logs[0]), /password|query-secret/)
})

test("rejects non-HTTP discovery URLs", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))

  const originalFetch = globalThis.fetch
  let fetched = false
  globalThis.fetch = async () => {
    fetched = true
    return Response.json({ data: [] })
  }
  t.after(() => {
    globalThis.fetch = originalFetch
  })

  const hooks = await plugin({} as never, { cachePath: join(directory, "cache.json") })
  const config = compatibleConfig()
  config.provider.proxy.options.baseURL = "file:///tmp/provider"
  await hooks.config?.(config as never)

  assert.equal(fetched, false)
  assert.equal(config.provider.proxy.models, undefined)
})

test("maps common limit aliases and normalizes modalities", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))

  const originalFetch = globalThis.fetch
  globalThis.fetch = async () =>
    Response.json({
      data: [
        {
          id: "aliased-model",
          max_model_len: 64_000,
          max_tokens: 4_000,
          input_modalities: ["TEXT", "vision"],
        },
      ],
    })
  t.after(() => {
    globalThis.fetch = originalFetch
  })

  const hooks = await plugin({} as never, { cachePath: join(directory, "cache.json") })
  const config = compatibleConfig()
  await hooks.config?.(config as never)

  assert.deepEqual(config.provider.proxy.models?.["aliased-model"], {
    id: "aliased-model",
    name: "aliased-model",
    limit: { context: 64_000, output: 4_000 },
    modalities: { input: ["text", "image"], output: ["text"] },
  })
})

test("follows bounded same-origin model pagination", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))

  const originalFetch = globalThis.fetch
  const requestedURLs: string[] = []
  globalThis.fetch = async (input) => {
    const url = input.toString()
    requestedURLs.push(url)
    return url.includes("after=first")
      ? Response.json({ data: [{ id: "second" }], has_more: false })
      : Response.json({ data: [{ id: "first" }], has_more: true, last_id: "first" })
  }
  t.after(() => {
    globalThis.fetch = originalFetch
  })

  const hooks = await plugin({} as never, { cachePath: join(directory, "cache.json") })
  const config = compatibleConfig()
  await hooks.config?.(config as never)

  assert.deepEqual(requestedURLs, ["https://proxy.example/v1/models", "https://proxy.example/v1/models?after=first"])
  assert.deepEqual(Object.keys(config.provider.proxy.models ?? {}), ["first", "second"])
})

test("merges concurrent cache updates", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))
  const cachePath = join(directory, "cache.json")

  const originalFetch = globalThis.fetch
  globalThis.fetch = async (input) => Response.json({ data: [{ id: new URL(input.toString()).hostname }] })
  t.after(() => {
    globalThis.fetch = originalFetch
  })

  const firstHooks = await plugin({} as never, { cachePath })
  const secondHooks = await plugin({} as never, { cachePath })
  const firstConfig = providerConfig("first", "https://first.example/v1")
  const secondConfig = providerConfig("second", "https://second.example/v1")
  await Promise.all([firstHooks.config?.(firstConfig as never), secondHooks.config?.(secondConfig as never)])

  const cache = JSON.parse(await readFile(cachePath, "utf8"))
  assert.equal(Object.keys(cache.providers).length, 2)
})

test("times out discovery requests", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))

  const originalFetch = globalThis.fetch
  globalThis.fetch = async (_input, init) =>
    new Promise<Response>((_resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("Mock fetch did not receive an abort signal")), 1_000)
      init?.signal?.addEventListener(
        "abort",
        () => {
          clearTimeout(timer)
          reject(init.signal?.reason)
        },
        { once: true },
      )
    })
  t.after(() => {
    globalThis.fetch = originalFetch
  })

  const hooks = await plugin({} as never, {
    cachePath: join(directory, "cache.json"),
    timeoutMs: 1,
  })
  const config = compatibleConfig()
  await hooks.config?.(config as never)

  assert.equal(config.provider.proxy.models, undefined)
})

test("maps custom modes and speed tiers", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))

  const originalFetch = globalThis.fetch
  globalThis.fetch = async () =>
    Response.json({
      data: [
        {
          id: "tiered-model",
          modes: {
            fast: {
              cost: { input: 2, output: 4 },
              provider: {
                body: { service_tier: "priority" },
                headers: { "x-mode": "fast", ignored: 42 },
              },
            },
          },
          additional_speed_tiers: ["flex"],
          service_tiers: [{ id: "flex-tier", name: "flex" }],
        },
      ],
    })
  t.after(() => {
    globalThis.fetch = originalFetch
  })

  const hooks = await plugin({} as never, { cachePath: join(directory, "cache.json") })
  const config = compatibleConfig()
  await hooks.config?.(config as never)

  assert.deepEqual(config.provider.proxy.models?.["tiered-model-fast"], {
    id: "tiered-model-fast",
    name: "tiered-model Fast",
    options: { serviceTier: "priority" },
    headers: { "x-mode": "fast" },
    cost: { input: 2, output: 4 },
  })
  const flex = config.provider.proxy.models?.["tiered-model-flex"] as { options: { serviceTier: string } } | undefined
  assert.equal(flex?.options.serviceTier, "flex-tier")
})

test("discovers models from a real HTTP endpoint", async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "opencode-models-discovery-"))
  t.after(() => rm(directory, { recursive: true, force: true }))

  let requestURL = ""
  let authorization = ""
  const server = createServer((request, response) => {
    requestURL = request.url ?? ""
    authorization = request.headers.authorization ?? ""
    response.setHeader("content-type", "application/json")
    response.end(JSON.stringify({ data: [{ id: "http-model" }] }))
  })
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve))
  t.after(() => new Promise<void>((resolve, reject) => server.close((error) => (error ? reject(error) : resolve()))))

  const address = server.address()
  assert.ok(address && typeof address === "object")
  const hooks = await plugin({} as never, { cachePath: join(directory, "cache.json") })
  const config = compatibleConfig()
  config.provider.proxy.options.baseURL = `http://127.0.0.1:${address.port}/v1`
  config.provider.proxy.options.apiKey = "http-secret"
  await hooks.config?.(config as never)

  assert.equal(requestURL, "/v1/models")
  assert.equal(authorization, "Bearer http-secret")
  assert.ok(config.provider.proxy.models?.["http-model"])
})

function providerConfig(providerID: string, baseURL: string) {
  return {
    provider: {
      [providerID]: {
        npm: "@ai-sdk/openai-compatible",
        options: { baseURL },
      },
    },
  }
}

function compatibleConfig() {
  return {
    provider: {
      proxy: {
        npm: "@ai-sdk/openai-compatible",
        options: { baseURL: "https://proxy.example/v1", apiKey: undefined as string | undefined },
        models: undefined as Record<string, unknown> | undefined,
      },
    },
  }
}

function providerModel(id: string) {
  return {
    id,
    providerID: "openai",
    api: { id, url: "https://proxy.example/v1", npm: "@ai-sdk/openai" },
    name: "Template Model",
    capabilities: {
      temperature: true,
      reasoning: true,
      attachment: true,
      toolcall: true,
      input: { text: true, audio: false, image: true, video: false, pdf: true },
      output: { text: true, audio: false, image: false, video: false, pdf: false },
      interleaved: false,
    },
    cost: { input: 10, output: 20, cache: { read: 1, write: 2 } },
    limit: { context: 1_000, output: 100 },
    status: "active" as const,
    options: { templateOption: true },
    headers: { "x-template": "true" },
    release_date: "2025-01-01",
  }
}
