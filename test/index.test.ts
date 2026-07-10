import assert from "node:assert/strict"
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
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
  globalThis.fetch = async () => Response.json({ data: [{ id: "discovered-model" }] })
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
