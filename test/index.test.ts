import assert from "node:assert/strict"
import { mkdtemp, rm } from "node:fs/promises"
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
