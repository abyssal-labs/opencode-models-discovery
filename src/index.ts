import { mkdir, readFile, writeFile } from "node:fs/promises"
import { dirname, join } from "node:path"
import { homedir } from "node:os"
import type { Plugin } from "@opencode-ai/plugin"

type ProviderConfig = {
  npm?: string
  name?: string
  whitelist?: string[]
  blacklist?: string[]
  models?: Record<string, ModelConfig>
  options?: {
    apiKey?: string
    baseURL?: string
    modelsDiscovery?: ProviderDiscoveryOptions
    [key: string]: unknown
  }
  [key: string]: unknown
}

type ModelConfig = {
  id?: string
  name?: string
  limit?: {
    context?: number
    input?: number
    output?: number
  }
  modalities?: {
    input?: string[]
    output?: string[]
  }
  [key: string]: unknown
}

type OpenCodeConfig = {
  provider?: Record<string, ProviderConfig>
}

type ProviderDiscoveryOptions = {
  refreshIntervalMs?: number
  refreshIntervalHours?: number
  include?: string[]
  exclude?: string[]
  overrideExisting?: boolean
}

type PluginOptions = {
  enabled?: boolean
  refreshIntervalMs?: number
  refreshIntervalHours?: number
  cachePath?: string
  providers?: {
    include?: string[]
    exclude?: string[]
  }
  overrideExisting?: boolean
}

type ModelsResponse = {
  data?: RemoteModel[]
  models?: RemoteModel[]
}

type RemoteModel = {
  id?: string
  slug?: string
  name?: string
  metadata?: Record<string, unknown>
  [key: string]: unknown
}

type Cache = {
  providers: Record<
    string,
    {
      checkedAt: number
      models: RemoteModel[]
    }
  >
}

const DEFAULT_REFRESH_INTERVAL_MS = 24 * 60 * 60 * 1000
const DEFAULT_CACHE_PATH = join(homedir(), ".cache", "opencode-models-discovery", "models-cache.json")
const DEFAULT_MAX_OUTPUT_TOKENS = 128_000
const OPENAI_SDKS = new Set(["@ai-sdk/openai", "@ai-sdk/openai-compatible"])

const plugin: Plugin = async (_input, options = {}) => {
  const pluginOptions = normalizePluginOptions(options as PluginOptions)

  return {
    config: async (cfg: OpenCodeConfig) => {
      if (pluginOptions.enabled === false) return
      if (!cfg.provider) return

      const cache = await readCache(pluginOptions.cachePath)
      let cacheChanged = false

      for (const [providerID, provider] of Object.entries(cfg.provider)) {
        const providerOptions = normalizeProviderOptions(provider.options?.modelsDiscovery, pluginOptions)
        if (!shouldHandleProvider(providerID, provider)) continue
        if (!matchesProviderFilter(providerID, providerOptions)) continue

        const baseURL = provider.options?.baseURL
        // Native OpenAI OAuth/API providers have no custom baseURL; discovery is only for wrappers/proxies.
        if (!baseURL) continue

        const cacheKey = `${providerID}:${baseURL}`
        const cached = cache.providers[cacheKey]
        const now = Date.now()
        const refreshIntervalMs = providerOptions.refreshIntervalMs ?? pluginOptions.refreshIntervalMs

        let models = cached?.models
        if (!cached || now - cached.checkedAt >= refreshIntervalMs) {
          const discovered = await fetchModels({
            baseURL,
            apiKey: expandEnv(provider.options?.apiKey),
          })
          if (discovered) {
            models = discovered
            cache.providers[cacheKey] = { checkedAt: now, models }
            cacheChanged = true
          } else if (cached) {
            cache.providers[cacheKey] = { ...cached, checkedAt: now }
            cacheChanged = true
          }
        }

        if (models?.length) {
          applyModels(provider, models, providerOptions.overrideExisting ?? pluginOptions.overrideExisting)
        }
      }

      if (cacheChanged) await writeCache(pluginOptions.cachePath, cache)
    },
  }
}

function normalizePluginOptions(options: PluginOptions) {
  return {
    enabled: options.enabled,
    refreshIntervalMs: intervalMs(options.refreshIntervalMs, options.refreshIntervalHours),
    cachePath: options.cachePath ?? DEFAULT_CACHE_PATH,
    providers: options.providers ?? {},
    overrideExisting: options.overrideExisting ?? true,
  }
}

function normalizeProviderOptions(options: ProviderDiscoveryOptions | undefined, pluginOptions: ReturnType<typeof normalizePluginOptions>) {
  return {
    refreshIntervalMs: intervalMs(options?.refreshIntervalMs, options?.refreshIntervalHours, pluginOptions.refreshIntervalMs),
    include: options?.include ?? pluginOptions.providers.include,
    exclude: options?.exclude ?? pluginOptions.providers.exclude,
    overrideExisting: options?.overrideExisting,
  }
}

function intervalMs(ms?: number, hours?: number, fallback = DEFAULT_REFRESH_INTERVAL_MS) {
  if (typeof ms === "number" && Number.isFinite(ms) && ms >= 0) return ms
  if (typeof hours === "number" && Number.isFinite(hours) && hours >= 0) return hours * 60 * 60 * 1000
  return fallback
}

function shouldHandleProvider(providerID: string, provider: ProviderConfig) {
  if (providerID === "openai") return true
  if (provider.npm && OPENAI_SDKS.has(provider.npm)) return true
  return false
}

function matchesProviderFilter(providerID: string, options: ReturnType<typeof normalizeProviderOptions>) {
  if (options.include?.length) return options.include.includes(providerID)
  if (options.exclude?.length) return !options.exclude.includes(providerID)
  return true
}

function modelsURL(baseURL: string) {
  const base = baseURL.replace(/\/+$/, "")
  return `${base}/models`
}

async function fetchModels(input: { baseURL: string; apiKey?: string }) {
  const headers: Record<string, string> = { accept: "application/json" }
  if (input.apiKey) headers.authorization = `Bearer ${input.apiKey}`

  try {
    const response = await fetch(modelsURL(input.baseURL), {
      headers,
      signal: AbortSignal.timeout(10_000),
    })
    if (!response.ok) return undefined

    const body = (await response.json()) as ModelsResponse | RemoteModel[]
    const models = Array.isArray(body) ? body : body.data ?? body.models ?? []
    return models.filter((model) => modelID(model) !== undefined)
  } catch {
    return undefined
  }
}

function applyModels(provider: ProviderConfig, models: RemoteModel[], overrideExisting: boolean) {
  provider.models ??= {}

  for (const remote of models) {
    const id = modelID(remote)
    if (!id) continue
    const existing = provider.models[id] ?? {}
    if (!overrideExisting && provider.models[id]) continue

    const mapped = mapRemoteModel(remote, existing)
    if (!mapped) continue

    provider.models[id] = {
      ...existing,
      ...mapped,
      limit: {
        ...existing.limit,
        ...mapped.limit,
      },
      modalities: {
        ...existing.modalities,
        ...mapped.modalities,
      },
    }
  }
}

function mapRemoteModel(remote: RemoteModel, existing: ModelConfig): ModelConfig | undefined {
  const metadata = remote.metadata ?? {}
  const context = numberValue(metadata.context_window, metadata.context, remote.context_window, remote.context)
  const input = numberValue(metadata.input_context_window, metadata.input, remote.input_context_window, remote.input)
  const output =
    numberValue(metadata.max_output_tokens, metadata.output, remote.max_output_tokens, remote.output) ??
    DEFAULT_MAX_OUTPUT_TOKENS
  const id = modelID(remote)
  if (!id) return undefined

  const mapped: ModelConfig = {
    id: existing.id ?? id,
    name: stringValue(remote.name, metadata.display_name, remote.display_name, existing.name, id),
  }

  if (context !== undefined || input !== undefined || output !== undefined) {
    mapped.limit = {}
    if (context !== undefined) mapped.limit.context = context
    if (input !== undefined) mapped.limit.input = input
    if (output !== undefined) mapped.limit.output = output
  }

  const inputModalities = stringArray(metadata.input_modalities, remote.input_modalities)
  const outputModalities = stringArray(metadata.output_modalities, remote.output_modalities)
  if (inputModalities || outputModalities) {
    mapped.modalities = {}
    if (inputModalities) mapped.modalities.input = inputModalities
    if (outputModalities) mapped.modalities.output = outputModalities
  }

  return mapped
}

function modelID(model: RemoteModel) {
  return stringValue(model.id, model.slug)
}

function numberValue(...values: unknown[]) {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) return value
    if (typeof value === "string" && value.trim() && Number.isFinite(Number(value))) return Number(value)
  }
  return undefined
}

function stringValue(...values: unknown[]) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) return value
  }
  return undefined
}

function stringArray(...values: unknown[]) {
  for (const value of values) {
    if (Array.isArray(value) && value.every((item) => typeof item === "string")) return value
  }
  return undefined
}

function expandEnv(value: string | undefined) {
  if (!value) return undefined
  const match = value.match(/^\{env:([A-Za-z_][A-Za-z0-9_]*)\}$/)
  if (match) return process.env[match[1]]
  return value
}

async function readCache(path: string): Promise<Cache> {
  try {
    const parsed = JSON.parse(await readFile(path, "utf8")) as Cache
    if (parsed && typeof parsed === "object" && parsed.providers) return parsed
  } catch {
    // Missing or invalid cache should not break opencode startup.
  }
  return { providers: {} }
}

async function writeCache(path: string, cache: Cache) {
  await mkdir(dirname(path), { recursive: true })
  await writeFile(path, `${JSON.stringify(cache, null, 2)}\n`)
}

export default plugin
