import { mkdir, readFile, writeFile } from "node:fs/promises"
import { dirname, join } from "node:path"
import { homedir } from "node:os"
import type { Plugin } from "@opencode-ai/plugin"
import type { Model as ProviderModel, Provider as ProviderInfo } from "@opencode-ai/sdk/v2"

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
  fallbackContextTokens?: number
  fallbackOutputTokens?: number
  include?: string[]
  exclude?: string[]
  overrideExisting?: boolean
}

type PluginOptions = {
  enabled?: boolean
  refreshIntervalMs?: number
  refreshIntervalHours?: number
  fallbackContextTokens?: number
  fallbackOutputTokens?: number
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

type RemoteMode = {
  cost?: Record<string, unknown>
  provider?: {
    body?: Record<string, unknown>
    headers?: Record<string, string>
  }
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
const OPENAI_SDKS = new Set(["@ai-sdk/openai", "@ai-sdk/openai-compatible"])

const plugin: Plugin = async (_input, options = {}) => {
  const pluginOptions = normalizePluginOptions(options as PluginOptions)
  const capturedProviders = new Map<string, ProviderConfig>()

  return {
    config: async (cfg: OpenCodeConfig) => {
      if (pluginOptions.enabled === false) return
      if (!cfg.provider) return

      const cache = await readCache(pluginOptions.cachePath)
      let cacheChanged = false

      for (const [providerID, provider] of Object.entries(cfg.provider)) {
        if (providerID === "openai") {
          capturedProviders.set(providerID, provider)
          continue
        }
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
          }
        }

        if (models?.length) {
          const overrideExisting = providerOptions.overrideExisting ?? pluginOptions.overrideExisting
          if (overrideExisting) provider.models = {}
          applyModels(provider, models, overrideExisting)
        }
      }

      if (cacheChanged) await writeCache(pluginOptions.cachePath, cache)
    },
    provider: {
      id: "openai",
      async models(provider, ctx) {
        if (ctx.auth?.type === "oauth") return provider.models
        const configProvider = capturedProviders.get(provider.id)
        const providerOptions = normalizeProviderOptions(configProvider?.options?.modelsDiscovery, pluginOptions)
        const baseURL = configProvider?.options?.baseURL
        if (!baseURL) return provider.models
        if (!matchesProviderFilter(provider.id, providerOptions)) return provider.models

        const cache = await readCache(pluginOptions.cachePath)
        const cacheKey = `${provider.id}:${baseURL}`
        const cached = cache.providers[cacheKey]
        const now = Date.now()
        const refreshIntervalMs = providerOptions.refreshIntervalMs ?? pluginOptions.refreshIntervalMs

        let models = cached?.models
        if (!cached || now - cached.checkedAt >= refreshIntervalMs) {
          const discovered = await fetchModels({
            baseURL,
            apiKey: expandEnv(configProvider?.options?.apiKey),
          })
          if (discovered) {
            models = discovered
            cache.providers[cacheKey] = { checkedAt: now, models }
            await writeCache(pluginOptions.cachePath, cache)
          }
        }

        return models?.length
          ? applyProviderModels(
              provider,
              models,
              providerOptions.overrideExisting ?? pluginOptions.overrideExisting,
              providerOptions,
            )
          : provider.models
      },
    },
  }
}

function normalizePluginOptions(options: PluginOptions) {
  return {
    enabled: options.enabled,
    refreshIntervalMs: intervalMs(options.refreshIntervalMs, options.refreshIntervalHours),
    fallbackContextTokens: tokenLimit(options.fallbackContextTokens),
    fallbackOutputTokens: tokenLimit(options.fallbackOutputTokens),
    cachePath: options.cachePath ?? DEFAULT_CACHE_PATH,
    providers: options.providers ?? {},
    overrideExisting: options.overrideExisting ?? true,
  }
}

function normalizeProviderOptions(options: ProviderDiscoveryOptions | undefined, pluginOptions: ReturnType<typeof normalizePluginOptions>) {
  return {
    refreshIntervalMs: intervalMs(options?.refreshIntervalMs, options?.refreshIntervalHours, pluginOptions.refreshIntervalMs),
    fallbackContextTokens: tokenLimit(options?.fallbackContextTokens) ?? pluginOptions.fallbackContextTokens,
    fallbackOutputTokens: tokenLimit(options?.fallbackOutputTokens) ?? pluginOptions.fallbackOutputTokens,
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

function tokenLimit(value: number | undefined) {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : undefined
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

    const nextModel: ModelConfig = {
      ...existing,
      ...mapped,
    }
    if (existing.limit || mapped.limit) {
      nextModel.limit = {
        ...existing.limit,
        ...mapped.limit,
      }
    }
    if (existing.modalities || mapped.modalities) {
      nextModel.modalities = {
        ...existing.modalities,
        ...mapped.modalities,
      }
    }
    provider.models[id] = nextModel

    for (const [mode, modeConfig] of Object.entries(remoteModes(remote))) {
      const modeID = `${id}-${mode}`
      const existingMode = provider.models[modeID] ?? {}
      if (!overrideExisting && provider.models[modeID]) continue

      const mappedMode = mapRemoteModel(remote, existingMode)
      if (!mappedMode) continue

      const nextMode: ModelConfig = {
        ...existingMode,
        ...mappedMode,
        id: existingMode.id ?? modeID,
        name: existingMode.name ?? `${mapped.name ?? id} ${titleCase(mode)}`,
      }
      if (nextModel.limit || mappedMode.limit || existingMode.limit) {
        nextMode.limit = {
          ...nextModel.limit,
          ...existingMode.limit,
          ...mappedMode.limit,
        }
      }
      if (nextModel.modalities || mappedMode.modalities || existingMode.modalities) {
        nextMode.modalities = {
          ...nextModel.modalities,
          ...existingMode.modalities,
          ...mappedMode.modalities,
        }
      }
      applyModeConfig(nextMode, modeConfig)
      provider.models[modeID] = nextMode
    }
  }
}

function applyProviderModels(
  provider: ProviderInfo,
  models: RemoteModel[],
  overrideExisting: boolean,
  options: Pick<ReturnType<typeof normalizeProviderOptions>, "fallbackContextTokens" | "fallbackOutputTokens">,
) {
  const next = overrideExisting ? {} : { ...provider.models }
  const template = Object.values(provider.models)[0]

  for (const remote of models) {
    const id = modelID(remote)
    if (!id) continue

    const existing = provider.models[id]
    if (!overrideExisting && existing) continue
    if (!existing && !template) continue

    const base = existing
      ? providerModelBase(provider, existing, remote)
      : discoveredProviderModel(provider, template, remote, options)
    const applied = applyRemoteToProviderModel(base, remote)
    next[id] = applied

    for (const [mode, modeConfig] of Object.entries(remoteModes(remote))) {
      const modeID = `${id}-${mode}`
      const existingMode = next[modeID]
      const baseMode = existingMode ? applyRemoteToProviderModel(existingMode, remote) : cloneProviderMode(applied, modeID, mode)
      next[modeID] = applyProviderModeConfig(baseMode, modeConfig)
    }
  }

  return next
}

function discoveredProviderModel(
  provider: ProviderInfo,
  template: ProviderModel,
  remote: RemoteModel,
  options: Pick<ReturnType<typeof normalizeProviderOptions>, "fallbackContextTokens" | "fallbackOutputTokens">,
): ProviderModel {
  const metadata = remote.metadata ?? {}
  const id = modelID(remote) ?? template.id

  return {
    id,
    providerID: provider.id,
    api: { ...template.api, id },
    name: stringValue(remote.name, metadata.display_name, remote.display_name, id) ?? id,
    capabilities: {
      temperature: false,
      reasoning: false,
      attachment: false,
      toolcall: false,
      input: { text: true, audio: false, image: false, video: false, pdf: false },
      output: { text: true, audio: false, image: false, video: false, pdf: false },
      interleaved: false,
    },
    cost: { input: 0, output: 0, cache: { read: 0, write: 0 } },
    limit: {
      context: options.fallbackContextTokens ?? template.limit.context,
      output: options.fallbackOutputTokens ?? template.limit.output,
    },
    status: "active",
    options: {},
    headers: {},
    release_date: "",
  }
}

function providerModelBase(provider: ProviderInfo, existing: ProviderModel, remote: RemoteModel): ProviderModel {
  const metadata = remote.metadata ?? {}
  const id = modelID(remote) ?? existing.id

  return {
    ...existing,
    id,
    providerID: provider.id,
    api: { ...existing.api, id },
    name: stringValue(remote.name, metadata.display_name, remote.display_name, id) ?? id,
    options: { ...existing.options },
    headers: { ...existing.headers },
  }
}

function applyRemoteToProviderModel(existing: ProviderModel, remote: RemoteModel): ProviderModel {
  const metadata = remote.metadata ?? {}
  const architecture = objectValue(remote.architecture)
  const context = numberValue(
    metadata.context_window,
    metadata.context_length,
    metadata.context,
    remote.context_window,
    remote.context_length,
    remote.context,
  )
  const input = numberValue(metadata.input_context_window, metadata.input, remote.input_context_window, remote.input)
  const output = numberValue(
    metadata.max_output_tokens,
    metadata.output,
    remote.max_output_tokens,
    remote.output,
  )
  const inputModalities = stringArray(metadata.input_modalities, remote.input_modalities, architecture?.input_modalities)
  const outputModalities = stringArray(metadata.output_modalities, remote.output_modalities, architecture?.output_modalities)

  return {
    ...existing,
    name: existing.name || stringValue(remote.name, metadata.display_name, remote.display_name) || existing.id,
    limit: {
      ...existing.limit,
      ...(context !== undefined ? { context } : {}),
      ...(input !== undefined ? { input } : {}),
      ...(output !== undefined ? { output } : {}),
    },
    capabilities: {
      ...existing.capabilities,
      input: inputModalities ? modalitiesToCapabilities(inputModalities, existing.capabilities.input) : existing.capabilities.input,
      output: outputModalities
        ? modalitiesToCapabilities(outputModalities, existing.capabilities.output)
        : existing.capabilities.output,
    },
  }
}

function modalitiesToCapabilities<T extends Record<string, boolean>>(modalities: string[], existing: T): T {
  const next = { ...existing }
  for (const key of Object.keys(next)) next[key as keyof T] = modalities.includes(key) as T[keyof T]
  return next
}

function mapRemoteModel(remote: RemoteModel, existing: ModelConfig): ModelConfig | undefined {
  const metadata = remote.metadata ?? {}
  const architecture = objectValue(remote.architecture)
  const context = numberValue(
    metadata.context_window,
    metadata.context_length,
    metadata.context,
    remote.context_window,
    remote.context_length,
    remote.context,
  )
  const input = numberValue(metadata.input_context_window, metadata.input, remote.input_context_window, remote.input)
  const discoveredOutput = numberValue(
    metadata.max_output_tokens,
    metadata.output,
    remote.max_output_tokens,
    remote.output,
  )
  const id = modelID(remote)
  if (!id) return undefined

  const mapped: ModelConfig = {
    id: existing.id ?? id,
    name: existing.name ?? stringValue(remote.name, metadata.display_name, remote.display_name, id),
  }

  if (context !== undefined || input !== undefined || discoveredOutput !== undefined) {
    mapped.limit = {}
    if (context !== undefined) mapped.limit.context = context
    if (input !== undefined) mapped.limit.input = input
    if (discoveredOutput !== undefined) mapped.limit.output = discoveredOutput
  }

  const inputModalities = stringArray(metadata.input_modalities, remote.input_modalities, architecture?.input_modalities)
  const outputModalities = stringArray(metadata.output_modalities, remote.output_modalities, architecture?.output_modalities)
  if (inputModalities || outputModalities) {
    mapped.modalities = {
      input: inputModalities ?? existing.modalities?.input ?? ["text"],
      output: outputModalities ?? existing.modalities?.output ?? ["text"],
    }
  }

  return mapped
}

function remoteModes(remote: RemoteModel): Record<string, RemoteMode> {
  const metadata = remote.metadata ?? {}
  const experimental = objectValue(remote.experimental) ?? objectValue(metadata.experimental)
  const discovered = objectValue(experimental?.modes) ?? objectValue(remote.modes) ?? objectValue(metadata.modes)
  const modes: Record<string, RemoteMode> = {}

  if (discovered) {
    for (const [mode, value] of Object.entries(discovered)) {
      const config = objectValue(value)
      modes[mode] = config ? (config as RemoteMode) : {}
    }
  }

  for (const mode of stringArray(remote.additional_speed_tiers, metadata.additional_speed_tiers) ?? []) {
    modes[mode] ??= speedTierMode(remote, mode)
  }

  return modes
}

function speedTierMode(remote: RemoteModel, mode: string): RemoteMode {
  const metadata = remote.metadata ?? {}
  const serviceTier = serviceTierID(remote, mode) ?? serviceTierID(metadata, mode) ?? mode
  return { provider: { body: { service_tier: serviceTier } } }
}

function serviceTierID(source: Record<string, unknown>, mode: string) {
  const defaultTier = stringValue(source.default_service_tier)
  const tiers = arrayValue(source.service_tiers)
  if (!tiers) return defaultTier

  const normalizedMode = mode.toLowerCase()
  const tier = tiers.map(objectValue).find((item) => {
    if (!item) return false
    const id = stringValue(item.id)?.toLowerCase()
    const name = stringValue(item.name)?.toLowerCase()
    return id === normalizedMode || name === normalizedMode
  })
  return stringValue(tier?.id, defaultTier)
}

function applyModeConfig(model: ModelConfig, mode: RemoteMode) {
  const provider = objectValue(mode.provider)
  const body = objectValue(provider?.body)
  const headers = objectValue(provider?.headers)
  const cost = objectValue(mode.cost)

  if (body) model.options = { ...objectValue(model.options), ...camelizeKeys(body) }
  if (headers) model.headers = { ...objectValue(model.headers), ...stringRecord(headers) }
  if (cost) model.cost = { ...objectValue(model.cost), ...cost }
}

function cloneProviderMode(base: ProviderModel, id: string, mode: string): ProviderModel {
  return {
    ...base,
    id,
    name: `${base.name || base.id} ${titleCase(mode)}`,
    options: { ...base.options },
    headers: { ...base.headers },
  }
}

function applyProviderModeConfig(model: ProviderModel, mode: RemoteMode): ProviderModel {
  const provider = objectValue(mode.provider)
  const body = objectValue(provider?.body)
  const headers = objectValue(provider?.headers)
  const cost = objectValue(mode.cost)

  return {
    ...model,
    ...(cost ? { cost: applyProviderCost(model.cost, cost) } : {}),
    options: body ? { ...model.options, ...camelizeKeys(body) } : model.options,
    headers: headers ? { ...model.headers, ...stringRecord(headers) } : model.headers,
  }
}

function applyProviderCost(existing: ProviderModel["cost"], cost: Record<string, unknown>): ProviderModel["cost"] {
  return {
    ...existing,
    input: numberValue(cost.input) ?? existing.input,
    output: numberValue(cost.output) ?? existing.output,
    cache: {
      ...existing.cache,
      read: numberValue(cost.cache_read) ?? existing.cache.read,
      write: numberValue(cost.cache_write) ?? existing.cache.write,
    },
  }
}

function camelizeKeys(values: Record<string, unknown>) {
  return Object.fromEntries(Object.entries(values).map(([key, value]) => [key.replace(/_([a-z])/g, (_, char) => char.toUpperCase()), value]))
}

function stringRecord(values: Record<string, unknown>) {
  return Object.fromEntries(Object.entries(values).filter((entry): entry is [string, string] => typeof entry[1] === "string"))
}

function titleCase(value: string) {
  return value
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map((part) => part[0]?.toUpperCase() + part.slice(1))
    .join(" ")
}

function objectValue(value: unknown) {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : undefined
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

function arrayValue(...values: unknown[]) {
  for (const value of values) {
    if (Array.isArray(value)) return value
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
    const parsed = objectValue(JSON.parse(await readFile(path, "utf8")))
    const providers = objectValue(parsed?.providers)
    if (!providers) return { providers: {} }

    const valid: Cache["providers"] = {}
    for (const [key, value] of Object.entries(providers)) {
      const entry = objectValue(value)
      const checkedAt = numberValue(entry?.checkedAt)
      if (checkedAt === undefined || checkedAt < 0 || !Array.isArray(entry?.models)) continue

      const models = entry.models
        .map(objectValue)
        .filter((model): model is RemoteModel => model !== undefined && modelID(model) !== undefined)
      valid[key] = { checkedAt, models }
    }
    return { providers: valid }
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
