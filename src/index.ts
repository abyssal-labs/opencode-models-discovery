import { mkdir, open, readFile, rename, rm, stat, writeFile } from "node:fs/promises"
import { dirname, join } from "node:path"
import { homedir } from "node:os"
import { createHash, randomUUID } from "node:crypto"
import { setTimeout as delay } from "node:timers/promises"
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

export type ProviderDiscoveryOptions = {
  refreshIntervalMs?: number
  refreshIntervalHours?: number
  fallbackContextTokens?: number
  fallbackOutputTokens?: number
  maxResponseBytes?: number
  maxPages?: number
  timeoutMs?: number
  headers?: Record<string, string>
  include?: string[]
  exclude?: string[]
  overrideExisting?: boolean
}

export type PluginOptions = {
  enabled?: boolean
  refreshIntervalMs?: number
  refreshIntervalHours?: number
  fallbackContextTokens?: number
  fallbackOutputTokens?: number
  maxResponseBytes?: number
  maxPages?: number
  timeoutMs?: number
  headers?: Record<string, string>
  cachePath?: string
  providers?: {
    include?: string[]
    exclude?: string[]
  }
  overrideExisting?: boolean
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
const DEFAULT_MAX_RESPONSE_BYTES = 5 * 1024 * 1024
const DEFAULT_MAX_PAGES = 10
const DEFAULT_TIMEOUT_MS = 10_000
const OPENAI_SDKS = new Set(["@ai-sdk/openai", "@ai-sdk/openai-compatible"])

const plugin: Plugin = async (input, options = {}) => {
  validatePluginOptions(options)
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

        const apiKey = expandEnv(provider.options?.apiKey)
        const headers = expandEnvRecord(providerOptions.headers)
        const cacheKey = modelCacheKey(providerID, { baseURL, apiKey, headers })
        const refreshed = await refreshModels(cache, cacheKey, {
          providerID,
          baseURL,
          apiKey,
          headers,
          maxResponseBytes: providerOptions.maxResponseBytes,
          maxPages: providerOptions.maxPages,
          timeoutMs: providerOptions.timeoutMs,
          refreshIntervalMs: providerOptions.refreshIntervalMs,
          log: (message, extra) => writeLog(input, message, extra),
        })
        cacheChanged ||= refreshed.changed

        if (refreshed.models) {
          const overrideExisting = providerOptions.overrideExisting ?? pluginOptions.overrideExisting
          if (overrideExisting) provider.models = {}
          applyModels(provider, refreshed.models, overrideExisting)
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
        const apiKey = expandEnv(configProvider?.options?.apiKey)
        const headers = expandEnvRecord(providerOptions.headers)
        const cacheKey = modelCacheKey(provider.id, { baseURL, apiKey, headers })
        const refreshed = await refreshModels(cache, cacheKey, {
          providerID: provider.id,
          baseURL,
          apiKey,
          headers,
          maxResponseBytes: providerOptions.maxResponseBytes,
          maxPages: providerOptions.maxPages,
          timeoutMs: providerOptions.timeoutMs,
          refreshIntervalMs: providerOptions.refreshIntervalMs,
          log: (message, extra) => writeLog(input, message, extra),
        })
        if (refreshed.changed) await writeCache(pluginOptions.cachePath, cache)

        return refreshed.models
          ? applyProviderModels(
              provider,
              refreshed.models,
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
    maxResponseBytes: positiveInteger(options.maxResponseBytes) ?? DEFAULT_MAX_RESPONSE_BYTES,
    maxPages: positiveInteger(options.maxPages) ?? DEFAULT_MAX_PAGES,
    timeoutMs: positiveInteger(options.timeoutMs) ?? DEFAULT_TIMEOUT_MS,
    headers: options.headers,
    cachePath: options.cachePath ?? defaultCachePath(),
    providers: options.providers ?? {},
    overrideExisting: options.overrideExisting ?? true,
  }
}

function defaultCachePath() {
  const root = process.env.XDG_CACHE_HOME || process.env.LOCALAPPDATA || join(homedir(), ".cache")
  return join(root, "opencode-models-discovery", "models-cache.json")
}

function normalizeProviderOptions(value: unknown, pluginOptions: ReturnType<typeof normalizePluginOptions>) {
  validateProviderOptions(value)
  const options = value as ProviderDiscoveryOptions | undefined
  return {
    refreshIntervalMs: intervalMs(
      options?.refreshIntervalMs,
      options?.refreshIntervalHours,
      pluginOptions.refreshIntervalMs,
    ),
    fallbackContextTokens: tokenLimit(options?.fallbackContextTokens) ?? pluginOptions.fallbackContextTokens,
    fallbackOutputTokens: tokenLimit(options?.fallbackOutputTokens) ?? pluginOptions.fallbackOutputTokens,
    maxResponseBytes: positiveInteger(options?.maxResponseBytes) ?? pluginOptions.maxResponseBytes,
    maxPages: positiveInteger(options?.maxPages) ?? pluginOptions.maxPages,
    timeoutMs: positiveInteger(options?.timeoutMs) ?? pluginOptions.timeoutMs,
    headers: options?.headers ?? pluginOptions.headers,
    include: options?.include ?? pluginOptions.providers.include,
    exclude: options?.exclude ?? pluginOptions.providers.exclude,
    overrideExisting: options?.overrideExisting,
  }
}

function validatePluginOptions(value: unknown): asserts value is PluginOptions {
  const options = requireOptionsObject(value, "plugin options")
  const allowed = new Set([
    "enabled",
    "refreshIntervalMs",
    "refreshIntervalHours",
    "fallbackContextTokens",
    "fallbackOutputTokens",
    "maxResponseBytes",
    "maxPages",
    "timeoutMs",
    "cachePath",
    "providers",
    "overrideExisting",
    "headers",
  ])
  rejectUnknownOptions(options, allowed, "plugin options")
  validateCommonOptions(options, "plugin options")
  if (options.enabled !== undefined && typeof options.enabled !== "boolean") invalidOption("plugin options", "enabled")
  if (options.cachePath !== undefined && (typeof options.cachePath !== "string" || !options.cachePath.trim())) {
    invalidOption("plugin options", "cachePath")
  }
  if (options.overrideExisting !== undefined && typeof options.overrideExisting !== "boolean") {
    invalidOption("plugin options", "overrideExisting")
  }
  if (options.providers !== undefined) {
    const providers = requireOptionsObject(options.providers, "plugin options.providers")
    rejectUnknownOptions(providers, new Set(["include", "exclude"]), "plugin options.providers")
    validateStringArray(providers.include, "plugin options.providers", "include")
    validateStringArray(providers.exclude, "plugin options.providers", "exclude")
  }
}

function validateProviderOptions(value: unknown): asserts value is ProviderDiscoveryOptions | undefined {
  if (value === undefined) return
  const options = requireOptionsObject(value, "provider modelsDiscovery options")
  rejectUnknownOptions(
    options,
    new Set([
      "refreshIntervalMs",
      "refreshIntervalHours",
      "fallbackContextTokens",
      "fallbackOutputTokens",
      "maxResponseBytes",
      "maxPages",
      "timeoutMs",
      "include",
      "exclude",
      "overrideExisting",
      "headers",
    ]),
    "provider modelsDiscovery options",
  )
  validateCommonOptions(options, "provider modelsDiscovery options")
  validateStringArray(options.include, "provider modelsDiscovery options", "include")
  validateStringArray(options.exclude, "provider modelsDiscovery options", "exclude")
  if (options.overrideExisting !== undefined && typeof options.overrideExisting !== "boolean") {
    invalidOption("provider modelsDiscovery options", "overrideExisting")
  }
}

function validateCommonOptions(options: Record<string, unknown>, scope: string) {
  for (const key of ["refreshIntervalMs", "refreshIntervalHours"] as const) {
    const value = options[key]
    if (value !== undefined && (typeof value !== "number" || !Number.isFinite(value) || value < 0)) {
      invalidOption(scope, key)
    }
  }
  for (const key of [
    "fallbackContextTokens",
    "fallbackOutputTokens",
    "maxResponseBytes",
    "maxPages",
    "timeoutMs",
  ] as const) {
    const value = options[key]
    if (value !== undefined && (typeof value !== "number" || !Number.isSafeInteger(value) || value <= 0)) {
      invalidOption(scope, key)
    }
  }
  if (options.headers !== undefined) {
    const headers = requireOptionsObject(options.headers, `${scope}.headers`)
    if (Object.values(headers).some((header) => typeof header !== "string")) invalidOption(scope, "headers")
  }
}

function requireOptionsObject(value: unknown, scope: string) {
  const options = objectValue(value)
  if (!options) throw new TypeError(`${scope} must be an object`)
  return options
}

function rejectUnknownOptions(options: Record<string, unknown>, allowed: Set<string>, scope: string) {
  const unknown = Object.keys(options).find((key) => !allowed.has(key))
  if (unknown) throw new TypeError(`${scope}.${unknown} is not supported`)
}

function validateStringArray(value: unknown, scope: string, key: string) {
  if (value !== undefined && (!Array.isArray(value) || value.some((item) => typeof item !== "string"))) {
    invalidOption(scope, key)
  }
}

function invalidOption(scope: string, key: string): never {
  throw new TypeError(`${scope}.${key} has an invalid value`)
}

function intervalMs(ms?: number, hours?: number, fallback = DEFAULT_REFRESH_INTERVAL_MS) {
  if (typeof ms === "number" && Number.isFinite(ms) && ms >= 0) return ms
  if (typeof hours === "number" && Number.isFinite(hours) && hours >= 0) return hours * 60 * 60 * 1000
  return fallback
}

function tokenLimit(value: number | undefined) {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : undefined
}

function positiveInteger(value: number | undefined) {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0 ? value : undefined
}

function shouldHandleProvider(providerID: string, provider: ProviderConfig) {
  if (providerID === "openai") return true
  if (provider.npm && OPENAI_SDKS.has(provider.npm)) return true
  return false
}

function matchesProviderFilter(providerID: string, options: ReturnType<typeof normalizeProviderOptions>) {
  if (options.exclude?.includes(providerID)) return false
  if (options.include?.length) return options.include.includes(providerID)
  return true
}

function modelsURL(baseURL: string) {
  const url = new URL(baseURL)
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new TypeError("Discovery baseURL must use HTTP or HTTPS")
  }
  url.pathname = `${url.pathname.replace(/\/+$/, "")}/models`
  url.hash = ""
  return url
}

function modelCacheKey(
  providerID: string,
  input: { baseURL: string; apiKey?: string; headers?: Record<string, string> },
) {
  const headers = Object.entries(input.headers ?? {}).sort(([left], [right]) => left.localeCompare(right))
  const digest = createHash("sha256")
    .update(JSON.stringify({ baseURL: input.baseURL, apiKey: input.apiKey, headers }))
    .digest("hex")
    .slice(0, 24)
  return `${providerID}:${digest}`
}

async function refreshModels(
  cache: Cache,
  cacheKey: string,
  input: {
    providerID: string
    baseURL: string
    apiKey?: string
    headers?: Record<string, string>
    maxResponseBytes: number
    maxPages: number
    timeoutMs: number
    refreshIntervalMs: number
    log: (message: string, extra: Record<string, unknown>) => Promise<void>
  },
) {
  const cached = cache.providers[cacheKey]
  const now = Date.now()
  if (cached && now - cached.checkedAt < input.refreshIntervalMs) {
    return { models: cached.models, changed: false }
  }

  const result = await fetchModels(input)
  if (!result.ok) {
    await input.log("Model discovery failed", {
      providerID: input.providerID,
      baseURL: redactedURL(input.baseURL),
      reason: result.reason,
      ...(result.status === undefined ? {} : { status: result.status }),
      usingStaleCache: cached !== undefined,
    })
    return { models: cached?.models, changed: false }
  }

  cache.providers[cacheKey] = { checkedAt: now, models: result.models }
  return { models: result.models, changed: true }
}

function redactedURL(value: string) {
  try {
    const url = new URL(value)
    url.username = ""
    url.password = ""
    url.search = ""
    url.hash = ""
    return url.toString()
  } catch {
    return "invalid URL"
  }
}

async function fetchModels(input: {
  baseURL: string
  apiKey?: string
  headers?: Record<string, string>
  maxResponseBytes: number
  maxPages: number
  timeoutMs: number
}) {
  const headers = new Headers(input.headers)
  if (!headers.has("accept")) headers.set("accept", "application/json")
  if (input.apiKey && !headers.has("authorization")) headers.set("authorization", `Bearer ${input.apiKey}`)

  try {
    const initialURL = modelsURL(input.baseURL)
    let url = initialURL
    const visited = new Set<string>()
    const models = new Map<string, RemoteModel>()

    for (let page = 0; page < input.maxPages; page += 1) {
      if (visited.has(url.href)) return { ok: false as const, reason: "Pagination loop detected" }
      visited.add(url.href)

      const response = await fetch(url, {
        headers,
        signal: AbortSignal.timeout(input.timeoutMs),
      })
      if (!response.ok) return { ok: false as const, reason: "HTTP error", status: response.status }

      const body = await responseJSON(response, input.maxResponseBytes)
      const responseObject = objectValue(body)
      const values = Array.isArray(body)
        ? body
        : Array.isArray(responseObject?.data)
          ? responseObject.data
          : Array.isArray(responseObject?.models)
            ? responseObject.models
            : undefined
      if (!values) return { ok: false as const, reason: "Invalid or oversized JSON response" }

      const pageModels = values
        .map(objectValue)
        .filter((model): model is RemoteModel => model !== undefined && modelID(model) !== undefined)
      if (values.length > 0 && pageModels.length === 0) {
        return { ok: false as const, reason: "Response contains no valid models" }
      }
      for (const model of pageModels) models.set(modelID(model)!, model)

      const next = nextPageURL(responseObject, url)
      if (!next) return { ok: true as const, models: [...models.values()] }
      if (next.origin !== initialURL.origin) {
        return { ok: false as const, reason: "Cross-origin pagination URL rejected" }
      }
      url = next
    }
    return { ok: false as const, reason: `Pagination exceeded ${input.maxPages} pages` }
  } catch (error) {
    return {
      ok: false as const,
      reason: error instanceof Error ? error.message : "Unknown discovery error",
    }
  }
}

function nextPageURL(response: Record<string, unknown> | undefined, current: URL) {
  if (!response) return undefined
  const pagination = objectValue(response.pagination)
  const next = stringValue(response.next_page, response.next, pagination?.next_page, pagination?.next)
  if (next) return new URL(next, current)
  if (response.has_more !== true) return undefined

  const lastID = stringValue(response.last_id)
  if (!lastID) return undefined
  const url = new URL(current)
  url.searchParams.set("after", lastID)
  return url
}

async function writeLog(input: Parameters<Plugin>[0], message: string, extra: Record<string, unknown>) {
  try {
    await input.client.app.log({
      body: {
        service: "opencode-models-discovery",
        level: "warn",
        message,
        extra,
      },
    })
  } catch {
    // Logging failures must not make provider initialization fail.
  }
}

async function responseJSON(response: Response, maxBytes: number) {
  const contentLength = numberValue(response.headers.get("content-length"))
  if (contentLength !== undefined && contentLength > maxBytes) return undefined
  if (!response.body) return undefined

  const reader = response.body.getReader()
  const chunks: Uint8Array[] = []
  let total = 0
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    total += value.byteLength
    if (total > maxBytes) {
      await reader.cancel()
      return undefined
    }
    chunks.push(value)
  }

  const bytes = new Uint8Array(total)
  let offset = 0
  for (const chunk of chunks) {
    bytes.set(chunk, offset)
    offset += chunk.byteLength
  }
  return JSON.parse(new TextDecoder().decode(bytes)) as unknown
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
      const baseMode = existingMode
        ? applyRemoteToProviderModel(existingMode, remote)
        : cloneProviderMode(applied, modeID, mode)
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
    metadata.max_model_len,
    metadata.max_context_length,
    remote.context_window,
    remote.context_length,
    remote.context,
    remote.max_model_len,
    remote.max_context_length,
  )
  const input = numberValue(
    metadata.input_context_window,
    metadata.max_input_tokens,
    metadata.input,
    remote.input_context_window,
    remote.max_input_tokens,
    remote.input,
  )
  const output = numberValue(
    metadata.max_output_tokens,
    metadata.max_completion_tokens,
    metadata.max_tokens,
    metadata.output,
    remote.max_output_tokens,
    remote.max_completion_tokens,
    remote.max_tokens,
    remote.output,
  )
  const inputModalities = modelModalities(
    metadata.input_modalities,
    remote.input_modalities,
    architecture?.input_modalities,
  )
  const outputModalities = modelModalities(
    metadata.output_modalities,
    remote.output_modalities,
    architecture?.output_modalities,
  )

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
      input: inputModalities
        ? modalitiesToCapabilities(inputModalities, existing.capabilities.input)
        : existing.capabilities.input,
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
    metadata.max_model_len,
    metadata.max_context_length,
    remote.context_window,
    remote.context_length,
    remote.context,
    remote.max_model_len,
    remote.max_context_length,
  )
  const input = numberValue(
    metadata.input_context_window,
    metadata.max_input_tokens,
    metadata.input,
    remote.input_context_window,
    remote.max_input_tokens,
    remote.input,
  )
  const discoveredOutput = numberValue(
    metadata.max_output_tokens,
    metadata.max_completion_tokens,
    metadata.max_tokens,
    metadata.output,
    remote.max_output_tokens,
    remote.max_completion_tokens,
    remote.max_tokens,
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

  const inputModalities = modelModalities(
    metadata.input_modalities,
    remote.input_modalities,
    architecture?.input_modalities,
  )
  const outputModalities = modelModalities(
    metadata.output_modalities,
    remote.output_modalities,
    architecture?.output_modalities,
  )
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
  return Object.fromEntries(
    Object.entries(values).map(([key, value]) => [key.replace(/_([a-z])/g, (_, char) => char.toUpperCase()), value]),
  )
}

function stringRecord(values: Record<string, unknown>) {
  return Object.fromEntries(
    Object.entries(values).filter((entry): entry is [string, string] => typeof entry[1] === "string"),
  )
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

function modelModalities(...values: unknown[]) {
  const modalities = stringArray(...values)
  if (!modalities) return undefined
  return [
    ...new Set(modalities.map((modality) => (modality.toLowerCase() === "vision" ? "image" : modality.toLowerCase()))),
  ]
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

function expandEnvRecord(values: Record<string, string> | undefined) {
  if (!values) return undefined
  return Object.fromEntries(
    Object.entries(values).flatMap(([key, value]) => {
      const expanded = expandEnv(value)
      return expanded === undefined ? [] : [[key, expanded]]
    }),
  )
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
  const release = await acquireCacheLock(path)
  try {
    const current = await readCache(path)
    const merged: Cache = { providers: { ...current.providers } }
    for (const [key, entry] of Object.entries(cache.providers)) {
      if (!merged.providers[key] || entry.checkedAt >= merged.providers[key].checkedAt) {
        merged.providers[key] = entry
      }
    }

    const temporaryPath = `${path}.${process.pid}.${randomUUID()}.tmp`
    try {
      await writeFile(temporaryPath, `${JSON.stringify(merged, null, 2)}\n`, { mode: 0o600 })
      await rename(temporaryPath, path)
      cache.providers = merged.providers
    } finally {
      await rm(temporaryPath, { force: true })
    }
  } finally {
    await release()
  }
}

async function acquireCacheLock(path: string) {
  const lockPath = `${path}.lock`
  for (let attempt = 0; attempt < 100; attempt += 1) {
    try {
      const handle = await open(lockPath, "wx", 0o600)
      return async () => {
        await handle.close()
        await rm(lockPath, { force: true })
      }
    } catch (error) {
      if (!isFileExistsError(error)) throw error
      try {
        const lock = await stat(lockPath)
        if (Date.now() - lock.mtimeMs > 30_000) await rm(lockPath, { force: true })
      } catch {
        // Another process released the lock while it was being inspected.
      }
      await delay(25)
    }
  }
  throw new Error(`Timed out waiting for model cache lock: ${path}`)
}

function isFileExistsError(error: unknown): error is NodeJS.ErrnoException {
  return error instanceof Error && "code" in error && error.code === "EEXIST"
}

export default plugin
