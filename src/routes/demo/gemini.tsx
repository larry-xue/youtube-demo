import { createFileRoute } from '@tanstack/react-router'
import {
  chat,
  type ContentPart,
  type ModelMessage,
  type StreamChunk,
} from '@tanstack/ai'
import {
  GeminiTextModels,
  createGeminiChat,
  type GeminiTextModel,
} from '@tanstack/ai-gemini'
import {
  AlertTriangle,
  Check,
  CircleStop,
  KeyRound,
  Plus,
  Sparkles,
  Trash2,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'

type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  createdAt: number
  attachments?: string[]
  status?: 'streaming' | 'done' | 'error'
  finishReason?: string
  usage?: {
    promptTokens: number
    completionTokens: number
    totalTokens: number
  }
  error?: string
}

type SessionSettings = {
  model: GeminiTextModel
  temperature: number
  topP: number
  maxTokens: number
  systemPrompt: string
  youtubeMode: boolean
  youtubeUrls: string
}

type Session = {
  id: string
  title: string
  createdAt: number
  updatedAt: number
  messages: ChatMessage[]
  settings: SessionSettings
}

const STORAGE_SESSIONS = 'gemini-demo.sessions.v1'
const STORAGE_ACTIVE = 'gemini-demo.active.v1'
const STORAGE_API_KEY = 'gemini-demo.apiKey.v1'
const STORAGE_CONFIG = 'gemini-demo.config.v1'

const DEFAULT_SETTINGS: SessionSettings = {
  model: 'gemini-2.5-flash',
  temperature: 0.7,
  topP: 0.9,
  maxTokens: 102400000,
  systemPrompt:
    'You are Gemini Studio. Be concise, helpful, and structure output with short sections.',
  youtubeMode: true,
  youtubeUrls: '',
}

const DATE_FORMATTER = new Intl.DateTimeFormat('en-US', {
  hour: '2-digit',
  minute: '2-digit',
})

export const Route = createFileRoute('/demo/gemini')({
  component: GeminiDemo,
})

function GeminiDemo() {
  const [defaultConfig, setDefaultConfig] = useState<SessionSettings>(() => {
    const storedConfig = loadFromStorage<SessionSettings>(STORAGE_CONFIG)
    return storedConfig
      ? { ...DEFAULT_SETTINGS, ...storedConfig }
      : { ...DEFAULT_SETTINGS }
  })

  const initialSessionRef = useRef<Session | null>(null)
  if (!initialSessionRef.current) {
    initialSessionRef.current = createSession(defaultConfig)
  }

  const [sessions, setSessions] = useState<Session[]>(() => [
    initialSessionRef.current!,
  ])
  const [activeSessionId, setActiveSessionId] = useState<string>(
    () => initialSessionRef.current!.id,
  )
  const [input, setInput] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [showKey, setShowKey] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [statusMessage, setStatusMessage] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const streamingRef = useRef<{ sessionId: string; messageId: string } | null>(
    null,
  )

  const activeSession = useMemo(
    () => sessions.find((session) => session.id === activeSessionId),
    [sessions, activeSessionId],
  )

  useEffect(() => {
    if (typeof window === 'undefined') {
      return
    }

    const storedSessions = loadFromStorage<Session[]>(STORAGE_SESSIONS)
    const storedActive = loadFromStorage<string>(STORAGE_ACTIVE)
    const storedKey = loadFromStorage<string>(STORAGE_API_KEY)
    const storedConfig = loadFromStorage<SessionSettings>(STORAGE_CONFIG)

    if (storedSessions?.length) {
      setSessions(storedSessions)
      setActiveSessionId(storedActive || storedSessions[0].id)
    } else {
      const resolvedConfig = storedConfig
        ? { ...DEFAULT_SETTINGS, ...storedConfig }
        : { ...DEFAULT_SETTINGS }
      setDefaultConfig(resolvedConfig)
      const initial = createSession(resolvedConfig)
      setSessions([initial])
      setActiveSessionId(initial.id)
    }

    if (storedKey) {
      setApiKey(storedKey)
    }
    if (storedConfig) {
      setDefaultConfig({ ...DEFAULT_SETTINGS, ...storedConfig })
    }
  }, [])

  useEffect(() => {
    if (typeof window === 'undefined') {
      return
    }
    saveToStorage(STORAGE_SESSIONS, sessions)
    saveToStorage(STORAGE_ACTIVE, activeSessionId)
  }, [sessions, activeSessionId])

  useEffect(() => {
    if (typeof window === 'undefined') {
      return
    }
    saveToStorage(STORAGE_API_KEY, apiKey)
  }, [apiKey])

  const handleNewSession = () => {
    const next = createSession(defaultConfig)
    setSessions((prev) => [next, ...prev])
    setActiveSessionId(next.id)
  }

  const handleDeleteSession = (sessionId: string) => {
    setSessions((prev) => {
      const remaining = prev.filter((session) => session.id !== sessionId)
      if (remaining.length === 0) {
        const fallback = createSession(defaultConfig)
        setActiveSessionId(fallback.id)
        return [fallback]
      }
      if (sessionId === activeSessionId) {
        setActiveSessionId(remaining[0].id)
      }
      return remaining
    })
  }

  const handleClearMessages = () => {
    if (!activeSession) return
    setSessions((prev) =>
      prev.map((session) =>
        session.id === activeSession.id
          ? {
              ...session,
              messages: [],
              updatedAt: Date.now(),
            }
          : session,
      ),
    )
  }

  const updateSessionById = (
    sessionId: string,
    updater: (session: Session) => Session,
  ) => {
    setSessions((prev) =>
      prev.map((session) =>
        session.id === sessionId ? updater(session) : session,
      ),
    )
  }

  const handleSend = async () => {
    if (!activeSession || !input.trim() || isStreaming) return
    if (!apiKey.trim()) {
      setStatusMessage('Add a Gemini API key before sending.')
      return
    }

    const youtubeMode = activeSession.settings.youtubeMode ?? false
    const youtubeUrls = parseYoutubeUrls(
      activeSession.settings.youtubeUrls ?? '',
    )
    if (youtubeMode && youtubeUrls.length === 0) {
      setStatusMessage('Add at least one YouTube URL for this mode.')
      return
    }

    const youtubeLimit = getYoutubeLimit(activeSession.settings.model)
    if (youtubeMode && youtubeUrls.length > youtubeLimit) {
      setStatusMessage(
        `This model supports up to ${youtubeLimit} YouTube URL${
          youtubeLimit === 1 ? '' : 's'
        } per request.`,
      )
      return
    }

    const userMessage: ChatMessage = {
      id: createId('user'),
      role: 'user',
      content: input.trim(),
      createdAt: Date.now(),
      attachments: youtubeMode ? youtubeUrls : undefined,
    }
    const assistantMessage: ChatMessage = {
      id: createId('assistant'),
      role: 'assistant',
      content: '',
      createdAt: Date.now(),
      status: 'streaming',
    }

    setInput('')
    setStatusMessage(null)
    setIsStreaming(true)

    const sessionId = activeSession.id
    const requestMessages = [
      ...activeSession.messages.map(toModelMessage),
      toModelMessage(userMessage),
    ]

    updateSessionById(sessionId, (session) => ({
      ...session,
      messages: [...session.messages, userMessage, assistantMessage],
      updatedAt: Date.now(),
      title:
        session.title === 'New Session'
          ? deriveTitle(userMessage)
          : session.title,
    }))

    abortRef.current?.abort()
    const abortController = new AbortController()
    abortRef.current = abortController
    streamingRef.current = { sessionId, messageId: assistantMessage.id }

    try {
      const adapter = createGeminiChat(
        activeSession.settings.model,
        apiKey.trim(),
      )

      for await (const chunk of chat({
        adapter,
        messages: requestMessages,
        systemPrompts: activeSession.settings.systemPrompt
          ? [activeSession.settings.systemPrompt]
          : undefined,
        temperature: activeSession.settings.temperature,
        topP: activeSession.settings.topP,
        maxTokens: activeSession.settings.maxTokens,
        abortController,
      })) {
        handleStreamChunk(chunk, sessionId, assistantMessage.id)
      }
    } catch (error) {
      updateAssistantMessage(sessionId, assistantMessage.id, (message) => ({
        ...message,
        status: 'error',
        error: error instanceof Error ? error.message : 'Unknown error.',
      }))
    } finally {
      setIsStreaming(false)
      streamingRef.current = null
    }
  }

  const handleStreamChunk = (
    chunk: StreamChunk,
    sessionId: string,
    assistantId: string,
  ) => {
    if (chunk.type === 'content') {
      updateAssistantMessage(sessionId, assistantId, (message) => ({
        ...message,
        content: chunk.content,
      }))
    }

    if (chunk.type === 'done') {
      updateAssistantMessage(sessionId, assistantId, (message) => ({
        ...message,
        status: 'done',
        finishReason: chunk.finishReason,
        usage: chunk.usage,
      }))
    }

    if (chunk.type === 'error') {
      updateAssistantMessage(sessionId, assistantId, (message) => ({
        ...message,
        status: 'error',
        error: chunk.error.message,
      }))
    }
  }

  const updateAssistantMessage = (
    sessionId: string,
    assistantId: string,
    updater: (message: ChatMessage) => ChatMessage,
  ) => {
    updateSessionById(sessionId, (session) => ({
      ...session,
      messages: session.messages.map((message) =>
        message.id === assistantId ? updater(message) : message,
      ),
      updatedAt: Date.now(),
    }))
  }

  const handleStop = () => {
    abortRef.current?.abort()
    setIsStreaming(false)
    if (streamingRef.current) {
      updateAssistantMessage(
        streamingRef.current.sessionId,
        streamingRef.current.messageId,
        (message) => ({
          ...message,
          status: 'error',
          error: 'Canceled by user.',
        }),
      )
      streamingRef.current = null
    }
  }

  const handleSettingsChange = (
    patch: Partial<SessionSettings>,
  ) => {
    if (!activeSession) return
    const nextSettings = { ...activeSession.settings, ...patch }
    updateSessionById(activeSession.id, (session) => ({
      ...session,
      settings: nextSettings,
      updatedAt: Date.now(),
    }))
    setDefaultConfig(nextSettings)
    saveToStorage(STORAGE_CONFIG, nextSettings)
  }

  return (
    <div
      className="gemini-demo min-h-screen pb-12 text-white"
      style={{
        backgroundImage:
          'radial-gradient(1200px 520px at 20% 10%, rgba(255, 212, 174, 0.18), transparent 60%), radial-gradient(900px 420px at 90% 20%, rgba(120, 212, 255, 0.16), transparent 60%), linear-gradient(135deg, #0c0d12 0%, #121722 45%, #0d0f14 100%)',
      }}
    >
      <div className="mx-auto w-full max-w-7xl px-4 pt-8">
        <header className="mb-6 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-white/10 text-amber-200 shadow-[0_12px_30px_rgba(255,183,77,0.25)]">
              <Sparkles className="h-6 w-6" />
            </div>
            <div>
              <p className="text-sm uppercase tracking-[0.2em] text-white/60">
                TanStack AI + Gemini
              </p>
              <h1 className="text-2xl font-semibold tracking-tight">
                Gemini Studio Demo
              </h1>
            </div>
          </div>
          <div className="flex flex-wrap gap-3">
            <button
              className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm transition hover:border-white/30 hover:bg-white/10"
              onClick={handleNewSession}
            >
              <Plus className="h-4 w-4" />
              New session
            </button>
            <button
              className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm transition hover:border-white/30 hover:bg-white/10"
              onClick={handleClearMessages}
            >
              <Trash2 className="h-4 w-4" />
              Clear chat
            </button>
          </div>
        </header>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[240px_minmax(0,1fr)_320px]">
          <section className="animate-in fade-in duration-700">
            <div className="rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur-xl shadow-[0_24px_60px_rgba(0,0,0,0.35)]">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-white/60">
                  Sessions
                </h2>
                <button
                  className="rounded-full border border-white/10 bg-white/10 px-3 py-1 text-xs uppercase tracking-[0.2em] text-white/70 transition hover:border-white/30"
                  onClick={handleNewSession}
                >
                  New
                </button>
              </div>
              <div className="flex flex-col gap-2">
                {sessions.map((session) => (
                  <button
                    key={session.id}
                    onClick={() => setActiveSessionId(session.id)}
                    className={`group flex items-start justify-between rounded-2xl border px-3 py-3 text-left transition ${
                      session.id === activeSessionId
                        ? 'border-amber-200/50 bg-amber-200/10 text-white shadow-[0_16px_36px_rgba(255,199,128,0.12)]'
                        : 'border-white/10 bg-white/5 text-white/70 hover:border-white/25 hover:text-white'
                    }`}
                  >
                    <div className="flex flex-1 flex-col gap-1">
                      <span className="text-sm font-medium">
                        {session.title}
                      </span>
                      <span className="text-xs text-white/50">
                        {session.messages.length} messages -{' '}
                        {DATE_FORMATTER.format(session.updatedAt)}
                      </span>
                    </div>
                    <button
                      className="ml-2 rounded-full border border-white/10 bg-white/5 p-1 text-white/60 opacity-0 transition hover:border-rose-300/50 hover:text-rose-200 group-hover:opacity-100"
                      aria-label="Delete session"
                      onClick={(event) => {
                        event.stopPropagation()
                        handleDeleteSession(session.id)
                      }}
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </button>
                ))}
              </div>
            </div>
          </section>

          <section className="animate-in fade-in duration-700">
            <div className="flex h-full flex-col rounded-[32px] border border-white/10 bg-white/5 p-5 backdrop-blur-xl shadow-[0_30px_70px_rgba(0,0,0,0.35)]">
              <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="text-xs uppercase tracking-[0.3em] text-white/50">
                    Active Model
                  </p>
                  <p className="text-lg font-semibold text-white">
                    {activeSession?.settings.model ?? DEFAULT_SETTINGS.model}
                  </p>
                </div>
                <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/10 px-3 py-1 text-xs text-white/70">
                  <Check className="h-3 w-3 text-emerald-200" />
                  Streaming enabled
                </div>
              </div>

              <div className="flex-1 space-y-4 overflow-y-auto pr-2">
                {activeSession?.messages.length === 0 && (
                  <div className="rounded-2xl border border-dashed border-white/20 bg-white/5 p-6 text-center text-sm text-white/60">
                    Start a new conversation. Your chat history stays in local
                    storage.
                  </div>
                )}
                {activeSession?.messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex ${
                      message.role === 'user' ? 'justify-end' : 'justify-start'
                    }`}
                  >
                    <div
                      className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-[0_12px_24px_rgba(0,0,0,0.2)] ${
                        message.role === 'user'
                          ? 'bg-gradient-to-br from-amber-200/20 to-amber-100/10 text-white'
                          : 'bg-white/10 text-white/90'
                      }`}
                    >
                      <p className="whitespace-pre-wrap">{message.content}</p>
                      {message.attachments?.length ? (
                        <div className="mt-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-[11px] text-white/70">
                          <p className="text-white/50">YouTube URLs</p>
                          <ul className="mt-1 space-y-1">
                            {message.attachments.map((url) => (
                              <li key={url} className="break-all">
                                {url}
                              </li>
                            ))}
                          </ul>
                        </div>
                      ) : null}
                      <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-white/50">
                        <span>{DATE_FORMATTER.format(message.createdAt)}</span>
                        {message.status === 'streaming' && (
                          <span className="animate-pulse text-amber-200">
                            streaming
                          </span>
                        )}
                        {message.finishReason && (
                          <span>finish: {message.finishReason}</span>
                        )}
                        {message.usage && (
                          <span>
                            tokens {message.usage.totalTokens}
                          </span>
                        )}
                      </div>
                      {message.error && (
                        <div className="mt-2 flex items-center gap-2 text-xs text-rose-200">
                          <AlertTriangle className="h-3 w-3" />
                          {message.error}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-4 rounded-2xl border border-white/10 bg-white/5 p-4">
                <div className="mb-2 flex items-center justify-between text-xs text-white/60">
                  <span>Prompt</span>
                  <span>{input.length} chars</span>
                </div>
                <textarea
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && !event.shiftKey) {
                      event.preventDefault()
                      handleSend()
                    }
                  }}
                  rows={4}
                  placeholder="Draft a prompt... (Enter to send, Shift+Enter for newline)"
                  className="w-full resize-none rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-sm text-white placeholder-white/40 outline-none focus:border-amber-200/60"
                />
                <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
                  <div className="text-xs text-white/50">
                    {statusMessage || 'Tip: Save API key in the right panel.'}
                  </div>
                  <div className="flex items-center gap-2">
                    {isStreaming ? (
                      <button
                        onClick={handleStop}
                        className="flex items-center gap-2 rounded-full border border-rose-300/40 bg-rose-300/10 px-4 py-2 text-xs uppercase tracking-[0.2em] text-rose-100 transition hover:border-rose-200"
                      >
                        <CircleStop className="h-4 w-4" />
                        Stop
                      </button>
                    ) : (
                      <button
                        onClick={handleSend}
                        className="flex items-center gap-2 rounded-full border border-amber-200/40 bg-amber-200/20 px-4 py-2 text-xs uppercase tracking-[0.2em] text-amber-50 transition hover:border-amber-200"
                      >
                        <Sparkles className="h-4 w-4" />
                        Send
                      </button>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className="animate-in fade-in duration-700">
            <div className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-xl shadow-[0_30px_70px_rgba(0,0,0,0.35)]">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-white/60">
                  Configuration
                </h2>
                <div className="flex items-center gap-2 text-xs text-white/50">
                  <KeyRound className="h-4 w-4 text-amber-200" />
                  Local only
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="text-xs uppercase tracking-[0.2em] text-white/50">
                    Gemini API Key
                  </label>
                  <div className="mt-2 flex items-center gap-2 rounded-2xl border border-white/10 bg-black/30 px-3 py-2">
                    <KeyRound className="h-4 w-4 text-white/40" />
                    <input
                      type={showKey ? 'text' : 'password'}
                      value={apiKey}
                      onChange={(event) => setApiKey(event.target.value)}
                      placeholder="AIza..."
                      className="flex-1 bg-transparent text-sm text-white outline-none placeholder:text-white/40"
                    />
                    <button
                      onClick={() => setShowKey((prev) => !prev)}
                      className="rounded-full border border-white/10 px-2 py-1 text-[11px] uppercase tracking-[0.2em] text-white/60 transition hover:border-white/30"
                    >
                      {showKey ? 'Hide' : 'Show'}
                    </button>
                  </div>
                </div>

                <div>
                  <label className="text-xs uppercase tracking-[0.2em] text-white/50">
                    Model
                  </label>
                  <select
                    value={
                      activeSession?.settings.model ?? DEFAULT_SETTINGS.model
                    }
                    onChange={(event) =>
                      handleSettingsChange({
                        model: event.target.value as GeminiTextModel,
                      })
                    }
                    className="mt-2 w-full rounded-2xl border border-white/10 bg-black/40 px-3 py-2 text-sm text-white outline-none focus:border-amber-200/60"
                  >
                    {GeminiTextModels.map((model) => (
                      <option key={model} value={model}>
                        {model}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-xs uppercase tracking-[0.2em] text-white/50">
                      Temperature
                    </label>
                    <input
                      type="number"
                      min={0}
                      max={2}
                      step={0.1}
                      value={
                        activeSession?.settings.temperature ??
                        DEFAULT_SETTINGS.temperature
                      }
                      onChange={(event) =>
                        handleSettingsChange({
                          temperature: Number(event.target.value),
                        })
                      }
                      className="mt-2 w-full rounded-2xl border border-white/10 bg-black/40 px-3 py-2 text-sm text-white outline-none focus:border-amber-200/60"
                    />
                  </div>
                  <div>
                    <label className="text-xs uppercase tracking-[0.2em] text-white/50">
                      Top P
                    </label>
                    <input
                      type="number"
                      min={0}
                      max={1}
                      step={0.05}
                      value={
                        activeSession?.settings.topP ?? DEFAULT_SETTINGS.topP
                      }
                      onChange={(event) =>
                        handleSettingsChange({
                          topP: Number(event.target.value),
                        })
                      }
                      className="mt-2 w-full rounded-2xl border border-white/10 bg-black/40 px-3 py-2 text-sm text-white outline-none focus:border-amber-200/60"
                    />
                  </div>
                </div>

                <div>
                  <label className="text-xs uppercase tracking-[0.2em] text-white/50">
                    Max Output Tokens
                  </label>
                  <input
                    type="number"
                    min={64}
                    max={65536}
                    step={64}
                    value={
                      activeSession?.settings.maxTokens ??
                      DEFAULT_SETTINGS.maxTokens
                    }
                    onChange={(event) =>
                      handleSettingsChange({
                        maxTokens: Number(event.target.value),
                      })
                    }
                    className="mt-2 w-full rounded-2xl border border-white/10 bg-black/40 px-3 py-2 text-sm text-white outline-none focus:border-amber-200/60"
                  />
                </div>

                <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs uppercase tracking-[0.2em] text-white/50">
                        YouTube Mode
                      </p>
                      <p className="text-sm text-white/70">
                        Pass YouTube URLs directly to Gemini.
                      </p>
                    </div>
                    <label className="flex items-center gap-2 text-xs text-white/60">
                      <input
                        type="checkbox"
                        checked={activeSession?.settings.youtubeMode ?? false}
                        onChange={(event) =>
                          handleSettingsChange({
                            youtubeMode: event.target.checked,
                          })
                        }
                        className="h-4 w-4 rounded border-white/20 bg-black/40 text-amber-200"
                      />
                      Enable
                    </label>
                  </div>
                  {activeSession?.settings.youtubeMode && (
                    <div className="mt-3">
                      <label className="text-xs uppercase tracking-[0.2em] text-white/50">
                        YouTube URLs
                      </label>
                      <textarea
                        rows={4}
                        value={
                          activeSession?.settings.youtubeUrls ??
                          DEFAULT_SETTINGS.youtubeUrls
                        }
                        onChange={(event) =>
                          handleSettingsChange({
                            youtubeUrls: event.target.value,
                          })
                        }
                        placeholder="https://www.youtube.com/watch?v=..."
                        className="mt-2 w-full resize-none rounded-2xl border border-white/10 bg-black/40 px-3 py-2 text-sm text-white outline-none focus:border-amber-200/60"
                      />
                      <p className="mt-2 text-[11px] text-white/50">
                        Limits: free tier is 8 hours/day. Gemini 2.5+ supports
                        up to 10 videos per request. Older models allow 1.
                      </p>
                    </div>
                  )}
                </div>

                <div>
                  <label className="text-xs uppercase tracking-[0.2em] text-white/50">
                    System Prompt
                  </label>
                  <textarea
                    rows={5}
                    value={
                      activeSession?.settings.systemPrompt ??
                      DEFAULT_SETTINGS.systemPrompt
                    }
                    onChange={(event) =>
                      handleSettingsChange({
                        systemPrompt: event.target.value,
                      })
                    }
                    className="mt-2 w-full resize-none rounded-2xl border border-white/10 bg-black/40 px-3 py-2 text-sm text-white outline-none focus:border-amber-200/60"
                  />
                </div>

                <div className="rounded-2xl border border-white/10 bg-white/5 p-3 text-xs text-white/60">
                  API keys and sessions are stored in local storage for quick
                  demos. For production, proxy requests through a server.
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}

function createSession(settings: SessionSettings = DEFAULT_SETTINGS): Session {
  const now = Date.now()
  return {
    id: createId('session'),
    title: 'New Session',
    createdAt: now,
    updatedAt: now,
    messages: [],
    settings: { ...settings },
  }
}

function createId(prefix: string) {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return `${prefix}-${crypto.randomUUID()}`
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
}

function loadFromStorage<T>(key: string): T | null {
  if (typeof window === 'undefined') return null
  try {
    const value = window.localStorage.getItem(key)
    return value ? (JSON.parse(value) as T) : null
  } catch {
    return null
  }
}

function saveToStorage<T>(key: string, value: T) {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(key, JSON.stringify(value))
  } catch {
    // Ignore storage errors.
  }
}

function toModelMessage(message: ChatMessage): ModelMessage {
  if (message.role === 'user' && message.attachments?.length) {
    const parts: ContentPart[] = [
      { type: 'text', content: message.content },
      ...message.attachments.map((url) => ({
        type: 'video',
        source: { type: 'file', value: url },
      })),
    ]
    return {
      role: message.role,
      content: parts,
    }
  }
  return {
    role: message.role,
    content: message.content,
  }
}

function deriveTitle(message: ChatMessage) {
  const trimmed = message.content.trim()
  if (!trimmed) return 'New Session'
  return trimmed.length > 28 ? `${trimmed.slice(0, 28)}...` : trimmed
}

function parseYoutubeUrls(input: string) {
  return input
    .split(/\s+/)
    .map((entry) => entry.trim())
    .filter(Boolean)
}

function getYoutubeLimit(model: string) {
  return model.includes('2.5') ? 10 : 1
}
