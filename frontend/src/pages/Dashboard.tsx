import React, { useState, useEffect, useCallback, useRef } from 'react'
import { Square, PlayCircle, Trash2, Loader2 } from 'lucide-react'
import ProviderStatus from '../components/ProviderStatus'

// ── Types ──────────────────────────────────────────────────────────────────

interface Prompt {
  id: number
  topic: string
  prompt: string
}

interface RunSummary {
  id: string
  prompt_id: number
  prompt_text: string
  mode: string
  status: string
  started_at: string | null
  completed_at: string | null
  total_corrections: number
  video_provider_used: string | null
  video_provider_choice: string | null
  has_final_video: boolean
  final_video_path: string | null
}

interface StageAttempt {
  id: string
  stage_name: string
  attempt_number: number
  status: string
  output_path: string | null
  human_decision: string | null
  reviewer_notes: string | null
  created_at: string | null
  // Script features
  readability_score: number | null
  lexical_diversity: number | null
  prompt_coverage: number | null
  sentence_redundancy: number | null
  entity_consistency: number | null
  topic_coherence: number | null
  factual_conflict_flag: number | null
  prompt_ambiguity: number | null
  // Audio features
  phoneme_error_rate: number | null
  silence_ratio: number | null
  speaking_rate_variance: number | null
  energy_variance: number | null
  tts_word_count_match: number | null
  // Visual features
  clip_similarity: number | null
  aesthetic_score: number | null
  blur_score: number | null
  object_match_score: number | null
  colour_tone_match: number | null
  visual_provider: string | null
  // Video features
  av_sync_error_ms: number | null
  transition_smoothness: number | null
  duration_deviation_s: number | null
  // Cross-stage
  prior_stage_corrections: number | null
  cumulative_risk_score: number | null
  api_retry_count: number | null
  is_fallback_video: boolean | null
}

interface RunStatus extends RunSummary {
  stages: StageAttempt[]
}

// ── Constants ──────────────────────────────────────────────────────────────

const STATUS_COLOURS: Record<string, string> = {
  running: 'bg-blue-900 text-blue-300',
  completed: 'bg-green-900 text-green-300',
  accepted: 'bg-green-600 text-green-100',
  failed: 'bg-red-900 text-red-300',
  // "stopped" uses slate (cool grey) — clearly distinct from the red of "failed"
  stopped: 'bg-slate-600 text-slate-100',
  pending: 'bg-blue-900 text-blue-300',
  queued: 'bg-gray-700 text-gray-400',
  pending_review: 'bg-yellow-900 text-yellow-300',
  rejected: 'bg-orange-900 text-orange-300',
}

const MODE_LABELS: Record<string, string> = {
  A: 'Mode A (Auto)',
  B: 'Mode B (HITL)',
  C: 'Mode C (ML)',
}

const STAGE_ICONS: Record<string, string> = {
  script: '📝',
  audio: '🎵',
  visual: '🎬',
  video: '📹',
}

const STAGE_FEATURES: Record<string, Array<[string, string]>> = {
  script: [
    ['readability_score', 'Readability'],
    ['lexical_diversity', 'Lexical Diversity'],
    ['prompt_coverage', 'Prompt Coverage'],
    ['sentence_redundancy', 'Redundancy'],
    ['entity_consistency', 'Entity Consistency'],
    ['topic_coherence', 'Topic Coherence'],
    ['factual_conflict_flag', 'Conflict Flag'],
    ['prompt_ambiguity', 'Ambiguity'],
  ],
  audio: [
    ['phoneme_error_rate', 'Phoneme Error Rate'],
    ['silence_ratio', 'Silence Ratio'],
    ['speaking_rate_variance', 'Rate Variance'],
    ['energy_variance', 'Energy Variance'],
    ['tts_word_count_match', 'Word Count Match'],
  ],
  visual: [
    ['clip_similarity', 'CLIP Similarity'],
    ['aesthetic_score', 'Aesthetic Score'],
    ['blur_score', 'Blur Score'],
    ['object_match_score', 'Object Match'],
    ['colour_tone_match', 'Colour Tone Match'],
  ],
  video: [
    ['av_sync_error_ms', 'AV Sync Error (ms)'],
    ['transition_smoothness', 'Transition Smoothness'],
    ['duration_deviation_s', 'Duration Deviation (s)'],
  ],
}

// ── Toast system ───────────────────────────────────────────────────────────

type ToastType = 'success' | 'error' | 'warning' | 'info'

interface Toast {
  id: number
  msg: string
  type: ToastType
}

const TOAST_COLOURS: Record<ToastType, string> = {
  success: 'bg-green-800 text-green-100 border-green-700',
  error:   'bg-red-800 text-red-100 border-red-700',
  warning: 'bg-yellow-800 text-yellow-100 border-yellow-700',
  info:    'bg-blue-800 text-blue-100 border-blue-700',
}

const TOAST_ICONS: Record<ToastType, string> = {
  success: '✓',
  error:   '✕',
  warning: '⚠',
  info:    'ℹ',
}

let _toastId = 0

function ToastContainer({
  toasts,
  onDismiss,
}: {
  toasts: Toast[]
  onDismiss: (id: number) => void
}) {
  if (!toasts.length) return null
  return (
    <div className="fixed top-4 right-4 z-[60] flex flex-col gap-2 w-80">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`flex items-start gap-2 px-4 py-3 rounded-lg text-sm shadow-lg border ${TOAST_COLOURS[t.type]} animate-in`}
        >
          <span className="font-bold shrink-0 mt-0.5">{TOAST_ICONS[t.type]}</span>
          <span className="flex-1">{t.msg}</span>
          <button
            onClick={() => onDismiss(t.id)}
            className="shrink-0 opacity-60 hover:opacity-100 leading-none ml-1"
          >
            ×
          </button>
        </div>
      ))}
    </div>
  )
}

// ── Small shared components ────────────────────────────────────────────────

function StatusBadge({ status }: { status: string }) {
  const cls = STATUS_COLOURS[status] ?? 'bg-gray-700 text-gray-300'
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${cls}`}>
      {status.replace('_', ' ')}
    </span>
  )
}

function StageProgressBar({ stages }: { stages: StageAttempt[] }) {
  const stageNames = ['script', 'audio', 'visual', 'video']
  return (
    <div className="flex gap-1 mt-1">
      {stageNames.map((name) => {
        const attemptsForStage = stages.filter((s) => s.stage_name === name)
        const latest = attemptsForStage.sort((a, b) => b.attempt_number - a.attempt_number)[0]
        // A stage is "done" if it was accepted or completed without rejection
        const isDone =
          latest?.status === 'accepted' ||
          latest?.status === 'completed' ||
          latest?.human_decision === 'accept'
        const bgCls = isDone
          ? 'bg-green-500'
          : latest?.status === 'pending_review'
          ? 'bg-yellow-500'
          : latest?.status === 'failed'
          ? 'bg-red-600'
          : latest?.status === 'rejected'
          ? 'bg-orange-500'
          : latest?.status === 'running' || latest?.status === 'pending'
          ? 'bg-blue-600'
          : 'bg-gray-700'
        const label = latest ? `${name}: ${latest.status}` : `${name}: not started`
        return <div key={name} title={label} className={`h-2 flex-1 rounded-full ${bgCls}`} />
      })}
    </div>
  )
}

// ── Script preview (loads JSON from backend) ───────────────────────────────

function ScriptPreview({ runId }: { runId: string }) {
  const [scenes, setScenes] = useState<Array<{ scene_number: number; narration: string; visual_prompt: string }>>([])
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    fetch(`/runs/${runId}/output/script`)
      .then((r) => (r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`)))
      .then((d) => setScenes(d?.scenes ?? []))
      .catch((e) => setErr(String(e)))
  }, [runId])

  if (err) return <p className="text-red-400 text-xs">{err}</p>
  if (!scenes.length) return <p className="text-gray-500 text-xs">Loading script…</p>

  return (
    <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
      {scenes.map((sc) => (
        <div key={sc.scene_number} className="bg-gray-800 rounded-lg p-2.5 border border-gray-700">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-medium text-blue-300 bg-blue-900/40 px-1.5 py-0.5 rounded">
              Scene {sc.scene_number}
            </span>
          </div>
          <p className="text-xs text-white mb-0.5">{sc.narration}</p>
          <p className="text-xs text-gray-500 italic">{sc.visual_prompt}</p>
        </div>
      ))}
    </div>
  )
}

// ── Feature scores table ───────────────────────────────────────────────────

function FeatureScores({ attempt }: { attempt: StageAttempt }) {
  const features = STAGE_FEATURES[attempt.stage_name] ?? []
  const attemptMap = attempt as unknown as Record<string, unknown>
  const rows = features.filter(([key]) => attemptMap[key] !== null)
  if (!rows.length) return <p className="text-gray-500 text-xs">No features extracted yet.</p>

  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-1">
      {rows.map(([key, label]) => {
        const val = attemptMap[key]
        const num = typeof val === 'number' ? val : null
        return (
          <div key={key} className="flex justify-between text-xs py-0.5 border-b border-gray-800">
            <span className="text-gray-400">{label}</span>
            <span className={`font-mono font-medium ${num !== null && num > 0.7 ? 'text-green-400' : num !== null && num < 0.3 ? 'text-red-400' : 'text-gray-200'}`}>
              {num !== null ? num.toFixed(3) : String(val)}
            </span>
          </div>
        )
      })}
    </div>
  )
}

// ── Visual media grid (provider-aware: images for ken_burns, clips for modal) ──

function VisualMediaGrid({ runId, provider }: { runId: string; provider: string | null }) {
  const isModal = provider === 'modal'
  const endpoint = isModal ? `/runs/${runId}/output/scenes` : `/runs/${runId}/output/images`
  const [files, setFiles] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch(endpoint)
      .then((r) => (r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`)))
      .then((names: string[]) => { setFiles(names); setLoading(false) })
      .catch((e) => { setError(String(e)); setLoading(false) })
  }, [endpoint])

  if (loading) return <p className="text-gray-500 text-sm">Loading {isModal ? 'clips' : 'images'}…</p>
  if (error) return <p className="text-red-400 text-sm">Failed to load: {error}</p>
  if (!files.length) return (
    <div className="rounded-lg border border-red-800/50 bg-red-900/20 p-3 text-sm text-red-300">
      No {isModal ? 'scene clips' : 'images'} found for this run.
    </div>
  )

  return (
    <div className="space-y-1">
      <p className="text-xs text-gray-500">
        {files.length} {isModal ? 'clip' : 'image'}{files.length !== 1 ? 's' : ''} generated
        {provider && <span className="ml-2 text-gray-600">({provider})</span>}
      </p>
      <div className="flex flex-wrap gap-2">
        {files.map((name) =>
          isModal ? (
            <div key={name} className="rounded overflow-hidden flex-shrink-0"
              style={{ width: 320, height: 180, background: 'black' }}>
              <video
                src={`/runs/${runId}/output/scenes/${name}`}
                controls
                muted
                style={{ width: '100%', height: '100%', objectFit: 'contain', background: 'black' }}
                title={name}
              />
            </div>
          ) : (
            <div key={name} className="rounded overflow-hidden flex-shrink-0"
              style={{ width: 120, height: 213, background: 'black' }}>
              <img
                src={`/runs/${runId}/output/images/${name}`}
                alt={name}
                style={{ width: '100%', height: '100%', objectFit: 'contain', background: 'black' }}
                loading="lazy"
                title={name}
              />
            </div>
          )
        )}
      </div>
    </div>
  )
}


// ── Inline review panel ────────────────────────────────────────────────────

function ReviewPanel({
  attempt,
  runId,
  runText,
  videoProvider,
  onDecision,
  onClose,
  onMediaError,
}: {
  attempt: StageAttempt
  runId: string
  runText: string
  videoProvider: string | null
  onDecision: (attemptId: string, decision: 'accept' | 'reject', notes: string) => Promise<void>
  onClose: () => void
  onMediaError: (msg: string) => void
}) {
  const [notes, setNotes] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submit = async (decision: 'accept' | 'reject') => {
    setSubmitting(true)
    setError(null)
    try {
      await onDecision(attempt.id, decision, notes)
      // onDecision closes the modal via setSelectedRun(null) on success
    } catch (e) {
      setError(String(e))
      setSubmitting(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 border border-yellow-700/50 rounded-xl w-full max-w-2xl max-h-[90vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-800">
          <div className="flex items-center gap-3">
            <span className="text-xl">{STAGE_ICONS[attempt.stage_name] ?? '📋'}</span>
            <div>
              <h2 className="font-semibold capitalize">
                Review: {attempt.stage_name} stage
              </h2>
              <p className="text-xs text-gray-400">
                Attempt {attempt.attempt_number} · {runText}
              </p>
            </div>
          </div>
          <button onClick={onClose} className="text-gray-500 hover:text-white text-xl leading-none">×</button>
        </div>

        <div className="overflow-y-auto flex-1 px-5 py-4 space-y-5">
          {/* Output preview */}
          <div>
            <h3 className="text-sm font-medium text-gray-300 mb-2">Output Preview</h3>
            {attempt.stage_name === 'script' ? (
              <ScriptPreview runId={runId} />
            ) : attempt.stage_name === 'audio' ? (
              <div className="space-y-2">
                <audio
                  controls
                  style={{ width: '100%' }}
                  onError={() => onMediaError('Audio file could not be loaded — it may be empty or corrupt.')}
                >
                  <source src={`/runs/${runId}/output/audio`} type="audio/wav" />
                  <source src={`/runs/${runId}/output/audio`} type="audio/mpeg" />
                  Your browser does not support the audio element.
                </audio>
                <div className="flex flex-wrap gap-4 text-xs text-gray-500">
                  {attempt.silence_ratio !== null && (
                    <span>Silence: <span className="text-gray-300">{(attempt.silence_ratio! * 100).toFixed(1)}%</span></span>
                  )}
                  {attempt.speaking_rate_variance !== null && (
                    <span>Rate variance: <span className="text-gray-300">{attempt.speaking_rate_variance!.toFixed(4)}</span></span>
                  )}
                  {attempt.tts_word_count_match !== null && (
                    <span>Word match: <span className={attempt.tts_word_count_match! >= 0.9 ? 'text-green-400' : 'text-yellow-400'}>{(attempt.tts_word_count_match! * 100).toFixed(0)}%</span></span>
                  )}
                  {attempt.phoneme_error_rate !== null && (
                    <span>Phoneme error: <span className={attempt.phoneme_error_rate! < 0.1 ? 'text-green-400' : 'text-red-400'}>{attempt.phoneme_error_rate!.toFixed(3)}</span></span>
                  )}
                </div>
              </div>
            ) : attempt.stage_name === 'visual' ? (
              <VisualMediaGrid runId={runId} provider={attempt.visual_provider} />
            ) : attempt.stage_name === 'video' ? (
              <div className="flex justify-center">
                {(() => {
                  const videoUrl = `/runs/${runId}/video`
                  const isModalProvider = videoProvider === 'modal'
                  return (
                    <div style={isModalProvider
                      ? { maxWidth: 640, width: '100%', aspectRatio: '16/9', background: 'black', borderRadius: 6 }
                      : { maxWidth: 280, width: '100%', aspectRatio: '9/16', background: 'black', borderRadius: 6 }
                    }>
                      <video
                        controls
                        style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                        onError={(e) => {
                          console.error('[ReviewPanel] video load error:', e, 'src:', videoUrl)
                          onMediaError('Video file could not be loaded — it may be empty or corrupt.')
                        }}
                      >
                        <source src={videoUrl} type="video/mp4" />
                        Your browser does not support video playback.
                      </video>
                    </div>
                  )
                })()}</div>
            ) : attempt.output_path ? (
              <video
                controls
                className="w-full rounded"
                src={`/api/media?path=${encodeURIComponent(attempt.output_path)}`}
                onError={() => onMediaError('Video file could not be loaded — it may be empty or corrupt.')}
              />
            ) : (
              <p className="text-gray-500 text-sm">No output file available yet.</p>
            )}
          </div>

          {/* Feature scores */}
          <div>
            <h3 className="text-sm font-medium text-gray-300 mb-2">Extracted Feature Scores</h3>
            <FeatureScores attempt={attempt} />
          </div>

          {/* Cross-stage info */}
          <div className="flex gap-4 text-xs text-gray-500">
            {attempt.prior_stage_corrections !== null && (
              <span>Prior corrections: <span className="text-white">{attempt.prior_stage_corrections}</span></span>
            )}
            {attempt.cumulative_risk_score !== null && (
              <span>Cumulative risk: <span className="text-white">{attempt.cumulative_risk_score?.toFixed(3)}</span></span>
            )}
            {attempt.is_fallback_video && (
              <span className="text-orange-400">Ken Burns fallback</span>
            )}
          </div>
        </div>

        {/* Decision controls */}
        <div className="px-5 py-4 border-t border-gray-800 space-y-3">
          {error && <p className="text-red-400 text-xs">{error}</p>}
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Reviewer notes (optional)…"
            rows={2}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:border-blue-500"
          />
          <div className="flex gap-3">
            <button
              onClick={() => submit('reject')}
              disabled={submitting}
              className="flex-1 py-2.5 bg-red-700 hover:bg-red-600 rounded-lg text-sm font-medium disabled:opacity-50 transition-colors"
            >
              {submitting ? 'Submitting…' : 'Reject'}
            </button>
            <button
              onClick={() => submit('accept')}
              disabled={submitting}
              className="flex-1 py-2.5 bg-green-700 hover:bg-green-600 rounded-lg text-sm font-medium disabled:opacity-50 transition-colors"
            >
              {submitting ? 'Submitting…' : 'Accept'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Final video player ─────────────────────────────────────────────────────

function FinalVideoPlayer({ runId }: { runId: string }) {
  const [isLandscape, setIsLandscape] = useState<boolean | null>(null)
  const videoUrl = `/runs/${runId}/video`
  return (
    <div className="space-y-3">
      <h3 className="text-sm font-medium text-gray-300">Final Video</h3>
      <div className="flex justify-center">
        <div style={isLandscape === true
          ? { maxWidth: 640, width: '100%', aspectRatio: '16/9', background: 'black', borderRadius: 6 }
          : { maxWidth: 280, width: '100%', aspectRatio: '9/16', background: 'black', borderRadius: 6 }
        }>
          <video
            controls
            src={videoUrl}
            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
            onLoadedMetadata={(e) => {
              const v = e.currentTarget
              setIsLandscape(v.videoWidth > v.videoHeight)
            }}
          >
            <source src={videoUrl} type="video/mp4" />
            Your browser does not support video playback.
          </video>
        </div>
      </div>
    </div>
  )
}


// ── Start run modal ────────────────────────────────────────────────────────

function StartRunModal({
  prompts,
  onClose,
  onStart,
}: {
  prompts: Prompt[]
  onClose: () => void
  onStart: (promptId: number, mode: string, videoProviderChoice: string) => Promise<void>
}) {
  const [selectedPrompt, setSelectedPrompt] = useState(1)
  const [selectedMode, setSelectedMode] = useState('B')
  const [videoProvider, setVideoProvider] = useState<'modal' | 'ken_burns'>('ken_burns')
  const [loading, setLoading] = useState(false)

  const handleStart = async () => {
    setLoading(true)
    await onStart(selectedPrompt, selectedMode, videoProvider)
    setLoading(false)
    onClose()
  }

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
      <div className="bg-gray-900 border border-gray-700 rounded-xl p-6 w-full max-w-lg shadow-2xl">
        <h2 className="text-xl font-semibold mb-4">Start New Pipeline Run</h2>

        <div className="mb-4">
          <label className="block text-sm text-gray-400 mb-1">Select Prompt</label>
          <select
            value={selectedPrompt}
            onChange={(e) => setSelectedPrompt(Number(e.target.value))}
            className="w-full bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-sm"
          >
            {prompts.map((p) => (
              <option key={p.id} value={p.id}>
                {p.id}. {p.topic} — {p.prompt}
              </option>
            ))}
          </select>
        </div>

        <div className="mb-4">
          <label className="block text-sm text-gray-400 mb-1">Video Generation</label>
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={() => setVideoProvider('ken_burns')}
              className={`px-4 py-2.5 rounded-lg text-sm font-medium border transition-colors ${
                videoProvider === 'ken_burns'
                  ? 'bg-gray-600 border-gray-500 text-white'
                  : 'bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-500'
              }`}
            >
              Ken Burns (Fast)
            </button>
            <button
              onClick={() => setVideoProvider('modal')}
              className={`px-4 py-2.5 rounded-lg text-sm font-medium border transition-colors ${
                videoProvider === 'modal'
                  ? 'bg-purple-700 border-purple-500 text-white'
                  : 'bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-500'
              }`}
            >
              Modal (AI Video)
            </button>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            {videoProvider === 'ken_burns' && 'Local FFmpeg zoom/pan effect. Fast, no API needed.'}
            {videoProvider === 'modal' && 'CogVideoX-2b on Modal A10G GPU. Requires MODAL_ENDPOINT_URL in .env.'}
          </p>
        </div>

        <div className="mb-6">
          <label className="block text-sm text-gray-400 mb-1">Pipeline Mode</label>
          <div className="grid grid-cols-3 gap-2">
            {(['A', 'B', 'C'] as const).map((m) => (
              <button
                key={m}
                onClick={() => setSelectedMode(m)}
                className={`px-4 py-2 rounded-lg text-sm font-medium border transition-colors ${
                  selectedMode === m
                    ? 'bg-blue-600 border-blue-500 text-white'
                    : 'bg-gray-800 border-gray-600 text-gray-300 hover:border-gray-400'
                }`}
              >
                {MODE_LABELS[m]}
              </button>
            ))}
          </div>
          <p className="text-xs text-gray-500 mt-2">
            {selectedMode === 'A' && 'Fully automated. No human review.'}
            {selectedMode === 'B' && 'Human reviews each stage. Generates ML training data.'}
            {selectedMode === 'C' && 'ML model predicts when human review is needed.'}
          </p>
        </div>

        <div className="flex gap-3 justify-end">
          <button onClick={onClose} className="px-4 py-2 text-sm text-gray-400 hover:text-white">
            Cancel
          </button>
          <button
            onClick={handleStart}
            disabled={loading}
            className="px-5 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-sm font-medium disabled:opacity-50 transition-colors"
          >
            {loading ? 'Starting…' : 'Start Run'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Main Dashboard ─────────────────────────────────────────────────────────

export default function Dashboard() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [selectedRun, setSelectedRun] = useState<RunStatus | null>(null)
  const [prompts, setPrompts] = useState<Prompt[]>([])
  const [showStartModal, setShowStartModal] = useState(false)
  const [toasts, setToasts] = useState<Toast[]>([])
  const [stageMap, setStageMap] = useState<Record<string, StageAttempt[]>>({})
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [stoppingId, setStoppingId] = useState<string | null>(null)
  const [resumingId, setResumingId] = useState<string | null>(null)
  // Track previous run statuses to detect transitions (running → failed/stopped)
  const prevStatusRef = useRef<Record<string, string>>({})
  // Track which runs have already triggered a timeout toast to avoid repeats
  const timeoutToastedRef = useRef<Set<string>>(new Set())

  const dismissToast = (id: number) =>
    setToasts((prev) => prev.filter((t) => t.id !== id))

  const showToast = useCallback((msg: string, type: ToastType = 'success') => {
    const id = ++_toastId
    setToasts((prev) => [...prev, { id, msg, type }])
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 5000)
  }, [])

  const fetchRuns = useCallback(async () => {
    try {
      const res = await fetch('/runs')
      if (!res.ok) return
      const data: RunSummary[] = await res.json()

      // Detect status transitions and surface toasts
      for (const run of data) {
        const prev = prevStatusRef.current[run.id]
        if (prev && prev !== run.status) {
          const topic = run.prompt_text.slice(0, 50)
          if (prev === 'running' && run.status === 'failed') {
            showToast(`Run failed: "${topic}"`, 'error')
          } else if (prev === 'running' && run.status === 'completed') {
            showToast(`Run completed: "${topic}"`, 'success')
          }
        }
        prevStatusRef.current[run.id] = run.status
      }

      setRuns(data)

      // Keep stage progress for running rows
      const runningIds = data.filter((r) => r.status === 'running').map((r) => r.id)
      const map: Record<string, StageAttempt[]> = {}
      await Promise.all(
        runningIds.map(async (id) => {
          const r = await fetch(`/runs/${id}/status`)
          if (r.ok) {
            const s: RunStatus = await r.json()
            map[id] = s.stages

            // Detect review timeouts — reviewer_notes set by orchestrator on timeout
            for (const stage of s.stages) {
              const timeoutKey = `${id}:${stage.id}`
              if (
                stage.reviewer_notes?.includes('Auto-accepted: reviewer timeout') &&
                !timeoutToastedRef.current.has(timeoutKey)
              ) {
                timeoutToastedRef.current.add(timeoutKey)
                const run = data.find((r) => r.id === id)
                const topic = run?.prompt_text.slice(0, 40) ?? id.slice(0, 8)
                showToast(
                  `Review timed out — ${stage.stage_name} auto-accepted: "${topic}"`,
                  'warning',
                )
              }
            }
          }
        }),
      )
      setStageMap((prev) => ({ ...prev, ...map }))

      // Refresh selected run if open
      setSelectedRun((prev) => {
        if (!prev) return prev
        const still = data.find((r) => r.id === prev.id)
        if (!still) return prev
        return prev
      })
    } catch {
      // Non-fatal
    }
  }, [showToast])

  const fetchPrompts = useCallback(async () => {
    const res = await fetch('/prompts').catch(() => null)
    if (res?.ok) setPrompts(await res.json())
  }, [])

  useEffect(() => {
    fetchRuns()
    fetchPrompts()
    const interval = setInterval(fetchRuns, 3000)
    return () => clearInterval(interval)
  }, [fetchRuns, fetchPrompts])

  // Click a run row — fetch full status and open review panel if pending_review
  const handleRowClick = async (runId: string) => {
    try {
      const res = await fetch(`/runs/${runId}/status`)
      if (!res.ok) return
      const data: RunStatus = await res.json()
      setSelectedRun(data)
    } catch {
      showToast('Could not load run details.', 'error')
    }
  }

  // Find the stage currently awaiting review in the selected run.
  // Exclude any attempt that already has a human_decision recorded — the review
  // endpoint now sets status immediately, but this guard handles any race window.
  const pendingStage =
    selectedRun?.stages.find((s) => s.status === 'pending_review' && !s.human_decision) ?? null

  const handleDecision = async (attemptId: string, decision: 'accept' | 'reject', notes: string) => {
    const runId = selectedRun?.id
    const res = await fetch(`/review/${attemptId}/decide`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision, notes }),
    })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(`HTTP ${res.status}${text ? ': ' + text : ''}`)
    }
    // Close modal immediately — feels responsive
    setSelectedRun(null)
    showToast(`${decision === 'accept' ? '✓ Accepted' : '✗ Rejected'} — pipeline continuing…`)
    // Optimistically clear pending_review from stageMap so the banner disappears
    // right away instead of waiting for the orchestrator to process the decision
    if (runId) {
      setStageMap((prev) => ({
        ...prev,
        [runId]: (prev[runId] ?? []).map((s) =>
          s.status === 'pending_review'
            ? { ...s, status: decision === 'accept' ? 'accepted' : 'rejected', human_decision: decision }
            : s
        ),
      }))
    }
    await fetchRuns()
  }

  const handleDeleteRun = async (runId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (!confirm('Delete this run and all its output files? This cannot be undone.')) return
    setDeletingId(runId)
    try {
      const res = await fetch(`/runs/${runId}`, { method: 'DELETE' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      showToast('Run deleted.')
      if (selectedRun?.id === runId) setSelectedRun(null)
      await fetchRuns()
    } catch (e) {
      showToast(`Delete failed: ${e}`, 'error')
    } finally {
      setDeletingId(null)
    }
  }

  const handleResumeRun = async (runId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setResumingId(runId)
    try {
      const res = await fetch(`/runs/${runId}/resume`, { method: 'POST' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      showToast('Run resumed — pipeline continuing from last stage.', 'info')
      setRuns((prev) => prev.map((r) => r.id === runId ? { ...r, status: 'running' } : r))
      await fetchRuns()
    } catch (e) {
      showToast(`Resume failed: ${e}`, 'error')
    } finally {
      setResumingId(null)
    }
  }

  const handleStopRun = async (runId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (!confirm('Stop this run? The pipeline will be halted and marked as stopped.')) return
    setStoppingId(runId)
    try {
      const res = await fetch(`/runs/${runId}/stop`, { method: 'POST' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      showToast('Run stopped.', 'info')
      if (selectedRun?.id === runId) setSelectedRun(null)
      // Optimistically update run status in the list
      setRuns((prev) => prev.map((r) => r.id === runId ? { ...r, status: 'stopped' } : r))
      // Clear pending review badge
      setStageMap((prev) => ({
        ...prev,
        [runId]: (prev[runId] ?? []).map((s) =>
          s.status === 'pending_review' ? { ...s, status: 'stopped' } : s
        ),
      }))
      await fetchRuns()
    } catch (e) {
      showToast(`Stop failed: ${e}`, 'error')
    } finally {
      setStoppingId(null)
    }
  }

  const handleStartRun = async (promptId: number, mode: string, videoProviderChoice: string) => {
    const res = await fetch('/runs/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt_id: promptId, mode, video_provider_choice: videoProviderChoice }),
    })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    await fetchRuns()
  }

  const formatDuration = (start: string | null, end: string | null) => {
    if (!start) return '—'
    const secs = Math.round((new Date(end ?? Date.now()).getTime() - new Date(start).getTime()) / 1000)
    if (secs < 60) return `${secs}s`
    return `${Math.floor(secs / 60)}m ${secs % 60}s`
  }

  const hasPendingReview = (runId: string) =>
    (stageMap[runId] ?? []).some((s) => s.status === 'pending_review' && !s.human_decision)

  const awaitingReview = runs.filter((r) => r.status === 'running' && hasPendingReview(r.id)).length

  return (
    <div className="max-w-7xl mx-auto px-4">
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />

      {/* Sticky header — title + stats bar */}
      <div className="sticky top-[49px] z-10 bg-gray-950 pt-4 pb-3">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h1 className="text-xl font-bold leading-tight">Pipeline Dashboard</h1>
            <p className="text-xs text-gray-400 mt-0.5">
              {runs.length} runs · {runs.filter((r) => r.status === 'running').length} active ·{' '}
              <span className="text-yellow-400">{awaitingReview} awaiting review</span>
            </p>
          </div>
          <button
            onClick={() => setShowStartModal(true)}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-sm font-medium transition-colors"
          >
            + Start New Run
          </button>
        </div>
        <ProviderStatus
          runs={runs}
          awaitingReview={awaitingReview}
          modeLabels={MODE_LABELS}
        />
      </div>

      {/* Full-width runs table */}
      <div className="pb-8">
        <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-800 text-gray-400 text-left">
              <th className="px-4 py-3 font-medium">Run ID</th>
              <th className="px-4 py-3 font-medium">Topic</th>
              <th className="px-4 py-3 font-medium">Mode</th>
              <th className="px-4 py-3 font-medium">Status</th>
              <th className="px-4 py-3 font-medium">Corrections</th>
              <th className="px-4 py-3 font-medium">Provider</th>
              <th className="px-4 py-3 font-medium">Duration</th>
              <th className="px-4 py-3 font-medium"></th>
            </tr>
          </thead>
          <tbody>
            {runs.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center text-gray-500">
                  No runs yet. Click "Start New Run" to begin.
                </td>
              </tr>
            ) : (
              runs.map((run) => {
                const needsReview = run.status === 'running' && hasPendingReview(run.id)
                const isDeleting = deletingId === run.id
                const isStopping = stoppingId === run.id
                return (
                  <tr
                    key={run.id}
                    onClick={() => handleRowClick(run.id)}
                    className={`border-b border-gray-800/50 cursor-pointer transition-colors ${
                      needsReview
                        ? 'bg-yellow-900/10 hover:bg-yellow-900/20'
                        : 'hover:bg-gray-800/30'
                    }`}
                  >
                    <td className="px-4 py-3 font-mono text-xs text-gray-400">
                      {run.id.slice(0, 8)}…
                    </td>
                    <td className="px-4 py-3">
                      <div className="font-medium">
                        {prompts.find((p) => p.id === run.prompt_id)?.topic ?? `#${run.prompt_id}`}
                      </div>
                      <div className="text-xs text-gray-500 truncate max-w-xs">{run.prompt_text}</div>
                      {run.status === 'running' && stageMap[run.id] && (
                        <StageProgressBar stages={stageMap[run.id]} />
                      )}
                      {needsReview && (
                        <span className="inline-block mt-1 text-xs text-yellow-400 font-medium">
                          ⚠ Awaiting your review — click row to open
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <span className="px-2 py-0.5 rounded text-xs font-medium bg-purple-900 text-purple-300">
                        {run.mode}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <StatusBadge status={run.status} />
                    </td>
                    <td className="px-4 py-3 text-center">{run.total_corrections}</td>
                    <td className="px-4 py-3">
                      {run.video_provider_choice === 'modal' ? (
                        <span className="px-2 py-0.5 rounded text-xs font-medium bg-purple-900 text-purple-300">
                          Modal AI
                        </span>
                      ) : run.video_provider_choice === 'ken_burns' ? (
                        <span className="px-2 py-0.5 rounded text-xs font-medium bg-gray-700 text-gray-300">
                          Ken Burns
                        </span>
                      ) : (
                        <span className="text-xs text-gray-500">—</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-xs text-gray-400">
                      {formatDuration(run.started_at, run.completed_at)}
                    </td>
                    <td className="px-4 py-3" onClick={(e) => e.stopPropagation()}>
                      <div className="flex gap-2 items-center">
                        {/* Stop — visible for active runs */}
                        {run.status === 'running' && (
                          <button
                            onClick={(e) => handleStopRun(run.id, e)}
                            disabled={isStopping}
                            title="Stop this run"
                            className="p-2 rounded bg-red-500 hover:bg-red-600 text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            {isStopping
                              ? <Loader2 size={16} className="animate-spin" />
                              : <Square size={16} />}
                          </button>
                        )}
                        {/* Resume — visible for stopped runs */}
                        {run.status === 'stopped' && (
                          <button
                            onClick={(e) => handleResumeRun(run.id, e)}
                            disabled={resumingId === run.id}
                            title="Resume from current stage"
                            className="p-2 rounded bg-green-500 hover:bg-green-600 text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            {resumingId === run.id
                              ? <Loader2 size={16} className="animate-spin" />
                              : <PlayCircle size={16} />}
                          </button>
                        )}
                        {/* Delete — always visible */}
                        <button
                          onClick={(e) => handleDeleteRun(run.id, e)}
                          disabled={isDeleting}
                          title="Delete this run"
                          className="p-2 rounded bg-gray-500 hover:bg-gray-600 text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {isDeleting
                            ? <Loader2 size={16} className="animate-spin" />
                            : <Trash2 size={16} />}
                        </button>
                      </div>
                    </td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>
      </div>

      {/* Inline review panel — shown when a pending_review stage is found */}
      {selectedRun && pendingStage && (
        <ReviewPanel
          attempt={pendingStage}
          runId={selectedRun.id}
          runText={`Run ${selectedRun.id.slice(0, 8)}… · ${prompts.find((p) => p.id === selectedRun.prompt_id)?.topic ?? selectedRun.prompt_text}`}
          videoProvider={selectedRun.video_provider_choice}
          onDecision={handleDecision}
          onClose={() => setSelectedRun(null)}
          onMediaError={(msg) => showToast(msg, 'warning')}
        />
      )}

      {/* Run detail modal — full stage breakdown with features and decisions */}
      {selectedRun && !pendingStage && (
        <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4" onClick={() => setSelectedRun(null)}>
          <div
            className="bg-gray-900 border border-gray-800 rounded-xl w-full max-w-2xl max-h-[90vh] flex flex-col shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-gray-800">
              <div>
                <h2 className="font-semibold">
                  {prompts.find((p) => p.id === selectedRun.prompt_id)?.topic ?? selectedRun.prompt_text}
                </h2>
                <p className="text-xs text-gray-400">
                  Run {selectedRun.id.slice(0, 8)}… · Mode {selectedRun.mode} ·{' '}
                  <StatusBadge status={selectedRun.status} />
                </p>
              </div>
              <button onClick={() => setSelectedRun(null)} className="text-gray-500 hover:text-white text-xl leading-none">×</button>
            </div>

            {/* Stage list */}
            <div className="overflow-y-auto flex-1 px-5 py-4 space-y-4">
              {/* Final video player — shown at top for completed runs */}
              {selectedRun.status === 'completed' && (
                <FinalVideoPlayer runId={selectedRun.id} />
              )}

              {selectedRun.stages.length === 0 && (
                <p className="text-gray-500 text-sm">No stages started yet.</p>
              )}
              {/* Group by stage name, show all attempts */}
              {['script', 'audio', 'visual', 'video'].map((stageName) => {
                const attempts = selectedRun.stages.filter((s) => s.stage_name === stageName)
                if (!attempts.length) return null
                return (
                  <div key={stageName} className="border border-gray-800 rounded-lg overflow-hidden">
                    <div className="flex items-center gap-2 px-4 py-2 bg-gray-800/60">
                      <span>{STAGE_ICONS[stageName]}</span>
                      <span className="font-medium capitalize text-sm">{stageName}</span>
                      <span className="text-xs text-gray-500">({attempts.length} attempt{attempts.length > 1 ? 's' : ''})</span>
                    </div>
                    {attempts.map((attempt) => {
                      const features = STAGE_FEATURES[stageName] ?? []
                      const hasFeatures = features.some(([k]) => (attempt as unknown as Record<string, unknown>)[k] !== null)
                      return (
                        <div key={attempt.id} className="px-4 py-3 border-t border-gray-800/50">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs text-gray-400">Attempt {attempt.attempt_number}</span>
                            <div className="flex items-center gap-2">
                              {attempt.human_decision && (
                                <span className={`text-xs px-2 py-0.5 rounded font-medium ${
                                  attempt.human_decision === 'accept'
                                    ? 'bg-green-900 text-green-300'
                                    : 'bg-red-900 text-red-300'
                                }`}>
                                  {attempt.human_decision === 'accept' ? '✓ Accepted' : '✗ Rejected'}
                                </span>
                              )}
                              <StatusBadge status={attempt.status} />
                            </div>
                          </div>
                          {attempt.reviewer_notes && (
                            <p className="text-xs text-gray-400 italic mb-2 bg-gray-800/50 rounded px-2 py-1">
                              "{attempt.reviewer_notes}"
                            </p>
                          )}
                          {attempt.visual_provider && (
                            <p className="text-xs text-gray-500 mb-2">Provider: <span className="text-gray-300">{attempt.visual_provider}</span></p>
                          )}
                          {attempt.is_fallback_video && (
                            <p className="text-xs text-orange-400 mb-2">Ken Burns (image fallback)</p>
                          )}
                          {hasFeatures && (
                            <details className="mt-1">
                              <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-300">
                                Feature scores
                              </summary>
                              <div className="mt-2">
                                <FeatureScores attempt={attempt} />
                              </div>
                            </details>
                          )}
                        </div>
                      )
                    })}
                  </div>
                )
              })}

              {/* Script preview for completed runs */}
              {(selectedRun.status === 'completed' || selectedRun.stages.some((s) => s.stage_name === 'script' && s.status === 'completed')) && (
                <div>
                  <h3 className="text-sm font-medium text-gray-300 mb-2">Script</h3>
                  <ScriptPreview runId={selectedRun.id} />
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="px-5 py-3 border-t border-gray-800 flex justify-between items-center">
              <span className="text-xs text-gray-500">
                {selectedRun.total_corrections} correction{selectedRun.total_corrections !== 1 ? 's' : ''} ·{' '}
                {selectedRun.video_provider_used ? `Provider: ${selectedRun.video_provider_used}` : 'No video provider'}
              </span>
              <button
                onClick={() => setSelectedRun(null)}
                className="px-4 py-1.5 text-sm bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {showStartModal && (
        <StartRunModal
          prompts={prompts}
          onClose={() => setShowStartModal(false)}
          onStart={(promptId, mode, vpc) => handleStartRun(promptId, mode, vpc)}
        />
      )}
    </div>
  )
}
