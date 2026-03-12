import React, { useState, useEffect, useCallback } from 'react'

// ── Types ──────────────────────────────────────────────────────────────────

interface PendingReview {
  id: string
  run_id: string
  stage_name: string
  attempt_number: number
  output_path: string | null
  prompt_used: string | null
  created_at: string | null
}

interface ScriptScene {
  scene_number: number
  narration: string
  visual_prompt: string
  estimated_start_s?: number
  estimated_duration_s?: number
}

// ── Helper: relative API URL ───────────────────────────────────────────────

const API = '/review'

// ── Scene preview components ───────────────────────────────────────────────

function ScriptPreview({ outputPath }: { outputPath: string }) {
  const [scenes, setScenes] = useState<ScriptScene[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // Fetch the script JSON file via a dedicated endpoint
    fetch(`/api/file?path=${encodeURIComponent(outputPath)}`)
      .then((r) => r.json())
      .then((data) => setScenes(data?.scenes ?? []))
      .catch(() => setError('Could not load script file.'))
  }, [outputPath])

  if (error) return <div className="text-red-400 text-sm">{error}</div>

  return (
    <div className="space-y-3">
      {scenes.map((scene) => (
        <div
          key={scene.scene_number}
          className="bg-gray-800 rounded-lg p-3 border border-gray-700"
        >
          <div className="flex items-center gap-2 mb-2">
            <span className="bg-blue-900 text-blue-300 text-xs px-2 py-0.5 rounded font-medium">
              Scene {scene.scene_number}
            </span>
            {scene.estimated_duration_s && (
              <span className="text-xs text-gray-500">
                ~{scene.estimated_duration_s.toFixed(1)}s
              </span>
            )}
          </div>
          <p className="text-sm text-white mb-1">{scene.narration}</p>
          <p className="text-xs text-gray-400 italic">{scene.visual_prompt}</p>
        </div>
      ))}
    </div>
  )
}

function AudioPreview({ outputPath }: { outputPath: string }) {
  const audioSrc = `/api/media?path=${encodeURIComponent(outputPath)}`
  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <p className="text-sm text-gray-400 mb-3">Generated narration audio:</p>
      <audio controls className="w-full" src={audioSrc}>
        Your browser does not support the audio element.
      </audio>
      <p className="text-xs text-gray-600 mt-2">File: {outputPath.split('/').pop()}</p>
    </div>
  )
}

function VideoPreview({ outputPath, label }: { outputPath: string; label: string }) {
  const videoSrc = `/api/media?path=${encodeURIComponent(outputPath)}`
  return (
    <div className="bg-gray-800 rounded-lg p-3 border border-gray-700">
      <p className="text-xs text-gray-400 mb-2">{label}</p>
      <video controls className="w-full rounded" src={videoSrc}>
        Your browser does not support the video element.
      </video>
    </div>
  )
}

function VisualPreview({ outputPath }: { outputPath: string }) {
  // outputPath is the scenes dir; show individual clips
  const scenesDir = outputPath.replace('/scenes', '')
  return (
    <div>
      <p className="text-sm text-gray-400 mb-3">
        Individual scene clips (8 scenes × 6 seconds):
      </p>
      <div className="grid grid-cols-2 gap-3">
        {Array.from({ length: 8 }, (_, i) => {
          const sceneNum = String(i + 1).padStart(2, '0')
          const path = `${outputPath}/scene_${sceneNum}.mp4`
          return (
            <VideoPreview
              key={i}
              outputPath={path}
              label={`Scene ${sceneNum}`}
            />
          )
        })}
      </div>
    </div>
  )
}

// ── Stage output preview router ────────────────────────────────────────────

function StageOutputPreview({
  stageName,
  outputPath,
}: {
  stageName: string
  outputPath: string
}) {
  if (stageName === 'script') return <ScriptPreview outputPath={outputPath} />
  if (stageName === 'audio') return <AudioPreview outputPath={outputPath} />
  if (stageName === 'visual') return <VisualPreview outputPath={outputPath} />
  if (stageName === 'video') return <VideoPreview outputPath={outputPath} label="Final assembled video" />
  return <p className="text-gray-400 text-sm">No preview available.</p>
}

// ── Main ReviewStage page ──────────────────────────────────────────────────

export default function ReviewStage() {
  const [pending, setPending] = useState<PendingReview | null>(null)
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [notes, setNotes] = useState('')
  const [lastResult, setLastResult] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const fetchPending = useCallback(async () => {
    try {
      const res = await fetch(`${API}/pending`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setPending(data)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchPending()
    const interval = setInterval(fetchPending, 5000)
    return () => clearInterval(interval)
  }, [fetchPending])

  const handleSubmit = async (decision: 'accept' | 'reject') => {
    if (!pending) return
    setSubmitting(true)
    setError(null)
    try {
      const res = await fetch(`${API}/${pending.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decision, notes }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setLastResult(
        `${decision === 'accept' ? 'Accepted' : 'Rejected'}: ${pending.stage_name} (attempt ${pending.attempt_number})`,
      )
      setNotes('')
      setPending(null)
      setTimeout(fetchPending, 1000)
    } catch (e) {
      setError(String(e))
    } finally {
      setSubmitting(false)
    }
  }

  const STAGE_ICONS: Record<string, string> = {
    script: '📝',
    audio: '🎵',
    visual: '🎬',
    video: '📹',
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="mb-6">
        <h1 className="text-2xl font-bold">Human Review Interface</h1>
        <p className="text-gray-400 text-sm mt-1">Mode B: Review and accept or reject each stage output.</p>
      </div>

      {error && (
        <div className="bg-red-900/40 border border-red-700 rounded-lg p-3 mb-4 text-sm text-red-300">
          {error}
        </div>
      )}

      {lastResult && (
        <div className="bg-green-900/40 border border-green-700 rounded-lg p-3 mb-4 text-sm text-green-300">
          {lastResult}
        </div>
      )}

      {loading ? (
        <div className="text-center py-16 text-gray-500">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-3" />
          Loading review queue...
        </div>
      ) : !pending ? (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-12 text-center">
          <div className="text-5xl mb-4">✓</div>
          <h2 className="text-xl font-semibold mb-2">No Pending Reviews</h2>
          <p className="text-gray-400 text-sm">
            All stages are up to date. Start a Mode B run to generate review tasks.
          </p>
          <p className="text-gray-600 text-xs mt-4">Auto-refreshing every 5 seconds...</p>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Stage header */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="flex items-center gap-3 mb-2">
                  <span className="text-2xl">
                    {STAGE_ICONS[pending.stage_name] ?? '📋'}
                  </span>
                  <div>
                    <h2 className="text-lg font-semibold capitalize">
                      Stage: {pending.stage_name}
                    </h2>
                    <p className="text-xs text-gray-400">
                      Attempt {pending.attempt_number} · Run {pending.run_id.slice(0, 8)}…
                    </p>
                  </div>
                </div>
                {pending.prompt_used && (
                  <p className="text-sm text-gray-300 mt-2 bg-gray-800 rounded-lg px-3 py-2">
                    Prompt: {pending.prompt_used}
                  </p>
                )}
              </div>
              <div className="text-xs text-gray-500 whitespace-nowrap">
                {pending.created_at
                  ? new Date(pending.created_at).toLocaleTimeString()
                  : '—'}
              </div>
            </div>
          </div>

          {/* Output preview */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h3 className="font-medium mb-4 text-gray-300">Stage Output Preview</h3>
            {pending.output_path ? (
              <StageOutputPreview
                stageName={pending.stage_name}
                outputPath={pending.output_path}
              />
            ) : (
              <p className="text-gray-500 text-sm">No output file available.</p>
            )}
          </div>

          {/* Review controls */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h3 className="font-medium mb-3 text-gray-300">Your Decision</h3>

            <div className="mb-4">
              <label className="block text-sm text-gray-400 mb-1">
                Reviewer Notes (optional)
              </label>
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Describe any issues observed..."
                rows={3}
                className="w-full bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:border-blue-500"
              />
            </div>

            <div className="flex gap-3">
              <button
                onClick={() => handleSubmit('reject')}
                disabled={submitting}
                className="flex-1 py-3 bg-red-700 hover:bg-red-600 rounded-lg font-medium disabled:opacity-50 transition-colors"
              >
                Reject
              </button>
              <button
                onClick={() => handleSubmit('accept')}
                disabled={submitting}
                className="flex-1 py-3 bg-green-700 hover:bg-green-600 rounded-lg font-medium disabled:opacity-50 transition-colors"
              >
                Accept
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
