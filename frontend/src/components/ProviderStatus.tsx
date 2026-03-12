import { useState, useEffect, useCallback } from 'react'

interface ProviderEntry {
  allocated: number
  used: number
  exhausted: boolean
}

type BudgetData = Record<string, ProviderEntry>

interface RunSummarySlim {
  mode: string
  status: string
  video_provider_choice: string | null
  total_corrections: number
}

interface ProviderStatusProps {
  runs?: RunSummarySlim[]
  awaitingReview?: number
  modeLabels?: Record<string, string>
}

const PROVIDER_LABELS: Record<string, string> = {
  modal: 'Modal.com',
  ken_burns: 'Ken Burns',
}

export default function ProviderStatus({ runs = [], awaitingReview = 0, modeLabels = {} }: ProviderStatusProps) {
  const [budget, setBudget] = useState<BudgetData>({})
  const [resetLoading, setResetLoading] = useState<string | null>(null)

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch('/providers/status')
      if (res.ok) setBudget(await res.json())
    } catch { /* non-fatal */ }
  }, [])

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 30000)
    return () => clearInterval(interval)
  }, [fetchStatus])

  const handleReset = async (name: string) => {
    setResetLoading(name)
    try {
      const res = await fetch(`/providers/${name}/reset`, { method: 'POST' })
      if (!res.ok) throw new Error()
      await fetchStatus()
    } catch { /* non-fatal */ } finally { setResetLoading(null) }
  }

  const pStats = (name: string) => {
    const pr = runs.filter((r) => r.video_provider_choice === name)
    return { total: pr.length, done: pr.filter((r) => r.status === 'completed').length }
  }

  const mStats = (m: string) => {
    const mr = runs.filter((r) => r.mode === m)
    return { total: mr.length, done: mr.filter((r) => r.status === 'completed').length }
  }

  const totalCorrections = runs.reduce((a, r) => a + r.total_corrections, 0)
  const totalRuns = runs.length
  const activRuns = runs.filter((r) => r.status === 'running').length
  const doneRuns = runs.filter((r) => r.status === 'completed').length

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl px-4 py-2.5 flex items-center gap-0 text-xs">

      {/* Overall summary */}
      <div className="flex items-center gap-3 pr-4 border-r border-gray-700 flex-shrink-0">
        <span className="text-gray-400 font-medium">Runs</span>
        <span className="text-gray-300 font-semibold">{totalRuns}</span>
        <span className="text-blue-400">{activRuns} active</span>
        {awaitingReview > 0 && <span className="text-yellow-400">{awaitingReview} awaiting</span>}
        <span className="text-green-400">{doneRuns} done</span>
        <button onClick={fetchStatus} className="text-gray-600 hover:text-gray-400 ml-1">↻</button>
      </div>

      {/* Providers */}
      {(['modal', 'ken_burns'] as const).map((name) => {
        const entry = budget[name] ?? { exhausted: false }
        const s = pStats(name)
        const dot = entry.exhausted ? 'bg-red-500' : 'bg-green-400'
        return (
          <div key={name} className="flex items-center gap-2.5 px-4 border-r border-gray-700 flex-shrink-0">
            <span className={`w-1.5 h-1.5 rounded-full ${dot}`} />
            <span className="text-gray-300 font-medium">{PROVIDER_LABELS[name]}</span>
            {s.total > 0 && (
              <span className="text-gray-500">
                <span className="text-green-400">{s.done}</span>/{s.total}
              </span>
            )}
            {entry.exhausted ? (
              <span className="text-red-400 font-medium">
                Unavail
                <button onClick={() => handleReset(name)} disabled={resetLoading === name} className="ml-1 hover:text-white">↺</button>
              </span>
            ) : (
              <span className="text-green-400 font-medium">Ready</span>
            )}
          </div>
        )
      })}

      {/* Mode breakdown */}
      {(['A', 'B', 'C'] as const).map((m) => {
        const s = mStats(m)
        return (
          <div key={m} className="flex items-center gap-2 px-4 border-r border-gray-700 flex-shrink-0">
            <span className="text-gray-500">{modeLabels[m] ?? `Mode ${m}`}</span>
            <span>
              <span className="text-green-400 font-semibold">{s.done}</span>
              <span className="text-gray-600">/{s.total}</span>
            </span>
          </div>
        )
      })}

      {/* Corrections */}
      <div className="flex items-center gap-2 pl-4 flex-shrink-0">
        <span className="text-gray-500">Corrections</span>
        <span className="text-white font-semibold">{totalCorrections}</span>
      </div>
    </div>
  )
}
