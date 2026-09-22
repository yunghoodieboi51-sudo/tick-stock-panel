import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Play, RefreshCw } from 'lucide-react'
import { useSearchParams } from 'react-router-dom'
import { api, type DailyScanCandidate } from '@/lib/api'
import { QK } from '@/lib/queryKeys'
import { storage } from '@/lib/storage'
import { useDataStatus } from '@/lib/useSharedQueries'
import { PageHeader } from '@/components/PageHeader'
import { CandidateDecisionDetail } from '@/components/daily-scan/CandidateDecisionDetail'
import { DEFAULT_DAILY_SCAN_SIZING, SizingSettings, type DailyScanSizing } from '@/components/daily-scan/SizingSettings'

const score = (value: number | null | undefined) => value == null ? '—' : value.toFixed(2)

export function TradingPlan() {
  const [searchParams] = useSearchParams()
  const queryClient = useQueryClient()
  const [runId, setRunId] = useState(() => searchParams.get('run_id') ?? '')
  const [selected, setSelected] = useState<DailyScanCandidate | null>(null)
  const [sizing, setSizing] = useState(() => storage.dailyScanSizing.get(DEFAULT_DAILY_SCAN_SIZING))
  const dataStatus = useDataStatus({ staleTime: 60_000 })
  const status = useQuery({ queryKey: QK.dailyScanStatus, queryFn: api.dailyScanStatus, refetchInterval: query => query.state.data?.state === 'RUNNING' ? 1500 : false })
  const runs = useQuery({ queryKey: QK.dailyScanRuns(), queryFn: () => api.dailyScanRuns(), refetchInterval: status.data?.state === 'RUNNING' ? 1500 : false })
  const run = useQuery({ queryKey: QK.dailyScanRun(runId), queryFn: () => api.dailyScanRun(runId), enabled: Boolean(runId) })
  const start = useMutation({ mutationFn: () => api.dailyScanStart(sizing), onSuccess: manifest => { setRunId(manifest.run_id); setSelected(null); queryClient.invalidateQueries({ queryKey: QK.dailyScan }) } })

  useEffect(() => { if (!runId && runs.data?.items[0]?.run_id) setRunId(runs.data.items[0].run_id) }, [runId, runs.data])
  useEffect(() => { if (!selected && run.data?.result?.candidates[0]) setSelected(run.data.result.candidates[0]) }, [run.data, selected])
  useEffect(() => {
    const symbol = searchParams.get('symbol')
    if (!symbol || !run.data?.result) return
    setSelected([...run.data.result.candidates, ...run.data.result.rejected].find(item => item.symbol === symbol) ?? null)
  }, [run.data, searchParams])

  const updateSizing = (next: DailyScanSizing) => {
    setSizing(next)
    storage.dailyScanSizing.set(next)
  }
  const candidates = run.data?.result?.candidates ?? []

  return <div className="min-h-full bg-base">
    <PageHeader title="交易计划" subtitle="V4 趋势策略 · 研究阶段 / 尚未完成历史有效性验证" right={<button type="button" disabled={status.data?.state === 'RUNNING' || start.isPending} onClick={() => start.mutate()} className="inline-flex items-center gap-1.5 rounded-btn bg-accent px-3 py-2 text-xs font-medium text-white disabled:opacity-50"><Play className="h-3.5 w-3.5" />运行 V4 扫描</button>} />
    <div className="grid gap-3 p-4 lg:grid-cols-[320px_minmax(0,1fr)]">
      <section className="rounded-lg border border-border bg-surface p-3">
        <SizingSettings sizing={sizing} onChange={updateSizing} />
        <label className="mt-4 block text-xs text-muted">历史运行</label>
        <select value={runId} onChange={event => { setRunId(event.target.value); setSelected(null) }} className="mt-2 w-full rounded border border-border bg-base p-2 text-xs"><option value="">选择运行记录</option>{runs.data?.items.map(item => <option key={item.run_id} value={item.run_id}>{item.completed_at ?? item.started_at} · {item.state} · {item.candidate_count ?? 0} 候选</option>)}</select>
        {status.data?.state === 'RUNNING' && <p className="mt-3 flex items-center gap-2 text-xs text-accent"><RefreshCw className="h-3 w-3 animate-spin" />{status.data.latest?.stage ?? 'running'}</p>}
        {run.data?.manifest.state === 'FAILED' && <p className="mt-3 text-xs text-danger">扫描失败：{run.data.manifest.error ?? 'unknown error'}</p>}
        {run.data?.manifest.state === 'COMPLETED' && candidates.length === 0 && <p className="mt-3 text-xs text-muted">当前规则下没有候选股票。</p>}
        <div className="mt-3 space-y-1">{candidates.map(item => <button key={item.symbol} type="button" onClick={() => setSelected(item)} className={`w-full rounded p-2 text-left text-xs hover:bg-elevated ${selected?.symbol === item.symbol ? 'bg-elevated' : ''}`}><span className="font-mono">#{item.rank} {item.symbol}</span><span className="float-right">{score(item.final_score)}</span><div className="mt-1 text-muted">{item.name} · {item.entry_pattern_primary ?? '—'}</div></button>)}</div>
      </section>
      <section className="rounded-lg border border-border bg-surface p-4">{!selected ? <p className="text-sm text-muted">请选择已完成的扫描结果，或运行新的 V4 扫描。</p> : <CandidateDecisionDetail candidate={selected} completedAt={run.data?.manifest.completed_at} dataDate={run.data?.result?.trade_date} currentDataDate={dataStatus.data?.enriched?.latest_date} policy={run.data?.result?.fundamental_veto_policy} />}</section>
    </div>
  </div>
}
