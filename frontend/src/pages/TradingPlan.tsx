import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Play, RefreshCw } from 'lucide-react'
import { useSearchParams } from 'react-router-dom'
import { api, type DailyScanCandidate } from '@/lib/api'
import { QK } from '@/lib/queryKeys'
import { storage } from '@/lib/storage'
import { PageHeader } from '@/components/PageHeader'

const DEFAULT_SIZING = { account_size: 100000, max_position_pct: 0.2, max_candidates: 10, risk_per_trade_pct: 0.01 }
const money = (value: number | null | undefined) => value == null ? '—' : value.toFixed(2)

export function TradingPlan() {
  const [searchParams] = useSearchParams()
  const queryClient = useQueryClient()
  const [runId, setRunId] = useState(() => searchParams.get('run_id') ?? '')
  const [selected, setSelected] = useState<DailyScanCandidate | null>(null)
  const [sizing, setSizing] = useState(() => storage.dailyScanSizing.get(DEFAULT_SIZING))
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
  const candidates = run.data?.result?.candidates ?? []
  const updateSizing = (key: keyof typeof DEFAULT_SIZING, value: number) => {
    const next = { ...sizing, [key]: value }
    setSizing(next)
    storage.dailyScanSizing.set(next)
  }

  return <div className="min-h-full bg-base">
    <PageHeader title="交易计划" subtitle="V4 趋势策略 · 研究阶段 / 尚未完成历史有效性验证" right={<button type="button" disabled={status.data?.state === 'RUNNING' || start.isPending} onClick={() => start.mutate()} className="inline-flex items-center gap-1.5 rounded-btn bg-accent px-3 py-2 text-xs font-medium text-white disabled:opacity-50"><Play className="h-3.5 w-3.5" />运行 V4 扫描</button>} />
    <div className="grid gap-3 p-4 lg:grid-cols-[320px_minmax(0,1fr)]">
      <section className="rounded-lg border border-border bg-surface p-3">
        <div className="grid grid-cols-2 gap-2 text-xs">{([['account_size', '账户规模'], ['max_position_pct', '单股上限'], ['risk_per_trade_pct', '单笔风险'], ['max_candidates', '候选上限']] as const).map(([key, label]) => <label key={key} className="text-muted">{label}<input type="number" min="0" step={key === 'max_candidates' ? 1 : 0.01} value={sizing[key]} onChange={event => updateSizing(key, Number(event.target.value))} className="mt-1 w-full rounded border border-border bg-base p-1.5 text-foreground" /></label>)}</div>
        <label className="mt-4 block text-xs text-muted">历史运行</label><select value={runId} onChange={event => { setRunId(event.target.value); setSelected(null) }} className="mt-2 w-full rounded border border-border bg-base p-2 text-xs"><option value="">选择运行记录</option>{runs.data?.items.map(item => <option key={item.run_id} value={item.run_id}>{item.started_at} · {item.state} · {item.candidate_count ?? 0} 候选</option>)}</select>
        {status.data?.state === 'RUNNING' && <p className="mt-3 flex items-center gap-2 text-xs text-accent"><RefreshCw className="h-3 w-3 animate-spin" />{status.data.latest?.stage ?? 'running'}</p>}
        {run.data?.manifest.state === 'FAILED' && <p className="mt-3 text-xs text-danger">扫描失败：{run.data.manifest.error ?? 'unknown error'}</p>}
        {run.data?.manifest.state === 'COMPLETED' && candidates.length === 0 && <p className="mt-3 text-xs text-muted">当前规则下没有候选股票。</p>}
        <div className="mt-3 space-y-1">{candidates.map(item => <button key={item.symbol} type="button" onClick={() => setSelected(item)} className={`w-full rounded p-2 text-left text-xs hover:bg-elevated ${selected?.symbol === item.symbol ? 'bg-elevated' : ''}`}><span className="font-mono">#{item.rank} {item.symbol}</span><span className="float-right">{money(item.final_score)}</span><div className="mt-1 text-muted">{item.name} · {item.entry_pattern_primary ?? '—'}</div></button>)}</div>
      </section>
      <section className="rounded-lg border border-border bg-surface p-4">
        {!selected ? <p className="text-sm text-muted">请选择已完成的扫描结果，或运行新的 V4 扫描。</p> : <>
          <div className="flex items-start justify-between"><div><h2 className="text-base font-semibold">{selected.name || selected.symbol}</h2><p className="font-mono text-xs text-muted">{selected.symbol} · {selected.entry_pattern_primary ?? '—'} · Score {money(selected.final_score)}</p></div><span className="rounded bg-warning/10 px-2 py-1 text-[10px] text-warning">Research / Unvalidated</span></div>
          <p className="mt-2 text-xs text-muted">信号价格基准：{selected.signal_price_basis}；计划价格基准：{selected.execution_price_basis ?? 'unavailable'}</p>
          <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">{[['计划参考价', selected.execution_reference_price], ['买入区间下限', selected.plan.buy_zone_low], ['追价上限', selected.plan.chase_limit_price], ['计划止损', selected.plan.stop_loss_price]].map(([label, value]) => <div key={String(label)} className="rounded border border-border p-2"><div className="text-[10px] text-muted">{label}</div><div className="mt-1 font-mono text-sm">{money(value as number | null)}</div></div>)}</div>
          <div className="mt-4 grid gap-4 lg:grid-cols-2"><div><h3 className="text-xs font-semibold">入选原因</h3><ul className="mt-2 space-y-1 text-xs text-secondary">{selected.why_selected.map(item => <li key={item}>• {item}</li>)}</ul><h3 className="mt-4 text-xs font-semibold">评分拆分</h3><div className="mt-2 grid grid-cols-2 gap-1 text-xs">{Object.entries(selected.score_breakdown).map(([key, value]) => <span key={key} className="rounded bg-base px-2 py-1">{key}: {money(value)}</span>)}</div></div><div><h3 className="text-xs font-semibold">风控与仓位</h3><p className="mt-2 text-xs text-secondary">止损 {selected.plan.stop_loss_pct == null ? '—' : `${(selected.plan.stop_loss_pct * 100).toFixed(1)}%`} · 移动止损 {selected.plan.trailing_stop_pct == null ? '—' : `${(selected.plan.trailing_stop_pct * 100).toFixed(1)}%`} · 最长 {selected.plan.max_hold_days ?? '—'} 日</p><p className="mt-2 text-xs text-secondary">预估 {selected.plan.estimated_shares ?? '—'} 股 / {money(selected.plan.estimated_order_value)}；{selected.plan.lot_rule_status ?? '—'}</p><p className="mt-3 text-xs">基本面：<b>{selected.fundamental.status}</b> {selected.fundamental.reason_codes.join(', ')}</p><ul className="mt-3 space-y-1 text-xs text-warning">{[...selected.warnings, ...(selected.plan.warnings ?? []), ...selected.plan.invalidation_conditions].map(item => <li key={item}>• {item}</li>)}</ul></div></div>
        </>}
      </section>
    </div>
  </div>
}
