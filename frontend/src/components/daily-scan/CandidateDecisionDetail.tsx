import { StockDailyKChart } from '@/components/StockDailyKChart'
import type { DailyScanCandidate } from '@/lib/api'
import { FundamentalVetoExplanation } from './FundamentalVetoExplanation'

const money = (value: number | null | undefined) => value == null ? '—' : value.toFixed(2)
const percent = (value: number | null | undefined) => value == null ? '—' : `${(value * 100).toFixed(1)}%`

function chartRange(asOf: string | null | undefined) {
  if (!asOf || !/^\d{4}-\d{2}-\d{2}/.test(asOf)) return undefined
  const end = asOf.slice(0, 10)
  const startDate = new Date(`${end}T00:00:00Z`)
  startDate.setUTCMonth(startDate.getUTCMonth() - 9)
  return { start: startDate.toISOString().slice(0, 10), end }
}

export function CandidateDecisionDetail({
  candidate,
  completedAt,
  dataDate,
  currentDataDate,
  policy,
}: {
  candidate: DailyScanCandidate
  completedAt?: string | null
  dataDate?: string | null
  currentDataDate?: string | null
  policy?: Record<string, unknown> | null
}) {
  const historical = Boolean(currentDataDate && dataDate && currentDataDate !== dataDate)
  const range = chartRange(dataDate ?? candidate.trade_date)
  return <div className="space-y-4">
    <div className="flex flex-wrap items-start justify-between gap-2">
      <div><h2 className="text-base font-semibold">{candidate.name || candidate.symbol}</h2><p className="font-mono text-xs text-muted">{candidate.symbol} · {candidate.entry_pattern_primary ?? '—'} · Score {money(candidate.final_score)}</p></div>
      <span className="rounded bg-warning/10 px-2 py-1 text-[10px] text-warning">Research / Unvalidated</span>
    </div>
    <p className="text-xs text-muted">策略数据日期：{dataDate ?? candidate.trade_date ?? '—'} · 扫描完成：{completedAt ?? '—'}{historical ? ' · 当前查看历史运行，不代表最新数据' : ''}</p>
    <section className="rounded border border-border bg-base p-3"><h3 className="text-xs font-semibold">技术诊断</h3><div className="mt-2 grid grid-cols-2 gap-1 text-xs sm:grid-cols-3">{Object.entries(candidate.score_breakdown).map(([key, item]) => <span key={key} className="rounded bg-surface px-2 py-1">{key}: {money(item)}</span>)}</div><p className="mt-2 text-xs text-secondary">主形态：{candidate.entry_pattern_primary ?? '—'} · 匹配：{candidate.entry_patterns_matched.join(', ') || '—'}</p></section>
    <section className="rounded border border-border bg-base p-3"><h3 className="text-xs font-semibold">价格图表</h3><p className="mt-1 text-[11px] text-muted">技术展示价格基准：FORWARD_ADJUSTED（含 MA20 / MA60）。图表截至本次策略数据日；RAW 计划参考价不叠加到此图，避免混用价格口径。</p><StockDailyKChart symbol={candidate.symbol} dateRange={range} height={360} showIndicatorControls={false} showLimitMarkers={false} visibleBars={150} /></section>
    <section className="rounded border border-border bg-base p-3"><h3 className="text-xs font-semibold">价格口径与交易计划</h3><p className="mt-1 text-[11px] text-muted">信号/展示：{candidate.signal_price_basis}；执行计划：{candidate.execution_price_basis ?? 'RAW unavailable'}。RAW 缺失时不会以调整价替代。</p><div className="mt-2 grid gap-2 text-xs sm:grid-cols-2 lg:grid-cols-4">{[['计划参考价', candidate.execution_reference_price], ['买入区间下限', candidate.plan.buy_zone_low], ['追价上限', candidate.plan.chase_limit_price], ['计划止损', candidate.plan.stop_loss_price]].map(([label, item]) => <div key={String(label)} className="rounded bg-surface p-2"><div className="text-[10px] text-muted">{label}</div><div className="mt-1 font-mono">{money(item as number | null)}</div></div>)}</div><p className="mt-2 text-xs text-secondary">止损 {percent(candidate.plan.stop_loss_pct)} · 移动止损 {percent(candidate.plan.trailing_stop_pct)} · 最长 {candidate.plan.max_hold_days ?? '—'} 日 · 预计 {candidate.plan.estimated_shares ?? '—'} 股 / {money(candidate.plan.estimated_order_value)}</p><ul className="mt-2 space-y-1 text-xs text-warning">{[...candidate.warnings, ...(candidate.plan.warnings ?? []), ...candidate.plan.invalidation_conditions].map(item => <li key={item}>• {item}</li>)}</ul></section>
    <FundamentalVetoExplanation candidate={candidate} policy={policy} />
  </div>
}
