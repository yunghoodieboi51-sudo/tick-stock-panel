import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api, type DailyScanCandidate } from '@/lib/api'
import { QK } from '@/lib/queryKeys'

const price = (value: number | null | undefined) => value == null ? '—' : value.toFixed(2)

export function V4DailyScanWorkspace({ onBack }: { onBack: () => void }) {
  const [runId, setRunId] = useState('')
  const [tab, setTab] = useState<'candidates' | 'rejected'>('candidates')
  const [search, setSearch] = useState('')
  const [sort, setSort] = useState<'rank' | 'score'>('rank')
  const [pattern, setPattern] = useState('')
  const [fundamental, setFundamental] = useState('')
  const runs = useQuery({ queryKey: QK.dailyScanRuns(), queryFn: () => api.dailyScanRuns() })
  const run = useQuery({ queryKey: QK.dailyScanRun(runId), queryFn: () => api.dailyScanRun(runId), enabled: Boolean(runId) })

  useEffect(() => {
    if (runId || !runs.data) return
    const latest = runs.data.items.find(item => item.state === 'COMPLETED') ?? runs.data.items[0]
    if (latest) setRunId(latest.run_id)
  }, [runId, runs.data])

  const rows = tab === 'candidates' ? run.data?.result?.candidates ?? [] : run.data?.result?.rejected ?? []
  const patterns = useMemo(() => [...new Set(rows.map(row => row.entry_pattern_primary).filter(Boolean))] as string[], [rows])
  const items = useMemo(() => rows.filter(row => {
    const query = search.trim().toLowerCase()
    return (!query || row.symbol.toLowerCase().includes(query) || row.name?.toLowerCase().includes(query))
      && (!pattern || row.entry_pattern_primary === pattern)
      && (!fundamental || row.fundamental.status === fundamental)
  }).sort((left, right) => sort === 'rank'
    ? (left.rank ?? Number.MAX_SAFE_INTEGER) - (right.rank ?? Number.MAX_SAFE_INTEGER)
    : (right.final_score ?? -Infinity) - (left.final_score ?? -Infinity)), [rows, search, pattern, fundamental, sort])

  const state = run.data?.manifest.state
  return <div className="min-h-full bg-base p-4">
    <div className="mb-4 flex flex-wrap items-start justify-between gap-3"><div><h1 className="text-lg font-semibold">V4 Daily Scan</h1><p className="mt-1 text-xs text-muted">V4 Trend Strategy · 研究阶段 / 尚未完成历史有效性验证</p></div><button type="button" onClick={onBack} className="rounded-btn border border-border px-3 py-1.5 text-xs text-secondary">返回策略列表</button></div>
    <div className="rounded-card border border-border bg-surface p-3"><div className="flex flex-wrap gap-2"><select value={runId} onChange={event => { setRunId(event.target.value); setSearch(''); setPattern(''); setFundamental('') }} className="rounded border border-border bg-base p-2 text-xs"><option value="">选择运行记录</option>{runs.data?.items.map(item => <option key={item.run_id} value={item.run_id}>{item.completed_at ?? item.started_at} · {item.state} · {item.candidate_count ?? 0} 候选</option>)}</select>{state === 'RUNNING' && <span className="p-2 text-xs text-accent">正在扫描：{run.data?.manifest.stage}</span>}{state === 'FAILED' && <span className="p-2 text-xs text-danger">扫描失败：{run.data?.manifest.error ?? 'unknown error'}</span>}</div>
      {state === 'COMPLETED' && <><div className="mt-3 flex gap-2 border-b border-border"><button type="button" onClick={() => setTab('candidates')} className={`px-2 py-2 text-xs ${tab === 'candidates' ? 'border-b-2 border-accent text-accent' : 'text-muted'}`}>Candidates ({run.data?.result?.candidates.length ?? 0})</button><button type="button" onClick={() => setTab('rejected')} className={`px-2 py-2 text-xs ${tab === 'rejected' ? 'border-b-2 border-accent text-accent' : 'text-muted'}`}>Rejected / Vetoed ({run.data?.result?.rejected.length ?? 0})</button></div><div className="mt-3 flex flex-wrap gap-2"><input value={search} onChange={event => setSearch(event.target.value)} placeholder="搜索代码或名称" className="rounded border border-border bg-base p-2 text-xs" /><select value={sort} onChange={event => setSort(event.target.value as 'rank' | 'score')} className="rounded border border-border bg-base p-2 text-xs"><option value="rank">Rank</option><option value="score">Score</option></select><select value={pattern} onChange={event => setPattern(event.target.value)} className="rounded border border-border bg-base p-2 text-xs"><option value="">所有形态</option>{patterns.map(item => <option key={item} value={item}>{item}</option>)}</select><select value={fundamental} onChange={event => setFundamental(event.target.value)} className="rounded border border-border bg-base p-2 text-xs"><option value="">所有基本面状态</option><option value="PASS">PASS</option><option value="VETO">VETO</option><option value="UNKNOWN">UNKNOWN</option></select></div>
      {items.length === 0 ? <p className="py-8 text-center text-sm text-muted">{tab === 'candidates' ? '当前规则下没有候选股票。' : '本次扫描没有保存被基本面否决的技术候选。'}</p> : <div className="mt-3 overflow-x-auto"><table className="w-full text-left text-xs"><thead className="border-b border-border text-muted"><tr>{tab === 'candidates' && <th className="p-2">Rank</th>}<th className="p-2">Symbol</th><th className="p-2">Name</th><th className="p-2">{tab === 'candidates' ? 'RAW Plan Ref' : 'Technical Score'}</th><th className="p-2">Score</th><th className="p-2">Pattern</th><th className="p-2">Fundamental</th>{tab === 'rejected' && <th className="p-2">Veto Reason</th>}<th className="p-2">Plan</th></tr></thead><tbody>{items.map(item => <Row key={`${item.symbol}-${item.rank ?? 'veto'}`} item={item} rejected={tab === 'rejected'} />)}</tbody></table></div>}</>}</div>
  </div>
}

function Row({ item, rejected }: { item: DailyScanCandidate; rejected: boolean }) {
  const missingRaw = !rejected && item.execution_reference_price == null
  return <tr className="border-b border-border/60"><>{!rejected && <td className="p-2 font-mono">{item.rank ?? '—'}</td>}</><td className="p-2 font-mono">{item.symbol}</td><td className="p-2">{item.name ?? '—'}</td><td className="p-2 font-mono">{rejected ? price(item.final_score) : <>{price(item.execution_reference_price)}{missingRaw && <span className="ml-1 text-warning">RAW unavailable</span>}</>}</td><td className="p-2 font-mono">{price(item.final_score)}</td><td className="p-2">{item.entry_pattern_primary ?? '—'}</td><td className="p-2">{item.fundamental.status}</td>{rejected && <td className="p-2 text-warning">{item.fundamental.reason_codes.join(', ') || '—'}</td>}<td className="p-2"><Link to={`/trading-plan?run_id=${encodeURIComponent(item.run_id)}&symbol=${encodeURIComponent(item.symbol)}`} className="text-accent hover:underline">Plan</Link></td></tr>
}
