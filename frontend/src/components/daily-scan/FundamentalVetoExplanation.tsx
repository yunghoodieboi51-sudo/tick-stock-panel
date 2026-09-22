import type { DailyScanCandidate, DailyScanFundamentalMetrics } from '@/lib/api'

type Policy = Record<string, unknown> | null | undefined

const DEFINITIONS: Record<string, { label: string; metric?: keyof DailyScanFundamentalMetrics; threshold?: string; direction: 'min' | 'max' | 'missing' }> = {
  NEGATIVE_ROE: { label: 'ROE 低于策略允许的最低阈值', metric: 'roe_latest', threshold: 'roe_min', direction: 'min' },
  SEVERE_PROFIT_DECLINE: { label: '净利润同比增速低于策略阈值', metric: 'net_income_yoy_latest', threshold: 'net_income_yoy_min', direction: 'min' },
  SEVERE_REVENUE_DECLINE: { label: '营业收入同比增速低于策略阈值', metric: 'revenue_yoy_latest', threshold: 'revenue_yoy_min', direction: 'min' },
  NEGATIVE_NET_MARGIN: { label: '净利率低于策略允许的最低阈值', metric: 'net_margin_latest', threshold: 'net_margin_min', direction: 'min' },
  HIGH_DEBT: { label: '资产负债率高于策略允许的最高阈值', metric: 'debt_ratio_latest', threshold: 'debt_ratio_max', direction: 'max' },
  ABNORMAL_GROSS_MARGIN: { label: '毛利率低于策略允许的最低阈值', metric: 'gross_margin_latest', threshold: 'gross_margin_min', direction: 'min' },
  MISSING_FINANCIAL_DATA: { label: '当前扫描时点可用的 PIT 核心财务字段不足，按 fail-closed 规则否决', direction: 'missing' },
}

const metricLabels: Record<string, string> = {
  roe_latest: 'ROE', revenue_yoy_latest: '营收同比', net_income_yoy_latest: '净利润同比',
  net_margin_latest: '净利率', debt_ratio_latest: '资产负债率', gross_margin_latest: '毛利率',
}

const value = (item: number | null | undefined) => item == null ? '—' : `${item.toFixed(2)}%`

export function FundamentalVetoExplanation({
  candidate,
  policy,
}: {
  candidate: DailyScanCandidate
  policy?: Policy
}) {
  const evidence = candidate.fundamental.evidence
  const metrics = evidence?.metrics
  const codes = candidate.fundamental.reason_codes
  // V4.8 runs predate immutable evidence.  Their historic PASS flag alone
  // must not be presented as a newly verified financial pass.
  const displayStatus = !evidence || evidence.availability === 'MISSING'
    ? (candidate.fundamental.status === 'PASS' ? 'UNKNOWN' : candidate.fundamental.status)
    : candidate.fundamental.status
  return <section className="rounded border border-border bg-base p-3">
    <div className="flex items-center justify-between gap-2">
      <h3 className="text-xs font-semibold">基本面 PIT 诊断</h3>
      <span className={displayStatus === 'VETO' ? 'text-xs text-danger' : 'text-xs text-accent'}>{displayStatus}</span>
    </div>
    <p className="mt-1 text-[11px] text-muted">{evidence?.availability === 'PIT_AVAILABLE' ? `目标日期 ${evidence.effective_as_of ?? '—'} 的已生效 PIT 值` : '本次运行未保存可用的 PIT 财务证据；不以当前财务数据替代。'}</p>
    <div className="mt-2 grid grid-cols-2 gap-1 text-xs sm:grid-cols-3">
      {(Object.entries(metricLabels) as [keyof DailyScanFundamentalMetrics, string][]).map(([key, label]) => <span key={key} className="rounded bg-surface px-2 py-1">{label}: {value(metrics?.[key])}</span>)}
    </div>
    {codes.length > 0 ? <ul className="mt-3 space-y-1 text-xs">
      {codes.map(code => {
        const detail = DEFINITIONS[code]
        const actual = detail?.metric ? metrics?.[detail.metric] : undefined
        const threshold = detail?.threshold ? policy?.[detail.threshold] : undefined
        return <li key={code} className="rounded border border-danger/20 bg-danger/5 p-2 text-secondary">
          <span className="font-mono text-danger">{code}</span><span className="ml-2">{detail?.label ?? '未识别的基本面否决原因'}</span>
          {detail?.metric && <span className="ml-2 text-muted">{metricLabels[detail.metric]} {value(actual)} / 阈值 {typeof threshold === 'number' ? `${threshold.toFixed(2)}%` : '—'}</span>}
        </li>
      })}
    </ul> : <p className="mt-3 text-xs text-secondary">未触发已记录的基本面否决原因。</p>}
  </section>
}
