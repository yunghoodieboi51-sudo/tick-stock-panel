export const DEFAULT_DAILY_SCAN_SIZING = {
  account_size: 100000,
  max_position_pct: 0.2,
  max_candidates: 10,
  risk_per_trade_pct: 0.01,
}

export type DailyScanSizing = typeof DEFAULT_DAILY_SCAN_SIZING

export function SizingSettings({ sizing, onChange }: { sizing: DailyScanSizing; onChange: (next: DailyScanSizing) => void }) {
  const update = (key: keyof DailyScanSizing, value: number) => onChange({ ...sizing, [key]: value })
  return <div className="grid grid-cols-2 gap-2 text-xs">
    {([['account_size', '账户规模'], ['max_position_pct', '单股上限'], ['risk_per_trade_pct', '单笔风险'], ['max_candidates', '候选上限']] as const).map(([key, label]) => <label key={key} className="text-muted">{label}<input type="number" min="0" step={key === 'max_candidates' ? 1 : 0.01} value={sizing[key]} onChange={event => update(key, Number(event.target.value))} className="mt-1 w-full rounded border border-border bg-base p-1.5 text-foreground" /></label>)}
    <p className="col-span-2 mt-1 text-[11px] text-muted">现有确定性分配语义：账户总额度上限等于账户规模；按 100 股整手估算，不新增独立总分配参数。</p>
  </div>
}
