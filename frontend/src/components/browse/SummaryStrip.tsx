import type { AggregatedStats } from "../../types";
import { fmtN, fmtDur } from "../../utils";

export function SummaryStrip({ stats }: { stats: AggregatedStats }) {
  return (
    <div className="flex items-center gap-2 px-4 py-2.5 bg-bg-card shadow-sm ring-1 ring-border/30 rounded-xl mb-4 text-sm flex-wrap">
      <span className="font-mono font-bold">{stats.traceCount}</span>
      <span className="text-text-muted">traces</span>
      <Dot />
      <span className="font-mono font-medium text-blue">{fmtN(stats.totalIn)}</span>
      <span className="text-text-muted">in</span>
      <span className="text-text-dim">/</span>
      <span className="font-mono font-medium text-orange">{fmtN(stats.totalOut)}</span>
      <span className="text-text-muted">out tokens</span>
      <Dot />
      <span className="font-mono font-medium">{fmtDur(stats.avgDuration)}</span>
      <span className="text-text-muted">avg</span>
      <Dot />
      <span className={`font-mono font-medium ${stats.totalErrors > 0 ? "text-err" : ""}`}>
        {stats.totalErrors}
      </span>
      <span className={stats.totalErrors > 0 ? "text-err" : "text-text-muted"}>
        errors
      </span>
    </div>
  );
}

function Dot() {
  return <span className="text-text-dim select-none">&middot;</span>;
}
