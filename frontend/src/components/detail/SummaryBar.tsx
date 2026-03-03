import { useNavigate } from "react-router-dom";
import type { ParsedTrace } from "../../types";
import { fmtDur, fmtN } from "../../utils";

interface Props {
  parsedTrace: ParsedTrace;
  dir: string;
}

export function SummaryBar({ parsedTrace, dir }: Props) {
  const navigate = useNavigate();
  const { traceStart, traceEnd } = parsedTrace;
  const stats = traceEnd?.stats ?? {};
  const hasErrors = (stats.errors ?? 0) > 0;

  return (
    <div className="flex items-center gap-3 px-6 py-2.5 bg-bg-card shadow-sm shadow-black/8 shrink-0 flex-wrap">
      <button
        onClick={() => navigate(dir ? `/dir/${dir}` : "/")}
        className="bg-bg-surface ring-1 ring-border/50 text-text-muted px-3 py-1.5 rounded-lg text-sm cursor-pointer flex items-center gap-1.5 shrink-0 hover:ring-accent hover:text-accent transition-all"
      >
        &larr; Back
      </button>

      <div className="flex items-center gap-2 text-sm flex-wrap">
        <span className="font-mono font-medium">{fmtDur(traceEnd?.duration_s)}</span>
        <Dot />
        <span className="font-mono font-medium">{stats.spans ?? 0}</span>
        <span className="text-text-muted">spans</span>
        <Dot />
        <span className="font-mono font-medium text-blue">{fmtN(stats.total_input_tokens ?? 0)}</span>
        {((stats.total_cache_read_tokens ?? 0) + (stats.total_cache_creation_tokens ?? 0)) > 0 && (
          <span className="font-mono text-text-dim">({fmtN((stats.total_cache_read_tokens ?? 0) + (stats.total_cache_creation_tokens ?? 0))} cached)</span>
        )}
        <span className="text-text-muted">in</span>
        <span className="text-text-dim">/</span>
        <span className="font-mono font-medium text-orange">{fmtN(stats.total_output_tokens ?? 0)}</span>
        <span className="text-text-muted">out tokens</span>
        <Dot />
        <span className="font-mono font-medium">{stats.events ?? 0}</span>
        <span className="text-text-muted">events</span>
        <Dot />
        <span className={`font-mono font-medium ${hasErrors ? "text-err" : ""}`}>
          {stats.errors ?? 0}
        </span>
        <span className={`${hasErrors ? "text-err" : "text-text-muted"}`}>
          errors
        </span>
      </div>

      {traceStart?.tags && traceStart.tags.length > 0 && (
        <div className="flex items-center gap-1.5 ml-1">
          {traceStart.tags.map((t) => (
            <span
              key={t}
              className="inline-block px-2 py-0.5 rounded bg-accent/10 text-accent text-[11px] font-medium font-mono"
            >
              {t}
            </span>
          ))}
        </div>
      )}

      {traceStart?.thread_id && (
        <>
          <Dot />
          <span className="text-sm text-text-muted">thread</span>
          <span className="text-sm font-mono font-medium">{traceStart.thread_id}</span>
        </>
      )}
    </div>
  );
}

function Dot() {
  return <span className="text-text-dim select-none">&middot;</span>;
}
