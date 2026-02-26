import { useNavigate } from "react-router-dom";
import type { TraceSummary } from "../../types";
import { TraceIcon } from "../../icons";
import { fmtN, fmtDur, fmtTime, summarizeInput } from "../../utils";

export function TraceRow({ trace }: { trace: TraceSummary }) {
  const navigate = useNavigate();
  const stats = trace.stats ?? {};
  const hasErr = (stats.errors ?? 0) > 0;
  const displayName = trace.file.split("/").pop()?.replace(/\.jsonl$/, "") ?? "";
  const desc = summarizeInput(trace.input);

  return (
    <div
      className="flex items-center gap-3 px-4 py-3 border-b border-border/50 cursor-pointer transition-colors hover:bg-bg-hover last:border-b-0"
      onClick={() => navigate(`/trace/${encodeURIComponent(trace.file)}`)}
    >
      <div
        className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${hasErr ? "bg-err-dim text-err" : "bg-ok/10 text-ok"}`}
      >
        <TraceIcon className="w-[18px] h-[18px]" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-[15px] font-medium truncate">{displayName}</div>
        <div className="text-xs text-text-muted mt-0.5 truncate">
          {desc}
          {trace.tags && trace.tags.length > 0 && (
            <>
              {desc ? " · " : ""}
              {trace.tags.slice(0, 3).map((tag) => (
                <span
                  key={tag}
                  className="inline-block px-[7px] py-0.5 rounded bg-accent/10 text-accent text-[11px] font-medium font-mono mr-0.5"
                >
                  {tag}
                </span>
              ))}
            </>
          )}
        </div>
      </div>
      <div className="hidden sm:flex items-center gap-4 shrink-0">
        <MetaItem>
          <span className="font-mono text-[13px] font-medium">
            <span className="text-blue">
              {fmtN(stats.total_input_tokens ?? 0)}
            </span>
            {" / "}
            <span className="text-orange">
              {fmtN(stats.total_output_tokens ?? 0)}
            </span>
          </span>
          <span className="text-[10px] text-text-dim uppercase tracking-wide">
            tokens
          </span>
        </MetaItem>
        <MetaItem>
          <span className="font-mono text-[13px] font-medium">
            {fmtDur(trace.duration_s)}
          </span>
          <span className="text-[10px] text-text-dim uppercase tracking-wide">
            duration
          </span>
        </MetaItem>
        <MetaItem>
          <span className="font-mono text-[13px] font-medium">
            {stats.spans ?? 0}
          </span>
          <span className="text-[10px] text-text-dim uppercase tracking-wide">
            spans
          </span>
        </MetaItem>
        <MetaItem>
          <span className="font-mono text-[13px] font-medium">
            {fmtTime(trace.ts)}
          </span>
          <span className="text-[10px] text-text-dim uppercase tracking-wide">
            time
          </span>
        </MetaItem>
        <MetaItem>
          {hasErr ? (
            <span className="inline-block px-[7px] py-0.5 rounded bg-err-dim text-err text-[11px] font-medium font-mono">
              {stats.errors} error{stats.errors !== 1 ? "s" : ""}
            </span>
          ) : (
            <span className="inline-block px-[7px] py-0.5 rounded bg-ok-dim text-ok text-[11px] font-medium font-mono">
              ok
            </span>
          )}
        </MetaItem>
      </div>
      <div className="text-text-dim text-base shrink-0 ml-1">&rsaquo;</div>
    </div>
  );
}

function MetaItem({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex flex-col items-end min-w-[70px]">{children}</div>
  );
}
