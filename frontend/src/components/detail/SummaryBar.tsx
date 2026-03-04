import { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import type { ParsedTrace } from "../../types";
import { fmtDur, fmtN, type ErrorSpanRef } from "../../utils";

interface Props {
  parsedTrace: ParsedTrace;
  dir: string;
  errorSpans: ErrorSpanRef[];
  onSelectError: (spanId: string) => void;
  onNextError: () => void;
  onPrevError: () => void;
  currentErrorIndex: number;
}

export function SummaryBar({
  parsedTrace,
  dir,
  errorSpans,
  onSelectError,
  onNextError,
  onPrevError,
  currentErrorIndex,
}: Props) {
  const navigate = useNavigate();
  const { traceStart, traceEnd } = parsedTrace;
  const stats = traceEnd?.stats ?? {};
  const hasErrors = (stats.errors ?? 0) > 0;

  const [errDropdownOpen, setErrDropdownOpen] = useState(false);
  const errDropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (errDropdownRef.current && !errDropdownRef.current.contains(e.target as Node)) {
        setErrDropdownOpen(false);
      }
    }
    document.addEventListener("click", handleClick);
    return () => document.removeEventListener("click", handleClick);
  }, []);

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

        {/* Error count with dropdown and navigation */}
        <div className="relative flex items-center gap-1" ref={errDropdownRef}>
          <button
            onClick={() => hasErrors && setErrDropdownOpen((v) => !v)}
            className={`flex items-center gap-1 ${hasErrors ? "cursor-pointer hover:bg-err-dim rounded px-1.5 py-0.5 -mx-1.5 -my-0.5 transition-colors" : ""}`}
            disabled={!hasErrors}
          >
            <span className={`font-mono font-medium ${hasErrors ? "text-err" : ""}`}>
              {stats.errors ?? 0}
            </span>
            <span className={`${hasErrors ? "text-err" : "text-text-muted"}`}>
              errors
            </span>
            {hasErrors && (
              <span className="text-[10px] text-err ml-0.5">&#9662;</span>
            )}
          </button>

          {hasErrors && currentErrorIndex >= 0 && (
            <span className="text-[11px] text-err font-mono ml-1">
              {currentErrorIndex + 1}/{errorSpans.length}
            </span>
          )}

          {hasErrors && (
            <div className="flex flex-col -my-1 ml-0.5">
              <button
                onClick={(e) => { e.stopPropagation(); onPrevError(); }}
                className="text-err hover:bg-err-dim rounded px-0.5 transition-colors text-[10px] leading-none py-0.5 cursor-pointer"
                title="Previous error (Shift+E)"
              >
                &#9650;
              </button>
              <button
                onClick={(e) => { e.stopPropagation(); onNextError(); }}
                className="text-err hover:bg-err-dim rounded px-0.5 transition-colors text-[10px] leading-none py-0.5 cursor-pointer"
                title="Next error (E)"
              >
                &#9660;
              </button>
            </div>
          )}

          {errDropdownOpen && (
            <div className="absolute top-full mt-1.5 left-0 bg-bg-card ring-1 ring-border/50 rounded-xl min-w-[320px] max-w-[420px] shadow-xl z-20 py-1.5 max-h-72 overflow-y-auto">
              <div className="px-3.5 py-2 text-[11px] font-semibold text-text-dim uppercase tracking-wide border-b border-border/30">
                Errors ({errorSpans.length})
              </div>
              {errorSpans.map((err) => (
                <div
                  key={err.spanId}
                  className="px-3.5 py-2.5 cursor-pointer transition-colors hover:bg-bg-hover border-b border-border/20 last:border-b-0"
                  onClick={() => {
                    onSelectError(err.spanId);
                    setErrDropdownOpen(false);
                  }}
                >
                  <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full shrink-0 bg-err" />
                    <span className="text-[13px] font-medium truncate">{err.name}</span>
                  </div>
                  {(err.errorType || err.errorMessage) && (
                    <div className="ml-3.5 mt-1 text-[12px] text-text-muted truncate">
                      {err.errorType && (
                        <span className="text-err font-mono">{err.errorType}</span>
                      )}
                      {err.errorType && err.errorMessage && (
                        <span className="text-text-dim"> — </span>
                      )}
                      {err.errorMessage && (
                        <span>
                          {err.errorMessage.length > 60
                            ? err.errorMessage.slice(0, 60) + "\u2026"
                            : err.errorMessage}
                        </span>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
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
