import type { ParsedSpan, ParsedTrace, TraceEvent } from "../../types";
import { fmtDur, fmtTimeFull, kindBadgeClass } from "../../utils";
import { JsonSection } from "./JsonSection";
import { TokenBar } from "./TokenBar";

interface Props {
  spanId: string | null;
  parsedTrace: ParsedTrace;
}

export function SpanDetail({ spanId, parsedTrace }: Props) {
  if (!spanId) {
    return (
      <div className="flex flex-col items-center justify-center text-text-dim gap-2 py-16">
        <div className="text-sm">Select a span to view details</div>
      </div>
    );
  }

  // Trace I/O
  if (spanId === "__trace__") {
    return <TraceIODetail parsedTrace={parsedTrace} />;
  }

  // Log event
  if (spanId.startsWith("ev-")) {
    const evId = spanId.slice(3);
    const ev =
      parsedTrace.logEvents.find((e) => e.id === evId) ??
      findEventInChildren(evId, parsedTrace);
    if (ev) return <EventDetail event={ev} />;
  }

  // Child span
  if (spanId.startsWith("child:")) {
    const parts = spanId.split(":");
    if (parts.length >= 3) {
      const childName = parts[1]!;
      const childSpanId = parts.slice(2).join(":");
      const child = parsedTrace.childTracers.get(childName);
      if (child?.parsed) {
        const span = child.parsed.spans.find((s) => s.id === childSpanId);
        if (span) return <SpanContent span={span} />;
      }
    }
  }

  // Regular span
  const span = parsedTrace.spans.find((s) => s.id === spanId);
  if (!span) {
    return (
      <div className="flex flex-col items-center justify-center text-text-dim gap-2 py-16">
        <div className="text-sm">Span not found</div>
      </div>
    );
  }

  return <SpanContent span={span} />;
}

function SpanContent({ span }: { span: ParsedSpan }) {
  const meta = span.metadata as Record<string, unknown> | undefined;
  const tokenUsage = meta?.token_usage as
    | {
        input_tokens?: number;
        output_tokens?: number;
        reasoning_tokens?: number;
        cache_read_tokens?: number;
      }
    | undefined;

  return (
    <div>
      {/* Header */}
      <div className="flex items-center gap-3 mb-6 flex-wrap">
        <span className="text-xl font-semibold">{span.name}</span>
        {span.kind && (
          <span
            className={`inline-block px-2.5 py-1 rounded text-[11px] font-medium font-mono ${kindBadgeClass(span.kind)}`}
          >
            {span.kind}
          </span>
        )}
        <span
          className={`inline-block px-2.5 py-1 rounded text-[11px] font-medium font-mono ${span.status === "error" ? "bg-err-dim text-err" : "bg-ok-dim text-ok"}`}
        >
          {span.status ?? "ok"}
        </span>
        <span className="font-mono text-sm text-text-muted">
          {fmtDur(span.duration_s)}
        </span>
      </div>

      {/* Timing */}
      {(span.ts_start || span.ts_end) && (
        <div className="mb-6">
          <SectionTitle>Timing</SectionTitle>
          <div className="bg-bg-card rounded-xl ring-1 ring-border/30 overflow-hidden">
          <table className="w-full border-collapse">
            <tbody>
              {span.ts_start && (
                <tr>
                  <td className="px-4 py-2.5 text-sm border-b border-border/40 text-text-muted whitespace-nowrap w-[120px]">
                    Start
                  </td>
                  <td className="px-4 py-2.5 text-sm border-b border-border/40 font-mono">
                    {fmtTimeFull(span.ts_start)}
                  </td>
                </tr>
              )}
              {span.ts_end && (
                <tr>
                  <td className="px-4 py-2.5 text-sm border-b border-border/40 text-text-muted whitespace-nowrap w-[120px]">
                    End
                  </td>
                  <td className="px-4 py-2.5 text-sm border-b border-border/40 font-mono">
                    {fmtTimeFull(span.ts_end)}
                  </td>
                </tr>
              )}
              {span.duration_s != null && (
                <tr>
                  <td className="px-4 py-2.5 text-sm border-b border-border/40 text-text-muted whitespace-nowrap w-[120px]">
                    Duration
                  </td>
                  <td className="px-4 py-2.5 text-sm border-b border-border/40 font-mono">
                    {span.duration_s.toFixed(3)}s
                  </td>
                </tr>
              )}
            </tbody>
          </table>
          </div>
        </div>
      )}

      {/* Tags */}
      {span.tags.length > 0 && (
        <div className="mb-6">
          <SectionTitle>Tags</SectionTitle>
          <div className="flex gap-2 flex-wrap">
            {span.tags.map((tag) => (
              <span
                key={tag}
                className="inline-block px-2.5 py-1 rounded bg-accent/10 text-accent text-[11px] font-medium font-mono"
              >
                {tag}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Error */}
      {span.error && (
        <div className="mb-6">
          <SectionTitle>Error</SectionTitle>
          <div className="bg-err-dim ring-1 ring-err/20 rounded-xl p-5">
            <div className="font-semibold text-err text-sm">
              {span.error.type ?? "Error"}
            </div>
            <div className="text-text text-sm mt-2 font-mono leading-relaxed">
              {span.error.message ?? ""}
            </div>
          </div>
        </div>
      )}

      {/* Token usage */}
      {tokenUsage && (
        <TokenBar
          inputTokens={tokenUsage.input_tokens ?? 0}
          outputTokens={tokenUsage.output_tokens ?? 0}
          reasoningTokens={tokenUsage.reasoning_tokens}
          cacheReadTokens={tokenUsage.cache_read_tokens}
        />
      )}

      {/* JSON sections */}
      {span.input != null && <JsonSection title="Input" value={span.input} />}
      {span.output != null && (
        <JsonSection title="Output" value={span.output} />
      )}

      {meta && Object.keys(meta).length > 0 && (() => {
        const m = { ...meta };
        delete m.token_usage;
        return Object.keys(m).length > 0 ? (
          <JsonSection title="Metadata" value={m} />
        ) : null;
      })()}
    </div>
  );
}

function TraceIODetail({ parsedTrace }: { parsedTrace: ParsedTrace }) {
  const { traceStart, traceEnd } = parsedTrace;
  return (
    <div>
      <div className="flex items-center gap-3 mb-6">
        <span className="text-xl font-semibold">Trace I/O</span>
        <span className="inline-block px-2.5 py-1 rounded bg-yellow/12 text-yellow text-[11px] font-medium font-mono">
          trace
        </span>
      </div>
      {traceStart?.input != null && (
        <JsonSection title="Trace Input" value={traceStart.input} />
      )}
      {traceEnd?.output != null && (
        <JsonSection title="Trace Output" value={traceEnd.output} />
      )}
      {traceStart?.metadata &&
        Object.keys(traceStart.metadata).length > 0 && (
          <JsonSection title="Trace Metadata" value={traceStart.metadata} />
        )}
    </div>
  );
}

function EventDetail({ event }: { event: TraceEvent }) {
  return (
    <div>
      <div className="flex items-center gap-3 mb-6">
        <span className="text-xl font-semibold">{event.name}</span>
        <span className="inline-block px-2.5 py-1 rounded bg-yellow/12 text-yellow text-[11px] font-medium font-mono">
          event
        </span>
      </div>
      {event.ts && (
        <div className="mb-6">
          <SectionTitle>Timestamp</SectionTitle>
          <div className="font-mono text-sm">{fmtTimeFull(event.ts)}</div>
        </div>
      )}
      {event.data != null && <JsonSection title="Data" value={event.data} />}
    </div>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-sm font-semibold text-text-muted mb-3">
      {children}
    </div>
  );
}

function findEventInChildren(
  evId: string,
  parsed: ParsedTrace,
): TraceEvent | undefined {
  for (const [, child] of parsed.childTracers) {
    if (!child.parsed) continue;
    const ev = child.parsed.logEvents.find((e) => e.id === evId);
    if (ev) return ev;
  }
  return undefined;
}
