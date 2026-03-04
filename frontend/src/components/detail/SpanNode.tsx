import type { ParsedSpan } from "../../types";
import { fmtDur, kindBadgeClass } from "../../utils";

interface Props {
  span: ParsedSpan;
  depth: number;
  t0: number;
  dur: number;
  qualifiedId?: string;
  selected: boolean;
  onSelect: (id: string) => void;
  hasErrorDescendant?: boolean;
}

function extractToolNames(span: ParsedSpan): string[] {
  if (span.kind !== "llm" || !span.output) return [];
  const out = span.output as Record<string, unknown>;
  const calls = out.tool_calls as Array<{ name?: string }> | undefined;
  if (!Array.isArray(calls)) return [];
  return calls.map((c) => c.name).filter((n): n is string => !!n);
}

export function SpanNode({
  span,
  depth,
  t0,
  dur,
  qualifiedId,
  selected,
  onSelect,
  hasErrorDescendant,
}: Props) {
  const s0 = span.ts_start ? new Date(span.ts_start).getTime() : t0;
  const s1 = span.ts_end ? new Date(span.ts_end).getTime() : s0;
  const left = (((s0 - t0) / dur) * 100).toFixed(1);
  const width = Math.max(((s1 - s0) / dur) * 100, 1).toFixed(1);
  const isErr = span.status === "error";
  const spanId = qualifiedId ?? span.id;
  const toolNames = extractToolNames(span);

  const barColor =
    span.kind === "llm"
      ? "bg-blue/70"
      : span.kind === "tool"
        ? "bg-purple/70"
        : "bg-accent/70";

  return (
    <div
      className={`flex items-center py-2 pr-4 cursor-pointer transition-colors border-l-3 ${
        selected
          ? "bg-bg-selected border-l-accent"
          : isErr
            ? "border-l-err hover:bg-bg-hover"
            : "border-l-transparent hover:bg-bg-hover"
      }`}
      style={{ paddingLeft: `calc(12px + ${depth} * 20px)` }}
      data-span-id={spanId}
      onClick={() => onSelect(spanId)}
    >
      <div className="flex-1 min-w-0 flex items-center gap-2">
        <span
          className={`w-1.5 h-1.5 rounded-full shrink-0 ${isErr ? "bg-err" : "bg-ok"}`}
        />
        {span.kind && (
          <span
            className={`inline-block px-1.5 py-0.5 rounded text-[11px] font-medium font-mono shrink-0 ${kindBadgeClass(span.kind)}`}
          >
            {span.kind}
          </span>
        )}
        <span className="text-[13px] font-medium truncate">{span.name}</span>
        {!isErr && hasErrorDescendant && (
          <span className="text-[10px] text-err/60 shrink-0" title="Has error descendants">!</span>
        )}
        {toolNames.length > 0 && (
          <span className="text-[11px] text-text-dim truncate shrink-0">
            {toolNames.join(", ")}
          </span>
        )}
      </div>
      <div className="flex items-center gap-3 shrink-0 ml-3">
        <div className="w-[140px] shrink-0 relative h-4">
          <div className="absolute top-[7px] left-0 right-0 h-0.5 bg-border/60 rounded-sm" />
          <div
            className={`absolute top-[3px] h-2.5 rounded-sm min-w-[3px] ${isErr ? "bg-err/70" : barColor}`}
            style={{ left: `${left}%`, width: `${width}%` }}
          />
        </div>
        <span className="text-[13px] text-text-muted font-mono whitespace-nowrap min-w-[52px] text-right">
          {fmtDur(span.duration_s)}
        </span>
      </div>
    </div>
  );
}
