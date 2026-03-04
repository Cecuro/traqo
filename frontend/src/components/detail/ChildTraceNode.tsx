import { useCallback, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import type { ChildTracer, ParsedTrace } from "../../types";
import { fmtDur, fmtN, buildErrorAncestorSet } from "../../utils";
import { SpanNode } from "./SpanNode";

interface Props {
  child: ChildTracer;
  depth: number;
  t0: number;
  dur: number;
  selected: boolean;
  selectedSpanId: string | null;
  onSelect: (id: string) => void;
  onToggle: (name: string) => void;
}

export function ChildTraceNode({
  child,
  depth,
  t0,
  dur: _,
  selected,
  selectedSpanId,
  onSelect,
  onToggle,
}: Props) {
  const navigate = useNavigate();
  const chevron = child.expanded ? "\u25BC" : "\u25B6";
  const childHasErrors = useMemo(
    () => child.parsed?.spans.some((s) => s.status === "error") ?? false,
    [child.parsed],
  );

  const statsHtml = child.stats ? (
    <span className="font-mono text-[11px] text-text-muted whitespace-nowrap truncate">
      <span className="text-blue">{fmtN(child.stats.total_input_tokens)}</span>
      /
      <span className="text-orange">
        {fmtN(child.stats.total_output_tokens)}
      </span>
      {" · "}
      {fmtDur(child.stats.duration_s)} · {child.stats.spans} spans
    </span>
  ) : (
    <span className="text-yellow text-[11px]">running…</span>
  );

  const handleOpen = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      navigate(`/trace/${encodeURIComponent(child.file)}`);
    },
    [navigate, child.file],
  );

  return (
    <>
      <div
        className={`flex items-center py-2 pr-4 cursor-pointer transition-colors border-l-3 border-l-orange bg-orange/4 hover:bg-orange/8 ${
          selected ? "bg-bg-selected" : ""
        } ${child.expanded ? "border-l-orange" : ""}`}
        style={{ paddingLeft: `calc(12px + ${depth} * 20px)` }}
        onClick={() => onToggle(child.name)}
      >
        <div className="flex-1 min-w-0 flex items-center gap-2">
          <span className="text-[10px] text-text-muted shrink-0 w-3.5 text-center transition-transform">
            {chevron}
          </span>
          <span className="inline-block px-1.5 py-0.5 rounded bg-orange/15 text-orange text-[11px] font-medium font-mono">
            child
          </span>
          <span className="text-[13px] font-medium truncate">{child.name}</span>
          {childHasErrors && (
            <span className="w-1.5 h-1.5 rounded-full shrink-0 bg-err" title="Contains errors" />
          )}
          {statsHtml}
        </div>
        <div className="flex items-center gap-2.5 shrink-0">
          <button
            onClick={handleOpen}
            className="text-[11px] text-text-dim no-underline font-mono whitespace-nowrap px-2.5 py-1 rounded transition-colors hover:text-accent hover:bg-accent/10"
          >
            open ↗
          </button>
        </div>
      </div>

      {child.expanded && (
        <div
          className="border-l-2 border-dashed border-orange/25"
          style={{
            marginLeft: `calc(12px + ${depth} * 20px + 7px)`,
          }}
        >
          {child.loading && (
            <div className="px-4 py-3 text-xs text-text-muted flex items-center gap-2">
              <span
                className="inline-block w-3.5 h-3.5 border-2 border-border border-t-orange rounded-full"
                style={{ animation: "spin 0.7s linear infinite" }}
              />
              Loading child trace…
            </div>
          )}
          {child.fetchError && (
            <div className="px-4 py-2 text-xs text-err bg-err-dim rounded m-1">
              Failed to load: {child.fetchError}
            </div>
          )}
          {child.parsed && (
            <ChildSpanTree
              parsed={child.parsed}
              childName={child.name}
              depth={depth + 1}
              parentT0={t0}
              selectedSpanId={selectedSpanId}
              onSelect={onSelect}
              onToggle={onToggle}
            />
          )}
        </div>
      )}
    </>
  );
}

function ChildSpanTree({
  parsed,
  childName,
  depth,
  parentT0,
  selectedSpanId,
  onSelect,
  onToggle,
}: {
  parsed: ParsedTrace;
  childName: string;
  depth: number;
  parentT0: number;
  selectedSpanId: string | null;
  onSelect: (id: string) => void;
  onToggle: (name: string) => void;
}) {
  const { spans, logEvents, traceStart, traceEnd, childTracers } = parsed;
  const t0 = traceStart?.ts
    ? new Date(traceStart.ts).getTime()
    : parentT0;
  const t1 = traceEnd?.ts ? new Date(traceEnd.ts).getTime() : t0;
  const dur = t1 - t0 || 1;

  // Build tree
  const roots: typeof spans = [];
  const childrenMap = new Map<string, typeof spans>();
  for (const s of spans) {
    if (!s.parent_id) {
      roots.push(s);
    } else {
      const arr = childrenMap.get(s.parent_id) ?? [];
      arr.push(s);
      childrenMap.set(s.parent_id, arr);
    }
  }
  const sortFn = (a: { ts_start?: string }, b: { ts_start?: string }) =>
    (a.ts_start ?? "") < (b.ts_start ?? "") ? -1 : 1;
  for (const c of childrenMap.values()) c.sort(sortFn);
  roots.sort(sortFn);

  const errorAncestors = useMemo(() => buildErrorAncestorSet(spans), [spans]);

  // Interleave root spans and grandchild tracers by timestamp
  type Item =
    | { type: "span"; span: (typeof spans)[0]; ts: string }
    | { type: "child"; child: ChildTracer; ts: string };

  const items: Item[] = [];
  for (const r of roots) items.push({ type: "span", span: r, ts: r.ts_start ?? "" });
  for (const [, gc] of childTracers) {
    items.push({ type: "child", child: gc, ts: gc.startedTs ?? "" });
  }
  items.sort((a, b) => (a.ts < b.ts ? -1 : 1));

  function renderSubtree(
    span: (typeof spans)[0],
    d: number,
  ): React.ReactNode {
    const qid = `child:${childName}:${span.id}`;
    return (
      <div key={qid}>
        <SpanNode
          span={span}
          depth={d}
          t0={t0}
          dur={dur}
          qualifiedId={qid}
          selected={selectedSpanId === qid}
          onSelect={onSelect}
          hasErrorDescendant={errorAncestors.has(span.id)}
        />
        {(childrenMap.get(span.id) ?? []).map((ch) =>
          renderSubtree(ch, d + 1),
        )}
      </div>
    );
  }

  return (
    <>
      {items.map((item, i) =>
        item.type === "span" ? (
          renderSubtree(item.span, depth)
        ) : (
          <ChildTraceNode
            key={`gc-${item.child.name}-${i}`}
            child={item.child}
            depth={depth}
            t0={t0}
            dur={dur}
            selected={false}
            selectedSpanId={selectedSpanId}
            onSelect={onSelect}
            onToggle={onToggle}
          />
        ),
      )}
      {logEvents.map((ev) => (
        <div
          key={`ev-${ev.id}`}
          className="flex items-center py-2 pr-4 cursor-pointer opacity-60 hover:bg-bg-hover border-l-3 border-l-transparent"
          style={{ paddingLeft: `calc(12px + ${depth} * 20px)` }}
          onClick={() => onSelect(`ev-${ev.id}`)}
        >
          <div className="flex-1 min-w-0 flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full shrink-0 bg-yellow" />
            <span className="inline-block px-1.5 py-0.5 rounded bg-yellow/12 text-yellow text-[11px] font-medium font-mono">
              event
            </span>
            <span className="text-[13px] font-medium truncate">
              {ev.name}
            </span>
          </div>
        </div>
      ))}
      {items.length === 0 && logEvents.length === 0 && (
        <div
          className="text-text-dim text-xs"
          style={{
            padding: `8px 0 8px calc(12px + ${depth} * 20px)`,
          }}
        >
          No spans
        </div>
      )}
    </>
  );
}
