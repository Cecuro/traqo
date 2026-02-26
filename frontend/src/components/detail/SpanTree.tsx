import { useCallback } from "react";
import type { ParsedTrace, ParsedSpan, ChildTracer } from "../../types";
import { fmtDur } from "../../utils";
import { fetchTrace } from "../../api";
import { parseEvents } from "../../utils";
import { useTraceDetailDispatch } from "../../context";
import { SpanNode } from "./SpanNode";
import { ChildTraceNode } from "./ChildTraceNode";

const childCache = new Map<string, ParsedTrace>();

interface Props {
  parsedTrace: ParsedTrace;
  selectedSpanId: string | null;
  onSelect: (id: string) => void;
}

export function SpanTree({ parsedTrace, selectedSpanId, onSelect }: Props) {
  const traceDispatch = useTraceDetailDispatch();
  const { spans, logEvents, traceStart, traceEnd, childTracers } =
    parsedTrace;

  const t0 = traceStart?.ts ? new Date(traceStart.ts).getTime() : 0;
  const t1 = traceEnd?.ts ? new Date(traceEnd.ts).getTime() : 0;
  const dur = t1 - t0 || 1;

  // Build parent-child tree
  const roots: ParsedSpan[] = [];
  const children = new Map<string, ParsedSpan[]>();
  for (const s of spans) {
    if (!s.parent_id) {
      roots.push(s);
    } else {
      const arr = children.get(s.parent_id) ?? [];
      arr.push(s);
      children.set(s.parent_id, arr);
    }
  }
  const sortFn = (a: { ts_start?: string }, b: { ts_start?: string }) =>
    (a.ts_start ?? "") < (b.ts_start ?? "") ? -1 : 1;
  for (const c of children.values()) c.sort(sortFn);
  roots.sort(sortFn);

  const handleToggleChild = useCallback(
    (childName: string) => {
      const child = childTracers.get(childName);
      if (!child) return;

      if (child.expanded) {
        traceDispatch({
          type: "UPDATE_CHILD",
          name: childName,
          update: (c) => {
            c.expanded = false;
          },
        });
      } else {
        if (childCache.has(child.file)) {
          traceDispatch({
            type: "UPDATE_CHILD",
            name: childName,
            update: (c) => {
              c.expanded = true;
              c.parsed = childCache.get(child.file)!;
            },
          });
        } else {
          traceDispatch({
            type: "UPDATE_CHILD",
            name: childName,
            update: (c) => {
              c.expanded = true;
              c.loading = true;
            },
          });
          fetchTrace(child.file)
            .then((data) => {
              const parsed = parseEvents(data.events ?? [], child.file);
              childCache.set(child.file, parsed);
              traceDispatch({
                type: "UPDATE_CHILD",
                name: childName,
                update: (c) => {
                  c.parsed = parsed;
                  c.fetchError = null;
                  c.loading = false;
                },
              });
            })
            .catch((e) => {
              traceDispatch({
                type: "UPDATE_CHILD",
                name: childName,
                update: (c) => {
                  c.fetchError =
                    e instanceof Error ? e.message : String(e);
                  c.loading = false;
                },
              });
            });
        }
      }
    },
    [childTracers, traceDispatch],
  );

  // Interleave root spans and child tracers by timestamp
  type Item =
    | { type: "span"; span: ParsedSpan; ts: string }
    | { type: "child"; child: ChildTracer; ts: string };

  const items: Item[] = [];
  for (const r of roots) items.push({ type: "span", span: r, ts: r.ts_start ?? "" });
  for (const [, ct] of childTracers) {
    items.push({ type: "child", child: ct, ts: ct.startedTs ?? "" });
  }
  items.sort((a, b) => (a.ts < b.ts ? -1 : 1));

  function renderSubtree(span: ParsedSpan, depth: number): React.ReactNode {
    return (
      <div key={span.id}>
        <SpanNode
          span={span}
          depth={depth}
          t0={t0}
          dur={dur}
          selected={selectedSpanId === span.id}
          onSelect={onSelect}
        />
        {(children.get(span.id) ?? []).map((ch) =>
          renderSubtree(ch, depth + 1),
        )}
      </div>
    );
  }

  return (
    <div className="py-2">
      {/* Trace I/O node */}
      {(traceStart?.input != null || traceEnd?.output != null) && (
        <div
          className={`flex items-center py-2 pr-4 pl-3 cursor-pointer opacity-70 transition-colors border-l-3 ${
            selectedSpanId === "__trace__"
              ? "bg-bg-selected border-l-accent"
              : "border-l-transparent hover:bg-bg-hover"
          }`}
          onClick={() => onSelect("__trace__")}
        >
          <div className="flex-1 min-w-0 flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full shrink-0 bg-ok" />
            <span className="inline-block px-1.5 py-0.5 rounded bg-yellow/12 text-yellow text-[11px] font-medium font-mono">
              trace
            </span>
            <span className="text-[13px] font-medium">Trace I/O</span>
          </div>
          <div className="flex items-center gap-3 shrink-0 ml-3">
            <div className="w-[140px] shrink-0 relative h-4">
              <div className="absolute top-[7px] left-0 right-0 h-0.5 bg-border/60 rounded-sm" />
              <div
                className="absolute top-[3px] h-2.5 rounded-sm min-w-[3px] bg-accent/70"
                style={{ left: "0%", width: "100%" }}
              />
            </div>
            <span className="text-[13px] text-text-muted font-mono whitespace-nowrap min-w-[52px] text-right">
              {fmtDur(traceEnd?.duration_s)}
            </span>
          </div>
        </div>
      )}

      {/* Spans and child tracers */}
      {items.map((item, i) =>
        item.type === "span" ? (
          renderSubtree(item.span, 0)
        ) : (
          <ChildTraceNode
            key={`ct-${item.child.name}-${i}`}
            child={item.child}
            depth={0}
            t0={t0}
            dur={dur}
            selected={false}
            selectedSpanId={selectedSpanId}
            onSelect={onSelect}
            onToggle={handleToggleChild}
          />
        ),
      )}

      {/* Log events */}
      {logEvents.map((ev) => (
        <div
          key={`ev-${ev.id}`}
          className={`flex items-center py-2 pr-4 pl-3 cursor-pointer opacity-60 transition-colors border-l-3 ${
            selectedSpanId === `ev-${ev.id}`
              ? "bg-bg-selected border-l-accent"
              : "border-l-transparent hover:bg-bg-hover"
          }`}
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
        <div className="flex flex-col items-center justify-center text-text-dim gap-2 py-12">
          <div className="text-sm">No spans</div>
        </div>
      )}
    </div>
  );
}
