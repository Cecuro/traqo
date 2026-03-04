import type {
  TraceSummary,
  TraceEvent,
  ParsedSpan,
  ParsedTrace,
  ChildTracer,
  AggregatedStats,
  FolderItem,
} from "./types";

// ── Formatting ─────────────────────────────────────────────

export function fmtDur(s?: number | null): string {
  if (s == null) return "\u2013";
  if (s < 0.001) return "<1ms";
  if (s < 1) return `${(s * 1000).toFixed(0)}ms`;
  if (s < 60) return `${s.toFixed(2)}s`;
  return `${(s / 60).toFixed(1)}m`;
}

export function fmtTime(ts?: string | null): string {
  if (!ts) return "\u2013";
  return new Date(ts).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function fmtTimeFull(ts?: string | null): string {
  if (!ts) return "\u2013";
  return new Date(ts).toISOString().replace("T", " ").replace("Z", " UTC");
}

export function fmtN(n?: number | null): string {
  if (n == null) return "0";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(n);
}

export function debounce<T extends (...args: never[]) => void>(
  fn: T,
  ms: number,
): (...args: Parameters<T>) => void {
  let t: ReturnType<typeof setTimeout>;
  return (...args: Parameters<T>) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

// ── Data processing ────────────────────────────────────────

export function summarizeInput(input: unknown): string {
  if (input == null) return "";
  if (typeof input === "string")
    return input.length > 80 ? input.slice(0, 80) + "\u2026" : input;
  if (typeof input === "object" && input !== null) {
    const obj = input as Record<string, unknown>;
    for (const key of [
      "query",
      "question",
      "prompt",
      "message",
      "text",
      "input",
    ]) {
      if (obj[key] && typeof obj[key] === "string") {
        const v = obj[key] as string;
        return v.length > 80 ? v.slice(0, 80) + "\u2026" : v;
      }
    }
    const keys = Object.keys(obj);
    return keys.length ? keys.join(", ") : "";
  }
  return String(input).slice(0, 80);
}

function resolveChildKey(
  parentFileKey: string | null,
  childData: Record<string, unknown>,
): string {
  const fallback =
    (childData.child_file as string) ||
    (childData.child_name as string) + ".jsonl";
  if (!parentFileKey) return fallback;
  const parts = parentFileKey.split("/");
  parts[parts.length - 1] = fallback;
  return parts.join("/");
}

export function parseEvents(
  events: TraceEvent[],
  parentFileKey: string | null,
): ParsedTrace {
  const starts = new Map<string, TraceEvent>();
  const ends = new Map<string, TraceEvent>();
  const logEvents: TraceEvent[] = [];
  const childTracers = new Map<string, ChildTracer>();
  let traceStart: TraceEvent | null = null;
  let traceEnd: TraceEvent | null = null;

  for (const ev of events) {
    switch (ev.type) {
      case "trace_start":
        traceStart = ev;
        break;
      case "trace_end":
        traceEnd = ev;
        break;
      case "span_start":
        if (ev.id) starts.set(ev.id, ev);
        break;
      case "span_end":
        if (ev.id) ends.set(ev.id, ev);
        break;
      case "event": {
        if (ev.name === "child_started" && ev.data) {
          const name = ev.data.child_name as string;
          const file = resolveChildKey(parentFileKey, ev.data);
          childTracers.set(name, {
            name,
            file,
            startedTs: ev.ts,
            stats: null,
            parsed: null,
            expanded: false,
            fetchError: null,
            loading: false,
          });
        } else if (ev.name === "child_ended" && ev.data) {
          const name = ev.data.child_name as string;
          const entry = childTracers.get(name);
          if (entry) {
            entry.stats = {
              spans: ev.data.spans as number | undefined,
              total_input_tokens: ev.data.total_input_tokens as
                | number
                | undefined,
              total_output_tokens: ev.data.total_output_tokens as
                | number
                | undefined,
              duration_s: ev.data.duration_s as number | undefined,
            };
            if (ev.data.child_file)
              entry.file = resolveChildKey(parentFileKey, ev.data);
          }
        } else {
          logEvents.push(ev);
        }
        break;
      }
    }
  }

  const spans: ParsedSpan[] = [];
  for (const [id, end] of ends) {
    const start = starts.get(id) ?? ({} as TraceEvent);
    spans.push({
      id,
      parent_id: end.parent_id,
      name: end.name ?? "",
      kind: end.kind ?? start.kind,
      status: end.status,
      duration_s: end.duration_s,
      input: start.input,
      output: end.output,
      metadata: end.metadata ?? start.metadata,
      tags: end.tags ?? start.tags ?? [],
      error: end.error,
      ts_start: start.ts,
      ts_end: end.ts,
    });
  }

  return { spans, logEvents, traceStart, traceEnd, childTracers };
}

export function tracesInDir(
  traces: TraceSummary[],
  prefix: string,
): TraceSummary[] {
  if (!prefix) return traces;
  return traces.filter((t) => t.file.startsWith(prefix + "/"));
}

export function aggregateStats(traces: TraceSummary[]): AggregatedStats {
  let spans = 0,
    errors = 0,
    tokIn = 0,
    tokOut = 0,
    dur = 0;
  for (const t of traces) {
    const s = t.stats ?? {};
    spans += s.spans ?? 0;
    errors += s.errors ?? 0;
    tokIn += s.total_input_tokens ?? 0;
    tokOut += s.total_output_tokens ?? 0;
    dur += t.duration_s ?? 0;
  }
  return {
    traceCount: traces.length,
    totalSpans: spans,
    totalErrors: errors,
    totalIn: tokIn,
    totalOut: tokOut,
    totalDuration: dur,
    avgDuration: traces.length ? dur / traces.length : 0,
  };
}

export function itemsAtDir(
  dirPrefix: string,
  traces: TraceSummary[],
  allTraces: TraceSummary[],
): { dirs: FolderItem[]; traces: TraceSummary[] } {
  const dirs = new Map<string, string>();
  const direct: TraceSummary[] = [];
  const prefix = dirPrefix ? dirPrefix + "/" : "";

  for (const t of traces) {
    const rel = dirPrefix ? t.file.slice(prefix.length) : t.file;
    const slash = rel.indexOf("/");
    if (slash === -1) {
      direct.push(t);
    } else {
      const name = rel.slice(0, slash);
      if (!dirs.has(name)) dirs.set(name, prefix + name);
    }
  }

  const dirList: FolderItem[] = [];
  for (const [name, path] of dirs) {
    const sub = tracesInDir(allTraces, path);
    const agg = aggregateStats(sub);
    let latest = "";
    for (const t of sub) if (t.ts && t.ts > latest) latest = t.ts;
    dirList.push({ type: "folder", name, path, latestTs: latest, ...agg });
  }
  dirList.sort((a, b) => a.name.localeCompare(b.name));

  return { dirs: dirList, traces: direct };
}

export function kindBadgeClass(kind: string): string {
  if (kind === "llm") return "bg-blue/12 text-blue";
  if (kind === "tool") return "bg-purple/12 text-purple";
  return "bg-yellow/12 text-yellow";
}

// ── Error navigation helpers ──────────────────────────────

export interface ErrorSpanRef {
  spanId: string;
  name: string;
  errorType?: string;
  errorMessage?: string;
}

/** Collect all error spans from a parsed trace (including expanded child tracers). */
export function collectErrorSpans(parsed: ParsedTrace): ErrorSpanRef[] {
  const errors: ErrorSpanRef[] = [];
  const sorted = [...parsed.spans].sort((a, b) =>
    (a.ts_start ?? "") < (b.ts_start ?? "") ? -1 : 1,
  );
  for (const span of sorted) {
    if (span.status === "error") {
      errors.push({
        spanId: span.id,
        name: span.name,
        errorType: span.error?.type,
        errorMessage: span.error?.message,
      });
    }
  }
  for (const [childName, child] of parsed.childTracers) {
    if (!child.parsed) continue;
    const childSorted = [...child.parsed.spans].sort((a, b) =>
      (a.ts_start ?? "") < (b.ts_start ?? "") ? -1 : 1,
    );
    for (const span of childSorted) {
      if (span.status === "error") {
        errors.push({
          spanId: `child:${childName}:${span.id}`,
          name: span.name,
          errorType: span.error?.type,
          errorMessage: span.error?.message,
        });
      }
    }
  }
  return errors;
}

/**
 * Returns a Set of span IDs that have at least one error descendant.
 * Walks up from every error span to the root, marking each ancestor.
 */
export function buildErrorAncestorSet(spans: ParsedSpan[]): Set<string> {
  const parentMap = new Map<string, string>();
  for (const s of spans) {
    if (s.parent_id) parentMap.set(s.id, s.parent_id);
  }
  const ancestors = new Set<string>();
  for (const s of spans) {
    if (s.status !== "error") continue;
    let current = s.parent_id;
    while (current) {
      if (ancestors.has(current)) break;
      ancestors.add(current);
      current = parentMap.get(current);
    }
  }
  return ancestors;
}
