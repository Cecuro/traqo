/** Shape returned by /api/traces — TraceSummary.to_dict() */
export interface TraceSummary {
  file: string;
  ts?: string;
  input?: unknown;
  tags?: string[];
  thread_id?: string;
  duration_s?: number;
  stats?: TraceStats;
}

export interface TraceStats {
  spans?: number;
  errors?: number;
  events?: number;
  total_input_tokens?: number;
  total_output_tokens?: number;
  total_cache_read_tokens?: number;
  total_cache_creation_tokens?: number;
}

/** Shape returned by /api/trace?file=... */
export interface TraceResponse {
  file: string;
  events: TraceEvent[];
  error?: string;
}

export interface TraceEvent {
  type:
    | "trace_start"
    | "trace_end"
    | "span_start"
    | "span_end"
    | "event";
  id?: string;
  ts?: string;
  name?: string;
  kind?: string;
  status?: string;
  input?: unknown;
  output?: unknown;
  metadata?: Record<string, unknown>;
  tags?: string[];
  parent_id?: string;
  duration_s?: number;
  error?: { type?: string; message?: string };
  data?: Record<string, unknown>;
  stats?: TraceStats;
  thread_id?: string;
}

export interface ParsedSpan {
  id: string;
  parent_id?: string;
  name: string;
  kind?: string;
  status?: string;
  duration_s?: number;
  input?: unknown;
  output?: unknown;
  metadata?: Record<string, unknown>;
  tags: string[];
  error?: { type?: string; message?: string };
  ts_start?: string;
  ts_end?: string;
}

export interface ChildTracer {
  name: string;
  file: string;
  startedTs?: string;
  stats: {
    spans?: number;
    total_input_tokens?: number;
    total_output_tokens?: number;
    duration_s?: number;
  } | null;
  parsed: ParsedTrace | null;
  expanded: boolean;
  fetchError: string | null;
  loading: boolean;
}

export interface ParsedTrace {
  spans: ParsedSpan[];
  logEvents: TraceEvent[];
  traceStart: TraceEvent | null;
  traceEnd: TraceEvent | null;
  childTracers: Map<string, ChildTracer>;
}

export interface AggregatedStats {
  traceCount: number;
  totalSpans: number;
  totalErrors: number;
  totalIn: number;
  totalOut: number;
  totalDuration: number;
  avgDuration: number;
}

export interface FolderItem {
  type: "folder";
  name: string;
  path: string;
  latestTs: string;
  traceCount: number;
  totalSpans: number;
  totalErrors: number;
  totalIn: number;
  totalOut: number;
  totalDuration: number;
  avgDuration: number;
}

/** Reference stub for externalized span input */
export interface ContentRef {
  _ref: string;
  _size: number;
}

/** Shape returned by /api/content?file=...&span_id=... */
export interface ContentResponse {
  span_id: string;
  input: unknown;
  error?: string;
}

export function isContentRef(value: unknown): value is ContentRef {
  return (
    typeof value === "object" &&
    value !== null &&
    "_ref" in value &&
    "_size" in value &&
    typeof (value as ContentRef)._ref === "string" &&
    typeof (value as ContentRef)._size === "number"
  );
}
