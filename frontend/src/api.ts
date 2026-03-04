import type { TraceSummary, TraceResponse, ContentResponse } from "./types";

export async function fetchTraces(): Promise<TraceSummary[]> {
  const res = await fetch("/api/traces");
  if (!res.ok) throw new Error(`Server error ${res.status}`);
  return res.json();
}

export async function fetchTrace(file: string): Promise<TraceResponse> {
  const res = await fetch(
    `/api/trace?file=${encodeURIComponent(file)}`,
  );
  if (!res.ok) throw new Error(`Server error ${res.status}`);
  const data: TraceResponse = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

export async function fetchContent(
  file: string,
  spanId: string,
): Promise<ContentResponse> {
  const res = await fetch(
    `/api/content?file=${encodeURIComponent(file)}&span_id=${encodeURIComponent(spanId)}`,
  );
  if (!res.ok) throw new Error(`Server error ${res.status}`);
  const data: ContentResponse = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}
