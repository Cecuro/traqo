import type { TraceSummary, TraceResponse } from "./types";

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
