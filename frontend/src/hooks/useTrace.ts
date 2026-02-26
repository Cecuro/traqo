import { useEffect, useRef } from "react";
import { fetchTrace } from "../api";
import { parseEvents } from "../utils";
import {
  useAppDispatch,
  useTraceDetailState,
  useTraceDetailDispatch,
} from "../context";

export function useTrace(file: string | undefined) {
  const appDispatch = useAppDispatch();
  const { parsedTrace } = useTraceDetailState();
  const traceDispatch = useTraceDetailDispatch();
  const loadedFile = useRef<string | null>(null);

  useEffect(() => {
    if (!file || file === loadedFile.current) return;
    loadedFile.current = file;

    let cancelled = false;
    appDispatch({ type: "SET_LOADING", loading: true });
    appDispatch({ type: "SET_ERROR", error: null });

    fetchTrace(file)
      .then((data) => {
        if (cancelled) return;
        const parsed = parseEvents(data.events ?? [], file);
        traceDispatch({ type: "SET_PARSED_TRACE", parsed });
      })
      .catch((e) => {
        if (cancelled) return;
        appDispatch({
          type: "SET_ERROR",
          error:
            "Failed to load trace: " +
            (e instanceof Error ? e.message : String(e)),
        });
      })
      .finally(() => {
        if (!cancelled) appDispatch({ type: "SET_LOADING", loading: false });
      });

    return () => {
      cancelled = true;
      loadedFile.current = null;
    };
  }, [file, appDispatch, traceDispatch]);

  // Clear parsed trace when leaving
  useEffect(() => {
    return () => {
      traceDispatch({ type: "SET_PARSED_TRACE", parsed: null });
    };
  }, [traceDispatch]);

  return parsedTrace;
}
