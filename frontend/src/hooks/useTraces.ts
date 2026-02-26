import { useEffect, useCallback } from "react";
import { fetchTraces } from "../api";
import { useAppState, useAppDispatch } from "../context";

export function useTraces() {
  const { traces } = useAppState();
  const dispatch = useAppDispatch();

  const load = useCallback(async () => {
    dispatch({ type: "SET_LOADING", loading: true });
    try {
      const data = await fetchTraces();
      dispatch({ type: "SET_TRACES", traces: data });
    } catch (e) {
      dispatch({
        type: "SET_ERROR",
        error:
          "Failed to load traces: " +
          (e instanceof Error ? e.message : String(e)),
      });
      dispatch({ type: "SET_TRACES", traces: [] });
    } finally {
      dispatch({ type: "SET_LOADING", loading: false });
    }
  }, [dispatch]);

  useEffect(() => {
    load();
  }, [load]);

  return { traces, refresh: load };
}
