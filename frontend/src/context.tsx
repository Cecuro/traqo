import {
  createContext,
  useContext,
  useReducer,
  useState,
  useEffect,
  useCallback,
  type Dispatch,
  type ReactNode,
} from "react";
import type { TraceSummary, ParsedTrace } from "./types";

// ── Theme ──
type Theme = "light" | "dark" | "system";

interface ThemeCtxValue {
  theme: Theme;
  resolved: "light" | "dark";
  setTheme: (t: Theme) => void;
  toggle: () => void;
}

const ThemeCtx = createContext<ThemeCtxValue>({
  theme: "system",
  resolved: "light",
  setTheme: () => {},
  toggle: () => {},
});

function resolveTheme(theme: Theme): "light" | "dark" {
  if (theme !== "system") return theme;
  if (typeof window === "undefined") return "light";
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(() => {
    try {
      return (localStorage.getItem("traqo-theme") as Theme) ?? "system";
    } catch {
      return "system";
    }
  });
  const [resolved, setResolved] = useState<"light" | "dark">(() =>
    resolveTheme(theme),
  );

  const applyResolved = useCallback(
    (t: Theme) => {
      const r = resolveTheme(t);
      setResolved(r);
      document.documentElement.classList.toggle("dark", r === "dark");
    },
    [],
  );

  const setTheme = useCallback(
    (t: Theme) => {
      setThemeState(t);
      try {
        localStorage.setItem("traqo-theme", t);
      } catch {}
      applyResolved(t);
    },
    [applyResolved],
  );

  const toggle = useCallback(() => {
    setTheme(resolved === "dark" ? "light" : "dark");
  }, [resolved, setTheme]);

  // Apply on mount
  useEffect(() => {
    applyResolved(theme);
  }, [applyResolved, theme]);

  // Listen for system preference changes when in "system" mode
  useEffect(() => {
    if (theme !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => applyResolved("system");
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, [theme, applyResolved]);

  return (
    <ThemeCtx.Provider value={{ theme, resolved, setTheme, toggle }}>
      {children}
    </ThemeCtx.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeCtx);
}

export interface AppState {
  traces: TraceSummary[];
  loading: boolean;
  error: string | null;
  statusFilter: "all" | "ok" | "error";
  selectedTags: Set<string>;
  searchQuery: string;
}

export type AppAction =
  | { type: "SET_TRACES"; traces: TraceSummary[] }
  | { type: "SET_LOADING"; loading: boolean }
  | { type: "SET_ERROR"; error: string | null }
  | { type: "SET_STATUS_FILTER"; filter: "all" | "ok" | "error" }
  | { type: "TOGGLE_TAG"; tag: string }
  | { type: "SET_SEARCH"; query: string }
  | { type: "CLEAR_FILTERS" };

const initialState: AppState = {
  traces: [],
  loading: false,
  error: null,
  statusFilter: "all",
  selectedTags: new Set(),
  searchQuery: "",
};

function reducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case "SET_TRACES":
      return { ...state, traces: action.traces };
    case "SET_LOADING":
      return { ...state, loading: action.loading };
    case "SET_ERROR":
      return { ...state, error: action.error };
    case "SET_STATUS_FILTER":
      return { ...state, statusFilter: action.filter };
    case "TOGGLE_TAG": {
      const next = new Set(state.selectedTags);
      if (next.has(action.tag)) next.delete(action.tag);
      else next.add(action.tag);
      return { ...state, selectedTags: next };
    }
    case "SET_SEARCH":
      return { ...state, searchQuery: action.query };
    case "CLEAR_FILTERS":
      return {
        ...state,
        statusFilter: "all",
        selectedTags: new Set(),
        searchQuery: "",
      };
    default:
      return state;
  }
}

const AppStateCtx = createContext<AppState>(initialState);
const AppDispatchCtx = createContext<Dispatch<AppAction>>(() => {});

// Separate context for parsed trace to avoid re-renders on browse view
export interface TraceDetailState {
  parsedTrace: ParsedTrace | null;
  selectedSpanId: string | null;
}

export type TraceDetailAction =
  | { type: "SET_PARSED_TRACE"; parsed: ParsedTrace | null }
  | { type: "SELECT_SPAN"; spanId: string | null }
  | { type: "UPDATE_CHILD"; name: string; update: (child: ParsedTrace["childTracers"] extends Map<string, infer V> ? V : never) => void };

const traceDetailInitial: TraceDetailState = {
  parsedTrace: null,
  selectedSpanId: null,
};

function traceDetailReducer(
  state: TraceDetailState,
  action: TraceDetailAction,
): TraceDetailState {
  switch (action.type) {
    case "SET_PARSED_TRACE":
      return { ...state, parsedTrace: action.parsed, selectedSpanId: null };
    case "SELECT_SPAN":
      return { ...state, selectedSpanId: action.spanId };
    case "UPDATE_CHILD": {
      if (!state.parsedTrace) return state;
      const child = state.parsedTrace.childTracers.get(action.name);
      if (!child) return state;
      action.update(child);
      // Force re-render by creating a new map
      const newTracers = new Map(state.parsedTrace.childTracers);
      return {
        ...state,
        parsedTrace: { ...state.parsedTrace, childTracers: newTracers },
      };
    }
    default:
      return state;
  }
}

const TraceDetailStateCtx =
  createContext<TraceDetailState>(traceDetailInitial);
const TraceDetailDispatchCtx = createContext<Dispatch<TraceDetailAction>>(
  () => {},
);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [traceState, traceDispatch] = useReducer(
    traceDetailReducer,
    traceDetailInitial,
  );

  return (
    <AppStateCtx.Provider value={state}>
      <AppDispatchCtx.Provider value={dispatch}>
        <TraceDetailStateCtx.Provider value={traceState}>
          <TraceDetailDispatchCtx.Provider value={traceDispatch}>
            {children}
          </TraceDetailDispatchCtx.Provider>
        </TraceDetailStateCtx.Provider>
      </AppDispatchCtx.Provider>
    </AppStateCtx.Provider>
  );
}

export function useAppState() {
  return useContext(AppStateCtx);
}
export function useAppDispatch() {
  return useContext(AppDispatchCtx);
}
export function useTraceDetailState() {
  return useContext(TraceDetailStateCtx);
}
export function useTraceDetailDispatch() {
  return useContext(TraceDetailDispatchCtx);
}
