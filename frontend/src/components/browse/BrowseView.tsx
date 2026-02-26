import { useMemo } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useAppState, useTheme } from "../../context";
import { useTraces } from "../../hooks/useTraces";
import { useKeyboard } from "../../hooks/useKeyboard";
import {
  tracesInDir,
  aggregateStats,
  itemsAtDir,
} from "../../utils";
import { LoadingOverlay } from "../LoadingOverlay";
import { SummaryStrip } from "./SummaryStrip";
import { FilterBar } from "./FilterBar";
import { FolderRow } from "./FolderRow";
import { TraceRow } from "./TraceRow";
import { FolderIcon } from "../../icons";

export function BrowseView() {
  const location = useLocation();
  const navigate = useNavigate();
  const { traces, statusFilter, selectedTags, searchQuery } = useAppState();
  const { refresh } = useTraces();
  const { toggle: toggleTheme } = useTheme();

  // Extract dir from route: /dir/some/path → "some/path", / → ""
  const dir = location.pathname.startsWith("/dir/")
    ? location.pathname.slice(5)
    : "";

  const scoped = useMemo(
    () => (dir ? tracesInDir(traces, dir) : traces),
    [traces, dir],
  );

  const filtered = useMemo(() => {
    const search = searchQuery.toLowerCase();
    return scoped.filter((t) => {
      if (search) {
        const hay = [
          t.file,
          t.thread_id ?? "",
          ...(t.tags ?? []),
          JSON.stringify(t.input ?? ""),
        ]
          .join(" ")
          .toLowerCase();
        if (!hay.includes(search)) return false;
      }
      if (
        selectedTags.size &&
        !(t.tags ?? []).some((tag) => selectedTags.has(tag))
      )
        return false;
      if (statusFilter === "ok" && (t.stats?.errors ?? 0) > 0) return false;
      if (statusFilter === "error" && (t.stats?.errors ?? 0) === 0)
        return false;
      return true;
    });
  }, [scoped, searchQuery, selectedTags, statusFilter]);

  const items = useMemo(
    () => itemsAtDir(dir, filtered, traces),
    [dir, filtered, traces],
  );

  const aggStats = useMemo(() => aggregateStats(scoped), [scoped]);

  const sorted = useMemo(
    () =>
      [...items.traces].sort((a, b) =>
        (b.ts ?? "") > (a.ts ?? "") ? 1 : -1,
      ),
    [items.traces],
  );

  useKeyboard(
    useMemo(
      () => ({
        onRefresh: refresh,
        onToggleTheme: toggleTheme,
        onEscape: () => {
          if (dir) {
            const parent = dir.includes("/")
              ? "/dir/" + dir.slice(0, dir.lastIndexOf("/"))
              : "/";
            navigate(parent);
          }
        },
      }),
      [refresh, toggleTheme, dir, navigate],
    ),
  );

  const folderName = dir ? dir.split("/").pop() : "";
  const parentDir = dir.includes("/")
    ? "/dir/" + dir.slice(0, dir.lastIndexOf("/"))
    : "/";

  return (
    <div className="max-w-6xl mx-auto px-6 py-6 relative">
      <LoadingOverlay />

      {dir && (
        <div className="flex items-center gap-3 mb-4">
          <button
            className="bg-transparent border-none text-text-muted w-8 h-8 rounded-lg text-lg cursor-pointer flex items-center justify-center hover:bg-bg-hover hover:text-text transition-colors"
            onClick={() => navigate(parentDir)}
            title="Go back"
          >
            &larr;
          </button>
          <div className="w-8 h-8 rounded-lg bg-accent/10 text-accent flex items-center justify-center shrink-0">
            <FolderIcon className="w-4 h-4" />
          </div>
          <div className="text-lg font-semibold">{folderName}</div>
        </div>
      )}

      <SummaryStrip stats={aggStats} />
      <FilterBar dir={dir} />

      {filtered.length < scoped.length && (
        <div className="text-xs text-text-dim mb-2.5">
          Showing {filtered.length} of {scoped.length} traces
        </div>
      )}

      {items.dirs.length === 0 && sorted.length === 0 ? (
        <div className="flex flex-col items-center justify-center text-text-dim gap-2 py-12">
          <div className="text-sm">No traces found</div>
        </div>
      ) : (
        <div className="rounded-xl shadow-sm ring-1 ring-border/50 overflow-hidden">
          {items.dirs.map((d) => (
            <FolderRow key={d.path} folder={d} />
          ))}
          {sorted.map((t) => (
            <TraceRow key={t.file} trace={t} />
          ))}
        </div>
      )}

      <footer className="py-4 text-center text-xs text-text-dim">
        Made with &hearts; &middot;{" "}
        <a
          href="https://github.com/Cecuro/traqo"
          target="_blank"
          rel="noopener"
          className="text-text-muted no-underline hover:text-text"
        >
          GitHub
        </a>
      </footer>
    </div>
  );
}
