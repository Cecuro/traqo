import { useMemo } from "react";
import { useAppState, useAppDispatch } from "../../context";
import { tracesInDir } from "../../utils";
import { Dropdown } from "./Dropdown";

interface Props {
  dir: string;
}

export function FilterBar({ dir }: Props) {
  const { traces, statusFilter, selectedTags, searchQuery } = useAppState();
  const dispatch = useAppDispatch();

  const scoped = useMemo(
    () => (dir ? tracesInDir(traces, dir) : traces),
    [traces, dir],
  );

  const tagItems = useMemo(() => {
    const counts = new Map<string, number>();
    for (const t of scoped) {
      for (const tag of t.tags ?? []) {
        counts.set(tag, (counts.get(tag) ?? 0) + 1);
      }
    }
    return [...counts.entries()]
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .map(([tag, count]) => ({ value: tag, label: tag, count }));
  }, [scoped]);

  const hasFilters =
    searchQuery || statusFilter !== "all" || selectedTags.size > 0;

  return (
    <div className="flex gap-3 mb-3 flex-wrap items-center">
      <input
        type="text"
        value={searchQuery}
        onChange={(e) =>
          dispatch({ type: "SET_SEARCH", query: e.target.value })
        }
        placeholder="Search by name, tag, thread ID..."
        className="bg-bg-card ring-1 ring-border/50 text-text px-3.5 py-2 rounded-xl text-sm outline-none min-w-60 focus:ring-accent transition-shadow"
      />
      <Dropdown
        items={[
          { value: "all", label: "All statuses" },
          { value: "ok", label: "OK only" },
          { value: "error", label: "Errors only" },
        ]}
        selected={statusFilter}
        onSelect={(v) =>
          dispatch({
            type: "SET_STATUS_FILTER",
            filter: v as "all" | "ok" | "error",
          })
        }
        label={
          statusFilter === "all"
            ? "All statuses"
            : statusFilter === "ok"
              ? "OK only"
              : "Errors only"
        }
      />
      <Dropdown
        items={tagItems}
        selected={selectedTags}
        onSelect={(v) => dispatch({ type: "TOGGLE_TAG", tag: v })}
        label="All tags"
        multi
      />
      {hasFilters && (
        <button
          onClick={() => dispatch({ type: "CLEAR_FILTERS" })}
          className="bg-transparent ring-1 ring-border/50 text-text-muted px-3.5 py-2 rounded-xl text-sm cursor-pointer flex items-center gap-1.5 hover:ring-accent hover:text-accent transition-all"
        >
          &times; Clear
        </button>
      )}
    </div>
  );
}
