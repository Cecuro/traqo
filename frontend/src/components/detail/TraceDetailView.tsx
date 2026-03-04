import { useEffect, useMemo, useCallback, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useTrace } from "../../hooks/useTrace";
import { useKeyboard } from "../../hooks/useKeyboard";
import { useTraceDetailState, useTraceDetailDispatch, useTheme } from "../../context";
import { collectErrorSpans } from "../../utils";
import { LoadingOverlay } from "../LoadingOverlay";
import { SummaryBar } from "./SummaryBar";
import { SpanTree } from "./SpanTree";
import { SpanDetail } from "./SpanDetail";

export function TraceDetailView() {
  const { file, spanId: routeSpanId } = useParams<{
    file: string;
    spanId?: string;
  }>();
  const navigate = useNavigate();
  const decodedFile = file ? decodeURIComponent(file) : undefined;
  const parsedTrace = useTrace(decodedFile);
  const { selectedSpanId } = useTraceDetailState();
  const traceDispatch = useTraceDetailDispatch();
  const treeRef = useRef<HTMLDivElement>(null);
  const { toggle: toggleTheme } = useTheme();

  // Dir from file path
  const dir = useMemo(() => {
    if (!decodedFile) return "";
    const parts = decodedFile.split("/");
    return parts.length > 1 ? parts.slice(0, -1).join("/") : "";
  }, [decodedFile]);

  // Auto-select span on load
  useEffect(() => {
    if (!parsedTrace) return;

    if (routeSpanId) {
      traceDispatch({ type: "SELECT_SPAN", spanId: routeSpanId });
      return;
    }

    // Auto-select trace I/O or first root span
    if (
      parsedTrace.traceStart?.input ||
      parsedTrace.traceEnd?.output
    ) {
      traceDispatch({ type: "SELECT_SPAN", spanId: "__trace__" });
    } else if (parsedTrace.spans.length) {
      const roots = parsedTrace.spans
        .filter((s) => !s.parent_id)
        .sort((a, b) =>
          (a.ts_start ?? "") < (b.ts_start ?? "") ? -1 : 1,
        );
      if (roots[0]) {
        traceDispatch({ type: "SELECT_SPAN", spanId: roots[0].id });
      }
    }
  }, [parsedTrace, routeSpanId, traceDispatch]);

  const handleSelectSpan = useCallback(
    (id: string) => {
      traceDispatch({ type: "SELECT_SPAN", spanId: id });
      // Update URL without history push
      if (decodedFile) {
        const newPath = `/trace/${encodeURIComponent(decodedFile)}/${encodeURIComponent(id)}`;
        window.history.replaceState(null, "", `#${newPath}`);
      }
    },
    [traceDispatch, decodedFile],
  );

  // Keyboard navigation for span tree
  const navigateSpans = useCallback(
    (direction: "up" | "down") => {
      if (!treeRef.current) return;
      const nodes = Array.from(
        treeRef.current.querySelectorAll("[data-span-id]"),
      );
      if (!nodes.length) return;

      const idx = nodes.findIndex(
        (n) =>
          (n as HTMLElement).dataset.spanId === selectedSpanId,
      );
      let next: number;
      if (direction === "up") {
        next = idx <= 0 ? nodes.length - 1 : idx - 1;
      } else {
        next = idx >= nodes.length - 1 ? 0 : idx + 1;
      }

      const node = nodes[next] as HTMLElement | undefined;
      if (node?.dataset.spanId) {
        handleSelectSpan(node.dataset.spanId);
        node.scrollIntoView({ block: "nearest" });
      }
    },
    [selectedSpanId, handleSelectSpan],
  );

  // Error navigation
  const errorSpans = useMemo(
    () => (parsedTrace ? collectErrorSpans(parsedTrace) : []),
    [parsedTrace],
  );

  const currentErrorIndex = useMemo(() => {
    if (!selectedSpanId) return -1;
    return errorSpans.findIndex((e) => e.spanId === selectedSpanId);
  }, [selectedSpanId, errorSpans]);

  const handleSelectError = useCallback(
    (spanId: string) => {
      handleSelectSpan(spanId);
      requestAnimationFrame(() => {
        if (!treeRef.current) return;
        const el = treeRef.current.querySelector(`[data-span-id="${CSS.escape(spanId)}"]`);
        el?.scrollIntoView({ block: "nearest", behavior: "smooth" });
      });
    },
    [handleSelectSpan],
  );

  const handleNextError = useCallback(() => {
    if (errorSpans.length === 0) return;
    const nextIndex = currentErrorIndex < errorSpans.length - 1
      ? currentErrorIndex + 1
      : 0;
    const target = errorSpans[nextIndex];
    if (target) handleSelectError(target.spanId);
  }, [errorSpans, currentErrorIndex, handleSelectError]);

  const handlePrevError = useCallback(() => {
    if (errorSpans.length === 0) return;
    const prevIndex = currentErrorIndex > 0
      ? currentErrorIndex - 1
      : errorSpans.length - 1;
    const target = errorSpans[prevIndex];
    if (target) handleSelectError(target.spanId);
  }, [errorSpans, currentErrorIndex, handleSelectError]);

  useKeyboard(
    useMemo(
      () => ({
        onEscape: () => navigate(dir ? `/dir/${dir}` : "/"),
        onUp: () => navigateSpans("up"),
        onDown: () => navigateSpans("down"),
        onToggleTheme: toggleTheme,
        onNextError: handleNextError,
        onPrevError: handlePrevError,
      }),
      [navigate, dir, navigateSpans, toggleTheme, handleNextError, handlePrevError],
    ),
  );

  if (!parsedTrace) {
    return (
      <div className="h-full relative">
        <LoadingOverlay />
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <SummaryBar
        parsedTrace={parsedTrace}
        dir={dir}
        errorSpans={errorSpans}
        onSelectError={handleSelectError}
        onNextError={handleNextError}
        onPrevError={handlePrevError}
        currentErrorIndex={currentErrorIndex}
      />
      <div className="flex flex-1 overflow-hidden max-[900px]:flex-col">
        <div
          ref={treeRef}
          className="w-[500px] min-w-[360px] max-w-[55%] border-r border-border overflow-y-auto shrink-0 bg-bg-surface max-[900px]:w-full max-[900px]:max-w-full max-[900px]:border-r-0 max-[900px]:border-b max-[900px]:border-border max-[900px]:max-h-[40vh]"
        >
          <SpanTree
            parsedTrace={parsedTrace}
            selectedSpanId={selectedSpanId}
            onSelect={handleSelectSpan}
          />
        </div>
        <div className="flex-1 overflow-y-auto py-6 px-8">
          <SpanDetail
            spanId={selectedSpanId}
            parsedTrace={parsedTrace}
            fileKey={decodedFile}
          />
        </div>
      </div>
    </div>
  );
}
