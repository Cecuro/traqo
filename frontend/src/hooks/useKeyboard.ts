import { useEffect } from "react";

interface KeyboardOptions {
  onEscape?: () => void;
  onHelp?: () => void;
  onRefresh?: () => void;
  onUp?: () => void;
  onDown?: () => void;
  onToggleTheme?: () => void;
}

export function useKeyboard(opts: KeyboardOptions) {
  useEffect(() => {
    function handler(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;

      if (e.key === "Escape") {
        opts.onEscape?.();
        return;
      }
      if (e.key === "?") {
        e.preventDefault();
        opts.onHelp?.();
        return;
      }
      if (e.key === "r") {
        e.preventDefault();
        opts.onRefresh?.();
        return;
      }
      if (e.key === "t") {
        e.preventDefault();
        opts.onToggleTheme?.();
        return;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        opts.onUp?.();
      }
      if (e.key === "ArrowDown") {
        e.preventDefault();
        opts.onDown?.();
      }
    }

    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [opts]);
}
