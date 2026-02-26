import { useState, useCallback } from "react";
import { JsonViewer } from "./JsonViewer";

interface Props {
  title: string;
  value: unknown;
  defaultOpen?: boolean;
}

export function JsonSection({ title, value, defaultOpen = true }: Props) {
  const [open, setOpen] = useState(defaultOpen);
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      const text =
        typeof value === "string" ? value : JSON.stringify(value, null, 2);
      navigator.clipboard.writeText(text).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1200);
      });
    },
    [value],
  );

  return (
    <div className="mb-6">
      <div
        className="text-sm font-semibold text-text-muted mb-3 cursor-pointer flex items-center gap-2 select-none"
        onClick={() => setOpen((v) => !v)}
      >
        <span
          className={`text-[10px] transition-transform ${open ? "" : "-rotate-90"}`}
        >
          &#9660;
        </span>
        {title}
        <button
          onClick={handleCopy}
          className="ml-auto bg-transparent ring-1 ring-border/50 text-text-dim text-[11px] cursor-pointer px-2.5 py-1 rounded-lg font-mono hover:text-text-muted transition-colors"
        >
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>
      {open && <JsonViewer value={value} />}
    </div>
  );
}
