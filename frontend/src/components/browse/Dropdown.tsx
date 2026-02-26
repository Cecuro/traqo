import { useState, useRef, useEffect } from "react";

interface DropdownItem {
  value: string;
  label: string;
  count?: number;
}

interface Props {
  items: DropdownItem[];
  selected: string | Set<string>;
  onSelect: (value: string) => void;
  label: string;
  multi?: boolean;
}

export function Dropdown({ items, selected, onSelect, label, multi }: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("click", handleClick);
    return () => document.removeEventListener("click", handleClick);
  }, []);

  const isSelected = (v: string) =>
    selected instanceof Set ? selected.has(v) : selected === v;

  const selCount = selected instanceof Set ? selected.size : 0;

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((v) => !v)}
        className={`bg-bg-card ring-1 ring-border/50 text-text-muted px-3.5 py-2 rounded-xl text-sm cursor-pointer whitespace-nowrap flex items-center gap-1.5 transition-all hover:ring-accent hover:text-text ${open ? "ring-accent text-text" : ""}`}
      >
        {label}
        {multi && selCount > 0 && (
          <span className="bg-accent text-white text-[10px] font-semibold px-1.5 rounded-lg leading-4 min-w-[16px] text-center">
            {selCount}
          </span>
        )}
        <span className="text-[10px] text-text-dim">&#9662;</span>
      </button>
      {open && (
        <div className="absolute top-full mt-1.5 left-0 bg-bg-card ring-1 ring-border/50 rounded-xl min-w-[180px] shadow-xl z-20 py-1.5 max-h-60 overflow-y-auto">
          {items.map((item) => (
            <div
              key={item.value}
              className={`px-3.5 py-2 text-sm cursor-pointer text-text-muted flex items-center gap-2 whitespace-nowrap transition-colors hover:bg-bg-hover hover:text-text ${isSelected(item.value) ? "text-accent" : ""}`}
              onClick={() => {
                onSelect(item.value);
                if (!multi) setOpen(false);
              }}
            >
              <span className="w-3.5 text-xs shrink-0 text-center">
                {isSelected(item.value) ? "\u2713" : ""}
              </span>
              {item.label}
              {item.count != null && (
                <span className="text-text-dim ml-auto">{item.count}</span>
              )}
            </div>
          ))}
          {items.length === 0 && (
            <div className="px-3.5 py-2 text-sm text-text-dim">
              None
            </div>
          )}
        </div>
      )}
    </div>
  );
}
