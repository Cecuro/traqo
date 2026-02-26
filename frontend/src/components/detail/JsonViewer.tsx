/** Syntax-highlighted JSON renderer */
export function JsonViewer({ value }: { value: unknown }) {
  if (typeof value === "string") {
    // Multi-line or long strings: show as plain text
    if (value.includes("\n") || value.length > 100) {
      return (
        <div className="bg-bg-card ring-1 ring-border/30 rounded-xl p-5 font-mono text-[13px] leading-relaxed overflow-x-auto whitespace-pre-wrap break-words max-h-[500px] overflow-y-auto">
          {value}
        </div>
      );
    }
    return (
      <div className="bg-bg-card ring-1 ring-border/30 rounded-xl p-5 font-mono text-[13px] leading-relaxed overflow-x-auto whitespace-pre-wrap break-words max-h-[500px] overflow-y-auto">
        <span className="text-ok">"{value}"</span>
      </div>
    );
  }

  return (
    <div className="bg-bg-card ring-1 ring-border/30 rounded-xl p-5 font-mono text-[13px] leading-relaxed overflow-x-auto whitespace-pre-wrap break-words max-h-[500px] overflow-y-auto">
      <Highlight value={value} indent={0} />
    </div>
  );
}

function Highlight({ value, indent }: { value: unknown; indent: number }) {
  if (value === null || value === undefined)
    return <span className="text-text-dim">null</span>;
  if (typeof value === "boolean")
    return <span className="text-purple">{String(value)}</span>;
  if (typeof value === "number")
    return <span className="text-orange">{value}</span>;
  if (typeof value === "string") {
    const display =
      value.length > 2000 ? value.slice(0, 2000) + "\u2026" : value;
    const escaped = display.replace(/\n/g, "\n" + " ".repeat(indent + 2));
    return <span className="text-ok">"{escaped}"</span>;
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return <>{"[]"}</>;
    const pad = " ".repeat(indent + 2);
    const end = " ".repeat(indent);
    return (
      <>
        {"[\n"}
        {value.map((item, i) => (
          <span key={i}>
            {pad}
            <Highlight value={item} indent={indent + 2} />
            {i < value.length - 1 ? ",\n" : "\n"}
          </span>
        ))}
        {end}
        {"]"}
      </>
    );
  }
  if (typeof value === "object") {
    const keys = Object.keys(value as Record<string, unknown>);
    if (keys.length === 0) return <>{"{ }"}</>;
    const pad = " ".repeat(indent + 2);
    const end = " ".repeat(indent);
    const obj = value as Record<string, unknown>;
    return (
      <>
        {"{\n"}
        {keys.map((k, i) => (
          <span key={k}>
            {pad}
            <span className="text-accent">"{k}"</span>
            {": "}
            <Highlight value={obj[k]} indent={indent + 2} />
            {i < keys.length - 1 ? ",\n" : "\n"}
          </span>
        ))}
        {end}
        {"}"}
      </>
    );
  }
  return <>{String(value)}</>;
}
