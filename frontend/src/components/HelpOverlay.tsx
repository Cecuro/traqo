interface Props {
  onClose: () => void;
}

export function HelpOverlay({ onClose }: Props) {
  return (
    <div
      className="fixed inset-0 bg-black/40 z-[100] flex items-center justify-center"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="bg-bg-card shadow-2xl ring-1 ring-border/50 rounded-2xl p-8 max-w-[540px] w-[90%] max-h-[80vh] overflow-y-auto relative">
        <button
          onClick={onClose}
          className="absolute top-5 right-5 bg-transparent border-none text-text-dim text-xl cursor-pointer leading-none hover:text-text"
        >
          &times;
        </button>

        <h2 className="text-lg font-semibold mb-4">Welcome to traqo</h2>
        <p className="text-sm text-text-muted leading-relaxed mb-1.5">
          traqo records what your LLM application does — every API call, tool
          use, and decision — as structured trace files.
        </p>

        <Section title="Navigating">
          <p>
            <strong>Folders</strong> group related traces. Click a folder to see
            what's inside. Use the breadcrumb path at the top to go back.
          </p>
          <p>
            <strong>Traces</strong> are individual runs of your application.
            Click one to see every step it took.
          </p>
        </Section>

        <Section title="Inside a trace">
          <p>
            The left panel shows the <strong>span tree</strong> — a timeline of
            operations your app performed. Each span is one step (an LLM call, a
            tool invocation, etc).
          </p>
          <p>
            Click any span to see its <strong>input</strong>,{" "}
            <strong>output</strong>, <strong>token usage</strong>, and{" "}
            <strong>timing</strong> in the right panel.
          </p>
        </Section>

        <Section title="What the colors mean">
          <ShortcutRow keyText="llm" keyColor="text-blue">
            LLM / model call
          </ShortcutRow>
          <ShortcutRow keyText="tool" keyColor="text-purple">
            Tool or function call
          </ShortcutRow>
          <ShortcutRow keyText="ok" keyColor="text-ok">
            Completed successfully
          </ShortcutRow>
          <ShortcutRow keyText="error" keyColor="text-err">
            Something went wrong
          </ShortcutRow>
        </Section>

        <Section title="What the numbers mean">
          <ShortcutRow
            keyText={
              <>
                <span className="text-blue">in</span> /{" "}
                <span className="text-orange">out</span>
              </>
            }
          >
            Input / output tokens (LLM usage)
          </ShortcutRow>
          <ShortcutRow keyText="spans">
            Number of operations in a trace
          </ShortcutRow>
          <ShortcutRow keyText="duration">Wall-clock time</ShortcutRow>
        </Section>

        <Section title="Keyboard shortcuts">
          <ShortcutRow keyText="Esc">Go back / close dialog</ShortcutRow>
          <ShortcutRow keyText="?">Toggle this help guide</ShortcutRow>
          <ShortcutRow keyText="t">Toggle light / dark theme</ShortcutRow>
          <ShortcutRow keyText="↑ / ↓">
            Navigate spans in trace view
          </ShortcutRow>
          <ShortcutRow keyText="r">Refresh trace list</ShortcutRow>
          <ShortcutRow keyText="e / E">Next / previous error</ShortcutRow>
        </Section>

        <Section title="Tips">
          <p>
            Use <strong>Search</strong> to filter by name, tags, or thread ID.
            Use <strong>Status</strong> to show only errors.
          </p>
          <p>
            Organize traces into folders by choosing your file path:{" "}
            <code className="bg-bg-surface px-1.5 py-0.5 rounded text-xs font-mono text-text">
              Tracer("my-agent", trace_dir="traces/")
            </code>
          </p>
        </Section>
      </div>
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <>
      <h3 className="text-[13px] font-semibold text-accent mt-5 mb-2 uppercase tracking-wide">
        {title}
      </h3>
      <div className="text-sm text-text-muted leading-relaxed [&>p]:mb-1.5">
        {children}
      </div>
    </>
  );
}

function ShortcutRow({
  keyText,
  keyColor,
  children,
}: {
  keyText: React.ReactNode;
  keyColor?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex gap-3 items-center py-1.5 text-sm">
      <span
        className={`bg-bg-surface ring-1 ring-border/50 px-2 py-0.5 rounded-md font-mono text-[11px] text-text-muted min-w-[80px] text-center ${keyColor ?? ""}`}
      >
        {keyText}
      </span>
      <span>{children}</span>
    </div>
  );
}
