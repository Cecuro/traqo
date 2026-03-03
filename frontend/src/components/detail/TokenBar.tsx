import { fmtN } from "../../utils";

interface Props {
  inputTokens: number;
  outputTokens: number;
  reasoningTokens?: number;
  cacheReadTokens?: number;
  cacheCreationTokens?: number;
}

export function TokenBar({
  inputTokens,
  outputTokens,
  reasoningTokens,
  cacheReadTokens,
  cacheCreationTokens,
}: Props) {
  const total = inputTokens + outputTokens || 1;
  const inPct = ((inputTokens / total) * 100).toFixed(1);
  const outPct = ((outputTokens / total) * 100).toFixed(1);

  return (
    <div className="mb-6">
      <div className="text-sm font-semibold text-text-muted mb-3">
        Token Usage
      </div>
      <div className="font-mono text-sm mb-2.5">
        <span className="text-blue">{fmtN(inputTokens)} input</span>
        {" / "}
        <span className="text-orange">{fmtN(outputTokens)} output</span>
      </div>
      <div className="flex h-3 rounded-full overflow-hidden bg-border/40">
        <div className="bg-blue rounded-l-full" style={{ width: `${inPct}%` }} />
        <div className="bg-orange rounded-r-full" style={{ width: `${outPct}%` }} />
      </div>
      {reasoningTokens != null && reasoningTokens > 0 && (
        <div className="mt-2 text-sm text-text-muted">
          Reasoning: {fmtN(reasoningTokens)}
        </div>
      )}
      {cacheReadTokens != null && cacheReadTokens > 0 && (
        <div className="mt-1 text-sm text-text-muted">
          Cache read: {fmtN(cacheReadTokens)}
        </div>
      )}
      {cacheCreationTokens != null && cacheCreationTokens > 0 && (
        <div className="mt-1 text-sm text-text-muted">
          Cache creation: {fmtN(cacheCreationTokens)}
        </div>
      )}
    </div>
  );
}
