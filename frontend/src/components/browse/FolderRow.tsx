import { useNavigate } from "react-router-dom";
import type { FolderItem } from "../../types";
import { FolderIcon } from "../../icons";
import { fmtN, fmtDur, fmtTime } from "../../utils";

export function FolderRow({ folder }: { folder: FolderItem }) {
  const navigate = useNavigate();
  const hasErr = folder.totalErrors > 0;

  return (
    <div
      className="flex items-center gap-3 px-4 py-3 border-b border-border/50 cursor-pointer transition-colors hover:bg-bg-hover"
      onClick={() => navigate(`/dir/${folder.path}`)}
    >
      <div className="w-8 h-8 rounded-lg bg-accent/10 text-accent flex items-center justify-center shrink-0">
        <FolderIcon className="w-[18px] h-[18px]" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-[15px] font-medium truncate">{folder.name}</div>
        <div className="text-xs text-text-muted mt-0.5 truncate">
          {folder.traceCount} trace{folder.traceCount !== 1 ? "s" : ""}
          {folder.latestTs ? ` · latest ${fmtTime(folder.latestTs)}` : ""}
        </div>
      </div>
      <div className="hidden sm:flex items-center gap-4 shrink-0">
        <MetaItem>
          <span className="font-mono text-[13px] font-medium">
            <span className="text-blue">{fmtN(folder.totalIn)}</span>
            {" / "}
            <span className="text-orange">{fmtN(folder.totalOut)}</span>
          </span>
          <span className="text-[10px] text-text-dim uppercase tracking-wide">
            tokens
          </span>
        </MetaItem>
        <MetaItem>
          <span className="font-mono text-[13px] font-medium">
            {fmtDur(folder.avgDuration)}
          </span>
          <span className="text-[10px] text-text-dim uppercase tracking-wide">
            avg
          </span>
        </MetaItem>
        <MetaItem>
          {hasErr ? (
            <span className="text-err text-[13px] font-medium font-mono">
              {folder.totalErrors}
            </span>
          ) : (
            <span className="inline-block px-[7px] py-0.5 rounded bg-ok-dim text-ok text-[11px] font-medium font-mono">
              ok
            </span>
          )}
          {hasErr && (
            <span className="text-[10px] text-text-dim uppercase tracking-wide">
              errors
            </span>
          )}
        </MetaItem>
      </div>
      <div className="text-text-dim text-base shrink-0 ml-1">&rsaquo;</div>
    </div>
  );
}

function MetaItem({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex flex-col items-end min-w-[70px]">{children}</div>
  );
}
