import { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useTheme } from "../context";
import { Breadcrumb } from "./Breadcrumb";
import { HelpOverlay } from "./HelpOverlay";
import { SunIcon, MoonIcon } from "../icons";

export function Header() {
  const navigate = useNavigate();
  const location = useLocation();
  const [helpOpen, setHelpOpen] = useState(false);
  const { resolved, toggle } = useTheme();
  const isDetail = location.pathname.startsWith("/trace/");

  return (
    <>
      <header className="flex items-center gap-3 px-6 py-2.5 shrink-0 bg-bg-card shadow-sm shadow-black/8">
        <span
          className="font-mono text-lg font-bold text-accent tracking-tight cursor-pointer"
          onClick={() => navigate("/")}
        >
          traqo
        </span>
        {isDetail && (
          <button
            onClick={() => navigate(-1)}
            className="bg-transparent border-none text-text-muted w-8 h-8 rounded-lg text-base cursor-pointer flex items-center justify-center hover:bg-bg-hover hover:text-text transition-colors"
            title="Back to traces"
          >
            &larr;
          </button>
        )}
        <Breadcrumb />
        <div className="ml-auto flex items-center gap-1">
          <button
            onClick={toggle}
            className="bg-transparent border-none text-text-muted w-8 h-8 rounded-lg cursor-pointer flex items-center justify-center hover:bg-bg-hover hover:text-text transition-colors"
            title={`Switch to ${resolved === "dark" ? "light" : "dark"} theme (t)`}
          >
            {resolved === "dark" ? (
              <SunIcon className="w-4 h-4" />
            ) : (
              <MoonIcon className="w-4 h-4" />
            )}
          </button>
          <button
            onClick={() => setHelpOpen((v) => !v)}
            className="bg-transparent border-none text-text-muted w-8 h-8 rounded-lg text-sm cursor-pointer flex items-center justify-center font-semibold hover:bg-bg-hover hover:text-text transition-colors"
            title="Help (?)"
          >
            ?
          </button>
        </div>
      </header>
      {helpOpen && <HelpOverlay onClose={() => setHelpOpen(false)} />}
    </>
  );
}
