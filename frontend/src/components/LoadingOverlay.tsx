import { useAppState } from "../context";

export function LoadingOverlay() {
  const { loading } = useAppState();

  if (!loading) return null;

  return (
    <div className="absolute inset-0 flex items-center justify-center bg-bg/70 z-50">
      <div className="w-7 h-7 border-3 border-border border-t-accent rounded-full animate-spin" style={{ animation: "spin 0.7s linear infinite" }} />
    </div>
  );
}
