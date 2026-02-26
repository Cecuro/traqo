import { useAppState, useAppDispatch } from "../context";

export function ErrorBanner() {
  const { error } = useAppState();
  const dispatch = useAppDispatch();

  if (!error) return null;

  return (
    <div className="px-4 py-2.5 bg-err-dim border-b border-err/25 text-err text-[13px] flex items-center gap-3 shrink-0">
      <span>{error}</span>
      <button
        onClick={() => dispatch({ type: "SET_ERROR", error: null })}
        className="ml-auto bg-transparent border-none text-err text-lg cursor-pointer"
      >
        &times;
      </button>
    </div>
  );
}
