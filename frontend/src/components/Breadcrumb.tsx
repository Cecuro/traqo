import { useLocation, useNavigate } from "react-router-dom";

export function Breadcrumb() {
  const location = useLocation();
  const navigate = useNavigate();

  const parts = buildBreadcrumb(location.pathname);

  return (
    <nav className="text-sm text-text-muted flex items-center gap-1.5">
      {parts.map((p, i) => (
        <span key={i} className="flex items-center gap-1.5">
          {i > 0 && <span className="text-text-dim/60">/</span>}
          {p.link ? (
            <a
              className="text-text-muted no-underline cursor-pointer hover:text-text"
              onClick={(e) => {
                e.preventDefault();
                navigate(p.link!);
              }}
            >
              {p.label}
            </a>
          ) : (
            <span className="text-text">{p.label}</span>
          )}
        </span>
      ))}
    </nav>
  );
}

interface Crumb {
  label: string;
  link: string | null;
}

function buildBreadcrumb(pathname: string): Crumb[] {
  const crumbs: Crumb[] = [];

  if (pathname === "/") {
    crumbs.push({ label: "traces", link: null });
    return crumbs;
  }

  if (pathname.startsWith("/dir/")) {
    const dirPath = pathname.slice(5); // after "/dir/"
    crumbs.push({ label: "traces", link: "/" });
    const segments = dirPath.split("/").filter(Boolean);
    segments.forEach((seg, i) => {
      const isLast = i === segments.length - 1;
      if (isLast) {
        crumbs.push({ label: seg, link: null });
      } else {
        crumbs.push({
          label: seg,
          link: "/dir/" + segments.slice(0, i + 1).join("/"),
        });
      }
    });
    return crumbs;
  }

  if (pathname.startsWith("/trace/")) {
    const rest = pathname.slice(7); // after "/trace/"
    // rest could be "file" or "file/spanId"
    const decoded = decodeURIComponent(rest);
    // The file part may contain slashes (directory structure)
    // We need the file name for display
    const fileParts = decoded.split("/");
    const fileName = fileParts[fileParts.length - 1]!.replace(/\.jsonl(?:\.gz)?$/, "");

    crumbs.push({ label: "traces", link: "/" });

    // If file was in a subdirectory, show dir crumbs
    if (fileParts.length > 1) {
      for (let i = 0; i < fileParts.length - 1; i++) {
        crumbs.push({
          label: fileParts[i]!,
          link: "/dir/" + fileParts.slice(0, i + 1).join("/"),
        });
      }
    }

    crumbs.push({ label: fileName, link: null });
    return crumbs;
  }

  crumbs.push({ label: "traces", link: null });
  return crumbs;
}
