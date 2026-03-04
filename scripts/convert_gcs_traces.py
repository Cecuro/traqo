"""Convert raw .jsonl traces in a GCS bucket to compressed split format.

Three phases for maximum throughput:
  1. Bulk download with gsutil -m cp (parallel transfers)
  2. Compress locally with multiprocessing
  3. Bulk upload with gsutil -m cp (parallel transfers)

Skips files that already have a .jsonl.gz counterpart in GCS.

Usage:
    python scripts/convert_gcs_traces.py gs://bucket/prefix/
    python scripts/convert_gcs_traces.py gs://bucket/prefix/ --dry-run
    python scripts/convert_gcs_traces.py gs://bucket/prefix/ --delete-originals
    python scripts/convert_gcs_traces.py gs://bucket/prefix/ --workers 8
"""

from __future__ import annotations

import argparse
import multiprocessing
import subprocess
import sys
import time
from pathlib import Path


def _gsutil(*args: str, capture: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = ["gsutil"] + list(args)
    return subprocess.run(cmd, capture_output=capture, text=True, check=True)


def _gsutil_ls(uri: str) -> list[dict]:
    """List objects with size info using gsutil ls -l."""
    result = _gsutil("ls", "-l", uri)
    entries = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("TOTAL:"):
            continue
        parts = line.split(None, 2)
        if len(parts) >= 3:
            size = int(parts[0])
            name = parts[2]
            entries.append({"name": name, "size": size})
    return entries


def _compress_one(raw_path: Path) -> dict:
    """Compress a single .jsonl file. Returns result dict."""
    from traqo.compress import split_and_compress

    try:
        t0 = time.time()
        main_path, content_path = split_and_compress(raw_path)
        elapsed = time.time() - t0

        main_size = main_path.stat().st_size
        content_size = content_path.stat().st_size if content_path else 0
        raw_size = raw_path.stat().st_size
        ratio = raw_size / (main_size + content_size) if (main_size + content_size) > 0 else 0

        return {
            "raw": str(raw_path),
            "main": str(main_path),
            "content": str(content_path) if content_path else None,
            "raw_size": raw_size,
            "main_size": main_size,
            "content_size": content_size,
            "ratio": ratio,
            "time": elapsed,
            "error": None,
        }
    except Exception as e:
        return {
            "raw": str(raw_path),
            "error": str(e),
        }


def convert_bucket(
    uri: str,
    *,
    dry_run: bool = False,
    delete_originals: bool = False,
    workers: int = 4,
) -> None:
    if not uri.startswith("gs://"):
        print(f"Error: expected gs:// URI, got {uri}", file=sys.stderr)
        sys.exit(1)

    uri = uri.rstrip("/") + "/"

    # --- Phase 0: List and filter ---
    print(f"Listing objects in {uri} ...")
    entries = _gsutil_ls(uri)

    all_names = {e["name"] for e in entries}
    raw_entries = [
        e
        for e in entries
        if e["name"].endswith(".jsonl")
        and not e["name"].endswith(".content.jsonl.zst")
    ]

    to_convert = []
    already_done = []
    for e in raw_entries:
        gz_name = e["name"] + ".gz"
        if gz_name in all_names:
            already_done.append(e["name"])
        else:
            to_convert.append(e)

    total_bytes = sum(e["size"] for e in to_convert)
    print(f"Found {len(raw_entries)} raw .jsonl files")
    print(f"  Already converted: {len(already_done)}")
    print(f"  To convert: {len(to_convert)} ({total_bytes / 1e9:.1f} GB)")
    print()

    if dry_run:
        for e in to_convert[:20]:
            size_mb = e["size"] / 1e6
            name = e["name"].split("/")[-1]
            print(f"  Would convert: {name} ({size_mb:.1f} MB)")
        if len(to_convert) > 20:
            print(f"  ... and {len(to_convert) - 20} more")
        return

    if not to_convert:
        print("Nothing to convert.")
        return

    # Use a local work directory (not temp — we want to keep files between phases)
    work_dir = Path("_convert_work")
    raw_dir = work_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Bulk download ---
    # Write list of GCS paths to download
    urls_file = work_dir / "download_urls.txt"
    urls_file.write_text("\n".join(e["name"] for e in to_convert) + "\n")

    # Check what's already downloaded
    already_local = {f.name for f in raw_dir.iterdir() if f.is_file()}
    to_download = [
        e for e in to_convert if e["name"].split("/")[-1] not in already_local
    ]

    if to_download:
        dl_urls_file = work_dir / "download_urls_remaining.txt"
        dl_urls_file.write_text("\n".join(e["name"] for e in to_download) + "\n")

        dl_bytes = sum(e["size"] for e in to_download)
        print(
            f"Phase 1: Downloading {len(to_download)} files ({dl_bytes / 1e9:.1f} GB) ..."
        )
        t0 = time.time()
        # gsutil -m cp: parallel multi-threaded copy
        result = subprocess.run(
            [
                "gsutil",
                "-m",
                "cp",
                "-I",  # read URLs from stdin
                str(raw_dir) + "/",
            ],
            input="\n".join(e["name"] for e in to_download),
            text=True,
            capture_output=True,
        )
        dl_time = time.time() - t0

        if result.returncode != 0:
            print(f"Warning: gsutil -m cp returned {result.returncode}", file=sys.stderr)
            if result.stderr:
                # Print last few lines of stderr
                for line in result.stderr.strip().splitlines()[-5:]:
                    print(f"  {line}", file=sys.stderr)

        downloaded = sum(1 for f in raw_dir.iterdir() if f.is_file() and f.name.endswith(".jsonl"))
        print(f"  Downloaded {downloaded} files in {dl_time:.0f}s")
    else:
        print(f"Phase 1: All {len(to_convert)} files already downloaded locally")

    print()

    # --- Phase 2: Compress in parallel ---
    raw_files = sorted(
        f for f in raw_dir.iterdir() if f.is_file() and f.name.endswith(".jsonl")
    )

    # Skip files that are already compressed
    to_compress = []
    for f in raw_files:
        gz_path = f.parent / (f.stem + ".jsonl.gz")
        if not gz_path.exists():
            to_compress.append(f)

    print(f"Phase 2: Compressing {len(to_compress)} files with {workers} workers ...")
    t0 = time.time()

    with multiprocessing.Pool(workers) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(_compress_one, to_compress)):
            raw_name = Path(result["raw"]).name
            if result["error"]:
                print(f"  [{i+1}/{len(to_compress)}] {raw_name} FAILED: {result['error']}")
            else:
                print(
                    f"  [{i+1}/{len(to_compress)}] {raw_name} "
                    f"-> {result['main_size']/1e3:.0f} KB + {result['content_size']/1e3:.0f} KB "
                    f"({result['ratio']:.0f}x, {result['time']:.1f}s)"
                )
            results.append(result)

    compress_time = time.time() - t0
    successful = [r for r in results if not r["error"]]
    failed = [r for r in results if r["error"]]
    print(
        f"  Compressed {len(successful)}/{len(to_compress)} in {compress_time:.0f}s"
    )
    if failed:
        print(f"  Failed: {len(failed)}")
    print()

    # --- Phase 3: Bulk upload ---
    # Collect all compressed files to upload
    upload_files_gz = sorted(raw_dir.glob("*.jsonl.gz"))
    upload_files_zst = sorted(raw_dir.glob("*.content.jsonl.zst"))

    print(
        f"Phase 3: Uploading {len(upload_files_gz)} main + "
        f"{len(upload_files_zst)} content files ..."
    )
    t0 = time.time()

    # Upload .jsonl.gz (no Content-Encoding — we want the file stored as-is)
    if upload_files_gz:
        result = subprocess.run(
            [
                "gsutil",
                "-m",
                "-h", "Content-Type:application/gzip",
                "cp",
            ]
            + [str(f) for f in upload_files_gz]
            + [uri],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Warning: main file upload returned {result.returncode}", file=sys.stderr)
            for line in (result.stderr or "").strip().splitlines()[-5:]:
                print(f"  {line}", file=sys.stderr)

    # Upload .content.jsonl.zst
    if upload_files_zst:
        result = subprocess.run(
            [
                "gsutil",
                "-m",
                "-h", "Content-Type:application/zstd",
                "cp",
            ]
            + [str(f) for f in upload_files_zst]
            + [uri],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Warning: content file upload returned {result.returncode}", file=sys.stderr)
            for line in (result.stderr or "").strip().splitlines()[-5:]:
                print(f"  {line}", file=sys.stderr)

    upload_time = time.time() - t0
    total_upload_size = sum(f.stat().st_size for f in upload_files_gz) + sum(
        f.stat().st_size for f in upload_files_zst
    )
    print(f"  Uploaded {total_upload_size / 1e6:.0f} MB in {upload_time:.0f}s")
    print()

    # --- Phase 4: Optional cleanup ---
    if delete_originals:
        print(f"Phase 4: Deleting {len(to_convert)} original .jsonl files from GCS ...")
        result = subprocess.run(
            ["gsutil", "-m", "rm"]
            + [e["name"] for e in to_convert],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Warning: delete returned {result.returncode}", file=sys.stderr)

    # --- Summary ---
    total_raw = sum(e["size"] for e in to_convert)
    total_compressed = sum(f.stat().st_size for f in upload_files_gz) + sum(
        f.stat().st_size for f in upload_files_zst
    )
    ratio = total_raw / total_compressed if total_compressed > 0 else 0

    print("=" * 60)
    print(f"Total: {total_raw / 1e9:.1f} GB -> {total_compressed / 1e6:.0f} MB ({ratio:.0f}x)")
    print(f"Files: {len(to_convert)} raw -> {len(upload_files_gz)} main + {len(upload_files_zst)} content")
    if failed:
        print(f"Failed: {len(failed)}")
    print()
    print(f"Local work dir: {work_dir.resolve()}")
    print("Run 'rm -rf _convert_work' to clean up local files when done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw .jsonl traces in GCS to compressed format"
    )
    parser.add_argument("uri", help="GCS URI: gs://bucket/prefix/")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done"
    )
    parser.add_argument(
        "--delete-originals",
        action="store_true",
        help="Delete .jsonl originals after conversion",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel compression workers (default: 4)",
    )
    args = parser.parse_args()

    convert_bucket(
        args.uri,
        dry_run=args.dry_run,
        delete_originals=args.delete_originals,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
