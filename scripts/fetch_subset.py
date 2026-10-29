"""Download selected members from the Zenodo clothing zip with HTTP range requests.

The archive is about 27 GB. This script reads the central directory once, then
fetches only the files you ask for (JSON annotations, or front and back images).
Brand-label images are never requested.
"""

from __future__ import annotations

import argparse
import json
import struct
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from remotezip import RemoteZip

DEFAULT_URL = (
    "https://zenodo.org/records/13788681/files/clothing_v3.zip?download=1"
)
DATASET_DOI = "10.5281/zenodo.13788681"
USER_AGENT = "garment-routing/0.1 (research; CC-BY attribution)"


def resolve_url(url: str) -> str:
    """Follow redirects once so later range requests hit the file host directly."""
    headers = {"User-Agent": USER_AGENT, "Range": "bytes=0-0"}
    for attempt in range(6):
        response = requests.get(url, stream=True, allow_redirects=True, timeout=120, headers=headers)
        try:
            if response.status_code == 429:
                wait = int(response.headers.get("Retry-After", 15 * (attempt + 1)))
                time.sleep(min(wait, 90))
                continue
            response.raise_for_status()
            return response.url
        finally:
            response.close()
    raise RuntimeError("rate limited while resolving the dataset URL")


def ranged_get(session: requests.Session, url: str, start: int, end: int) -> requests.Response:
    """GET one byte range, backing off when Zenodo answers 429."""
    response = None
    for attempt in range(6):
        response = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=(30, 180))
        if response.status_code != 429:
            return response
        wait = int(response.headers.get("Retry-After", "5"))
        time.sleep(max(1, min(wait, 20)))
    return response


def load_manifest(url: str, cache_path: Path) -> list[dict]:
    """Return zip member metadata, cached locally after the first listing."""
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Listing {url}", flush=True)
    started = time.time()
    with RemoteZip(url, headers={"User-Agent": USER_AGENT}) as archive:
        members = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            members.append(
                {
                    "name": info.filename.replace("\\", "/"),
                    "header_offset": int(info.header_offset),
                    "compress_size": int(info.compress_size),
                    "file_size": int(info.file_size),
                    "compress_type": int(info.compress_type),
                }
            )
    cache_path.write_text(json.dumps(members), encoding="utf-8")
    print(f"Cached {len(members)} members in {time.time() - started:.1f}s", flush=True)
    return members


def decode_zip_member(blob: bytes, compress_size: int) -> bytes:
    """Slice one local-file entry out of a range response and inflate it."""
    if len(blob) < 30 or blob[:4] != b"PK\x03\x04":
        raise ValueError("response is not a zip local file header")
    local_comp = struct.unpack_from("<H", blob, 8)[0]
    local_csz = struct.unpack_from("<I", blob, 18)[0]
    name_len = struct.unpack_from("<H", blob, 26)[0]
    extra_len = struct.unpack_from("<H", blob, 28)[0]
    compressed_len = local_csz or compress_size
    start = 30 + name_len + extra_len
    end = start + compressed_len
    if end > len(blob):
        raise ValueError(
            f"truncated zip entry: have {len(blob)} bytes, need {end} "
            f"(name {name_len}, extra {extra_len}, data {compressed_len})"
        )
    data = blob[start:end]
    if local_comp == 0:
        return data
    if local_comp == 8:
        return zlib.decompress(data, -15)
    raise ValueError(f"unsupported zip compression method {local_comp}")


def _session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def _write_atomic(destination: Path, payload: bytes) -> None:
    """Write via a sibling partial file so an interrupted download is not treated as done."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def download_member(
    file_url: str,
    member: dict,
    destination: Path,
    session: requests.Session,
    timeout: tuple[float, float] = (30, 180),
) -> None:
    """Download one zip member to ``destination`` using a single range request."""
    cushion = 512
    last_error: Exception | None = None
    for _ in range(2):
        start = int(member["header_offset"])
        end = start + 30 + cushion + int(member["compress_size"]) - 1
        response = ranged_get(session, file_url, start, end)
        try:
            if response.status_code != 206:
                raise RuntimeError(
                    f"expected HTTP 206 for a range request, got {response.status_code}. "
                    "Refusing to read the response body."
                )
            payload = decode_zip_member(response.content, int(member["compress_size"]))
        except ValueError as exc:
            last_error = exc
            if "truncated" in str(exc):
                cushion = max(cushion * 8, 4096)
                continue
            raise
        else:
            _write_atomic(destination, payload)
            return
    raise RuntimeError(f"failed to decode {member['name']}: {last_error}")


def _safe_relative(name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        raise ValueError(f"unsafe zip path: {name}")
    return relative


def group_runs(members: list[dict]) -> list[list[dict]]:
    """Group zip members that sit next to each other so one range read can grab the run.

    Annotation JSON files are stored back to back. Reading each one alone spends
    the whole round trip on a few hundred bytes.
    """
    ordered = sorted(members, key=lambda member: int(member["header_offset"]))
    runs: list[list[dict]] = []
    current: list[dict] = []
    for member in ordered:
        if current:
            previous = current[-1]
            gap = int(member["header_offset"]) - int(previous["header_offset"])
            if gap > int(previous["compress_size"]) + 2000:
                runs.append(current)
                current = []
        current.append(member)
    if current:
        runs.append(current)
    return runs


def _member_at(blob: bytes, offset: int) -> tuple[str, bytes, int]:
    """Return the filename, inflated bytes, and the offset of the next entry."""
    if blob[offset : offset + 4] != b"PK\x03\x04":
        raise ValueError("not a zip local header")
    flag, comp, csz, name_len, extra_len = (
        struct.unpack_from("<H", blob, offset + 6)[0],
        struct.unpack_from("<H", blob, offset + 8)[0],
        struct.unpack_from("<I", blob, offset + 18)[0],
        struct.unpack_from("<H", blob, offset + 26)[0],
        struct.unpack_from("<H", blob, offset + 28)[0],
    )
    name = blob[offset + 30 : offset + 30 + name_len].decode("utf-8" if flag & 0x800 else "cp437")
    start = offset + 30 + name_len + extra_len
    end = start + csz
    if end > len(blob):
        raise ValueError("truncated zip entry")
    data = blob[start:end]
    if comp == 8:
        data = zlib.decompress(data, -15)
    elif comp != 0:
        raise ValueError(f"unsupported zip compression method {comp}")
    return name.replace("\\", "/"), data, end


def download_run(
    file_url: str,
    run: list[dict],
    destination_root: Path,
    session: requests.Session,
) -> None:
    """Download a contiguous run and write any members that are not already on disk."""
    start = int(run[0]["header_offset"])
    last = run[-1]
    end = int(last["header_offset"]) + 30 + 1024 + int(last["compress_size"]) - 1
    response = ranged_get(session, file_url, start, end)
    if response.status_code != 206:
        raise RuntimeError(
            f"expected HTTP 206 for a range request, got {response.status_code}. "
            "Refusing to read the response body."
        )
    blob = response.content
    found: dict[str, bytes] = {}
    offset = 0
    while offset + 30 <= len(blob) and blob[offset : offset + 4] == b"PK\x03\x04":
        try:
            name, payload, offset = _member_at(blob, offset)
        except ValueError:
            break
        found[name] = payload
    missing = [member for member in run if member["name"] not in found]
    for member in missing:
        target = destination_root / _safe_relative(member["name"])
        download_member(file_url, member, target, session)
    for member in run:
        payload = found.get(member["name"])
        if payload is None:
            continue
        target = destination_root / _safe_relative(member["name"])
        if target.exists() and target.stat().st_size > 0:
            continue
        _write_atomic(target, payload)
    relative = Path(name)
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        raise ValueError(f"unsafe zip path: {name}")
    return relative


def select_members(members: list[dict], mode: str, names: set[str] | None) -> list[dict]:
    """Choose JSON files, or explicit archive paths."""
    if names is not None:
        wanted = {name.replace("\\", "/") for name in names}
        return [member for member in members if member["name"] in wanted]
    if mode == "json":
        return [
            member
            for member in members
            if member["name"].endswith(".json") and "json_errors" not in member["name"].split("/")
        ]
    raise ValueError(f"unknown selection mode {mode}")


class _Throttle:
    """Minimum gap between the start of successive range requests."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = max(0.0, float(min_interval))
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            if delay > 0:
                time.sleep(delay)
            self._next = time.monotonic() + self.min_interval


def read_name_list(path: Path) -> set[str]:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return set(lines)


def fetch(
    url: str,
    output_dir: Path,
    manifest_path: Path,
    mode: str,
    names: set[str] | None,
    workers: int,
    limit: int | None,
    min_interval: float = 0.0,
) -> dict:
    """Download the selected members.

    A non-empty final file is skipped, so running the same command again resumes.
    Partial files end in ``.partial`` and are downloaded again.
    """
    members = load_manifest(url, manifest_path)
    chosen = select_members(members, mode, names)
    if limit is not None:
        chosen = chosen[:limit]
    file_url = resolve_url(url)
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    skipped = 0
    for member in chosen:
        target = output_dir / _safe_relative(member["name"])
        if target.exists() and target.stat().st_size > 0:
            skipped += 1
            continue
        pending.append((member, target))
    print(
        f"Selected {len(chosen)} files, {skipped} already present, {len(pending)} to download, "
        f"workers {max(1, workers)}, min interval {min_interval:.2f}s",
        flush=True,
    )
    throttle = _Throttle(min_interval)
    errors: list[dict] = []
    done = 0
    finished_runs = 0
    lock = threading.Lock()
    thread_state = threading.local()
    runs = group_runs([member for member, _target in pending]) if mode == "json" else [[member] for member, _target in pending]

    def worker(run: list[dict]) -> int:
        if not hasattr(thread_state, "session"):
            thread_state.session = _session()
        last: Exception | None = None
        for attempt in range(4):
            try:
                throttle.wait()
                if mode == "json":
                    download_run(file_url, run, output_dir, thread_state.session)
                else:
                    member = run[0]
                    download_member(file_url, member, output_dir / _safe_relative(member["name"]), thread_state.session)
                return len(run)
            except Exception as exc:  # noqa: BLE001 - retry then record
                last = exc
                time.sleep(min(30.0, 1.5 * (attempt + 1)))
        raise RuntimeError(str(last))

    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(worker, run): run for run in runs}
        for future in as_completed(futures):
            run = futures[future]
            try:
                count = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad run should not stop the rest
                errors.append({"name": run[0]["name"], "error": str(exc), "run_length": len(run)})
                count = 0
            with lock:
                done += count
                finished_runs += 1
                if finished_runs % 10 == 0 or done >= len(pending):
                    rate = done / max(time.time() - started, 1e-6)
                    print(
                        f"Downloaded {done}/{len(pending)} files across {finished_runs}/{len(runs)} runs ({rate:.1f} files/s)",
                        flush=True,
                    )
    error_path = output_dir.parent / "fetch_errors.jsonl"
    if errors:
        with error_path.open("a", encoding="utf-8") as handle:
            for row in errors:
                handle.write(json.dumps(row) + "\n")
    summary = {
        "doi": DATASET_DOI,
        "url": url,
        "mode": mode,
        "selected": len(chosen),
        "already_present": skipped,
        "downloaded": len(pending) - len(errors),
        "errors": len(errors),
    }
    print(json.dumps(summary), flush=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("data/zip_manifest.json"))
    parser.add_argument("--json", action="store_true", help="Download every annotation JSON")
    parser.add_argument(
        "--members",
        type=Path,
        help="Text file of archive paths to download (front and back images)",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--min-interval",
        type=float,
        default=None,
        help="Seconds between range requests. Image downloads default to 0.4; JSON defaults to 0.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    if args.json == bool(args.members):
        parser.error("pass exactly one of --json or --members")
    names = read_name_list(args.members) if args.members else None
    mode = "json" if args.json else "members"
    min_interval = 0.0 if mode == "json" else 0.4
    if args.min_interval is not None:
        min_interval = args.min_interval
    fetch(args.url, args.out, args.manifest, mode, names, args.workers, args.limit, min_interval)


if __name__ == "__main__":
    main()
