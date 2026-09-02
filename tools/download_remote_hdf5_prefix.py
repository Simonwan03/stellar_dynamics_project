from __future__ import annotations

import argparse
import io
import json
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import h5py
import numpy as np


ALL_KEYS = (
    "APs_complete",
    "LANs_complete",
    "angMomentum_complete",
    "aratios_complete",
    "eccs_complete",
    "energy_complete",
    "incs_complete",
    "masses_complete",
    "mratios_complete",
    "periods_complete",
    "positions_complete",
    "smaxes_complete",
    "tAnos_complete",
    "times_complete",
    "velocities_complete",
)
ORBITAL15_KEYS = (
    "aratios_complete",
    "mratios_complete",
    "eccs_complete",
    "incs_complete",
    "APs_complete",
    "LANs_complete",
    "times_complete",
)
TRAJECTORY_KEYS = (
    "positions_complete",
    "velocities_complete",
    "times_complete",
)
RESULT_NAME = "results_batch_{batch:02d}_sampling_{sampling_id}.hdf5"


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    offset: int
    attrs: dict[str, Any]

    @property
    def tail_shape(self) -> tuple[int, ...]:
        return self.shape[2:]

    @property
    def tail_size(self) -> int:
        return max(1, int(np.prod(self.tail_shape, dtype=np.int64)))

    @property
    def row_bytes(self) -> int:
        return int(self.shape[1] * self.tail_size * self.dtype.itemsize)

    def prefix_bytes(self, prefix_len: int) -> int:
        return int(prefix_len * self.tail_size * self.dtype.itemsize)


@dataclass(frozen=True)
class RemoteMetadata:
    url: str
    file_size: int
    file_attrs: dict[str, Any]
    datasets: dict[str, DatasetInfo]
    source_time_end: float | None


@dataclass(frozen=True)
class RangeSpec:
    key: str
    row: int
    start: int
    end: int


class HttpRangeFile:
    """Small random-access read-only file object backed by HTTP Range requests."""

    def __init__(self, url: str, block_size: int = 4 * 1024 * 1024) -> None:
        self.url = url
        self.block_size = int(block_size)
        self.position = 0
        self._closed = False
        self._cache: dict[int, bytes] = {}
        self.requests = 0
        self.bytes_read = 0
        self.file_size = _head_size(url)

    @property
    def closed(self) -> bool:
        return self._closed

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_position = offset
        elif whence == io.SEEK_CUR:
            new_position = self.position + offset
        elif whence == io.SEEK_END:
            new_position = self.file_size + offset
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        if new_position < 0:
            raise ValueError("Cannot seek to a negative position")
        self.position = int(new_position)
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.file_size - self.position
        size = min(int(size), max(0, self.file_size - self.position))
        if size <= 0:
            return b""

        end_position = self.position + size
        output = bytearray()
        while self.position < end_position:
            block_id = self.position // self.block_size
            block = self._load_block(block_id)
            block_start = block_id * self.block_size
            within_block = self.position - block_start
            take = min(end_position - self.position, len(block) - within_block)
            output.extend(block[within_block : within_block + take])
            self.position += take
        return bytes(output)

    def readinto(self, buffer: bytearray | memoryview) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "HttpRangeFile":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _load_block(self, block_id: int) -> bytes:
        cached = self._cache.get(block_id)
        if cached is not None:
            return cached
        start = block_id * self.block_size
        end = min(self.file_size, start + self.block_size) - 1
        data = _request_range(self.url, [(start, end)], timeout=120.0)[(start, end)]
        self._cache[block_id] = data
        self.requests += 1
        self.bytes_read += len(data)
        return data


def _head_size(url: str) -> int:
    request = Request(url, method="HEAD", headers={"Accept-Encoding": "identity"})
    with urlopen(request, timeout=60.0) as response:
        value = response.headers.get("Content-Length")
    if value is None:
        raise RuntimeError(f"HEAD response for {url} did not contain Content-Length")
    return int(value)


def _request_range(
    url: str,
    ranges: list[tuple[int, int]],
    *,
    timeout: float,
    retries: int = 4,
) -> dict[tuple[int, int], bytes]:
    if not ranges:
        return {}
    range_header = "bytes=" + ",".join(f"{start}-{end}" for start, end in ranges)
    request = Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "Range": range_header,
            "User-Agent": "stellar-dynamics-prefix-downloader/1.0",
        },
    )

    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout) as response:
                if response.status != 206:
                    raise RuntimeError(
                        f"Server did not honor Range request for {url}: status={response.status}"
                    )
                body = response.read()
                content_type = response.headers.get("Content-Type", "")
                content_range = response.headers.get("Content-Range")
            if "multipart/byteranges" in content_type.lower():
                parsed = _parse_multipart_ranges(body, content_type)
            elif content_range is not None:
                match = _CONTENT_RANGE_RE.search(content_range)
                if match is None:
                    raise RuntimeError(f"Invalid Content-Range header: {content_range}")
                start, end = int(match.group(1)), int(match.group(2))
                parsed = {(start, end): body}
            else:
                raise RuntimeError(
                    f"Range response for {url} had neither multipart nor Content-Range metadata"
                )
            for start, end in ranges:
                data = parsed.get((start, end))
                if data is None:
                    raise RuntimeError(f"Range response omitted bytes {start}-{end} from {url}")
                if len(data) != end - start + 1:
                    raise RuntimeError(
                        f"Range length mismatch for {url}: requested {start}-{end}, got {len(data)} bytes"
                    )
            return parsed
        except (HTTPError, OSError, RuntimeError):
            if attempt + 1 >= retries:
                raise
            time.sleep(float(2**attempt))
    raise AssertionError("unreachable")


_CONTENT_RANGE_RE = re.compile(r"bytes\s+(\d+)-(\d+)/(?:\d+|\*)", re.IGNORECASE)


def _parse_multipart_ranges(body: bytes, content_type: str) -> dict[tuple[int, int], bytes]:
    match = re.search(r"boundary=\"?([^\";]+)", content_type, re.IGNORECASE)
    if match is None:
        raise RuntimeError(f"Multipart response has no boundary: {content_type}")
    marker = b"--" + match.group(1).encode("ascii")
    parsed: dict[tuple[int, int], bytes] = {}
    for part in body.split(marker):
        if part.startswith(b"\r\n"):
            part = part[2:]
        if part.endswith(b"\r\n"):
            part = part[:-2]
        if part == b"--":
            continue
        if not part:
            continue
        separator = part.find(b"\r\n\r\n")
        if separator < 0:
            raise RuntimeError("Malformed multipart Range response")
        header_block = part[:separator].decode("ascii", errors="strict")
        payload = part[separator + 4 :]
        content_range = None
        for line in header_block.split("\r\n"):
            if line.lower().startswith("content-range:"):
                content_range = line.split(":", 1)[1].strip()
                break
        if content_range is None:
            raise RuntimeError("Multipart Range part has no Content-Range")
        match = _CONTENT_RANGE_RE.search(content_range)
        if match is None:
            raise RuntimeError(f"Invalid multipart Content-Range: {content_range}")
        parsed[(int(match.group(1)), int(match.group(2)))] = payload
    return parsed


def parse_key_spec(spec: str) -> tuple[str, ...]:
    normalized = spec.strip().lower()
    if normalized == "all":
        return ALL_KEYS
    if normalized == "orbital15":
        return ORBITAL15_KEYS
    if normalized in {"trajectory", "raw_state18"}:
        return TRAJECTORY_KEYS
    requested = {item.strip() for item in spec.split(",") if item.strip()}
    unknown = sorted(requested - set(ALL_KEYS))
    if unknown:
        raise ValueError(f"Unknown dataset key(s): {unknown}. Available keys: {list(ALL_KEYS)}")
    if not requested:
        raise ValueError("--keys must select at least one dataset")
    # Existing loaders expect a time grid when reading a prefix directory.
    requested.add("times_complete")
    return tuple(key for key in ALL_KEYS if key in requested)


def inspect_remote(url: str, block_size: int) -> RemoteMetadata:
    remote = HttpRangeFile(url, block_size=block_size)
    file_attrs: dict[str, Any]
    datasets: dict[str, DatasetInfo]
    try:
        with h5py.File(remote, "r", driver="fileobj") as handle:
            file_attrs = {str(key): value for key, value in handle.attrs.items()}
            datasets = {}
            for name, dataset in handle.items():
                if not isinstance(dataset, h5py.Dataset):
                    continue
                if dataset.ndim < 2:
                    raise ValueError(f"Expected {url}:{name} to have [N,T,...] shape, got {dataset.shape}")
                if dataset.chunks is not None or dataset.compression is not None:
                    raise ValueError(
                        f"{url}:{name} is chunked/compressed. The exact contiguous Range path cannot be used."
                    )
                offset = dataset.id.get_offset()
                if offset is None:
                    raise ValueError(f"Could not determine contiguous data offset for {url}:{name}")
                datasets[str(name)] = DatasetInfo(
                    name=str(name),
                    shape=tuple(int(dim) for dim in dataset.shape),
                    dtype=np.dtype(dataset.dtype),
                    offset=int(offset),
                    attrs={str(key): value for key, value in dataset.attrs.items()},
                )
        source_time_end = None
        time_info = datasets.get("times_complete")
        if time_info is not None:
            start = time_info.offset + (time_info.shape[1] - 1) * time_info.row_bytes
            end = start + time_info.dtype.itemsize - 1
            raw = _request_range(url, [(start, end)], timeout=120.0)[(start, end)]
            source_time_end = float(np.frombuffer(raw, dtype=time_info.dtype, count=1)[0])
    finally:
        remote.close()
    return RemoteMetadata(
        url=url,
        file_size=remote.file_size,
        file_attrs=file_attrs,
        datasets=datasets,
        source_time_end=source_time_end,
    )


def validate_selection(metadata: RemoteMetadata, keys: tuple[str, ...], prefix_len: int) -> tuple[int, int]:
    missing = sorted(set(keys) - set(metadata.datasets))
    if missing:
        raise KeyError(f"Remote file {metadata.url} is missing datasets: {missing}")
    first = metadata.datasets[keys[0]]
    row_count, time_steps = first.shape[:2]
    if prefix_len <= 0 or prefix_len > time_steps:
        raise ValueError(f"prefix_len={prefix_len} is outside the remote time range [1, {time_steps}]")
    for key in keys[1:]:
        info = metadata.datasets[key]
        if info.shape[:2] != (row_count, time_steps):
            raise ValueError(
                f"Inconsistent [N,T] shape for {key}: {info.shape[:2]} vs {(row_count, time_steps)}"
            )
    for key in keys:
        info = metadata.datasets[key]
        end = info.offset + info.row_bytes * row_count
        if end > metadata.file_size:
            raise ValueError(f"Dataset {key} extends beyond the remote file size")
    return row_count, time_steps


def make_block_specs(
    metadata: RemoteMetadata,
    infos: tuple[DatasetInfo, ...],
    row_start: int,
    row_stop: int,
    prefix_len: int,
) -> list[RangeSpec]:
    specs: list[RangeSpec] = []
    for row in range(row_start, row_stop):
        for info in infos:
            start = info.offset + row * info.row_bytes
            end = start + info.prefix_bytes(prefix_len) - 1
            specs.append(RangeSpec(info.name, row, start, end))
    return specs


def fetch_block(
    url: str,
    specs: list[RangeSpec],
    infos: dict[str, DatasetInfo],
    prefix_len: int,
    timeout: float,
) -> dict[str, np.ndarray]:
    raw_parts = _request_range(url, [(spec.start, spec.end) for spec in specs], timeout=timeout)
    grouped: dict[str, list[np.ndarray]] = {key: [] for key in infos}
    for spec in specs:
        raw = raw_parts[(spec.start, spec.end)]
        info = infos[spec.key]
        values = np.frombuffer(raw, dtype=info.dtype).reshape((prefix_len, *info.tail_shape))
        grouped[spec.key].append(values)
    return {key: np.stack(values, axis=0) for key, values in grouped.items()}


def _iter_ordered_fetches(
    *,
    url: str,
    metadata: RemoteMetadata,
    infos: tuple[DatasetInfo, ...],
    row_count: int,
    prefix_len: int,
    rows_per_request: int,
    workers: int,
    timeout: float,
) -> Iterable[tuple[int, int, dict[str, np.ndarray]]]:
    blocks = [
        (start, min(row_count, start + rows_per_request))
        for start in range(0, row_count, rows_per_request)
    ]
    if workers <= 1:
        for row_start, row_stop in blocks:
            specs = make_block_specs(metadata, infos, row_start, row_stop, prefix_len)
            yield row_start, row_stop, fetch_block(
                url,
                specs,
                {info.name: info for info in infos},
                prefix_len,
                timeout,
            )
        return

    def submit(executor: ThreadPoolExecutor, row_start: int, row_stop: int) -> Future[dict[str, np.ndarray]]:
        specs = make_block_specs(metadata, infos, row_start, row_stop, prefix_len)
        return executor.submit(
            fetch_block,
            url,
            specs,
            {info.name: info for info in infos},
            prefix_len,
            timeout,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: dict[int, tuple[int, int, Future[dict[str, np.ndarray]]]] = {}
        next_block = 0
        while next_block < min(workers, len(blocks)):
            row_start, row_stop = blocks[next_block]
            pending[next_block] = (row_start, row_stop, submit(executor, row_start, row_stop))
            next_block += 1
        for block_index in range(len(blocks)):
            row_start, row_stop, future = pending.pop(block_index)
            result = future.result()
            yield row_start, row_stop, result
            if next_block < len(blocks):
                next_start, next_stop = blocks[next_block]
                pending[next_block] = (next_start, next_stop, submit(executor, next_start, next_stop))
                next_block += 1


def output_matches(path: Path, keys: tuple[str, ...], row_count: int, prefix_len: int, dtype_mode: str) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as handle:
            if tuple(handle.keys()) != tuple(sorted(keys)):
                return False
            for key in keys:
                dataset = handle[key]
                if dataset.shape[:2] != (row_count, prefix_len):
                    return False
                if dtype_mode == "float32" and dataset.dtype != np.dtype("float32"):
                    return False
            return True
    except OSError:
        return False


def write_prefix_shard(
    metadata: RemoteMetadata,
    output_path: Path,
    keys: tuple[str, ...],
    prefix_len: int,
    dtype_mode: str,
    max_ranges: int,
    workers: int,
    timeout: float,
    max_rows: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    row_count, time_steps = validate_selection(metadata, keys, prefix_len)
    if max_rows is not None and max_rows > 0:
        row_count = min(row_count, max_rows)
    if output_matches(output_path, keys, row_count, prefix_len, dtype_mode) and not overwrite:
        print(f"[skip] {output_path} already matches the requested prefix")
        return {"path": str(output_path), "rows": row_count, "skipped": True}
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists (use --overwrite): {output_path}")

    infos = tuple(metadata.datasets[key] for key in keys)
    if len(infos) > max_ranges:
        raise ValueError(f"Need at least {len(infos)} max ranges for {len(infos)} selected datasets")
    rows_per_request = max(1, max_ranges // len(infos))
    output_dtype = None if dtype_mode == "preserve" else np.dtype("float32")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.part")
    if temporary_path.exists():
        # A shard is committed atomically only after all rows are written.
        # Restart an interrupted shard, while completed sibling shards remain
        # reusable on the next invocation.
        temporary_path.unlink()
        print(f"[restart] removing incomplete shard {temporary_path}", flush=True)

    with h5py.File(temporary_path, "w") as output:
        for key, value in metadata.file_attrs.items():
            output.attrs[key] = value
        output.attrs["prefix_len"] = np.int64(prefix_len)
        output.attrs["source_time_steps"] = np.int64(time_steps)
        if metadata.source_time_end is not None:
            output.attrs["source_time_end"] = np.float64(metadata.source_time_end)
        output.attrs["source_url"] = metadata.url
        output.attrs["selected_keys"] = np.asarray(keys, dtype="S")
        output.attrs["storage_dtype_mode"] = dtype_mode

        for info in infos:
            shape = (row_count, prefix_len, *info.tail_shape)
            output.create_dataset(info.name, shape=shape, dtype=output_dtype or info.dtype)
            for attr_key, attr_value in info.attrs.items():
                output[info.name].attrs[attr_key] = attr_value

        for row_start, row_stop, block in _iter_ordered_fetches(
            url=metadata.url,
            metadata=metadata,
            infos=infos,
            row_count=row_count,
            prefix_len=prefix_len,
            rows_per_request=rows_per_request,
            workers=workers,
            timeout=timeout,
        ):
            for key in keys:
                values = block[key]
                if output_dtype is not None:
                    values = values.astype(output_dtype, copy=False)
                output[key][row_start:row_stop] = values
            print(f"[rows] {output_path.name}: {row_stop}/{row_count}", flush=True)
        output.flush()

    temporary_path.replace(output_path)
    bytes_written = sum(
        row_count * prefix_len * max(1, int(np.prod(metadata.datasets[key].tail_shape, dtype=np.int64)))
        * int((output_dtype or metadata.datasets[key].dtype).itemsize)
        for key in keys
    )
    return {
        "path": str(output_path),
        "rows": row_count,
        "prefix_len": prefix_len,
        "dtype_mode": dtype_mode,
        "keys": list(keys),
        "estimated_data_bytes": int(bytes_written),
        "skipped": False,
    }


def download_file(url: str, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        print(f"[skip] {path} already exists")
        return
    temporary = path.with_name(f".{path.name}.part")
    if temporary.exists():
        temporary.unlink()
        print(f"[restart] removing incomplete file {temporary}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        url,
        headers={"Accept-Encoding": "identity", "User-Agent": "stellar-dynamics-prefix-downloader/1.0"},
    )
    with urlopen(request, timeout=120.0) as response, temporary.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    temporary.replace(path)


def parse_shards(spec: str | None) -> list[int]:
    if spec is None or not spec.strip():
        return list(range(40))
    values = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    if not values or any(value < 0 for value in values):
        raise ValueError("--shards must contain non-negative batch numbers, e.g. 0,1,7")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download only the first time prefix from contiguous remote HDF5 result shards."
    )
    parser.add_argument("--base-url", required=True, help="Remote sampling directory URL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sampling-id", type=int, default=3)
    parser.add_argument("--shards", default=None, help="Comma-separated batch IDs; default is 0..39")
    parser.add_argument("--prefix-len", type=int, default=512)
    parser.add_argument(
        "--keys",
        default="all",
        help="all, orbital15, trajectory, or comma-separated HDF5 keys (times_complete is added automatically)",
    )
    parser.add_argument("--dtype", choices=["preserve", "float32"], default="preserve")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-ranges",
        type=int,
        default=64,
        help="Maximum byte ranges per HTTP request; 64 stays below common Apache header limits",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--metadata-block-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--max-rows", type=int, default=0, help="Debug-only cap per shard; 0 keeps all rows")
    parser.add_argument("--copy-completion-table", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.prefix_len <= 0 or args.workers <= 0 or args.max_ranges <= 0:
        raise ValueError("prefix_len, workers, and max_ranges must be positive")
    keys = parse_key_spec(args.keys)
    shards = parse_shards(args.shards)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = args.base_url.rstrip("/")
    summaries: list[dict[str, Any]] = []

    for batch_id in shards:
        filename = RESULT_NAME.format(batch=batch_id, sampling_id=args.sampling_id)
        url = f"{base_url}/{filename}"
        print(f"[inspect] {url}", flush=True)
        metadata = inspect_remote(url, block_size=args.metadata_block_size)
        row_count, time_steps = validate_selection(metadata, keys, args.prefix_len)
        estimated_rows = min(row_count, args.max_rows) if args.max_rows > 0 else row_count
        output_dtype = np.dtype("float32") if args.dtype == "float32" else metadata.datasets[keys[0]].dtype
        estimated_bytes = sum(
            estimated_rows
            * args.prefix_len
            * max(1, int(np.prod(metadata.datasets[key].tail_shape, dtype=np.int64)))
            * output_dtype.itemsize
            for key in keys
        )
        print(
            f"[shape] batch={batch_id} rows={row_count} T={time_steps} prefix={args.prefix_len} "
            f"keys={len(keys)} output={estimated_bytes / 1e9:.3f} GB",
            flush=True,
        )
        if args.dry_run:
            summaries.append(
                {
                    "batch": batch_id,
                    "url": url,
                    "rows": estimated_rows,
                    "time_steps": time_steps,
                    "estimated_data_bytes": int(estimated_bytes),
                }
            )
            continue
        summary = write_prefix_shard(
            metadata=metadata,
            output_path=output_dir / filename,
            keys=keys,
            prefix_len=args.prefix_len,
            dtype_mode=args.dtype,
            max_ranges=args.max_ranges,
            workers=args.workers,
            timeout=args.timeout,
            max_rows=args.max_rows if args.max_rows > 0 else None,
            overwrite=args.overwrite,
        )
        summary["batch"] = batch_id
        summaries.append(summary)

    if args.copy_completion_table and not args.dry_run:
        completion_name = f"completed_systems_sampling_{args.sampling_id}.txt"
        download_file(f"{base_url}/{completion_name}", output_dir / completion_name, args.overwrite)

    manifest = {
        "sampling_id": args.sampling_id,
        "base_url": base_url,
        "prefix_len": args.prefix_len,
        "keys": list(keys),
        "dtype": args.dtype,
        "shards": shards,
        "summaries": summaries,
    }
    (output_dir / "prefix_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] manifest written to {output_dir / 'prefix_manifest.json'}")


if __name__ == "__main__":
    main()
