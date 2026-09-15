#!/usr/bin/env python3
"""Finalize deferred camera video spools for robologger episodes.

This script converts raw spool chunks under each camera *.zarr/_spool folder
into *.mp4 files inside the same zarr directory, updates zarr attrs to mark
the episode as finalized, and deletes the spool directory after successful
encoding.

Accepts one or more episode_* directories, or any parent directory (run / task /
project / root) which is searched recursively for episode_* dirs to finalize.
Per-episode failures are isolated: the batch continues and a summary is printed.

Usage:
    # single episode
    python3 scripts/data_tools/finalize_episode_video_spools.py /path/to/episode_000123
    # whole run (recursively finalizes every episode under it)
    python3 scripts/data_tools/finalize_episode_video_spools.py /path/to/run_001
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import zarr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="One or more episode_* dirs, or parent dirs (run/task/project/root) "
        "to search recursively for episodes",
    )
    parser.add_argument(
        "--codec",
        default="libx264",
        help="FFmpeg video codec to use for encoded MP4 output (default: libx264)",
    )
    parser.add_argument(
        "--bitrate",
        default="5M",
        help="FFmpeg target bitrate for encoded MP4 output (default: 5M)",
    )
    parser.add_argument(
        "--preset",
        default="fast",
        help="FFmpeg preset for encoded MP4 output (default: fast)",
    )
    parser.add_argument(
        "--keep-spool",
        action="store_true",
        help="Keep raw spool directories after successful encoding",
    )
    return parser.parse_args()


def load_attrs(zarr_dir: Path) -> dict:
    zattrs_path = zarr_dir / ".zattrs"
    if not zattrs_path.exists():
        raise FileNotFoundError(f"Missing zarr attrs file: {zattrs_path}")
    with open(zattrs_path, "r") as f:
        return json.load(f)


def iter_camera_zarrs(episode_dir: Path):
    for zarr_dir in sorted(episode_dir.glob("*.zarr")):
        if zarr_dir.name == "metadata.zarr":
            continue
        attrs = load_attrs(zarr_dir)
        camera_configs = attrs.get("camera_configs")
        if isinstance(camera_configs, dict):
            yield zarr_dir, attrs, camera_configs


def collect_spool_chunks(spool_dir: Path, camera_name: str):
    chunk_paths = sorted(spool_dir.glob(f"{camera_name}_chunk_*.bin"))
    if not chunk_paths:
        raise FileNotFoundError(f"No spool chunks found for {camera_name} in {spool_dir}")
    return chunk_paths


def encode_camera_spool(
    zarr_dir: Path,
    zarr_group: zarr.Group,
    spool_dir: Path,
    camera_name: str,
    config: dict,
    codec: str,
    bitrate: str,
    preset: str,
) -> None:
    width = int(config["width"])
    height = int(config["height"])
    fps = str(config["fps"])
    frame_bytes = width * height * 3
    mp4_path = zarr_dir / f"{camera_name}.mp4"

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        fps,
        "-i",
        "-",
        "-c:v",
        codec,
        "-pix_fmt",
        "yuv420p",
        "-preset",
        preset,
        "-b:v",
        bitrate,
        str(mp4_path),
    ]

    timestamp_key = f"{camera_name}_timestamps"
    expected_timestamps = None
    if timestamp_key in zarr_group:
        expected_timestamps = np.array(zarr_group[timestamp_key])

    process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    total_frames = 0

    try:
        for chunk_path in collect_spool_chunks(spool_dir, camera_name):
            chunk_id = chunk_path.stem.split("_")[-1]
            ts_path = spool_dir / f"{camera_name}_chunk_{chunk_id}_timestamps.npy"
            if not ts_path.exists():
                raise FileNotFoundError(f"Missing timestamps file for chunk: {ts_path}")

            raw = np.fromfile(chunk_path, dtype=np.uint8)
            if raw.size % frame_bytes != 0:
                raise ValueError(
                    f"Corrupt chunk {chunk_path}: {raw.size} bytes is not divisible by frame size {frame_bytes}"
                )

            frame_count = raw.size // frame_bytes
            chunk_timestamps = np.load(ts_path)
            if len(chunk_timestamps) != frame_count:
                raise ValueError(
                    f"Timestamp mismatch for {chunk_path}: {frame_count} frames but {len(chunk_timestamps)} timestamps"
                )

            if process.stdin is None:
                raise RuntimeError(f"FFmpeg stdin unavailable while encoding {camera_name}")

            process.stdin.write(raw.tobytes())
            total_frames += frame_count

        if process.stdin is not None:
            process.stdin.close()

        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg failed for {camera_name} in {zarr_dir}:\n{stderr}")

        if expected_timestamps is not None and len(expected_timestamps) != total_frames:
            raise ValueError(
                f"Frame count mismatch for {camera_name} in {zarr_dir}: "
                f"encoded {total_frames} frames but zarr has {len(expected_timestamps)} timestamps"
            )
    except Exception:
        process.kill()
        process.wait()
        if mp4_path.exists():
            mp4_path.unlink()
        raise
    finally:
        if process.stderr:
            process.stderr.close()


def finalize_camera_zarr(
    zarr_dir: Path,
    attrs: dict,
    camera_configs: dict,
    codec: str,
    bitrate: str,
    preset: str,
    keep_spool: bool,
) -> int:
    """Encode this camera zarr's spools to mp4. Returns the number of cameras encoded
    (0 when the spool dir is absent, i.e. already finalized)."""
    spool_dir = zarr_dir / attrs.get("video_spool_dir", "_spool")
    if not spool_dir.exists():
        print(f"Skipping {zarr_dir.name}: no spool directory")
        return 0

    zarr_group = zarr.open_group(zarr_dir, mode="a")

    for camera_name in camera_configs:
        print(f"Encoding {zarr_dir.name}/{camera_name} -> mp4")
        encode_camera_spool(
            zarr_dir=zarr_dir,
            zarr_group=zarr_group,
            spool_dir=spool_dir,
            camera_name=camera_name,
            config=camera_configs[camera_name],
            codec=codec,
            bitrate=bitrate,
            preset=preset,
        )

    zarr_group.attrs["video_storage_backend"] = "mp4_finalized_v1"
    zarr_group.attrs["video_encoding_deferred"] = False

    if not keep_spool:
        shutil.rmtree(spool_dir)
        print(f"Deleted spool: {spool_dir}")

    return len(camera_configs)


def dir_has_camera_zarr(path: Path) -> bool:
    """True if `path` directly contains a finalize-able camera .zarr."""
    try:
        return any(iter_camera_zarrs(path))
    except Exception:
        return False


def discover_episodes(paths) -> list[Path]:
    """Expand input paths into a de-duplicated, ordered list of episode dirs.

    A path that itself contains a camera .zarr is treated as a single episode;
    otherwise it is searched recursively for episode_* dirs containing camera .zarr.
    """
    episodes: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if not path.exists():
            print(f"Skipping {path}: does not exist")
            continue
        if dir_has_camera_zarr(path):
            candidates = [path]
        else:
            candidates = [
                p
                for p in sorted(path.rglob("episode_*"))
                if p.is_dir() and dir_has_camera_zarr(p)
            ]
        for ep in candidates:
            resolved = ep.resolve()
            if resolved not in seen:
                seen.add(resolved)
                episodes.append(ep)
    return episodes


def finalize_episode(episode_dir: Path, args: argparse.Namespace) -> str:
    """Finalize one episode. Returns "finalized" if any camera was encoded,
    "skipped" if there was nothing to encode (e.g. already finalized)."""
    found_camera_zarr = False
    encoded = 0
    for zarr_dir, attrs, camera_configs in iter_camera_zarrs(episode_dir):
        found_camera_zarr = True
        encoded += finalize_camera_zarr(
            zarr_dir=zarr_dir,
            attrs=attrs,
            camera_configs=camera_configs,
            codec=args.codec,
            bitrate=args.bitrate,
            preset=args.preset,
            keep_spool=args.keep_spool,
        )

    if not found_camera_zarr:
        return "skipped"
    return "finalized" if encoded else "skipped"


def main() -> None:
    args = parse_args()
    paths = [p.expanduser().resolve() for p in args.paths]

    episodes = discover_episodes(paths)
    if not episodes:
        print("No episodes found under: " + ", ".join(str(p) for p in paths))
        sys.exit(1)

    counts = {"finalized": 0, "skipped": 0, "failed": 0}
    for episode_dir in episodes:
        try:
            status = finalize_episode(episode_dir, args)
        except Exception as exc:
            counts["failed"] += 1
            print(f"FAILED {episode_dir.name}: {exc}")
            continue
        counts[status] += 1
        print(f"{status.upper()} {episode_dir.name}")

    total = len(episodes)
    print(
        f"\nFinalized {counts['finalized']}, skipped {counts['skipped']}, "
        f"failed {counts['failed']} of {total} episode(s)"
    )
    sys.exit(1 if counts["failed"] else 0)


if __name__ == "__main__":
    main()
