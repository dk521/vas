#!/usr/bin/env python3
"""Validate a locally downloaded VAS enrollment dataset.

Checks the manifest, local audio paths, embedded vectors, and decodability of
each recording with ffprobe. No files are modified.
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default="research_data/vas_enrollment_2026",
        help="Directory containing enrollment_manifest.csv and audio/.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Concurrent ffprobe processes.")
    return parser.parse_args()


def probe_audio(row, dataset_dir):
    audio_path = dataset_dir / row["audio_local_path"]
    result = {
        "sample_id": row["sample_id"],
        "participant_id": row["participant_id"],
        "prompt_type": row["prompt_type"],
        "audio_local_path": row["audio_local_path"],
        "status": "ok",
        "duration_seconds": "",
        "message": "",
    }
    if not audio_path.is_file():
        result.update(status="error", message="Audio file is missing")
        return result
    if audio_path.stat().st_size == 0:
        result.update(status="error", message="Audio file is empty")
        return result

    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
        "-of", "json", str(audio_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        result.update(status="error", message=completed.stderr.strip() or "ffprobe could not read file")
        return result
    try:
        probe = json.loads(completed.stdout)
        duration = float(probe["format"]["duration"])
        has_audio = any(stream.get("codec_type") == "audio" for stream in probe.get("streams", []))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        result.update(status="error", message=f"Invalid ffprobe output: {error}")
        return result
    if not has_audio or duration <= 0:
        result.update(status="error", message="No playable audio stream or zero duration")
        return result

    result["duration_seconds"] = f"{duration:.3f}"
    try:
        expected_duration = float(row["duration_seconds"])
        if abs(duration - expected_duration) > 1.0:
            result.update(status="warning", message=f"Duration differs from manifest ({expected_duration:.3f}s)")
    except ValueError:
        result.update(status="warning", message="Manifest duration is invalid")
    return result


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    manifest_path = dataset_dir / "enrollment_manifest.csv"
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    if not shutil.which("ffprobe"):
        raise SystemExit("ffprobe is required. Install FFmpeg first (for macOS: brew install ffmpeg).")

    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    if not rows:
        raise SystemExit("The manifest has no samples.")

    manifest_errors = []
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        manifest_errors.append("Duplicate sample IDs in manifest")
    for row in rows:
        try:
            embedding = json.loads(row["embedding_json"])
            if not isinstance(embedding, list) or not embedding:
                raise ValueError("embedding is not a non-empty list")
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            manifest_errors.append(f"Sample {row['sample_id']}: invalid embedding ({error})")

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(lambda row: probe_audio(row, dataset_dir), rows))

    report_path = dataset_dir / "validation_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    status_counts = Counter(result["status"] for result in results)
    prompt_counts = Counter((row["participant_id"], row["prompt_type"]) for row in rows)
    print(f"Manifest samples: {len(rows)}")
    print(f"Audio checks: {status_counts['ok']} ok, {status_counts['warning']} warning, {status_counts['error']} error")
    print(f"Validation report: {report_path}")
    print("Participant sample counts:")
    for participant_id in sorted({row["participant_id"] for row in rows}):
        free_speech = prompt_counts[(participant_id, "free_speech")]
        pangram = prompt_counts[(participant_id, "pangram")]
        print(f"  {participant_id}: free_speech={free_speech}, pangram={pangram}")

    if manifest_errors:
        print("Manifest errors:", *manifest_errors, sep="\n  ", file=sys.stderr)
    if manifest_errors or status_counts["error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
