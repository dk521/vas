#!/usr/bin/env python3
"""Download a VAS enrollment manifest and its audio into an analysis-ready layout."""

import argparse
import csv
from pathlib import Path

import boto3


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True, help="S3 bucket containing recordings and manifest.")
    parser.add_argument(
        "--manifest-key",
        default="exports/vas-enrollment/enrollment_manifest.csv",
        help="S3 key of the manifest exported by Django.",
    )
    parser.add_argument(
        "--include-identifiers",
        action="store_true",
        help="Also download the separately exported private participant lookup CSV.",
    )
    parser.add_argument(
        "--lookup-key",
        default="exports/vas-enrollment/participant_lookup.csv",
        help="S3 key of the private participant lookup CSV.",
    )
    parser.add_argument("--output-dir", required=True, help="Local directory for the analysis dataset.")
    parser.add_argument("--profile", default="vas-deploy-profile", help="AWS CLI/SSO profile name.")
    parser.add_argument("--region", default="ap-south-1", help="AWS Region.")
    parser.add_argument("--overwrite", action="store_true", help="Redownload files already present locally.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "enrollment_manifest.csv"

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    s3 = session.client("s3")
    s3.download_file(args.bucket, args.manifest_key, str(manifest_path))
    if args.include_identifiers:
        s3.download_file(args.bucket, args.lookup_key, str(output_dir / "participant_lookup.csv"))

    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    if not rows:
        raise RuntimeError("The manifest has no enrollment samples.")

    downloaded = skipped = 0
    for row in rows:
        destination = output_dir / row["audio_local_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not args.overwrite:
            skipped += 1
            continue
        s3.download_file(args.bucket, row["audio_s3_key"], str(destination))
        downloaded += 1

    print(f"Manifest: {manifest_path}")
    if args.include_identifiers:
        print(f"Private lookup: {output_dir / 'participant_lookup.csv'}")
    print(f"Samples: {len(rows)} | downloaded: {downloaded} | already present: {skipped}")
    print(f"Audio directory: {output_dir / 'audio'}")


if __name__ == "__main__":
    main()
