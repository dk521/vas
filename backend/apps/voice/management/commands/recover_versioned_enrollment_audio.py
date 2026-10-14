"""Recover enrollment audio that was overwritten under a repeated S3 key.

Older application versions stored browser recordings using the client filename.
Because every browser recording was named ``voice-recording.webm``, recordings in
the same session shared a key. In a versioned S3 bucket the older objects remain
available as non-current versions.
"""

from itertools import groupby
from pathlib import PurePosixPath

import boto3
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from apps.voice.models import EnrollmentSample


class Command(BaseCommand):
    help = (
        "Recover overwritten enrollment audio from S3 object versions. "
        "Runs as a dry run unless --apply is supplied."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--bucket",
            help="Versioned S3 bucket containing enrollment recordings. Defaults to AWS_STORAGE_BUCKET_NAME.",
        )
        parser.add_argument("--user-id", type=int, help="Recover samples for one user only.")
        parser.add_argument("--session-id", help="Recover samples for one enrollment session UUID only.")
        parser.add_argument(
            "--apply",
            action="store_true",
            help="Copy each historical version to a unique key and update EnrollmentSample.audio_file.",
        )

    def handle(self, *args, **options):
        bucket = options["bucket"] or getattr(settings, "AWS_STORAGE_BUCKET_NAME", "")
        if not bucket:
            raise CommandError("Provide --bucket or set AWS_STORAGE_BUCKET_NAME.")

        samples = EnrollmentSample.objects.select_related(
            "voice_profile__user", "enrollment_session"
        ).filter(audio_file__startswith="enrollment/")
        if options["user_id"]:
            samples = samples.filter(voice_profile__user_id=options["user_id"])
        if options["session_id"]:
            samples = samples.filter(enrollment_session__session_id=options["session_id"])

        samples = samples.order_by("audio_file", "created_at", "id")
        collisions = []
        for audio_key, group in groupby(samples, key=lambda sample: sample.audio_file.name):
            grouped_samples = list(group)
            if len(grouped_samples) > 1:
                collisions.append((audio_key, grouped_samples))

        if not collisions:
            self.stdout.write(self.style.WARNING("No repeated enrollment audio keys were found."))
            return

        s3 = boto3.client("s3", region_name=getattr(settings, "AWS_REGION", None))
        recovered = 0
        skipped = 0

        for audio_key, grouped_samples in collisions:
            versions = self._get_versions(s3, bucket, audio_key)
            if len(versions) != len(grouped_samples):
                skipped += 1
                self.stderr.write(
                    self.style.WARNING(
                        f"Skipping {audio_key}: found {len(grouped_samples)} database samples but "
                        f"{len(versions)} S3 object versions. No changes were made for this key."
                    )
                )
                continue

            self.stdout.write(
                f"{audio_key}: {len(grouped_samples)} samples and {len(versions)} S3 versions."
            )
            for sample, version in zip(grouped_samples, versions):
                new_key = self._recovered_key(sample, audio_key)
                action = f"sample {sample.id}: {version['VersionId']} -> {new_key}"
                if not options["apply"]:
                    self.stdout.write(f"  [DRY RUN] {action}")
                    continue

                # A server-side copy preserves the original version and avoids
                # downloading participants' recordings onto the operator's disk.
                s3.copy_object(
                    Bucket=bucket,
                    Key=new_key,
                    CopySource={
                        "Bucket": bucket,
                        "Key": audio_key,
                        "VersionId": version["VersionId"],
                    },
                )
                with transaction.atomic():
                    sample.audio_file.name = new_key
                    sample.save(update_fields=["audio_file"])
                recovered += 1
                self.stdout.write(self.style.SUCCESS(f"  Recovered {action}"))

        if not options["apply"]:
            self.stdout.write(
                self.style.WARNING("Dry run only. Review the mapping, then rerun with --apply to make changes.")
            )
        elif skipped:
            self.stdout.write(self.style.WARNING(f"Recovered {recovered} samples; skipped {skipped} key(s)."))
        else:
            self.stdout.write(self.style.SUCCESS(f"Recovered {recovered} enrollment recordings."))

    @staticmethod
    def _get_versions(s3, bucket, audio_key):
        versions = []
        paginator = s3.get_paginator("list_object_versions")
        for page in paginator.paginate(Bucket=bucket, Prefix=audio_key):
            versions.extend(version for version in page.get("Versions", []) if version["Key"] == audio_key)
        return sorted(versions, key=lambda version: (version["LastModified"], version["VersionId"]))

    @staticmethod
    def _recovered_key(sample, original_key):
        extension = PurePosixPath(original_key).suffix.lower()
        return (
            f"enrollment/{sample.voice_profile.user_id}/"
            f"{sample.enrollment_session.session_id}/sample-{sample.id}{extension}"
        )
