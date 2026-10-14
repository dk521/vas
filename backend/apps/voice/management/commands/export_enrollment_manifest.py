"""Export a privacy-conscious manifest for local enrollment-data analysis."""

import csv
import io
import json
from pathlib import PurePosixPath

import boto3
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.voice.models import EnrollmentSample


class Command(BaseCommand):
    help = "Export enrollment metadata and embedding vectors as a CSV manifest in S3."

    def add_arguments(self, parser):
        parser.add_argument("--bucket", help="Destination S3 bucket. Defaults to AWS_STORAGE_BUCKET_NAME.")
        parser.add_argument(
            "--key",
            default="exports/vas-enrollment/enrollment_manifest.csv",
            help="Destination object key for the CSV manifest.",
        )
        parser.add_argument(
            "--include-identifiers",
            action="store_true",
            help="Also export a separate private participant lookup CSV containing usernames and emails.",
        )
        parser.add_argument(
            "--lookup-key",
            default="exports/vas-enrollment/participant_lookup.csv",
            help="Destination key for the private participant lookup CSV.",
        )

    def handle(self, *args, **options):
        bucket = options["bucket"] or getattr(settings, "AWS_STORAGE_BUCKET_NAME", "")
        if not bucket:
            raise CommandError("Provide --bucket or set AWS_STORAGE_BUCKET_NAME.")

        samples = list(
            EnrollmentSample.objects.select_related(
                "voice_profile__user", "enrollment_session", "pangram"
            ).order_by("voice_profile__user_id", "created_at", "id")
        )
        if not samples:
            raise CommandError("No enrollment samples were found.")

        participant_ids = {
            user_id: f"P{index:03d}"
            for index, user_id in enumerate(
                sorted({sample.voice_profile.user_id for sample in samples}), start=1
            )
        }
        fieldnames = [
            "sample_id",
            "participant_id",
            "session_id",
            "prompt_type",
            "pangram_id",
            "pangram_text",
            "audio_s3_key",
            "audio_local_path",
            "duration_seconds",
            "sample_rate",
            "model_version",
            "device",
            "created_at",
            "embedding_json",
        ]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()

        for sample in samples:
            participant_id = participant_ids[sample.voice_profile.user_id]
            extension = PurePosixPath(sample.audio_file.name).suffix.lower()
            audio_local_path = f"audio/{participant_id}/{sample.prompt_type}/sample-{sample.id}{extension}"
            writer.writerow(
                {
                    "sample_id": sample.id,
                    "participant_id": participant_id,
                    "session_id": sample.enrollment_session.session_id,
                    "prompt_type": sample.prompt_type,
                    "pangram_id": sample.pangram_id or "",
                    "pangram_text": sample.pangram.text if sample.pangram else "",
                    "audio_s3_key": sample.audio_file.name,
                    "audio_local_path": audio_local_path,
                    "duration_seconds": sample.duration_seconds,
                    "sample_rate": sample.sample_rate,
                    "model_version": sample.model_version,
                    "device": sample.device,
                    "created_at": sample.created_at.isoformat(),
                    "embedding_json": json.dumps(sample.embedding, separators=(",", ":")),
                }
            )

        s3 = boto3.client("s3", region_name=getattr(settings, "AWS_REGION", None))
        s3.put_object(
            Bucket=bucket,
            Key=options["key"],
            Body=buffer.getvalue().encode("utf-8"),
            ContentType="text/csv; charset=utf-8",
        )
        if options["include_identifiers"]:
            lookup_buffer = io.StringIO()
            lookup_writer = csv.DictWriter(
                lookup_buffer, fieldnames=["participant_id", "user_id", "username", "email"]
            )
            lookup_writer.writeheader()
            users_by_id = {sample.voice_profile.user_id: sample.voice_profile.user for sample in samples}
            for user_id in sorted(users_by_id):
                user = users_by_id[user_id]
                lookup_writer.writerow(
                    {
                        "participant_id": participant_ids[user_id],
                        "user_id": user_id,
                        "username": user.username,
                        "email": user.email,
                    }
                )
            s3.put_object(
                Bucket=bucket,
                Key=options["lookup_key"],
                Body=lookup_buffer.getvalue().encode("utf-8"),
                ContentType="text/csv; charset=utf-8",
            )
            self.stdout.write(
                self.style.WARNING(
                    f"Exported private participant lookup to s3://{bucket}/{options['lookup_key']}"
                )
            )
        self.stdout.write(
            self.style.SUCCESS(
                f"Exported {len(samples)} samples for {len(participant_ids)} participants to s3://{bucket}/{options['key']}"
            )
        )
