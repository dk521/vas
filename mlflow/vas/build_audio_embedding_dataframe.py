#!/usr/bin/env python3
"""Build a private Pandas DataFrame from local audio and ECAPA embeddings.

The output contains participant identity data and biometric embeddings. Keep it
local: the research_data directory is intentionally excluded from Git.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import mlflow
import numpy as np
import pandas as pd
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIRECTORY = PROJECT_ROOT / "research_data" / "vas_enrollment_2026"
SHARED_MLFLOW_DIRECTORY = PROJECT_ROOT / "mlflow" / "learning"
MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_DIRECTORY = PROJECT_ROOT / "backend" / "ml_models" / "ecapa"
DEFAULT_TRACKING_URI = f"sqlite:///{SHARED_MLFLOW_DIRECTORY / 'mlflow.db'}"
DEFAULT_ARTIFACT_LOCATION = (DATASET_DIRECTORY / "mlartifacts").resolve().as_uri()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DATASET_DIRECTORY / "enrollment_manifest.csv",
        help="CSV containing sample metadata and local audio paths.",
    )
    parser.add_argument(
        "--participants",
        type=Path,
        default=DATASET_DIRECTORY / "participant_lookup.csv",
        help="Private CSV mapping participant IDs to usernames and emails.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATASET_DIRECTORY / "private_ecapa_embedding_dataframe.pkl",
        help="Private pickle output. It preserves NumPy embedding arrays.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N rows; use this to run a small smoke test.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="Local SQLite MLflow tracking URI. Remote URIs are rejected for this private dataset.",
    )
    parser.add_argument(
        "--artifact-location",
        default=DEFAULT_ARTIFACT_LOCATION,
        help="Local directory where MLflow stores the private DataFrame artifact.",
    )
    parser.add_argument(
        "--experiment-name",
        default="vas-private-ecapa-embeddings",
        help="Local MLflow experiment name.",
    )
    return parser.parse_args()


def require_local_path(uri, allowed_schemes, allowed_root):
    """Prevent accidental upload of participant PII and biometric embeddings."""
    parsed = urlparse(uri)
    if parsed.scheme not in allowed_schemes:
        raise ValueError(
            "This builder contains PII and biometric embeddings. Use local SQLite and file URIs only."
        )
    local_path = Path(unquote(parsed.path)).resolve()
    allowed_root = allowed_root.resolve()
    if allowed_root not in (local_path, *local_path.parents):
        raise ValueError(f"MLflow path must stay inside {allowed_root}.")


def configure_local_experiment(experiment_name, artifact_location):
    """Create or reuse an experiment whose artifacts stay on this machine."""
    client = mlflow.MlflowClient()
    existing_experiment = client.get_experiment_by_name(experiment_name)
    if existing_experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)


def load_model():
    """Load ECAPA once; never load it once per audio recording."""
    return SpeakerRecognition.from_hparams(
        source=MODEL_SOURCE,
        savedir=str(MODEL_DIRECTORY),
        local_strategy=LocalStrategy.COPY,
    )


def convert_to_wav(audio_path):
    """Convert browser WebM audio to mono 16 kHz WAV, matching the backend."""
    temporary_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temporary_wav.close()
    wav_path = Path(temporary_wav.name)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                str(wav_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return wav_path
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        wav_path.unlink(missing_ok=True)
        message = getattr(error, "stderr", None) or str(error)
        raise RuntimeError(f"FFmpeg conversion failed for {audio_path}: {message}") from error


def generate_embedding(verification_model, wav_path):
    """Return the normalized 192-dimensional ECAPA embedding and duration."""
    signal = verification_model.load_audio(str(wav_path))
    batched_signal = signal.unsqueeze(0)
    embedding = verification_model.encode_batch(batched_signal)
    embedding = embedding.squeeze().detach().cpu().numpy().astype(np.float32)

    if embedding.shape != (192,):
        raise ValueError(f"Expected a 192-dimensional embedding, received {embedding.shape}")
    duration_seconds = float(signal.numel() / 16000)
    return embedding, duration_seconds


def build_embedding_dataframe(manifest_path, participants_path, limit=None):
    """Create one private DataFrame row per locally downloaded recording."""
    manifest = pd.read_csv(manifest_path)
    participants = pd.read_csv(participants_path)
    samples = manifest.merge(
        participants[["participant_id", "user_id", "username", "email"]],
        on="participant_id",
        how="left",
        validate="many_to_one",
    )
    if samples[["user_id", "username", "email"]].isna().any().any():
        missing = samples.loc[samples["email"].isna(), "participant_id"].unique().tolist()
        raise ValueError(f"No participant identity mapping for: {missing}")
    if limit is not None:
        samples = samples.head(limit)

    verification_model = load_model()
    records = []
    failures = []
    total = len(samples)

    for position, row in enumerate(samples.itertuples(index=False), start=1):
        audio_path = DATASET_DIRECTORY / row.audio_local_path
        wav_path = None
        try:
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            wav_path = convert_to_wav(audio_path)
            embedding, duration_seconds = generate_embedding(verification_model, wav_path)
            records.append(
                {
                    "sample_id": row.sample_id,
                    "participant_id": row.participant_id,
                    "user_id": row.user_id,
                    "user": row.username,
                    "email": row.email,
                    "speech_type": "pangrams" if row.prompt_type == "pangram" else "free_speech",
                    "duration_seconds": duration_seconds,
                    "audio_local_path": row.audio_local_path,
                    "embedding": embedding,
                    "embedding_dimension": int(embedding.size),
                    "model_version": MODEL_SOURCE,
                }
            )
        except Exception as error:  # Preserve the batch run; report failed files afterwards.
            failures.append({"sample_id": row.sample_id, "audio_local_path": row.audio_local_path, "error": str(error)})
        finally:
            if wav_path is not None:
                wav_path.unlink(missing_ok=True)

        if position % 10 == 0 or position == total:
            print(f"Processed {position}/{total} recordings; successful embeddings: {len(records)}")

    dataframe = pd.DataFrame(records)
    failure_dataframe = pd.DataFrame(failures)
    return dataframe, failure_dataframe


def main():
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    participants_path = args.participants.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    if not participants_path.is_file():
        raise SystemExit(f"Participant lookup not found: {participants_path}")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_path}. Use --overwrite to replace it.")
    try:
        require_local_path(args.tracking_uri, {"sqlite"}, PROJECT_ROOT)
        require_local_path(args.artifact_location, {"file"}, DATASET_DIRECTORY)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    mlflow.set_tracking_uri(args.tracking_uri)
    Path(unquote(urlparse(args.artifact_location).path)).mkdir(parents=True, exist_ok=True)
    configure_local_experiment(args.experiment_name, args.artifact_location)
    with mlflow.start_run(run_name="build-private-ecapa-dataframe"):
        mlflow.set_tags(
            {
                "data_classification": "private_pii_and_biometric_embeddings",
                "model_source": MODEL_SOURCE,
            }
        )
        mlflow.log_params(
            {
                "manifest_name": manifest_path.name,
                "participant_lookup_name": participants_path.name,
                "requested_limit": args.limit if args.limit is not None else "all",
                "audio_format": "webm_to_mono_16khz_wav",
                "embedding_dimension": 192,
            }
        )

        dataframe, failures = build_embedding_dataframe(manifest_path, participants_path, args.limit)
        if dataframe.empty:
            raise SystemExit("No embeddings were generated.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_pickle(output_path)
        mlflow.log_artifact(str(output_path), artifact_path="private_dataset")
        mlflow.log_metrics(
            {
                "recordings_requested": len(dataframe) + len(failures),
                "embeddings_generated": len(dataframe),
                "embedding_failures": len(failures),
                "participants": dataframe["participant_id"].nunique(),
                "free_speech_recordings": int((dataframe["speech_type"] == "free_speech").sum()),
                "pangram_recordings": int((dataframe["speech_type"] == "pangrams").sum()),
            }
        )
        print(f"Saved {len(dataframe)} private rows to {output_path}")
        print(f"Columns: {', '.join(dataframe.columns)}")

        if not failures.empty:
            failure_path = output_path.with_suffix(".failures.csv")
            failures.to_csv(failure_path, index=False)
            mlflow.log_artifact(str(failure_path), artifact_path="private_dataset")
            print(f"Failed recordings: {len(failures)}. Details: {failure_path}")

        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        print(f"MLflow artifact URI: {mlflow.get_artifact_uri()}")


if __name__ == "__main__":
    main()
