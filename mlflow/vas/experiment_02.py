"""Experiment 02: repeated centroid verification, Free Speech vs Pangrams.

This repeats the Experiment 01 protocol with deterministic random splits from
seed 42 through 84 inclusive.  Each repetition uses six recordings to build a
participant centroid and four unseen recordings as queries.

The parent MLflow run contains the average and standard deviation across all
repetitions.  Nested runs contain the metrics for one individual random split.
"""

from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATAFRAME_PATH = (
    PROJECT_ROOT
    / "research_data"
    / "vas_enrollment_2026"
    / "private_ecapa_embedding_dataframe.pkl"
)

RANDOM_STATES = range(42, 85)  # 42, 43, ..., 84: 43 reproducible repetitions.
ENROLLMENT_COUNT = 6
TARGET_FAR = 0.01


def cosine_similarity(vector_a, vector_b):
    """Cosine similarity; a higher value means the speakers are more similar."""
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )


def build_centroid(enrollment_embeddings):
    """Average six ECAPA embeddings, then normalize the result to unit length."""
    centroid = np.vstack(enrollment_embeddings).mean(axis=0)
    return centroid / np.linalg.norm(centroid)


def get_paired_prompt_frames(dataframe):
    """Keep only people with ten recordings in both prompt-type conditions."""
    free_speech = dataframe[dataframe["speech_type"] == "free_speech"].copy()
    pangrams = dataframe[dataframe["speech_type"] == "pangrams"].copy()

    free_complete_ids = set(
        free_speech.groupby("user_id").filter(lambda group: len(group) == 10)["user_id"]
    )
    pangram_complete_ids = set(
        pangrams.groupby("user_id").filter(lambda group: len(group) == 10)["user_id"]
    )
    paired_user_ids = sorted(free_complete_ids & pangram_complete_ids)

    return (
        free_speech[free_speech["user_id"].isin(paired_user_ids)],
        pangrams[pangrams["user_id"].isin(paired_user_ids)],
        paired_user_ids,
    )


def make_splits(prompt_dataframe, random_state):
    """For each participant select six of ten recordings without replacement."""
    rng = np.random.default_rng(random_state)
    splits = {}

    for user_id, rows in prompt_dataframe.groupby("user_id"):
        enrollment_index = rng.choice(
            rows.index.to_numpy(),
            size=ENROLLMENT_COUNT,
            replace=False,
        )
        splits[user_id] = {
            "enroll": rows.loc[enrollment_index],
            "test": rows.drop(enrollment_index),
        }

    return splits


def score_trials(splits, speech_type, random_state):
    """Create every genuine and impostor trial for one prompt type and split."""
    trials = []

    for claimed_user_id, claimed_split in splits.items():
        centroid = build_centroid(claimed_split["enroll"]["embedding"])

        # Genuine queries: the claimed participant's held-out recordings.
        for _, query in claimed_split["test"].iterrows():
            trials.append({
                "random_state": random_state,
                "speech_type": speech_type,
                "claimed_user_id": claimed_user_id,
                "query_user_id": claimed_user_id,
                "trial_type": "genuine",
                "query_duration_seconds": query["duration_seconds"],
                "score": cosine_similarity(centroid, query["embedding"]),
            })

        # Impostor queries: every other participant's held-out recordings.
        for query_user_id, query_split in splits.items():
            if query_user_id == claimed_user_id:
                continue

            for _, query in query_split["test"].iterrows():
                trials.append({
                    "random_state": random_state,
                    "speech_type": speech_type,
                    "claimed_user_id": claimed_user_id,
                    "query_user_id": query_user_id,
                    "trial_type": "impostor",
                    "query_duration_seconds": query["duration_seconds"],
                    "score": cosine_similarity(centroid, query["embedding"]),
                })

    return pd.DataFrame(trials)


def calculate_metrics(trials):
    """Calculate EER, ROC-AUC, and TAR at the desired maximum FAR."""
    genuine_scores = trials.loc[trials["trial_type"] == "genuine", "score"].to_numpy()
    impostor_scores = trials.loc[trials["trial_type"] == "impostor", "score"].to_numpy()

    thresholds = np.concatenate([
        [np.inf],
        np.sort(np.unique(np.concatenate([genuine_scores, impostor_scores])))[::-1],
        [-np.inf],
    ])

    sweep_rows = []
    for threshold in thresholds:
        far = np.mean(impostor_scores >= threshold)
        frr = np.mean(genuine_scores < threshold)
        tar = np.mean(genuine_scores >= threshold)
        sweep_rows.append({
            "threshold": threshold,
            "far": far,
            "frr": frr,
            "tar": tar,
            "far_frr_gap": abs(far - frr),
        })

    threshold_sweep = pd.DataFrame(sweep_rows)
    eer_row = threshold_sweep.loc[threshold_sweep["far_frr_gap"].idxmin()]
    secure_rows = threshold_sweep[threshold_sweep["far"] <= TARGET_FAR]
    secure_row = secure_rows.loc[secure_rows["tar"].idxmax()]

    # Thresholds are descending, so FAR is ascending and can be integrated.
    roc_auc = np.trapezoid(threshold_sweep["tar"], threshold_sweep["far"])

    return {
        "genuine_trials": len(genuine_scores),
        "impostor_trials": len(impostor_scores),
        "genuine_mean": genuine_scores.mean(),
        "genuine_std": genuine_scores.std(),
        "impostor_mean": impostor_scores.mean(),
        "impostor_std": impostor_scores.std(),
        "eer": (eer_row["far"] + eer_row["frr"]) / 2,
        "eer_threshold": eer_row["threshold"],
        "far_at_eer": eer_row["far"],
        "frr_at_eer": eer_row["frr"],
        "roc_auc": roc_auc,
        "tar_at_1pct_far": secure_row["tar"],
        "threshold_at_1pct_far": secure_row["threshold"],
    }


def log_repeat_stability_plot(metrics_dataframe):
    """Show whether EER and AUC remain stable across deterministic splits."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)

    for speech_type, group in metrics_dataframe.groupby("speech_type"):
        axes[0].plot(group["random_state"], group["eer"], marker="o", label=speech_type)
        axes[1].plot(group["random_state"], group["roc_auc"], marker="o", label=speech_type)

    axes[0].set_title("EER Stability Across Splits")
    axes[0].set_xlabel("Random seed")
    axes[0].set_ylabel("EER (lower is better)")
    axes[0].legend()
    axes[0].grid()

    axes[1].set_title("ROC-AUC Stability Across Splits")
    axes[1].set_xlabel("Random seed")
    axes[1].set_ylabel("ROC-AUC (higher is better)")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid()

    fig.tight_layout()
    mlflow.log_figure(fig, "plots/repeat_stability.png")
    plt.close(fig)


def log_metric_distribution_plot(metrics_dataframe):
    """Compare the distribution of outcomes from all 43 repetitions."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    prompt_order = ["free_speech", "pangrams"]

    axes[0].boxplot(
        [
            metrics_dataframe.loc[metrics_dataframe["speech_type"] == prompt, "eer"]
            for prompt in prompt_order
        ],
        tick_labels=["Free Speech", "Pangrams"],
    )
    axes[0].set_title("EER Across Repetitions")
    axes[0].set_ylabel("EER (lower is better)")
    axes[0].grid()

    axes[1].boxplot(
        [
            metrics_dataframe.loc[metrics_dataframe["speech_type"] == prompt, "roc_auc"]
            for prompt in prompt_order
        ],
        tick_labels=["Free Speech", "Pangrams"],
    )
    axes[1].set_title("ROC-AUC Across Repetitions")
    axes[1].set_ylabel("ROC-AUC (higher is better)")
    axes[1].set_ylim(0, 1)
    axes[1].grid()

    fig.tight_layout()
    mlflow.log_figure(fig, "plots/repeated_split_metric_distributions.png")
    plt.close(fig)


def run_experiment():
    dataframe = pd.read_pickle(DATAFRAME_PATH)
    dataframe = dataframe[["user_id", "speech_type", "duration_seconds", "embedding"]]

    free_speech, pangrams, paired_user_ids = get_paired_prompt_frames(dataframe)
    prompt_frames = {
        "free_speech": free_speech,
        "pangrams": pangrams,
    }

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("vas_experiment_02_repeated_splits")

    metric_rows = []
    all_trials = []

    with mlflow.start_run(run_name="centroid-prompt-comparison-seeds-42-to-84"):
        mlflow.log_params({
            "seed_start": min(RANDOM_STATES),
            "seed_end": max(RANDOM_STATES),
            "repetitions": len(RANDOM_STATES),
            "paired_participant_count": len(paired_user_ids),
            "enrollment_recordings_per_user": ENROLLMENT_COUNT,
            "test_recordings_per_user": 10 - ENROLLMENT_COUNT,
            "target_far": TARGET_FAR,
        })

        for random_state in RANDOM_STATES:
            with mlflow.start_run(run_name=f"seed-{random_state}", nested=True):
                seed_metrics = {}

                for speech_type, prompt_dataframe in prompt_frames.items():
                    splits = make_splits(prompt_dataframe, random_state)
                    trials = score_trials(splits, speech_type, random_state)
                    metrics = calculate_metrics(trials)

                    metric_rows.append({
                        "random_state": random_state,
                        "speech_type": speech_type,
                        **metrics,
                    })
                    all_trials.append(trials)
                    seed_metrics.update({
                        f"{speech_type}_{metric}": value
                        for metric, value in metrics.items()
                    })

                mlflow.log_param("random_state", random_state)
                mlflow.log_metrics(seed_metrics)

        metrics_dataframe = pd.DataFrame(metric_rows)
        trials_dataframe = pd.concat(all_trials, ignore_index=True)

        summary = (
            metrics_dataframe.groupby("speech_type")
            .agg(
                eer_mean=("eer", "mean"),
                eer_std=("eer", "std"),
                roc_auc_mean=("roc_auc", "mean"),
                roc_auc_std=("roc_auc", "std"),
                tar_at_1pct_far_mean=("tar_at_1pct_far", "mean"),
                tar_at_1pct_far_std=("tar_at_1pct_far", "std"),
            )
            .reset_index()
        )

        # Log the parent-run metrics that you compare in the MLflow UI.
        parent_metrics = {}
        for speech_type, group in metrics_dataframe.groupby("speech_type"):
            for metric in ["eer", "roc_auc", "tar_at_1pct_far"]:
                parent_metrics[f"{speech_type}_{metric}_mean"] = group[metric].mean()
                parent_metrics[f"{speech_type}_{metric}_std"] = group[metric].std()
        mlflow.log_metrics(parent_metrics)

        # These artifacts retain the score-level data needed for later analysis.
        mlflow.log_table(metrics_dataframe, "tables/repeat_metrics.json")
        mlflow.log_table(summary, "tables/metric_summary.json")
        with tempfile.TemporaryDirectory() as temporary_directory:
            trial_path = Path(temporary_directory) / "trial_scores.csv"
            trials_dataframe.to_csv(trial_path, index=False)
            mlflow.log_artifact(str(trial_path), artifact_path="tables")

        log_repeat_stability_plot(metrics_dataframe)
        log_metric_distribution_plot(metrics_dataframe)

        print("\nAverage metrics across seeds 42–84:")
        print(
            metrics_dataframe.groupby("speech_type")[
                ["eer", "roc_auc", "tar_at_1pct_far"]
            ].agg(["mean", "std"])
        )


if __name__ == "__main__":
    run_experiment()
