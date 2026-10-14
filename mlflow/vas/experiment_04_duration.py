"""Experiment 04: does recording duration affect verification performance?

For each held-out query recording, this experiment calculates:
  - genuine score: query vs its own speaker centroid
  - max impostor score: query vs the most similar other-speaker centroid
  - score margin: genuine score minus max impostor score

It then evaluates duration policies such as "require at least 4 seconds".
"""

import mlflow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_02 import (
    DATAFRAME_PATH,
    RANDOM_STATES,
    build_centroid,
    cosine_similarity,
    get_paired_prompt_frames,
    make_splits,
)
from experiment_03 import calculate_metrics_from_scores


MINIMUM_DURATIONS = [0, 1, 2, 3, 4, 5]


def build_duration_trials():
    """Re-run the 43 deterministic splits while retaining query duration."""
    dataframe = pd.read_pickle(DATAFRAME_PATH)
    dataframe = dataframe[
        ["sample_id", "user_id", "speech_type", "duration_seconds", "embedding"]
    ]
    free_speech, pangrams, paired_user_ids = get_paired_prompt_frames(dataframe)

    score_rows = []
    query_rows = []

    for random_state in RANDOM_STATES:
        for speech_type, prompt_dataframe in {
            "free_speech": free_speech,
            "pangrams": pangrams,
        }.items():
            splits = make_splits(prompt_dataframe, random_state)
            centroids = {
                user_id: build_centroid(split["enroll"]["embedding"])
                for user_id, split in splits.items()
            }

            for owner_user_id, split in splits.items():
                for _, query in split["test"].iterrows():
                    genuine_score = cosine_similarity(
                        centroids[owner_user_id], query["embedding"]
                    )
                    impostor_scores = []

                    for claimed_user_id, centroid in centroids.items():
                        if claimed_user_id == owner_user_id:
                            continue

                        impostor_score = cosine_similarity(centroid, query["embedding"])
                        impostor_scores.append(impostor_score)
                        score_rows.append({
                            "random_state": random_state,
                            "speech_type": speech_type,
                            "query_sample_id": query["sample_id"],
                            "query_user_id": owner_user_id,
                            "query_duration_seconds": query["duration_seconds"],
                            "trial_type": "impostor",
                            "score": impostor_score,
                        })

                    score_rows.append({
                        "random_state": random_state,
                        "speech_type": speech_type,
                        "query_sample_id": query["sample_id"],
                        "query_user_id": owner_user_id,
                        "query_duration_seconds": query["duration_seconds"],
                        "trial_type": "genuine",
                        "score": genuine_score,
                    })
                    query_rows.append({
                        "random_state": random_state,
                        "speech_type": speech_type,
                        "query_sample_id": query["sample_id"],
                        "query_user_id": owner_user_id,
                        "duration_seconds": query["duration_seconds"],
                        "genuine_score": genuine_score,
                        "max_impostor_score": max(impostor_scores),
                        "score_margin": genuine_score - max(impostor_scores),
                    })

    return (
        pd.DataFrame(score_rows),
        pd.DataFrame(query_rows),
        paired_user_ids,
    )


def make_duration_policy_table(score_trials):
    """Evaluate EER/AUC/TAR after applying each minimum-duration policy."""
    policy_rows = []

    for speech_type, speech_trials in score_trials.groupby("speech_type"):
        total_unique_queries = speech_trials["query_sample_id"].nunique()

        for minimum_duration in MINIMUM_DURATIONS:
            included_trials = speech_trials[
                speech_trials["query_duration_seconds"] >= minimum_duration
            ]
            included_queries = included_trials["query_sample_id"].nunique()

            genuine_scores = included_trials.loc[
                included_trials["trial_type"] == "genuine", "score"
            ].to_numpy()
            impostor_scores = included_trials.loc[
                included_trials["trial_type"] == "impostor", "score"
            ].to_numpy()

            metrics = calculate_metrics_from_scores(genuine_scores, impostor_scores)
            policy_rows.append({
                "speech_type": speech_type,
                "minimum_duration_seconds": minimum_duration,
                "unique_query_coverage": included_queries / total_unique_queries,
                "genuine_trials": len(genuine_scores),
                "impostor_trials": len(impostor_scores),
                **metrics,
            })

    return pd.DataFrame(policy_rows)


def make_duration_correlation_table(query_summary):
    """Correlate duration with query-level quality after averaging split repeats."""
    # A recording appears in multiple random test splits. Average it first so
    # frequently selected recordings do not receive extra statistical weight.
    sample_level = (
        query_summary.groupby(["speech_type", "query_sample_id"], as_index=False)
        .agg(
            duration_seconds=("duration_seconds", "first"),
            genuine_score=("genuine_score", "mean"),
            max_impostor_score=("max_impostor_score", "mean"),
            score_margin=("score_margin", "mean"),
        )
    )

    correlation_rows = []
    for speech_type, group in sample_level.groupby("speech_type"):
        for outcome in ["genuine_score", "max_impostor_score", "score_margin"]:
            correlation_rows.append({
                "speech_type": speech_type,
                "outcome": outcome,
                "pearson_correlation": group["duration_seconds"].corr(group[outcome]),
                "sample_count": len(group),
            })

    return sample_level, pd.DataFrame(correlation_rows)


def log_duration_scatter_plot(sample_level):
    """Visualize whether longer utterances give larger genuine-impostor margins."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for axis, (speech_type, group) in zip(
        axes, sample_level.groupby("speech_type", sort=True)
    ):
        axis.scatter(group["duration_seconds"], group["score_margin"], alpha=0.7)
        slope, intercept = np.polyfit(
            group["duration_seconds"], group["score_margin"], 1
        )
        x_values = np.linspace(group["duration_seconds"].min(), group["duration_seconds"].max(), 100)
        axis.plot(x_values, slope * x_values + intercept, color="tab:red")
        axis.set_title(speech_type.replace("_", " ").title())
        axis.set_xlabel("Query duration (seconds)")
        axis.grid()

    axes[0].set_ylabel("Genuine score − maximum impostor score")
    fig.suptitle("Duration vs Query-Level Verification Margin")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/duration_vs_score_margin.png")
    plt.close(fig)


def log_duration_policy_plot(policy_table):
    """Show the accuracy-versus-coverage trade-off of a duration requirement."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)

    for speech_type, group in policy_table.groupby("speech_type"):
        axes[0].plot(
            group["minimum_duration_seconds"], group["eer"], marker="o", label=speech_type
        )
        axes[1].plot(
            group["minimum_duration_seconds"], group["unique_query_coverage"], marker="o", label=speech_type
        )

    axes[0].set_title("Verification Error vs Duration Requirement")
    axes[0].set_xlabel("Minimum query duration (seconds)")
    axes[0].set_ylabel("EER (lower is better)")
    axes[0].legend()
    axes[0].grid()

    axes[1].set_title("Participant Recording Coverage")
    axes[1].set_xlabel("Minimum query duration (seconds)")
    axes[1].set_ylabel("Fraction of recordings retained")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid()

    fig.tight_layout()
    mlflow.log_figure(fig, "plots/duration_policy_tradeoff.png")
    plt.close(fig)


def run_experiment():
    score_trials, query_summary, participant_ids = build_duration_trials()
    policy_table = make_duration_policy_table(score_trials)
    sample_level, correlation_table = make_duration_correlation_table(query_summary)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("vas_experiment_04_duration")

    with mlflow.start_run(run_name="duration-policy-analysis"):
        mlflow.log_params({
            "paired_participant_count": len(participant_ids),
            "split_repetitions": len(RANDOM_STATES),
            "duration_policies_seconds": ",".join(map(str, MINIMUM_DURATIONS)),
        })
        mlflow.log_table(policy_table, "tables/duration_policy_metrics.json")
        mlflow.log_table(correlation_table, "tables/duration_correlations.json")
        mlflow.log_table(sample_level, "tables/query_level_duration_summary.json")

        log_duration_scatter_plot(sample_level)
        log_duration_policy_plot(policy_table)

        print("\nDuration correlations:")
        print(correlation_table.to_string(index=False))
        print("\nDuration policy metrics:")
        print(policy_table.to_string(index=False))


if __name__ == "__main__":
    run_experiment()
