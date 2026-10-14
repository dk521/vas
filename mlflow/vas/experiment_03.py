"""Experiment 03: paired participant-bootstrap confidence intervals.

Experiment 02 measures how sensitive results are to different 6/4 recording
splits. This experiment measures uncertainty from having only a finite set of
participants. It resamples the same paired participants for Free Speech and
Pangrams, then reports bootstrap confidence intervals for their difference.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from experiment_02 import (
    DATAFRAME_PATH,
    RANDOM_STATES,
    get_paired_prompt_frames,
    make_splits,
    score_trials,
)


BOOTSTRAP_SAMPLES = 1_000
BOOTSTRAP_RANDOM_STATE = 20260917
TARGET_FAR = 0.01


def calculate_metrics_from_scores(genuine_scores, impostor_scores):
    """Compute verification metrics with vectorized threshold sweeping.

    This is equivalent to the Experiment 02 calculation, but fast enough to
    repeat one thousand times during bootstrap sampling.
    """
    genuine_scores = np.asarray(genuine_scores, dtype=float)
    impostor_scores = np.asarray(impostor_scores, dtype=float)

    thresholds = np.concatenate([
        [np.inf],
        np.sort(np.unique(np.concatenate([genuine_scores, impostor_scores])))[::-1],
        [-np.inf],
    ])

    genuine_sorted = np.sort(genuine_scores)
    impostor_sorted = np.sort(impostor_scores)

    # acceptance rule: score >= threshold
    frr = np.searchsorted(genuine_sorted, thresholds, side="left") / len(genuine_scores)
    far = (
        len(impostor_scores)
        - np.searchsorted(impostor_sorted, thresholds, side="left")
    ) / len(impostor_scores)
    tar = 1 - frr

    eer_index = np.abs(far - frr).argmin()
    secure_index = np.where(far <= TARGET_FAR)[0][np.argmax(tar[far <= TARGET_FAR])]

    return {
        "eer": float((far[eer_index] + frr[eer_index]) / 2),
        "eer_threshold": float(thresholds[eer_index]),
        "roc_auc": float(np.trapezoid(tar, far)),
        "tar_at_1pct_far": float(tar[secure_index]),
        "threshold_at_1pct_far": float(thresholds[secure_index]),
    }


def build_repeated_trial_data():
    """Re-create Experiment 02's 43 deterministic split evaluations."""
    dataframe = pd.read_pickle(DATAFRAME_PATH)
    dataframe = dataframe[["user_id", "speech_type", "duration_seconds", "embedding"]]
    free_speech, pangrams, paired_user_ids = get_paired_prompt_frames(dataframe)

    all_trials = []
    for random_state in RANDOM_STATES:
        for speech_type, prompt_dataframe in {
            "free_speech": free_speech,
            "pangrams": pangrams,
        }.items():
            splits = make_splits(prompt_dataframe, random_state)
            all_trials.append(score_trials(splits, speech_type, random_state))

    return pd.concat(all_trials, ignore_index=True), paired_user_ids


def make_participant_clusters(trials, participant_ids):
    """Group all repeated-split scores by claimed participant and prompt type."""
    clusters = {"free_speech": {}, "pangrams": {}}

    for speech_type in clusters:
        speech_trials = trials[trials["speech_type"] == speech_type]
        for user_id in participant_ids:
            user_trials = speech_trials[speech_trials["claimed_user_id"] == user_id]
            clusters[speech_type][user_id] = {
                "genuine": user_trials.loc[
                    user_trials["trial_type"] == "genuine", "score"
                ].to_numpy(),
                "impostor": user_trials.loc[
                    user_trials["trial_type"] == "impostor", "score"
                ].to_numpy(),
            }

    return clusters


def metrics_for_participant_sample(clusters, sampled_participant_ids):
    """Calculate both arms after sampling the same participant IDs with replacement."""
    results = {}

    for speech_type, speech_clusters in clusters.items():
        genuine_scores = np.concatenate([
            speech_clusters[user_id]["genuine"]
            for user_id in sampled_participant_ids
        ])
        impostor_scores = np.concatenate([
            speech_clusters[user_id]["impostor"]
            for user_id in sampled_participant_ids
        ])
        results[speech_type] = calculate_metrics_from_scores(
            genuine_scores,
            impostor_scores,
        )

    return results


def bootstrap_comparison(clusters, participant_ids):
    """Generate paired bootstrap deltas: Pangrams minus Free Speech."""
    rng = np.random.default_rng(BOOTSTRAP_RANDOM_STATE)
    rows = []

    for bootstrap_index in range(BOOTSTRAP_SAMPLES):
        sampled_ids = rng.choice(participant_ids, size=len(participant_ids), replace=True)
        results = metrics_for_participant_sample(clusters, sampled_ids)
        free = results["free_speech"]
        pangrams = results["pangrams"]

        rows.append({
            "bootstrap_index": bootstrap_index,
            "free_speech_eer": free["eer"],
            "pangrams_eer": pangrams["eer"],
            "delta_eer": pangrams["eer"] - free["eer"],
            "free_speech_roc_auc": free["roc_auc"],
            "pangrams_roc_auc": pangrams["roc_auc"],
            "delta_roc_auc": pangrams["roc_auc"] - free["roc_auc"],
            "free_speech_tar_at_1pct_far": free["tar_at_1pct_far"],
            "pangrams_tar_at_1pct_far": pangrams["tar_at_1pct_far"],
            "delta_tar_at_1pct_far": (
                pangrams["tar_at_1pct_far"] - free["tar_at_1pct_far"]
            ),
        })

    return pd.DataFrame(rows)


def create_confidence_interval_summary(point_estimate, bootstrap_results):
    """Summarize 95% percentile intervals for each paired prompt-type delta."""
    summaries = []
    for metric, interpretation in [
        ("eer", "negative favors pangrams"),
        ("roc_auc", "positive favors pangrams"),
        ("tar_at_1pct_far", "positive favors pangrams"),
    ]:
        delta_column = f"delta_{metric}"
        lower, upper = np.percentile(bootstrap_results[delta_column], [2.5, 97.5])
        point_delta = (
            point_estimate["pangrams"][metric]
            - point_estimate["free_speech"][metric]
        )

        summaries.append({
            "metric": metric,
            "point_delta_pangrams_minus_free_speech": point_delta,
            "ci_95_low": lower,
            "ci_95_high": upper,
            "zero_in_ci": lower <= 0 <= upper,
            "interpretation": interpretation,
        })

    return pd.DataFrame(summaries)


def log_bootstrap_delta_distributions(bootstrap_results, interval_summary):
    """Show whether each paired metric difference is consistently above/below zero."""
    metric_details = [
        ("delta_eer", "EER difference\n(Pangrams − Free Speech)"),
        ("delta_roc_auc", "ROC-AUC difference\n(Pangrams − Free Speech)"),
        ("delta_tar_at_1pct_far", "TAR @ 1% FAR difference\n(Pangrams − Free Speech)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for axis, (column, title) in zip(axes, metric_details):
        metric = column.removeprefix("delta_")
        summary = interval_summary[interval_summary["metric"] == metric].iloc[0]
        axis.hist(bootstrap_results[column], bins=30, alpha=0.75)
        axis.axvline(0, color="black", linestyle="--", label="No difference")
        axis.axvline(
            summary["point_delta_pangrams_minus_free_speech"],
            color="tab:red",
            label="Point estimate",
        )
        axis.axvspan(summary["ci_95_low"], summary["ci_95_high"], alpha=0.2)
        axis.set_title(title)
        axis.set_xlabel("Difference")
        axis.set_ylabel("Bootstrap samples")
        axis.legend()
        axis.grid()

    fig.suptitle("Paired Participant-Bootstrap Confidence Intervals")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/paired_bootstrap_delta_distributions.png")
    plt.close(fig)


def run_experiment():
    trials, participant_ids = build_repeated_trial_data()
    clusters = make_participant_clusters(trials, participant_ids)

    # Point estimates use every paired participant exactly once.
    point_estimate = metrics_for_participant_sample(clusters, participant_ids)
    bootstrap_results = bootstrap_comparison(clusters, participant_ids)
    interval_summary = create_confidence_interval_summary(
        point_estimate,
        bootstrap_results,
    )

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("vas_experiment_03_paired_bootstrap")

    with mlflow.start_run(run_name="paired-participant-bootstrap-95ci"):
        mlflow.log_params({
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_random_state": BOOTSTRAP_RANDOM_STATE,
            "paired_participant_count": len(participant_ids),
            "split_repetitions": len(RANDOM_STATES),
            "target_far": TARGET_FAR,
        })

        metrics_to_log = {}
        for _, row in interval_summary.iterrows():
            metric = row["metric"]
            metrics_to_log[f"delta_{metric}"] = row[
                "point_delta_pangrams_minus_free_speech"
            ]
            metrics_to_log[f"delta_{metric}_ci_95_low"] = row["ci_95_low"]
            metrics_to_log[f"delta_{metric}_ci_95_high"] = row["ci_95_high"]
        mlflow.log_metrics(metrics_to_log)

        mlflow.log_table(interval_summary, "tables/paired_bootstrap_ci_summary.json")
        mlflow.log_table(bootstrap_results, "tables/paired_bootstrap_samples.json")
        log_bootstrap_delta_distributions(bootstrap_results, interval_summary)

        print("\nPaired participant-bootstrap 95% confidence intervals:")
        print(interval_summary.to_string(index=False))


if __name__ == "__main__":
    run_experiment()
