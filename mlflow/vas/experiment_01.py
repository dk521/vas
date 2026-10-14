from pathlib import Path
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt


DATAFRAME_PATH = Path("/Users/matrix/Desktop/Projects/vas/research_data/vas_enrollment_2026/private_ecapa_embedding_dataframe.pkl")
RANDOM_STATE=50
THRESHOLD = np.linspace(-1, 1, 401)

class ExperimentService:
    def __init__(self, dataframe_path=DATAFRAME_PATH):
        self.df = pd.read_pickle(dataframe_path)
        self.df = self.df[['user_id', 'speech_type', 'duration_seconds','embedding'] ]

    @staticmethod
    def enroll_test_split(df, random_state=RANDOM_STATE):
        rng = np.random.default_rng(random_state)

        split_dict = {}

        for user_id, rows in df.groupby('user_id'):
            enroll_index = rng.choice(
                rows.index.to_numpy(),
                size=6,
                replace=False   
            )

            split_dict[user_id] = {
                'enroll': rows.loc[enroll_index],
                'test': rows.drop(enroll_index)
            }
        return split_dict

    def paired_prompt_embeddings(self):
        """Split the same complete participant cohort for a fair comparison."""
        free_speech = self.df[self.df['speech_type'] == 'free_speech']
        pangrams = self.df[self.df['speech_type'] == 'pangrams']

        # A user belongs in the experiment only if they have all ten recordings
        # for BOTH prompt types. Otherwise the arms would use different people.
        free_complete = set(
            free_speech.groupby('user_id').filter(lambda rows: len(rows) == 10)['user_id']
        )
        pangram_complete = set(
            pangrams.groupby('user_id').filter(lambda rows: len(rows) == 10)['user_id']
        )
        paired_user_ids = sorted(free_complete & pangram_complete)

        free_speech = free_speech[free_speech['user_id'].isin(paired_user_ids)]
        pangrams = pangrams[pangrams['user_id'].isin(paired_user_ids)]

        return (
            self.enroll_test_split(free_speech),
            self.enroll_test_split(pangrams),
            paired_user_ids,
        )



def calculate_centroid(embeddings):
    return np.vstack(embeddings).mean(axis=0)

def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)



import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("vas_experiment_01")

experiment_service = ExperimentService()
(
    free_speech_embeddings,
    pangrams_embeddings,
    paired_user_ids,
) = experiment_service.paired_prompt_embeddings()



def calculate_scores(splits):
    genuine_scores = []
    impostor_scores = []

    for claimed_user_id, claimed_split in splits.items():
        centroid = calculate_centroid(claimed_split["enroll"]["embedding"])
        centroid = centroid / np.linalg.norm(centroid)

        # Genuine: this user's test recordings vs this user's centroid.
        for query_embedding in claimed_split["test"]["embedding"]:
            score = cosine_similarity(centroid, query_embedding)
            genuine_scores.append(score)

        # Impostor: every other user's TEST recordings vs claimed user's centroid.
        for query_user_id, query_split in splits.items():
            if query_user_id == claimed_user_id:
                continue

            for query_embedding in query_split["test"]["embedding"]:
                score = cosine_similarity(centroid, query_embedding)
                impostor_scores.append(score)

    return genuine_scores, impostor_scores

def false_reject_rate(theta, scores):
    # genuine
    return np.mean(np.array(scores) < theta)

def true_accept_rate(theta, scores):
    # genuine
    return np.mean(np.array(scores) >= theta)

def false_accept_rate(theta, scores):
    # imposter
    return np.mean(np.array(scores) >= theta)


def calculate_eer(genuine_scores, impostor_scores):
    """Return the discrete threshold where FAR and FRR are closest.

    A verification system accepts a query when score >= threshold.  At the
    equal-error-rate (EER) operating point, false accepts and false rejects
    should be approximately equal.  With finite trial data, they usually do
    not meet at exactly the same value, so we select the closest point.
    """
    genuine_scores = np.asarray(genuine_scores, dtype=float)
    impostor_scores = np.asarray(impostor_scores, dtype=float)

    thresholds = np.concatenate([
        [np.inf],
        np.sort(np.unique(np.concatenate([genuine_scores, impostor_scores])))[::-1],
        [-np.inf],
    ])

    rows = []
    for threshold in thresholds:
        far = false_accept_rate(threshold, impostor_scores)
        frr = false_reject_rate(threshold, genuine_scores)
        tar = true_accept_rate(threshold, genuine_scores)
        rows.append({
            "threshold": threshold,
            "far": far,
            "frr": frr,
            "tar": tar,
            "far_frr_gap": abs(far - frr),
        })

    threshold_sweep = pd.DataFrame(rows)
    eer_row = threshold_sweep.loc[threshold_sweep["far_frr_gap"].idxmin()]

    return {
        "threshold": float(eer_row["threshold"]),
        "eer": float((eer_row["far"] + eer_row["frr"]) / 2),
        "far": float(eer_row["far"]),
        "frr": float(eer_row["frr"]),
        "tar": float(eer_row["tar"]),
    }, threshold_sweep



def log_roc_curve(genuine_scores, impostor_scores, name):
    genuine_scores = np.array(genuine_scores, dtype=float)
    impostor_scores = np.array(impostor_scores, dtype=float)

    all_scores = np.concatenate([
        genuine_scores,
        impostor_scores,
    ])

    # Use actual cosine-similarity scores as thresholds.
    thresholds = np.concatenate([
        [np.inf],
        np.sort(np.unique(all_scores))[::-1],
        [-np.inf],
    ])

    fars = []
    tars = []

    for threshold in thresholds:
        far = np.mean(impostor_scores >= threshold)
        tar = np.mean(genuine_scores >= threshold)

        fars.append(far)
        tars.append(tar)

    fars = np.array(fars)
    tars = np.array(tars)

    auc = np.trapezoid(tars, fars)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        fars,
        tars,
        label=f"{name}, AUC={auc:.3f}",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        label="Random baseline",
    )

    ax.set_xlabel("False Accept Rate")
    ax.set_ylabel("True Accept Rate")
    ax.set_title(f"ROC Curve: {name}")
    ax.legend()
    ax.grid()

    mlflow.log_figure(
        fig,
        f"plots/{name.lower().replace(' ', '_')}_roc_curve.png",
    )

    plt.close(fig)

    return auc




def log_score_distribution(genuine_scores, impostor_scores, name):
    genuine_scores = np.array(genuine_scores, dtype=float)
    impostor_scores = np.array(impostor_scores, dtype=float)

    genuine_mean = np.mean(genuine_scores)
    genuine_std = np.std(genuine_scores)

    impostor_mean = np.mean(impostor_scores)
    impostor_std = np.std(impostor_scores)

    all_scores = np.concatenate([
        genuine_scores,
        impostor_scores,
    ])

    bins = np.linspace(
        all_scores.min(),
        all_scores.max(),
        30,
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        genuine_scores,
        bins=bins,
        alpha=0.6,
        label=f"Genuine: μ={genuine_mean:.3f}, σ={genuine_std:.3f}",
    )

    ax.hist(
        impostor_scores,
        bins=bins,
        alpha=0.6,
        label=f"Impostor: μ={impostor_mean:.3f}, σ={impostor_std:.3f}",
    )

    ax.axvline(
        genuine_mean,
        linestyle="--",
        label="Genuine mean",
    )

    ax.axvline(
        impostor_mean,
        linestyle="--",
        label="Impostor mean",
    )

    ax.set_xlabel("Cosine similarity score")
    ax.set_ylabel("Number of trials")
    ax.set_title(f"Score Distribution: {name}")
    ax.legend()
    ax.grid()

    mlflow.log_figure(
        fig,
        f"plots/{name.lower().replace(' ', '_')}_score_distribution.png",
    )

    plt.close(fig)

    return {
        "genuine_mean": genuine_mean,
        "genuine_std": genuine_std,
        "impostor_mean": impostor_mean,
        "impostor_std": impostor_std,
    }


def log_combined_roc_curve(free_genuine, free_impostor, pangram_genuine, pangram_impostor):
    """Put both ROC curves on one axis for direct prompt-type comparison."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for genuine, impostor, label in [
        (free_genuine, free_impostor, "Free Speech"),
        (pangram_genuine, pangram_impostor, "Pangrams"),
    ]:
        genuine = np.asarray(genuine, dtype=float)
        impostor = np.asarray(impostor, dtype=float)
        thresholds = np.sort(np.unique(np.concatenate([genuine, impostor])))[::-1]
        fars = [false_accept_rate(theta, impostor) for theta in thresholds]
        tars = [true_accept_rate(theta, genuine) for theta in thresholds]
        auc = np.trapezoid(tars, fars)
        ax.plot(fars, tars, label=f"{label} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random baseline")
    ax.set_xlabel("False Accept Rate (FAR)")
    ax.set_ylabel("True Accept Rate (TAR)")
    ax.set_title("Prompt-Type ROC Comparison")
    ax.legend()
    ax.grid()
    mlflow.log_figure(fig, "plots/prompt_type_roc_comparison.png")
    plt.close(fig)


def log_eer_tradeoff(threshold_sweep, eer_result, name):
    """Visualize exactly why the EER threshold was selected."""
    plot_data = threshold_sweep[np.isfinite(threshold_sweep["threshold"])].sort_values(
        "threshold"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(plot_data["threshold"], plot_data["far"], label="FAR")
    ax.plot(plot_data["threshold"], plot_data["frr"], label="FRR")
    ax.axvline(
        eer_result["threshold"],
        color="black",
        linestyle="--",
        label=f"EER threshold = {eer_result['threshold']:.3f}",
    )
    ax.scatter(
        [eer_result["threshold"]],
        [eer_result["eer"]],
        color="black",
        zorder=3,
        label=f"EER = {eer_result['eer']:.2%}",
    )
    ax.set_xlabel("Cosine-similarity threshold")
    ax.set_ylabel("Error rate")
    ax.set_ylim(0, 1)
    ax.set_title(f"FAR / FRR Trade-off: {name}")
    ax.legend()
    ax.grid()
    mlflow.log_figure(
        fig,
        f"plots/{name.lower().replace(' ', '_')}_eer_tradeoff.png",
    )
    plt.close(fig)


def log_prompt_distribution_comparison(free_genuine, free_impostor, pangram_genuine, pangram_impostor):
    """Use shared axes so score overlap is visually comparable across arms."""
    all_scores = np.concatenate([
        free_genuine, free_impostor, pangram_genuine, pangram_impostor,
    ])
    bins = np.linspace(all_scores.min(), all_scores.max(), 30)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    for ax, genuine, impostor, title in [
        (axes[0], free_genuine, free_impostor, "Free Speech"),
        (axes[1], pangram_genuine, pangram_impostor, "Pangrams"),
    ]:
        ax.hist(genuine, bins=bins, alpha=0.6, density=True, label="Genuine")
        ax.hist(impostor, bins=bins, alpha=0.6, density=True, label="Impostor")
        ax.set_title(title)
        ax.set_xlabel("Cosine similarity")
        ax.grid()

    axes[0].set_ylabel("Density")
    axes[0].legend()
    fig.suptitle("Genuine vs Impostor Score Separation")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/prompt_type_score_distribution_comparison.png")
    plt.close(fig)


def log_metric_comparison(free_auc, pangram_auc, free_eer, pangram_eer):
    """A compact visual summary: higher AUC and lower EER are better."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    labels = ["Free Speech", "Pangrams"]

    auc_bars = axes[0].bar(labels, [free_auc, pangram_auc])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("Higher is better")
    axes[0].bar_label(auc_bars, fmt="%.3f")

    eer_bars = axes[1].bar(labels, [free_eer, pangram_eer])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Equal Error Rate")
    axes[1].set_title("Lower is better")
    axes[1].bar_label(
        eer_bars,
        labels=[f"{value:.2%}" for value in [free_eer, pangram_eer]],
    )

    fig.suptitle("Prompt-Type Verification Summary")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/prompt_type_metric_comparison.png")
    plt.close(fig)



with mlflow.start_run() as run:


    free_speech_genuine_score, free_speech_imposter_score = calculate_scores(
        free_speech_embeddings
    )

    pangram_genuine_score, pangram_imposter_score = calculate_scores(
        pangrams_embeddings
    )


    mlflow.log_params({
        'random_state': RANDOM_STATE,
        'paired_participant_count': len(paired_user_ids),
        'enrollment_recordings_per_user': 6,
        'test_recordings_per_user': 4,
    })

    mlflow.log_metrics({
        'free_speech_genuine_mean': np.mean(free_speech_genuine_score),
        'free_speech_genuine_std': np.std(free_speech_genuine_score),
        'free_speech_imposter_mean': np.mean(free_speech_imposter_score),
        'free_speech_imposter_std': np.std(free_speech_imposter_score),        

        'pangrams_genuine_mean': np.mean(pangram_genuine_score),
        'pangrams_genuine_std': np.std(pangram_genuine_score),
        'pangrams_imposter_mean': np.mean(pangram_imposter_score),
        'pangrams_imposter_std': np.std(pangram_imposter_score),        


    })
    # free speech
    for step, theta in enumerate(THRESHOLD):
        mlflow.log_metrics({
            'FRR_free_speech' : false_reject_rate(theta, free_speech_genuine_score),
            'TAR_free_speech' : true_accept_rate(theta, free_speech_genuine_score),
            'FAR_free_speech' : false_accept_rate(theta, free_speech_imposter_score)
        },
        step=step)

    # pangrams
    for step, theta in enumerate(THRESHOLD):
        mlflow.log_metrics({
            'FRR_pangram' : false_reject_rate(theta, pangram_genuine_score),
            'TAR_pangram' : true_accept_rate(theta, pangram_genuine_score),
            'FAR_pangram' : false_accept_rate(theta, pangram_imposter_score)
        },
        step=step)

    free_eer, free_threshold_sweep = calculate_eer(
        free_speech_genuine_score,
        free_speech_imposter_score,
    )
    pangram_eer, pangram_threshold_sweep = calculate_eer(
        pangram_genuine_score,
        pangram_imposter_score,
    )

    mlflow.log_metrics({
        "free_speech_eer": free_eer["eer"],
        "free_speech_eer_threshold": free_eer["threshold"],
        "free_speech_far_at_eer": free_eer["far"],
        "free_speech_frr_at_eer": free_eer["frr"],
        "pangrams_eer": pangram_eer["eer"],
        "pangrams_eer_threshold": pangram_eer["threshold"],
        "pangrams_far_at_eer": pangram_eer["far"],
        "pangrams_frr_at_eer": pangram_eer["frr"],
    })

    # These tables let you inspect the FAR/FRR trade-off at every real score.
    mlflow.log_table(
        free_threshold_sweep,
        artifact_file="tables/free_speech_threshold_sweep.json",
    )
    mlflow.log_table(
        pangram_threshold_sweep,
        artifact_file="tables/pangrams_threshold_sweep.json",
    )

    print("Free speech EER:", free_eer)
    print("Pangrams EER:", pangram_eer)

    log_eer_tradeoff(free_threshold_sweep, free_eer, "Free Speech")
    log_eer_tradeoff(pangram_threshold_sweep, pangram_eer, "Pangrams")

    free_auc = log_roc_curve(
        free_speech_genuine_score,
        free_speech_imposter_score,
        "Free Speech",
    )

    pangram_auc = log_roc_curve(
        pangram_genuine_score,
        pangram_imposter_score,
        "Pangrams",
    )

    mlflow.log_metric("free_speech_roc_auc", free_auc)
    mlflow.log_metric("pangrams_roc_auc", pangram_auc)

    log_combined_roc_curve(
        free_speech_genuine_score,
        free_speech_imposter_score,
        pangram_genuine_score,
        pangram_imposter_score,
    )


    free_stats = log_score_distribution(
        free_speech_genuine_score,
        free_speech_imposter_score,
        "Free Speech",
    )

    pangram_stats = log_score_distribution(
        pangram_genuine_score,
        pangram_imposter_score,
        "Pangrams",
    )

    log_prompt_distribution_comparison(
        free_speech_genuine_score,
        free_speech_imposter_score,
        pangram_genuine_score,
        pangram_imposter_score,
    )
    log_metric_comparison(
        free_auc,
        pangram_auc,
        free_eer["eer"],
        pangram_eer["eer"],
    )
