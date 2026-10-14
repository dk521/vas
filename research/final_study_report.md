# Voice Authentication Study: Free Speech vs Pangrams

## 1. Study objective

This study compares two enrollment prompt types for speaker verification:

- **Free speech:** participants speak naturally in their own words.
- **Pangrams:** participants read a fixed sentence designed to cover varied phonetic sounds.

The question is not simply which condition has a higher average similarity score. The question is:

> Which prompt type better separates a genuine speaker from an impostor while keeping genuine users accepted at a secure operating point?

The system uses a pre-trained **ECAPA-TDNN** speaker-recognition model. Each recording becomes a speaker embedding, and cosine similarity is used to compare embeddings.

---

## 2. Dataset used in the analysis

- 300 enrollment recordings were exported locally.
- 14 participants had complete, paired data: 10 free-speech and 10 pangram recordings each.
- Only these 14 paired participants were used in the comparison. This prevents one prompt type from being evaluated on a different group of people.
- Each experimental split used 6 recordings for enrollment and 4 held-out recordings for testing.

The experiment data and MLflow artifacts are local and contain biometric-derived scores. They should not be committed to Git or made public.

---

## 3. Reproducible workflow

### Step 0 — Collect and export recordings

**Input:** browser recordings uploaded through the Django application.

**Output:** a local study directory containing audio files plus:

- `enrollment_manifest.csv` — sample metadata and audio locations.
- `participant_lookup.csv` — private participant mapping.

**Why this comes first:** no evaluation is possible without both prompt types and a reliable mapping from an audio file to a participant and condition.

### Step 1 — Build the embedding dataset

**Script:** `mlflow/vas/build_audio_embedding_dataframe.py`

**Input:** exported local audio and manifests.

**Output:** `research_data/vas_enrollment_2026/private_ecapa_embedding_dataframe.pkl`

Each row contains a participant identifier, prompt type, recording duration, local audio path, and one 192-dimensional ECAPA embedding.

**Why this is needed next:** caching embeddings means every later experiment compares the same model outputs. The evaluation becomes fast, reproducible, and does not repeatedly load the neural model.

### Step 2 — Start local MLflow

```bash
/Users/matrix/Desktop/Projects/vas/.vas-cs/bin/mlflow server \
  --backend-store-uri sqlite:////Users/matrix/Desktop/Projects/vas/mlflow/learning/mlflow.db \
  --host 127.0.0.1 \
  --port 5000 \
  --allowed-hosts "localhost:*,127.0.0.1:*"
```

Open `http://127.0.0.1:5000` to inspect runs, metrics, tables, and plots.

### Step 3 — Experiment 01: one illustrative split

**Script:** `mlflow/vas/experiment_01.py`

```bash
/Users/matrix/Desktop/Projects/vas/.vas-cs/bin/python mlflow/vas/experiment_01.py
```

For one deterministic random seed:

1. Select 6 of a participant's 10 recordings for enrollment.
2. Average those 6 embeddings into a normalized speaker centroid.
3. Score the 4 held-out recordings against their own centroid: **genuine trials**.
4. Score every other participant's held-out recordings against that centroid: **impostor trials**.
5. Plot score distributions, ROC curves, and the FAR/FRR trade-off.

**Why this is useful:** it teaches and visualizes the evaluation mechanics.

**Why it is not enough:** one split can accidentally select unusually easy or difficult recordings. It provides a point estimate, not a stable conclusion.

### Step 4 — Experiment 02: repeated 6/4 splits

**Script:** `mlflow/vas/experiment_02.py`

```bash
/Users/matrix/Desktop/Projects/vas/.vas-cs/bin/python mlflow/vas/experiment_02.py
```

The same protocol is repeated for random seeds 42 through 84 inclusive: 43 deterministic splits.

**Why this is the next step:** it tests whether the result depends heavily on which recordings were chosen for enrollment.

Experiment 02 logs:

- a nested MLflow run for each seed;
- average and standard deviation across seeds;
- EER and AUC stability plots;
- box plots showing the distribution of metrics across splits;
- score-level trial data for later analysis.

### Step 5 — Experiment 03: paired participant bootstrap

**Script:** `mlflow/vas/experiment_03.py`

```bash
/Users/matrix/Desktop/Projects/vas/.vas-cs/bin/python mlflow/vas/experiment_03.py
```

Experiment 02 answers, “Does the result survive different recording splits?” Experiment 03 answers, “How uncertain is the difference because the study has only 14 participants?”

It resamples the same paired participant IDs with replacement 1,000 times and computes the delta:

`Pangrams − Free Speech`

for EER, ROC-AUC, and TAR at 1% FAR. The 2.5th and 97.5th percentiles form a 95% bootstrap confidence interval.

**Why pairing matters:** both conditions come from the same people. Resampling the same participant IDs in both arms prevents participant identity from confounding the comparison.

### Step 6 — Experiment 04: duration policy analysis

**Script:** `mlflow/vas/experiment_04_duration.py`

```bash
/Users/matrix/Desktop/Projects/vas/.vas-cs/bin/python mlflow/vas/experiment_04_duration.py
```

For each held-out recording, this experiment measures:

- genuine score;
- maximum impostor score across other speaker centroids;
- score margin = genuine score − maximum impostor score.

It evaluates minimum-duration policies from 0 to 5 seconds and plots the accuracy-versus-coverage trade-off.

**Why this is last:** duration guidance should be supported by observed verification performance, not guessed before the core free-speech versus pangram comparison is established.

---

## 4. Metric guide

| Metric | Meaning | Better direction | Interpretation |
|---|---|---|---|
| Genuine score mean | Average similarity between a held-out recording and its true speaker centroid | Higher | Genuine recordings are more consistent with enrollment recordings. |
| Genuine score standard deviation | Spread of genuine scores | Lower | A user's recordings are more consistent across attempts. |
| Impostor score mean | Average similarity between a query and a wrong speaker centroid | Lower | Wrong speakers look less similar to the claimed identity. |
| Score margin | Genuine score − highest impostor score | Higher | A query is safely separated from the most confusing wrong identity. |
| FAR | False Accept Rate: fraction of impostor trials accepted at a threshold | Lower | Better security. Low FAR means fewer wrong speakers gain access. |
| FRR | False Reject Rate: fraction of genuine trials rejected at a threshold | Lower | Better usability. Low FRR means fewer real users are blocked. |
| TAR | True Accept Rate: fraction of genuine trials accepted | Higher | TAR = 1 − FRR. Higher means more real users are accepted. |
| EER | Equal Error Rate: error where FAR and FRR are closest | Lower | A threshold-independent summary of verification difficulty. |
| EER threshold | Threshold used at the EER operating point | Neither | It is a balance point, not automatically the production threshold. |
| ROC-AUC | Ability to rank genuine scores above impostor scores across all thresholds | Higher | 1.0 is perfect separation; 0.5 is random ranking. |
| TAR @ 1% FAR | Genuine acceptance when FAR is constrained to at most 1% | Higher | Useful security-focused operating metric. |
| Metric standard deviation across splits | Sensitivity to different enrollment/test recording selections | Lower | Lower values mean the result is more stable. |
| 95% CI of prompt-type delta | Bootstrap uncertainty around `Pangrams − Free Speech` | Depends on metric | If it excludes zero in the favorable direction, the improvement is supported at this confidence level. |
| Coverage | Fraction of recordings retained by a policy such as “at least 3 seconds” | Higher, subject to accuracy | A policy that improves accuracy but excludes most users may not be practical. |

---

## 5. Observations from the completed experiments

### Experiment 01 — one split

Experiment 01 is the visual, single-split demonstration. It shows how a centroid is built, how genuine and impostor scores are created, and how changing a threshold creates the FAR/FRR trade-off.

**Observation:** a single split can show a smaller or larger gap depending on the chosen six enrollment recordings. It should be used to explain the method, not as the final claim.

### Experiment 02 — average over 43 splits

| Metric | Free speech mean ± SD | Pangrams mean ± SD | Difference |
|---|---:|---:|---:|
| EER | 5.07% ± 1.70% | 2.64% ± 1.22% | Pangrams lower by 2.43 percentage points |
| ROC-AUC | 0.9801 ± 0.0110 | 0.9969 ± 0.0020 | Pangrams higher by 0.0168 |
| TAR @ 1% FAR | 93.15% ± 2.70% | 96.14% ± 2.02% | Pangrams higher by 2.99 percentage points |

**Observations:**

- Pangrams have lower average EER.
- Pangrams have higher average ROC-AUC.
- Pangrams have higher average genuine acceptance under a strict 1% FAR constraint.
- Pangram AUC varies less across enrollment/test splits, indicating more stable score separation.

### Experiment 03 — 1,000 paired participant bootstrap samples

| Difference: Pangrams − Free speech | Point estimate | 95% CI | Observation |
|---|---:|---:|---|
| EER | −2.49 percentage points | −6.02 to +0.20 pp | Direction favors pangrams, but the interval includes no difference. |
| ROC-AUC | +0.0169 | +0.0030 to +0.0339 | Interval excludes zero. Pangrams have better overall score separation in this pilot. |
| TAR @ 1% FAR | +2.78 percentage points | −1.58 to +7.60 pp | Direction favors pangrams, but the interval includes no difference. |

**Observations:**

- The strongest supported result is improved ROC-AUC for pangrams.
- EER and TAR point estimates favor pangrams, but the current participant sample is too small to make a 95%-confidence claim for those metrics.

### Experiment 04 — duration analysis

| Prompt type | Duration relationship with score margin | Practical policy | EER | Coverage |
|---|---:|---|---:|---:|
| Free speech | Pearson r = 0.52 | Require at least 3 seconds | 3.90% | 84.3% |
| Pangrams | Pearson r = 0.07 | Require at least 2 seconds | 2.12% | 97.9% |

Additional observations:

- Free-speech EER without a duration filter was 5.07%; a 3-second requirement reduced it to 3.90%.
- A 5-second free-speech requirement produced a lower EER of 3.67%, but retained only 58.6% of recordings. It is not the preferred usability trade-off.
- Pangrams already perform well with shorter recordings. A 4-second requirement retained only 67.9% of recordings without improving EER over the 2-second policy.
- A 5-second pangram requirement worsened EER to 2.80% and retained only 39.3% of recordings.

---

## 6. Conclusions supported by the current data

1. **Pangrams provide better overall speaker-score separation than free speech in this pilot.**
   - Supported by higher mean ROC-AUC in Experiment 02 and a positive 95% bootstrap CI for the ROC-AUC difference in Experiment 03.

2. **Pangrams are the better default enrollment prompt for this ECAPA centroid-based evaluation.**
   - They achieved lower mean EER and higher mean TAR at 1% FAR across 43 splits.
   - This statement should be described as a pilot result, not a universal claim.

3. **Free speech benefits from a minimum duration requirement.**
   - Three seconds is the best observed accuracy–coverage compromise in this dataset.

4. **Pangrams do not need a four-second requirement in this dataset.**
   - A two-second minimum retained 97.9% of recordings and achieved the best practical EER.

5. **Prompt-specific guidance is better than one shared recording rule.**
   - Suggested UI guidance: free speech ≥3 seconds; pangram ≥2 seconds.

---

## 7. Conclusions that require more data

Do not claim these yet:

1. **“Pangrams have significantly lower EER.”**
   - The EER confidence interval includes zero. Recruit more paired participants before making this claim.

2. **“Pangrams have significantly higher TAR at 1% FAR.”**
   - The TAR confidence interval includes zero.

3. **“Three seconds is the universal free-speech requirement.”**
   - It is the best observed policy in this pilot, not a validated universal threshold.

4. **“Two seconds is always enough for pangrams.”**
   - Validate on more speakers, microphones, acoustic environments, and pangram text variants.

5. **“The current application threshold should be changed to the experimental EER threshold.”**
   - The deployed Django application currently uses maximum similarity against individual enrollment references, while the study uses a centroid. Thresholds transfer only when scoring methods and evaluation conditions match.

---

## 8. Limitations and the next data collection round

### Current limitations

- Only 14 paired participants are available for the core comparison.
- The same participants contribute to model-condition selection and evaluation; there is no participant-disjoint final test set.
- Repeated random splits reuse recordings. They test split sensitivity but do not create 43 independent datasets.
- The model is pre-trained on VoxCeleb-like speaker data and was not fine-tuned on this collection environment.
- This is a zero-effort impostor evaluation. It does not test replay attacks, synthetic voice attacks, or intentionally mimicked speakers.
- Microphone, room noise, device type, and recording-day effects may confound the duration and prompt-type findings.

### Recommended next collection round

1. Recruit at least 30–50 additional paired participants.
2. Collect on multiple days to measure session-to-session robustness.
3. Record known device and environment labels.
4. Reserve a participant-disjoint test set before selecting a final threshold.
5. Evaluate the exact production scoring rule, or change production to centroid scoring and re-calibrate it.
6. Add spoof/replay evaluation before presenting this as a security-ready authentication system.

---

## 9. Interview-ready summary

> I built a controlled pilot to compare free speech and fixed pangrams for ECAPA-based speaker verification. I used paired participants, six enrollment recordings and four held-out recordings, then evaluated genuine and impostor trials using cosine similarity to a speaker centroid. A single split was useful for visualization, but I used 43 repeated splits to measure stability and a paired participant bootstrap to quantify uncertainty. Pangrams showed significantly better ROC-AUC, while EER and TAR improvements were favorable but not statistically conclusive with only 14 participants. A follow-up duration analysis showed that free speech benefits from a three-second minimum, while pangrams remain robust at two seconds. The next step is a larger, participant-disjoint evaluation with matched production scoring and spoof testing.
