# NeuroSense AI: academic demonstration guide

A step-by-step script for presenting NeuroSense AI, about 11 minutes long. It is written for the presenter and assumes
no earlier knowledge of the project.

**What to say first:** NeuroSense AI is a research prototype that estimates a binary category of *self-reported* valence
from EEG. It does not detect a person's exact emotions, and it is not a medical or diagnostic tool.

## Before the session

### Start

PostgreSQL must be running. Open PowerShell in the project folder and run:

```powershell
cd C:\Users\gandl\OneDrive\Documents\NeuroSense_AI
powershell -ExecutionPolicy Bypass -File scripts\start.ps1
```

Wait for `backend healthy` and `frontend ready`. This takes about 15–30 seconds. Then check it:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\status.ps1
```

You should see, for both ports, `listening on 127.0.0.1 (loopback only)`, plus `status=ok` and
`active model: valence_binary-random_forest-20261008-180558`.

### Sign in

1. Open **http://127.0.0.1:5173** in the browser.
2. Enter your email and password, then choose **Sign in**. If you have no account yet, choose the option to create one.
3. Check that Overview shows "Service available" and both datasets' results. "Not available" on a dataset card means its
   result file or configuration could not be read: see "If something goes wrong".

Signing in, creating an account and running analyses are all saved to the database, as they are in normal use. Starting,
checking and stopping the application are not.

### Stop

After the session, from the same folder:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stop.ps1
```

You should see `frontend: stopped`, `backend: stopped` and `Done (2 stopped).` It stops only the processes that
`start.ps1` started, and leaves every other program alone. `status.ps1` then reports `not running` for both ports.

## The demonstration

### 1. Overview: the honest headline (1 minute)

Open **Overview** and point to "Datasets and measured performance".

| Dataset | Result | What it means |
|---|---|---|
| DEAP | 50.0% (95% CI 48.1–52.0%), read from the saved result file | Exploratory, 12 participants, new person and new videos. There is no independent test. |
| FACED | 56.6% (95% CI 51.8–61.2%) | Strict held-out test of 25 unseen people × 9 unseen clips, evaluated once. |

Say clearly: 50% is chance, neither result reaches the 80% target, and the two numbers cannot be compared directly
because the data and protocols differ.

### 2. Dataset: two separate datasets (2 minutes)

Open **Dataset** under Research tools.

- **DEAP tab:** the 12 authorised participants, the DEAP processing pipeline, and a searchable participant table.
- **FACED tab:** 123 validated participants, 3,444 trials and FACED's own pipeline. Search for `12` to show the
  directory, choose a participant, and show the 28 recordings with each person's rating and evaluation status.

Point out that each dataset has its own channels, sampling rate, features and model.

### 3. Analyze: the full workflow (4 minutes)

From a participant's recording list, click **Analyze** on a trial marked **Strict held-out**. Good examples: Participant
000 (`sub-000`) with video 1, 6 or 7.

1. Show the strip at the top: dataset, pipeline and model. For FACED it reads `faced-v2` and
   `faced-valence-lr_C0.01-S1`.
2. Run **Analyze this trial with the FACED model**.
3. Read the result aloud: the estimate, the participant's own rating and the evaluation status.
4. Read **How to read this**. A single trial is weak evidence, and a match or mismatch says little about the model.
5. Change the video and show that the old result disappears straight away.
6. Switch to the **DEAP** tab: the pipeline and model change to the DEAP pipeline and the active Random Forest. Search
   for a participant, run an analysis, and open the saved result.

### 4. History: every analysis is a labelled record (1 minute)

Open **History**. Each row is marked DEAP, FACED, Uploaded file or Synthetic test, and a synthetic test never shows an
estimate. Filter by dataset, search for `sub-000`, and open a FACED record: it keeps its model version, pipeline,
evaluation status and the participant's rating.

### 5. Model & Evidence: why the numbers should be trusted (2 minutes)

Open **Model & Evidence**, FACED tab.

- The strict held-out result, with its confidence interval and majority baseline.
- The confusion matrix.
- Per-participant results. The mean per-participant balanced accuracy is 50.8%, lower than the pooled 56.6%, so part of
  the pooled score comes from differences between people.
- The model was chosen on development folds only, then frozen before the test set was evaluated once.
- The exploratory follow-ups: intended video category 55.8%, and enhanced EEG cleaning gave no improvement.

Then open the DEAP tab and show the research summary and limitations, including the 0.805 leakage demonstration: a
random segment split gives an impressive number that is invalid.

### 6. Close (1 minute)

The main finding is a negative one, reported honestly. Under strict evaluation, neither dataset supports reliable
cross-person valence estimation from these features. More model types and more cleaning did not change that.

## Likely questions

**Why not report the 80% target as met?** It was not met. The best strict result is 56.6%, and the upper end of its
confidence interval is 61.2%.

**Why is FACED better than DEAP?** That cannot be concluded. They differ in participants, stimuli, labels and protocols.

**Can the DEAP model analyse FACED recordings?** No. The two use different channels, sampling rates and feature sets. The
app runs each dataset through its own pipeline, and it refuses a FACED recording on the DEAP path and the reverse. If the
FACED model is unavailable, analysis stops; the DEAP model is never substituted.

**Has the frozen DEAP test group been used?** No. The eight frozen participants have never been accessed, because the
decision on official DEAP data access is still pending.

**Could more cleaning or a bigger model reach 80%?** The enhanced cleaning experiment found no improvement, and on DEAP
the model comparisons found none either. Any further attempt needs a new question or representation, declared in advance,
and a new untouched test set.

**Is it clinically useful?** No. It has no clinical validity and must not be used to describe anyone's emotional state.

## If something goes wrong

| Symptom | What to do |
|---|---|
| Sign-in page says the service is unreachable | Run `scripts\status.ps1`, then `scripts\start.ps1` |
| `start.ps1` says the database is not reachable | Start PostgreSQL, then run it again |
| `start.ps1` says a port is already in use | The app may already be running. Run `scripts\status.ps1`, then `scripts\stop.ps1` if needed |
| The DEAP figure on Overview says "Not available" | Check that `docs/results/exploratory/cross_participant_lvo_rf/summary.json` exists. Nothing is shown in its place |
| FACED tab says it is not available | Check `NEUROSENSE_FACED_DATASET_DIR` and that `outputs/faced/validation_report.json` exists |
| "No FACED model available" | Check that `outputs/faced/models/` contains the model and its card; DEAP remains usable |
| A FACED analysis takes a few seconds | Normal: the recording is read and processed on demand |

## Further reading

- [FACED integration and evaluation](FACED_INTEGRATION_AND_EVALUATION.md)
- [FACED stimulus-category experiment](FACED_STIMULUS_CATEGORY_EXPERIMENT.md)
- [FACED noise-removal experiment](FACED_NOISE_REMOVAL_EXPERIMENT.md)
- [Research findings and next steps](RESEARCH_FINDINGS_AND_NEXT_STEPS.md)
- [Evaluation protocol](EVALUATION_PROTOCOL.md)
