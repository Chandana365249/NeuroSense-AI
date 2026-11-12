# Deploying NeuroSense AI

The deployment has two parts:
- **Backend on Render (free web service).** One Docker container: the FastAPI backend serves the API under `/api` and
  the built frontend at `/`.
- **Website on Vercel (free).** It serves the frontend and forwards every `/api` request to the Render backend
  (`vercel.json` at the repository root).

Hugging Face Docker Spaces now need a paid plan, so they are not used.

## What is online and what stays local

| Feature | Online | Local |
|---|---|---|
| Sign-in, dashboard, history, result pages | yes | yes |
| Analyse an uploaded EEG file (DEAP format, 32 channels, 128 Hz) | yes | yes |
| Example EEG files to try (real FACED EEG, converted) | yes | yes |
| Valence, arousal, dominance and liking estimates, explanations and flashcards | yes | yes |
| Model descriptions and model-level explanations | yes | yes |
| FACED: browse all 123 participants, analyse any trial, explanations, personalised estimates | yes (from saved features) | yes (from the recordings) |
| FACED: the raw EEG-trace card | no | yes |
| DEAP: browse participants and analyse their recordings | no (option hidden) | yes |

The app adapts automatically:
- **DEAP.** When no DEAP recording is installed, the "DEAP recording" input is hidden and the page opens on "Your EEG
  file". The DEAP licence does not allow its recordings to be redistributed, so no recording and no feature table
  derived from one is shipped.
- **FACED.** When the FACED recordings are missing, FACED runs from `outputs/faced/features_faced-v2.npz`. This table
  holds the features the pipeline computes from the recordings, so predictions, explanations and personalised
  estimates are identical; the check found a difference of 0 on real trials (see `tests/test_faced_features_mode.py`).
  FACED is CC-BY 4.0 and is credited in the app. Only the raw signal is missing online, so the EEG-trace card is not
  shown.
- **Examples.** `frontend/public/samples/` holds three real FACED clips converted to the upload format, written by
  `python -m scripts.make_sample_eeg`. They come from development participants and were chosen by each viewer's own
  rating, never by model output. Each file states how it was converted, including that AF3 and AF4 are estimated
  from neighbouring channels.

The bundle script refuses `.dat`, `.bdf`, `.npz`, `.npy`, database, log and `.env` files. The only exception is the
FACED feature table. The bundle also includes model files trained on DEAP and FACED data. Check that the DEAP end-user
licence allows you to publish them.

## Memory

Measured on the deployment bundle (no datasets), the backend peaks at about **430 MB** (including the 40 MB FACED feature table) after several analyses with
full explanations. That fits Render's free 512 MB instance.

## Steps

1. **Build the bundle** (from `backend/`): `python -m scripts.make_deploy_bundle`. This writes `deploy/space/`, which
   is ignored by the main repository. A rebuild keeps its `.git` folder, so the same folder can be pushed again.

2. **Put the bundle in its own private GitHub repository** (for example `neurosense-ai-backend`). It holds the
   model files, which the main repository deliberately leaves out.

3. **Create the Render service** at <https://dashboard.render.com>: *New → Web Service*. Then:
   - Choose that repository, language **Docker**, instance type **Free**.
   - Add the environment variable `NEUROSENSE_JWT_SECRET_KEY` with a random value, generated with
     `python -c "import secrets; print(secrets.token_urlsafe(48))"`. Never put it in git.
   - The container listens on `$PORT`, which Render sets.

4. **Point Vercel at the backend.** `frontend/vercel.json` forwards `/api` to https://neurosenseai-backend.onrender.com, then deploy the
   frontend with `npx vercel --prod` from `frontend/`.

## Notes

- **Cold starts:** free Render services sleep after about 15 minutes without traffic. The first request after that
  takes about a minute.
- **The database is SQLite inside the container.** Accounts, results and stored uploads are reset when the service
  restarts or redeploys.
- **Upload size:** Vercel forwards requests up to about 4.5 MB. Larger EEG files work on the Render URL directly.
- **Updating:** rebuild the bundle, then push it to the backend repository. Render redeploys automatically.
- **Local version:** unchanged. `scripts\start.ps1` still runs it on 127.0.0.1 with the full datasets.
