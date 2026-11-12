# Deploying NeuroSense AI

The deployment has two parts:
- **Backend on Render (free web service).** One Docker container: the FastAPI backend serves the API under `/api` and
  the built frontend at `/`.
- **Website on Vercel (free).** It serves the frontend and forwards every `/api` request to the Render backend
  (`frontend/vercel.json`).

Hugging Face Docker Spaces now need a paid plan, so they are not used.

## What is online and what stays local

| Feature | Online | Local |
|---|---|---|
| Sign-in, dashboard, history, result pages | yes | yes |
| Analyse an uploaded EEG file (DEAP format, 32 channels, 128 Hz) | yes | yes |
| Valence, arousal, dominance and liking estimates, explanations and flashcards | yes | yes |
| Model descriptions and model-level explanations | yes | yes |
| Browsing DEAP or FACED participants, personalised estimates | no | yes |

The DEAP licence does not allow its recordings to be redistributed, so no recording and no feature table derived
from one is shipped. The bundle script refuses `.dat`, `.bdf`, `.npz`, `.npy`, database, log and `.env` files. The
bundle does include model files trained on DEAP and FACED data. Check that the DEAP end-user licence allows you to
publish them. FACED is CC-BY 4.0 and must be credited.

## Memory

Measured on the deployment bundle (no datasets), the backend peaks at about **390 MB** after several analyses with
full explanations. That fits Render's free 512 MB instance.

## Steps

1. **Build the bundle** (from `backend/`): `python -m scripts.make_deploy_bundle`. This writes `deploy/space/`, which
   is ignored by the main repository.

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
