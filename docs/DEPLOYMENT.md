# Deploying NeuroSense AI

The app runs as one Docker container: the FastAPI backend serves the API under `/api` and the built frontend at `/`.
The steps below use a Hugging Face Space (free Docker hosting with 16 GB of memory). The same Dockerfile works on any
Docker host (Render, Railway, a VM): only step 3 differs.

## What is online and what stays local

| Feature | Online | Local |
|---|---|---|
| Sign-in, history, result details | yes | yes |
| Analyse an uploaded EEG file (DEAP format, 32 channels, 128 Hz) | yes | yes |
| Valence, arousal, dominance and liking estimates with calibrated probabilities | yes | yes |
| "Why did the model make this prediction?" for uploads | yes | yes |
| Model & Evidence, including the model-level explanations | yes | yes |
| Browsing DEAP or FACED participants, personalised estimates | no | yes |

The DEAP licence does not allow its recordings to be redistributed, so no recording and no feature table derived
from one is shipped. The bundle script refuses `.dat`, `.bdf`, `.npz`, `.npy`, database, log and `.env` files.

The bundle does include model files trained on DEAP and FACED data: the active DEAP Random Forest and the DEAP affect
models. Check that the DEAP end-user licence allows you to publish models trained on it before making the Space
public. FACED is CC-BY 4.0, and the Space README must credit it.

## Steps

1. **Build the bundle** (from `backend/`):

   ```
   python -m scripts.make_deploy_bundle
   ```

   This writes `deploy/space/`, which is ignored by git. It holds the Dockerfile, a Space README, the source code and
   the model files (about 60 MB).

2. **Create the Space** at <https://huggingface.co/new-space>:
   - SDK: **Docker**, blank template;
   - visibility: **Private** to start with.

3. **Add the secret.** Open the Space's *Settings → Variables and secrets* and add a secret named
   `NEUROSENSE_JWT_SECRET_KEY` with a random value. Generate one with:

   ```
   python -c "import secrets; print(secrets.token_urlsafe(48))"
   ```

   Never put the secret in the bundle or in git.

4. **Upload the bundle.** Install the tools with `pip install -U huggingface_hub`, then:

   ```
   huggingface-cli login
   huggingface-cli upload <your-username>/<space-name> deploy/space . --repo-type space
   ```

   The Space builds the image (about 5–10 minutes) and starts it. The app is then at
   `https://<your-username>-<space-name>.hf.space`.

5. **First sign-in.** Register an account in the app.

## Notes

- **The database is SQLite inside the container.** It is reset when the Space restarts, unless persistent storage is
  enabled in the Space settings. If you enable it, set `NEUROSENSE_DATABASE_URL=sqlite:////data/neurosense.sqlite3`.
- **Stored uploads** (`/app/outputs/uploads`) are also lost on a restart. After that, an older upload's explanation
  asks for the file again.
- **Updating:** rebuild the bundle and upload it again. The Space rebuilds itself.
- **Local version:** unchanged. `scripts\start.ps1` still runs it on 127.0.0.1 with the full datasets.
