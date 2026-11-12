"""Assemble a self-contained deployment folder (a Docker app for Render or any Docker host) in deploy/space/.

Usage (from backend/):  python -m scripts.make_deploy_bundle

What goes in: the backend and frontend source, the saved research results, and only the model artifacts the app
serves (the active DEAP model, the FACED model, the affect models, the precomputed global explanations, evidence
JSON and graphs). What never goes in: the DEAP and FACED recordings, extracted feature tables, the database,
stored uploads, logs and any .env file. Without the recordings, the online app analyses uploaded EEG files and shows
the evidence pages; browsing dataset participants stays a local feature (the DEAP licence does not allow
redistribution of its recordings).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "deploy" / "space"
FORBIDDEN = (".env", ".dat", ".bdf", ".sqlite3", ".db", ".npz", ".npy", ".log")

DOCKERFILE = """\
# NeuroSense AI: API + built frontend in one container. Listens on $PORT (set by Render; 7860 otherwise).
FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN useradd -m -u 1000 user
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY --chown=user backend/ backend/
COPY --chown=user outputs/ outputs/
COPY --chown=user docs/ docs/
COPY --from=web --chown=user /web/dist frontend/dist
RUN mkdir -p /app/data && chown user /app/data
USER user
# The JWT secret is NOT set here: add NEUROSENSE_JWT_SECRET_KEY as a secret environment variable on the host.
ENV NEUROSENSE_ENVIRONMENT=production \\
    NEUROSENSE_OUTPUT_DIR=/app/outputs \\
    NEUROSENSE_DATASET_DIR=/app/Dataset/DEAP \\
    NEUROSENSE_FACED_DATASET_DIR=/app/Dataset/FACED/v1.1.4 \\
    NEUROSENSE_DATABASE_URL=sqlite:////app/data/neurosense.sqlite3 \\
    NEUROSENSE_FRONTEND_DIST_DIR=/app/frontend/dist \\
    NEUROSENSE_CORS_ORIGINS=[]
EXPOSE 7860
WORKDIR /app/backend
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:create_app --factory --host 0.0.0.0 --port ${PORT:-7860}"]
"""

README = """\
---
title: NeuroSense AI
emoji: 🧠
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# NeuroSense AI (research prototype)

Estimates self-reported valence, arousal, dominance and liking from EEG, with calibrated probabilities and
explanations computed from the model that made each estimate. Research use only; not a medical or diagnostic device.

Online, you can analyse your own EEG upload (DEAP format: 32 channels at 128 Hz) and read the model evidence.
Browsing dataset participants needs the DEAP and FACED recordings and is available in the local version only.

Required secret: `NEUROSENSE_JWT_SECRET_KEY` (generate with
`python -c "import secrets; print(secrets.token_urlsafe(48))"`). The SQLite database lives in the container and is
reset when the Space restarts unless persistent storage is enabled.
"""


def copy_tree(src: Path, dst: Path, skip_dirs: tuple[str, ...] = ("__pycache__", "node_modules", "dist", ".pytest_cache")) -> None:
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        if any(part in skip_dirs for part in rel.parts) or path.is_dir():
            continue
        if path.name.endswith(FORBIDDEN) or path.name == ".env":
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def copy_file(rel: str) -> None:
    src = ROOT / rel
    if not src.exists():
        print("  (missing, skipped)", rel)
        return
    target = OUT / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    (OUT / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (OUT / "README.md").write_text(README, encoding="utf-8")

    for rel in ("backend/app", "backend/alembic"):
        copy_tree(ROOT / rel, OUT / rel)
    for rel in ("backend/alembic.ini", "backend/requirements.txt", "backend/scripts/__init__.py", "backend/scripts/create_admin.py"):
        copy_file(rel)
    for rel in ("frontend/src", "frontend/public"):
        if (ROOT / rel).exists():
            copy_tree(ROOT / rel, OUT / rel)
    for p in (ROOT / "frontend").glob("*"):
        if p.is_file() and (p.name in ("index.html", "package.json", "package-lock.json") or p.name.startswith("vite.config")):
            copy_file(f"frontend/{p.name}")
    copy_tree(ROOT / "docs" / "results", OUT / "docs" / "results")

    # Model artifacts actually served by the app.
    active = json.loads((ROOT / "outputs/models/active_model.json").read_text(encoding="utf-8"))["version"]
    copy_file("outputs/models/active_model.json")
    for name in ("metadata.json", "model.joblib"):
        copy_file(f"outputs/models/{active}/{name}")
    card = json.loads((ROOT / "outputs/faced/models/model_card.json").read_text(encoding="utf-8"))
    for rel in ("outputs/faced/models/model_card.json", f"outputs/faced/models/{card['artifact']}",
                "outputs/faced/validation_report.json", "outputs/faced/protocol_split_v1.json", "outputs/faced/final_results.json",
                "outputs/faced/stimulus_category/final_results.json", "outputs/faced/cleaning/results.json"):
        copy_file(rel)
    copy_tree(ROOT / "outputs/affect", OUT / "outputs/affect")
    copy_tree(ROOT / "outputs/cache/xai", OUT / "outputs/cache/xai")
    for rel in ("outputs/metrics", "outputs/graphs"):
        if (ROOT / rel).exists():
            copy_tree(ROOT / rel, OUT / rel)

    files = [p for p in OUT.rglob("*") if p.is_file()]
    bad = [p for p in files if p.name.endswith(FORBIDDEN) or "Dataset" in p.parts or p.name == ".env"]
    assert not bad, f"forbidden files in the bundle: {bad[:5]}"
    size = sum(p.stat().st_size for p in files)
    print(f"wrote {OUT} ({len(files)} files, {size / 1e6:.1f} MB); active DEAP model {active}, FACED model {card['name']}")


if __name__ == "__main__":
    main()
