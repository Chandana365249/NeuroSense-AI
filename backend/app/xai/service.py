"""Explanations of saved predictions, recomputed from the same model and the same EEG.

A prediction is explained only if recomputing it with the recorded model gives the stored output again (to 1e-6).
Otherwise, and for synthetic test signals, unsupported models or uploads whose signal is not re-sent, the response
says that no explanation is available and why. Nothing here trains, changes or activates a model, or writes to the
database.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.affect.service import AffectService
from app.core.config import Settings
from app.ml.dataset import DatasetError, DEAPDataset
from app.ml.faced import dataset as fd
from app.ml.faced import pipeline as fpipe
from app.ml.participants import is_frozen
from app.models import Prediction
from app.services.faced_service import FacedService
from app.services.model_service import ModelService
from app.services.upload_store import load_upload
from app.xai import attribution as A

TOP = 12
MEANING = {
    "probability": "SHAP values: how much each feature moved this model's high-valence score away from its average "
                   "score over the training data. They add up exactly to the model's score for this trial.",
    "log_odds": "Contributions on the log-odds scale: coefficient x standardised feature value. With the intercept they "
                "add up exactly to the model's decision score for this trial.",
}
CAVEATS = [
    "This shows what this model relied on for this recording. It is not evidence of where or how an emotion arises "
    "in the brain, and a highly weighted channel is not the source of a feeling.",
    "Contributions are not probabilities or confidence; the calibrated confidence is shown separately.",
    "Gamma-band and frontal features are easily affected by muscle and eye-movement activity.",
    "No clinical validity.",
]
DEAP_RATINGS = (("valence", "Valence"), ("arousal", "Arousal"), ("dominance", "Dominance"), ("liking", "Liking"))
FACED_RATINGS = (("Valence", "Valence"), ("Arousal", "Arousal"), ("Liking", "Liking"), ("Familiarity", "Familiarity"))


def unavailable(reason: str, **extra: Any) -> dict[str, Any]:
    return {"available": False, "reason": reason, **extra}


def global_path(settings: Settings, dataset: str, version: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in version)
    return settings.output_dir / "cache" / "xai" / f"global_{dataset}_{safe}.json"


def _matrix(info: list[dict[str, Any]], values: np.ndarray, channels: tuple[str, ...], bands: list[str], family: str) -> list[list[float]]:
    """(channel, band) table of one band-specific family, e.g. log band power."""
    at = {(f["channel"], f["band"]): v for f, v in zip(info, values, strict=True) if f["family"] == family}
    return [[round(float(at.get((c, b), np.nan)), 5) for b in bands] for c in channels]


def trial_payload(*, dataset: str, model_version: str, algorithm: str, names: list[str], channels: tuple[str, ...],
                  X: np.ndarray, att: A.Attribution, probabilities: np.ndarray, stored_high: float, segments: np.ndarray,
                  kept: list[int], fs: int, ratings: list[dict[str, Any]], segment_seconds: float) -> dict[str, Any]:
    info = A.feature_info(names)
    phi = att.values.mean(axis=0)
    raw = X.astype(np.float64).mean(axis=0)
    order = np.argsort(phi)
    pick = lambda idx: [{**info[j], "value": float(raw[j]), "contribution": float(phi[j])} for j in idx]  # noqa: E731
    bands = [b for b in A.BAND_LABELS]
    trial_output = att.base_value + float(phi.sum())
    step = max(1, int(round(fs / 64)))
    seg_band = [[float(sum(v for f, v in zip(info, row, strict=True) if f["band"] == b)) for b in [*bands, "broadband"]]
                for row in att.values]
    return {
        "available": True, "reason": None, "dataset": dataset, "model_version": model_version, "algorithm": algorithm,
        "method": att.method, "output_kind": att.output, "meaning": MEANING[att.output], "caveats": CAVEATS,
        "target_class": "high_valence",
        "base_value": att.base_value, "trial_output": trial_output,
        "probability_high_valence": float(probabilities.mean()),
        "verification": {"stored_probability_high_valence": stored_high, "recomputed_probability_high_valence": float(probabilities.mean()),
                         "matches_stored_prediction": True, "max_segment_reconstruction_error": att.max_abs_error,
                         "trial_reconstruction_error": abs(trial_output - float(att.segment_outputs.mean())),
                         "segments": int(X.shape[0])},
        "top_positive": pick(order[::-1][:TOP]), "top_negative": pick(order[:TOP]),
        "by_band": [{**g, "label": A.label_for(g["group"], "band")} for g in A.grouped(info, phi, "band")],
        "by_channel": A.grouped(info, phi, "channel", only_kind="channel"),
        "by_pair": A.grouped(info, phi, "channel", only_kind="pair"),
        "by_family": [{**g, "label": A.label_for(g["group"], "family")} for g in A.grouped(info, phi, "family")],
        "channels": list(channels), "bands": bands,
        "contribution_map": _matrix(info, phi, channels, bands, "logpow"),
        "trial_view": {
            "segment_seconds": segment_seconds, "kept_segments": [int(i) for i in kept],
            "segment_probability_high": [round(float(p), 4) for p in probabilities],
            "segment_band_contribution": {"bands": [*bands, "broadband"], "values": [[round(x, 6) for x in r] for r in seg_band]},
            "log_band_power": _matrix(info, raw, channels, bands, "logpow"),
            "relative_band_power": _matrix(info, raw, channels, bands, "relpow"),
            "signal": {"sampling_rate": fs / step, "unit": "microvolts after preprocessing (4–45 Hz band-pass)",
                       "segments": [{"index": int(i), "data": np.round(seg[:, ::step].astype(np.float64), 1).tolist()}
                                    for i, seg in zip(kept, segments, strict=True)]},
        },
        "ratings": ratings,
    }


class XaiService:
    def __init__(self, settings: Settings, model_service: ModelService, dataset: DEAPDataset, faced: FacedService,
                 affect: AffectService | None = None) -> None:
        self.settings = settings
        self.model_service = model_service
        self.dataset = dataset
        self.faced = faced
        self.affect = affect or AffectService(settings)
        self._tables: dict[str, Any] = {}

    # ------------------------------------------------------------ local explanations
    def explain(self, item: Prediction, eeg: dict[str, Any] | None = None) -> dict[str, Any]:
        ref = item.source_reference or {}
        if ref.get("synthetic_test_signal") or (eeg or {}).get("synthetic_test_signal"):
            return unavailable("This result came from a synthetic test signal, not recorded EEG. Explaining it would "
                               "describe noise, so no explanation is produced.")
        if item.input_source == "faced_reference":
            return self._explain_faced(item)
        if item.input_source in ("deap_reference", "upload"):
            return self._explain_deap(item, eeg)
        return unavailable(f"Results from '{item.input_source}' cannot be explained.")

    def _explain_deap(self, item: Prediction, eeg: dict[str, Any] | None) -> dict[str, Any]:
        snap = self.model_service.snapshot()
        version = item.model.version if item.model else None
        if version != snap.version:
            return unavailable(f"This result was made by model {version or 'not recorded'}, which is no longer active. "
                               f"Explanations are computed with the model that made the prediction, so none is shown.")
        ref = item.source_reference or {}
        ratings: list[dict[str, Any]] = []
        if item.input_source == "deap_reference":
            pid, trial = ref.get("participant_id"), ref.get("trial_index")
            if pid not in self.settings.development_participants or is_frozen(str(pid)):
                return unavailable("This participant is not available on this server.")
            try:
                signal, values = self.dataset.load_trial(pid, int(trial))
            except DatasetError as exc:
                return unavailable(f"The recording could not be read: {exc}")
            includes_baseline = True
            ratings = [{"name": label, "value": round(float(v), 2), "scale": [1, 9]}
                       for (_, label), v in zip(DEAP_RATINGS, values, strict=True)]
            context = {"participant": str(pid), "trial": int(trial),
                       "ratings": {k: float(v) for (k, _), v in zip(DEAP_RATINGS, values, strict=True)}}
        else:
            eeg = eeg or load_upload(self.settings, item.user_id, item.id)
            if eeg is None:
                return unavailable("This upload was analysed before uploads were stored. Send the same signal again "
                                   "to explain this result.", needs_signal=True)
            context = {}
            signal = np.asarray(eeg["signal"], dtype=np.float64)
            if [int(signal.shape[0]), int(signal.shape[1]), int(eeg.get("sampling_rate", 0))] != \
                    [ref.get("channels"), ref.get("samples"), ref.get("sampling_rate")]:
                return unavailable("The signal sent does not match the shape of the analysed upload.")
            includes_baseline = bool(eeg.get("includes_baseline", False))
        pre = snap.preprocessor.preprocess(signal, has_baseline=includes_baseline)
        if pre.segments.shape[0] == 0:
            return unavailable("Every segment was rejected by the quality checks.")
        if snap.extractor is None:
            return self._explain_cnn(item, snap, pre, ratings)
        X = snap.extractor.extract(pre.segments)  # exactly the input the prediction used
        stored = float((item.probabilities or {}).get("high_valence", np.nan))
        try:
            att = A.explain(snap.model, X)
        except TypeError as exc:
            return unavailable(f"No reliable explanation is implemented for this model ({exc}).")
        return self._finish(item, "DEAP", snap.version, snap.meta.get("algorithm", "unknown"), snap.extractor.feature_names,
                            snap.extractor.channel_names, X, att, stored, pre, snap.meta["preprocessing"]["sampling_rate"],
                            ratings, float(snap.meta["preprocessing"]["segment_seconds"]), context=context)

    def _explain_cnn(self, item: Prediction, snap: Any, pre: Any, ratings: list[dict[str, Any]]) -> dict[str, Any]:
        """Raw-signal CNN: integrated gradients per channel and per second. Band power is shown as a description of
        the signal only; the CNN has no band features, so no band attribution is claimed."""
        from app.ml.factory import make_feature_extractor

        model = snap.model
        stored = float((item.probabilities or {}).get("high_valence", np.nan))
        probabilities = model.predict_proba(pre.segments)[:, A.high_class_index(model)]
        if not np.isfinite(stored) or abs(float(probabilities.mean()) - stored) > A.TOLERANCE:
            return unavailable("Recomputing this result did not reproduce the stored probability, so an explanation "
                               "could not be tied to it.", stored=stored, recomputed=float(probabilities.mean()))
        att = A.explain(model, pre.segments)
        if att.max_abs_error > A.CNN_TOLERANCE:
            return unavailable("The integrated-gradient attributions did not add up to the model output within tolerance.")
        fs = int(snap.meta["preprocessing"]["sampling_rate"])
        describe = make_feature_extractor(self.settings)
        channels = describe.channel_names
        per_channel = att.values.sum(axis=2)  # (segments, channels)
        n_sec = att.values.shape[2] // fs
        per_second = att.values[:, :, : n_sec * fs].reshape(len(att.values), len(channels), n_sec, fs).sum(axis=(1, 3))
        phi = per_channel.mean(axis=0)
        info = A.feature_info(describe.feature_names)
        raw = describe.extract(pre.segments).astype(np.float64).mean(axis=0)
        bands = list(A.BAND_LABELS)
        step = max(1, int(round(fs / 64)))
        trial_output = att.base_value + float(phi.sum())
        return {
            "available": True, "reason": None, "dataset": "DEAP", "model_version": snap.version,
            "algorithm": snap.meta.get("algorithm", "cnn"), "method": att.method, "output_kind": "log_odds",
            "meaning": ("Integrated gradients on the log-odds of higher valence: how much each channel's raw signal moved "
                        "the network's output away from its output for a flat signal at each channel's training mean. "
                        "They add up to the network's log-odds up to a small, measured integration error."),
            "caveats": CAVEATS, "target_class": "high_valence", "base_value": att.base_value, "trial_output": trial_output,
            "probability_high_valence": float(probabilities.mean()),
            "verification": {"stored_probability_high_valence": stored, "recomputed_probability_high_valence": float(probabilities.mean()),
                             "matches_stored_prediction": True, "max_segment_reconstruction_error": att.max_abs_error,
                             "trial_reconstruction_error": abs(trial_output - float(att.segment_outputs.mean())),
                             "segments": int(pre.segments.shape[0]), "tolerance": A.CNN_TOLERANCE},
            "top_positive": [], "top_negative": [], "by_band": [], "by_pair": [], "by_family": [],
            "by_channel": [{"group": c, "contribution": float(v)} for c, v in sorted(zip(channels, phi, strict=True), key=lambda kv: -abs(kv[1]))],
            "by_second": {"seconds": n_sec, "values": [[round(float(x), 6) for x in r] for r in per_second]},
            "channels": list(channels), "bands": bands, "contribution_map": None,
            "trial_view": {
                "segment_seconds": float(snap.meta["preprocessing"]["segment_seconds"]),
                "kept_segments": [int(i) for i in pre.kept_segment_indices],
                "segment_probability_high": [round(float(x), 4) for x in probabilities],
                "segment_band_contribution": None,
                "log_band_power": _matrix(info, raw, channels, bands, "logpow"),
                "relative_band_power": _matrix(info, raw, channels, bands, "relpow"),
                "signal": {"sampling_rate": fs / step, "unit": "microvolts after preprocessing (4–45 Hz band-pass)",
                           "segments": [{"index": int(i), "data": np.round(seg[:, ::step].astype(np.float64), 1).tolist()}
                                        for i, seg in zip(pre.kept_segment_indices, pre.segments, strict=True)]},
            },
            "ratings": ratings, "prediction_id": item.id, "predicted_label": item.predicted_label, "affect": None,
        }

    def _explain_faced(self, item: Prediction) -> dict[str, Any]:
        ref = item.source_reference or {}
        card = self.faced.card()
        if not card or card.get("name") != ref.get("model_version") or self.faced.incompatibility(card):
            return unavailable("The FACED model that made this result is not available in the same version, so it cannot be explained.")
        participant = fd.load_participant(self.faced.root, ref["participant_id"])
        trial = next((t for t in participant.trials if t.video_index == ref.get("video_index")), None)
        if trial is None:
            return unavailable("The FACED trial of this result could not be found.")
        pre = fpipe._preprocessor.preprocess(fpipe.trial_signal(participant, trial), has_baseline=False)
        if pre.segments.shape[0] == 0:
            return unavailable("Every segment was rejected by the quality checks.")
        X = fpipe._extractor.extract(pre.segments).astype(np.float64)  # the same features faced_service.predict uses
        model = self.faced._load_model(card)
        try:
            att = A.explain(model, X)
        except TypeError as exc:
            return unavailable(f"No reliable explanation is implemented for this model ({exc}).")
        ratings = [{"name": label, "value": round(float(trial.ratings[key]), 2), "scale": [0, 7]}
                   for key, label in FACED_RATINGS if trial.ratings.get(key) is not None]
        stored = float((item.probabilities or {}).get("high_valence", np.nan))
        context = {"participant": ref["participant_id"], "trial": int(ref["video_index"]),
                   "ratings": {k.lower(): float(trial.ratings[k]) for k, _ in FACED_RATINGS if trial.ratings.get(k) is not None}}
        return self._finish(item, "FACED", card["name"], card["algorithm"].get("name", "unknown"), fpipe.feature_names(),
                            fd.FACED_CHANNELS, X, att, stored, pre, fpipe.TARGET_FS, ratings,
                            float(fpipe.PREPROCESSING.segment_seconds), model, context=context)

    def _finish(self, item: Prediction, dataset: str, version: str, algorithm: str, names: list[str], channels: tuple[str, ...],
                X: np.ndarray, att: A.Attribution, stored: float, pre: Any, fs: int, ratings: list[dict[str, Any]],
                segment_seconds: float, model: Any = None, context: dict[str, Any] | None = None) -> dict[str, Any]:
        model = model if model is not None else self.model_service.snapshot().model
        probabilities = model.predict_proba(X)[:, A.high_class_index(model)]
        if not np.isfinite(stored) or abs(float(probabilities.mean()) - stored) > A.TOLERANCE:
            return unavailable("Recomputing this result did not reproduce the stored probability, so an explanation "
                               "could not be tied to it.", stored=stored, recomputed=float(probabilities.mean()))
        if att.max_abs_error > A.tolerance_for(att):
            return unavailable("The attributions did not add up to the model output within tolerance.")
        out = trial_payload(dataset=dataset, model_version=version, algorithm=algorithm, names=names, channels=channels, X=X,
                            att=att, probabilities=probabilities, stored_high=stored, segments=pre.segments,
                            kept=pre.kept_segment_indices, fs=fs, ratings=ratings, segment_seconds=segment_seconds)
        out["prediction_id"] = item.id
        out["predicted_label"] = item.predicted_label
        context = context or {}
        out["affect"] = self.affect.estimate(dataset, X, stored, context.get("participant"), context.get("trial"),
                                             context.get("ratings"))
        return out

    # ------------------------------------------------------------ personalised estimates
    def personal(self, item: Prediction) -> dict[str, Any]:
        """Per-dimension models trained only on the same person's other rated trials (dataset recordings only)."""
        try:
            return self._personal(item)
        except (FileNotFoundError, DatasetError):
            return unavailable("Personalised estimates need this person's other recordings, which are not available on "
                               "this server (dataset recordings are only installed in the local version).")

    def _personal(self, item: Prediction) -> dict[str, Any]:
        ref = item.source_reference or {}
        if item.input_source == "deap_reference":
            pid, trial = str(ref.get("participant_id")), int(ref.get("trial_index", -1))
            if pid not in self.settings.development_participants or is_frozen(pid):
                return unavailable("This participant is not available on this server.")
            table = self._deap_table()
            rows = table.participant == pid
            ratings = {int(t): {k: float(r[i]) for i, (k, _) in enumerate(DEAP_RATINGS)}
                       for t, r in zip(table.trial[rows], table.ratings[rows], strict=True)}
            return self.affect.personal("DEAP", table.X[rows].astype(np.float64), table.trial[rows], ratings, trial)
        if item.input_source == "faced_reference":
            pid, video = str(ref.get("participant_id")), int(ref.get("video_index", -1))
            z = self._faced_table()
            rows = z["participant"] == pid
            keys = [str(k) for k in z["rating_keys"]]
            ratings = {int(v): {k.lower(): float(r[keys.index(k)]) for k, _ in FACED_RATINGS[:3]}
                       for v, r in zip(z["video"][rows], z["ratings"][rows], strict=True)}
            return self.affect.personal("FACED", z["X"][rows].astype(np.float64), z["video"][rows].astype(int), ratings, video)
        return unavailable("Personalised estimates need a dataset recording with the person's own ratings of other trials.")

    def _deap_table(self) -> Any:
        if "deap" not in self._tables:
            from app.ml.factory import make_feature_extractor, make_preprocessor
            from app.ml.feature_table import build_feature_table

            table = build_feature_table(self.dataset, make_preprocessor(self.settings), make_feature_extractor(self.settings),
                                        self.settings.cache_dir)
            assert not any(is_frozen(q) for q in set(table.participant.tolist()))
            self._tables["deap"] = table
        return self._tables["deap"]

    def _faced_table(self) -> Any:
        if "faced" not in self._tables:
            path = self.faced.out / f"features_{fpipe.PIPELINE_VERSION}.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(self.faced.out / f"features_{fpipe.PIPELINE_VERSION}.npz", allow_pickle=False) as z:
                self._tables["faced"] = {k: z[k] for k in ("X", "participant", "video", "ratings", "rating_keys")}
        return self._tables["faced"]

    # ------------------------------------------------------------ global explanations (precomputed)
    def global_view(self, dataset: str) -> dict[str, Any]:
        if dataset == "deap":
            meta = self.model_service.metadata
            version = meta["version"] if meta else None
        elif dataset == "faced":
            card = self.faced.card()
            version = card.get("name") if card else None
        else:
            return unavailable("Unknown dataset.")
        if not version:
            return unavailable("No model is available for this dataset.")
        path = global_path(self.settings, dataset, version)
        if not path.exists():
            return unavailable("The global explanation has not been computed for this model yet. Run "
                               "`python -m app.xai.build_global` from backend/.", model_version=version)
        data = json.loads(path.read_text(encoding="utf-8"))
        return {**data, "available": True, "reason": None}
