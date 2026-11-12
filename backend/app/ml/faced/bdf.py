"""Minimal, dependency-free reader for 24-bit BDF (BioSemi Data Format) files.

Only what FACED needs: parse the header, then decode a sample window of selected signals through a read-only memory
map, so a 60-80 minute recording is never loaded whole. Values are returned in the file's physical unit, scaled
with the standard EDF/BDF calibration: physical = (digital - dig_min) * (phys_max - phys_min) / (dig_max - dig_min)
+ phys_min.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class BDFError(ValueError):
    """Raised when a file is not a readable 24-bit BDF file."""


@dataclass(frozen=True)
class BDFSignal:
    label: str
    physical_dimension: str
    physical_min: float
    physical_max: float
    digital_min: int
    digital_max: int
    samples_per_record: int

    @property
    def gain(self) -> float:
        return (self.physical_max - self.physical_min) / (self.digital_max - self.digital_min)


@dataclass(frozen=True)
class BDFHeader:
    path: Path
    header_bytes: int
    n_records: int
    record_duration: float
    signals: tuple[BDFSignal, ...]

    @property
    def labels(self) -> list[str]:
        return [s.label for s in self.signals]

    @property
    def record_bytes(self) -> int:
        return sum(s.samples_per_record for s in self.signals) * 3

    def sampling_rate(self, index: int) -> float:
        return self.signals[index].samples_per_record / self.record_duration

    @property
    def n_samples(self) -> int:
        return self.n_records * self.signals[0].samples_per_record


def _field(raw: bytes, start: int, width: int) -> str:
    return raw[start:start + width].decode("ascii", errors="replace").strip()


def read_header(path: Path | str) -> BDFHeader:
    path = Path(path)
    with path.open("rb") as f:
        general = f.read(256)
        if len(general) < 256 or general[:1] != b"\xff" or general[1:8] != b"BIOSEMI":
            raise BDFError(f"{path.name}: not a BDF file (missing 0xFF 'BIOSEMI' identifier)")
        try:
            header_bytes = int(_field(general, 184, 8))
            n_records = int(_field(general, 236, 8))
            record_duration = float(_field(general, 244, 8))
            ns = int(_field(general, 252, 4))
        except ValueError as exc:
            raise BDFError(f"{path.name}: malformed general header") from exc
        if ns <= 0 or header_bytes != 256 * (ns + 1) or record_duration <= 0:
            raise BDFError(f"{path.name}: inconsistent header (signals={ns}, header bytes={header_bytes})")
        raw = f.read(256 * ns)

    def block(offset_per: int, width: int) -> list[str]:
        base = sum(w for _, w in _LAYOUT[:offset_per]) * ns
        return [_field(raw, base + i * width, width) for i in range(ns)]

    labels, dims = block(0, 16), block(2, 8)
    pmin, pmax, dmin, dmax = (block(3, 8), block(4, 8), block(5, 8), block(6, 8))
    spr = block(8, 8)
    try:
        signals = tuple(
            BDFSignal(labels[i], dims[i], float(pmin[i]), float(pmax[i]), int(dmin[i]), int(dmax[i]), int(spr[i]))
            for i in range(ns)
        )
    except ValueError as exc:
        raise BDFError(f"{path.name}: malformed signal header") from exc
    expected = header_bytes + n_records * sum(s.samples_per_record for s in signals) * 3
    actual = path.stat().st_size
    if n_records < 0 or actual < expected:
        raise BDFError(f"{path.name}: file is truncated ({actual} bytes, header implies {expected})")
    return BDFHeader(path, header_bytes, n_records, record_duration, signals)


# EDF/BDF per-signal header fields in order: (name, width).
_LAYOUT = (("label", 16), ("transducer", 80), ("physical_dimension", 8), ("physical_min", 8), ("physical_max", 8),
           ("digital_min", 8), ("digital_max", 8), ("prefiltering", 80), ("samples_per_record", 8), ("reserved", 32))


def read_window(header: BDFHeader, channels: list[int], start: int, stop: int) -> np.ndarray:
    """Physical values of `channels` for samples [start, stop) -> (len(channels), stop - start) float64.

    Requires every selected signal to share the first signal's samples-per-record (true for FACED)."""
    spr = header.signals[channels[0]].samples_per_record
    if any(header.signals[c].samples_per_record != spr for c in channels):
        raise BDFError("selected signals have different sampling rates")
    if not 0 <= start < stop <= header.n_samples:
        raise BDFError(f"window [{start}, {stop}) outside the recording (0, {header.n_samples})")
    r0, r1 = start // spr, -(-stop // spr)
    data = np.memmap(header.path, dtype=np.uint8, mode="r", offset=header.header_bytes,
                     shape=(header.n_records, header.record_bytes))
    offsets = np.cumsum([0] + [s.samples_per_record * 3 for s in header.signals])
    out = np.empty((len(channels), stop - start))
    for k, c in enumerate(channels):
        chunk = np.asarray(data[r0:r1, offsets[c]:offsets[c] + spr * 3]).reshape(-1, 3).astype(np.int32)
        digital = chunk[:, 0] | (chunk[:, 1] << 8) | (chunk[:, 2] << 16)
        digital = np.where(digital >= 1 << 23, digital - (1 << 24), digital)
        s = header.signals[c]
        values = (digital - s.digital_min) * s.gain + s.physical_min
        out[k] = values[start - r0 * spr: stop - r0 * spr]
    del data
    return out
