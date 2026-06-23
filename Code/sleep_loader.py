from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
import json
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


# -----------------------------
# Data containers
# -----------------------------

@dataclass
class ChannelInfo:
    id: int
    name: str
    enabled: bool
    offset: float | None = None
    gain: float | None = None
    rate: float | None = None
    acquisition_range_max: float | None = None
    record_type: str | None = None


@dataclass
class FileSegment:
    filename: str
    tstart: datetime
    duration_ms: int

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000.0


@dataclass
class HypnogramStatus:
    level: int
    label: str
    key: int
    color: int


@dataclass
class ExperimentMetadata:
    exp_path: Path
    file_origin: str | None
    sampling_rate: float
    n_channels: int
    header_offset: int
    channels: list[ChannelInfo]
    acquisition_files: list[FileSegment]
    video_files: list[FileSegment] = field(default_factory=list)
    comment_files: list[FileSegment] = field(default_factory=list)
    hypnogram_files: list[FileSegment] = field(default_factory=list)
    hypnogram_sampling_rate: float | None = None
    hypnogram_statuses: list[HypnogramStatus] = field(default_factory=list)

    @property
    def hypnogram_epoch_s(self) -> float | None:
        if self.hypnogram_sampling_rate and self.hypnogram_sampling_rate > 0:
            return 1.0 / self.hypnogram_sampling_rate
        return None

    @property
    def channel_names(self) -> list[str]:
        return [c.name for c in self.channels if c.enabled]


@dataclass
class LoadedSegment:
    file_path: Path
    tstart: datetime
    duration_s_declared: float
    time_s: np.ndarray
    data: dict[str, np.ndarray]
    dtype_used: str
    inferred_scale: str = "raw"


@dataclass
class LoadedRecording:
    metadata: ExperimentMetadata
    segments: list[LoadedSegment]
    hypnogram: pd.DataFrame | None = None

    def to_metadata_json(self, path: str | Path) -> None:
        payload = asdict(self.metadata)
        payload["exp_path"] = str(self.metadata.exp_path)
        Path(path).write_text(json.dumps(payload, default=str, indent=2))


# -----------------------------
# XML parsing helpers
# -----------------------------


def _text(node: ET.Element | None, default: str | None = None) -> str | None:
    if node is None or node.text is None:
        return default
    return node.text.strip()



def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() == "true"



def _parse_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt



def _parse_file_segments(parent: ET.Element | None) -> list[FileSegment]:
    if parent is None:
        return []
    out: list[FileSegment] = []
    for file_node in parent.findall("./Files/File"):
        filename = _text(file_node.find("FileName"))
        tstart = _text(file_node.find("TStart"))
        duration = _text(file_node.find("Duration"), "0")
        if not filename or not tstart:
            continue
        out.append(
            FileSegment(
                filename=filename,
                tstart=_parse_datetime(tstart),
                duration_ms=int(float(duration)),
            )
        )
    return out



def _parse_channels(acq_node: ET.Element) -> list[ChannelInfo]:
    channel_nodes = acq_node.findall("./EnabledChannel/Channel") or acq_node.findall("./Channels/Channel")
    channels: list[ChannelInfo] = []
    for node in channel_nodes:
        channels.append(
            ChannelInfo(
                id=int(_text(node.find("Id"), "0")),
                name=_text(node.find("Name"), "") or "",
                enabled=_parse_bool(_text(node.find("Enable")), default=True),
                offset=float(_text(node.find("Offset"), "nan")),
                gain=float(_text(node.find("Gain"), "nan")),
                rate=float(_text(node.find("Rate"), "nan")),
                acquisition_range_max=float(_text(node.find("AcquisitionRangeMax"), "nan")),
                record_type=_text(node.find("RecordType")),
            )
        )
    return channels



def parse_exp(exp_path: str | Path) -> ExperimentMetadata:
    exp_path = Path(exp_path)
    root = ET.parse(exp_path).getroot()

    file_origin = _text(root.find("FileOrigin"))

    acq = root.find("Acquisition")
    if acq is None:
        raise ValueError("Could not find <Acquisition> block in experiment file.")

    sampling_rate = float(_text(acq.find("SamplingRate"), "nan"))
    n_channels = int(_text(acq.find("NbChan"), "0"))
    header_offset = int(_text(acq.find("HeaderOffset"), "0"))
    channels = _parse_channels(acq)
    acquisition_files = _parse_file_segments(acq)

    video_files: list[FileSegment] = []
    for video_node in root.findall("./Videos/Video"):
        video_files.extend(_parse_file_segments(video_node))

    comment_files = _parse_file_segments(root.find("Comment"))

    hyp = root.find("Hypnogram")
    hypnogram_files = _parse_file_segments(hyp)
    hypnogram_sampling_rate = None
    hypnogram_statuses: list[HypnogramStatus] = []
    if hyp is not None:
        sr = _text(hyp.find("SamplingRate"))
        hypnogram_sampling_rate = float(sr) if sr else None
        for node in hyp.findall("./Statuses/Status"):
            hypnogram_statuses.append(
                HypnogramStatus(
                    level=int(_text(node.find("Level"), "0")),
                    label=_text(node.find("Label"), "") or "",
                    key=int(_text(node.find("Key"), "0")),
                    color=int(_text(node.find("Color"), "0")),
                )
            )

    return ExperimentMetadata(
        exp_path=exp_path,
        file_origin=file_origin,
        sampling_rate=sampling_rate,
        n_channels=n_channels,
        header_offset=header_offset,
        channels=channels,
        acquisition_files=acquisition_files,
        video_files=video_files,
        comment_files=comment_files,
        hypnogram_files=hypnogram_files,
        hypnogram_sampling_rate=hypnogram_sampling_rate,
        hypnogram_statuses=hypnogram_statuses,
    )


# -----------------------------
# Path resolution
# -----------------------------


def resolve_associated_path(
    exp_path: str | Path,
    filename: str,
    data_root: str | Path | None = None,
) -> Path:
    """
    Resolve a file referenced in the .exp.

    Search order:
    1. data_root / filename                (if data_root is provided)
    2. exp_path.parent / filename
    3. exp_path.parent / basename(filename)
    4. Path(file_origin) / basename(filename)  -> handled externally if desired
    """
    exp_path = Path(exp_path)

    candidates: list[Path] = []
    if data_root is not None:
        data_root = Path(data_root)
        candidates.append(data_root / filename)
        candidates.append(data_root / Path(filename).name)

    candidates.append(exp_path.parent / filename)
    candidates.append(exp_path.parent / Path(filename).name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0] if candidates else exp_path.parent / Path(filename).name


# -----------------------------
# Binary loading
# -----------------------------


def infer_bin_layout(
    file_path: str | Path,
    n_channels: int,
    header_offset: int = 0,
    candidate_dtypes: tuple[str, ...] = ("<u2", "<i2", "<i4", "<f4"),
) -> pd.DataFrame:
    """
    Rank plausible binary layouts based on whether the payload can be reshaped
    into [n_samples, n_channels].

    This does NOT guarantee the correct dtype. It only checks file-size consistency.
    """
    file_path = Path(file_path)
    payload_nbytes = file_path.stat().st_size - header_offset
    rows = []

    for dtype in candidate_dtypes:
        itemsize = np.dtype(dtype).itemsize
        divisible = payload_nbytes % (itemsize * n_channels) == 0
        n_samples = payload_nbytes // (itemsize * n_channels) if divisible else np.nan
        rows.append(
            {
                "dtype": dtype,
                "itemsize": itemsize,
                "divisible": divisible,
                "n_samples_per_channel": n_samples,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["divisible", "itemsize"], ascending=[False, True]).reset_index(drop=True)
    return out



def load_bin_segment(
    file_path: str | Path,
    channel_names: list[str],
    dtype: str = "<u2",
    header_offset: int = 0,
    interleave: Literal["sample", "channel"] = "sample",
) -> dict[str, np.ndarray]:
    """
    Load a binary segment.

    Parameters
    ----------
    dtype:
        Raw dtype, e.g. '<i2' for little-endian int16.
    interleave:
        'sample' assumes [s0_ch0, s0_ch1, ..., s1_ch0, s1_ch1, ...]
        'channel' assumes [ch0_all_samples, ch1_all_samples, ...]
    """
    file_path = Path(file_path)
    raw = np.fromfile(file_path, dtype=np.dtype(dtype), offset=header_offset)
    n_channels = len(channel_names)

    if raw.size % n_channels != 0:
        raise ValueError(
            f"Raw payload size {raw.size} is not divisible by n_channels={n_channels}. "
            f"Check dtype={dtype}, header_offset={header_offset}, or the interleave assumption."
        )

    n_samples = raw.size // n_channels

    if interleave == "sample":
        arr = raw.reshape(n_samples, n_channels)
    elif interleave == "channel":
        arr = raw.reshape(n_channels, n_samples).T
    else:
        raise ValueError("interleave must be 'sample' or 'channel'.")

    return {name: arr[:, idx].copy() for idx, name in enumerate(channel_names)}



def load_recording_segments(
    metadata: ExperimentMetadata,
    data_root: str | Path | None = None,
    dtype: str = "<u2",
    interleave: Literal["sample", "channel"] = "sample",
) -> list[LoadedSegment]:
    loaded: list[LoadedSegment] = []
    channel_names = metadata.channel_names

    for seg in metadata.acquisition_files:
        path = resolve_associated_path(metadata.exp_path, seg.filename, data_root=data_root)
        if not path.exists():
            raise FileNotFoundError(f"Could not find binary segment: {seg.filename}\nTried: {path}")

        data = load_bin_segment(
            path,
            channel_names=channel_names,
            dtype=dtype,
            header_offset=metadata.header_offset,
            interleave=interleave,
        )
        n_samples = len(next(iter(data.values())))
        time_s = np.arange(n_samples, dtype=float) / metadata.sampling_rate

        loaded.append(
            LoadedSegment(
                file_path=path,
                tstart=seg.tstart,
                duration_s_declared=seg.duration_s,
                time_s=time_s,
                data=data,
                dtype_used=dtype,
            )
        )

    return loaded



def concatenate_segments(
    segments: list[LoadedSegment],
    fs: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    """
    Concatenate segments on one continuous relative timeline.

    Returns
    -------
    time_s_global
    data_global
    segment_table
    """
    if not segments:
        raise ValueError("No segments provided.")

    channel_names = list(segments[0].data.keys())
    global_time = []
    global_data = {name: [] for name in channel_names}
    segment_rows = []

    t_cursor = 0.0
    for idx, seg in enumerate(segments):
        n_samples = len(next(iter(seg.data.values())))
        seg_duration_loaded = n_samples / fs
        t = t_cursor + np.arange(n_samples, dtype=float) / fs
        global_time.append(t)
        for name in channel_names:
            global_data[name].append(seg.data[name])
        segment_rows.append(
            {
                "segment_id": idx,
                "file": str(seg.file_path),
                "tstart_wallclock": seg.tstart.isoformat(),
                "global_t0_s": t_cursor,
                "global_t1_s": t_cursor + seg_duration_loaded,
                "duration_s_declared": seg.duration_s_declared,
                "duration_s_loaded": seg_duration_loaded,
                "duration_diff_s": seg_duration_loaded - seg.duration_s_declared,
            }
        )
        t_cursor += seg_duration_loaded

    time_s_global = np.concatenate(global_time)
    data_global = {name: np.concatenate(parts) for name, parts in global_data.items()}
    segment_table = pd.DataFrame(segment_rows)
    return time_s_global, data_global, segment_table


# -----------------------------
# Hypnogram loading
# -----------------------------


def _read_file_bytes_maybe_zip(path: str | Path) -> bytes:
    path = Path(path)
    if path.suffix.lower() != ".zip":
        return path.read_bytes()

    import zipfile

    with zipfile.ZipFile(path, "r") as zf:
        members = [m for m in zf.namelist() if not m.endswith("/") and "__MACOSX" not in m]
        if not members:
            raise ValueError(f"Zip file contains no usable members: {path}")
        return zf.read(members[0])



def _load_plaintext_hypnogram_from_bytes(raw_bytes: bytes) -> np.ndarray:
    text = raw_bytes.decode(errors="ignore").strip()
    if not text:
        return np.array([], dtype=object)

    tokens = [
        tok.strip()
        for tok in text.replace("\r", "\n").replace(",", " ").replace(";", " ").split()
        if tok.strip()
    ]
    return np.array(tokens, dtype=object)



def _load_binary_u16_hypnogram_from_bytes(raw_bytes: bytes) -> np.ndarray:
    if len(raw_bytes) % 2 != 0:
        raise ValueError("Binary hypnogram byte length is not divisible by 2.")
    return np.frombuffer(raw_bytes, dtype="<u2").copy()

def choose_hypnogram_sample_dt(
    dt_est_s: float,
    epoch_s: float | None = None,
    tol_1hz: float = 0.02,
    tol_epoch: float = 0.10,
) -> float:
    """
    Snap estimated hypnogram sample spacing to an exact convention when the evidence is strong.
    """
    if not np.isfinite(dt_est_s):
        return dt_est_s

    if abs(dt_est_s - 1.0) <= tol_1hz:
        return 1.0

    if epoch_s is not None and np.isfinite(epoch_s):
        if abs(dt_est_s - epoch_s) <= max(tol_epoch, 0.05 * epoch_s):
            return float(epoch_s)

    return float(dt_est_s)
def load_single_recording_segment(
    metadata: ExperimentMetadata,
    seg_idx: int,
    data_root: str | Path | None = None,
    dtype: str = "<u2",
    interleave: Literal["sample", "channel"] = "sample",
) -> LoadedSegment:
    channel_names = metadata.channel_names
    seg = metadata.acquisition_files[seg_idx]

    path = resolve_associated_path(metadata.exp_path, seg.filename, data_root=data_root)
    if not path.exists():
        raise FileNotFoundError(f"Could not find binary segment: {seg.filename}\nTried: {path}")

    data = load_bin_segment(
        path,
        channel_names=channel_names,
        dtype=dtype,
        header_offset=metadata.header_offset,
        interleave=interleave,
    )

    n_samples = len(next(iter(data.values())))
    time_s = np.arange(n_samples, dtype=float) / metadata.sampling_rate

    return LoadedSegment(
        file_path=path,
        tstart=seg.tstart,
        duration_s_declared=seg.duration_s,
        time_s=time_s,
        data=data,
        dtype_used=dtype,
    )
def infer_hypnogram_format(
    hyp_file: str | Path,
    known_label_keys: set[int] | None = None,
    expected_duration_s: float | None = None,
    epoch_s: float | None = None,
) -> dict[str, Any]:
    raw_bytes = _read_file_bytes_maybe_zip(hyp_file)
    printable_ratio = sum(1 for b in raw_bytes[:512] if 32 <= b <= 126 or b in (9, 10, 13)) / max(min(len(raw_bytes), 512), 1)

    info: dict[str, Any] = {
        "path": str(hyp_file),
        "size_bytes": len(raw_bytes),
        "printable_ratio": printable_ratio,
    }

    if printable_ratio > 0.85:
        tokens = _load_plaintext_hypnogram_from_bytes(raw_bytes)
        info["format"] = "text"
        info["n_samples"] = int(tokens.size)
        info["unique_values"] = sorted(set(tokens.tolist()))[:20]
        return info

    arr = _load_binary_u16_hypnogram_from_bytes(raw_bytes)
    unique_vals = sorted(set(arr.tolist()))
    info["format"] = "binary_u16"
    info["n_samples"] = int(arr.size)
    info["unique_values"] = unique_vals[:20]

    if known_label_keys is not None:
        info["all_values_known"] = set(unique_vals).issubset(set(known_label_keys))

    if expected_duration_s is not None and arr.size > 0:
        dt_est = expected_duration_s / arr.size
        info["dt_est_s"] = float(dt_est)
        if epoch_s is not None:
            if abs(dt_est - 1.0) < 0.1:
                info["timing_guess"] = "1 Hz label stream"
            elif abs(dt_est - epoch_s) < max(0.25, 0.1 * epoch_s):
                info["timing_guess"] = f"{epoch_s:g} s epoch stream"
            else:
                info["timing_guess"] = "unknown"

    return info



def load_hypnogram_segment(
    hyp_file: str | Path,
    epoch_s: float,
    tstart_wallclock: datetime,
    expected_duration_s: float | None = None,
    label_map: dict[str | int, str] | None = None,
    status_key_to_label: dict[int, str] | None = None,
) -> pd.DataFrame:
    """
    Load ONE hypnogram segment into a table.

    Supports:
    - plain-text token files
    - binary uint16 files containing state codes

    The uploaded .H example is a binary uint16 file whose values match the state
    keys from the .exp. Its timing appears much closer to a 1-second label stream
    than to a direct 5-second epoch list, so we infer the sample interval from the
    declared segment duration when possible.
    """
    raw_bytes = _read_file_bytes_maybe_zip(hyp_file)
    printable_ratio = sum(1 for b in raw_bytes[:512] if 32 <= b <= 126 or b in (9, 10, 13)) / max(min(len(raw_bytes), 512), 1)

    if printable_ratio > 0.85:
        raw_labels = _load_plaintext_hypnogram_from_bytes(raw_bytes)
        sample_dt_s = epoch_s
    else:
        raw_codes = _load_binary_u16_hypnogram_from_bytes(raw_bytes)
        if expected_duration_s is not None and raw_codes.size > 0:
            sample_dt_s = expected_duration_s / raw_codes.size
        else:
            sample_dt_s = epoch_s
        if status_key_to_label:
            raw_labels = np.array([status_key_to_label.get(int(x), str(int(x))) for x in raw_codes], dtype=object)
        else:
            raw_labels = raw_codes.astype(object)

    if raw_labels.size == 0:
        raise ValueError(f"Empty hypnogram file: {hyp_file}")

    t0 = np.arange(raw_labels.size, dtype=float) * sample_dt_s
    t1 = t0 + sample_dt_s
    df = pd.DataFrame(
        {
            "sample_id_local": np.arange(raw_labels.size, dtype=int),
            "t0_s_local": t0,
            "t1_s_local": t1,
            "dt_s": sample_dt_s,
            "label_raw": raw_labels,
            "tstart_wallclock": tstart_wallclock,
        }
    )
    if label_map:
        df["label_mapped"] = df["label_raw"].map(label_map).fillna(df["label_raw"])
    else:
        df["label_mapped"] = df["label_raw"]
    return df



def load_hypnogram_all_segments(
    metadata: ExperimentMetadata,
    data_root: str | Path | None = None,
    label_map: dict[str | int, str] | None = None,
) -> pd.DataFrame:
    epoch_s = metadata.hypnogram_epoch_s
    if epoch_s is None:
        raise ValueError("No hypnogram sampling rate found in metadata.")

    status_key_to_label = {int(s.key): s.label for s in metadata.hypnogram_statuses}

    dfs = []
    for seg_id, seg in enumerate(metadata.hypnogram_files):
        path = resolve_associated_path(metadata.exp_path, seg.filename, data_root=data_root)
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find hypnogram file: {seg.filename}\nTried: {path}"
            )

        df = load_hypnogram_segment(
            path,
            epoch_s=epoch_s,
            tstart_wallclock=seg.tstart,
            expected_duration_s=seg.duration_s,
            label_map=label_map,
            status_key_to_label=status_key_to_label,
        )
        df["segment_id"] = seg_id
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


# -----------------------------
# Epoch alignment / export helpers
# -----------------------------


def make_signal_epoch_table(
    total_duration_s: float,
    epoch_s: float,
    fs: float,
) -> pd.DataFrame:
    n_epochs = int(np.floor(total_duration_s / epoch_s))
    t0 = np.arange(n_epochs, dtype=float) * epoch_s
    t1 = t0 + epoch_s
    return pd.DataFrame(
        {
            "epoch_id": np.arange(n_epochs, dtype=int),
            "t0_s": t0,
            "t1_s": t1,
            "sample_i0": np.round(t0 * fs).astype(int),
            "sample_i1": np.round(t1 * fs).astype(int),
        }
    )



def align_hypnogram_to_signal_epochs(
    signal_epochs: pd.DataFrame,
    hypnogram_df: pd.DataFrame,
    mode: Literal["overlap", "nearest_start"] = "overlap",
) -> pd.DataFrame:
    """
    Join the hypnogram to a signal epoch table.

    If the hypnogram is sampled more finely than the target epochs (for example
    1-second labels aligned onto 5-second signal epochs), overlap mode assigns
    the modal label across all overlapping hypnogram samples.
    """
    out = signal_epochs.copy()
    out["label_raw"] = None
    out["label_mapped"] = None
    out["hyp_segment_id"] = pd.NA
    out["n_hyp_samples"] = 0

    if hypnogram_df.empty:
        return out

    if mode == "nearest_start":
        hyp_starts = (
            hypnogram_df["t0_s_global"].to_numpy()
            if "t0_s_global" in hypnogram_df.columns
            else hypnogram_df["t0_s_local"].to_numpy()
        )
        for idx, row in out.iterrows():
            k = int(np.argmin(np.abs(hyp_starts - row["t0_s"])))
            out.at[idx, "label_raw"] = hypnogram_df.iloc[k]["label_raw"]
            out.at[idx, "label_mapped"] = hypnogram_df.iloc[k]["label_mapped"]
            out.at[idx, "hyp_segment_id"] = hypnogram_df.iloc[k]["segment_id"]
            out.at[idx, "n_hyp_samples"] = 1
        return out

    if "t0_s_global" not in hypnogram_df.columns:
        hyp = hypnogram_df.copy()
        ref = hyp["tstart_wallclock"].min()
        hyp["t0_s_global"] = (hyp["tstart_wallclock"] - ref).dt.total_seconds() + hyp["t0_s_local"]
        hyp["t1_s_global"] = (hyp["tstart_wallclock"] - ref).dt.total_seconds() + hyp["t1_s_local"]
    else:
        hyp = hypnogram_df

    for idx, row in out.iterrows():
        mask = (hyp["t0_s_global"].to_numpy() < row["t1_s"]) & (hyp["t1_s_global"].to_numpy() > row["t0_s"])
        if np.any(mask):
            sub = hyp.loc[mask]
            raw_mode = sub["label_raw"].mode(dropna=True)
            mapped_mode = sub["label_mapped"].mode(dropna=True)
            out.at[idx, "label_raw"] = raw_mode.iloc[0] if len(raw_mode) else None
            out.at[idx, "label_mapped"] = mapped_mode.iloc[0] if len(mapped_mode) else None
            out.at[idx, "hyp_segment_id"] = sub.iloc[0]["segment_id"]
            out.at[idx, "n_hyp_samples"] = int(len(sub))

    return out



def default_label_map() -> dict[str, str]:
    return {
        "WK": "Awake",
        "SWS": "NREM",
        "PS": "REM",
        "ND": "Undefined",
        "TR": "Undefined",
        "Artef": "Undefined",
    }


# -----------------------------
# Debug / QC helpers
# -----------------------------


def summarise_experiment(metadata: ExperimentMetadata) -> dict[str, Any]:
    return {
        "exp_path": str(metadata.exp_path),
        "sampling_rate": metadata.sampling_rate,
        "n_channels": metadata.n_channels,
        "channel_names": metadata.channel_names,
        "header_offset": metadata.header_offset,
        "n_acquisition_segments": len(metadata.acquisition_files),
        "n_video_segments": len(metadata.video_files),
        "n_comment_segments": len(metadata.comment_files),
        "n_hyp_segments": len(metadata.hypnogram_files),
        "hypnogram_epoch_s": metadata.hypnogram_epoch_s,
        "hypnogram_labels": [s.label for s in metadata.hypnogram_statuses],
    }



def qc_segment_lengths(metadata: ExperimentMetadata, loaded_segments: list[LoadedSegment]) -> pd.DataFrame:
    rows = []
    for meta_seg, data_seg in zip(metadata.acquisition_files, loaded_segments, strict=True):
        n_samples = len(next(iter(data_seg.data.values())))
        loaded_duration_s = n_samples / metadata.sampling_rate
        rows.append(
            {
                "file": meta_seg.filename,
                "declared_duration_s": meta_seg.duration_s,
                "loaded_duration_s": loaded_duration_s,
                "diff_s": loaded_duration_s - meta_seg.duration_s,
                "n_samples": n_samples,
                "dtype_used": data_seg.dtype_used,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------
# Local inspection helpers
# -----------------------------


def inspect_file_head(path: str | Path, nbytes: int = 256) -> dict[str, Any]:
    """
    Inspect the first bytes of any file. Useful when a .H file cannot be uploaded.
    """
    path = Path(path)
    raw = path.read_bytes()[:nbytes]
    ascii_preview = ''.join(chr(b) if 32 <= b <= 126 or b in (9, 10, 13) else '.' for b in raw)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "head_hex": raw.hex(' '),
        "head_ascii": ascii_preview,
    }



def inspect_hypnogram_file_local(path: str | Path, nbytes: int = 256) -> dict[str, Any]:
    """
    Quick local check for whether a .H file looks like text or binary.
    """
    info = inspect_file_head(path, nbytes=nbytes)
    raw = Path(path).read_bytes()[:nbytes]
    printable_ratio = sum(1 for b in raw if 32 <= b <= 126 or b in (9, 10, 13)) / max(len(raw), 1)
    info["looks_text_like"] = printable_ratio > 0.85
    if info["looks_text_like"]:
        try:
            text = Path(path).read_text(errors="ignore")
            sample_tokens = text.replace("\r", "\n").replace(",", " ").replace(";", " ").split()[:30]
            info["sample_tokens"] = sample_tokens
        except Exception as exc:
            info["text_read_error"] = repr(exc)
    return info



def preview_loaded_channels(
    data: dict[str, np.ndarray],
    fs: float,
    start_s: float = 0.0,
    duration_s: float = 10.0,
) -> pd.DataFrame:
    """
    Small preview table to compare channels before plotting.
    """
    i0 = int(round(start_s * fs))
    i1 = int(round((start_s + duration_s) * fs))
    rows = []
    for name, x in data.items():
        seg = np.asarray(x[i0:i1])
        rows.append(
            {
                "channel": name,
                "n": len(seg),
                "min": float(np.min(seg)) if len(seg) else np.nan,
                "max": float(np.max(seg)) if len(seg) else np.nan,
                "mean": float(np.mean(seg)) if len(seg) else np.nan,
                "std": float(np.std(seg)) if len(seg) else np.nan,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------
# Example usage
# -----------------------------

if __name__ == "__main__":
    exp_path = "/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/converted/LC_PD_wk2_M1_default.exp"

    meta = parse_exp(exp_path)
    print("Metadata summary:")
    print(json.dumps(summarise_experiment(meta), indent=2))

    print("\nHypnogram statuses:")
    for status in meta.hypnogram_statuses:
        print(status)

    print("\nTo continue:")
    print("1) Put the .bin and .H files in the same folder as the .exp, or set data_root=...")
    print("2) Run infer_bin_layout() on one .bin file")
    print("3) Load one segment with load_bin_segment(...)")
    print("4) Visually inspect EEG / EMG / TTL before trusting the dtype and interleave")
