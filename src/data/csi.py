"""src/data/csi.py — CSI (CALVIN with Speech Instructions) dataset + TTS synthesis.

B3 ablation only. Reuses the original CALVIN trajectories; the text instruction is
converted to a SYNTHESIZED 16 kHz mono wav via TTS and frame-aligned to the
8-frame CALVIN action window (same wav segment repeated over the window, matching the
CSI alignment convention). This is synthesized speech, not human natural speech —
see docs/limitations.md.

TTS priority: edge-tts (natural-ish neural voices, requires network) -> espeak-ng
(fully local fallback).
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

SAMPLE_RATE = 16000
WINDOW = 8  # CALVIN frame-alignment window (CSI convention)


# --------------------------------------------------------------------------- #
# TTS synthesis
# --------------------------------------------------------------------------- #
def synth_tts_edge(text: str, out_wav: Path, voice: str = "en-US-JennyNeural") -> bool:
    import edge_tts

    async def _run():
        comm = edge_tts.Communicate(text, voice)
        await comm.save(str(out_wav))

    asyncio.run(_run())
    return out_wav.exists() and out_wav.stat().st_size > 0


def synth_tts_espeak(text: str, out_wav: Path) -> bool:
    """Local fallback: espeak-ng -> wav, resampled to 16 kHz mono."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(["espeak-ng", "-w", tmp.name, "-s", "150", text],
                       check=True, capture_output=True, timeout=30)
        return convert_to_16k(tmp.name, out_wav)
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def convert_to_16k(src_wav: Path, dst_wav: Path) -> bool:
    """Convert any wav to 16 kHz mono via ffmpeg (or librosa fallback)."""
    try:
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-i", str(src_wav), "-ar", "16000", "-ac", "1",
                            str(dst_wav)], check=True, capture_output=True)
            return True
        import librosa
        import soundfile as sf
        wav, sr = librosa.load(str(src_wav), sr=SAMPLE_RATE, mono=True)
        sf.write(str(dst_wav), wav, SAMPLE_RATE)
        return True
    except Exception:
        return False


def synthesize(text: str, out_wav: Path) -> bool:
    """Synthesize `text` to 16 kHz mono wav. Returns True on success.

    edge-tts outputs 24 kHz — always resample to 16 kHz mono afterwards."""
    tmp = Path(str(out_wav) + ".raw.wav")
    if synth_tts_edge(text, tmp) or synth_tts_espeak(text, tmp):
        return convert_to_16k(tmp, out_wav)
    return False


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CSIDataset(Dataset):
    """B3 dataset: (image window, frame-aligned audio wav, actions, lang).

    Trajectory npz files are shared with CALVIN preprocessing (calvin_root/<split>/<id>.npz);
    synthesized wavs live in csi_root/wav/<sha256(lang)>.wav (same keying as rollout).
    """

    def __init__(self, csi_root: Path, calvin_root: Path, split: str = "sft_train",
                 chunk_size: int = None):
        self.csi_root = Path(csi_root)
        self.calvin_root = Path(calvin_root)
        self.split = split
        # CSI reuses CALVIN trajectory ids; split manifests are shared with CALVIN.
        from .splits import load_manifest
        manifest = load_manifest(self.calvin_root / "splits" / f"calvin_{split}.json")
        self.ids: List[str] = sorted(manifest["ids"])
        self.chunk_size = chunk_size
        self._index = None
        if chunk_size and chunk_size > 0:
            self._index = []
            for i in range(len(self.ids)):
                with np.load(self._npz_path(i)) as f:
                    t = int(f["actions"].shape[0])
                for s in range(0, t, chunk_size):
                    self._index.append((i, s, min(s + chunk_size, t)))

    def _npz_path(self, i: int) -> Path:
        traj_id = self.ids[i]
        p = self.calvin_root / self.split / f"{traj_id}.npz"
        if not p.exists() and self.split == "sft_val":
            p = self.calvin_root / "sft_train" / f"{traj_id}.npz"
        return p

    def __len__(self):
        return len(self._index) if self._index is not None else len(self.ids)

    def __getitem__(self, idx: int):
        if self._index is not None:
            i, s, e = self._index[idx]
            sl = slice(s, e)
        else:
            i, sl = idx, slice(0, None)
        with np.load(self._npz_path(i)) as f:
            rgb = torch.from_numpy(f["rgb"][sl]).float() / 255.0   # (K,3,224,224)
            actions = torch.from_numpy(f["actions"][sl])            # (K,7)
            lang = str(f["lang"])
            proprio = f["proprio"][sl] if "proprio" in f else np.zeros(
                (actions.shape[0], 8), dtype=np.float32)
        wav = self._load_wav(lang)
        return {"pixel_values": rgb, "actions": actions, "lang": lang, "audio_wav": wav,
                "id": self.ids[i], "proprio": torch.from_numpy(proprio)}

    def _load_wav(self, lang: str) -> Optional[torch.Tensor]:
        import hashlib
        import soundfile as sf
        name = hashlib.sha256(lang.encode()).hexdigest()[:16]
        wav_path = self.csi_root / "wav" / f"{name}.wav"
        if not wav_path.exists():
            return None  # missing TTS -> caller may drop audio (graceful degradation)
        audio, sr = sf.read(str(wav_path), dtype="float32")
        assert sr == SAMPLE_RATE, f"{wav_path}: expected {SAMPLE_RATE} Hz, got {sr}"
        return torch.from_numpy(audio)  # (T_audio,), fed to Whisper encoder


def align_wav_to_windows(wav: np.ndarray, n_frames: int, window: int = WINDOW) -> np.ndarray:
    """Frame-align a full-instruction wav to CALVIN's 8-frame windows.

    Splits the instruction audio into ceil(n_frames/window) segments; each segment is
    the audio for its window. Returns list of segment arrays (len = ceil(n_frames/8)).
    """
    n_win = max(1, int(np.ceil(n_frames / window)))
    seg_len = len(wav) // n_win
    segs = [wav[i * seg_len:(i + 1) * seg_len] for i in range(n_win)]
    return np.asarray(segs, dtype=object)  # object array of variable-length segments
