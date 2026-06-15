#!/usr/bin/env python3
"""
recompute codebook signatures in true audio space,
extracts librosa features from original WAV files and averages per codebook entry
fixes the retrieval problem where all audio mapped to the same codebook entry

usage:
    python recompute_signatures_audio.py --wav-dir edm_hse_id_001-004_wav
"""

import argparse
import pickle
import sys
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

sys.path.insert(0, ".")


def extract_audio_features(wav_path, sr=44100):
    """
    Extract the same 4 features used at inference time.
    Returns (4,) array: [centroid_hz, rms, rolloff_hz, zcr]
    """
    try:
        y, sr = librosa.load(wav_path, sr=sr)
        if len(y) == 0:
            return None

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
        rms      = librosa.feature.rms(y=y)[0].mean()
        rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean()
        zcr      = librosa.feature.zero_crossing_rate(y)[0].mean()

        return np.array([centroid, rms, rolloff, zcr], dtype=np.float32)
    except Exception as e:
        print(f"  Warning: failed on {wav_path.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-dir",     type=str,
                        default="edm_hse_id_001-004_wav")
    parser.add_argument("--sequences",   type=str,
                        default="codebook_sequences.pkl")
    parser.add_argument("--out",         type=str,
                        default="codebook_signatures_audio.pkl")
    parser.add_argument("--sr",          type=int, default=44100)
    args = parser.parse_args()

    # ── Load codebook sequences ───────────────────────────────────────────────
    print("Loading codebook sequences...")
    with open(args.sequences, "rb") as f:
        seq_data = pickle.load(f)

    sequences  = seq_data["sequences"]   # (N, 16) int32
    vocab_size = seq_data["vocab_size"]  # 256
    N          = len(sequences)
    print(f"  {N} sequences, vocab={vocab_size}")

    # dominant code per sample = mode of its 16-step sequence
    from scipy import stats
    dominant_codes = stats.mode(sequences, axis=1).mode.flatten()  # (N,)
    print(f"  Dominant codes computed for {N} samples")

    # ── Load WAV files in sorted order ────────────────────────────────────────
    print(f"\nScanning WAV files in {args.wav_dir}...")
    wav_dir  = Path(args.wav_dir)
    wav_files = sorted(list(wav_dir.glob("*.wav")))
    print(f"  Found {len(wav_files)} WAV files")

    if len(wav_files) != N:
        print(f"  WARNING: {len(wav_files)} WAVs != {N} pkl samples")
        print(f"  Using min({len(wav_files)}, {N}) = {min(len(wav_files), N)}")
        n_process = min(len(wav_files), N)
    else:
        n_process = N

    # ── Extract features per WAV ──────────────────────────────────────────────
    print(f"\nExtracting audio features from {n_process} WAV files...")
    feature_sum   = np.zeros((vocab_size, 4), dtype=np.float64)
    feature_count = np.zeros(vocab_size, dtype=np.int64)
    all_features  = np.zeros((n_process, 4), dtype=np.float32)

    for i, wav_path in enumerate(tqdm(wav_files[:n_process])):
        feats = extract_audio_features(wav_path, sr=args.sr)
        if feats is None:
            # use zeros for failed files
            feats = np.zeros(4, dtype=np.float32)

        all_features[i] = feats
        code = int(dominant_codes[i])
        feature_sum[code]   += feats
        feature_count[code] += 1

    # ── Average per codebook entry ────────────────────────────────────────────
    print("\nComputing per-entry averages...")
    signatures = np.zeros((vocab_size, 4), dtype=np.float32)
    for c in range(vocab_size):
        if feature_count[c] > 0:
            signatures[c] = feature_sum[c] / feature_count[c]

    # for unused entries, use global mean
    global_mean = all_features.mean(axis=0)
    unused = feature_count == 0
    signatures[unused] = global_mean
    print(f"  Used entries: {(~unused).sum()} / {vocab_size}")
    print(f"  Unused entries (set to global mean): {unused.sum()}")

    # ── Print feature ranges ──────────────────────────────────────────────────
    feat_names = ["centroid_hz", "rms", "rolloff_hz", "zcr"]
    print(f"\nSignature feature ranges (audio space):")
    for i, name in enumerate(feat_names):
        print(f"  {name:15s}: [{signatures[:,i].min():.4f}, {signatures[:,i].max():.4f}]")

    print(f"\nGlobal audio feature stats (all {n_process} files):")
    for i, name in enumerate(feat_names):
        print(f"  {name:15s}: mean={all_features[:,i].mean():.4f}  "
              f"std={all_features[:,i].std():.4f}")

    # ── Normalization params ──────────────────────────────────────────────────
    sig_min   = signatures.min(axis=0)
    sig_max   = signatures.max(axis=0)
    sig_range = sig_max - sig_min
    sig_range[sig_range == 0] = 1.0

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "signatures_raw":  signatures,          # (256, 4) in audio space
        "sig_min":         sig_min,
        "sig_max":         sig_max,
        "feature_names":   feat_names,
        "feature_count":   feature_count,       # how many samples per entry
        "global_mean":     global_mean,
    }
    with open(args.out, "wb") as f:
        pickle.dump(out, f)

    print(f"\nSaved: {args.out}")
    print(f"\nNow update end_to_end.py and generate.py to use:")
    print(f"  --signatures {args.out}")
    print(f"\nAnd remove the feature mapping/rescaling in get_start_code_from_audio")
    print(f"since audio and codebook features are now in the same space.")


if __name__ == "__main__":
    main()
