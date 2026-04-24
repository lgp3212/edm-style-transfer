#!/usr/bin/env python3
"""
EDM Style Transfer — Audio-Conditioned Drum Pattern Generation
--------------------------------------------------------------
Takes arbitrary input audio, extracts spectral features,
finds the nearest EDM rhythm archetype in the VQ-VAE codebook,
reconstructs a dense drum pattern via full model forward pass,
and maps user sounds onto that pattern with Phase 2 velocity.

Usage:
    python end_to_end.py --input your_audio.wav
    python end_to_end.py --input your_audio.wav --tempo 128 --output my_output.wav
    python end_to_end.py --input your_audio.wav --no-click
"""

import argparse
import os
import pickle
import sys

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, ".")
from utils.model import VQVAE
from train_phase2 import Phase2VelocityModel
from generate import get_start_code_from_audio


########################################
# AUDIO UTILS
########################################

def apply_fade(segment, sr, fade_ms=5):
    fade_samples = int(sr * fade_ms / 1000)
    fade_samples = min(fade_samples, len(segment) // 2)
    if fade_samples <= 1:
        return segment
    segment = segment.copy()
    segment[:fade_samples]  *= np.linspace(0, 1, fade_samples)
    segment[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    return segment


def apply_reverb(audio, sr, room_size=0.3, wet=0.3):
    delays = [0.029, 0.037, 0.041, 0.043]
    decays = [0.7,   0.6,   0.5,   0.4  ]
    reverb = np.zeros(len(audio))
    for delay, decay in zip(delays, decays):
        d = int(delay * sr * room_size)
        if d < len(audio):
            delayed = np.zeros(len(audio))
            delayed[d:] = audio[:-d] * decay
            reverb += delayed
    return audio * (1 - wet) + reverb * wet


def add_click_track(output, sr, tempo, bars=4):
    spb         = (60.0 / tempo) / 4
    total_steps = bars * 16
    click       = np.zeros(len(output))
    for step in range(total_steps):
        t0 = int(step * spb * sr)
        if t0 >= len(click):
            break
        if step % 16 == 0:
            freq, amp, dur = 1000, 0.6, 0.02
        elif step % 4 == 0:
            freq, amp, dur = 800,  0.2, 0.015
        else:
            continue
        n   = int(dur * sr)
        t   = np.linspace(0, dur, n)
        hit = np.sin(2 * np.pi * freq * t) * np.exp(-t * 200) * amp
        end = min(t0 + n, len(click))
        click[t0:end] += hit[:end - t0]
    return output + click


########################################
# AUDIO DECOMPOSITION
########################################

def decompose_user_audio(audio_path, sr=44100):
    print(f"  Loading: {audio_path}")
    y, sr = librosa.load(audio_path, sr=sr)

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times  = librosa.frames_to_time(onset_frames, sr=sr)
    print(f"  Detected {len(onset_times)} onsets")

    segments = []
    for i, t in enumerate(onset_times):
        start = int(t * sr)
        dur   = min(onset_times[i+1] - t, 0.25) if i < len(onset_times)-1 else 0.25
        seg   = apply_fade(y[start:int((t+dur)*sr)], sr)
        if len(seg) == 0:
            continue
        features = np.array([
            librosa.feature.spectral_centroid(y=seg, sr=sr)[0].mean(),
            librosa.feature.rms(y=seg)[0].mean(),
            librosa.feature.spectral_rolloff(y=seg, sr=sr)[0].mean(),
            librosa.feature.zero_crossing_rate(seg)[0].mean(),
        ])
        segments.append({"onset_time": t, "audio": seg, "features": features})

    return segments, sr


########################################
# CLUSTERING
########################################

def cluster_user_sounds(segments, n_clusters=9):
    from sklearn.cluster import KMeans
    n_clusters = min(n_clusters, len(segments))
    features   = np.stack([s["features"] for s in segments])
    features   = (features - features.mean(0)) / (features.std(0) + 1e-8)
    labels     = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(features)
    return labels


########################################
# PATTERN GENERATION
########################################

def generate_pattern(start_code, vqvae, phase2, sequences, data):
    """
    Find a real training sample assigned to start_code,
    run it through the full VQ-VAE forward pass (encoder → quantizer → decoder),
    then predict velocities with Phase 2.

    Using full forward pass guarantees dense, valid patterns — the model
    was trained end-to-end on these inputs so it always produces good output.
    Audio conditioning works through start_code selection.
    """
    # find training samples whose first code matches start_code
    matching = np.where(sequences[:, 0] == start_code)[0]

    if len(matching) == 0:
        # no exact match — find nearest dominant code
        print(f"  No exact match for code {start_code}, using nearest...")
        # compute dominant code for all sequences
        from scipy import stats
        dominant = stats.mode(sequences, axis=1).mode.flatten()
        # find closest code
        diffs = np.abs(dominant - start_code)
        matching = np.where(diffs == diffs.min())[0]

    # pick randomly among matches for variety
    sample_idx = int(matching[np.random.randint(len(matching))])
    print(f"  Using training sample {sample_idx} "
          f"(from {len(matching)} matches for code {start_code})")

    # full forward pass through VQ-VAE
    sample = torch.tensor(data[sample_idx].astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        label, logits = vqvae(sample)
        rhythm        = label.squeeze(0).cpu().numpy()   # (27, 64) binary

        # Phase 2: predict velocities at hit positions
        x        = torch.tensor(rhythm.astype(np.float32)).unsqueeze(0)
        vel      = phase2(x).squeeze(0).cpu().numpy()
        velocity = rhythm * vel                           # (27, 64) float

    return rhythm, velocity


########################################
# AUDIO SYNTHESIS
########################################

def map_sounds_to_pattern(segments, labels, rhythm, velocity,
                          tempo=128, bars=4, sr=44100):
    print("  Synthesizing output...")
    total_steps   = bars * 16
    step_duration = 60 / tempo / 4
    total_length  = int(total_steps * step_duration * sr)
    output        = np.zeros(total_length)

    sound_bank = {}
    for seg, label in zip(segments, labels):
        sound_bank.setdefault(int(label), []).append(seg["audio"])

    n_drums     = rhythm.shape[0]
    sounds_used = 0

    for step in range(total_steps):
        for drum in range(n_drums):
            if rhythm[drum, step] <= 0:
                continue
            vel = float(velocity[drum, step])
            if vel <= 0.01:
                continue

            cluster = drum % len(sound_bank)
            if cluster not in sound_bank:
                continue

            sound = sound_bank[cluster][sounds_used % len(sound_bank[cluster])].copy()
            sound = sound * vel
            sound = sound[:int(step_duration * sr * 2)]

            start = int(step * step_duration * sr)
            end   = min(start + len(sound), len(output))
            output[start:end] += sound[:end - start]
            sounds_used += 1

    output = apply_reverb(output, sr, room_size=0.4, wet=0.25)
    peak   = np.max(np.abs(output))
    if peak > 0:
        output = output / peak * 0.8

    print(f"  Used {sounds_used} sound events")
    return output


########################################
# FULL PIPELINE
########################################

def edm_transfer(
    input_path,
    phase1_ckpt,
    phase2_ckpt,
    signatures_path,
    sequences_path,
    data_path,
    output_path="edm_output.wav",
    tempo=128,
    click=True,
):
    print("\n" + "="*60)
    print("EDM STYLE TRANSFER")
    print("="*60)

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n[0/4] Loading models...")
    phase1 = VQVAE.load_from_checkpoint(
        phase1_ckpt,
        ch=128, num_pitch=27, latent_dim=16, num_embed=256, thres=0.5,
        map_location="cpu", weights_only=False,
    )
    phase1.eval()

    phase2 = Phase2VelocityModel(phase1_ckpt=phase1_ckpt)
    ckpt   = torch.load(phase2_ckpt, map_location="cpu", weights_only=False)
    phase2.load_state_dict(ckpt["state_dict"])
    phase2.eval()

    # ── Load data ─────────────────────────────────────────────────────────────
    with open(sequences_path, "rb") as f:
        seq_data  = pickle.load(f)
    sequences = seq_data["sequences"]   # (N, 16) int32

    with open(data_path, "rb") as f:
        raw_data = pickle.load(f)
    data = (raw_data > 0).astype(np.float32)
    if data.shape[1] != 27:
        data = np.transpose(data, (0, 2, 1))

    # ── Decompose input audio ─────────────────────────────────────────────────
    print("\n[1/4] Decomposing input audio...")
    segments, sr = decompose_user_audio(input_path)
    if len(segments) == 0:
        print("ERROR: No sound events detected.")
        return

    # ── Audio conditioning ────────────────────────────────────────────────────
    print("\n[2/4] Audio conditioning...")
    start_code = get_start_code_from_audio(input_path, signatures_path)
    if start_code is None:
        start_code = int(np.random.choice(256))
        print(f"  No onsets detected, using random code {start_code}")

    # ── Generate pattern ──────────────────────────────────────────────────────
    print("\n[3/4] Generating EDM pattern...")
    rhythm, velocity = generate_pattern(start_code, phase1, phase2, sequences, data)
    hits = int((rhythm > 0).sum())
    print(f"  Rhythm hits : {hits} / {rhythm.size} ({rhythm.mean()*100:.1f}% density)")
    if velocity.max() > 0:
        print(f"  Velocity    : mean={velocity[velocity>0].mean():.3f}  max={velocity.max():.3f}")

    # ── Synthesize ────────────────────────────────────────────────────────────
    print("\n[4/4] Synthesizing output...")
    labels = cluster_user_sounds(segments, n_clusters=min(27, len(segments)))
    output = map_sounds_to_pattern(segments, labels, rhythm, velocity,
                                   tempo=tempo, sr=sr)

    if click:
        output = add_click_track(output, sr, tempo)

    sf.write(output_path, output, sr)

    print(f"\n{'='*60}")
    print(f"Input      : {input_path}")
    print(f"Output     : {output_path}")
    print(f"Start code : {start_code}  tempo={tempo}bpm  hits={hits}")
    print(f"{'='*60}\n✓ Done!")

    return output


########################################
# MAIN
########################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        type=str, required=True)
    parser.add_argument("--output",       type=str, default="edm_output.wav")
    parser.add_argument("--phase1-ckpt",  type=str,
                        default="phase1/VQVAE_phase1-epoch=091-val_loss=0.0064.ckpt")
    parser.add_argument("--phase2-ckpt",  type=str,
                        default="phase2/velocity-epoch=021-val_loss=0.0181-val_mae=0.0847.ckpt")
    parser.add_argument("--signatures",   type=str,
                        default="codebook_signatures_audio.pkl")
    parser.add_argument("--sequences",    type=str,
                        default="codebook_sequences.pkl")
    parser.add_argument("--data",         type=str,
                        default="edm_hse_27drums_full.pkl")
    parser.add_argument("--tempo",        type=int, default=128)
    parser.add_argument("--no-click",     action="store_true",
                        help="Disable click track")
    args = parser.parse_args()

    edm_transfer(
        input_path=args.input,
        phase1_ckpt=args.phase1_ckpt,
        phase2_ckpt=args.phase2_ckpt,
        signatures_path=args.signatures,
        sequences_path=args.sequences,
        data_path=args.data,
        output_path=args.output,
        tempo=args.tempo,
        click=not args.no_click,
    )
