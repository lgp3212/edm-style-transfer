#!/usr/bin/env python3
"""
Generate novel EDM drum patterns using the LSTM prior + VQ-VAE decoder.
Optionally condition on input audio via spectral retrieval of starting code.

Usage:
    # generate 10 random patterns
    python generate.py

    # condition on input audio
    python generate.py --input your_audio.wav

    # generate with specific starting code
    python generate.py --start-code 45

    # control diversity (higher temp = more random, lower = more conservative)
    python generate.py --temp 0.8 --top-k 5
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pretty_midi
import torch

sys.path.insert(0, ".")
from utils.model import VQVAE
from utils.lstm_layers import LSTM_Decoder

DRUM_INDEX_TO_NOTE = {
    0: 27, 1: 31, 2: 36, 3: 37, 4: 38, 5: 39, 6: 40,
    7: 41, 8: 42, 9: 43, 10: 44, 11: 45, 12: 46, 13: 48,
    14: 50, 15: 54, 16: 57, 17: 62, 18: 63, 19: 64,
    20: 69, 21: 70, 22: 75, 23: 76, 24: 77, 25: 56, 26: 55,
}

DRUM_NAMES = [
    "SFX", "Stick", "Kick", "Rimshot", "Snare", "Clap", "Snare Sm",
    "Low Tom", "HH Cl", "High Tom", "HH Acc", "Low Tom2", "HH Op",
    "Mid Tom", "High Tom2", "Tamb", "Rev Rise", "Conga M",
    "Conga H", "Conga L", "Cabasa", "Maracas", "Claves",
    "WBlock H", "WBlock L", "Cowbell", "Splash",
]


# ── LSTM wrapper for sampling ─────────────────────────────────────────────────
class LSTMPriorInference:
    def __init__(self, ckpt_path, vocab_size=256, embed_size=128,
                 hidden_size=512, num_layers=2):
        self.decoder = LSTM_Decoder(
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
        )
        # load weights from lightning checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = {k.replace("decoder.", ""): v
                 for k, v in ckpt["state_dict"].items()
                 if k.startswith("decoder.")}
        self.decoder.load_state_dict(state)
        self.decoder.eval()
        self.vocab_size = vocab_size

    def generate(self, start_code, seq_len=16, temp=1.0, top_k=0, top_p=0):
        """
        Autoregressively generate a sequence of codebook indices.

        Args:
            start_code : int, starting codebook index
            seq_len    : total sequence length (should match VQ-VAE encoder output)
            temp       : sampling temperature (1.0=normal, <1=conservative, >1=random)
            top_k      : top-k sampling (0=disabled)
            top_p      : nucleus sampling (0=disabled)

        Returns:
            sequence : (seq_len,) int array
        """
        inf = -float('Inf')
        sequence = [start_code]

        inp = torch.tensor([[start_code]], dtype=torch.long)  # (1, 1)
        h, c = self.decoder.init_state(1)

        with torch.no_grad():
            for _ in range(seq_len - 1):
                logits, h, c = self.decoder.predict(inp, h, c)  # (vocab,)
                logits = logits.unsqueeze(0) if logits.dim() == 1 else logits

                # temperature
                logits = logits / temp

                # top-k
                if top_k > 0:
                    remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
                    logits[remove] = inf

                # top-p (nucleus)
                if top_p > 0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum_prob = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cum_prob > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0]  = 0
                    remove_idx = remove.scatter(1, sorted_idx, remove)
                    logits[remove_idx] = inf

                prob  = torch.softmax(logits / temp, dim=-1)
                label = torch.multinomial(prob, 1)
                sequence.append(label.item())
                inp = label.unsqueeze(0) if label.dim() == 1 else label

        return np.array(sequence, dtype=np.int32)


# ── Audio conditioning ────────────────────────────────────────────────────────
def get_start_code_from_audio(audio_path, signatures_path):
    import librosa
    
    y, sr = librosa.load(audio_path, sr=44100)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times  = librosa.frames_to_time(onset_frames, sr=sr)

    segments = []
    for i, t in enumerate(onset_times):
        start = int(t * sr)
        dur   = min(onset_times[i+1] - t, 0.25) if i < len(onset_times)-1 else 0.25
        seg   = y[start:int((t+dur)*sr)]
        if len(seg) == 0:
            continue
        features = np.array([
            librosa.feature.spectral_centroid(y=seg, sr=sr)[0].mean(),
            librosa.feature.rms(y=seg)[0].mean(),
            librosa.feature.spectral_rolloff(y=seg, sr=sr)[0].mean(),
            librosa.feature.zero_crossing_rate(seg)[0].mean(),
        ])
        segments.append(features)

    if not segments:
        return None

    # simplified get_start_code_from_audio after recompute
    audio_features = np.stack(segments).mean(axis=0)  # [centroid_hz, rms, rolloff_hz, zcr]

    with open(signatures_path, "rb") as f:
        sigs = pickle.load(f)

    sig_min   = sigs["sig_min"]   # (4,)
    sig_max   = sigs["sig_max"]   # (4,)
    sig_range = sig_max - sig_min
    sig_range[sig_range == 0] = 1.0

    # clip audio features to codebook range then normalize
    audio_norm = (audio_features - sig_min) / sig_range
    audio_norm = np.clip(audio_norm, 0, 1)

    sigs_norm  = (sigs["signatures_raw"] - sig_min) / sig_range

    dists     = np.linalg.norm(sigs_norm - audio_norm, axis=1)
    best_code = int(np.argmin(dists))
    return best_code


# ── MIDI export ───────────────────────────────────────────────────────────────
def pattern_to_midi(pattern, bpm=128, output_path="output.mid"):
    """pattern: (27, 64) float, values in [0,1]"""
    pm    = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    spb   = (60.0 / bpm) / 4  # seconds per 16th note

    for d in range(pattern.shape[0]):
        note = DRUM_INDEX_TO_NOTE.get(d, 38)
        for s in range(pattern.shape[1]):
            v = float(pattern[d, s])
            if v <= 0:
                continue
            vel   = max(1, min(127, int(v * 127)))
            start = s * spb
            drums.notes.append(pretty_midi.Note(
                velocity=vel, pitch=note,
                start=start, end=start + spb * 0.9
            ))

    pm.instruments.append(drums)
    pm.write(output_path)


# ── Print grid ────────────────────────────────────────────────────────────────
def print_grid(pattern, title=""):
    steps = pattern.shape[1]
    markers = "".join("|" if i%16==0 else ("+" if i%4==0 else " ") for i in range(steps))
    print(f"\n{'─'*72}")
    if title: print(f"  {title}")
    print(f"{'─'*72}")
    print(f"  {'Drum':<12} {markers}")
    for d in range(pattern.shape[0]):
        row = "".join("█" if pattern[d,t] > 0 else "·" for t in range(steps))
        if (pattern[d] > 0).any():
            print(f"  {DRUM_NAMES[d]:<12} {row}")
    print(f"{'─'*72}")
    print(f"  Hits: {(pattern>0).sum()}  Density: {(pattern>0).mean()*100:.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lstm-ckpt",   type=str,
                        default="lstm_prior/lstm_prior-epoch=020-val_loss=2.7994-val_acc=0.3988.ckpt",
                        help="Path to LSTM checkpoint")
    parser.add_argument("--phase1-ckpt", type=str,
                        default="phase1/VQVAE_phase1-epoch=091-val_loss=0.0064.ckpt")
    parser.add_argument("--sequences",   type=str, default="codebook_sequences.pkl")
    parser.add_argument("--signatures",  type=str, default="codebook_signatures_audio.pkl")
    parser.add_argument("--input",       type=str, default=None,
                        help="Optional input audio for conditioning")
    parser.add_argument("--start-code",  type=int, default=None,
                        help="Force a specific starting code (overrides audio)")
    parser.add_argument("--n",           type=int, default=10)
    parser.add_argument("--bpm",         type=int, default=128)
    parser.add_argument("--temp",        type=float, default=1.0)
    parser.add_argument("--top-k",       type=int, default=0)
    parser.add_argument("--out-dir",     type=str, default="generated_midi")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading models...")
    lstm = LSTMPriorInference(args.lstm_ckpt)

    vqvae = VQVAE.load_from_checkpoint(
        args.phase1_ckpt,
        ch=128, num_pitch=27, latent_dim=16, num_embed=256, thres=0.5,
        map_location="cpu", weights_only=False,
    )
    vqvae.eval()

    with open(args.sequences, "rb") as f:
        seq_data = pickle.load(f)
    prob_x1 = seq_data["prob_x1"]
    seq_len  = seq_data["seq_len"]

    # ── Determine starting code ───────────────────────────────────────────────
    if args.start_code is not None:
        start_code = args.start_code
        print(f"Using forced start code: {start_code}")
    elif args.input is not None:
        print("Conditioning on input audio...")
        start_code = get_start_code_from_audio(args.input, args.signatures)
        if start_code is None:
            start_code = int(np.random.choice(256, p=prob_x1))
    else:
        start_code = None  # sample fresh each time from prob_x1
        print("Generating unconditionally (sampling start code from prior)")

    # ── Generate ──────────────────────────────────────────────────────────────
    print(f"\nGenerating {args.n} patterns  (temp={args.temp}, top_k={args.top_k})")
    print(f"Output: ./{args.out_dir}/\n")

    densities = []

    for i in range(args.n):
        # pick start code
        sc = start_code if start_code is not None else int(np.random.choice(256, p=prob_x1))

        # generate sequence
        seq = lstm.generate(sc, seq_len=seq_len, temp=args.temp, top_k=args.top_k)

        # decode through VQ-VAE
        with torch.no_grad():
            idx       = torch.tensor(seq).unsqueeze(0).long()  # (1, T)
            label, _  = vqvae.decode_code(idx)
            pattern   = label.squeeze(0).cpu().numpy()          # (27, 64)

        density = float((pattern > 0).mean() * 100)
        densities.append(density)

        title = f"Pattern {i+1:02d}  start={sc}  density={density:.1f}%"
        print_grid(pattern, title=title)

        midi_path = os.path.join(args.out_dir, f"pattern_{i+1:02d}_start{sc}.mid")
        pattern_to_midi(pattern, bpm=args.bpm, output_path=midi_path)
        print(f"  → {midi_path}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"{'='*72}")
    print(f"  Generated {args.n} patterns")
    print(f"  Mean density : {np.mean(densities):.1f}%")
    print(f"  Max density  : {np.max(densities):.1f}%")
    print(f"  Min density  : {np.min(densities):.1f}%")
    print(f"  Output dir   : ./{args.out_dir}/")
    print(f"{'='*72}")
    print(f"\nDrag ./{args.out_dir}/ into GarageBand to listen.")
    if start_code is not None:
        print(f"All patterns conditioned on start code {start_code}.")
        print(f"Try --temp 0.7 for less variation, --temp 1.3 for more.")


if __name__ == "__main__":
    main()
