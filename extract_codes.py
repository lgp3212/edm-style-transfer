#!/usr/bin/env python3
"""
Extract codebook index sequences from training data.
Encodes every sample → sequence of 16 codes → saves as (N, 16) int array.

Run once before training the LSTM prior.

Usage:
    python extract_codes.py
"""

import argparse
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from utils.model import VQVAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-ckpt", type=str,
                        default="phase1/VQVAE_phase1-epoch=091-val_loss=0.0064.ckpt")
    parser.add_argument("--data",  type=str, default="edm_hse_27drums_full.pkl")
    parser.add_argument("--out",   type=str, default="codebook_sequences.pkl")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model...")
    model = VQVAE.load_from_checkpoint(
        args.phase1_ckpt,
        ch=128, num_pitch=27, latent_dim=16, num_embed=256, thres=0.5,
        map_location="cpu", weights_only=False,
    )
    model.eval()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    with open(args.data, "rb") as f:
        data = pickle.load(f)
    data = (data > 0).astype(np.float32)
    if data.shape[1] != 27:
        data = np.transpose(data, (0, 2, 1))
    print(f"  Data shape: {data.shape}")

    # ── Extract sequences ─────────────────────────────────────────────────────
    print("Extracting code sequences...")
    all_sequences = []

    with torch.no_grad():
        for i in range(0, len(data), args.batch_size):
            batch  = torch.tensor(data[i:i+args.batch_size])
            z      = model.encoder(batch)

            # replicate quantize forward to get indices
            x      = z.transpose(1, 2)                          # (B, T, D)
            x_flat = x.detach().reshape(-1, model.quantize.latent_dim)
            dist   = (
                x_flat.pow(2).sum(1, keepdim=True)
                - 2 * x_flat @ model.quantize.embed
                + model.quantize.embed.pow(2).sum(0, keepdim=True)
            )
            idx_flat = torch.argmin(dist, 1)                    # (B*T,)
            T        = x.shape[1]
            idx_seq  = idx_flat.reshape(batch.shape[0], T)      # (B, T)

            all_sequences.append(idx_seq.cpu().numpy())

            if (i // args.batch_size) % 10 == 0:
                print(f"  {i}/{len(data)}")

    sequences = np.concatenate(all_sequences, axis=0).astype(np.int32)  # (N, T)
    print(f"\nSequences shape: {sequences.shape}")
    print(f"Sequence length: {sequences.shape[1]} (time steps after encoder)")
    print(f"Vocab size used: {len(np.unique(sequences))} / 256")

    # ── Compute prob_x1 (starting code distribution) ─────────────────────────
    # used at inference to sample the first code
    first_codes = sequences[:, 0]
    counts      = np.bincount(first_codes, minlength=256).astype(float)
    prob_x1     = counts / counts.sum()
    print(f"Top 5 starting codes: {np.argsort(counts)[::-1][:5].tolist()}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "sequences": sequences,   # (N, T) int32
        "prob_x1":   prob_x1,     # (256,) starting code distribution
        "seq_len":   sequences.shape[1],
        "vocab_size": 256,
    }
    with open(args.out, "wb") as f:
        pickle.dump(out, f)

    print(f"\nSaved: {args.out}")
    print(f"Next: python train_lstm_modal.py")


if __name__ == "__main__":
    main()
