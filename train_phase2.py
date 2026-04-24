#!/usr/bin/env python3
"""
Phase 2: Fine-tune velocity decoder on top of frozen Phase 1 VQ-VAE.
Encoder + quantizer are frozen. Only the velocity decoder is trained.

Usage:
    python train_phase2.py --phase1-ckpt phase1/VQVAE_phase1-epoch=091-val_loss=0.0064.ckpt
    python train_phase2.py --phase1-ckpt phase1/VQVAE_phase1-epoch=091-val_loss=0.0064.ckpt --epochs 100
"""

import argparse
import os
import pickle

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, ".")

from utils.data import DatasetSampler
from utils.model import VQVAE
from utils.cnn_layers import CNNBlock, ResBlock


# ── Velocity decoder (same structure as Phase 1 decoder, no binarization) ────
class VelocityDecoder(nn.Module):
    def __init__(self, ch, num_pitch, latent_dim):
        super().__init__()
        self.first_layer = CNNBlock(latent_dim, ch, kernel_size=3, padding=1)
        self.cnn_layer   = nn.ModuleList([CNNBlock(ch, ch, kernel_size=3, stride=1, padding=1) for _ in range(2)])
        self.res_layer   = nn.ModuleList([ResBlock(ch, kernel_size=3, padding=1) for _ in range(2)])
        self.final_layer = nn.Conv1d(ch, num_pitch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.first_layer(x)
        x = nn.LeakyReLU(0.2)(x)
        for i in range(2):
            x = nn.Upsample(scale_factor=2)(x)
            x = self.cnn_layer[i](x)
            x = self.res_layer[i](x)
        x = self.final_layer(x)
        return torch.sigmoid(x)  # output in [0, 1], no thresholding


# ── Lightning module ──────────────────────────────────────────────────────────
class Phase2VelocityModel(pl.LightningModule):
    def __init__(self, phase1_ckpt, ch=128, num_pitch=27, latent_dim=16, num_embed=256):
        super().__init__()
        self.save_hyperparameters()

        # Load Phase 1 and freeze encoder + quantizer
        phase1 = VQVAE.load_from_checkpoint(
            phase1_ckpt,
            ch=ch, num_pitch=num_pitch, latent_dim=latent_dim,
            num_embed=num_embed, thres=0.5,
            map_location="cpu", weights_only=False,
        )
        self.encoder  = phase1.encoder
        self.quantize = phase1.quantize

        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.quantize.parameters():
            p.requires_grad = False

        # New trainable velocity decoder
        self.velocity_decoder = VelocityDecoder(ch, num_pitch, latent_dim)

        print(f"Frozen params:    {sum(p.numel() for p in self.encoder.parameters()) + sum(p.numel() for p in self.quantize.parameters()):,}")
        print(f"Trainable params: {sum(p.numel() for p in self.velocity_decoder.parameters()):,}")

    def forward(self, x):
        with torch.no_grad():
            z         = self.encoder(x)
            quant_z, _, _ = self.quantize(z)
        return self.velocity_decoder(quant_z)

    def _masked_mse(self, pred, target):
        hit_mask  = (target > 0.01).float()
        rest_mask = 1 - hit_mask

        if hit_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred.device)

        # MSE on hit positions (learn velocity)
        hit_loss  = ((pred - target) ** 2 * hit_mask).sum()  / hit_mask.sum()

        # Penalize any nonzero prediction at rest positions
        rest_loss = (pred ** 2 * rest_mask).sum() / rest_mask.sum()

        return hit_loss + rest_loss

    def training_step(self, batch, batch_idx):
        x    = batch
        pred = self(x)
        loss = self._masked_mse(pred, x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x    = batch
        pred = self(x)
        loss = self._masked_mse(pred, x)

        # Also log MAE for interpretability
        mask = (x > 0.01).float()
        if mask.sum() > 0:
            mae = ((pred - x).abs() * mask).sum() / mask.sum()
            self.log("val_mae", mae, prog_bar=True)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = optim.AdamW(self.velocity_decoder.parameters(), lr=1e-3)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=1e-5)
        return [opt], [sch]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-ckpt", type=str,
                        default="phase1/VQVAE_phase1-epoch=091-val_loss=0.0064.ckpt")
    parser.add_argument("--data",        type=str, default="edm_hse_27drums_continuous.pkl")
    parser.add_argument("--epochs",      type=int, default=100)
    parser.add_argument("--batch-size",  type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed",        type=int, default=0)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    print("=" * 70)
    print("PHASE 2: VELOCITY DECODER FINE-TUNING")
    print("=" * 70)

    # ── Data ─────────────────────────────────────────────────────────────────
    print(f"\n[1/3] Loading {args.data}...")
    with open(args.data, "rb") as f:
        data = pickle.load(f)

    data = data.astype(np.float32)
    if data.shape[1] != 27:
        data = np.transpose(data, (0, 2, 1))
    print(f"  Shape: {data.shape}  density: {(data > 0).mean()*100:.2f}%")

    num_train  = int(len(data) * 0.8)
    train_data = data[:num_train]
    val_data   = data[num_train:]
    print(f"  Train: {len(train_data)}   Val: {len(val_data)}")

    kw = dict(batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    train_loader = DataLoader(DatasetSampler(train_data), shuffle=True,  **kw)
    val_loader   = DataLoader(DatasetSampler(val_data),   shuffle=False, **kw)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n[2/3] Building Phase 2 model...")
    model = Phase2VelocityModel(phase1_ckpt=args.phase1_ckpt)

    # ── Trainer ───────────────────────────────────────────────────────────────
    os.makedirs("checkpoints/phase2", exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/phase2",
        filename="velocity-{epoch:03d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )
    logger = TensorBoardLogger(
        save_dir="lightning_logs", name="phase2",
        version=f"velocity_seed{args.seed}",
    )

    use_gpu = torch.cuda.is_available()
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1 if use_gpu else "auto",
        callbacks=[ckpt_cb],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    print(f"\n[3/3] Training for {args.epochs} epochs...")
    print("=" * 70)
    trainer.fit(model, train_loader, val_loader)

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    print(f"Best checkpoint : {ckpt_cb.best_model_path}")
    print(f"Best val loss   : {ckpt_cb.best_model_score:.6f}")


if __name__ == "__main__":
    main()
