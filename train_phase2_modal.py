"""
Modal wrapper for Phase 2 velocity decoder fine-tuning.

Setup (one time):
    export MODAL_PROFILE=personal
    modal volume put vqvae-data edm_hse_27drums_continuous.pkl /data/edm_hse_27drums_continuous.pkl
    modal volume put vqvae-data phase1/VQVAE_phase1-epoch=091-val_loss=0.0064.ckpt /data/checkpoints/phase1/VQVAE_phase1-epoch=091-val_loss=0.0064.ckpt

Run:
    modal run train_phase2_modal.py
    modal run train_phase2_modal.py --epochs 150
"""

import modal

app = modal.App("vqvae-phase2")

volume = modal.Volume.from_name("vqvae-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2",
        "torch==2.2.0",
        "torchaudio==2.2.0",
        "pytorch-lightning==2.2.0",
        "tensorboard",
        "prdc",
        "librosa",
        "soundfile",
        "scikit-learn",
        "scipy",
        "einops",
        "encodec",
        "vocos",
        "pedalboard",
        "h5py",
        "tqdm",
        "matplotlib",
        "pandas",
        "gdown",
        "pretty_midi",
    )
    .add_local_dir("utils", remote_path="/root/utils")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 4,
    volumes={"/data": volume},
)
def train(
    epochs: int = 100,
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 0,
    data_path: str = "/data/edm_hse_27drums_continuous.pkl",
    phase1_ckpt: str = "/data/checkpoints/phase1/VQVAE_phase1-epoch=091-val_loss=0.0064.ckpt",
):
    import os, pickle, sys
    sys.path.insert(0, "/root")

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    from torch.utils.data import DataLoader

    from utils.data import DatasetSampler
    from utils.model import VQVAE
    from utils.cnn_layers import CNNBlock, ResBlock

    # ── Velocity decoder ──────────────────────────────────────────────────────
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
            return torch.sigmoid(x)

    # ── Lightning module ──────────────────────────────────────────────────────
    class Phase2VelocityModel(pl.LightningModule):
        def __init__(self, phase1_ckpt, ch=128, num_pitch=27, latent_dim=16, num_embed=256):
            super().__init__()

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

            self.velocity_decoder = VelocityDecoder(ch, num_pitch, latent_dim)

            frozen    = sum(p.numel() for p in self.encoder.parameters()) + sum(p.numel() for p in self.quantize.parameters())
            trainable = sum(p.numel() for p in self.velocity_decoder.parameters())
            print(f"  Frozen params:    {frozen:,}")
            print(f"  Trainable params: {trainable:,}")

        def forward(self, x):
            with torch.no_grad():
                z             = self.encoder(x)
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
            pred = self(batch)
            loss = self._masked_mse(pred, batch)
            self.log("train_loss", loss, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            pred = self(batch)
            loss = self._masked_mse(pred, batch)

            mask = (batch > 0.01).float()
            if mask.sum() > 0:
                mae = ((pred - batch).abs() * mask).sum() / mask.sum()
                self.log("val_mae", mae, prog_bar=True)

            self.log("val_loss", loss, prog_bar=True)
            return loss

        def configure_optimizers(self):
            opt = optim.AdamW(self.velocity_decoder.parameters(), lr=1e-3)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=1e-5)
            return [opt], [sch]

    # ── Data ─────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("PHASE 2: VELOCITY DECODER FINE-TUNING — Modal / A10G")
    print("=" * 70)

    pl.seed_everything(seed)

    print(f"\n[1/3] Loading {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    data = data.astype(np.float32)
    if data.shape[1] != 27:
        data = np.transpose(data, (0, 2, 1))
    print(f"  Shape: {data.shape}  density: {(data > 0).mean()*100:.2f}%")

    num_train  = int(len(data) * 0.8)
    train_data = data[:num_train]
    val_data   = data[num_train:]
    print(f"  Train: {len(train_data)}   Val: {len(val_data)}")

    kw = dict(batch_size=batch_size, pin_memory=True, num_workers=num_workers)
    train_loader = DataLoader(DatasetSampler(train_data), shuffle=True,  **kw)
    val_loader   = DataLoader(DatasetSampler(val_data),   shuffle=False, **kw)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n[2/3] Building Phase 2 model...")
    model = Phase2VelocityModel(phase1_ckpt=phase1_ckpt)

    # ── Trainer ───────────────────────────────────────────────────────────────
    ckpt_dir = "/data/checkpoints/phase2"
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        monitor="val_loss",
        dirpath=ckpt_dir,
        filename="velocity-{epoch:03d}-{val_loss:.4f}-{val_mae:.4f}",
        save_top_k=3,
        mode="min",
    )
    logger = TensorBoardLogger(
        save_dir="/data/logs", name="phase2",
        version=f"velocity_seed{seed}",
    )

    print(f"\n[3/3] Training {epochs} epochs, batch_size={batch_size}...")
    print("=" * 70)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[ckpt_cb],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_loader)

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    print(f"Best checkpoint : {ckpt_cb.best_model_path}")
    print(f"Best val loss   : {ckpt_cb.best_model_score:.6f}")
    print(f"\nDownload with:")
    print(f"  modal volume get vqvae-data checkpoints/phase2/ ./checkpoints/phase2/")

    volume.commit()


@app.local_entrypoint()
def main(
    epochs: int = 100,
    batch_size: int = 256,
    seed: int = 0,
):
    print(f"Launching Phase 2: epochs={epochs}, batch_size={batch_size}, seed={seed}")
    train.remote(epochs=epochs, batch_size=batch_size, seed=seed)
