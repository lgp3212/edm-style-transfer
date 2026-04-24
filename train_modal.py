"""
Modal wrapper for Phase 1 VQ-VAE binary drum training.
Codebook usage + perplexity logged every epoch to catch collapse early.

Setup (one time):
    export MODAL_PROFILE=personal
    modal volume create vqvae-data
    modal volume put vqvae-data edm_hse_27drums_full.pkl /data/edm_hse_27drums_full.pkl

Run:
    modal run train_modal.py
    modal run train_modal.py --epochs 50
    modal run train_modal.py --epochs 300 --batch-size 512
"""

import modal

app = modal.App("vqvae-phase1")

volume = modal.Volume.from_name("vqvae-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # numpy pinned to <2 to avoid torch compatibility crash
        "numpy<2",
        "torch==2.2.0",
        "torchaudio==2.2.0",
        "pytorch-lightning==2.2.0",
        "tensorboard",
        "pretty_midi",
        # your utils deps
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
    epochs: int = 30,
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 0,
    data_path: str = "/data/data/edm_hse_27drums_full.pkl",
):
    import os, pickle, sys
    sys.path.insert(0, "/root")

    import numpy as np
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, Callback
    from pytorch_lightning.loggers import TensorBoardLogger
    from torch.utils.data import DataLoader

    from utils.data import DatasetSampler
    from utils.model import VQVAE

    # ── Codebook monitor ──────────────────────────────────────────────────────
    class CodebookMonitor(Callback):
        """
        Logs every epoch:
          - Perplexity : model's own value (batch-averaged). Want ~num_embed.
          - Usage %    : fraction of codes seen this epoch.  Want > 80%.

        Collapse signals:
          - Perplexity << num_embed  (codes clustering in a small region)
          - Usage % dropping epoch over epoch
          - red warning if usage < 20%
        """
        def __init__(self, num_embed: int):
            self.num_embed   = num_embed
            self.all_idx     = []
            self.all_perplex = []

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if hasattr(pl_module, "last_encodings") and pl_module.last_encodings is not None:
                self.all_idx.append(pl_module.last_encodings.detach().cpu())
            if hasattr(pl_module, "last_perplex") and pl_module.last_perplex is not None:
                self.all_perplex.append(pl_module.last_perplex.detach().cpu().item())

        def on_train_epoch_end(self, trainer, pl_module):
            epoch = trainer.current_epoch

            # perplexity — use model's own computation (already correct)
            if self.all_perplex:
                mean_perplex = sum(self.all_perplex) / len(self.all_perplex)
                self.all_perplex = []
            else:
                mean_perplex = float("nan")
                print(f"[Codebook | Epoch {epoch:03d}] WARNING: no perplexity — "
                      "add last_perplex to training_step in model.py")

            # usage — fraction of codebook entries seen this epoch
            if self.all_idx:
                all_enc      = torch.cat(self.all_idx, dim=0)
                self.all_idx = []
                unique       = torch.unique(all_enc)
                usage_pct    = 100.0 * len(unique) / self.num_embed
            else:
                usage_pct = float("nan")
                print(f"[Codebook | Epoch {epoch:03d}] WARNING: no indices — "
                      "add last_encodings to training_step in model.py")

            status = (
                "OK" if usage_pct >= 80
                else "LOW" if usage_pct >= 20
                else "*** COLLAPSE ***"
            )
            print(
                f"[Codebook | Epoch {epoch:03d}]  "
                f"Perplexity: {mean_perplex:6.1f} / {self.num_embed}  "
                f"Usage: {usage_pct:5.1f}%  [{status}]"
            )

            if trainer.logger:
                trainer.logger.experiment.add_scalar(
                    "codebook/perplexity", mean_perplex, epoch)
                trainer.logger.experiment.add_scalar(
                    "codebook/usage_pct",  usage_pct,    epoch)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("PHASE 1: VQ-VAE BINARY RHYTHM TRAINING (27 DRUMS) — Modal / A10G")
    print("=" * 70)

    pl.seed_everything(seed)

    print(f"\n[1/4] Loading {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    print(f"  Raw shape : {data.shape}  dtype: {data.dtype}")
    data = (data > 0).astype(np.float32)
    print(f"  Binarized — density: {data.mean()*100:.2f}% hits")

    if data.shape[1] != 27:
        data = np.transpose(data, (0, 2, 1))
    print(f"  Final shape: {data.shape}")

    num_train  = int(len(data) * 0.8)
    train_data = data[:num_train]
    val_data   = data[num_train:]
    print(f"  Train: {len(train_data)}   Val: {len(val_data)}")

    kw = dict(batch_size=batch_size, pin_memory=True, num_workers=num_workers)
    train_loader = DataLoader(DatasetSampler(train_data), shuffle=True,  **kw)
    val_loader   = DataLoader(DatasetSampler(val_data),   shuffle=False, **kw)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n[2/4] Building model...")
    num_embed = 256
    model     = VQVAE(ch=128, num_pitch=27, latent_dim=16, num_embed=num_embed, thres=0.5)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    ckpt_dir = "/data/checkpoints/phase1"
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        monitor="val_loss",
        dirpath=ckpt_dir,
        filename="VQVAE_phase1-{epoch:03d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )
    swa_cb = StochasticWeightAveraging(
        swa_epoch_start=0.7,
        swa_lrs=5e-5,
        annealing_epochs=min(20, max(1, int(epochs * 0.1))),
    )
    codebook_cb = CodebookMonitor(num_embed=num_embed)

    logger = TensorBoardLogger(
        save_dir="/data/logs", name="phase1",
        version=f"binary_seed{seed}",
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    print(f"\n[3/4] Training {epochs} epochs, batch_size={batch_size}, A10G GPU...")
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[ckpt_cb, swa_cb, codebook_cb],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        deterministic=True,
    )

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")

    trainer.fit(model, train_loader, val_loader)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Best checkpoint : {ckpt_cb.best_model_path}")
    print(f"Best val loss   : {ckpt_cb.best_model_score:.4f}")
    print(f"\nDownload checkpoints:")
    print(f"  modal volume get vqvae-data checkpoints/phase1/ ./checkpoints/")

    volume.commit()


@app.local_entrypoint()
def main(
    epochs: int = 30,
    batch_size: int = 256,
    seed: int = 0,
):
    print(f"Launching: epochs={epochs}, batch_size={batch_size}, seed={seed}")
    train.remote(epochs=epochs, batch_size=batch_size, seed=seed)
