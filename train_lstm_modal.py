"""
Modal wrapper for LSTM prior training over VQ-VAE codebook sequences.

The LSTM learns: given previous codebook indices, predict the next one.
At inference: audio features → starting code → LSTM generates full sequence
→ VQ-VAE decoder → novel EDM drum pattern.

Setup:
    export MODAL_PROFILE=personal
    python extract_codes.py  # run locally first
    modal volume put vqvae-data codebook_sequences.pkl /data/data/codebook_sequences.pkl

Run:
    modal run train_lstm_modal.py
    modal run train_lstm_modal.py --epochs 200
"""

import modal

app = modal.App("vqvae-lstm-prior")

volume = modal.Volume.from_name("vqvae-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2",
        "torch==2.2.0",
        "pytorch-lightning==2.2.0",
        "tensorboard",
        "tqdm",
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
    epochs: int = 200,
    batch_size: int = 512,
    embed_size: int = 128,
    hidden_size: int = 512,
    num_layers: int = 2,
    seed: int = 0,
    sequences_path: str = "/data/codebook_sequences.pkl",
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
    from torch.utils.data import DataLoader, Dataset

    from utils.lstm_layers import LSTM_Decoder

    # ── Dataset ───────────────────────────────────────────────────────────────
    class CodeSequenceDataset(Dataset):
        """
        Each item: (input_seq, target_seq)
        input  = codes[:-1]  (length T-1)
        target = codes[1:]   (length T-1, shifted by 1)
        """
        def __init__(self, sequences):
            self.sequences = torch.tensor(sequences, dtype=torch.long)

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            seq    = self.sequences[idx]
            return seq[:-1], seq[1:]  # input, target

    # ── Lightning module ──────────────────────────────────────────────────────
    class LSTMPrior(pl.LightningModule):
        def __init__(self, vocab_size=256, embed_size=128,
                     hidden_size=512, num_layers=2):
            super().__init__()
            self.save_hyperparameters()
            self.decoder = LSTM_Decoder(
                embed_size=embed_size,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
            )
            self.criterion = nn.CrossEntropyLoss()

        def forward(self, x):
            return self.decoder(x)

        def training_step(self, batch, batch_idx):
            inputs, targets = batch               # (B, T-1)
            logits  = self.decoder(inputs)        # (B*(T-1), vocab) or (B, T-1, vocab)

            # handle shape from LSTM_Decoder
            if logits.dim() == 2:
                # (B*(T-1), vocab) — already flattened by squeeze in forward
                targets_flat = targets.reshape(-1)
            else:
                logits       = logits.reshape(-1, logits.shape[-1])
                targets_flat = targets.reshape(-1)

            loss     = self.criterion(logits, targets_flat)
            # accuracy
            preds    = logits.argmax(dim=-1)
            acc      = (preds == targets_flat).float().mean()

            self.log("train_loss", loss, prog_bar=True)
            self.log("train_acc",  acc,  prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            inputs, targets = batch
            logits  = self.decoder(inputs)

            if logits.dim() == 2:
                targets_flat = targets.reshape(-1)
            else:
                logits       = logits.reshape(-1, logits.shape[-1])
                targets_flat = targets.reshape(-1)

            loss  = self.criterion(logits, targets_flat)
            preds = logits.argmax(dim=-1)
            acc   = (preds == targets_flat).float().mean()

            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc",  acc,  prog_bar=True)
            return loss

        def configure_optimizers(self):
            opt = optim.AdamW(self.parameters(), lr=1e-3)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=1e-5)
            return [opt], [sch]

    # ── Data ─────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("LSTM PRIOR TRAINING — Modal / A10G")
    print("=" * 70)

    pl.seed_everything(seed)

    print(f"\n[1/3] Loading sequences from {sequences_path}...")
    with open(sequences_path, "rb") as f:
        data = pickle.load(f)

    sequences = data["sequences"]   # (N, T)
    vocab_size = data["vocab_size"]
    seq_len    = data["seq_len"]

    print(f"  Sequences: {sequences.shape}  vocab={vocab_size}  seq_len={seq_len}")

    num_train  = int(len(sequences) * 0.8)
    train_seqs = sequences[:num_train]
    val_seqs   = sequences[num_train:]
    print(f"  Train: {len(train_seqs)}   Val: {len(val_seqs)}")

    kw = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(CodeSequenceDataset(train_seqs), shuffle=True,  **kw)
    val_loader   = DataLoader(CodeSequenceDataset(val_seqs),   shuffle=False, **kw)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n[2/3] Building LSTM prior...")
    model = LSTMPrior(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"  embed={embed_size}  hidden={hidden_size}  layers={num_layers}")
    print(f"  Params: {total:,}")

    # ── Trainer ───────────────────────────────────────────────────────────────
    ckpt_dir = "/data/checkpoints/lstm_prior"
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        monitor="val_loss",
        dirpath=ckpt_dir,
        filename="lstm_prior-{epoch:03d}-{val_loss:.4f}-{val_acc:.4f}",
        save_top_k=3,
        mode="min",
    )
    logger = TensorBoardLogger(
        save_dir="/data/logs", name="lstm_prior",
        version=f"seed{seed}",
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
    print("LSTM PRIOR TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best checkpoint : {ckpt_cb.best_model_path}")
    print(f"Best val loss   : {ckpt_cb.best_model_score:.4f}")
    print(f"\nDownload with:")
    print(f"  modal volume get vqvae-data checkpoints/lstm_prior/ ./checkpoints/lstm_prior/")

    volume.commit()


@app.local_entrypoint()
def main(
    epochs: int = 200,
    batch_size: int = 512,
    seed: int = 0,
):
    print(f"Launching LSTM prior: epochs={epochs}, batch_size={batch_size}")
    train.remote(epochs=epochs, batch_size=batch_size, seed=seed)
