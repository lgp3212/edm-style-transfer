#!/usr/bin/env python3
"""
Comprehensive VQ-VAE codebook analysis.
Produces a multi-panel figure for the DAFx paper.

Panels:
  1. UMAP of codebook vectors (colored by cluster)
  2. PCA of codebook vectors (colored by cluster)
  3. Cluster dendrogram
  4. Per-cluster drum density heatmap
  5. Spectral signature per cluster
  6. Codebook usage histogram

Usage:
    python analyze_codebook.py
    python analyze_codebook.py --n-clusters 8 --out codebook_analysis.png
"""

import argparse
import pickle
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import torch
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

sys.path.insert(0, ".")
from utils.model import VQVAE

DRUM_NAMES = [
    "SFX", "Stick", "Kick", "Rimshot", "Snare", "Clap", "Snare Sm",
    "Low Tom", "HH Cl", "High Tom", "HH Acc", "Low Tom2", "HH Op",
    "Mid Tom", "High Tom2", "Tamb", "Rev Rise", "Conga M",
    "Conga H", "Conga L", "Cabasa", "Maracas", "Claves",
    "WBlock H", "WBlock L", "Cowbell", "Splash",
]


def load_model_and_data(phase1_ckpt, data_path):
    print("Loading model...")
    model = VQVAE.load_from_checkpoint(
        phase1_ckpt,
        ch=128, num_pitch=27, latent_dim=16, num_embed=256, thres=0.5,
        map_location="cpu", weights_only=False,
    )
    model.eval()

    print("Loading data...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    data = (data > 0).astype(np.float32)
    if data.shape[1] != 27:
        data = np.transpose(data, (0, 2, 1))

    return model, data


def get_codebook_vectors(model):
    """Extract raw codebook embedding vectors. Shape: (256, latent_dim)"""
    return model.quantize.embed.T.detach().numpy()  # (num_embed, latent_dim)


def get_usage_and_patterns(model, data, batch_size=512):
    """
    Run all val data through the encoder and collect:
      - usage counts per codebook entry
      - average drum pattern per codebook entry
    """
    num_embed = model.quantize.num_embed
    usage     = np.zeros(num_embed, dtype=np.int64)
    patterns  = np.zeros((num_embed, 27, 64), dtype=np.float64)  # sum
    counts    = np.zeros(num_embed, dtype=np.int64)

    val_data = data[int(len(data) * 0.8):]
    print(f"Computing codebook usage on {len(val_data)} val samples...")

    with torch.no_grad():
        for i in range(0, len(val_data), batch_size):
            batch = torch.tensor(val_data[i:i+batch_size])
            z     = model.encoder(batch)

            # get indices directly from quantize
            x     = z.transpose(1, 2)
            x_flat = x.detach().reshape(-1, model.quantize.latent_dim)
            dist  = (
                x_flat.pow(2).sum(1, keepdim=True)
                - 2 * x_flat @ model.quantize.embed
                + model.quantize.embed.pow(2).sum(0, keepdim=True)
            )
            idx = torch.argmin(dist, 1)  # (batch * time,)

            # per-sample: use the most frequent index as the "pattern code"
            batch_size_actual = batch.shape[0]
            time_steps = idx.shape[0] // batch_size_actual
            idx_per_sample = idx.reshape(batch_size_actual, time_steps)
            dominant = torch.mode(idx_per_sample, dim=1).values  # (batch,)

            for j, code in enumerate(dominant.numpy()):
                usage[code]        += 1
                patterns[code]     += val_data[i + j]
                counts[code]       += 1

    # Average patterns
    avg_patterns = np.zeros_like(patterns)
    for c in range(num_embed):
        if counts[c] > 0:
            avg_patterns[c] = patterns[c] / counts[c]

    return usage, avg_patterns, counts


def compute_spectral_signatures(avg_patterns):
    """
    Compute spectral features per codebook entry from its average drum pattern.
    Features: density, top-drum concentration, rhythmic regularity, hit spread
    """
    sigs = []
    for pattern in avg_patterns:
        density      = pattern.mean()
        top_drum     = pattern.max(axis=1).mean()      # avg peak per drum
        regularity   = 0.0
        # check how regular hits are (autocorrelation at lag 16 = one bar)
        flat         = pattern.sum(axis=0)             # (64,)
        if flat.sum() > 0:
            flat_n   = flat / (flat.sum() + 1e-8)
            regularity = float(np.correlate(flat_n, flat_n, mode='full')[64])
        hit_spread   = (pattern > 0.01).sum()          # total hits

        sigs.append([density, top_drum, regularity, hit_spread])

    return np.array(sigs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-ckpt", type=str,
                        default="phase1/VQVAE_phase1-epoch=091-val_loss=0.0064.ckpt")
    parser.add_argument("--data",        type=str,
                        default="edm_hse_27drums_full.pkl")
    parser.add_argument("--n-clusters",  type=int, default=8)
    parser.add_argument("--out",         type=str, default="codebook_analysis.png")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ── Load ──────────────────────────────────────────────────────────────────
    model, data = load_model_and_data(args.phase1_ckpt, args.data)
    cb_vectors  = get_codebook_vectors(model)              # (256, 16)
    usage, avg_patterns, counts = get_usage_and_patterns(model, data)
    spectral    = compute_spectral_signatures(avg_patterns)

    print(f"Codebook vectors shape : {cb_vectors.shape}")
    print(f"Used entries           : {(usage > 0).sum()} / {len(usage)}")

    # ── Clustering ────────────────────────────────────────────────────────────
    print("Clustering codebook vectors...")
    scaler     = StandardScaler()
    cb_scaled  = scaler.fit_transform(cb_vectors)
    linkage_mat = linkage(pdist(cb_scaled), method='ward')
    cluster_labels = fcluster(linkage_mat, t=args.n_clusters, criterion='maxclust') - 1

    # ── Dimensionality reduction ───────────────────────────────────────────────
    print("Running PCA...")
    pca      = PCA(n_components=2, random_state=args.seed)
    cb_pca   = pca.fit_transform(cb_scaled)

    print("Running UMAP...")
    reducer  = umap.UMAP(n_components=2, random_state=args.seed, n_neighbors=15, min_dist=0.1)
    cb_umap  = reducer.fit_transform(cb_scaled)

    # ── Compute per-cluster avg drum density ──────────────────────────────────
    cluster_drum_density = np.zeros((args.n_clusters, 27))
    cluster_sizes        = np.zeros(args.n_clusters)
    for c in range(args.n_clusters):
        mask = cluster_labels == c
        if mask.sum() > 0:
            # average pattern across all codebook entries in this cluster
            cluster_avg = avg_patterns[mask].mean(axis=0)  # (27, 64)
            cluster_drum_density[c] = (cluster_avg > 0.01).mean(axis=1)  # per drum
            cluster_sizes[c] = mask.sum()

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("Plotting...")
    palette = sns.color_palette("tab10", args.n_clusters)
    colors  = [palette[c] for c in cluster_labels]

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle("VQ-VAE Codebook Analysis — Phase 1 (256 entries, 27 drums)",
                 fontsize=16, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: UMAP ─────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(cb_umap[:, 0], cb_umap[:, 1],
                          c=colors, s=30 + usage / usage.max() * 80,
                          alpha=0.8, edgecolors='none')
    ax1.set_title("UMAP of Codebook Vectors\n(size = usage frequency)", fontsize=11)
    ax1.set_xlabel("UMAP-1")
    ax1.set_ylabel("UMAP-2")
    for c in range(args.n_clusters):
        mask = cluster_labels == c
        cx, cy = cb_umap[mask, 0].mean(), cb_umap[mask, 1].mean()
        ax1.annotate(f"C{c}", (cx, cy), fontsize=8, fontweight='bold',
                     color=palette[c], ha='center')

    # ── Panel 2: PCA ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(cb_pca[:, 0], cb_pca[:, 1],
                c=colors, s=30 + usage / usage.max() * 80,
                alpha=0.8, edgecolors='none')
    ax2.set_title(f"PCA of Codebook Vectors\n(var explained: {pca.explained_variance_ratio_.sum()*100:.1f}%)", fontsize=11)
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    # ── Panel 3: Usage histogram ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    sorted_usage = np.sort(usage)[::-1]
    bar_colors   = [palette[cluster_labels[np.argsort(usage)[::-1][i]]]
                    for i in range(len(usage))]
    ax3.bar(range(len(sorted_usage)), sorted_usage, color=bar_colors, width=1.0)
    ax3.set_title("Codebook Entry Usage\n(sorted, colored by cluster)", fontsize=11)
    ax3.set_xlabel("Codebook entry (sorted by usage)")
    ax3.set_ylabel("Times used (val set)")
    ax3.axhline(usage.mean(), color='black', linestyle='--', alpha=0.5, label=f'mean={usage.mean():.1f}')
    ax3.legend(fontsize=8)

    # ── Panel 4: Dendrogram ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    dendrogram(linkage_mat, ax=ax4, no_labels=True, color_threshold=0,
               above_threshold_color='grey', truncate_mode='lastp', p=40)
    ax4.set_title(f"Hierarchical Clustering Dendrogram\n(truncated to last 40 merges, {args.n_clusters} clusters)", fontsize=11)
    ax4.set_xlabel("Codebook entries")
    ax4.set_ylabel("Distance")

    # ── Panel 5: Spectral signatures per cluster ──────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    cluster_spectral = np.zeros((args.n_clusters, 4))
    feat_names = ["Density", "Peak/drum", "Regularity", "Hit count"]
    for c in range(args.n_clusters):
        mask = cluster_labels == c
        if mask.sum() > 0:
            cluster_spectral[c] = spectral[mask].mean(axis=0)

    # normalize each feature to 0-1 for radar-style heatmap
    cluster_spectral_n = cluster_spectral.copy()
    for j in range(4):
        rng = cluster_spectral_n[:, j].max() - cluster_spectral_n[:, j].min()
        if rng > 0:
            cluster_spectral_n[:, j] = (cluster_spectral_n[:, j] - cluster_spectral_n[:, j].min()) / rng

    sns.heatmap(cluster_spectral_n,
                ax=ax5,
                xticklabels=feat_names,
                yticklabels=[f"C{c} (n={int(cluster_sizes[c])})" for c in range(args.n_clusters)],
                cmap="YlOrRd",
                annot=True, fmt=".2f",
                linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    ax5.set_title("Spectral Signatures per Cluster\n(normalized 0-1)", fontsize=11)
    ax5.tick_params(axis='x', rotation=30)

    # ── Panel 6: Per-cluster drum density heatmap ─────────────────────────────
    ax6 = fig.add_subplot(gs[2, :])
    # only show drums that have any activity
    active_drums = np.where(cluster_drum_density.max(axis=0) > 0.001)[0]
    heatmap_data = cluster_drum_density[:, active_drums]
    active_names = [DRUM_NAMES[i] for i in active_drums]

    sns.heatmap(heatmap_data,
                ax=ax6,
                xticklabels=active_names,
                yticklabels=[f"Cluster {c} (n={int(cluster_sizes[c])})" for c in range(args.n_clusters)],
                cmap="Blues",
                annot=True, fmt=".2f",
                linewidths=0.5,
                cbar_kws={"label": "Hit density", "shrink": 0.5})
    ax6.set_title("Average Drum Hit Density per Cluster\n(fraction of time steps with a hit)", fontsize=11)
    ax6.tick_params(axis='x', rotation=45)

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.savefig(args.out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {args.out}")

    # ── Print cluster summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"CLUSTER SUMMARY ({args.n_clusters} clusters)")
    print(f"{'='*60}")
    for c in range(args.n_clusters):
        mask     = cluster_labels == c
        top_drum = active_drums[np.argmax(cluster_drum_density[c, active_drums])] if len(active_drums) > 0 else -1
        print(f"  C{c}: {mask.sum():3d} entries  "
              f"avg_usage={usage[mask].mean():.1f}  "
              f"dominant_drum={DRUM_NAMES[top_drum] if top_drum >= 0 else 'none'}")


if __name__ == "__main__":
    main()
