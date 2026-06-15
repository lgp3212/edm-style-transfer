import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import umap

sys.path.insert(0, ".")
from utils.model import VQVAE

PHASE1_CKPT = "phase1/VQVAE_phase1-epoch=094-val_loss=0.0064.ckpt"
OUT_DIR     = "poster_assets"

import os
os.makedirs(OUT_DIR, exist_ok=True)

# load mdoel
print("Loading model...")
model = VQVAE.load_from_checkpoint(
    PHASE1_CKPT, ch=128, num_pitch=27, latent_dim=16, num_embed=256, thres=0.5,
    map_location="cpu", weights_only=False)
model.eval()

cb = model.quantize.embed.T.detach().numpy()  # (256, 16)

with open("edm_hse_27drums_full.pkl", "rb") as f:
    raw = pickle.load(f)
data = (raw > 0).astype("float32")
if data.shape[1] != 27:
    data = np.transpose(data, (0, 2, 1))

import torch
usage = np.zeros(256, dtype=np.int64)
with torch.no_grad():
    for i in range(0, len(data), 512):
        batch  = torch.tensor(data[i:i+512])
        z      = model.encoder(batch)
        x      = z.transpose(1, 2)
        x_flat = x.detach().reshape(-1, model.quantize.latent_dim)
        dist   = (x_flat.pow(2).sum(1, keepdim=True)
                  - 2 * x_flat @ model.quantize.embed
                  + model.quantize.embed.pow(2).sum(0, keepdim=True))
        idx    = torch.argmin(dist, 1)
        seq    = idx.reshape(batch.shape[0], -1)
        dom    = torch.mode(seq, dim=1).values
        for code in dom.numpy():
            usage[code] += 1

# cluster 
scaler   = StandardScaler()
cb_scaled = scaler.fit_transform(cb)
linkage_mat   = linkage(pdist(cb_scaled), method='ward')
cluster_labels = fcluster(linkage_mat, t=8, criterion='maxclust') - 1

# cluster names based on dominant drum
CLUSTER_NAMES = {
    0: "Kick + HH",
    1: "Snare heavy",
    2: "Textural",
    3: "HH closed",
    4: "Clap driven",
    5: "Sparse",
    6: "HH mid",
    7: "Kick groove",
}

CLUSTER_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e"
]

#umap
print("Running UMAP...")
reducer  = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
cb_umap  = reducer.fit_transform(cb_scaled)

#plot
fig, ax = plt.subplots(figsize=(8, 6))

for c in range(8):
    mask = cluster_labels == c
    sizes = 20 + usage[mask] / usage.max() * 120
    ax.scatter(cb_umap[mask, 0], cb_umap[mask, 1],
               c=CLUSTER_COLORS[c], s=sizes, alpha=0.85,
               edgecolors='white', linewidths=0.3,
               label=f"C{c}: {CLUSTER_NAMES[c]} (n={mask.sum()})")

ax.set_xlabel("UMAP-1", fontsize=12)
ax.set_ylabel("UMAP-2", fontsize=12)
ax.set_title("VQ-VAE Codebook — 256 Rhythm Archetypes\n(dot size = usage frequency)",
             fontsize=13, fontweight='bold')

legend = ax.legend(fontsize=9, loc='upper left',
                   framealpha=0.9, edgecolor='#cccccc',
                   title="Cluster (dominant character)", title_fontsize=9)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "fig_umap_poster.png")
plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out_path}")
