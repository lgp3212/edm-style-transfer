import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
import torch
from collections import defaultdict

sys.path.insert(0, ".")
from utils.model import VQVAE
from train_phase2 import Phase2VelocityModel

PHASE1_CKPT  = "phase1/VQVAE_phase1-epoch=091-val_loss=0.0064.ckpt"
PHASE2_CKPT  = "phase2/velocity-epoch=021-val_loss=0.0181-val_mae=0.0847.ckpt"
DATA_BINARY  = "edm_hse_27drums_full.pkl"
DATA_VEL     = "edm_hse_27drums_continuous.pkl"
SIGS_PATH    = "codebook_signatures_audio.pkl"
ESC50_PKL    = "esc50_results.pkl"
OUT_DIR      = "poster_assets"

os.makedirs(OUT_DIR, exist_ok=True)
metrics = {}


print("="*60)
print("Loading models...")
print("="*60)

phase1 = VQVAE.load_from_checkpoint(
    PHASE1_CKPT, ch=128, num_pitch=27, latent_dim=16, num_embed=256, thres=0.5,
    map_location="cpu", weights_only=False)
phase1.eval()

phase2 = Phase2VelocityModel(phase1_ckpt=PHASE1_CKPT)
ckpt   = torch.load(PHASE2_CKPT, map_location="cpu", weights_only=False)
phase2.load_state_dict(ckpt["state_dict"])
phase2.eval()
print("  Models loaded.")


print("\n" + "="*60)
print("Computing Phase 1 reconstruction metrics...")
print("="*60)

with open(DATA_BINARY, "rb") as f:
    raw = pickle.load(f)
data = (raw > 0).astype(np.float32)
if data.shape[1] != 27:
    data = np.transpose(data, (0, 2, 1))

val_data = data[int(len(data) * 0.8):]
print(f"  Val samples: {len(val_data)}")

hammings = []
with torch.no_grad():
    for i in range(0, min(500, len(val_data)), 32):
        batch = torch.tensor(val_data[i:i+32])
        label, _ = phase1(batch)
        recon = label.cpu().numpy()
        orig  = val_data[i:i+32]
        hd = float(np.abs(orig - recon).mean())
        hammings.append(hd)

mean_hamming = float(np.mean(hammings))
metrics["phase1_hamming_mean"] = mean_hamming
metrics["phase1_hamming_std"]  = float(np.std(hammings))

# codebook stats
cb_vectors = phase1.quantize.embed.T.detach().numpy()  # (256, 16)
metrics["codebook_size"] = 256

# compute perplexity over val set
all_perplex = []
with torch.no_grad():
    for i in range(0, min(500, len(val_data)), 32):
        batch = torch.tensor(val_data[i:i+32])
        z     = phase1.encoder(batch)
        _, _, perplex = phase1.quantize(z)
        all_perplex.append(perplex.item())

metrics["codebook_perplexity"] = float(np.mean(all_perplex))
metrics["codebook_utilization"] = 100.0  # from training logs

print(f"  Mean Hamming distance : {mean_hamming:.4f} ± {np.std(hammings):.4f}")
print(f"  Codebook perplexity   : {metrics['codebook_perplexity']:.1f} / 256")


print("\n" + "="*60)
print("Computing Phase 2 velocity metrics...")
print("="*60)

with open(DATA_VEL, "rb") as f:
    raw_vel = pickle.load(f).astype(np.float32)
if raw_vel.shape[1] != 27:
    raw_vel = np.transpose(raw_vel, (0, 2, 1))

val_vel = raw_vel[int(len(raw_vel) * 0.8):]

# training data velocity distribution
train_hits = raw_vel[:int(len(raw_vel) * 0.8)]
train_hits_flat = train_hits[train_hits > 0.01].flatten()

# predicted velocity distribution
pred_hits_flat = []
mae_vals = []

with torch.no_grad():
    for i in range(0, min(300, len(val_vel)), 32):
        batch_vel = torch.tensor(val_vel[i:i+32])
        # get binary rhythm from phase1
        label, _ = phase1(batch_vel)
        rhythm = label.cpu().numpy()
        # get velocities from phase2
        x   = torch.tensor(rhythm.astype(np.float32))
        vel = phase2(x).squeeze().cpu().numpy()
        if vel.ndim == 2:
            vel = vel[np.newaxis]
        velocity = rhythm * vel
        # collect hits
        for j in range(len(batch_vel)):
            target    = val_vel[i+j]
            rhythm_j  = rhythm[j] if j < len(rhythm) else rhythm[0]
            pred_vel  = vel[j] if j < len(vel) else vel[0]
            
            hit_mask_pred   = rhythm_j > 0.5      # predicted hits
            hit_mask_target = target > 0.01       # ground truth hits
            
            if hit_mask_pred.sum() > 0:
                pred_hits_flat.extend(pred_vel[hit_mask_pred].tolist())
            if hit_mask_target.sum() > 0 and hit_mask_pred.sum() > 0:
                # MAE only where both agree there's a hit
                common = hit_mask_pred & hit_mask_target
                if common.sum() > 0:
                    mae_vals.append(np.abs(pred_vel[common] - target[common]).mean())

pred_hits_flat = np.array(pred_hits_flat)
metrics["phase2_mae"]      = float(np.mean(mae_vals))
metrics["phase2_mae_std"]  = float(np.std(mae_vals))
metrics["phase2_val_loss"] = 0.0181  # from training logs

print(f"  Velocity MAE : {metrics['phase2_mae']:.4f} ± {metrics['phase2_mae_std']:.4f}")
print(f"  Val loss     : {metrics['phase2_val_loss']:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Velocity Distribution: Training Data vs Phase 2 Predictions",
             fontsize=13, fontweight='bold')

bins = np.linspace(0, 1, 25)

axes[0].hist(train_hits_flat, bins=bins, color="#3498db", alpha=0.85, edgecolor='white')
axes[0].set_title("Training data\n(continuous velocities)", fontsize=11)
axes[0].set_xlabel("Velocity (0–1)")
axes[0].set_ylabel("Count")
axes[0].axvline(train_hits_flat.mean(), color='black', linestyle='--',
                alpha=0.7, label=f"mean={train_hits_flat.mean():.2f}")
axes[0].legend(fontsize=9)

axes[1].hist(pred_hits_flat, bins=bins, color="#e74c3c", alpha=0.85, edgecolor='white')
axes[1].set_title("Phase 2 predictions\n(velocity decoder output)", fontsize=11)
axes[1].set_xlabel("Velocity (0–1)")
axes[1].set_ylabel("Count")
axes[1].axvline(pred_hits_flat.mean(), color='black', linestyle='--',
                alpha=0.7, label=f"mean={pred_hits_flat.mean():.2f}")
axes[1].legend(fontsize=9)

plt.tight_layout()
vel_fig_path = os.path.join(OUT_DIR, "fig_velocity_distribution.png")
plt.savefig(vel_fig_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {vel_fig_path}")


print("\n" + "="*60)
print("Computing audio conditioning metrics...")
print("="*60)

ESC50_CATEGORIES = {
    0:'dog',1:'rooster',2:'pig',3:'cow',4:'frog',5:'cat',6:'hen',
    7:'insects',8:'sheep',9:'crow',10:'rain',11:'sea_waves',
    12:'crackling_fire',13:'crickets',14:'chirping_birds',15:'water_drops',
    16:'wind',17:'pouring_water',18:'toilet_flush',19:'thunderstorm',
    20:'crying_baby',21:'sneezing',22:'clapping',23:'breathing',24:'coughing',
    25:'footsteps',26:'laughing',27:'brushing_teeth',28:'snoring',29:'drinking',
    30:'door_knock',31:'mouse_click',32:'keyboard_typing',33:'door_creak',
    34:'can_opening',35:'washing_machine',36:'vacuum_cleaner',37:'clock_alarm',
    38:'clock_tick',39:'glass_breaking',40:'helicopter',41:'chainsaw',
    42:'siren',43:'car_horn',44:'engine',45:'train',46:'church_bells',
    47:'airplane',48:'fireworks',49:'hand_saw'
}

SUPER_CATEGORIES = {
    'Animals':  ['dog','rooster','pig','cow','frog','cat','hen','insects','sheep','crow'],
    'Nature':   ['rain','sea_waves','crackling_fire','crickets','chirping_birds',
                 'water_drops','wind','pouring_water','toilet_flush','thunderstorm'],
    'Human':    ['crying_baby','sneezing','clapping','breathing','coughing',
                 'footsteps','laughing','brushing_teeth','snoring','drinking'],
    'Domestic': ['door_knock','mouse_click','keyboard_typing','door_creak',
                 'can_opening','washing_machine','vacuum_cleaner','clock_alarm',
                 'clock_tick','glass_breaking'],
    'Urban':    ['helicopter','chainsaw','siren','car_horn','engine',
                 'train','church_bells','airplane','fireworks','hand_saw'],
}
CAT_TO_SUPER = {c: s for s, cats in SUPER_CATEGORIES.items() for c in cats}

if os.path.exists(ESC50_PKL):
    with open(ESC50_PKL, "rb") as f:
        esc50_results = [r for r in pickle.load(f) if r['code'] >= 0]

    unique_codes = set(r['code'] for r in esc50_results)
    metrics["esc50_total_files"]   = len(esc50_results)
    metrics["esc50_unique_codes"]  = len(unique_codes)
    metrics["esc50_coverage_pct"]  = len(unique_codes) / 256 * 100

    super_unique = defaultdict(set)
    for r in esc50_results:
        sup = CAT_TO_SUPER.get(r['category'], 'Unknown')
        super_unique[sup].add(r['code'])

    for s, codes in super_unique.items():
        metrics[f"esc50_{s.lower()}_unique_codes"] = len(codes)

    print(f"  ESC-50 files        : {len(esc50_results)}")
    print(f"  Unique codes        : {len(unique_codes)} / 256 ({metrics['esc50_coverage_pct']:.0f}%)")
    for s in SUPER_CATEGORIES:
        print(f"  {s:12s} unique codes: {len(super_unique[s])}")
else:
    print("  esc50_results.pkl not found — skipping ESC-50 metrics")


print("\n" + "="*60)
print("Generating input→output example table figure...")
print("="*60)

# Cluster dominant drum names from earlier analysis
CLUSTER_CHARS = {
    0: "Kick + HH (dense)",
    1: "Snare heavy",
    2: "Maracas / textural",
    3: "HH closed (max)",
    4: "Clap driven",
    5: "Sparse / open HH",
    6: "HH closed (mid)",
    7: "Kick + HH closed",
}

# Example inputs with their observed codes and super-categories
examples = [
    ("Dog bark",       "Animals",  117, "Kick + HH"),
    ("Rain",           "Nature",   205, "Sparse / textural"),
    ("Crackling fire", "Nature",   205, "Sparse / textural"),
    ("Siren",          "Urban",    232, "HH closed (mid)"),
    ("Helicopter",     "Urban",    202, "Kick + HH"),
    ("Keyboard",       "Domestic", 48,  "Clap driven"),
    ("Footsteps",      "Human",    250, "Sparse"),
    ("Church bells",   "Urban",    25,  "Kick + HH"),
    ("Glass breaking", "Domestic", 110, "Snare / impact"),
    ("Chirping birds", "Nature",   244, "HH dense"),
]

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.axis('off')
fig.suptitle("Audio Conditioning: Input Sound → Codebook Archetype",
             fontsize=13, fontweight='bold', y=1.0)

col_labels = ["Input sound", "Category", "Codebook\nentry", "Rhythm archetype"]
col_widths  = [0.28, 0.18, 0.14, 0.36]

SUPER_COLORS_MPL = {
    'Animals':  '#fadbd8',
    'Nature':   '#d5f5e3',
    'Human':    '#d6eaf8',
    'Domestic': '#fdebd0',
    'Urban':    '#e8daef',
}

table_data = [[e[0], e[1], str(e[2]), e[3]] for e in examples]

table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
)

table.auto_set_font_size(False)
table.set_fontsize(11)

# style header
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#2c3e50')
    table[0, j].set_text_props(color='white', fontweight='bold')
    table[0, j].set_height(0.12)

# style rows
for i, (_, cat, _, _) in enumerate(examples):
    row_color = SUPER_COLORS_MPL.get(cat, '#ffffff')
    for j in range(len(col_labels)):
        table[i+1, j].set_facecolor(row_color)
        table[i+1, j].set_height(0.09)

# legend
legend_patches = [mpatches.Patch(color=SUPER_COLORS_MPL[s], label=s)
                  for s in SUPER_CATEGORIES]
ax.legend(handles=legend_patches, loc='lower center',
          bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=9,
          frameon=False)

plt.tight_layout()
table_fig_path = os.path.join(OUT_DIR, "fig_input_output_table.png")
plt.savefig(table_fig_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {table_fig_path}")


print("\n" + "="*60)
print("Generating pipeline diagram...")
print("="*60)

fig, ax = plt.subplots(figsize=(14, 3.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 3.5)
ax.axis('off')

boxes = [
    (0.3,  "Input\naudio",          "#d6eaf8", "#2980b9"),
    (2.3,  "Onset\ndetection",      "#eaf3de", "#27ae60"),
    (4.3,  "Spectral\nfeatures",    "#eaf3de", "#27ae60"),
    (6.3,  "Codebook\nretrieval",   "#fdebd0", "#e67e22"),
    (8.3,  "VQ-VAE\ndecoder",       "#fdebd0", "#e67e22"),
    (10.3, "Velocity\ndecoder",     "#fadbd8", "#e74c3c"),
    (12.3, "EDM\noutput",           "#d6eaf8", "#2980b9"),
]

BOX_W, BOX_H, BOX_Y = 1.6, 1.4, 1.0

for x, label, facecolor, edgecolor in boxes:
    rect = mpatches.FancyBboxPatch(
        (x, BOX_Y), BOX_W, BOX_H,
        boxstyle="round,pad=0.1",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + BOX_W/2, BOX_Y + BOX_H/2, label,
            ha='center', va='center', fontsize=10, fontweight='500',
            color='#2c3e50')

# arrows
arrow_props = dict(arrowstyle='->', color='#555555', lw=1.5)
for i in range(len(boxes) - 1):
    x_start = boxes[i][0] + BOX_W + 0.05
    x_end   = boxes[i+1][0] - 0.05
    y_mid   = BOX_Y + BOX_H / 2
    ax.annotate('', xy=(x_end, y_mid), xytext=(x_start, y_mid),
                arrowprops=arrow_props)

# labels below boxes
sublabels = [
    "Any sound",
    "librosa",
    "centroid\nRMS rolloff ZCR",
    "nearest-\nneighbor",
    "Phase 1\nfrozen",
    "Phase 2\nfrozen enc.",
    "WAV file",
]
for (x, _, _, _), sub in zip(boxes, sublabels):
    ax.text(x + BOX_W/2, BOX_Y - 0.25, sub,
            ha='center', va='top', fontsize=8,
            color='#7f8c8d', style='italic')

# trained badge
for i, (x, _, _, _) in enumerate(boxes):
    if i in [3, 4, 5]:
        ax.text(x + BOX_W/2, BOX_Y + BOX_H + 0.15, "trained",
                ha='center', va='bottom', fontsize=8,
                color='#e67e22', fontweight='500')

plt.tight_layout()
pipeline_fig_path = os.path.join(OUT_DIR, "fig_pipeline.png")
plt.savefig(pipeline_fig_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {pipeline_fig_path}")


#save 
print("\n" + "="*60)
print("Saving metrics summary...")
print("="*60)

metrics_path = os.path.join(OUT_DIR, "poster_metrics.txt")
with open(metrics_path, "w") as f:
    f.write("POSTER METRICS — DAFx26\n")
    f.write("="*50 + "\n\n")

    f.write("PHASE 1 — VQ-VAE RECONSTRUCTION\n")
    f.write(f"  Hamming distance (mean ± std) : {metrics['phase1_hamming_mean']:.4f} ± {metrics['phase1_hamming_std']:.4f}\n")
    f.write(f"  Codebook utilization          : {metrics['codebook_utilization']:.0f}%\n")
    f.write(f"  Codebook perplexity           : {metrics['codebook_perplexity']:.1f} / 256\n")
    f.write(f"  Codebook size                 : {metrics['codebook_size']}\n\n")

    f.write("PHASE 2 — VELOCITY DECODER\n")
    f.write(f"  MAE (mean ± std)              : {metrics['phase2_mae']:.4f} ± {metrics['phase2_mae_std']:.4f}\n")
    f.write(f"  Val loss                      : {metrics['phase2_val_loss']:.4f}\n\n")

    f.write("AUDIO CONDITIONING — ESC-50\n")
    if "esc50_total_files" in metrics:
        f.write(f"  Total files tested            : {metrics['esc50_total_files']}\n")
        f.write(f"  Unique codes activated        : {metrics['esc50_unique_codes']} / 256\n")
        f.write(f"  Codebook coverage             : {metrics['esc50_coverage_pct']:.0f}%\n")
        for s in SUPER_CATEGORIES:
            key = f"esc50_{s.lower()}_unique_codes"
            if key in metrics:
                f.write(f"  {s:12s} unique codes      : {metrics[key]}\n")

    f.write("\n" + "="*50 + "\n")
    f.write("FIGURES GENERATED\n")
    f.write(f"  fig_velocity_distribution.png\n")
    f.write(f"  fig_input_output_table.png\n")
    f.write(f"  fig_pipeline.png\n")
    f.write(f"  (use existing codebook_analysis.png + esc50_codebook_activation.png)\n")

print(f"  Saved: {metrics_path}")

# print summary to terminal
print("\n" + "="*60)
print("ALL METRICS SUMMARY")
print("="*60)
with open(metrics_path) as f:
    print(f.read())

print(f"\nAll assets saved to: {OUT_DIR}/")
print("Files:")
for fname in sorted(os.listdir(OUT_DIR)):
    print(f"  {fname}")
