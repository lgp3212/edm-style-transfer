#!/usr/bin/env python3
"""
Batch diversity test for EDM codebook audio conditioning.
Downloads samples from ESC-50 and GTZAN, runs audio conditioning,
and reports which codebook entries get triggered.

Usage:
    python batch_diversity_test.py
    python batch_diversity_test.py --n-per-class 3
"""

import argparse
import os
import pickle
import sys
import soundfile as sf
import numpy as np
from collections import defaultdict

sys.path.insert(0, ".")
from generate import get_start_code_from_audio


def download_esc50_samples(output_dir, n_per_class=2):
    """Download a sample of ESC-50 environmental sounds."""
    from datasets import load_dataset
    import soundfile as sf

    print("Downloading ESC-50 samples...")
    os.makedirs(output_dir, exist_ok=True)

    ds = load_dataset("ashraq/esc50", split="train")

    # pick a diverse set of categories
    target_categories = [
        "dog", "rain", "crying_baby", "clock_tick", "chainsaw",
        "crackling_fire", "helicopter", "sea_waves", "piano", "guitar"
    ]

    saved = []
    category_counts = defaultdict(int)

    for item in ds:
        cat = item["category"]
        if cat in target_categories and category_counts[cat] < n_per_class:
            path = os.path.join(output_dir, f"esc50_{cat}_{category_counts[cat]}.wav")
            audio = np.array(item["audio"]["array"])
            sr    = item["audio"]["sampling_rate"]
            sf.write(path, audio, sr)
            saved.append({"path": path, "category": cat, "dataset": "ESC-50"})
            category_counts[cat] += 1

    print(f"  Saved {len(saved)} ESC-50 files")
    return saved


def download_gtzan_samples(output_dir, n_per_genre=2):
    """Download a sample of GTZAN music genre files."""
    from datasets import load_dataset
    import soundfile as sf

    print("Downloading GTZAN samples...")
    os.makedirs(output_dir, exist_ok=True)

    ds = load_dataset("marsyas/gtzan", split="train", trust_remote_code=True)

    genres = ["blues", "classical", "country", "disco", "hiphop",
              "jazz", "metal", "pop", "reggae", "rock"]

    saved = []
    genre_counts = defaultdict(int)

    for item in ds:
        genre = item["genre"]
        if genre in genres and genre_counts[genre] < n_per_genre:
            path = os.path.join(output_dir, f"gtzan_{genre}_{genre_counts[genre]}.wav")
            audio = np.array(item["audio"]["array"])
            sr    = item["audio"]["sampling_rate"]
            sf.write(path, audio, sr)
            saved.append({"path": path, "category": genre, "dataset": "GTZAN"})
            genre_counts[genre] += 1
        if sum(genre_counts.values()) >= len(genres) * n_per_genre:
            break

    print(f"  Saved {len(saved)} GTZAN files")
    return saved


def run_batch_test(files, signatures_path, output_path="batch_results.txt"):
    """Run audio conditioning on all files and report codebook diversity."""

    print(f"\nRunning batch test on {len(files)} files...")
    print(f"Signatures: {signatures_path}\n")

    results = []
    code_to_files = defaultdict(list)

    for item in files:
        path     = item["path"]
        category = item["category"]
        dataset  = item["dataset"]

        try:
            code = get_start_code_from_audio(path, signatures_path)
            if code is None:
                code = -1
        except Exception as e:
            print(f"  ERROR on {path}: {e}")
            code = -1

        results.append({
            "file":     os.path.basename(path),
            "category": category,
            "dataset":  dataset,
            "code":     code,
        })
        code_to_files[code].append(f"{dataset}/{category}")
        print(f"  {dataset:6s} | {category:20s} → code {code:3d}")

    # ── Summary ───────────────────────────────────────────────────────────────
    codes_used   = set(r["code"] for r in results if r["code"] >= 0)
    total_files  = len([r for r in results if r["code"] >= 0])
    diversity    = len(codes_used) / total_files if total_files > 0 else 0

    print(f"\n{'='*60}")
    print(f"BATCH TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Total files tested : {total_files}")
    print(f"  Unique codes used  : {len(codes_used)} / 256")
    print(f"  Diversity ratio    : {diversity:.2f}  (1.0 = all different)")
    print(f"\n  Code distribution:")

    # group by code
    for code in sorted(codes_used):
        files_for_code = code_to_files[code]
        print(f"    Code {code:3d}: {', '.join(files_for_code)}")

    # per-dataset breakdown
    for dataset in ["ESC-50", "GTZAN"]:
        ds_results = [r for r in results if r["dataset"] == dataset and r["code"] >= 0]
        if ds_results:
            ds_codes = set(r["code"] for r in ds_results)
            print(f"\n  {dataset}: {len(ds_codes)} unique codes / {len(ds_results)} files "
                  f"({len(ds_codes)/len(ds_results):.2f} diversity)")

    # save results
    with open(output_path, "w") as f:
        f.write("file,category,dataset,code\n")
        for r in results:
            f.write(f"{r['file']},{r['category']},{r['dataset']},{r['code']}\n")
    print(f"\n  Results saved to: {output_path}")

    return results, codes_used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signatures",   type=str,
                        default="codebook_signatures_audio.pkl")
    parser.add_argument("--n-per-class",  type=int, default=2,
                        help="Samples per category/genre")
    parser.add_argument("--audio-dir",    type=str, default="batch_test_audio",
                        help="Directory to save downloaded audio")
    parser.add_argument("--output",       type=str, default="batch_results.csv",
                        help="CSV output file")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download if audio already exists")
    args = parser.parse_args()

    all_files = []

    if not args.skip_download:
        esc50_files = download_esc50_samples(
            os.path.join(args.audio_dir, "esc50"),
            n_per_class=args.n_per_class
        )
        gtzan_files = download_gtzan_samples(
            os.path.join(args.audio_dir, "gtzan"),
            n_per_class=args.n_per_class
        )
        all_files = esc50_files + gtzan_files
    else:
        # load from existing directory
        for dataset, subdir in [("ESC-50", "esc50"), ("GTZAN", "gtzan")]:
            d = os.path.join(args.audio_dir, subdir)
            if os.path.exists(d):
                for fname in sorted(os.listdir(d)):
                    if fname.endswith(".wav"):
                        parts   = fname.replace(".wav", "").split("_")
                        category = "_".join(parts[1:-1])
                        all_files.append({
                            "path":     os.path.join(d, fname),
                            "category": category,
                            "dataset":  dataset,
                        })

    if not all_files:
        print("No files found. Run without --skip-download first.")
        return

    results, codes_used = run_batch_test(
        all_files,
        signatures_path=args.signatures,
        output_path=args.output,
    )

    print(f"\n{'='*60}")
    print(f"INTERPRETATION FOR PAPER")
    print(f"{'='*60}")
    n = len([r for r in results if r["code"] >= 0])
    u = len(codes_used)
    if u / n >= 0.7:
        print(f"  STRONG: {u}/{n} files mapped to different codes ({u/n:.0%} diversity)")
        print(f"  → Audio conditioning is discriminating well across sound types")
    elif u / n >= 0.4:
        print(f"  MODERATE: {u}/{n} files mapped to different codes ({u/n:.0%} diversity)")
        print(f"  → Some discrimination but clusters exist — expected for similar genres")
    else:
        print(f"  WEAK: {u}/{n} files mapped to different codes ({u/n:.0%} diversity)")
        print(f"  → Audio conditioning may not be discriminating enough")


if __name__ == "__main__":
    main()
