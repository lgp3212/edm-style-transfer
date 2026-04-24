#!/usr/bin/env python3
"""
Process EDM-HSE with 27 Labeled Drums using JSON + Clustering
-------------------------------------------------------------

Uses JSON metadata to constrain clustering of onsets into drum types.
Creates labeled 27×64 arrays with velocity bins [0-4].

Usage:
    python preprocess_edm_hse_27drums.py --wav-dir /path/to/wavs --json-dir /path/to/json --max-files 100
"""

import numpy as np
import librosa
import pickle
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans
import argparse


# MIDI note to drum index mapping (0-26)
NOTE_TO_INDEX_27 = {
    27: 0,   # One-Shot SFX
    31: 1,   # Stick
    36: 2,   # Kick
    37: 3,   # Rimshot
    38: 4,   # Snare Big
    39: 5,   # Clap
    40: 6,   # Snare Small
    41: 7,   # Simple Low Tom
    42: 8,   # Closed Hat
    43: 9,   # Simple High Tom
    44: 10,  # Accent Hat
    45: 11,  # Low Tom
    46: 12,  # Open Hat
    48: 13,  # Mid Tom
    50: 14,  # High Tom
    54: 15,  # Tambourine
    57: 16,  # Reversed Rise
    62: 17,  # Mute Drumsynth Conga
    63: 18,  # High Drumsynth Conga
    64: 19,  # Low Drumsynth Conga
    69: 20,  # Cabasa
    70: 21,  # Maracas
    75: 22,  # Claves
    76: 23,  # High Wood Block
    77: 24,  # Low Wood Block
}


def detect_onsets(y, sr):
    """Detect onset times in audio"""
    onset_frames = librosa.onset.onset_detect(
        y=y, 
        sr=sr, 
        backtrack=True,
        units='frames'
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return onset_times


def extract_features(segment, sr):
    """
    Extract audio features for classification
    
    Returns: (centroid, rms, rolloff, zcr)
    """
    if len(segment) < 100:
        return np.array([0, 0, 0, 0])
    
    centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)[0].mean()
    rms = librosa.feature.rms(y=segment)[0].mean()
    rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)[0].mean()
    zcr = librosa.feature.zero_crossing_rate(segment)[0].mean()
    
    return np.array([centroid, rms, rolloff, zcr])


def extract_velocity(segment):
    """Extract velocity from segment amplitude"""
    if len(segment) == 0:
        return 0.0
    
    rms = librosa.feature.rms(y=segment)[0].mean()
    velocity = min(rms * 5, 1.0)  # Normalize to [0, 1]
    return velocity


def discretize_velocity(velocity): # changed from 4 bins
    """Return continuous velocity, 0.0 for no hit, (0,1] for hits."""
    if velocity <= 0.01:
        return 0.0
    return float(velocity)  


def map_clusters_to_drums(cluster_centers, valid_drums):
    """
    Map each cluster to one of the valid drums - DATA DRIVEN approach
    
    Uses principle: sort clusters by centroid, sort drums by MIDI note,
    map lowest cluster to lowest drum (lower notes = lower frequencies)
    
    Args:
        cluster_centers: (K, 4) array of cluster centroids
        valid_drums: list of MIDI note numbers present in file
    
    Returns:
        dict: {cluster_id: drum_note}
    """
    n_clusters = len(cluster_centers)
    n_drums = len(valid_drums)
    
    # Handle edge cases
    if n_clusters == 0 or n_drums == 0:
        return {}
    
    # Sort clusters by spectral centroid (low to high frequency)
    # Centroid is the first feature
    cluster_centroids = cluster_centers[:, 0]
    sorted_cluster_ids = np.argsort(cluster_centroids)
    
    # Sort valid drums by MIDI note number (low notes = low freq drums)
    # This is principled: MIDI note 36 (kick) < 42 (hat) < 70 (maracas)
    sorted_drums = sorted(valid_drums)
    
    # Map clusters to drums
    cluster_to_drum = {}
    
    if n_clusters <= n_drums:
        # More drums available than clusters found
        # Map each cluster to corresponding drum in sorted order
        for i, cluster_id in enumerate(sorted_cluster_ids):
            cluster_to_drum[cluster_id] = sorted_drums[i]
    else:
        # More clusters than drums (oversegmentation)
        # Distribute clusters evenly across available drums
        for i, cluster_id in enumerate(sorted_cluster_ids):
            drum_idx = int(i * n_drums / n_clusters)
            cluster_to_drum[cluster_id] = sorted_drums[drum_idx]
    
    return cluster_to_drum


def time_to_step(onset_time, tempo, n_steps=64):
    """Convert onset time to grid step"""
    bar_duration = 60 / tempo * 4  # Duration of one bar in seconds
    total_duration = bar_duration * 4  # 4 bars
    step = int((onset_time / total_duration) * n_steps)
    return min(max(step, 0), n_steps - 1)


def load_json_metadata(json_path):
    """
    Load metadata from individual JSON file
    
    Returns dict with 'pitch' and 'tempo' keys
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # JSON structure: {"filename": {"pitch": [...], "tempo": 120}}
    # Get the first (and only) key
    file_id = list(data.keys())[0]
    return data[file_id]


def process_single_file(wav_file, drums_present, tempo, sr=44100, n_steps=64):
    """
    Process one WAV file into labeled 27×64 array
    
    Args:
        wav_file: path to WAV file
        drums_present: list of MIDI notes present in file
        tempo: BPM
        sr: sample rate
        n_steps: number of time steps
    
    Returns:
        pianoroll: (27, n_steps) array with velocity bins [0-4]
    """
    # Load audio
    y, sr = librosa.load(wav_file, sr=sr)
    
    # Detect onsets
    onset_times = detect_onsets(y, sr)
    
    if len(onset_times) == 0:
        return np.zeros((27, n_steps), dtype=np.int32)
    
    # Extract features and velocities for each onset
    features_list = []
    
    for onset_time in onset_times:
        # Extract segment around onset
        start_sample = int(onset_time * sr)
        end_sample = int((onset_time + 0.1) * sr)  # 100ms window
        segment = y[start_sample:end_sample]
        
        if len(segment) < 100:
            continue
        
        features = extract_features(segment, sr)
        velocity = extract_velocity(segment)
        
        features_list.append((onset_time, features, velocity))
    
    if len(features_list) == 0:
        return np.zeros((27, n_steps), dtype=np.int32)
    
    # Cluster into K groups (K = number of drums present)
    K = len(drums_present)
    K = max(1, min(K, len(features_list)))  # Constrain K
    
    feature_matrix = np.array([f[1] for f in features_list])
    
    # Normalize features
    feature_matrix = (feature_matrix - feature_matrix.mean(axis=0)) / (feature_matrix.std(axis=0) + 1e-8)
    
    # Cluster
    if K == 1:
        cluster_labels = np.zeros(len(features_list), dtype=int)
        cluster_centers = feature_matrix.mean(axis=0, keepdims=True)
    else:
        kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix)
        cluster_centers = kmeans.cluster_centers_
    
    # Map clusters to drum types (DATA DRIVEN - no arbitrary centroids!)
    cluster_to_drum = map_clusters_to_drums(cluster_centers, drums_present)
    
    # Build 27×64 array
    pianoroll = np.zeros((27, n_steps), dtype=np.float32)
    
    for (onset_time, features, velocity), cluster_id in zip(features_list, cluster_labels):
        # Get drum type
        drum_note = cluster_to_drum.get(cluster_id)
        if drum_note is None or drum_note not in NOTE_TO_INDEX_27:
            continue
        
        drum_idx = NOTE_TO_INDEX_27[drum_note]
        
        # Quantize time to grid
        step = time_to_step(onset_time, tempo, n_steps)
        
        # Bin velocity
        vel_bin = discretize_velocity(velocity)
        
        # Place in pianoroll (take max if multiple hits on same step)
        pianoroll[drum_idx, step] = max(pianoroll[drum_idx, step], vel_bin)
    
    return pianoroll


def process_dataset(wav_dir, json_dir, output_path, max_files=None, n_steps=64):
    """
    Process entire EDM HSE dataset
    
    Args:
        wav_dir: directory containing WAV files
        json_dir: directory containing JSON metadata files (one per WAV)
        output_path: output .pkl file
        max_files: limit number of files (None = all)
        n_steps: number of time steps per pattern
    """
    print("="*70)
    print("PROCESSING EDM HSE WITH 27 LABELED DRUMS")
    print("="*70)
    
    # Find WAV files
    print(f"\n[1/3] Scanning for files...")
    wav_dir = Path(wav_dir)
    json_dir = Path(json_dir)
    
    wav_files = sorted(list(wav_dir.glob("*.wav")))
    
    if max_files:
        wav_files = wav_files[:max_files]
        print(f"  Limiting to {max_files} files")
    
    print(f"  Found: {len(wav_files)} WAV files")
    print(f"  WAV dir: {wav_dir}")
    print(f"  JSON dir: {json_dir}")
    
    # Process each file
    print(f"\n[2/3] Processing WAV files with clustering...")
    
    all_pianorolls = []
    failed_files = []
    skipped_files = []
    
    for wav_file in tqdm(wav_files, desc="Processing"):
        try:
            file_id = wav_file.stem
            
            # Find corresponding JSON file
            json_file = json_dir / f"{file_id}.json"
            
            if not json_file.exists():
                skipped_files.append(file_id)
                continue
            
            # Load metadata
            metadata = load_json_metadata(json_file)
            drums_present = metadata['pitch']
            tempo = metadata['tempo']
            
            # Process file
            pianoroll = process_single_file(
                wav_file, 
                drums_present, 
                tempo,
                n_steps=n_steps
            )
            
            all_pianorolls.append(pianoroll)
            
        except Exception as e:
            print(f"\n  ERROR processing {wav_file.name}: {e}")
            failed_files.append(wav_file.name)
            continue
    
    print(f"\n  Processed: {len(all_pianorolls)} files")
    print(f"  Skipped (no JSON): {len(skipped_files)} files")
    print(f"  Failed: {len(failed_files)} files")
    
    if len(all_pianorolls) == 0:
        print("  ERROR: No files successfully processed!")
        return None
    
    # Stack into array
    print(f"\n[3/3] Creating dataset array...")
    data = np.stack(all_pianorolls)  # (N, 27, n_steps)
    
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    
    # Analyze distribution
    print(f"\n  Velocity bin distribution:")
    for bin_idx in range(5):
        count = (data == bin_idx).sum()
        pct = count / data.size * 100
        if bin_idx == 0:
            print(f"    Bin {bin_idx} (no hit):   {count:10d} ({pct:5.2f}%)")
        else:
            print(f"    Bin {bin_idx} (velocity): {count:10d} ({pct:5.2f}%)")
    
    # Drum usage statistics
    print(f"\n  Drum track usage:")
    for drum_note, drum_idx in sorted(NOTE_TO_INDEX_27.items(), key=lambda x: x[1]):
        hits = (data[:, drum_idx, :] > 0).sum()
        if hits > 0:
            pct = hits / (data.shape[0] * data.shape[2]) * 100
            print(f"    Track {drum_idx:2d} (note {drum_note:2d}): {hits:6d} hits ({pct:5.2f}%)")
    
    # Save
    print(f"\n  Saving dataset...")
    print(f"  Output: {output_path}")
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    file_size = Path(output_path).stat().st_size / 1024 / 1024
    print(f"  File size: {file_size:.2f} MB")
    
    print("\n" + "="*70)
    print("DATASET PROCESSING COMPLETE!")
    print("="*70)
    
    print(f"\nDataset info:")
    print(f"  Samples: {len(data)}")
    print(f"  Shape: {data.shape} (samples, drums, timesteps)")
    print(f"  Velocity bins: 0-4 (0=no hit, 1-4=velocity levels)")
    print(f"  Sparsity: {(data == 0).sum() / data.size * 100:.2f}%")
    
    if len(failed_files) > 0:
        print(f"\nFailed files ({len(failed_files)}):")
        for f in failed_files[:10]:
            print(f"  - {f}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    
    if len(skipped_files) > 0:
        print(f"\nSkipped files (no matching JSON) ({len(skipped_files)}):")
        for f in skipped_files[:10]:
            print(f"  - {f}")
        if len(skipped_files) > 10:
            print(f"  ... and {len(skipped_files) - 10} more")
    
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process EDM HSE with 27 labeled drums using clustering'
    )
    
    parser.add_argument('--wav-dir', type=str, required=True,
                       help='Directory containing WAV files')
    parser.add_argument('--json-dir', type=str, required=True,
                       help='Directory containing JSON metadata files')
    parser.add_argument('--output', type=str, default='edm_hse_27drums_continuous.pkl',
                       help='Output .pkl file')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Limit number of files (None = all)')
    parser.add_argument('--steps', type=int, default=64,
                       help='Number of time steps per pattern')
    
    args = parser.parse_args()
    
    data = process_dataset(
        wav_dir=args.wav_dir,
        json_dir=args.json_dir,
        output_path=args.output,
        max_files=args.max_files,
        n_steps=args.steps
    )
    
    if data is not None:
        print("\n✓ Success! Dataset ready for training.")
        print(f"\nNext steps:")
        print(f"  1. Check output: {args.output}")
        print(f"  2. Inspect: python inspect_27drums.py {args.output}")
        print(f"  3. If looks good, process all files (remove --max-files)")
        print(f"  4. Train VQ-VAE with 27 drum inputs")
