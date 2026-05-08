"""
Data Preprocessing Script
===========================
Supports SMAP and MSL datasets (NASA public telemetry).

Usage:
    python scripts/preprocess.py --dataset smap --data_dir ./data
    python scripts/preprocess.py --dataset msl  --data_dir ./data
"""

import argparse
import os
import numpy as np


def load_smap_msl(data_dir: str, dataset: str):
    """
    Load SMAP or MSL dataset from the standard directory structure.

    Expected layout (from NASA Telemanom repo):
        data/
          train/  *.npy
          test/   *.npy
          labeled_anomalies.csv

    Args:
        data_dir: Root data directory.
        dataset:  'smap' or 'msl'.

    Returns:
        entities: List of (entity_id, train_arr, test_arr, labels) tuples.
    """
    import pandas as pd

    label_file = os.path.join(data_dir, 'labeled_anomalies.csv')
    if not os.path.exists(label_file):
        raise FileNotFoundError(
            f"labeled_anomalies.csv not found in {data_dir}.\n"
            "Download from: https://github.com/khundman/telemanom"
        )

    labels_df = pd.read_csv(label_file)
    labels_df = labels_df[labels_df['spacecraft'] == dataset.upper()]

    entities = []
    for _, row in labels_df.iterrows():
        entity_id = row['chan_id']
        train_path = os.path.join(data_dir, 'train', f'{entity_id}.npy')
        test_path = os.path.join(data_dir, 'test', f'{entity_id}.npy')

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"  [skip] {entity_id}: data files not found")
            continue

        train_arr = np.load(train_path)
        test_arr = np.load(test_path)

        # Parse anomaly labels
        anomaly_seqs = eval(row['anomaly_sequences'])
        labels = np.zeros(len(test_arr), dtype=int)
        for start, end in anomaly_seqs:
            labels[start:end + 1] = 1

        entities.append((entity_id, train_arr, test_arr, labels))
        print(f"  [ok] {entity_id}: train={train_arr.shape}, test={test_arr.shape}, "
              f"anomaly_ratio={labels.mean():.3f}")

    return entities


def normalize(train: np.ndarray, test: np.ndarray):
    """Z-score normalization using training set statistics."""
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-8
    return (train - mean) / std, (test - mean) / std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['smap', 'msl'], default='smap')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.data_dir, f'{args.dataset}_processed')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {args.dataset.upper()} from {args.data_dir} ...")
    entities = load_smap_msl(args.data_dir, args.dataset)

    print(f"\nNormalizing and saving to {output_dir} ...")
    for entity_id, train_arr, test_arr, labels in entities:
        train_norm, test_norm = normalize(train_arr, test_arr)
        np.save(os.path.join(output_dir, f'{entity_id}_train.npy'), train_norm)
        np.save(os.path.join(output_dir, f'{entity_id}_test.npy'), test_norm)
        np.save(os.path.join(output_dir, f'{entity_id}_labels.npy'), labels)

    print(f"\nDone. Processed {len(entities)} entities.")


if __name__ == '__main__':
    main()
