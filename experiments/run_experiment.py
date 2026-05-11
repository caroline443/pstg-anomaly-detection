"""
Main Experiment Runner
========================
Trains and evaluates the PSTG model with conditional causal graph.

Training mode:
  - joint:  All entities with the same channel count are merged into one
            training set. A single shared model is trained per channel-group,
            then evaluated on each entity individually.
  - single: (legacy) Each entity trains its own model independently.

Usage:
    python experiments/run_experiment.py --config configs/default.yaml
    python experiments/run_experiment.py --config configs/default.yaml --dataset smap --mode joint
    python experiments/run_experiment.py --config configs/default.yaml --dataset smap --mode single
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.pstg import PSTGModel
from src.pstg.threshold import DynamicThreshold


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logger(log_dir: str, run_name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{run_name}.log')

    logger = logging.getLogger('pstg')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f'Log file: {log_path}')
    return logger


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_windows(data: np.ndarray, window: int, stride: int = 1):
    """Slide a window over (T, C) data, return (N_windows, window, C)."""
    T, C = data.shape
    windows = []
    for i in range(0, T - window + 1, stride):
        windows.append(data[i:i + window])
    return np.stack(windows)  # (N_windows, window, C)


def group_entities_by_channels(processed_dir: str, entity_files: list) -> dict:
    """
    Group entity IDs by their channel count.

    Returns:
        groups: {n_channels: [entity_id, ...]}
    """
    groups = defaultdict(list)
    for eid in entity_files:
        train_data = np.load(os.path.join(processed_dir, f'{eid}_train.npy'))
        n_ch = train_data.shape[1]
        groups[n_ch].append(eid)
    return dict(groups)


def build_joint_train_loader(
    processed_dir: str,
    entity_ids: list,
    window: int,
    pred_len: int,
    stride: int,
    batch_size: int,
) -> DataLoader:
    """
    Merge training windows from all entities in the group into one DataLoader.
    Each entity is independently windowed then concatenated.
    """
    all_x, all_y = [], []
    for eid in entity_ids:
        train_data = np.load(os.path.join(processed_dir, f'{eid}_train.npy'))
        wins = make_windows(train_data, window + pred_len, stride=stride)
        if len(wins) == 0:
            continue
        x = torch.FloatTensor(wins[:, :window, :]).permute(0, 2, 1)   # (N, C, T)
        y = torch.FloatTensor(wins[:, window:, :]).permute(0, 2, 1)   # (N, C, F)
        all_x.append(x)
        all_y.append(y)

    if not all_x:
        return None

    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)

    loader = DataLoader(
        TensorDataset(all_x, all_y),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    return loader


# ── Training / evaluation ──────────────────────────────────────────────────────

def build_model(n_channels: int, config: dict, device: torch.device) -> PSTGModel:
    mc = config['model']
    dc = config['data']
    return PSTGModel(
        n_channels=n_channels,
        seq_len=dc['window_length'],
        pred_len=dc['prediction_length'],
        patch_sizes=mc['patch_sizes'],
        d_model=mc['d_model'],
        n_heads=mc['n_heads'],
        n_layers=mc['n_layers'],
        causal_hidden_dim=mc['causal_hidden_dim'],
        causal_lag=mc['causal_lag'],
        sparsity_k=max(1, int(n_channels * mc['sparsity_ratio'])),
        dropout=mc['dropout'],
    ).to(device)


def train_model(
    model: PSTGModel,
    train_loader: DataLoader,
    config: dict,
    device: torch.device,
    logger: logging.Logger,
    desc: str = 'training',
) -> PSTGModel:
    """Train model for the configured number of epochs."""
    tc = config['training']

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc['learning_rate'],
        weight_decay=tc['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc['T_max'], eta_min=tc['eta_min']
    )
    criterion = nn.MSELoss()

    for epoch in tqdm(range(tc['epochs']), desc=desc, leave=False):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            logger.info(f'  [{desc}] Epoch {epoch+1}/{tc["epochs"]}  loss={avg_loss:.4f}')

    return model


@torch.no_grad()
def predict(model: PSTGModel, loader: DataLoader, device: torch.device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device)
        pred = model(x)
        preds.append(pred.cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def compute_errors(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-window mean absolute error across channels and time steps."""
    return np.abs(preds - targets).mean(axis=(1, 2))


def evaluate(pred_labels: np.ndarray, true_labels: np.ndarray) -> dict:
    tp = np.sum((pred_labels == 1) & (true_labels == 1))
    fp = np.sum((pred_labels == 1) & (true_labels == 0))
    fn = np.sum((pred_labels == 0) & (true_labels == 1))
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    beta = 0.5
    f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)
    return {'precision': float(precision), 'recall': float(recall), f'F{beta}': float(f_beta)}


def detect_entity(
    model: PSTGModel,
    processed_dir: str,
    entity_id: str,
    config: dict,
    device: torch.device,
    logger: logging.Logger,
) -> dict:
    """
    Run anomaly detection on a single entity using a (pre-trained) shared model.
    Returns metrics dict.
    """
    dc = config['data']
    ac = config['anomaly_detection']
    W  = dc['window_length']
    F  = dc['prediction_length']

    train_data  = np.load(os.path.join(processed_dir, f'{entity_id}_train.npy'))
    test_data   = np.load(os.path.join(processed_dir, f'{entity_id}_test.npy'))
    true_labels = np.load(os.path.join(processed_dir, f'{entity_id}_labels.npy'))

    # Build per-entity loaders (stride=1 for test to get dense predictions)
    train_wins = make_windows(train_data, W + F, stride=dc['stride'])
    test_wins  = make_windows(test_data,  W + F, stride=1)

    if len(train_wins) == 0 or len(test_wins) == 0:
        logger.info(f'  [skip] {entity_id}: not enough windows')
        return None

    train_x = torch.FloatTensor(train_wins[:, :W, :]).permute(0, 2, 1)
    train_y = torch.FloatTensor(train_wins[:, W:, :]).permute(0, 2, 1)
    test_x  = torch.FloatTensor(test_wins[:, :W, :]).permute(0, 2, 1)
    test_y  = torch.FloatTensor(test_wins[:, W:, :]).permute(0, 2, 1)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=ac['test_batch_size'], shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=ac['test_batch_size'], shuffle=False,
    )

    # Predict
    preds,  targets  = predict(model, test_loader,  device)
    t_preds, t_tgts  = predict(model, train_loader, device)

    errors       = compute_errors(preds,   targets)
    train_errors = compute_errors(t_preds, t_tgts)

    # Threshold detection
    detector = DynamicThreshold(
        smoothing_base=ac['smoothing_base'],
        test_batch_size=ac['test_batch_size'],
        tuning_percentage=ac['tuning_percentage'],
        use_adaptive=ac.get('use_adaptive', True),
    )
    pred_labels_win = detector.detect(errors, train_errors=train_errors)
    logger.info(f'  {entity_id}: adaptive tuning_p={detector.calibrated_p:.4f}')

    # Map window labels back to time-step labels
    pred_labels_ts = np.zeros(len(test_data), dtype=int)
    for i, label in enumerate(pred_labels_win):
        if label == 1:
            pred_labels_ts[i:i + W + F] = 1

    min_len = min(len(pred_labels_ts), len(true_labels))
    metrics = evaluate(pred_labels_ts[:min_len], true_labels[:min_len])
    return metrics


# ── Joint training mode ────────────────────────────────────────────────────────

def run_joint(config: dict, processed_dir: str, entity_files: list,
              device: torch.device, logger: logging.Logger,
              log_dir: str, run_name: str) -> list:
    """
    Joint training: group entities by channel count, train one shared model
    per group, then evaluate each entity individually.
    """
    tc = config['training']
    dc = config['data']

    logger.info('Mode: JOINT training (shared model per channel-group)')

    # Group entities by channel count
    groups = group_entities_by_channels(processed_dir, entity_files)
    logger.info(f'Channel groups: { {k: len(v) for k, v in groups.items()} }')

    all_results = []
    csv_path = os.path.join(log_dir, f'{run_name}_results.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=['entity', 'precision', 'recall', 'F0.5', 'time_s']).writeheader()

    for n_ch, entity_ids in sorted(groups.items()):
        logger.info(f"\n{'='*60}")
        logger.info(f'Channel group: {n_ch} channels  ({len(entity_ids)} entities)')
        logger.info(f'Entities: {entity_ids}')

        # ── Build joint training loader ──────────────────────────────────────
        joint_loader = build_joint_train_loader(
            processed_dir=processed_dir,
            entity_ids=entity_ids,
            window=dc['window_length'],
            pred_len=dc['prediction_length'],
            stride=dc['stride'],
            batch_size=tc['batch_size'],
        )
        if joint_loader is None:
            logger.info('  [skip] no valid training windows in this group')
            continue

        total_windows = len(joint_loader.dataset)
        logger.info(f'  Joint training windows: {total_windows}')

        # ── Train shared model ───────────────────────────────────────────────
        model = build_model(n_ch, config, device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'  Model params: {n_params:,}')

        model = train_model(
            model, joint_loader, config, device, logger,
            desc=f'ch={n_ch} joint',
        )

        # ── Evaluate each entity individually ────────────────────────────────
        for entity_id in entity_ids:
            t0 = time.time()
            metrics = detect_entity(
                model, processed_dir, entity_id, config, device, logger
            )
            if metrics is None:
                continue
            elapsed = time.time() - t0

            logger.info(
                f'  {entity_id:8s}  P={metrics["precision"]:.3f}'
                f'  R={metrics["recall"]:.3f}  F0.5={metrics["F0.5"]:.3f}'
                f'  ({elapsed:.0f}s)'
            )
            row = {'entity': entity_id, **metrics, 'time_s': round(elapsed, 1)}
            all_results.append(row)

            with open(csv_path, 'a', newline='') as f:
                csv.DictWriter(
                    f, fieldnames=['entity', 'precision', 'recall', 'F0.5', 'time_s']
                ).writerow(row)

    return all_results


# ── Single (legacy) training mode ─────────────────────────────────────────────

def run_single(config: dict, processed_dir: str, entity_files: list,
               device: torch.device, logger: logging.Logger,
               log_dir: str, run_name: str) -> list:
    """
    Legacy mode: train one model per entity independently.
    """
    tc = config['training']
    dc = config['data']
    ac = config['anomaly_detection']

    logger.info('Mode: SINGLE training (one model per entity)')

    all_results = []
    csv_path = os.path.join(log_dir, f'{run_name}_results.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=['entity', 'precision', 'recall', 'F0.5', 'time_s']).writeheader()

    W = dc['window_length']
    F = dc['prediction_length']

    for entity_id in entity_files:
        logger.info(f"\n{'='*50}")
        logger.info(f'Entity: {entity_id}')
        t0 = time.time()

        train_data  = np.load(os.path.join(processed_dir, f'{entity_id}_train.npy'))
        test_data   = np.load(os.path.join(processed_dir, f'{entity_id}_test.npy'))
        true_labels = np.load(os.path.join(processed_dir, f'{entity_id}_labels.npy'))

        n_channels = train_data.shape[1]
        train_wins = make_windows(train_data, W + F, stride=dc['stride'])
        test_wins  = make_windows(test_data,  W + F, stride=1)

        if len(train_wins) < tc['batch_size']:
            logger.info(f'  [skip] only {len(train_wins)} training windows')
            continue

        train_x = torch.FloatTensor(train_wins[:, :W, :]).permute(0, 2, 1)
        train_y = torch.FloatTensor(train_wins[:, W:, :]).permute(0, 2, 1)
        test_x  = torch.FloatTensor(test_wins[:, :W, :]).permute(0, 2, 1)
        test_y  = torch.FloatTensor(test_wins[:, W:, :]).permute(0, 2, 1)

        train_loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=tc['batch_size'], shuffle=True, drop_last=True,
        )
        test_loader = DataLoader(
            TensorDataset(test_x, test_y),
            batch_size=ac['test_batch_size'], shuffle=False,
        )

        model = build_model(n_channels, config, device)
        model = train_model(model, train_loader, config, device, logger, desc=entity_id)

        preds,   targets  = predict(model, test_loader,  device)
        t_preds, t_tgts   = predict(model, train_loader, device)
        errors       = compute_errors(preds,   targets)
        train_errors = compute_errors(t_preds, t_tgts)

        detector = DynamicThreshold(
            smoothing_base=ac['smoothing_base'],
            test_batch_size=ac['test_batch_size'],
            tuning_percentage=ac['tuning_percentage'],
            use_adaptive=ac.get('use_adaptive', True),
        )
        pred_labels_win = detector.detect(errors, train_errors=train_errors)
        logger.info(f'  Adaptive tuning_p={detector.calibrated_p:.4f}')

        pred_labels_ts = np.zeros(len(test_data), dtype=int)
        for i, label in enumerate(pred_labels_win):
            if label == 1:
                pred_labels_ts[i:i + W + F] = 1

        min_len = min(len(pred_labels_ts), len(true_labels))
        metrics = evaluate(pred_labels_ts[:min_len], true_labels[:min_len])
        elapsed = time.time() - t0

        logger.info(
            f'  Results: P={metrics["precision"]:.3f}  R={metrics["recall"]:.3f}'
            f'  F0.5={metrics["F0.5"]:.3f}  ({elapsed:.0f}s)'
        )
        row = {'entity': entity_id, **metrics, 'time_s': round(elapsed, 1)}
        all_results.append(row)

        with open(csv_path, 'a', newline='') as f:
            csv.DictWriter(
                f, fieldnames=['entity', 'precision', 'recall', 'F0.5', 'time_s']
            ).writerow(row)

    return all_results


# ── Main ───────────────────────────────────────────────────────────────────────

def run(config: dict, data_dir: str, dataset: str, mode: str,
        device: torch.device, log_dir: str, run_name: str):
    logger = setup_logger(log_dir, run_name)
    logger.info(f'Dataset: {dataset}  Mode: {mode}  Device: {device}')
    logger.info(f'Config: {json.dumps(config, indent=2)}')

    processed_dir = os.path.join(data_dir, f'{dataset}_processed')
    entity_files = sorted([
        f.replace('_train.npy', '')
        for f in os.listdir(processed_dir) if f.endswith('_train.npy')
    ])
    logger.info(f'Found {len(entity_files)} entities')

    if mode == 'joint':
        all_results = run_joint(
            config, processed_dir, entity_files, device, logger, log_dir, run_name
        )
    else:
        all_results = run_single(
            config, processed_dir, entity_files, device, logger, log_dir, run_name
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info('Summary:')
    for r in all_results:
        logger.info(
            f"  {r['entity']:8s}  P={r['precision']:.3f}"
            f"  R={r['recall']:.3f}  F0.5={r['F0.5']:.3f}"
        )

    avg_f = float(np.mean([r['F0.5'] for r in all_results])) if all_results else 0.0
    logger.info(f'\nAverage F0.5: {avg_f:.3f}  (over {len(all_results)} entities)')

    summary_path = os.path.join(log_dir, f'{run_name}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({'avg_F0.5': avg_f, 'results': all_results, 'config': config}, f, indent=2)
    logger.info(f'Summary saved to: {summary_path}')

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   default='configs/default.yaml')
    parser.add_argument('--dataset',  choices=['smap', 'msl'], default='smap')
    parser.add_argument('--mode',     choices=['joint', 'single'], default='joint',
                        help='joint: shared model per channel-group; single: one model per entity')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--device',   default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--log_dir',  default='logs')
    args = parser.parse_args()

    run_name = f'{args.dataset}_{args.mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    config   = load_config(args.config)
    device   = torch.device(args.device)

    run(config, args.data_dir, args.dataset, args.mode, device, args.log_dir, run_name)


if __name__ == '__main__':
    main()
