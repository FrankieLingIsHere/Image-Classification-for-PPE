#!/usr/bin/env python3
"""Train a simple MLP rescorer using detections matched to GT as TP/FP labels.

This script:
- Loads the detection model (rcnn or ssd) using the existing evaluator loader
- Runs detections on the train split
- Matches detections to GT boxes (IoU >= 0.5 -> TP)
- Builds node feature vectors [score, cx, cy, w, h, area]
- Trains a small MLP (RelationalRescorer) to predict TP vs FP
"""
import os
import sys
import argparse
from pathlib import Path
import json
import torch
import numpy as np

# Ensure repo root is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.eval.evaluate_detection_performance import PPEDetectionEvaluator
from src.models.relational_rescorer import create_rescorer
from src.utils.utils import calculate_iou


def build_dataset(evaluator: PPEDetectionEvaluator, split='train', max_images=None):
    gt = evaluator._load_ground_truth(split)
    items = list(gt.items())
    if max_images:
        items = items[:max_images]

    X = []
    y = []

    for img_name, annotations in items:
        img_path = evaluator.data_dir / 'images' / img_name
        detections = evaluator._detect_image(img_path)

        # Convert GT boxes to [x1,y1,x2,y2]
        gt_boxes = [ann['bbox'] for ann in annotations]

        for d in detections:
            bbox = d['bbox']
            conf = float(d.get('confidence', 0.0))
            x1, y1, x2, y2 = bbox
            # compute normalized cx,cy,w,h,area using pixel coords
            img_w = 300
            img_h = 300
            try:
                img = Path(evaluator.data_dir) / 'images' / img_name
                from PIL import Image
                im = Image.open(img)
                img_w, img_h = im.size
            except Exception:
                pass

            cx = (x1 + x2) / 2.0 / img_w
            cy = (y1 + y2) / 2.0 / img_h
            w = max(0.0, (x2 - x1) / img_w)
            h = max(0.0, (y2 - y1) / img_h)
            area = w * h

            # Match to GT by IoU
            label = 0
            for g in gt_boxes:
                iou = calculate_iou([x1, y1, x2, y2], g)
                if iou >= 0.5:
                    label = 1
                    break

            X.append([conf, cx, cy, w, h, area])
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y


def train(args):
    evaluator = PPEDetectionEvaluator(args.model_path, args.data_dir, args.config_path, args.output_dir)
    print('[INFO] Building dataset from train split...')
    X, y = build_dataset(evaluator, split='train', max_images=args.max_images)
    print(f'[INFO] Built dataset X={X.shape} y={y.shape}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type = getattr(args, 'model_type', 'mlp') or 'mlp'
    if model_type == 'gat':
        model = create_rescorer('gat', node_feat_dim=X.shape[1], emb_dim=args.emb_dim, num_heads=2)
    else:
        model = create_rescorer('mlp', node_feat_dim=X.shape[1], hidden=args.hidden)
    model = model.to(device)
    # Use BCEWithLogitsLoss for numerical stability; models now output logits
    if args.class_balance:
        # compute positive class frequency for pos_weight
        pos = float(y.sum())
        neg = float(len(y) - pos)
        if pos == 0:
            pos_weight = torch.tensor(1.0, device=device)
        else:
            pos_weight = torch.tensor(neg / pos, device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f'[INFO] Using class-balanced BCEWithLogitsLoss pos_weight={pos_weight.item():.3f}')
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    epochs = args.epochs
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            # preds are logits, ensure yb shape matches
            if preds.dim() == 2:
                y_target = yb.unsqueeze(-1).float()
            else:
                y_target = yb.float()
            loss = criterion(preds, y_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg = total_loss / len(dataset) if len(dataset)>0 else 0.0
        print(f'[EPOCH {epoch+1}/{epochs}] loss={avg:.6f}')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (f'rescorer_{model_type}.pth' if model_type else 'rescorer.pth')
    torch.save(model.state_dict(), out_path)
    print(f'[OK] Saved rescorer weights to {out_path}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, default='models/rcnn_baseline.pth')
    p.add_argument('--data_dir', type=str, default='data')
    p.add_argument('--config_path', type=str, default='configs/best_runtime_config.yaml')
    p.add_argument('--output_dir', type=str, default='models')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--max_images', type=int, default=200)
    p.add_argument('--model_type', type=str, default='mlp', choices=['mlp', 'gat'],
                   help='Type of rescorer to train')
    p.add_argument('--class_balance', action='store_true',
                   help='Use class-balanced BCEWithLogitsLoss (pos_weight based on training set)')
    p.add_argument('--hidden', type=int, default=128, help='Hidden dim for MLP')
    p.add_argument('--emb_dim', type=int, default=128, help='Embedding dim for GAT')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
