#!/usr/bin/env python3
"""Quick Faster R-CNN baseline fine-tune + inference

Saves model to: models/rcnn_baseline.pth
Writes per-image detection JSONs to: outputs/rcnn_baseline/<image>_results.json

This is intentionally separated from existing SSD artifacts.
"""

import os
import json
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Make sure `src` package is importable
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from src.dataset.ppe_dataset import load_ppe_images_and_annotations


PPE_CLASSES = [
    'background', 'person', 'hard_hat', 'safety_vest', 'safety_gloves',
    'safety_boots', 'eye_protection', 'no_hard_hat', 'no_safety_vest',
    'no_safety_gloves', 'no_safety_boots', 'no_eye_protection'
]


class TorchvisionPPEDataset(Dataset):
    def __init__(self, data_dir, split='train', transforms=None):
        self.data_dir = data_dir
        self.split = split
        self.class2idx = {c: i for i, c in enumerate(PPE_CLASSES)}
        self.images_info = load_ppe_images_and_annotations(data_dir, self.class2idx, split)
        # allow simple augmentations via transforms (RandomHorizontalFlip mostly)
        self.transforms = transforms or T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        info = self.images_info[idx]
        img_path = info['filename']
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        # Build target dict expected by torchvision models
        boxes = []
        labels = []
        if info.get('detections'):
            for det in info['detections']:
                # det['bbox'] is [xmin, ymin, xmax, ymax] (integers)
                boxes.append(det['bbox'])
                labels.append(det['label'])

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            # Filter out invalid boxes with non-positive width/height
            if boxes.numel() > 0:
                x1 = boxes[:, 0]
                y1 = boxes[:, 1]
                x2 = boxes[:, 2]
                y2 = boxes[:, 3]
                valid_mask = (x2 > x1) & (y2 > y1)
                if valid_mask.sum().item() != boxes.size(0):
                    boxes = boxes[valid_mask]
                    labels = labels[valid_mask]
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        img_t = self.transforms(img)
        return img_t, target, info['img_id']


def get_model(num_classes):
    # load a model pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def collate_fn(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    ids = [b[2] for b in batch]
    return images, targets, ids


def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0.0
    # accumulate per-key losses so we can inspect which term is stuck
    loss_sums = {}
    iters = 0
    for imgs, targets, _ in data_loader:
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        iters += 1
        for k, v in loss_dict.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + float(v.item())

    # build averaged loss dict
    avg_loss = total_loss / iters if iters > 0 else 0.0
    avg_loss_components = {k: v / iters for k, v in loss_sums.items()}
    return avg_loss, avg_loss_components


def run_inference(model, dataset, device, out_dir):
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i in range(len(dataset)):
            img_t, _, img_id = dataset[i]
            w, h = Image.open(dataset.images_info[i]['filename']).size
            imgs = [img_t.to(device)]
            outputs = model(imgs)
            out = outputs[0]

            detections = []
            boxes = out.get('boxes', torch.empty((0, 4)))
            scores = out.get('scores', torch.empty((0,)))
            labels = out.get('labels', torch.empty((0,)))

            for b, s, l in zip(boxes, scores, labels):
                x1, y1, x2, y2 = b.cpu().numpy().tolist()
                # convert to fractional coordinates
                fx1 = x1 / w
                fy1 = y1 / h
                fx2 = x2 / w
                fy2 = y2 / h
                detections.append({
                    'class_id': int(l.item()),
                    'class_name': PPE_CLASSES[int(l.item())] if int(l.item()) < len(PPE_CLASSES) else f'class_{int(l.item())}',
                    'confidence': float(s.item()),
                    'bbox': [fx1, fy1, fx2, fy2]
                })

            out_json = out_dir / f"{img_id}_results.json"
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump({'image_id': img_id, 'detections': detections}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adamw'])
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--step_lr', action='store_true', help='Use StepLR scheduler')
    parser.add_argument('--step_size', type=int, default=4)
    parser.add_argument('--step_gamma', type=float, default=0.1)
    parser.add_argument('--freeze_backbone_epochs', type=int, default=0, help='Number of initial epochs to freeze backbone')
    parser.add_argument('--augment', action='store_true', help='Use simple augmentations (RandomHorizontalFlip)')
    parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--resume-start-epoch', type=int, default=None, help='(Optional) force trainer to treat resume checkpoint as having completed this many epochs (useful if checkpoint lacks epoch metadata)')
    parser.add_argument('--output_model', type=str, default='models/rcnn_baseline.pth')
    parser.add_argument('--output_dir', type=str, default='outputs/rcnn_baseline')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Datasets
    # setup transforms / augmentations
    train_transforms = None
    if args.augment:
        train_transforms = T.Compose([T.RandomHorizontalFlip(0.5), T.ToTensor()])

    train_ds = TorchvisionPPEDataset(args.data_dir, split='train', transforms=train_transforms)
    val_ds = TorchvisionPPEDataset(args.data_dir, split='test')

    if len(train_ds) == 0:
        print('No training images found; aborting')
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    num_classes = len(PPE_CLASSES)
    model = get_model(num_classes)
    model.to(device)

    # helper to collect trainable params depending on whether backbone is trainable
    def get_trainable_params(model, train_backbone=True):
        if train_backbone:
            return [p for p in model.parameters() if p.requires_grad]
        else:
            # exclude backbone parameters
            backbone_params = set([id(p) for p in model.backbone.parameters()]) if hasattr(model, 'backbone') else set()
            return [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_params]

    # initial decision whether to freeze backbone
    train_backbone = True if args.freeze_backbone_epochs <= 0 else False

    if not train_backbone and hasattr(model, 'backbone'):
        for p in model.backbone.parameters():
            p.requires_grad = False

    # optionally resume model weights
    start_epoch = 0
    if args.resume and args.resume_checkpoint:
        ckpt_path = Path(args.resume_checkpoint)
        if ckpt_path.exists():
            print(f'Loading checkpoint from {ckpt_path}')
            ckpt = torch.load(str(ckpt_path), map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            start_epoch = int(ckpt.get('epoch', 0))
            # allow explicit override from CLI when metadata is missing or you want to force a particular resume epoch
            if args.resume_start_epoch is not None:
                print(f'Overriding checkpoint epoch with --resume-start-epoch {args.resume_start_epoch}')
                start_epoch = int(args.resume_start_epoch)
            print(f'Resuming from checkpoint epoch {start_epoch}')
        else:
            print(f'Warning: resume checkpoint {ckpt_path} not found; starting from scratch')

    params = get_trainable_params(model, train_backbone=train_backbone)
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    scheduler = None
    if args.step_lr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_gamma)

    # Training loop (quick)
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    status_path = Path(args.output_dir) / 'training_status.json'
    progress_log = Path(args.output_dir) / 'training_progress.log'

    # start_epoch used when resuming; range is inclusive of start_epoch and exclusive of args.epochs
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    for epoch in range(start_epoch, args.epochs):
        avg_loss, loss_components = train_one_epoch(model, optimizer, train_loader, device)
        comp_str = ', '.join([f"{k}:{v:.4f}" for k, v in loss_components.items()])
        epoch_num = epoch + 1
        msg = f'Epoch {epoch_num}/{args.epochs}  avg loss: {avg_loss:.4f}  components: {comp_str}'
        print(msg)

        # save checkpoint per epoch
        torch.save({'epoch': epoch_num, 'model_state_dict': model.state_dict()}, args.output_model)
        print(f'Saved checkpoint to {args.output_model}')

        # write status JSON
        try:
            cur_lr = None
            if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
                cur_lr = float(optimizer.param_groups[0].get('lr', 0.0))
            status = {
                'epoch_completed': epoch_num,
                'total_epochs': args.epochs,
                'avg_loss': float(avg_loss),
                'loss_components': {k: float(v) for k, v in loss_components.items()},
                'lr': cur_lr
            }
            with open(status_path, 'w', encoding='utf-8') as sf:
                json.dump(status, sf, indent=2)
        except Exception:
            pass

        # append human-readable progress line
        try:
            with open(progress_log, 'a', encoding='utf-8') as pf:
                pf.write(msg + '\n')
        except Exception:
            pass

        # scheduler step
        if scheduler is not None:
            scheduler.step()

        # if we had frozen the backbone for initial epochs, unfreeze now and recreate optimizer
        if args.freeze_backbone_epochs > 0 and (epoch_num) == args.freeze_backbone_epochs:
            if hasattr(model, 'backbone'):
                print('Unfreezing backbone and recreating optimizer to fine-tune full model')
                for p in model.backbone.parameters():
                    p.requires_grad = True
                # recreate optimizer with full trainable params
                params = get_trainable_params(model, train_backbone=True)
                if args.optimizer == 'adamw':
                    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
                else:
                    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
                # recreate scheduler if requested
                if args.step_lr:
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_gamma)

    # Inference on test split
    run_inference(model, val_ds, device, args.output_dir)
    print('Inference complete, results in', args.output_dir)


if __name__ == '__main__':
    main()
