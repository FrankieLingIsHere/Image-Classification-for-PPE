import json
import csv
import argparse
import math


def iou(a, b):
    # a and b: [x1,y1,x2,y2]
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--csv', required=False)
    parser.add_argument('--top', type=int, default=5)
    args = parser.parse_args()

    with open(args.results, 'r', encoding='utf-8') as f:
        data = json.load(f)

    per_class = {}
    # load class metrics if CSV provided
    if args.csv:
        with open(args.csv, 'r', encoding='utf-8') as cf:
            r = csv.DictReader(cf)
            for row in r:
                cls = row['class']
                ap = float(row['ap'])
                det_count = int(row.get('det_count', 0))
                per_class[cls] = {'ap': ap, 'det_count': det_count}
    else:
        # fallback: read from JSON summary
        for cls, metrics in data.get('per_class_metrics', {}).items():
            per_class[cls] = {'ap': float(metrics.get('ap', 0)), 'det_count': metrics.get('det_count', 0)}

    targets = [c for c,v in per_class.items() if v['ap'] == 0.0 and v['det_count'] > 0]

    if not targets:
        print('No classes with AP==0 and detections > 0 found.')
        return

    print('Classes to inspect (AP==0 && det_count>0):')
    for t in targets:
        print(' -', t)
    print('\nScanning per-image detections...')

    # map image -> gt list
    image_map = {}
    for item in data.get('detection_results', []):
        image = item['image']
        gts = item.get('ground_truth', [])
        dets = item.get('detections', [])
        image_map[image] = {'gts': gts, 'dets': dets}

    for cls in targets:
        print('\n=== CLASS:', cls, '===')
        det_entries = []
        for image, vals in image_map.items():
            gts = [g for g in vals['gts'] if g['class'] == cls]
            dets = [d for d in vals['dets'] if d['class'] == cls]
            for d in dets:
                best = 0.0
                best_gt = None
                for g in gts:
                    cur = iou(d['bbox'], g['bbox'])
                    if cur > best:
                        best = cur
                        best_gt = g
                det_entries.append({'image': image, 'bbox': d['bbox'], 'conf': d.get('confidence', 0.0), 'best_iou': best, 'has_gt': len(gts)>0})

        if not det_entries:
            print(' No detections for class in filtered results (det_count appears >0 but none after filtering).')
            continue

        # sort by best_iou desc then conf
        det_entries = sorted(det_entries, key=lambda x: (x['best_iou'], x['conf']), reverse=True)
        total = len(det_entries)
        mean_best = sum(e['best_iou'] for e in det_entries)/total if total>0 else 0.0
        num_ge_05 = sum(1 for e in det_entries if e['best_iou'] >= 0.5)
        num_ge_03 = sum(1 for e in det_entries if e['best_iou'] >= 0.3)
        print(f' Total detections inspected: {total}')
        print(f' Mean(best IoU to same-class GT): {mean_best:.3f}')
        print(f' Detections with IoU>=0.5: {num_ge_05} (TP at eval IoU 0.5)')
        print(f' Detections with IoU>=0.3: {num_ge_03}')

        print('\n Top examples (best_iou, conf, has_gt, image):')
        for e in det_entries[:args.top]:
            print(f"  - IoU={e['best_iou']:.3f}, conf={e['conf']:.3f}, has_gt={e['has_gt']}, image={e['image']}, bbox={e['bbox']}")

        # show some low IoU examples
        low = [e for e in det_entries if e['best_iou'] < 0.2]
        print(f' Low-IoU detections (<0.2): {len(low)}')
        for e in low[:min(5, len(low))]:
            print(f"   - IoU={e['best_iou']:.3f}, conf={e['conf']:.3f}, image={e['image']}")

if __name__ == '__main__':
    main()
