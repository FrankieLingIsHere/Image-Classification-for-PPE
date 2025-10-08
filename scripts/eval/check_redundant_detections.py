"""Check redundant detections and mapping to ground truth for a single image

Moved to `scripts/eval/` for consistency with other evaluation helpers.
"""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    inter = interW * interH
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--json', required=True)
    p.add_argument('--annotation', required=True)
    p.add_argument('--iou_dup', type=float, default=0.5, help='IoU threshold to consider detections duplicates')
    p.add_argument('--iou_match', type=float, default=0.5, help='IoU threshold to match detection to GT')
    return p.parse_args()


def load_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Defensive parsing: handle missing nodes gracefully and provide clear errors
    size = root.find('size')
    if size is None:
        raise ValueError(f"Invalid annotation (missing <size>) in {xml_path}")

    width_node = size.find('width')
    height_node = size.find('height')
    if width_node is None or height_node is None or width_node.text is None or height_node.text is None:
        raise ValueError(f"Invalid annotation (missing width/height text) in {xml_path}")

    width = int(width_node.text)
    height = int(height_node.text)

    gts = []
    for obj in root.findall('object'):
        name_node = obj.find('name')
        bbox = obj.find('bndbox')
        if name_node is None or name_node.text is None or bbox is None:
            # skip malformed object entries but continue parsing others
            continue

        name = name_node.text

        xmin_node = bbox.find('xmin')
        ymin_node = bbox.find('ymin')
        xmax_node = bbox.find('xmax')
        ymax_node = bbox.find('ymax')
        if None in (xmin_node, ymin_node, xmax_node, ymax_node):
            continue

        try:
            x1 = float(xmin_node.text)
            y1 = float(ymin_node.text)
            x2 = float(xmax_node.text)
            y2 = float(ymax_node.text)
        except Exception:
            # skip malformed numeric fields
            continue

        gts.append({'class': name, 'bbox': [x1, y1, x2, y2]})

    return width, height, gts


def main():
    args = parse_args()
    jpath = Path(args.json)
    apath = Path(args.annotation)
    assert jpath.exists(), f"JSON not found: {jpath}"
    assert apath.exists(), f"Annotation not found: {apath}"

    with open(jpath, 'r') as f:
        data = json.load(f)

    width, height, gts = load_annotation(apath)

    dets = data.get('detections', [])

    for d in dets:
        bx = d['bbox']
        x1 = max(0.0, min(1.0, bx[0])) * width
        y1 = max(0.0, min(1.0, bx[1])) * height
        x2 = max(0.0, min(1.0, bx[2])) * width
        y2 = max(0.0, min(1.0, bx[3])) * height
        d['_abs_bbox'] = [x1, y1, x2, y2]

    duplicates = []
    n = len(dets)
    for i in range(n):
        for j in range(i+1, n):
            if dets[i]['class_name'] != dets[j]['class_name']:
                continue
            iou_val = iou(dets[i]['_abs_bbox'], dets[j]['_abs_bbox'])
            if iou_val >= args.iou_dup:
                duplicates.append((i, j, dets[i]['class_name'], iou_val, dets[i]['confidence'], dets[j]['confidence']))

    gt_matches = []
    for gi, gt in enumerate(gts):
        matches = []
        for di, d in enumerate(dets):
            iou_val = iou(gt['bbox'], d['_abs_bbox'])
            if iou_val >= args.iou_match:
                matches.append((di, d['class_name'], d['confidence'], iou_val))
        gt_matches.append((gi, gt['class'], len(matches), matches))

    print(f"Image: {jpath.name}")
    print(f"Image size (W x H): {width} x {height}")
    print(f"Total detections: {len(dets)}")

    from collections import Counter
    cnt = Counter([d['class_name'] for d in dets])
    print("Detections per class:")
    for cls, c in cnt.most_common():
        print(f"  {cls}: {c}")

    print('\nDuplicate detection pairs (same-class, IoU >= {:.2f}):'.format(args.iou_dup))
    for tup in duplicates:
        i,j,cls,iouv,c1,c2 = tup
        print(f"  Pair ({i},{j}) class={cls} iou={iouv:.3f} confs={c1:.3f},{c2:.3f}")

    print('\nGT matches (IoU >= {:.2f}):'.format(args.iou_match))
    for gi, gclass, mcount, matches in gt_matches:
        print(f"  GT[{gi}] class={gclass} matched_detections={mcount}")
        for di, cls, conf, iouv in matches:
            print(f"    - det[{di}] class={cls} conf={conf:.3f} iou={iouv:.3f}")

    dup_count = len(duplicates)
    multi_match_gt = sum(1 for _,_,c,_ in gt_matches if c > 1)
    print('\nSummary:')
    print(f"  Duplicate same-class pairs: {dup_count}")
    print(f"  GT boxes with >1 matching detections: {multi_match_gt} / {len(gts)}")


if __name__ == '__main__':
    main()
