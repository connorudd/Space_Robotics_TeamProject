#!/usr/bin/env python3
"""
build_yolo_dataset.py - turn the labelled folder-per-class dataset into a YOLO dataset.

    python3 build_yolo_dataset.py ~/cave_dataset ~/cave_yolo

Reads classes.txt and the .txt label files written by label_tool.py, and produces:

    cave_yolo/
        images/{train,val,test}/
        labels/{train,val,test}/   <- negatives get empty label files (YOLO "background" images)
        data.yaml                  <- point YOLO training at this
        preview/                   <- a few labelled images per class with boxes drawn: check these!
        dataset_summary.md         <- image and instance counts per split, for the report

Splitting: consecutive frames are near-identical, so a random split leaks training images
into val/test and inflates your scores. Instead, each folder is cut into blocks of consecutive
frames (by filename order) and whole blocks are assigned to train/val/test.
The output folder is deleted and rebuilt on every run; the source folders are never modified.
"""

import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2

IMAGE_EXTS = {'.png', '.jpg', '.jpeg'}
SPLITS = ('train', 'val', 'test')


def read_labels(label_path, num_classes):
    """Return a list of (class_id, xc, yc, w, h). Raises ValueError on a malformed line."""
    boxes = []
    for i, line in enumerate(label_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f'line {i}: expected 5 values, got {len(parts)}')
        class_id = int(parts[0])
        values = [float(v) for v in parts[1:]]
        if not 0 <= class_id < num_classes:
            raise ValueError(f'line {i}: class id {class_id} out of range')
        if not all(0.0 <= v <= 1.0 for v in values):
            raise ValueError(f'line {i}: coordinates not normalised to 0-1')
        boxes.append((class_id, *values))
    return boxes


def split_into_blocks(items, fractions, block_size, rng):
    """Assign whole blocks of consecutive items to train/val/test."""
    if not items:
        return {s: [] for s in SPLITS}

    # Shrink blocks for small folders so there are enough blocks (~7+) to hit the split fractions
    block_size = max(1, min(block_size, len(items) // 7))
    blocks = [items[i:i + block_size] for i in range(0, len(items), block_size)]
    rng.shuffle(blocks)

    n = len(blocks)
    n_val = max(1, round(fractions[1] * n)) if n >= 3 else 0
    n_test = max(1, round(fractions[2] * n)) if n >= 3 else 0
    n_train = n - n_val - n_test

    assigned = {'train': blocks[:n_train],
                'val': blocks[n_train:n_train + n_val],
                'test': blocks[n_train + n_val:]}
    return {s: [item for block in b for item in block] for s, b in assigned.items()}


def save_preview(image_path, boxes, classes, out_path):
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    for class_id, xc, yc, bw, bh in boxes:
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, classes[class_id], (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), image)


def markdown_table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', type=Path, help='labelled dataset folder (one folder per class)')
    parser.add_argument('out', type=Path, help='output folder for the YOLO dataset')
    parser.add_argument('--negatives', default='negatives', help='name of the negatives folder')
    parser.add_argument('--block-size', type=int, default=10,
                        help='consecutive frames kept together when splitting (default 10)')
    parser.add_argument('--split', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                        metavar=('TRAIN', 'VAL', 'TEST'))
    parser.add_argument('--seed', type=int, default=0, help='random seed, so splits are repeatable')
    parser.add_argument('--previews', type=int, default=3, help='preview images per class')
    args = parser.parse_args()

    root, out = args.root.expanduser(), args.out.expanduser()
    classes_file = root / 'classes.txt'
    if not classes_file.exists():
        raise SystemExit(f'{classes_file} not found - run label_tool.py first')
    classes = [line.strip() for line in classes_file.read_text().splitlines() if line.strip()]

    if out.exists():
        shutil.rmtree(out)
    for sub in ('images', 'labels'):
        for split in SPLITS:
            (out / sub / split).mkdir(parents=True)
    (out / 'preview').mkdir()

    rng = random.Random(args.seed)
    image_counts = defaultdict(Counter)      # folder -> split -> number of images
    instance_counts = defaultdict(Counter)   # class  -> split -> number of boxes
    background_counts = Counter()            # split  -> images with no boxes
    problems = []

    for folder in [root / c for c in classes] + [root / args.negatives]:
        if not folder.is_dir():
            problems.append(f'folder {folder.name}/ not found')
            continue
        is_negative = folder.name == args.negatives

        items = []
        for image_path in sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS):
            label_path = image_path.with_suffix('.txt')
            if is_negative:
                boxes = []
            elif not label_path.exists():
                problems.append(f'{folder.name}/{image_path.name}: not labelled yet - skipped')
                continue
            else:
                try:
                    boxes = read_labels(label_path, len(classes))
                except ValueError as e:
                    problems.append(f'{folder.name}/{image_path.name}: {e} - skipped')
                    continue
            items.append((image_path, boxes))

        for split, split_items in split_into_blocks(items, args.split, args.block_size, rng).items():
            for image_path, boxes in split_items:
                stem = f'{folder.name}_{image_path.stem}'.replace(' ', '_')  # no spaces in YOLO filenames
                shutil.copy2(image_path, out / 'images' / split / f'{stem}{image_path.suffix}')
                with open(out / 'labels' / split / f'{stem}.txt', 'w') as f:
                    for class_id, xc, yc, bw, bh in boxes:
                        f.write(f'{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n')

                image_counts[folder.name][split] += 1
                if boxes:
                    for box in boxes:
                        instance_counts[classes[box[0]]][split] += 1
                else:
                    background_counts[split] += 1

        if not is_negative:
            labelled = [item for item in items if item[1]]
            for image_path, boxes in rng.sample(labelled, min(args.previews, len(labelled))):
                save_preview(image_path, boxes, classes,
                             out / 'preview' / f'{folder.name}_{image_path.name}'.replace(' ', '_'))

    # data.yaml for YOLO training
    yaml_lines = [f'path: {out.resolve()}', 'train: images/train', 'val: images/val',
                  'test: images/test', 'names:']
    yaml_lines += [f'  {i}: {name}' for i, name in enumerate(classes)]
    (out / 'data.yaml').write_text('\n'.join(yaml_lines) + '\n')

    # Summary tables for the report
    folder_rows = []
    for name in classes + [args.negatives]:
        c = image_counts[name]
        folder_rows.append([name] + [c[s] for s in SPLITS] + [sum(c.values())])
    totals = [sum(r[i] for r in folder_rows) for i in range(1, 5)]
    folder_rows.append(['**Total**'] + totals)

    instance_rows = []
    for name in classes:
        c = instance_counts[name]
        instance_rows.append([name] + [c[s] for s in SPLITS] + [sum(c.values())])
    instance_rows.append(['background images (no boxes)'] +
                         [background_counts[s] for s in SPLITS] + [sum(background_counts.values())])

    summary = '\n\n'.join([
        '## Images per folder',
        markdown_table(['Folder', 'Train', 'Val', 'Test', 'Total'], folder_rows),
        '## Labelled instances (bounding boxes) per class',
        markdown_table(['Class', 'Train', 'Val', 'Test', 'Total'], instance_rows),
    ]) + '\n'
    (out / 'dataset_summary.md').write_text(summary)

    print(summary)
    if problems:
        print(f'{len(problems)} problem(s):')
        for p in problems:
            print(f'  - {p}')
    print(f'Dataset written to {out}. Check the images in {out / "preview"} before training.')


if __name__ == '__main__':
    main()
