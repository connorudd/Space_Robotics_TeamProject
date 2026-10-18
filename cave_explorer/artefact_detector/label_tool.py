#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

import cv2

IMAGE_EXTS = {'.png', '.jpg', '.jpeg'}
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)


def draw_box(image, box, label, colour):
    x, y, w, h = box
    cv2.rectangle(image, (x, y), (x + w, y + h), colour, 2)
    cv2.putText(image, label, (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, colour, 1, cv2.LINE_AA)


def draw_banner(image, text):
    cv2.rectangle(image, (0, 0), (image.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(image, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)


def label_image(window, image, class_id, classes):
    """Collect boxes for one image. Returns (boxes, final_key)."""
    while True:
        boxes = []
        display = image.copy()

        while True:
            x, y, w, h = cv2.selectROI(window, display, showCrosshair=False)
            if w == 0 or h == 0:
                break  # c / ESC pressed: no more boxes on this image

            # Ask which class this box is (default: the folder's class)
            pending = display.copy()
            draw_box(pending, (x, y, w, h), '?', YELLOW)
            draw_banner(pending, f'ENTER={classes[class_id]}  ' +
                        '  '.join(f'{i}={c}' for i, c in enumerate(classes)))
            cv2.imshow(window, pending)
            key = cv2.waitKey(0) & 0xFF

            box_class = class_id
            if ord('0') <= key <= ord('9') and key - ord('0') < len(classes):
                box_class = key - ord('0')

            boxes.append((box_class, x, y, w, h))
            draw_box(display, (x, y, w, h), classes[box_class], GREEN)

        summary = display.copy()
        draw_banner(summary, f'{len(boxes)} box(es) | any key: next  u: redo  x: reject  q: save+quit')
        cv2.imshow(window, summary)
        key = cv2.waitKey(0) & 0xFF
        if key != ord('u'):
            return boxes, key


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', type=Path, help='dataset folder containing one folder per class')
    parser.add_argument('--negatives', default='negatives', help='name of the negatives folder')
    parser.add_argument('--relabel', action='store_true', help='redo images that already have labels')
    args = parser.parse_args()

    root = args.root.expanduser()
    folder_names = sorted(d.name for d in root.iterdir()
                          if d.is_dir() and d.name != args.negatives and not d.name.startswith('_'))
    if not folder_names:
        raise SystemExit(f'No class folders found in {root}')

    # Keep the class IDs already used in existing labels: classes from classes.txt keep their
    # position, and any new folders (e.g. a newly added artefact) are appended at the end.
    classes_file = root / 'classes.txt'
    existing = []
    if classes_file.exists():
        existing = [c.strip() for c in classes_file.read_text().splitlines() if c.strip()]
    missing = [c for c in existing if c not in folder_names]
    if missing:
        raise SystemExit(f'classes.txt lists {missing} but there is no folder for them. Restore the '
                         f'folder(s), or delete classes.txt and ALL label .txt files to start over.')
    new = [c for c in folder_names if c not in existing]
    classes = existing + new
    class_dirs = [root / c for c in classes]
    if existing and new:
        print(f'New class(es) {new} added after the existing ones - existing labels are unchanged.')

    (root / 'classes.txt').write_text('\n'.join(classes) + '\n')
    print('Class IDs (saved to classes.txt):')
    for i, name in enumerate(classes):
        print(f'  {i}: {name}')

    window = 'label_tool'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    for class_id, folder in enumerate(class_dirs):
        images = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        todo = [f for f in images if args.relabel or not f.with_suffix('.txt').exists()]
        print(f'{folder.name}: {len(images) - len(todo)} already labelled, {len(todo)} to go')

        for n, image_path in enumerate(todo, 1):
            image = cv2.imread(str(image_path))
            if image is None:
                print(f'  could not read {image_path.name}, skipping')
                continue

            cv2.setWindowTitle(window, f'{folder.name}  [{n}/{len(todo)}]  {image_path.name}')
            boxes, key = label_image(window, image, class_id, classes)
            label_path = image_path.with_suffix('.txt')

            if key == ord('x'):
                rejected = root / '_rejected'
                rejected.mkdir(exist_ok=True)
                shutil.move(str(image_path), str(rejected / f'{folder.name}_{image_path.name}'))
                label_path.unlink(missing_ok=True)
                print(f'  rejected {image_path.name}')
                continue

            h, w = image.shape[:2]
            with open(label_path, 'w') as f:
                for c, x, y, bw, bh in boxes:
                    f.write(f'{c} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} '
                            f'{bw / w:.6f} {bh / h:.6f}\n')

            if key == ord('q'):
                cv2.destroyAllWindows()
                print('Progress saved. Run again to continue.')
                return

    cv2.destroyAllWindows()
    print('All images labelled.')


if __name__ == '__main__':
    main()
