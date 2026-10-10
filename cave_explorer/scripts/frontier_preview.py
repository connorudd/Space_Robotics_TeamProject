#!/usr/bin/env python3
"""Run frontier detection on saved map snapshots and draw the result (no ROS needed).

  python3 frontier_preview.py ~/map_snapshots/map_*.npz

For each map_X.npz writes map_X_frontiers.png next to it:
  grey unknown, white free, black occupied, red robot,
  green = chosen frontier, blue = other candidates, orange = unreachable, purple = too small,
  yellow cross = goal of the chosen frontier
and prints one line per frontier.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cave_explorer'))
from frontier_explorer import FrontierExplorer  # noqa: E402

COLOURS = {'chosen': (0, 200, 0), 'candidate': (255, 120, 0), 'unreachable': (0, 140, 255),
           'too_small': (200, 0, 200), 'blacklisted': (0, 0, 120)}   # BGR


def preview(path, fe):
    m = np.load(path)
    grid, res = m['grid'], float(m['resolution'])
    ox, oy = float(m['origin_x']), float(m['origin_y'])
    robot = (float(m['robot_x']), float(m['robot_y']))
    best, frontiers, reason = fe.plan(grid, res, ox, oy, robot)

    img = np.full(grid.shape + (3,), 128, dtype=np.uint8)
    img[(grid >= 0) & (grid < fe.occupied_threshold)] = (255, 255, 255)
    img[grid >= fe.occupied_threshold] = (0, 0, 0)
    for f in frontiers:
        for r, c in f.cells:
            img[r, c] = COLOURS[f.status]
    scale = 3
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    def px(x, y):  # world -> pixel in the scaled image (before the vertical flip)
        r, c = FrontierExplorer.world_to_cell(x, y, ox, oy, res)
        return int((c + 0.5) * scale), int((r + 0.5) * scale)

    cv2.circle(img, px(*robot), 5, (0, 0, 255), -1)
    if best is not None:
        cv2.drawMarker(img, px(best.goal[0], best.goal[1]), (0, 220, 255), cv2.MARKER_CROSS, 14, 3)
    img = cv2.flip(img, 0)  # row 0 is the bottom of the map
    out = path.replace('.npz', '_frontiers.png')
    cv2.imwrite(out, img)

    print(f'{os.path.basename(path)}: robot ({robot[0]:.1f}, {robot[1]:.1f}) -> {reason}')
    for f in sorted(frontiers, key=lambda f: f.path_distance):
        goal = f'goal ({f.goal[0]:.1f}, {f.goal[1]:.1f})' if f.goal else 'no goal'
        print(f'   {f.status:12s} {f.size:4d} cells, {goal}, path {f.path_distance:.1f} m')
    print(f'   wrote {out}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    explorer = FrontierExplorer()
    for p in sys.argv[1:]:
        preview(p, explorer)
