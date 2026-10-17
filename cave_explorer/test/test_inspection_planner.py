"""Tests for inspection_planner. Run with:  python3 test_inspection_planner.py   (or pytest)."""
import math
import sys

import numpy as np

try:
    from cave_explorer.inspection_planner import InspectionPlanner          # installed package
except ImportError:
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from cave_explorer.inspection_planner import InspectionPlanner          # source tree

RES = 0.3


def room(h=40, w=40):
    """A walled, fully mapped room, origin (0, 0)"""
    g = np.zeros((h, w), dtype=np.int8)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 100
    return g


def test_standoff_faces_artefact_at_ring_distance():
    ip = InspectionPlanner(radii=(2.5,))
    g = room()
    target = (6.0, 6.0)
    s = ip.standoffs(g, RES, 0.0, 0.0, (2.0, 2.0), target)
    assert s, 'expected standoffs in an open room'
    x, y, yaw, d = s[0]
    assert abs(math.hypot(x - target[0], y - target[1]) - 2.5) < 1e-6
    assert abs(math.atan2(target[1] - y, target[0] - x) - yaw) < 1e-6
    # Best first: the chosen standoff is on the robot's side of the artefact
    assert math.hypot(x - 2.0, y - 2.0) < math.hypot(target[0] - 2.0, target[1] - 2.0)


def test_standoff_not_behind_wall():
    """Artefact against a wall: no standoff on the far side of the wall"""
    ip = InspectionPlanner(radii=(2.5,))
    g = room(40, 60)
    g[:, 30] = 100                         # full-height wall at x = 9.0..9.3
    target = (8.5, 6.0)                    # artefact just west of the wall
    for x, y, _, _ in ip.standoffs(g, RES, 0.0, 0.0, (2.0, 6.0), target):
        assert x < 9.0, f'standoff {x:.1f}, {y:.1f} is behind the wall'


def test_no_line_of_sight_through_pillar():
    ip = InspectionPlanner(radii=(2.5,), n_angles=4)   # candidates E, N, W, S of the target
    g = room()
    target = (6.0, 6.0)
    g[19:22, 11:13] = 100                  # pillar between the target and the west candidate (3.5, 6.0)
    xs = [round(s[0], 1) for s in ip.standoffs(g, RES, 0.0, 0.0, (6.0, 2.0), target)]
    assert 3.5 not in xs, xs


def test_failed_standoff_excluded():
    ip = InspectionPlanner(radii=(2.5,))
    g = room()
    first = ip.standoffs(g, RES, 0.0, 0.0, (2.0, 2.0), (6.0, 6.0))[0]
    second = ip.standoffs(g, RES, 0.0, 0.0, (2.0, 2.0), (6.0, 6.0), failed=[first[:2]])[0]
    assert math.hypot(first[0] - second[0], first[1] - second[1]) >= 1.0


def test_unreachable_artefact_has_no_standoff():
    ip = InspectionPlanner()
    g = room(40, 60)
    g[:, 30] = 100                         # wall splitting the room, no gap
    s = ip.standoffs(g, RES, 0.0, 0.0, (2.0, 6.0), (14.0, 6.0))   # artefact on the other side
    assert s == []


def test_robot_outside_map():
    assert InspectionPlanner().standoffs(room(), RES, 0.0, 0.0, (-20.0, -20.0), (6.0, 6.0)) == []


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print('ok  ', name)
            except Exception as e:  # noqa
                fails += 1
                print('FAIL', name, repr(e))
    print('failures:', fails)
    sys.exit(1 if fails else 0)
