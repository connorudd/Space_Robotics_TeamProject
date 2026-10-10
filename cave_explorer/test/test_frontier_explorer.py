"""Tests for frontier_explorer. Run with:  python3 test_frontier_explorer.py   (or pytest)."""
import math
import sys
import time

import numpy as np

try:
    from cave_explorer.frontier_explorer import FrontierExplorer, UNKNOWN   # installed package
except ImportError:
    from frontier_explorer import FrontierExplorer, UNKNOWN                 # standalone

RES = 0.3


def empty(h, w, value=UNKNOWN):
    return np.full((h, w), value, dtype=np.int8)


def carve(grid, r0, r1, c0, c1, value=0):
    grid[r0:r1, c0:c1] = value


def corridor_map():
    """A 3 m wide corridor going east from the robot, open (unknown) at its far end, walled elsewhere.
    Returns grid, robot (x, y) in world coordinates with origin (0, 0)."""
    g = empty(30, 80)
    carve(g, 10, 20, 2, 60)          # free corridor rows 10-19, cols 2-59 (3 m wide, 17.4 m long)
    g[9, 1:61] = 100                 # north wall
    g[20, 1:61] = 100                # south wall
    g[9:21, 1] = 100                 # west wall (closed end); east end (col 60) is unknown
    robot = (5 * RES, 15 * RES)
    return g, robot


def test_frontier_mask_only_free_next_to_unknown():
    fe = FrontierExplorer()
    g, _ = corridor_map()
    m = fe.frontier_mask(g)
    rows, cols = np.nonzero(m)
    assert set(cols.tolist()) == {59}, 'only the open east end of the corridor is a frontier'
    assert set(rows.tolist()) == set(range(10, 20))


def test_corridor_goal_at_open_end():
    fe = FrontierExplorer()
    g, robot = corridor_map()
    best, frontiers, reason = fe.plan(g, RES, 0.0, 0.0, robot)
    assert best is not None, reason
    gx, gy, yaw = best.goal
    assert abs(gx - 59.5 * RES) < 1e-6 and abs(gy - 15.0 * RES) < 0.31, best.goal
    assert abs(yaw) < 0.1, 'faces east, into the unknown'
    assert 15.0 < best.path_distance < 17.0


def test_no_unknown_means_exploration_complete():
    fe = FrontierExplorer()
    g = empty(20, 20, 0)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 100
    best, frontiers, reason = fe.plan(g, RES, 0.0, 0.0, (3.0, 3.0))
    assert best is None and frontiers == [] and 'complete' in reason


def test_unknown_specks_are_not_frontiers():
    """Single unknown cells in mapped floor (gaps between lidar rays) must not become goals"""
    fe = FrontierExplorer(max_hole_cells=4)
    g = empty(20, 20, 0)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 100
    g[10, 10] = UNKNOWN
    g[5, 14:16] = UNKNOWN
    best, frontiers, reason = fe.plan(g, RES, 0.0, 0.0, (1.5, 1.5))
    assert best is None and frontiers == [], reason


def test_large_unknown_pocket_is_kept():
    fe = FrontierExplorer(max_hole_cells=4)
    g = empty(20, 20, 0)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 100
    g[8:12, 8:12] = UNKNOWN          # 16 cells: real unexplored space, not a speck
    best, frontiers, reason = fe.plan(g, RES, 0.0, 0.0, (1.5, 1.5))
    assert best is not None, reason


def test_small_clusters_ignored():
    """A frontier cluster smaller than min_cluster_size (a sliver of unknown at a wall) is not a goal"""
    fe = FrontierExplorer(min_cluster_size=5)
    g, robot = corridor_map()
    g[10:20, 60] = 100               # close the corridor end...
    g[10:13, 60] = UNKNOWN           # ...except a 3-cell sliver: 3 frontier cells
    best, frontiers, reason = fe.plan(g, RES, 0.0, 0.0, robot)
    assert best is None
    assert [f.status for f in frontiers] == ['too_small'], frontiers


def test_path_distance_beats_straight_line():
    """Frontier A is just behind a wall (close in a straight line, long way round by path).
    Frontier B is further in a straight line but on the robot's side of the wall.
    Nearest by straight line would pick A; nearest by path must pick B."""
    fe = FrontierExplorer()
    g = empty(60, 60)
    carve(g, 5, 55, 5, 55)                       # big free room
    g[4, 4:56] = g[55, 4:56] = 100
    g[4:56, 4] = g[4:56, 55] = 100
    g[10:55, 30] = 100                           # dividing wall, only gap at rows 5-9 (far end)
    g[38:50, 32:38] = UNKNOWN                    # A: unmapped pocket just east of the wall
    g[55, 6:12] = 0                              # B: 6-cell opening in the north wall, robot's side
    g[56:60, 6:12] = UNKNOWN
    robot = (27 * RES, 45 * RES)                 # west of the wall, next to it
    best, frontiers, reason = fe.plan(g, RES, 0.0, 0.0, robot)
    assert best is not None, reason
    usable = [f for f in frontiers if f.goal is not None]
    straight = min(usable, key=lambda f: math.hypot(f.goal[0] - robot[0], f.goal[1] - robot[1]))
    assert straight.goal[0] > 30 * RES, 'test set-up: A must be the nearest by straight line'
    assert best.goal[1] > 54 * RES, f'should choose the north opening B by path, chose {best.goal}'


def test_unreachable_frontier_not_chosen():
    fe = FrontierExplorer()
    g, robot = corridor_map()
    # A sealed pocket with its own frontier, not connected to the corridor
    carve(g, 24, 28, 30, 40)
    g[23, 29:41] = 100
    g[24:29, 29] = 100
    g[24:29, 40] = 100                           # pocket's south side (row 28/29) touches unknown
    best, frontiers, reason = fe.plan(g, RES, 0.0, 0.0, robot)
    statuses = sorted(f.status for f in frontiers)
    assert 'unreachable' in statuses, statuses
    assert best is not None and best.goal[0] > 50 * RES


def test_blacklist_skips_goal():
    fe = FrontierExplorer(blacklist_radius=2.0)
    g, robot = corridor_map()
    best, _, _ = fe.plan(g, RES, 0.0, 0.0, robot)
    best2, frontiers, reason = fe.plan(g, RES, 0.0, 0.0, robot, blacklist=[best.goal[:2]])
    assert best2 is None and frontiers[0].status == 'blacklisted', reason


def test_narrow_gap_not_traversable():
    """A 0.6 m gap is narrower than the robot (radius 0.5 m): the frontier beyond it is unreachable"""
    fe = FrontierExplorer(robot_radius=0.5)
    g, robot = corridor_map()
    g[10:20, 40] = 100
    g[15:17, 40] = 0                             # 2-cell (0.6 m) gap in a wall across the corridor
    best, frontiers, reason = fe.plan(g, RES, 0.0, 0.0, robot)
    assert best is None and 'no reachable' in reason, reason


def test_robot_on_noisy_cell_still_plans():
    fe = FrontierExplorer()
    g, robot = corridor_map()
    r, c = fe.world_to_cell(robot[0], robot[1], 0.0, 0.0, RES)
    g[r, c] = 100                                # a speck of map noise under the robot
    best, _, reason = fe.plan(g, RES, 0.0, 0.0, robot)
    assert best is not None, reason


def test_robot_outside_map():
    fe = FrontierExplorer()
    g, _ = corridor_map()
    best, _, reason = fe.plan(g, RES, 0.0, 0.0, (-50.0, -50.0))
    assert best is None and 'outside' in reason


def test_world_cell_round_trip():
    fe = FrontierExplorer
    for x, y in [(0.1, 0.1), (5.0, -3.2), (-7.45, 12.0)]:
        r, c = fe.world_to_cell(x, y, -10.0, -10.0, RES)
        wx, wy = fe.cell_to_world(r, c, -10.0, -10.0, RES)
        assert abs(wx - x) <= RES / 2 + 1e-9 and abs(wy - y) <= RES / 2 + 1e-9


def test_speed_on_real_sized_map():
    """216 x 171 cells is the size of the real SLAM map: planning must take well under a second"""
    fe = FrontierExplorer()
    g = empty(171, 216)
    carve(g, 20, 150, 20, 190)
    t0 = time.time()
    best, _, reason = fe.plan(g, RES, 0.0, 0.0, (30.0, 25.0))
    elapsed = time.time() - t0
    assert best is not None, reason
    assert elapsed < 2.0, f'{elapsed:.2f} s'
    print(f'      (planning on 216x171 took {elapsed:.2f} s)')


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
