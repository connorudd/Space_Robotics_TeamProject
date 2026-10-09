"""ROS-free artefact detector simulator.

Stands in for the Perception node while it is not ready. It reports artefacts
from the known world positions, but only when a real camera could plausibly see
them (range, field of view, line of sight), with noisy positions that improve as
the robot gets closer, and with optional fault modes so the planner can be
tested against a misbehaving detector.

Nothing in here imports ROS, so it can be unit tested with plain Python.
"""
import math
import random

FAULT_MODES = ['none', 'dropout', 'noise', 'late', 'false_positive']

# [type, x, y, z] in the map frame (map == Gazebo world frame).
GROUND_TRUTH = [
    ['green_alien', -5.4, -0.3, 0.2],
    ['green_crystals', 8.2, 5.2, -0.2],
    ['white_sphere', 52.0, 5.4, 0.8],
    ['ice_formation', 33.7, 10.6, 0.6],
    ['mushrooms_blue', 20.0, 10.2, 0.8],
    ['green_alien', 19.1, 20.8, 0.2],
    ['green_crystals', -3.3, 21.4, -0.2],
    ['white_sphere', 2.9, 23.3, 0.8],
    ['ice_formation', 18.0, 30.2, 0.6],
    ['mushrooms_blue', -4.7, 30.2, 0.8],
    ['green_alien', 54.4, 32.7, 0.2],
    ['green_crystals', 19.8, 35.8, -0.4],
    ['white_sphere', 46.7, 35.9, 0.8],
    ['ice_formation', -4.0, 39.0, 0.6],
    ['mushrooms_blue', 33.5, 45.9, 0.8],
    # Stop signs are on walls; z is not measured, 1.0 is a guess.
    ['stop_sign', 23.35, 4.7, 1.0],
    ['stop_sign', 45.55, 18.2, 1.0],
    ['stop_sign', 29.25, 28.35, 1.0],
    ['stop_sign', 7.95, 39.35, 1.0],
    ['stop_sign', 52.7, 47.15, 1.0],
]

# Where the 'false_positive' mode invents a stop sign (open floor near the start).
FALSE_POSITIVE_XY = (12.0, 1.5)


def wrap_to_pi(a):
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def bresenham(c0, r0, c1, r1):
    """Grid cells (col, row) on the line from (c0, r0) to (c1, r1), inclusive."""
    cells = []
    dc, dr = abs(c1 - c0), abs(r1 - r0)
    sc = 1 if c0 < c1 else -1
    sr = 1 if r0 < r1 else -1
    err = dc - dr
    c, r = c0, r0
    while True:
        cells.append((c, r))
        if c == c1 and r == r1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
    return cells


def grid_line_blocked(data, width, height, origin_x, origin_y, resolution,
                      x0, y0, x1, y1, occupied_threshold=65, skip_end_cells=2):
    """True if an occupied cell lies between (x0, y0) and (x1, y1).

    Unknown cells (-1) do not block, because the map is only partly explored and
    we do not want to hide an artefact behind something we have not mapped. The
    last skip_end_cells cells are ignored, since artefacts sit next to walls and
    the cell under the artefact itself may be marked occupied.
    """
    c0 = int((x0 - origin_x) / resolution)
    r0 = int((y0 - origin_y) / resolution)
    c1 = int((x1 - origin_x) / resolution)
    r1 = int((y1 - origin_y) / resolution)
    cells = bresenham(c0, r0, c1, r1)
    if skip_end_cells > 0:
        cells = cells[:-skip_end_cells]
    for c, r in cells:
        if c < 0 or r < 0 or c >= width or r >= height:
            continue
        if data[r * width + c] >= occupied_threshold:
            return True
    return False


class ArtifactSimulator:
    def __init__(self, min_range=2.0, max_range=8.0, hfov=math.radians(120.0),
                 noise_base=0.1, noise_per_metre=0.05, fault_mode='none',
                 dropout_prob=0.5, late_delay=5.0, seed=0,
                 ground_truth=None, false_positive_xy=None):
        if fault_mode not in FAULT_MODES:
            raise ValueError(f"Unknown fault_mode '{fault_mode}', expected one of {FAULT_MODES}")
        self.min_range = min_range
        self.max_range = max_range
        self.hfov = hfov
        self.noise_base = noise_base
        self.noise_per_metre = noise_per_metre
        self.fault_mode = fault_mode
        self.dropout_prob = dropout_prob
        self.late_delay = late_delay
        self.rng = random.Random(seed)
        self.truth = [list(t) for t in (GROUND_TRUTH if ground_truth is None else ground_truth)]
        self.false_positive_xy = FALSE_POSITIVE_XY if false_positive_xy is None else false_positive_xy
        self.records = {}        # truth index (or -1 for the fake) -> record dict
        self.first_visible = {}  # truth index -> time first visible (for 'late')
        self.next_id = 0

    # ---- geometry -------------------------------------------------------
    def in_view(self, robot_x, robot_y, robot_yaw, x, y):
        """Range and field-of-view test (no line of sight). Returns (visible, range)."""
        dx, dy = x - robot_x, y - robot_y
        rng = math.hypot(dx, dy)
        if rng < self.min_range or rng > self.max_range:
            return False, rng
        bearing = wrap_to_pi(math.atan2(dy, dx) - robot_yaw)
        return abs(bearing) <= self.hfov / 2.0, rng

    # ---- measurement ----------------------------------------------------
    def _sigma(self, rng):
        s = self.noise_base + self.noise_per_metre * rng
        return s * 3.0 if self.fault_mode == 'noise' else s

    def add_observation(self, key, typ, x, y, z, sigma, now, truth_xy=None):
        """Fuse one noisy measurement into the record for key (inverse-variance mean)."""
        rec = self.records.get(key)
        w = 1.0 / (sigma * sigma)
        if rec is None:
            rec = {'id': self.next_id, 'type': typ, 'sum_w': 0.0,
                   'sx': 0.0, 'sy': 0.0, 'sz': 0.0, 'n_obs': 0, 'last_seen': now}
            self.next_id += 1
            self.records[key] = rec
        rec['sum_w'] += w
        rec['sx'] += w * x
        rec['sy'] += w * y
        rec['sz'] += w * z
        rec['n_obs'] += 1
        rec['last_seen'] = now
        return rec

    def observe(self, now, robot_x, robot_y, robot_yaw, is_blocked=None):
        """Process one camera frame. is_blocked(x0,y0,x1,y1) -> bool gives line of sight."""
        for i, (typ, tx, ty, tz) in enumerate(self.truth):
            visible, rng = self.in_view(robot_x, robot_y, robot_yaw, tx, ty)
            if not visible:
                continue
            if is_blocked is not None and is_blocked(robot_x, robot_y, tx, ty):
                continue
            if self.fault_mode == 'late' and i not in self.records:
                t0 = self.first_visible.setdefault(i, now)
                if now - t0 < self.late_delay:
                    continue
            if self.fault_mode == 'dropout' and self.rng.random() < self.dropout_prob:
                continue
            s = self._sigma(rng)
            self.add_observation(i, typ,
                                 tx + self.rng.gauss(0.0, s),
                                 ty + self.rng.gauss(0.0, s),
                                 tz + self.rng.gauss(0.0, s * 0.5),
                                 s, now)

        if self.fault_mode == 'false_positive' and -1 not in self.records:
            fx, fy = self.false_positive_xy
            visible, rng = self.in_view(robot_x, robot_y, robot_yaw, fx, fy)
            if visible and (is_blocked is None or not is_blocked(robot_x, robot_y, fx, fy)):
                # One-off ghost: reported once, never confirmed again (n_obs stays 1).
                self.add_observation(-1, 'stop_sign', fx, fy, 1.0, self._sigma(rng), now)

    def reports(self):
        """Current artefact list, sorted by id."""
        out = []
        for rec in self.records.values():
            w = rec['sum_w']
            out.append({'id': rec['id'], 'type': rec['type'],
                        'x': round(rec['sx'] / w, 3), 'y': round(rec['sy'] / w, 3),
                        'z': round(rec['sz'] / w, 3),
                        'n_obs': rec['n_obs'], 'last_seen': round(rec['last_seen'], 3)})
        return sorted(out, key=lambda r: r['id'])
