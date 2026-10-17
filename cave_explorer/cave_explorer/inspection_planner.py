"""Standoff pose selection for close-range artefact inspection (ROS-free).

Given an artefact position, choose where the robot should stand to look at it:
candidate poses on rings around the artefact are kept only if they are
  - free in the map and at least robot_radius from any obstacle,
  - in line of sight of the artefact (no occupied cell on the straight line between them),
  - reachable from the robot (same path search as the frontier explorer),
  - not near a standoff that already failed,
and the reachable one with the shortest path wins. The pose faces the artefact so the
camera points at it.

The rings are deliberately not too close: Nav2 may not see a small artefact as an obstacle,
and the detector needs the artefact inside its working range to confirm it.
"""
import math

import numpy as np

from cave_explorer.frontier_explorer import FrontierExplorer


class InspectionPlanner:
    def __init__(self, radii=(2.5, 3.0, 2.0, 3.5), n_angles=24, robot_radius=0.5,
                 failed_radius=1.0, occupied_threshold=50):
        self.radii = radii                  # m, tried in order of preference
        self.n_angles = n_angles            # candidates per ring
        self.failed_radius = failed_radius  # m around a failed standoff that is skipped
        self.fe = FrontierExplorer(robot_radius=robot_radius, occupied_threshold=occupied_threshold)

    def line_of_sight(self, grid, res, ox, oy, x0, y0, x1, y1, skip_end=1.5):
        """True if no occupied cell lies on the segment, ignoring the last skip_end metres.
        The artefact's own body is mapped as occupied by the lidar, and its listed position can be
        up to ~1 m inside that outline (ice formations are ~1.5 m across), so it must not count
        as something blocking the view of itself."""
        length = math.hypot(x1 - x0, y1 - y0)
        steps = max(2, int(length / (res / 2.0)))
        h, w = grid.shape
        for i in range(steps + 1):
            t = i / steps
            if t * length > length - skip_end:
                break
            r, c = FrontierExplorer.world_to_cell(x0 + t * (x1 - x0), y0 + t * (y1 - y0), ox, oy, res)
            if 0 <= r < h and 0 <= c < w and grid[r, c] >= self.fe.occupied_threshold:
                return False
        return True

    def standoffs(self, grid, res, ox, oy, robot_xy, target_xy, failed=()):
        """All valid standoff poses as (x, y, yaw, path_distance), best first. Empty if none."""
        grid = self.fe.fill_small_holes(np.asarray(grid))
        traversable = self.fe.free_mask(grid) & (self.fe.clearance_m(grid, res) >= self.fe.robot_radius)
        start_cell = FrontierExplorer.world_to_cell(robot_xy[0], robot_xy[1], ox, oy, res)
        h, w = grid.shape
        if not (0 <= start_cell[0] < h and 0 <= start_cell[1] < w):
            return []
        start = self.fe.nearest_traversable(traversable, start_cell)
        if start is None:
            return []
        dist = FrontierExplorer.path_distances(traversable, start, res)

        tx, ty = target_xy
        out = []
        for ring, radius in enumerate(self.radii):
            for k in range(self.n_angles):
                a = 2.0 * math.pi * k / self.n_angles
                x, y = tx + radius * math.cos(a), ty + radius * math.sin(a)
                r, c = FrontierExplorer.world_to_cell(x, y, ox, oy, res)
                if not (0 <= r < h and 0 <= c < w) or not traversable[r, c] or not np.isfinite(dist[r, c]):
                    continue
                if any(math.hypot(x - fx, y - fy) < self.failed_radius for fx, fy in failed):
                    continue
                if not self.line_of_sight(grid, res, ox, oy, x, y, tx, ty):
                    continue
                yaw = math.atan2(ty - y, tx - x)   # face the artefact
                out.append((x, y, yaw, float(dist[r, c]), ring))
        # Preferred ring first, then shortest path; the ring index is dropped from the result
        out.sort(key=lambda s: (s[4], s[3]))
        return [s[:4] for s in out]
