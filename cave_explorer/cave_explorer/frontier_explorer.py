"""Frontier detection and goal selection on an occupancy grid (ROS-free).

A frontier cell is a free cell with at least one unknown neighbour: the boundary between what
the robot has mapped and what it has not. Frontier cells are grouped into 8-connected clusters
(one cluster ~ one opening into unexplored space), and each cluster gets one candidate goal.

The baseline strategy picks the cluster with the shortest path from the robot, measured by a
breadth-first search through traversable cells (free, and at least robot_radius from any
obstacle). Using path distance instead of straight-line distance matters in a cave: a frontier
on the other side of a wall can be 2 m away in a straight line and 40 m away by path. The same
search also tells us which frontiers are unreachable, so they are never sent to Nav2.

Only numpy and OpenCV (distance transform) are needed, so this can be unit tested and run on
saved map snapshots without ROS.
"""
import math
from collections import deque

import cv2
import numpy as np

UNKNOWN = -1
NEIGHBOURS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class Frontier:
    """One frontier cluster and the goal chosen for it"""

    def __init__(self, cells):
        self.cells = cells              # list of (row, col)
        self.size = len(cells)
        self.centroid = None            # (x, y) world
        self.goal = None                # (x, y, yaw) world, None if no usable goal
        self.path_distance = math.inf   # metres along the search, inf if unreachable
        self.status = 'candidate'       # candidate / too_small / unreachable / blacklisted / chosen

    def __repr__(self):
        return (f'Frontier(size={self.size}, status={self.status}, centroid={self.centroid}, '
                f'goal={self.goal}, path={self.path_distance:.1f})')


class FrontierExplorer:
    def __init__(self, robot_radius=0.5, min_cluster_size=5, blacklist_radius=2.0,
                 occupied_threshold=50, max_hole_cells=4):
        self.robot_radius = robot_radius            # m, matches the Nav2 robot_radius
        self.min_cluster_size = min_cluster_size    # cells; smaller clusters are usually map noise
        self.blacklist_radius = blacklist_radius    # m around a failed goal that is skipped
        self.occupied_threshold = occupied_threshold
        self.max_hole_cells = max_hole_cells        # unknown specks this small inside mapped space are noise

    def fill_small_holes(self, grid):
        """Treat tiny unknown specks (gaps between lidar rays, not unexplored space) as free.
        Without this, one unknown cell in open floor makes an 8-cell ring of 'frontier'."""
        unknown = (grid == UNKNOWN).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(unknown, connectivity=8)
        small = np.zeros(n, dtype=bool)
        small[1:] = stats[1:, cv2.CC_STAT_AREA] <= self.max_hole_cells   # label 0 is "not unknown"
        filled = grid.copy()
        filled[small[labels]] = 0
        return filled

    # ---------------------------------------------------------------- grid helpers
    @staticmethod
    def world_to_cell(x, y, origin_x, origin_y, resolution):
        return int(math.floor((y - origin_y) / resolution)), int(math.floor((x - origin_x) / resolution))

    @staticmethod
    def cell_to_world(row, col, origin_x, origin_y, resolution):
        return origin_x + (col + 0.5) * resolution, origin_y + (row + 0.5) * resolution

    def free_mask(self, grid):
        return (grid >= 0) & (grid < self.occupied_threshold)

    def frontier_mask(self, grid):
        """Free cells with at least one unknown 8-neighbour"""
        free = self.free_mask(grid)
        unknown = np.pad(grid == UNKNOWN, 1, constant_values=False)
        h, w = grid.shape
        near_unknown = np.zeros_like(free)
        for dr, dc in NEIGHBOURS_8:
            near_unknown |= unknown[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        return free & near_unknown

    def clearance_m(self, grid, resolution):
        """Distance (m) from each cell to the nearest occupied cell"""
        not_occupied = (grid < self.occupied_threshold).astype(np.uint8)  # unknown counts as not occupied
        return cv2.distanceTransform(not_occupied, cv2.DIST_L2, 5) * resolution

    @staticmethod
    def cluster(mask):
        """8-connected clusters of True cells, as lists of (row, col)"""
        h, w = mask.shape
        seen = np.zeros_like(mask)
        clusters = []
        for r0, c0 in zip(*np.nonzero(mask)):
            if seen[r0, c0]:
                continue
            seen[r0, c0] = True
            queue, cells = deque([(r0, c0)]), []
            while queue:
                r, c = queue.popleft()
                cells.append((int(r), int(c)))
                for dr, dc in NEIGHBOURS_8:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        queue.append((rr, cc))
            clusters.append(cells)
        return clusters

    @staticmethod
    def path_distances(traversable, start, resolution):
        """Breadth-first search over 8-connected traversable cells. Returns metres (inf = unreachable).
        Diagonal steps cost sqrt(2); it is a cheap approximation of the true path length."""
        h, w = traversable.shape
        dist = np.full((h, w), np.inf)
        dist[start] = 0.0
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            d = dist[r, c]
            for dr, dc in NEIGHBOURS_8:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and traversable[rr, cc]:
                    nd = d + (1.4142 if dr and dc else 1.0)
                    if nd < dist[rr, cc]:
                        dist[rr, cc] = nd
                        queue.append((rr, cc))
        return dist * resolution

    def nearest_traversable(self, traversable, cell, max_cells=5):
        """The robot's own cell may not count as traversable (map noise): use the nearest one that does"""
        r0, c0 = cell
        h, w = traversable.shape
        best = None
        for r in range(max(0, r0 - max_cells), min(h, r0 + max_cells + 1)):
            for c in range(max(0, c0 - max_cells), min(w, c0 + max_cells + 1)):
                if traversable[r, c]:
                    d = (r - r0) ** 2 + (c - c0) ** 2
                    if best is None or d < best[0]:
                        best = (d, (r, c))
        return None if best is None else best[1]

    # ---------------------------------------------------------------- main entry point
    def plan(self, grid, resolution, origin_x, origin_y, robot_xy, blacklist=()):
        """
        Find frontiers and choose the nearest reachable one.
        grid: 2-D int array, rows = y (OccupancyGrid data reshaped to (height, width)).
        Returns (chosen Frontier or None, list of every Frontier with its status, reason string).
        """
        grid = self.fill_small_holes(np.asarray(grid))
        traversable =self.free_mask(grid) & (self.clearance_m(grid, resolution) >= self.robot_radius)

        robot_cell = self.world_to_cell(robot_xy[0], robot_xy[1], origin_x, origin_y, resolution)
        if not (0 <= robot_cell[0] < grid.shape[0] and 0 <= robot_cell[1] < grid.shape[1]):
            return None, [], 'robot is outside the map'
        start = self.nearest_traversable(traversable, robot_cell)
        if start is None:
            return None, [], 'no traversable cell near the robot'
        dist = self.path_distances(traversable, start, resolution)

        frontiers = [Frontier(cells) for cells in self.cluster(self.frontier_mask(grid))]
        for f in frontiers:
            rows = np.array([rc[0] for rc in f.cells])
            cols = np.array([rc[1] for rc in f.cells])
            f.centroid = self.cell_to_world(rows.mean(), cols.mean(), origin_x, origin_y, resolution)
            if f.size < self.min_cluster_size:
                f.status = 'too_small'
                continue

            # Goal: the reachable cell of the cluster closest to its centroid (the middle of the
            # opening, so the robot sees as much of the unknown side as possible)
            reach = np.isfinite(dist[rows, cols])
            if not reach.any():
                f.status = 'unreachable'
                continue
            r_c, c_c = (rows.mean(), cols.mean())
            k = np.argmin(np.where(reach, (rows - r_c) ** 2 + (cols - c_c) ** 2, np.inf))
            gx, gy = self.cell_to_world(rows[k], cols[k], origin_x, origin_y, resolution)
            f.path_distance = float(dist[rows[k], cols[k]])
            # Face the direction of travel; the camera then looks into the unexplored side
            yaw = math.atan2(gy - robot_xy[1], gx - robot_xy[0])
            f.goal = (gx, gy, yaw)

            if any(math.hypot(gx - bx, gy - by) < self.blacklist_radius for bx, by in blacklist):
                f.status = 'blacklisted'

        candidates = [f for f in frontiers if f.status == 'candidate']
        if not candidates:
            if not frontiers:
                return None, frontiers, 'no frontiers left: exploration complete'
            return None, frontiers, 'no reachable, non-blacklisted frontier'
        best = min(candidates, key=lambda f: f.path_distance)
        best.status = 'chosen'
        return best, frontiers, f'{len(candidates)} candidate(s), nearest by path {best.path_distance:.1f} m'
