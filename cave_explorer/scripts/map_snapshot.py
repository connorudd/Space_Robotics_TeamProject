#!/usr/bin/env python3
"""Save the SLAM map (and the robot pose) to disk for offline planning development.

Each snapshot is written as:
  map_<sim seconds>.npz  grid (int8, rows = y), resolution, origin_x, origin_y, robot_x, robot_y, robot_yaw, stamp
  map_<sim seconds>.png  quick look: unknown grey, free white, occupied black, robot red

Usage (with the simulation, SLAM and Nav2 running):
  python3 map_snapshot.py                     # one snapshot into ~/map_snapshots
  python3 map_snapshot.py --count 6 --interval 60   # six snapshots, one per minute of sim time
"""
import argparse
import math
import os

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


class MapSnapshot(Node):
    def __init__(self, out_dir, count, interval):
        super().__init__('map_snapshot', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.out_dir_ = out_dir
        self.remaining_ = count
        self.interval_ = interval
        self.last_save_ = None
        self.tf_buffer_ = Buffer()
        self.tf_listener_ = TransformListener(self.tf_buffer_, self)
        # Volatile + reliable works with either durability on the publisher side; slam_toolbox
        # republishes the map every few seconds, so the next one arrives soon enough.
        self.create_subscription(OccupancyGrid, 'map', self.map_callback, 1)
        self.get_logger().info(f'Waiting for /map, saving {count} snapshot(s) to {out_dir}')

    def robot_pose(self):
        try:
            t = self.tf_buffer_.lookup_transform('map', 'base_link', Time())
        except TransformException:
            return None
        q = t.transform.rotation
        return t.transform.translation.x, t.transform.translation.y, 2.0 * math.atan2(q.z, q.w)

    def map_callback(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_save_ is not None and stamp - self.last_save_ < self.interval_:
            return

        pose = self.robot_pose()
        if pose is None:
            self.get_logger().warn('No map -> base_link transform yet, skipping this map',
                                   throttle_duration_sec=2.0)
            return

        info = msg.info
        grid = np.array(msg.data, dtype=np.int8).reshape(info.height, info.width)
        name = os.path.join(self.out_dir_, f'map_{stamp:08.1f}')
        np.savez_compressed(name + '.npz', grid=grid, resolution=info.resolution,
                            origin_x=info.origin.position.x, origin_y=info.origin.position.y,
                            robot_x=pose[0], robot_y=pose[1], robot_yaw=pose[2], stamp=stamp)
        cv2.imwrite(name + '.png', self.render(grid, info, pose))

        known = np.count_nonzero(grid >= 0) * info.resolution ** 2
        self.get_logger().info(f'Saved {name}.npz: {info.width}x{info.height} cells, '
                               f'{known:.0f} m^2 known, robot at ({pose[0]:.1f}, {pose[1]:.1f})')
        self.last_save_ = stamp
        self.remaining_ -= 1

    @staticmethod
    def render(grid, info, pose):
        img = np.full(grid.shape + (3,), 128, dtype=np.uint8)   # unknown: grey
        img[(grid >= 0) & (grid < 50)] = (255, 255, 255)        # free: white
        img[grid >= 50] = (0, 0, 0)                             # occupied: black
        col = int((pose[0] - info.origin.position.x) / info.resolution)
        row = int((pose[1] - info.origin.position.y) / info.resolution)
        cv2.circle(img, (col, row), 3, (0, 0, 255), -1)         # robot: red (BGR)
        img = cv2.flip(img, 0)                                  # row 0 is the bottom of the map
        return cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default=os.path.expanduser('~/map_snapshots'), help='output folder')
    parser.add_argument('--count', type=int, default=1, help='number of snapshots to save')
    parser.add_argument('--interval', type=float, default=60.0, help='sim seconds between snapshots')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rclpy.init()
    node = MapSnapshot(args.out, args.count, args.interval)
    try:
        while rclpy.ok() and node.remaining_ > 0:
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
