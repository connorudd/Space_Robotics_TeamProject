#!/usr/bin/env python3
"""Print the occupancy grid around a map-frame point: python3 probe_map.py X Y"""
import sys

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node


class MapProbe(Node):
    def __init__(self, x, y, half_width):
        super().__init__('map_probe')
        self.x_ = x
        self.y_ = y
        self.half_width_ = half_width
        self.done_ = False
        self.map_sub_ = self.create_subscription(OccupancyGrid, 'map', self.map_callback, 1)

    def map_callback(self, map_msg):
        """Print a window of cells centred on the probe point, +y up and +x right"""
        info = map_msg.info

        # World metres -> grid indices, same conversion Planning 1 will use
        col = int((self.x_ - info.origin.position.x) / info.resolution)
        row = int((self.y_ - info.origin.position.y) / info.resolution)
        print(f'Probe ({self.x_}, {self.y_}) -> col={col} row={row}, resolution {info.resolution:.2f} m')
        print('# occupied   . free   ? unknown   [x] probe cell')

        # Print rows from high y to low y so +y is up on screen
        for r in range(row + self.half_width_, row - self.half_width_ - 1, -1):
            line = ''
            for c in range(col - self.half_width_, col + self.half_width_ + 1):
                if 0 <= c < info.width and 0 <= r < info.height:
                    value = map_msg.data[r * info.width + c]
                    symbol = '?' if value < 0 else ('#' if value >= 65 else '.')
                else:
                    symbol = ' '
                line += f'[{symbol}]' if (r == row and c == col) else f' {symbol} '
            print(line)
        self.done_ = True


def main():
    rclpy.init()
    probe = MapProbe(float(sys.argv[1]), float(sys.argv[2]), 5)
    while rclpy.ok() and not probe.done_:
        rclpy.spin_once(probe)
    probe.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
