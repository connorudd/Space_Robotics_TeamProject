#!/usr/bin/env python3
"""Stand-in for the Perception artefact detector.

Publishes std_msgs/String JSON on /artifacts at ~2 Hz: a list of
{id, type, x, y, z, n_obs, last_seen} in the map frame. All the detection logic
lives in artifact_sim.py (ROS-free, unit tested); this node only feeds it the
robot pose and the map, and publishes the result.
"""
import json
import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener, TransformException

from cave_explorer.artifact_sim import ArtifactSimulator, FAULT_MODES, grid_line_blocked

TYPE_COLOURS = {
    'stop_sign': (1.0, 0.0, 0.0),
    'green_alien': (0.2, 1.0, 0.2),
    'green_crystals': (0.0, 0.8, 0.5),
    'white_sphere': (1.0, 1.0, 1.0),
    'ice_formation': (0.4, 0.8, 1.0),
    'mushrooms_blue': (0.2, 0.2, 1.0),
}


class ArtifactStub(Node):
    def __init__(self):
        super().__init__('artifact_stub_node')

        self.declare_parameter('fault_mode', 'none')
        self.declare_parameter('min_range', 2.0)
        self.declare_parameter('max_range', 8.0)
        self.declare_parameter('hfov_deg', 120.0)
        self.declare_parameter('noise_base', 0.1)
        self.declare_parameter('noise_per_metre', 0.05)
        self.declare_parameter('dropout_prob', 0.5)
        self.declare_parameter('late_delay', 5.0)
        self.declare_parameter('seed', 0)
        self.declare_parameter('use_line_of_sight', True)
        self.declare_parameter('observe_period', 1.0 / 3.0)   # matches the 3 Hz camera
        self.declare_parameter('publish_period', 0.5)

        p = self.get_parameter
        fault_mode = p('fault_mode').value
        if fault_mode not in FAULT_MODES:
            raise ValueError(f"fault_mode '{fault_mode}' is not one of {FAULT_MODES}")
        self.sim_ = ArtifactSimulator(
            min_range=p('min_range').value, max_range=p('max_range').value,
            hfov=math.radians(p('hfov_deg').value),
            noise_base=p('noise_base').value, noise_per_metre=p('noise_per_metre').value,
            fault_mode=fault_mode, dropout_prob=p('dropout_prob').value,
            late_delay=p('late_delay').value, seed=p('seed').value)
        self.use_los_ = p('use_line_of_sight').value

        self.map_ = None
        self.tf_buffer_ = Buffer()
        self.tf_listener_ = TransformListener(self.tf_buffer_, self)
        self.logged_ids_ = set()

        self.create_subscription(OccupancyGrid, 'map', self.map_callback, 1)
        self.pub_ = self.create_publisher(String, 'artifacts', 1)
        self.marker_pub_ = self.create_publisher(MarkerArray, 'marker_array_stub', 1)

        # Timers run on the node clock, which is sim time when use_sim_time is true.
        self.create_timer(p('observe_period').value, self.observe_callback)
        self.create_timer(p('publish_period').value, self.publish_callback)
        self.get_logger().info(f'Artifact stub running, fault_mode={fault_mode}')

    def map_callback(self, msg):
        self.map_ = msg

    def is_blocked(self, x0, y0, x1, y1):
        m = self.map_
        if not self.use_los_ or m is None:
            return False   # no map yet: do not hide anything
        return grid_line_blocked(m.data, m.info.width, m.info.height,
                                 m.info.origin.position.x, m.info.origin.position.y,
                                 m.info.resolution, x0, y0, x1, y1)

    def observe_callback(self):
        try:
            t = self.tf_buffer_.lookup_transform('map', 'base_link', Time(),
                                                 timeout=Duration(seconds=0.0))
        except TransformException:
            self.get_logger().warn('Waiting for the map -> base_link transform',
                                   throttle_duration_sec=5.0)
            return
        q = t.transform.rotation
        yaw = 2.0 * math.atan2(q.z, q.w)   # planar robot: only z and w are non-zero
        now = self.get_clock().now().nanoseconds * 1e-9
        self.sim_.observe(now, t.transform.translation.x, t.transform.translation.y,
                          yaw, self.is_blocked)

    def publish_callback(self):
        reports = self.sim_.reports()
        msg = String()
        msg.data = json.dumps(reports)
        self.pub_.publish(msg)

        # Log every new id, not just the newest: two can appear in the same publish period.
        for r in reports:
            if r['id'] not in self.logged_ids_:
                self.logged_ids_.add(r['id'])
                self.get_logger().info(
                    f"New artefact id {r['id']} {r['type']} at ({r['x']:.1f}, {r['y']:.1f}), "
                    f"{len(reports)} reported")

        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        stamp = self.get_clock().now().to_msg()
        for r in reports:
            colour = TYPE_COLOURS.get(r['type'], (1.0, 0.5, 0.0))
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = 'stub_artifacts'
            m.id = r['id']
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = r['x'], r['y'], r['z']
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.6
            m.color.r, m.color.g, m.color.b = colour
            m.color.a = 0.8
            arr.markers.append(m)

            txt = Marker()
            txt.header = m.header
            txt.ns = 'stub_labels'
            txt.id = r['id']
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x, txt.pose.position.y = r['x'], r['y']
            txt.pose.position.z = r['z'] + 0.8
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.5
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            txt.text = f"{r['id']}: {r['type']} (n={r['n_obs']})"
            arr.markers.append(txt)
        self.marker_pub_.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = ArtifactStub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
