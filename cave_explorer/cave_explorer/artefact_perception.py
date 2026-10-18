#!/usr/bin/env python3

import csv
import math
import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Int32, Int32MultiArray
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray

try:
    from cave_explorer.degradations import DEGRADATIONS, degrade, enhance
except ImportError:
    try:
        from degradations import DEGRADATIONS, degrade, enhance
    except ImportError:
        DEGRADATIONS, degrade, enhance = (), None, None

def depth_in_box(depth_m, box, shrink=0.4, min_valid_fraction=0.1, max_depth=8.0):
    """
    Robust distance to the object inside a bounding box.
    Uses the median depth of the central part of the box (so background pixels around the
    edges of the box are ignored). Returns None if there aren't enough valid depth pixels.
    depth_m: HxW float array in metres. box: (x1, y1, x2, y2) in depth image pixels.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx1 = int(x1 + w * (1 - shrink) / 2)
    cx2 = int(x2 - w * (1 - shrink) / 2)
    cy1 = int(y1 + h * (1 - shrink) / 2)
    cy2 = int(y2 - h * (1 - shrink) / 2)
    patch = depth_m[max(cy1, 0):max(cy2, cy1 + 1), max(cx1, 0):max(cx2, cx1 + 1)]
    if patch.size == 0:
        return None
    valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < max_depth)]
    if valid.size < max(1, min_valid_fraction * patch.size):
        return None
    return float(np.median(valid))


def detection_to_map(u, depth, fx, cx, robot_x, robot_y, robot_theta,
                     camera_x_offset=0.0, camera_y_offset=0.0):
    """
    Convert a detection to a 2D map position.
    u: horizontal pixel of the box centre. depth: distance along the camera's forward axis (m).
    fx, cx: camera intrinsics (pixels). The camera is assumed to face forward along the robot's x axis,
    mounted camera_x_offset ahead of / camera_y_offset left of base_link.
    """
    forward = depth                      # metres in front of the camera
    left = -(u - cx) * depth / fx        # pixels right of centre -> negative (to the robot's right)
    bx = camera_x_offset + forward       # point in the robot (base_link) frame
    by = camera_y_offset + left
    c, s = math.cos(robot_theta), math.sin(robot_theta)
    return robot_x + c * bx - s * by, robot_y + s * bx + c * by


class ArtefactMap:
    """
    Combines repeated detections into one estimate per artefact.
    Each observation is assigned to the nearest existing artefact within merge_radius (any class,
    so an occasional misclassification doesn't create a duplicate); otherwise it starts a new one.
    Position = median of all observations (robust to the odd bad depth reading).
    Class = majority vote. An artefact is only 'confirmed' after min_observations sightings,
    which filters out one-off false positives.
    """

    def __init__(self, merge_radius=2.0, min_observations=3, max_obs_per_artefact=300):
        self.merge_radius = merge_radius
        self.min_observations = min_observations
        self.max_obs = max_obs_per_artefact
        self.artefacts = [] 

    @staticmethod
    def _position(a):
        pts = np.array(a['obs'])
        return float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))

    @staticmethod
    def _label(a):
        return max(a['votes'], key=a['votes'].get)

    def add(self, x, y, label):
        """Add one observation. Returns (artefact, newly_confirmed)."""
        best, best_d = None, self.merge_radius
        for a in self.artefacts:
            ax, ay = self._position(a)
            d = math.hypot(x - ax, y - ay)
            if d < best_d:
                best, best_d = a, d
        if best is None:
            best = {'obs': [], 'votes': {}, 'count': 0}
            self.artefacts.append(best)

        was_confirmed = best['count'] >= self.min_observations
        best['obs'].append((x, y))
        best['obs'] = best['obs'][-self.max_obs:]
        best['votes'][label] = best['votes'].get(label, 0) + 1
        best['count'] += 1
        self._merge_close(best)
        return best, (not was_confirmed and best['count'] >= self.min_observations)

    def _merge_close(self, a):
        """If an artefact's estimate has drifted onto another one, merge them."""
        ax, ay = self._position(a)
        for other in list(self.artefacts):
            if other is a:
                continue
            ox, oy = self._position(other)
            if math.hypot(ax - ox, ay - oy) < self.merge_radius:
                a['obs'] = (a['obs'] + other['obs'])[-self.max_obs:]
                for k, v in other['votes'].items():
                    a['votes'][k] = a['votes'].get(k, 0) + v
                a['count'] += other['count']
                self.artefacts.remove(other)

    def confirmed(self):
        """List of (label, x, y, num_observations) for confirmed artefacts."""
        out = []
        for a in self.artefacts:
            if a['count'] >= self.min_observations:
                x, y = self._position(a)
                out.append((self._label(a), x, y, a['count']))
        return out

    def all_observations(self):
        return [p for a in self.artefacts for p in a['obs']]


# RViz colours (r, g, b) per class; unknown classes get white
CLASS_COLOURS = {
    'green_alien': (0.2, 1.0, 0.2),
    'green_crystals': (0.0, 0.6, 0.3),
    'ice_castles': (0.7, 0.9, 1.0),
    'mushrooms_blue': (0.2, 0.4, 1.0),
    'white_sphere': (1.0, 1.0, 1.0),
    'stop_sign': (1.0, 0.1, 0.1),
}

# Fixed integer ID for each artefact type, published on artefact_id / artefact_location_ids.
# Keyed by class NAME, so the IDs stay the same even if the YOLO class order changes after retraining.
# Share this table with whoever uses the topics (e.g. Planning 2). Unknown classes get -1.
ARTEFACT_IDS = {
    'green_alien': 1,
    'green_crystals': 2,
    'ice_castles': 3,
    'mushrooms_blue': 4,
    'white_sphere': 5,
    'stop_sign': 6,
}

def artefact_id(label):
    return ARTEFACT_IDS.get(label, -1)

class ArtefactPerception(Node):
    def __init__(self):
        super().__init__('artefact_perception_node')

        self.cv_bridge_ = CvBridge()
        self.tf_buffer_ = Buffer()
        self.tf_listener_ = TransformListener(self.tf_buffer_, self)

        # Topics
        self.declare_parameter('image_topic', 'camera/image')
        self.declare_parameter('depth_topic', 'camera/depth_image')
        self.declare_parameter('camera_info_topic', 'camera/camera_info')

        # Perception 1: dataset collection. Toggle while running:
        #   ros2 param set /artefact_perception_node save_images true
        self.declare_parameter('save_images', False)
        self.declare_parameter('dataset_dir', os.path.expanduser('~/cave_dataset'))
        self.dataset_dir_ = os.path.expanduser(self.get_parameter('dataset_dir').value)
        os.makedirs(os.path.join(self.dataset_dir_, 'images'), exist_ok=True)
        self.last_save_time_ = 0.0
        self.last_save_pose_ = None
        self.save_count_ = len(os.listdir(os.path.join(self.dataset_dir_, 'images')))  # don't overwrite

        # Perception 2: YOLO detector
        self.declare_parameter('yolo_model_path', os.path.expanduser(
            '~/ros2_ws/src/cave_explorer/artefact_detector/best.pt'))
        self.declare_parameter('detection_confidence', 0.5)
        model_path = os.path.expanduser(self.get_parameter('yolo_model_path').value)
        self.get_logger().info(f'Loading YOLO model from {model_path}')
        self.yolo_model_ = YOLO(model_path)
        self.get_logger().info(f'YOLO classes: {self.yolo_model_.names}')

        # Perception 3: localisation
        self.declare_parameter('camera_hfov', 1.047)          # rad, only used until camera_info arrives
        self.declare_parameter('camera_x_offset', 0.0)        # m, camera position ahead of base_link
        self.declare_parameter('camera_y_offset', 0.0)        # m, camera position left of base_link
        self.declare_parameter('localisation_min_confidence', 0.6)
        self.declare_parameter('max_depth', 8.0)              # m, ignore far (noisy) depth readings
        self.declare_parameter('merge_radius', 2.0)           # m, observations closer than this = same artefact
        self.declare_parameter('min_observations', 3)         # sightings needed before showing a marker
        self.declare_parameter('show_observations', True)     # also show every raw estimate as small dots
        self.artefact_map_ = ArtefactMap(self.get_parameter('merge_radius').value,
                                         self.get_parameter('min_observations').value)
        self.latest_depth_ = None
        self.latest_depth_stamp_ = None
        self.camera_fx_ = None
        self.camera_cx_ = None
        self.camera_info_width_ = None

        # Publishers
        self.detections_image_pub_ = self.create_publisher(Image, 'detections_image', 1)
        self.marker_pub_ = self.create_publisher(MarkerArray, 'marker_array_artifacts', 10)
        self.detected_pub_ = self.create_publisher(Bool, 'artefact_detected', 10)

        # Advanced 1: robust perception. Change while running, e.g.
        #   ros2 param set /artefact_perception_node degradation dust
        #   ros2 param set /artefact_perception_node degradation_level 2
        #   ros2 param set /artefact_perception_node enhance true
        self.declare_parameter('degradation', 'none')   # none, dust, low_light, motion_blur, low_res, dirty_lens
        self.declare_parameter('degradation_level', 0)  # 0 (off) to 3 (severe)
        self.declare_parameter('enhance', False)        # preprocessing before detection
        self.degraded_image_pub_ = self.create_publisher(Image, 'degraded_image', 1)
        self.enhanced_image_pub_ = self.create_publisher(Image, 'enhanced_image', 1)
        self.degradation_rng_ = np.random.default_rng()
        self.stats_setting_ = None
        self.locations_pub_ = self.create_publisher(PoseArray, 'artefact_locations', 10)
        self.id_pub_ = self.create_publisher(Int32, 'artefact_id', 10)
        self.location_ids_pub_ = self.create_publisher(Int32MultiArray, 'artefact_location_ids', 10)
        unknown = [n for n in self.yolo_model_.names.values() if n not in ARTEFACT_IDS]
        if unknown:
            self.get_logger().warn(f'No integer ID for classes {unknown} (published as -1) - add them to ARTEFACT_IDS')

        # Subscribers. Sensor-data QoS (best effort) connects to both reliable and best-effort
        # publishers; a default (reliable) subscriber silently receives nothing from a best-effort
        # publisher, which is a common reason a depth topic "doesn't arrive".
        image_topic = self.get_parameter('image_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        self.image_sub_ = self.create_subscription(
            Image, image_topic, self.image_callback, qos_profile_sensor_data)
        self.depth_sub_ = self.create_subscription(
            Image, depth_topic, self.depth_callback, qos_profile_sensor_data)
        self.camera_info_sub_ = self.create_subscription(
            CameraInfo, info_topic, self.camera_info_callback, qos_profile_sensor_data)
        self.get_logger().info(f'Subscribed to image: {image_topic}, depth: {depth_topic}, '
                               f'camera_info: {info_topic}')

        # Re-publish markers and locations every second so RViz / late subscribers always have them
        self.publish_timer_ = self.create_timer(1.0, self.publish_artefacts)

        self.get_logger().info('Artefact perception running')

    #######################################################
    # Robot pose

    def get_pose_2d(self, stamp=None):
        """Robot (x, y, theta) in the map frame, at the given time if available, else the latest."""
        times = [rclpy.time.Time.from_msg(stamp)] if stamp is not None else []
        times.append(rclpy.time.Time())
        for t in times:
            try:
                tf = self.tf_buffer_.lookup_transform('map', 'base_link', t)
                q = tf.transform.rotation
                theta = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                return tf.transform.translation.x, tf.transform.translation.y, theta
            except TransformException:
                continue
        self.get_logger().warn('No map -> base_link transform yet (is SLAM running?)',
                               throttle_duration_sec=5.0)
        return None

    #######################################################
    # Callbacks

    def depth_callback(self, depth_msg):
        """Keep the latest depth image, converted to metres"""
        depth = self.cv_bridge_.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0   # 16-bit depth is in millimetres
        self.latest_depth_ = np.asarray(depth, dtype=np.float32)
        if self.latest_depth_stamp_ is None:
            valid = self.latest_depth_[np.isfinite(self.latest_depth_) & (self.latest_depth_ > 0)]
            rng = f'{valid.min():.2f}-{valid.max():.2f}' if valid.size else 'no valid pixels'
            self.get_logger().info(
                f'First depth image received: {depth_msg.width}x{depth_msg.height}, '
                f'encoding {depth_msg.encoding}, depth range {rng} m')
        self.latest_depth_stamp_ = depth_msg.header.stamp

    def camera_info_callback(self, info_msg):
        """Camera intrinsics (focal length and centre, in pixels)"""
        self.camera_fx_ = info_msg.k[0]
        self.camera_cx_ = info_msg.k[2]
        self.camera_info_width_ = info_msg.width

    def image_callback(self, image_msg):
        # BGR for OpenCV and YOLO (the camera publishes RGB; passthrough would swap red and blue)
        image_bgr = self.cv_bridge_.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        # Perception 1: save the raw frame (before any boxes are drawn)
        if self.get_parameter('save_images').value:
            self.maybe_save_image(image_bgr, image_msg.header.stamp)

        # Advanced 1: degrade (simulated Martian conditions) and optionally enhance before detection.
        # The original stream stays on camera/image; depth is not degraded.
        processed = self.apply_robustness(image_bgr, image_msg.header)

        # Perception 2: detect
        detections = self.detect(processed, image_msg.header)
        self.detected_pub_.publish(Bool(data=len(detections) > 0))
        for d in sorted(detections, key=lambda d: d['confidence'], reverse=True):
            self.id_pub_.publish(Int32(data=artefact_id(d['class'])))
        self.update_robustness_stats(detections)

        # Perception 3: localise
        if detections:
            summary = ', '.join(f"{d['class']} ({d['confidence']:.2f})" for d in detections)
            self.get_logger().info(f'Detected: {summary}', throttle_duration_sec=1.0)
            self.localise(detections, image_msg.header.stamp, image_bgr.shape[1], image_bgr.shape[0])

    #######################################################
    # Perception 1: dataset collection

    def maybe_save_image(self, image_bgr, stamp):
        """Save a frame if enough time has passed and the robot has moved/turned enough"""
        if time.time() - self.last_save_time_ < 1.0:
            return

        pose = self.get_pose_2d()
        if pose is None:
            return

        if self.last_save_pose_ is not None:
            moved = math.hypot(pose[0] - self.last_save_pose_[0], pose[1] - self.last_save_pose_[1])
            dtheta = pose[2] - self.last_save_pose_[2]
            turned = abs(math.atan2(math.sin(dtheta), math.cos(dtheta)))
            if moved < 0.3 and turned < math.radians(10):
                return  # too similar to the last saved frame

        filename = f'img_{self.save_count_:05d}.png'
        cv2.imwrite(os.path.join(self.dataset_dir_, 'images', filename), image_bgr)
        with open(os.path.join(self.dataset_dir_, 'poses.csv'), 'a', newline='') as f:
            csv.writer(f).writerow([filename, stamp.sec, stamp.nanosec,
                                    f'{pose[0]:.3f}', f'{pose[1]:.3f}', f'{pose[2]:.3f}'])

        self.save_count_ += 1
        self.last_save_time_ = time.time()
        self.last_save_pose_ = pose
        self.get_logger().info(f'Saved {filename}')

    #######################################################
    # Advanced 1: robust perception

    def apply_robustness(self, image_bgr, header):
        """Apply the selected degradation (and enhancement), publish both streams, return the result"""
        kind = self.get_parameter('degradation').value
        level = self.get_parameter('degradation_level').value
        if degrade is None:
            if kind != 'none':
                self.get_logger().error('degradations.py not installed - degradation disabled',
                                        throttle_duration_sec=10.0)
            return image_bgr
        if kind not in ('none',) + DEGRADATIONS:
            self.get_logger().error(f'Unknown degradation "{kind}". Options: none, {", ".join(DEGRADATIONS)}',
                                    throttle_duration_sec=5.0)
            kind = 'none'

        processed = degrade(image_bgr, kind, level, self.degradation_rng_)
        msg = self.cv_bridge_.cv2_to_imgmsg(processed, encoding='bgr8')
        msg.header = header
        self.degraded_image_pub_.publish(msg)

        if self.get_parameter('enhance').value:
            processed = enhance(processed)
            msg = self.cv_bridge_.cv2_to_imgmsg(processed, encoding='bgr8')
            msg.header = header
            self.enhanced_image_pub_.publish(msg)
        return processed

    def update_robustness_stats(self, detections):
        """
        Live statistics for the current setting, logged every 50 frames: the fraction of frames with
        a detection and the mean top confidence. Park the robot facing an artefact and compare settings.
        """
        setting = (self.get_parameter('degradation').value,
                   self.get_parameter('degradation_level').value,
                   self.get_parameter('enhance').value)
        if setting != self.stats_setting_:
            self.stats_setting_, self.stats_frames_, self.stats_hits_, self.stats_conf_ = setting, 0, 0, 0.0
        self.stats_frames_ += 1
        if detections:
            self.stats_hits_ += 1
            self.stats_conf_ += max(d['confidence'] for d in detections)
        if self.stats_frames_ % 50 == 0:
            mean_conf = self.stats_conf_ / self.stats_hits_ if self.stats_hits_ else 0.0
            kind, level, enh = setting
            self.get_logger().info(
                f'[robustness] degradation={kind} level={level} enhance={enh}: artefact detected in '
                f'{100.0 * self.stats_hits_ / self.stats_frames_:.0f}% of {self.stats_frames_} frames, '
                f'mean top confidence {mean_conf:.2f}')

    #######################################################
    # Perception 2: detection

    def detect(self, image_bgr, header):
        """Run YOLO, publish the annotated image, return a list of detections"""
        confidence = self.get_parameter('detection_confidence').value
        result = self.yolo_model_.predict(image_bgr, conf=confidence, verbose=False)[0]

        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            detections.append({
                'class': result.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'box': (x1, y1, x2, y2),
            })

        annotated = result.plot()  # BGR image with boxes and labels drawn
        msg = self.cv_bridge_.cv2_to_imgmsg(annotated, encoding='bgr8')
        msg.header = header
        self.detections_image_pub_.publish(msg)
        return detections

    #######################################################
    # Perception 3: localisation

    def localise(self, detections, stamp, image_width, image_height):
        """
        Estimate the map position of each detection: direction from the box centre's pixel position
        (via the camera focal length), distance from the depth camera. Repeated sightings of the same
        artefact are combined in self.artefact_map_.
        """
        if self.latest_depth_ is None:
            self.get_logger().warn('No depth image yet - check depth_topic', throttle_duration_sec=5.0)
            return

        # Skip if the depth image is too old to match this colour image
        dt = abs((stamp.sec + stamp.nanosec * 1e-9) -
                 (self.latest_depth_stamp_.sec + self.latest_depth_stamp_.nanosec * 1e-9))
        if dt > 0.5:
            self.get_logger().warn(
                f'Depth and colour images are {dt:.2f} s apart - skipping localisation. If this '
                f'never goes away, the two cameras use different clocks (check use_sim_time)',
                throttle_duration_sec=5.0)
            return

        pose = self.get_pose_2d(stamp)
        if pose is None:
            return

        # Camera intrinsics: from camera_info if available, otherwise from the field of view
        if self.camera_fx_ is not None:
            scale = image_width / self.camera_info_width_
            fx, cx = self.camera_fx_ * scale, self.camera_cx_ * scale
        else:
            fx = (image_width / 2.0) / math.tan(self.get_parameter('camera_hfov').value / 2.0)
            cx = image_width / 2.0

        # The depth image may have a different resolution from the colour image
        depth = self.latest_depth_
        sx, sy = depth.shape[1] / image_width, depth.shape[0] / image_height

        min_conf = self.get_parameter('localisation_min_confidence').value
        max_depth = self.get_parameter('max_depth').value
        x_off = self.get_parameter('camera_x_offset').value
        y_off = self.get_parameter('camera_y_offset').value

        changed = False
        for d in detections:
            if d['confidence'] < min_conf:
                continue
            x1, y1, x2, y2 = d['box']
            if x1 <= 5 or x2 >= image_width - 5:
                continue  # cut off at the image edge: the box centre isn't the object's centre

            dist = depth_in_box(depth, (x1 * sx, y1 * sy, x2 * sx, y2 * sy), max_depth=max_depth)
            if dist is None:
                continue

            ax, ay = detection_to_map((x1 + x2) / 2.0, dist, fx, cx, *pose, x_off, y_off)
            _, newly_confirmed = self.artefact_map_.add(ax, ay, d['class'])
            changed = True
            if newly_confirmed:
                self.get_logger().warn(f'New artefact confirmed: {d["class"]} at ({ax:.2f}, {ay:.2f})')

        if changed:
            self.publish_artefacts()

    def publish_artefacts(self):
        """Publish markers for RViz and the confirmed locations for other nodes"""
        now = self.get_clock().now().to_msg()
        confirmed = self.artefact_map_.confirmed()

        # Locations for other nodes (e.g. planning)
        poses = PoseArray()
        poses.header.frame_id = 'map'
        poses.header.stamp = now
        for _, x, y, _ in confirmed:
            p = Pose()
            p.position.x, p.position.y = x, y
            p.orientation.w = 1.0
            poses.poses.append(p)
        self.locations_pub_.publish(poses)
        self.location_ids_pub_.publish(Int32MultiArray(data=[artefact_id(label) for label, _, _, _ in confirmed]))

        # RViz markers
        marker_array = MarkerArray()
        clear = Marker()
        clear.header.frame_id = 'map'
        clear.action = Marker.DELETEALL   # artefacts can merge, so ids change; clear old ones first
        marker_array.markers.append(clear)

        for i, (label, x, y, count) in enumerate(confirmed):
            r, g, b = CLASS_COLOURS.get(label, (1.0, 1.0, 1.0))

            sphere = Marker()
            sphere.header.frame_id = 'map'
            sphere.header.stamp = now
            sphere.ns = 'artefacts'
            sphere.id = i
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x, sphere.pose.position.y, sphere.pose.position.z = x, y, 0.5
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 1.0
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = r, g, b, 0.9
            marker_array.markers.append(sphere)

            text = Marker()
            text.header = sphere.header
            text.ns = 'artefact_labels'
            text.id = i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x, text.pose.position.y, text.pose.position.z = x, y, 1.6
            text.pose.orientation.w = 1.0
            text.scale.z = 0.6
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.text = f'{label} [{artefact_id(label)}] ({count})'
            marker_array.markers.append(text)

        if self.get_parameter('show_observations').value:
            points = self.artefact_map_.all_observations()
            if points:
                obs = Marker()
                obs.header.frame_id = 'map'
                obs.header.stamp = now
                obs.ns = 'observations'
                obs.id = 0
                obs.type = Marker.POINTS
                obs.action = Marker.ADD
                obs.pose.orientation.w = 1.0
                obs.scale.x = obs.scale.y = 0.1
                obs.color.r, obs.color.g, obs.color.b, obs.color.a = 1.0, 0.5, 0.0, 0.8
                obs.points = [Point(x=px, y=py, z=0.2) for px, py in points]
                marker_array.markers.append(obs)

        self.marker_pub_.publish(marker_array)


def main():
    rclpy.init()
    node = ArtefactPerception()
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
