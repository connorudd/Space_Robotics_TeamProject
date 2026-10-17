#!/usr/bin/env python3

import math
import random
from enum import Enum

import cv2  # OpenCV2
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Pose2D, PoseArray, PoseStamped, Point
from builtin_interfaces.msg import Duration as DurationMsg
from nav2_msgs.action import BackUp, NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

import json

from std_msgs.msg import Bool, Header, String

try:
    from sensor_msgs_py import point_cloud2   # part of ROS 2 Humble
except ImportError:
    point_cloud2 = None

from cave_explorer.frontier_explorer import FrontierExplorer
from cave_explorer.inspection_planner import InspectionPlanner


def wrap_angle(angle):
    """Function to wrap an angle between 0 and 2*Pi"""
    while angle < 0.0:
        angle = angle + 2 * math.pi

    while angle > 2 * math.pi:
        angle = angle - 2 * math.pi

    return angle

def pose2d_to_pose(pose_2d):
    """Convert a Pose2D to a full 3D Pose"""
    pose = Pose()

    pose.position.x = pose_2d.x
    pose.position.y = pose_2d.y

    pose.orientation.w = math.cos(pose_2d.theta / 2.0)
    pose.orientation.z = math.sin(pose_2d.theta / 2.0)

    return pose


class PlannerType(Enum):
    ERROR = 0
    MOVE_FORWARDS = 1
    RETURN_HOME = 2
    GO_TO_FIRST_ARTIFACT = 3
    RANDOM_WALK = 4
    RANDOM_GOAL = 5
    # Add more!


class State(Enum):
    """Top-level behaviour. APPROACHING, INSPECTING and RETURN_HOME arrive with Planning 2 and 3."""
    EXPLORING = 1
    DONE = 2       # nothing left to explore
    APPROACHING = 3  # driving to a standoff pose next to an artefact
    INSPECTING = 4   # at the standoff, waiting for the detector to confirm the artefact
    RETURNING_HOME = 5  # exploration complete or time budget used up


class GoalOutcome(Enum):
    """How the last Nav2 goal ended, for the state handlers to branch on"""
    NONE = 0       # no goal has ended since the handler last looked
    REACHED = 1    # arrived, or was cancelled within arrival_slack of the goal
    ABORTED = 2    # Nav2 gave up, or something else preempted the goal
    CANCELLED = 3  # cancelled by the stuck-goal check (or by someone else)
    PREEMPTED = 4  # cancelled on purpose to switch behaviour: not a failure


class CaveExplorer(Node):
    def __init__(self):
        super().__init__('cave_explorer_node')

        # Variables/Flags for mapping
        self.xlim_ = [0.0, 0.0]
        self.ylim_ = [0.0, 0.0]

        # Variables/Flags for perception
        self.artifact_found_ = False

        # Variables/Flags for planning
        self.state_ = None  # set by set_state() once the node is built
        self.planner_type_ = PlannerType.ERROR
        self.first_artifact_done_ = False  # reached, or abandoned after repeated failures
        self.return_home_done_ = False     # reached, or abandoned after repeated failures
        self.consecutive_failures_ = 0     # goals in a row that were aborted or cancelled
        self.goal_outcome_ = GoalOutcome.NONE

        # Marker for artifact locations
        # See https://wiki.ros.org/rviz/DisplayTypes/Marker
        self.marker_artifacts_ = Marker()
        self.marker_artifacts_.header.frame_id = "map"
        self.marker_artifacts_.ns = "artifacts"
        self.marker_artifacts_.id = 0
        self.marker_artifacts_.type = Marker.SPHERE_LIST
        self.marker_artifacts_.action = Marker.ADD
        self.marker_artifacts_.pose.position.x = 0.0
        self.marker_artifacts_.pose.position.y = 0.0
        self.marker_artifacts_.pose.position.z = 0.0
        self.marker_artifacts_.pose.orientation.x = 0.0
        self.marker_artifacts_.pose.orientation.y = 0.0
        self.marker_artifacts_.pose.orientation.z = 0.0
        self.marker_artifacts_.pose.orientation.w = 1.0
        self.marker_artifacts_.scale.x = 1.5
        self.marker_artifacts_.scale.y = 1.5
        self.marker_artifacts_.scale.z = 1.5
        self.marker_artifacts_.color.a = 1.0
        self.marker_artifacts_.color.r = 0.0
        self.marker_artifacts_.color.g = 1.0
        self.marker_artifacts_.color.b = 0.2
        # The template's stop-sign detector is off by default: the Perception node (artefact_perception)
        # publishes on the same topics (marker_array_artifacts, detections_image) and replaces it.
        self.declare_parameter('template_detector', False)
        self.template_detector_ = self.get_parameter('template_detector').value
        if self.template_detector_:
            self.marker_pub_ = self.create_publisher(MarkerArray, 'marker_array_artifacts', 10)

        # Remember the artifact locations
        # Array of type geometry_msgs.Point
        self.artifact_locations_ = []

        # Initialise CvBridge
        self.cv_bridge_ = CvBridge()

        # Prepare transformation to get robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Action client for nav2
        self.nav2_action_client_ = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        # Nav2's own back-up behaviour, used to get unstuck (behavior_server provides 'backup')
        self.backup_client_ = ActionClient(self, BackUp, 'backup')
        self.declare_parameter('stuck_move_threshold', 0.5)  # m; a failed goal moving less than this = wedged
        self.declare_parameter('backup_distance', 1.0)       # m to reverse when wedged
        self.last_goal_moved_ = None    # straight-line distance the robot moved during the last goal
        self.pending_outcome_ = None    # outcome held back while backing up
        self.backup_start_ = None       # sim time the back-up was sent, None when not backing up
        self.get_logger().warn('Waiting for navigate_to_pose action...')
        self.nav2_action_client_.wait_for_server()
        self.get_logger().warn('navigate_to_pose connected')
        self.ready_for_next_goal_ = True
        self.goal_handle_ = None  # active Nav2 goal; kept so it can be cancelled
        self.last_goal_pose2d_ = None  # most recent goal, kept for retry after a rejection
        self.retry_timer_ = None
        self.goal_sent_time_ = None  # when the outstanding goal request was sent; None once answered
        self.declare_parameter('print_feedback', False)

        # Stuck-goal handling: give up on a goal that goes into recovery or stops making progress
        # max_recoveries: cancel when Nav2's recovery count exceeds this. Not 0: in the 10 Oct run all 6
        # first-recovery cancels coincided with a bt_navigator "Timed out while waiting for action
        # server" warning, which Nav2 retries by itself. Treat 2 as a starting point to tune.
        self.declare_parameter('max_recoveries', 2)
        self.declare_parameter('stuck_timeout', 30.0)     # node-clock (sim) seconds without progress
        self.declare_parameter('progress_epsilon', 0.5)   # metres closer that count as progress
        self.last_feedback_ = None       # latest NavigateToPose feedback for the active goal
        self.best_distance_ = None       # smallest distance_remaining seen for the active goal
        self.last_progress_time_ = None  # node-clock time of the last real progress
        self.cancel_reason_ = None       # set once a cancel has been requested for the active goal
        self.blacklist_ = []             # (x, y) goals given up on, for the planners to avoid
        # arrival_slack: a goal cancelled this close (m) counts as reached. In run 4 a goal was cancelled
        # at 0.95 m after 3 recoveries, inside the 1.0 m tolerance, so calling that a failure is wrong.
        self.declare_parameter('arrival_slack', 1.5)
        self.cancel_near_goal_ = False   # the cancel was requested close enough to count as arrival
        self.first_distance_ = None      # first distance_remaining of the active goal (for the end-of-goal log)
        self.goal_start_pose_ = None     # robot pose when the active goal was accepted

        # Publisher for the goal pose visualisation
        self.current_goal_pub_ = self.create_publisher(PoseStamped, 'current_goal', 1)

        # Subscribe to the map topic to get current bounds
        self.map_sub_ = self.create_subscription(OccupancyGrid, 'map',  self.map_callback, 1)

        # Prepare image processing
        self.declare_parameter('computer_vision_model_filename', rclpy.Parameter.Type.STRING)
        if self.template_detector_:
            self.image_detections_pub_ = self.create_publisher(Image, 'detections_image', 1)
            self.computer_vision_model_ = cv2.CascadeClassifier(self.get_parameter('computer_vision_model_filename').value)
            self.image_sub_ = self.create_subscription(Image, 'camera/image', self.image_callback, 1)

        # Frontier exploration (Planning 1). exploration:=scripted falls back to the template route.
        self.declare_parameter('exploration', 'frontier')
        self.frontier_explorer_ = FrontierExplorer(robot_radius=0.5, min_cluster_size=5,
                                                   blacklist_radius=2.0, max_hole_cells=4)
        self.map_msg_ = None            # latest /map, planned on when a goal is needed
        self.visited_goals_ = []        # frontier goals already reached: never sent again
        self.next_plan_time_ = None     # wait before re-planning when no frontier was usable
        self.no_frontier_count_ = 0     # plans in a row that found no usable frontier
        self.planning_marker_pub_ = self.create_publisher(MarkerArray, 'marker_array_planning', 1)

        # Close-range inspection (Planning 2). Artefacts arrive as JSON on /artifacts:
        # a list of {id, type, x, y, z, n_obs, last_seen} in the map frame (last_seen in sim seconds).
        self.declare_parameter('min_observations', 2)    # ignore one-off detections (likely false positives)
        self.declare_parameter('inspect_dwell', 3.0)     # sim seconds to wait at the standoff
        self.inspection_planner_ = InspectionPlanner()
        self.artifacts_ = {}            # id -> latest report
        self.inspected_ = {}            # id -> True if confirmed at close range, False if not
        self.abandoned_ = set()         # ids given up on after repeated failures
        self.target_ = None             # report of the artefact being approached / inspected
        self.target_attempts_ = 0       # standoffs tried for the current target
        self.failed_standoffs_ = []     # (x, y) standoffs that failed for the current target
        self.inspect_start_ = None      # sim time the robot arrived at the standoff
        self.standoff_ = None           # (x, y, yaw, path) of the current standoff goal
        self.next_inspection_check_ = None  # sim time before which no new inspection is planned
        self.artifacts_sub_ = self.create_subscription(String, 'artifacts', self.artifacts_callback, 10)

        # The Perception node publishes confirmed artefacts (seen at least 3 times) as positions only
        # (PoseArray, map frame, re-sent every second) plus a Bool while any artefact is in view.
        # These are turned into the same reports as /artifacts: ids are assigned here by position.
        self.declare_parameter('perception_match_radius', 2.0)  # m, same as the Perception merge radius
        self.next_perception_id_ = 0
        self.locations_sub_ = self.create_subscription(PoseArray, 'artefact_locations',
                                                       self.artefact_locations_callback, 10)
        self.detected_sub_ = self.create_subscription(Bool, 'artefact_detected',
                                                      self.artefact_detected_callback, 10)
        self.inspection_marker_pub_ = self.create_publisher(MarkerArray, 'marker_array_inspection', 1)

        # Known artefacts as obstacles for Nav2. Low artefacts (the green crystals) sit below the
        # lidar's scan plane (lidar mounted 1 m up), so Nav2's costmaps never see them and the robot
        # drives into them (10 Oct runs: wedged against the crystals at about (6.3, 4.4)).
        self.declare_parameter('artifact_obstacle_radius', 0.8)   # m, disc of obstacle points per artefact
        self.artifact_obstacle_pub_ = self.create_publisher(PointCloud2, 'artifact_obstacles', 1)
        self.artifact_obstacle_timer_ = self.create_timer(1.0, self.publish_artifact_obstacles)
        if point_cloud2 is None:
            self.get_logger().error('sensor_msgs_py not found: artefacts will not be sent to Nav2 as obstacles')

        # Behaviour switching (Planning 3)
        self.declare_parameter('mission_time', 0.0)   # sim seconds before returning home; 0 = no limit
        self.declare_parameter('home_x', 0.0)         # the robot spawns at the map origin
        self.declare_parameter('home_y', 0.0)
        self.preempt_reason_ = None      # set when a goal is cancelled on purpose (not a failure)
        self.mission_start_ = None       # sim time of the first tick with a map -> base_link transform
        self.home_attempts_ = 0
        self.next_preempt_check_ = 0.0   # sim time of the next check for a reason to preempt
        self.state_pub_ = self.create_publisher(String, 'planner_state', 1)
        self.state_marker_pub_ = self.create_publisher(MarkerArray, 'marker_array_state', 1)
        self.state_marker_timer_ = self.create_timer(1.0, self.publish_state_display)

        self.set_state(State.EXPLORING, 'startup')

        # Timer for main loop
        self.main_loop_timer_ = self.create_timer(0.2, self.main_loop)

    def set_state(self, new_state, reason):
        """Change the top-level state, logging every transition with its reason"""

        old = self.state_.name if self.state_ is not None else 'none'
        self.get_logger().info(f'State {old} -> {new_state.name} ({reason})')
        self.state_ = new_state
        msg = String()
        msg.data = f'{new_state.name}: {reason}'
        self.state_pub_.publish(msg)
    
    def get_pose_2d(self):
        """Get the 2d pose of the robot"""

        # Lookup the latest transform
        try:
            t = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().error(f'Could not transform: {ex}')
            return

        # Return a Pose2D message
        pose = Pose2D()
        pose.x = t.transform.translation.x
        pose.y = t.transform.translation.y

        qw = t.transform.rotation.w
        qz = t.transform.rotation.z

        if qz >= 0.:
            pose.theta = wrap_angle(2. * math.acos(qw))
        else: 
            pose.theta = wrap_angle(-2. * math.acos(qw))

        return pose

    def map_callback(self, map_msg: OccupancyGrid):
        """New map received, so update x and y limits"""

        # Extract data from message
        map_origin = [map_msg.info.origin.position.x, 
                      map_msg.info.origin.position.y]
        map_resolution = map_msg.info.resolution
        map_height = map_msg.info.height
        map_width = map_msg.info.width

        # Set current limits
        self.xlim_ = [map_origin[0], map_origin[0]+map_width*map_resolution]
        self.ylim_ = [map_origin[1], map_origin[1]+map_height*map_resolution]

        # Keep the whole map for frontier planning (only processed when a new goal is needed)
        self.map_msg_ = map_msg

        # self.get_logger().warn('Map received:')
        # self.get_logger().warn(f'  xlim = [{self.xlim_[0]:.2f}, {self.xlim_[1]:.2f}]')
        # self.get_logger().warn(f'  ylim = [{self.ylim_[0]:.2f}, {self.ylim_[1]:.2f}]')
    
    def image_callback(self, image_msg):
        """
        Recieve an RGB image.
        Use this method to detect artifacts of interest.
        
        A simple method has been provided to begin with for detecting stop signs (which is not what we're actually looking for) 
        adapted from: https://www.geeksforgeeks.org/detect-an-object-with-opencv-python/
        """
    
        # Copy the image message to a cv image
        # see http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
        image = self.cv_bridge_.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')

        # Create a grayscale version (some simple models use this)
        # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Retrieve the pre-trained model
        stop_sign_model = self.computer_vision_model_

        # Detect artifacts in the image
        # The minSize is used to avoid very small detections that are probably noise
        detections = stop_sign_model.detectMultiScale(image, minSize=(20,20))

        # You can set "artifact_found_" to true to signal to "main_loop" that you have found a artifact
        # You may want to communicate more information
        # Since the "image_callback" and "main_loop" methods can run at the same time you should protect any shared variables
        # with a mutex
        # "artifact_found_" doesn't need a mutex because it's an atomic
        num_detections = len(detections)

        if num_detections > 0:
            self.artifact_found_ = True
        else:
            self.artifact_found_ = False

        # Draw a bounding box rectangle on the image for each detection
        for(x, y, width, height) in detections:
            cv2.rectangle(image, (x, y), (x + height, y + width), (0, 255, 0), 5)

        # Publish the image with the detection bounding boxes
        image_detection_message = self.cv_bridge_.cv2_to_imgmsg(image, encoding="rgb8")
        self.image_detections_pub_.publish(image_detection_message)

        if self.artifact_found_:
            self.get_logger().info('Artifact found!')
            self.localise_artifact()


    def localise_artifact(self):
        """
        INCOMPLETE:
        Compute the location of the artifact
        Save it to a list, publish rviz marker
        This version just uses the robot location rather than the artifact location
        You can find other examples of using RViz markers in the previous assignments template code
        """

        # Current location of the robot
        robot_pose = self.get_pose_2d()

        if robot_pose == None:
            self.get_logger().warn(f'localise_artifact: robot_pose is None.')
            return

        # Compute the location of the artifact
        # This is currently INCOMPLETE
        point = Point()
        point.x = robot_pose.x
        point.y = robot_pose.y
        point.z = 1.0

        # Save it
        self.artifact_locations_.append(point)

        # Publish the markers
        self.publish_artifact_markers()

    def publish_artifact_markers(self):
        """ Publish the artifact location markers"""

        # Update the locations
        self.marker_artifacts_.points = self.artifact_locations_

        # Create and publish the MarkerArray
        marker_array = MarkerArray()
        marker_array.markers = [self.marker_artifacts_]
        self.marker_pub_.publish(marker_array)


    def planner_go_to_pose2d(self, pose2d):
        """Go to a provided 2d pose"""

        # A goal is now in flight: main_loop waits until goal_reached_callback clears this
        self.ready_for_next_goal_ = False

        # Remember the goal so it can be resent if Nav2 rejects it
        self.last_goal_pose2d_ = pose2d

        # Send a goal to navigate_to_pose with self.nav2_action_client_
        action_goal = NavigateToPose.Goal()
        action_goal.pose.header.stamp = self.get_clock().now().to_msg()
        action_goal.pose.header.frame_id = 'map'
        action_goal.pose.pose = pose2d_to_pose(pose2d)

        # Publish visualisation
        self.current_goal_pub_.publish(action_goal.pose)

        # Send goal to action server. Feedback is always on: the stuck-goal check needs it.
        # (print_feedback now only controls whether each feedback message is logged.)
        self.get_logger().warn(f'Sending goal [{pose2d.x:.2f}, {pose2d.y:.2f}]...')
        self.send_goal_future_ = self.nav2_action_client_.send_goal_async(
            action_goal,
            feedback_callback=self.feedback_callback)
        self.send_goal_future_.add_done_callback(self.goal_response_callback)
        self.goal_sent_time_ = self.get_clock().now()

    def goal_response_callback(self, future):
        """The requested goal pose has been sent to the action server"""

        # A request cancelled by the watchdog still calls back: ignore it
        if future.cancelled():
            return

        # Nav2 answered, so the watchdog can stand down
        self.goal_sent_time_ = None

        goal_handle = future.result()
        if not goal_handle.accepted:
            # Nav2 is probably not active yet (startup or restart): resend the same pose shortly
            self.get_logger().error('Goal rejected, retrying in 2 s')
            self.retry_timer_ = self.create_timer(2.0, self.retry_goal_callback)
            return

        # Goal accepted: keep the handle so the goal can be cancelled, then wait for the result
        self.get_logger().warn(f'Goal accepted')
        # Start the stuck-goal tracking for this goal (before the handle is set, so the check
        # in main_loop never sees a handle without a progress time)
        self.last_feedback_ = None
        self.best_distance_ = None
        self.last_progress_time_ = self.get_clock().now()
        self.cancel_reason_ = None
        self.cancel_near_goal_ = False
        self.preempt_reason_ = None
        self.first_distance_ = None
        self.last_goal_moved_ = None
        self.goal_start_pose_ = self.get_pose_2d()
        self.goal_handle_ = goal_handle
        self.get_result_future_ = goal_handle.get_result_async()
        self.get_result_future_.add_done_callback(self.goal_reached_callback)

    def feedback_callback(self, feedback_msg):
        """Monitor the feedback from the action server"""

        # Ignore late feedback from a goal that has already ended (or not yet accepted)
        if self.goal_handle_ is None or feedback_msg.goal_id != self.goal_handle_.goal_id:
            return

        feedback = feedback_msg.feedback
        self.last_feedback_ = feedback

        if self.get_parameter('print_feedback').value:
            self.get_logger().info(f'{feedback.distance_remaining:.2f} m remaining')

        # Nav2 reports 0 before it has a path: that is not progress, so ignore it
        distance = feedback.distance_remaining
        if distance <= 0.0:
            return

        # The first feedback of a goal is computed from the PREVIOUS goal's path (10 Oct logs: the
        # first distance of each goal equalled the previous goal's last distance, e.g. 1.0 m for a
        # goal 20 m away). A path can never be shorter than the straight line from the robot to the
        # goal, so a shorter distance is stale: ignore it, or it becomes an unbeatable "best distance".
        pose = self.get_pose_2d()
        goal = self.last_goal_pose2d_
        if pose is not None and goal is not None:
            straight_line = math.hypot(goal.x - pose.x, goal.y - pose.y)
            if distance < straight_line - 0.5:
                return

        if self.first_distance_ is None:
            self.first_distance_ = distance

        # Progress = at least progress_epsilon closer than the best distance so far
        if self.best_distance_ is None or \
                distance < self.best_distance_ - self.get_parameter('progress_epsilon').value:
            self.best_distance_ = distance
            self.last_progress_time_ = self.get_clock().now()

    def check_goal_stuck(self):
        """Return a reason string if the active goal should be given up on, otherwise None"""

        # Nav2 has started its recovery behaviours (spin, back up, wait...)
        feedback = self.last_feedback_
        if feedback is not None and \
                feedback.number_of_recoveries > self.get_parameter('max_recoveries').value:
            return f'{feedback.number_of_recoveries} recovery attempt(s)'

        # No real progress for too long (also fires if feedback stops arriving)
        idle = (self.get_clock().now() - self.last_progress_time_).nanoseconds * 1e-9
        if idle > self.get_parameter('stuck_timeout').value:
            return f'no progress for {idle:.0f} s'

        return None

    def cancel_active_goal(self, reason):
        """Cancel the active goal, remember it as unreachable, and let the result callback finish up"""

        goal = self.last_goal_pose2d_
        feedback = self.last_feedback_
        remaining = f'{feedback.distance_remaining:.2f} m' if feedback is not None else 'unknown distance'

        # Close enough already: count it as reached, and do not blacklist a goal we got to
        self.cancel_near_goal_ = feedback is not None and \
            feedback.distance_remaining <= self.get_parameter('arrival_slack').value
        if self.cancel_near_goal_:
            self.get_logger().warn(f'Cancelling goal [{goal.x:.2f}, {goal.y:.2f}]: {reason}, '
                                   f'{remaining} remaining (close enough, counting it as reached)')
        else:
            self.blacklist_.append((goal.x, goal.y))
            self.get_logger().warn(f'Cancelling goal [{goal.x:.2f}, {goal.y:.2f}]: {reason}, '
                                   f'{remaining} remaining (blacklisted, {len(self.blacklist_)} total)')

        # Cancelling stops the robot (the controller publishes zero velocity). Do not use the
        # Nav2 panel's Reset/Pause for this: the robot keeps driving on its last command.
        self.cancel_reason_ = reason
        self.goal_handle_.cancel_goal_async()

    def goal_reached_callback(self, future):
        """The requested goal has finished: reached, aborted or cancelled"""

        status = future.result().status
        self.goal_handle_ = None

        # Report what actually happened, and record it for the state machine to branch on
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Goal reached!')
            self.goal_outcome_ = GoalOutcome.REACHED
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warn('Goal aborted by Nav2')
            self.goal_outcome_ = GoalOutcome.ABORTED
        elif status == GoalStatus.STATUS_CANCELED and self.preempt_reason_ is not None:
            self.get_logger().info(f'Goal preempted: {self.preempt_reason_}')
            self.goal_outcome_ = GoalOutcome.PREEMPTED
        elif status == GoalStatus.STATUS_CANCELED and self.cancel_near_goal_:
            self.get_logger().info('Goal cancelled near the goal: counted as reached')
            self.goal_outcome_ = GoalOutcome.REACHED
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warn('Goal cancelled')
            self.goal_outcome_ = GoalOutcome.CANCELLED
        else:
            self.get_logger().error(f'Goal ended with unexpected status {status}')
            self.goal_outcome_ = GoalOutcome.ABORTED

        # Record how the goal ended, so an abort or cancel can be diagnosed from the log
        feedback = self.last_feedback_
        if feedback is not None:
            # Path distance went from first_distance_ to remaining; moved is the straight-line
            # displacement of the robot. Moved far but remaining barely changed = detour, not stuck.
            first = f'{self.first_distance_:.1f}' if self.first_distance_ is not None else '?'
            end_pose = self.get_pose_2d()
            if self.goal_start_pose_ is not None and end_pose is not None:
                self.last_goal_moved_ = math.hypot(end_pose.x - self.goal_start_pose_.x,
                                                   end_pose.y - self.goal_start_pose_.y)
                moved = f'{self.last_goal_moved_:.1f}'
            else:
                self.last_goal_moved_ = None
                moved = '?'
            self.get_logger().info(f'Goal ended with {feedback.distance_remaining:.2f} m remaining '
                                   f'(started at {first} m, robot moved {moved} m), '
                                   f'{feedback.number_of_recoveries} recovery attempt(s)')
        self.last_feedback_ = None
        self.cancel_reason_ = None
        self.cancel_near_goal_ = False
        self.preempt_reason_ = None
        self.ready_for_next_goal_ = True

    def retry_goal_callback(self):
        """Resend the rejected goal once the retry delay has passed"""

        # One-shot: stop the timer so the goal is resent once per rejection
        self.retry_timer_.cancel()
        self.destroy_timer(self.retry_timer_)
        self.planner_go_to_pose2d(self.last_goal_pose2d_)

    def planner_move_forwards(self, distance):
        """Simply move forward by the specified distance"""

        pose_2d = self.get_pose_2d()

        pose_2d.x += distance * math.cos(pose_2d.theta)
        pose_2d.y += distance * math.sin(pose_2d.theta)

        self.planner_go_to_pose2d(pose_2d)

    def planner_go_to_first_artifact(self):
        """Go to a pre-specified artifact location"""

        goal_pose2d = Pose2D(
            x = 18.1,
            y = 6.6,
            theta = math.pi/2
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def planner_return_home(self):
        """Return to the origin"""

        goal_pose2d = Pose2D(
            x = 0.0,
            y = 0.0,
            theta = math.pi
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def planner_random_walk(self):
        """Go to a random location, which may be invalid"""

        # Select a random location
        goal_pose2d = Pose2D(
            x = random.uniform(self.xlim_[0], self.xlim_[1]),
            y = random.uniform(self.ylim_[0], self.ylim_[1]),
            theta = random.uniform(0, 2*math.pi)
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def planner_random_goal(self):
        """Go to a random location out of a predefined set"""

        # Hand picked set of goal locations
        random_goals = [[15.2, 2.2],
                        [30.7, 2.2],
                        [43.0, 11.3],
                        [36.6, 21.9],
                        [33.0, 30.4],
                        [40.4, 44.3],
                        [51.5, 37.8],
                        [16.0, 24.1],
                        [3.4, 33.5],
                        [7.9, 13.8],
                        [14.2, 37.7]]

        # Select a random location. Bounded: the old unbounded loop spun forever (blocking the whole
        # node) if no goal was inside the map bounds, e.g. before the first map arrives.
        goal_valid = False
        for _ in range(50):
            idx = random.randint(0,len(random_goals)-1)
            goal_x = random_goals[idx][0]
            goal_y = random_goals[idx][1]

            # Only accept this goal if it's within the current costmap bounds
            if goal_x > self.xlim_[0] and goal_x < self.xlim_[1] and \
               goal_y > self.ylim_[0] and goal_y < self.ylim_[1]:
                goal_valid = True
                break
            else:
                self.get_logger().warn(f'Goal [{goal_x}, {goal_y}] out of bounds')

        if not goal_valid:
            # Send nothing: main_loop stays ready and tries again next tick
            self.get_logger().warn('No random goal inside the map bounds yet', throttle_duration_sec=5.0)
            return

        goal_pose2d = Pose2D(
            x = goal_x,
            y = goal_y,
            theta = random.uniform(0, 2*math.pi)
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def main_loop(self):
        """
        Set the next goal pose and send to the action server
        See https://docs.nav2.org/concepts/index.html
        """
        
        # Don't do anything until SLAM is launched
        if not self.tf_buffer.can_transform(
                'map',
                'base_link',
                rclpy.time.Time()):
            self.get_logger().warn('Waiting for the map -> base_link transform (is SLAM running?)',
                                   throttle_duration_sec=2.0)
            return
        if self.mission_start_ is None:
            self.mission_start_ = self.now_sec()

        #######################################################
        # Update flags related to the progress of the current planner

        # Watchdog: Nav2 never answered the goal request (e.g. it restarted), so send it again
        if self.goal_sent_time_ is not None:
            waited = (self.get_clock().now() - self.goal_sent_time_).nanoseconds * 1e-9
            if waited > 5.0:
                self.get_logger().error('No response to goal request, resending')
                self.send_goal_future_.cancel()
                self.goal_sent_time_ = None
                self.planner_go_to_pose2d(self.last_goal_pose2d_)
                return

        # Stuck-goal check: cancel a goal that is in recovery or has stopped making progress
        if self.goal_handle_ is not None and self.cancel_reason_ is None:
            stuck_reason = self.check_goal_stuck()
            if stuck_reason is not None:
                self.cancel_active_goal(stuck_reason)

        # A back-up that never answers must not freeze the node
        if self.backup_start_ is not None and self.now_sec() - self.backup_start_ > 20.0:
            self.get_logger().error('Back-up did not finish in 20 s, carrying on')
            self.backup_done(None)

        # Behaviour switching that has to interrupt a goal in flight (Planning 3)
        if self.check_time_budget():
            return
        self.check_preemption()

        # Check if previous goal still running
        if not self.ready_for_next_goal_:
            return

        #######################################################
        # No goal in flight: hand the outcome of the last goal (if any) to the current state.
        # Each handler decides what to do next; sending a goal makes the node busy again.
        outcome = self.goal_outcome_
        self.goal_outcome_ = GoalOutcome.NONE

        # Wedged: the goal failed and the robot barely moved. Reverse first, then let the
        # state handle the failure as usual (backup_done hands the outcome back).
        if outcome in (GoalOutcome.ABORTED, GoalOutcome.CANCELLED) and self.last_goal_moved_ is not None \
                and self.last_goal_moved_ < self.get_parameter('stuck_move_threshold').value:
            if self.start_backup(outcome):
                return

        if self.state_ == State.EXPLORING:
            if self.get_parameter('exploration').value == 'scripted':
                self.state_exploring_scripted(outcome)
            else:
                self.state_exploring_frontier(outcome)
        elif self.state_ == State.APPROACHING:
            self.state_approaching(outcome)
        elif self.state_ == State.INSPECTING:
            self.state_inspecting()
        elif self.state_ == State.RETURNING_HOME:
            self.state_returning_home(outcome)
        elif self.state_ == State.DONE:
            self.get_logger().info('Mission finished, idle', throttle_duration_sec=30.0)
        else:
            # Log instead of crashing: the old destroy_node() here left a dead node spinning
            self.get_logger().error(f'No handler for state {self.state_}', throttle_duration_sec=5.0)

    def state_exploring_frontier(self, outcome):
        """
        EXPLORING with frontier goals (Planning 1 baseline): go to the nearest frontier by path.
        Reached goals are remembered as visited, failed goals are blacklisted; neither is chosen again.
        Three plans in a row without a usable frontier end the exploration.
        """

        # 1. Account for how the previous goal ended
        goal = self.last_goal_pose2d_
        if goal is not None and outcome == GoalOutcome.REACHED:
            self.consecutive_failures_ = 0
            self.visited_goals_.append((goal.x, goal.y))
        elif goal is not None and outcome in (GoalOutcome.ABORTED, GoalOutcome.CANCELLED):
            self.consecutive_failures_ += 1
            if (goal.x, goal.y) not in self.blacklist_:   # a stuck-goal cancel has already added it
                self.blacklist_.append((goal.x, goal.y))
            self.get_logger().warn(f'Frontier goal [{goal.x:.2f}, {goal.y:.2f}] failed ({outcome.name}), '
                                   f'blacklisted ({len(self.blacklist_)} total)')

        # 2. An artefact waiting for inspection takes priority over the next frontier
        if self.start_inspection_if_needed():
            return

        # 3. Wait for a map, a pose, and any re-plan delay
        if self.map_msg_ is None:
            self.get_logger().warn('Waiting for the first map', throttle_duration_sec=5.0)
            return
        now = self.get_clock().now()
        if self.next_plan_time_ is not None and now < self.next_plan_time_:
            return
        pose = self.get_pose_2d()
        if pose is None:
            return

        # 4. Plan on the latest map
        info = self.map_msg_.info
        grid = np.array(self.map_msg_.data, dtype=np.int8).reshape(info.height, info.width)
        best, frontiers, reason = self.frontier_explorer_.plan(
            grid, info.resolution, info.origin.position.x, info.origin.position.y,
            (pose.x, pose.y),
            # Known artefacts too: low artefacts may be invisible to the lidar (mounted 1 m up), and the
            # pocket behind one is never mapped, so a frontier next to an artefact lures the robot into it
            blacklist=self.blacklist_ + self.visited_goals_ +
            [(a['x'], a['y']) for a in self.artifacts_.values()])
        self.publish_frontier_markers(frontiers, best, info)

        if best is None:
            self.no_frontier_count_ += 1
            self.get_logger().warn(f'No frontier goal: {reason} ({self.no_frontier_count_}/3)')
            if self.no_frontier_count_ >= 3:
                self.start_return_home(reason)
            else:
                self.next_plan_time_ = now + Duration(seconds=2.0)  # let the map update first
            return

        # 5. Go there
        self.no_frontier_count_ = 0
        gx, gy, yaw = best.goal
        self.get_logger().info(f'Frontier goal [{gx:.2f}, {gy:.2f}]: {reason}, '
                               f'cluster of {best.size} cells')
        self.planner_go_to_pose2d(Pose2D(x=float(gx), y=float(gy), theta=float(yaw)))

    ###########################################################
    # Behaviour switching (Planning 3)

    def preempt_active_goal(self, reason):
        """Cancel the active goal on purpose. It ends as PREEMPTED: not blacklisted, no back-up."""
        self.get_logger().warn(f'Preempting the current goal: {reason}')
        self.preempt_reason_ = reason
        self.cancel_reason_ = reason   # stops the stuck-goal check cancelling it a second time
        self.goal_handle_.cancel_goal_async()

    def check_time_budget(self):
        """When mission_time is used up, stop what we are doing and go home. True while it takes over."""
        limit = self.get_parameter('mission_time').value
        if limit <= 0.0 or self.state_ in (State.RETURNING_HOME, State.DONE):
            return False
        if self.now_sec() - self.mission_start_ < limit:
            return False
        if self.goal_handle_ is not None:
            if self.preempt_reason_ is None:
                self.preempt_active_goal(f'time budget of {limit:.0f} s used up')
            return True     # wait for the cancel to finish
        if not self.ready_for_next_goal_:
            return True     # a goal request or back-up is still in progress
        self.start_return_home(f'time budget of {limit:.0f} s used up')
        return True

    def check_preemption(self):
        """Interrupt a goal in flight when something more important comes up (checked every 2 s)"""
        if self.goal_handle_ is None or self.preempt_reason_ is not None or self.cancel_reason_ is not None:
            return
        if self.now_sec() < self.next_preempt_check_:
            return
        self.next_preempt_check_ = self.now_sec() + 2.0

        # Exploring and an artefact is waiting: go and inspect it now, not after this frontier
        if self.state_ == State.EXPLORING and self.get_parameter('exploration').value != 'scripted':
            found = self.next_inspection_target()
            if found is not None:
                artifact = found[0]
                self.preempt_active_goal(f"artefact {artifact['id']} ({artifact['type']}) waiting for inspection")
            return

        # Approaching and the detector has moved the target: re-plan the standoff
        if self.state_ == State.APPROACHING and self.target_ is not None:
            latest = self.artifacts_.get(self.target_['id'])
            if latest is not None and math.hypot(latest['x'] - self.target_['x'],
                                                 latest['y'] - self.target_['y']) > 1.5:
                self.target_ = latest
                self.preempt_active_goal(f"artefact {latest['id']} moved to "
                                         f"({latest['x']:.1f}, {latest['y']:.1f})")

    def start_return_home(self, reason):
        """Drop whatever is happening and drive back to the start"""
        self.goal_outcome_ = GoalOutcome.NONE
        self.target_ = None
        self.standoff_ = None
        self.home_attempts_ = 0
        self.set_state(State.RETURNING_HOME, reason)
        self.send_home_goal()

    def send_home_goal(self):
        self.planner_go_to_pose2d(Pose2D(x=float(self.get_parameter('home_x').value),
                                         y=float(self.get_parameter('home_y').value), theta=0.0))

    def state_returning_home(self, outcome):
        """RETURNING_HOME: up to three tries, then stop wherever the robot is"""
        if outcome == GoalOutcome.REACHED:
            self.set_state(State.DONE, 'back home')
        elif outcome in (GoalOutcome.ABORTED, GoalOutcome.CANCELLED, GoalOutcome.PREEMPTED):
            self.home_attempts_ += 1
            if self.home_attempts_ < 3:
                self.get_logger().warn(f'Return home failed ({outcome.name}), trying again')
                self.send_home_goal()
            else:
                self.set_state(State.DONE, 'could not reach home after 3 attempts')

    def publish_state_display(self):
        """RViz: the current state as text above the robot, for the demo"""
        if self.state_ is None or not self.tf_buffer.can_transform('map', 'base_link', rclpy.time.Time()):
            return
        pose = self.get_pose_2d()
        if pose is None:
            return
        text = self.state_.name
        if self.state_ in (State.APPROACHING, State.INSPECTING) and self.target_ is not None:
            text += f" artefact {self.target_['id']} ({self.target_['type']})"
        confirmed = sum(1 for v in self.inspected_.values() if v)
        text += f'\ninspected {confirmed}/{len(self.artifacts_)}'
        limit = self.get_parameter('mission_time').value
        if limit > 0.0 and self.mission_start_ is not None:
            text += f'  time {self.now_sec() - self.mission_start_:.0f}/{limit:.0f} s'

        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'planner_state'
        m.id = 0
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = pose.x, pose.y, 2.5
        m.pose.orientation.w = 1.0
        m.scale.z = 0.6
        m.color.r = m.color.g = m.color.b = m.color.a = 1.0
        m.text = text
        markers = MarkerArray()
        markers.markers.append(m)
        self.state_marker_pub_.publish(markers)

    ###########################################################
    # Artefacts as obstacles for Nav2

    def publish_artifact_obstacles(self):
        """Publish a disc of points (map frame, 0.3 m up) around every trusted artefact.
        Nav2 marks them in a costmap layer of their own, which the lidar cannot clear."""
        if point_cloud2 is None:
            return
        radius = self.get_parameter('artifact_obstacle_radius').value
        min_obs = self.get_parameter('min_observations').value
        step = 0.15
        n = int(radius / step)
        offsets = [(i * step, j * step) for i in range(-n, n + 1) for j in range(-n, n + 1)
                   if (i * step) ** 2 + (j * step) ** 2 <= radius ** 2]
        points = [(float(a['x'] + dx), float(a['y'] + dy), 0.3)
                  for a in self.artifacts_.values() if a.get('n_obs', min_obs) >= min_obs
                  for dx, dy in offsets]
        header = Header()
        header.frame_id = 'map'
        header.stamp = self.get_clock().now().to_msg()
        self.artifact_obstacle_pub_.publish(point_cloud2.create_cloud_xyz32(header, points))

    ###########################################################
    # Getting unstuck

    def start_backup(self, outcome):
        """Reverse a short distance with Nav2's back-up behaviour. True if it was sent."""
        if not self.backup_client_.server_is_ready():
            self.get_logger().warn('Robot is wedged but the backup action is not available')
            return False
        distance = self.get_parameter('backup_distance').value
        self.get_logger().warn(f'Goal failed and the robot moved only {self.last_goal_moved_:.1f} m: '
                               f'backing up {distance:.1f} m')
        goal = BackUp.Goal()
        goal.target.x = float(distance)   # distance to travel; the behaviour drives it in reverse
        goal.speed = 0.15
        goal.time_allowance = DurationMsg(sec=10)
        self.pending_outcome_ = outcome
        self.last_goal_moved_ = None
        self.ready_for_next_goal_ = False
        self.backup_start_ = self.now_sec()
        future = self.backup_client_.send_goal_async(goal)
        future.add_done_callback(self.backup_response_callback)
        return True

    def backup_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Back-up rejected')
            self.backup_done(None)
            return
        goal_handle.get_result_async().add_done_callback(
            lambda result_future: self.backup_done(result_future.result().status))

    def backup_done(self, status):
        """Back-up finished (or gave up): hand the held-back outcome to the state handler"""
        if self.backup_start_ is None:
            return   # already handled (late callback after the 20 s guard)
        ok = status == GoalStatus.STATUS_SUCCEEDED
        self.get_logger().info(f'Back-up {"finished" if ok else "did not complete"} (status {status})')
        self.backup_start_ = None
        self.goal_outcome_ = self.pending_outcome_
        self.pending_outcome_ = None
        self.ready_for_next_goal_ = True

    ###########################################################
    # Close-range inspection (Planning 2)

    def now_sec(self):
        """Node clock (sim time) in seconds, the same clock the detector uses for last_seen"""
        return self.get_clock().now().nanoseconds * 1e-9

    def artifacts_callback(self, msg):
        """Keep the latest report of every artefact, keyed by the detector's id"""
        try:
            reports = json.loads(msg.data)
        except ValueError:
            self.get_logger().warn('Could not parse /artifacts JSON', throttle_duration_sec=5.0)
            return
        for report in reports:
            if not isinstance(report, dict) or not all(k in report for k in ('id', 'type', 'x', 'y')):
                self.get_logger().warn(f'Ignoring malformed artefact report: {report}', throttle_duration_sec=5.0)
                continue
            if report['id'] not in self.artifacts_:
                self.get_logger().info(f"New artefact {report['id']} ({report['type']}) "
                                       f"at ({report['x']:.1f}, {report['y']:.1f})")
            self.artifacts_[report['id']] = report

    def artefact_locations_callback(self, msg):
        """Confirmed positions from the Perception node -> reports keyed by a stable id of our own.
        A position within perception_match_radius of a known artefact updates it; otherwise it is new."""
        radius = self.get_parameter('perception_match_radius').value
        for pose in msg.poses:
            x, y = pose.position.x, pose.position.y
            best_id, best_d = None, radius
            for artifact_id, a in self.artifacts_.items():
                d = math.hypot(a['x'] - x, a['y'] - y)
                if d < best_d:
                    best_id, best_d = artifact_id, d
            if best_id is None:
                artifact_id = f'p{self.next_perception_id_}'
                self.next_perception_id_ += 1
                # Perception only publishes artefacts it has seen at least 3 times, so it is trusted
                self.artifacts_[artifact_id] = {'id': artifact_id, 'type': 'artefact', 'x': x, 'y': y,
                                                'z': 0.5, 'n_obs': 3, 'last_seen': self.now_sec()}
                self.get_logger().info(f'New artefact {artifact_id} from Perception at ({x:.1f}, {y:.1f})')
            else:
                # A new dict, not an in-place edit, so the 'target moved' check can compare old and new
                self.artifacts_[best_id] = {**self.artifacts_[best_id], 'x': x, 'y': y}

    def artefact_detected_callback(self, msg):
        """Perception sees an artefact right now: mark the known artefacts in the camera's view as seen.
        This is what confirms an inspection (seen again while the robot waits at the standoff)."""
        if not msg.data:
            return
        pose = self.get_pose_2d()
        if pose is None:
            return
        now = self.now_sec()
        for artifact_id, a in self.artifacts_.items():
            dx, dy = a['x'] - pose.x, a['y'] - pose.y
            bearing = math.atan2(dy, dx) - pose.theta
            bearing = math.atan2(math.sin(bearing), math.cos(bearing))
            if math.hypot(dx, dy) <= 8.0 and abs(bearing) <= math.radians(60):   # 120 deg camera
                self.artifacts_[artifact_id] = {**a, 'last_seen': now}

    def pending_artifacts(self):
        """Artefacts seen often enough to trust, not yet inspected or given up on"""
        min_obs = self.get_parameter('min_observations').value
        return [a for a in self.artifacts_.values()
                if a['id'] not in self.inspected_ and a['id'] not in self.abandoned_
                and a.get('n_obs', min_obs) >= min_obs]

    def find_standoff(self, artifact, pose):
        """Best standoff pose for the artefact, or None if none is reachable on the current map"""
        info = self.map_msg_.info
        grid = np.array(self.map_msg_.data, dtype=np.int8).reshape(info.height, info.width)
        standoffs = self.inspection_planner_.standoffs(
            grid, info.resolution, info.origin.position.x, info.origin.position.y,
            (pose.x, pose.y), (artifact['x'], artifact['y']), failed=self.failed_standoffs_)
        return standoffs[0] if standoffs else None

    def next_inspection_target(self):
        """(artefact, standoff) for the nearest waiting artefact that has a reachable standoff, or None"""
        if self.map_msg_ is None:
            return None
        if self.next_inspection_check_ is not None and self.now_sec() < self.next_inspection_check_:
            return None
        pending = self.pending_artifacts()
        if not pending:
            return None
        pose = self.get_pose_2d()
        if pose is None:
            return None

        pending.sort(key=lambda a: math.hypot(a['x'] - pose.x, a['y'] - pose.y))
        for artifact in pending:
            self.failed_standoffs_ = []
            standoff = self.find_standoff(artifact, pose)
            if standoff is not None:
                return artifact, standoff

        # None reachable yet: keep exploring, the map may open a way later. Check again in 5 s.
        self.get_logger().info(f'{len(pending)} artefact(s) waiting, no standoff reachable yet')
        self.next_inspection_check_ = self.now_sec() + 5.0
        return None

    def start_inspection_if_needed(self):
        """If an artefact is waiting, drive to a standoff next to the nearest one. True if a goal was sent."""
        found = self.next_inspection_target()
        if found is None:
            return False
        artifact, standoff = found
        self.target_ = artifact
        self.target_attempts_ = 0
        self.failed_standoffs_ = []
        self.set_state(State.APPROACHING, f"artefact {artifact['id']} ({artifact['type']}) "
                                          f"at ({artifact['x']:.1f}, {artifact['y']:.1f})")
        self.send_standoff_goal(standoff)
        return True

    def send_standoff_goal(self, standoff):
        self.standoff_ = standoff
        x, y, yaw, path = standoff
        self.get_logger().info(f"Standoff [{x:.2f}, {y:.2f}] for artefact {self.target_['id']}, "
                               f"path {path:.1f} m, attempt {self.target_attempts_ + 1}")
        self.planner_go_to_pose2d(Pose2D(x=float(x), y=float(y), theta=float(yaw)))
        self.publish_inspection_markers()

    def state_approaching(self, outcome):
        """APPROACHING: the standoff goal has ended"""
        if outcome == GoalOutcome.PREEMPTED:
            # The detector moved the target: plan a standoff for the new position (not a failed attempt)
            pose = self.get_pose_2d()
            standoff = self.find_standoff(self.target_, pose) if pose is not None else None
            if standoff is not None:
                self.send_standoff_goal(standoff)
            else:
                self.finish_inspection(None, f"artefact {self.target_['id']}: no standoff after it moved")
        elif outcome == GoalOutcome.REACHED:
            self.inspect_start_ = self.now_sec()
            self.set_state(State.INSPECTING, f"at the standoff for artefact {self.target_['id']}")
        elif outcome in (GoalOutcome.ABORTED, GoalOutcome.CANCELLED):
            self.retry_or_give_up(f'approach failed ({outcome.name})', final_result=None)

    def state_inspecting(self):
        """INSPECTING: wait at the standoff, then check the detector saw the artefact while we were here"""
        if self.now_sec() - self.inspect_start_ < self.get_parameter('inspect_dwell').value:
            return
        latest = self.artifacts_.get(self.target_['id'], self.target_)
        if latest.get('last_seen', -1.0) >= self.inspect_start_:
            self.finish_inspection(True, f"artefact {self.target_['id']} confirmed at close range")
        else:
            self.retry_or_give_up('not seen from the standoff', final_result=False)

    def retry_or_give_up(self, reason, final_result):
        """One more standoff for the same artefact, then give up.
        final_result: None = could not get there (abandoned), False = got there but not confirmed."""
        self.failed_standoffs_.append(self.standoff_[:2])
        self.target_attempts_ += 1
        pose = self.get_pose_2d()
        if self.target_attempts_ < 2 and pose is not None:
            standoff = self.find_standoff(self.target_, pose)
            if standoff is not None:
                self.get_logger().warn(f"Artefact {self.target_['id']}: {reason}, trying another standoff")
                if self.state_ != State.APPROACHING:
                    self.set_state(State.APPROACHING, f"retry for artefact {self.target_['id']}")
                self.send_standoff_goal(standoff)
                return
        self.finish_inspection(final_result, f"artefact {self.target_['id']}: {reason}, giving up")

    def finish_inspection(self, result, reason):
        """Record the result for the target and go back to exploring"""
        artifact_id = self.target_['id']
        if result is None:
            self.abandoned_.add(artifact_id)
        else:
            self.inspected_[artifact_id] = result
        confirmed = sum(1 for v in self.inspected_.values() if v)
        self.get_logger().info(f'Inspection finished: {reason}. Confirmed {confirmed}, '
                               f'unconfirmed {len(self.inspected_) - confirmed}, '
                               f'abandoned {len(self.abandoned_)}, known {len(self.artifacts_)}')
        self.target_ = None
        self.standoff_ = None
        self.publish_inspection_markers()
        self.set_state(State.EXPLORING, reason)

    def publish_inspection_markers(self):
        """RViz: current target and standoff, plus every artefact coloured by inspection result"""

        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        groups = {'confirmed': ([], (0.0, 1.0, 0.0)), 'unconfirmed': ([], (1.0, 0.6, 0.0)),
                  'abandoned': ([], (1.0, 0.0, 0.0)), 'pending': ([], (0.6, 0.6, 1.0))}
        for a in self.artifacts_.values():
            if a['id'] in self.abandoned_:
                key = 'abandoned'
            elif a['id'] in self.inspected_:
                key = 'confirmed' if self.inspected_[a['id']] else 'unconfirmed'
            else:
                key = 'pending'
            groups[key][0].append(Point(x=float(a['x']), y=float(a['y']), z=1.5))

        for i, (ns, (points, rgb)) in enumerate(groups.items()):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = ns
            m.id = i
            m.type = Marker.SPHERE_LIST
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.7
            m.color.r, m.color.g, m.color.b = rgb
            m.color.a = 1.0
            m.points = points
            markers.markers.append(m)

        if self.standoff_ is not None:
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = 'standoff'
            m.id = 0
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.pose = pose2d_to_pose(Pose2D(x=float(self.standoff_[0]), y=float(self.standoff_[1]),
                                           theta=float(self.standoff_[2])))
            m.pose.position.z = 0.3
            m.scale.x, m.scale.y, m.scale.z = 1.0, 0.25, 0.25
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 1.0, 1.0
            markers.markers.append(m)

        self.inspection_marker_pub_.publish(markers)

    ###########################################################

    def publish_frontier_markers(self, frontiers, best, info):
        """RViz: frontier cells by status, the chosen goal, and blacklisted / visited goals"""

        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        def new_marker(ns, marker_id, marker_type, rgb, scale):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = ns
            m.id = marker_id
            m.type = marker_type
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = scale
            m.color.r, m.color.g, m.color.b = rgb
            m.color.a = 1.0
            return m

        # Frontier cells, one POINTS marker per status (tiny clusters are left out as noise)
        colours = {'chosen': (0.0, 1.0, 0.0), 'candidate': (0.0, 0.5, 1.0),
                   'unreachable': (1.0, 0.5, 0.0), 'blacklisted': (0.5, 0.0, 0.0)}
        for i, (status, rgb) in enumerate(colours.items()):
            m = new_marker('frontiers', i, Marker.POINTS, rgb, info.resolution)
            for f in frontiers:
                if f.status == status:
                    for r, c in f.cells:
                        x, y = FrontierExplorer.cell_to_world(r, c, info.origin.position.x,
                                                              info.origin.position.y, info.resolution)
                        m.points.append(Point(x=float(x), y=float(y), z=0.1))
            markers.markers.append(m)

        # Chosen goal as an arrow along its heading
        if best is not None:
            m = new_marker('frontier_goal', 0, Marker.ARROW, (1.0, 1.0, 0.0), 1.0)
            m.scale.y = m.scale.z = 0.25
            m.pose = pose2d_to_pose(Pose2D(x=float(best.goal[0]), y=float(best.goal[1]), theta=float(best.goal[2])))
            m.pose.position.z = 0.3
            markers.markers.append(m)

        # Goals not to be chosen again: blacklisted (red) and visited (grey)
        for ns, points, rgb in (('blacklisted_goals', self.blacklist_, (1.0, 0.0, 0.0)),
                                ('visited_goals', self.visited_goals_, (0.6, 0.6, 0.6))):
            m = new_marker(ns, 0, Marker.SPHERE_LIST, rgb, 0.6)
            m.points = [Point(x=float(x), y=float(y), z=0.3) for x, y in points]
            markers.markers.append(m)

        self.planning_marker_pub_.publish(markers)

    def state_exploring_scripted(self, outcome):
        """
        EXPLORING with the template's scripted route: first artifact, home, then random goals.
        Kept as a fallback (exploration:=scripted) and for comparison with the frontier explorer.
        """

        # 1. Account for how the previous goal ended
        if outcome == GoalOutcome.REACHED:
            self.consecutive_failures_ = 0
            if self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT:
                self.get_logger().info('Reached first artifact')
            elif self.planner_type_ == PlannerType.RETURN_HOME:
                self.get_logger().info('Returned home')
            self.finish_route_stage()
        elif outcome in (GoalOutcome.ABORTED, GoalOutcome.CANCELLED):
            self.consecutive_failures_ += 1
            self.get_logger().warn(f'{self.planner_type_.name} goal failed ({outcome.name}), '
                                   f'{self.consecutive_failures_} failure(s) in a row')
            # A scripted goal gets one retry, then is abandoned so the route cannot stall on it.
            # Random goals are never retried: the next pick is a different attempt anyway.
            if self.consecutive_failures_ >= 2 and self.planner_type_ != PlannerType.RANDOM_GOAL:
                self.get_logger().warn(f'Abandoning {self.planner_type_.name} after '
                                       f'{self.consecutive_failures_} failures')
                self.consecutive_failures_ = 0
                self.finish_route_stage()

        # 2. Choose the next planner (after a single failure this picks the same one: a retry)
        if not self.first_artifact_done_:
            self.planner_type_ = PlannerType.GO_TO_FIRST_ARTIFACT
        elif not self.return_home_done_:
            self.planner_type_ = PlannerType.RETURN_HOME
        else:
            self.planner_type_ = PlannerType.RANDOM_GOAL

        # 3. Run it
        self.get_logger().info(f'Calling planner: {self.planner_type_.name}')
        if self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT:
            self.planner_go_to_first_artifact()
        elif self.planner_type_ == PlannerType.RETURN_HOME:
            self.planner_return_home()
        else:
            self.planner_random_goal()

    def finish_route_stage(self):
        """Mark the current scripted stage as done (reached or abandoned)"""

        if self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT:
            self.first_artifact_done_ = True
        elif self.planner_type_ == PlannerType.RETURN_HOME:
            self.return_home_done_ = True


def main():
    # Initialise
    rclpy.init()

    # Create the cave explorer
    cave_explorer = CaveExplorer()

    # Spin until Ctrl-C, then shut down cleanly
    try:
        rclpy.spin(cave_explorer)
    except KeyboardInterrupt:
        pass
    finally:
        cave_explorer.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()