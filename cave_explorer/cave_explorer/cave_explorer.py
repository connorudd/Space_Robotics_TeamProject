#!/usr/bin/env python3

import math
import random
from enum import Enum

import cv2  # OpenCV2
import rclpy
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Pose2D, PoseStamped, Point
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


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


class CaveExplorer(Node):
    def __init__(self):
        super().__init__('cave_explorer_node')

        # Variables/Flags for mapping
        self.xlim_ = [0.0, 0.0]
        self.ylim_ = [0.0, 0.0]

        # Variables/Flags for perception
        self.artifact_found_ = False

        # Variables/Flags for planning
        self.planner_type_ = PlannerType.ERROR
        self.reached_first_artifact_ = False
        self.returned_home_ = False

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
        self.first_distance_ = None      # first distance_remaining of the active goal (for the end-of-goal log)
        self.goal_start_pose_ = None     # robot pose when the active goal was accepted

        # Publisher for the goal pose visualisation
        self.current_goal_pub_ = self.create_publisher(PoseStamped, 'current_goal', 1)

        # Subscribe to the map topic to get current bounds
        self.map_sub_ = self.create_subscription(OccupancyGrid, 'map',  self.map_callback, 1)

        # Prepare image processing
        self.image_detections_pub_ = self.create_publisher(Image, 'detections_image', 1)
        self.declare_parameter('computer_vision_model_filename', rclpy.Parameter.Type.STRING)
        self.computer_vision_model_ = cv2.CascadeClassifier(self.get_parameter('computer_vision_model_filename').value)
        self.image_sub_ = self.create_subscription(Image, 'camera/image', self.image_callback, 1)

        # Timer for main loop
        self.main_loop_timer_ = self.create_timer(0.2, self.main_loop)
    
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
        self.first_distance_ = None
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

        # Report what actually happened, not just that the goal ended
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Goal reached!')
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warn('Goal aborted by Nav2')
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warn('Goal cancelled')
        else:
            self.get_logger().error(f'Goal ended with unexpected status {status}')

        # Record how the goal ended, so an abort or cancel can be diagnosed from the log
        feedback = self.last_feedback_
        if feedback is not None:
            # Path distance went from first_distance_ to remaining; moved is the straight-line
            # displacement of the robot. Moved far but remaining barely changed = detour, not stuck.
            first = f'{self.first_distance_:.1f}' if self.first_distance_ is not None else '?'
            end_pose = self.get_pose_2d()
            if self.goal_start_pose_ is not None and end_pose is not None:
                moved = f'{math.hypot(end_pose.x - self.goal_start_pose_.x, end_pose.y - self.goal_start_pose_.y):.1f}'
            else:
                moved = '?'
            self.get_logger().info(f'Goal ended with {feedback.distance_remaining:.2f} m remaining '
                                   f'(started at {first} m, robot moved {moved} m), '
                                   f'{feedback.number_of_recoveries} recovery attempt(s)')
        self.last_feedback_ = None
        self.cancel_reason_ = None
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

        # Select a random location
        goal_valid = False
        while not goal_valid:
            idx = random.randint(0,len(random_goals)-1)
            goal_x = random_goals[idx][0]
            goal_y = random_goals[idx][1]

            # Only accept this goal if it's within the current costmap bounds
            if goal_x > self.xlim_[0] and goal_x < self.xlim_[1] and \
               goal_y > self.ylim_[0] and goal_y < self.ylim_[1]:
                goal_valid = True
            else:
                self.get_logger().warn(f'Goal [{goal_x}, {goal_y}] out of bounds')

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

        # Check if previous goal still running
        if not self.ready_for_next_goal_:
            # self.get_logger().info(f'Previous goal still running')
            return

        self.ready_for_next_goal_ = False

        if self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT:
            self.get_logger().info('Successfully reached first artifact!')
            self.reached_first_artifact_ = True
        if self.planner_type_ == PlannerType.RETURN_HOME:
            self.get_logger().info('Successfully returned home!')
            self.returned_home_ = True

        #######################################################
        # Select the next planner to execute
        # Update this logic as you see fit!
        if not self.reached_first_artifact_:
            self.planner_type_ = PlannerType.GO_TO_FIRST_ARTIFACT
        elif not self.returned_home_:
            self.planner_type_ = PlannerType.RETURN_HOME
        else:
            self.planner_type_ = PlannerType.RANDOM_GOAL

        #######################################################
        # Execute the planner by calling the relevant method
        # Add your own planners here!
        self.get_logger().info(f'Calling planner: {self.planner_type_.name}')
        if self.planner_type_ == PlannerType.MOVE_FORWARDS:
            self.planner_move_forwards(10)
        elif self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT:
            self.planner_go_to_first_artifact()
        elif self.planner_type_ == PlannerType.RETURN_HOME:
            self.planner_return_home()
        elif self.planner_type_ == PlannerType.RANDOM_WALK:
            self.planner_random_walk()
        elif self.planner_type_ == PlannerType.RANDOM_GOAL:
            self.planner_random_goal()
        else:
            self.get_logger().error('No valid planner selected')
            self.destroy_node()


        #######################################################

def main():
    # Initialise
    rclpy.init()

    # Create the cave explorer
    cave_explorer = CaveExplorer()

    while rclpy.ok():
        rclpy.spin(cave_explorer)