"""
perception.launch.py - runs the artefact perception node (Perception 1, 2 and 3).

    ros2 launch cave_explorer perception.launch.py
    ros2 launch cave_explorer perception.launch.py save_images:=true
    ros2 launch cave_explorer perception.launch.py camera_x_offset:=0.3 confidence:=0.4

A standalone launch file: it is not included in, and does not include, any cave_explorer launch file.
Run it in its own terminal alongside the startup and navigation launches (and optionally autonomy or teleop).
It needs the simulator (camera) and the navigation launch (SLAM provides map -> base_link).
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = {
        # Change these defaults to suit your setup so you don't have to pass them every time
        'model': os.path.expanduser('~/ros2_ws/src/cave_explorer/artefact_detector/best.pt'),
        'confidence': '0.5',
        'save_images': 'false',
        'dataset_dir': os.path.expanduser('~/cave_dataset'),
        'camera_x_offset': '0.0',
        'image_topic': 'camera/image',
        'depth_topic': 'camera/depth/image',
        'camera_info_topic': 'camera/camera_info',
    }

    perception_node = Node(
        package='cave_explorer',
        executable='artefact_perception',
        name='artefact_perception_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'yolo_model_path': LaunchConfiguration('model'),
            'detection_confidence': LaunchConfiguration('confidence'),
            'save_images': LaunchConfiguration('save_images'),
            'dataset_dir': LaunchConfiguration('dataset_dir'),
            'camera_x_offset': LaunchConfiguration('camera_x_offset'),
            'image_topic': LaunchConfiguration('image_topic'),
            'depth_topic': LaunchConfiguration('depth_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
        }],
    )

    return LaunchDescription(
        [DeclareLaunchArgument(name, default_value=value) for name, value in args.items()]
        + [perception_node]
    )
