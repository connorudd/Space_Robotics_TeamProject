from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Local copy so the launch file does not need to import the package.
FAULT_MODES = ['none', 'dropout', 'noise', 'late', 'false_positive']


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    fault_mode = LaunchConfiguration('fault_mode')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='True',
                              description='Use simulation clock'),
        DeclareLaunchArgument('fault_mode', default_value='none', choices=FAULT_MODES,
                              description='Detector fault to simulate'),
        Node(
            package='cave_explorer',
            executable='artifact_stub',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time, 'fault_mode': fault_mode}],
        ),
    ])
