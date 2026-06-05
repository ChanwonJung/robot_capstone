"""Hybrid planning bring-up for the stop+resume / avoidance demo.

Runs the MoveIt hybrid planning components (global + local + manager) wired to
Isaac Sim via a Float64MultiArray->/joint_command adapter. The local planner
subscribes to /collision_object (fed by hazard_collision_injector_node), so the
arm reacts when a hazard blocks its path.

Manager logic is selectable via the manager_logic launch arg:
  stop_resume (default) -- SinglePlanExecution: halt in place on a collision and
                           resume once it clears (transient hazard, e.g. a flying
                           bottle).
  replan                -- ReplanInvalidatedTrajectory: on a collision, re-plan
                           the global trajectory from the current state so OMPL
                           routes AROUND a persistent hazard (e.g. an arm parked
                           in the EE path). Relies on the global planner's PSM
                           also seeing /collision_object (it does — both the
                           global and local planner private nodes subscribe).

This REPLACES panda_isaac_moveit (do not run both — they would both drive the
planning scene). Run alongside:
  - Isaac Sim scene (/joint_states, /joint_command, cameras, depth, the bottle)
  - YOLO top hazard node + hazard_collision_injector_node
  - a goal source publishing PoseStamped on /grasp_target_pose
    (the bt_pkg BT owns goal submission to /run_hybrid_planning)

Components:
  global_planner / local_planner / hybrid_planning_manager  (composable container)
  robot_state_publisher, static_tf (world->panda_link0), RViz
  hybrid_command_bridge_node  (local planner output -> Isaac /joint_command)
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def _load_yaml(package_name, file_path):
    abs_path = os.path.join(get_package_share_directory(package_name), file_path)
    try:
        with open(abs_path, "r") as f:
            return yaml.safe_load(f)
    except EnvironmentError:
        return None


def _manager_setup(context, *args, **kwargs):
    """Load the Hybrid Planning Manager with the logic plugin chosen at runtime.

    manager_logic:=stop_resume -> hybrid_planning_manager.yaml
                                  (SinglePlanExecution: halt + resume)
    manager_logic:=replan      -> hybrid_planning_manager_replan.yaml
                                  (ReplanInvalidatedTrajectory: re-plan around)

    The manager connects to the global/local planner action servers during its
    own init, so it must load AFTER they are up — otherwise it aborts with
    "Global planner action server not available after waiting". Hence the 8 s
    delay before loading it into the (already running) container.
    """
    logic = LaunchConfiguration("manager_logic").perform(context)
    fname = (
        "config/hybrid/hybrid_planning_manager_replan.yaml"
        if logic == "replan"
        else "config/hybrid/hybrid_planning_manager.yaml"
    )
    manager_param = _load_yaml("moveit_isaac_bridge_pkg", fname)
    manager_node = ComposableNode(
        package="moveit_hybrid_planning",
        plugin="moveit::hybrid_planning::HybridPlanningManager",
        name="hybrid_planning_manager",
        parameters=[manager_param, {"use_sim_time": False}],
    )
    return [
        TimerAction(
            period=8.0,
            actions=[
                LoadComposableNodes(
                    target_container="/hybrid_planning_container",
                    composable_node_descriptions=[manager_node],
                )
            ],
        )
    ]


def generate_launch_description():
    start_rviz_arg = DeclareLaunchArgument("start_rviz", default_value="true")
    rviz_config_arg = DeclareLaunchArgument("rviz_config", default_value="moveit.rviz")
    manager_logic_arg = DeclareLaunchArgument(
        "manager_logic",
        default_value="stop_resume",
        choices=["stop_resume", "replan"],
        description="Hybrid manager logic: 'stop_resume' (halt + resume) or "
        "'replan' (re-plan around a persistent hazard).",
    )

    bridge_pkg_share = get_package_share_directory("moveit_isaac_bridge_pkg")
    moveit_controllers = os.path.join(bridge_pkg_share, "config", "moveit_controllers.yaml")

    moveit_config = (
        MoveItConfigsBuilder(
            "moveit_resources_panda",
            package_name="moveit_resources_panda_moveit_config",
        )
        .robot_description(file_path="config/panda.urdf.xacro")
        .robot_description_semantic(file_path="config/panda.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .trajectory_execution(file_path=moveit_controllers)
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    global_planner_param = _load_yaml("moveit_isaac_bridge_pkg", "config/hybrid/global_planner.yaml")
    # Local planner config is selected at runtime by manager_logic — see
    # _build_local_planner below. The replan profile uses a higher
    # local_planning_frequency than stop_resume so the reactive layer
    # detects an injected obstacle before the arm crosses into its
    # collision volume (which would crash the planning pipeline through
    # OMPL CheckStartStateCollision → manager double-abort → SIGABRT).
    local_planner_yaml_default = _load_yaml(
        "moveit_isaac_bridge_pkg", "config/hybrid/local_planner.yaml")
    local_planner_yaml_replan = _load_yaml(
        "moveit_isaac_bridge_pkg", "config/hybrid/local_planner_replan.yaml")
    # Isaac in this scene does NOT publish /clock, so the whole stack runs on wall
    # time. use_sim_time:True would (a) crash the composable container on the
    # /clock subscription QoS override and (b) stall every node at time 0.
    sim_time = {"use_sim_time": False}

    # /joint_states is the canonical wall-time topic (joint_state_restamp_node
    # republishes Isaac's /joint_states_isaac here), so the planners + RSP just use
    # /joint_states directly — no remap needed.
    #
    # NOTE: an earlier attempt to add FixStartStateCollision to the OMPL
    # request_adapters list to recover from multi-link contact at the local
    # stop position was reverted — that adapter was removed from MoveIt 2 in
    # Jazzy (only Check* and the two non-Fix utility adapters remain), so
    # registering it crashes the container at plugin load.
    #
    # Instead we shrink the collision-check geometry of the gripper links
    # via negative link_padding. MoveIt's robot collision model adds this
    # padding around every link (positive = safety buffer); a negative value
    # is treated as a tight-fit shrink, so the planner sees the fingers as
    # ~1 cm smaller than they really are. Combined with the hazard box's
    # negative xy_margin (in hazard_collision_injector.launch.py), the
    # gripper can pass right next to the parked bottle without OMPL
    # registering a collision, which is what was crashing the manager.
    link_padding_override = {
        "robot_description_planning.link_padding.panda_leftfinger": -0.01,
        "robot_description_planning.link_padding.panda_rightfinger": -0.01,
        "robot_description_planning.link_padding.panda_hand": -0.005,
    }
    global_planner_node = ComposableNode(
        package="moveit_hybrid_planning",
        plugin="moveit::hybrid_planning::GlobalPlannerComponent",
        name="global_planner",
        parameters=[
            global_planner_param,
            moveit_config.to_dict(),
            link_padding_override,
            sim_time,
        ],
    )
    # local_planner is loaded AFTER the container is up, via _build_local_planner.
    # That OpaqueFunction reads manager_logic at runtime and picks the matching
    # yaml; baking local into the container's initial descriptions would force
    # us to pick one of the two yamls statically.
    def _build_local_planner(context):
        logic = LaunchConfiguration("manager_logic").perform(context)
        local_param = (
            local_planner_yaml_replan
            if logic == "replan"
            else local_planner_yaml_default
        )
        local_planner_node = ComposableNode(
            package="moveit_hybrid_planning",
            plugin="moveit::hybrid_planning::LocalPlannerComponent",
            name="local_planner",
            # Local needs the same link_padding shrink as global so its
            # collision check during ForwardTrajectory advance agrees with
            # the path the global planner just produced — otherwise local
            # stops the arm at the (un-shrunk) finger envelope and we're
            # back to the holding/SIGABRT cycle.
            parameters=[
                local_param,
                moveit_config.to_dict(),
                link_padding_override,
                sim_time,
            ],
        )
        # Load before the manager (manager has its own 8 s TimerAction) so
        # the manager's init can find local_planner's action server.
        return [
            TimerAction(
                period=2.0,
                actions=[
                    LoadComposableNodes(
                        target_container="/hybrid_planning_container",
                        composable_node_descriptions=[local_planner_node],
                    )
                ],
            )
        ]

    container = ComposableNodeContainer(
        name="hybrid_planning_container",
        namespace="/",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[global_planner_node],
        output="screen",
    )

    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "panda_link0"],
    )

    # Re-stamp Isaac joint states to wall time so MoveIt accepts them.
    restamp_node = Node(
        package="moveit_isaac_bridge_pkg",
        executable="joint_state_restamp_node",
        name="joint_state_restamp_node",
        output="screen",
        parameters=[sim_time],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description, sim_time],
    )

    command_bridge = Node(
        package="moveit_isaac_bridge_pkg",
        executable="hybrid_command_bridge_node",
        name="hybrid_command_bridge_node",
        output="screen",
        parameters=[sim_time],
    )

    # NOTE: hybrid_pose_client_node is intentionally omitted here.
    # The bt_pkg bt_executor_node now owns goal submission to /run_hybrid_planning
    # via its MoveAction BT node. Running both would cause double goal submissions
    # and hybrid planner thrash ("Unknown event" crashes).

    # Gripper action server — serves /gripper_command (control_msgs/GripperCommand)
    # for the BT's GripperAction node. Uses MoveIt panda_hand group + /joint_states
    # contact detection.
    gripper_server = Node(
        package="moveit_isaac_bridge_pkg",
        executable="gripper_action_server",
        name="gripper_action_server",
        output="screen",
        parameters=[sim_time],
    )

    rviz_config = PathJoinSubstitution(
        [FindPackageShare("moveit_isaac_bridge_pkg"), "config", LaunchConfiguration("rviz_config")]
    )
    rviz_node = Node(
        condition=IfCondition(LaunchConfiguration("start_rviz")),
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            sim_time,
        ],
    )

    return LaunchDescription(
        [
            start_rviz_arg,
            rviz_config_arg,
            manager_logic_arg,
            static_tf_node,
            restamp_node,
            robot_state_publisher,
            container,
            OpaqueFunction(function=_build_local_planner),
            OpaqueFunction(function=_manager_setup),
            command_bridge,
            gripper_server,
            rviz_node,
        ]
    )
