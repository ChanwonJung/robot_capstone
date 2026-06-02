"""Launch the hazard collision injector.

Turns top-view YOLO hazard detections into MoveIt collision objects on
/collision_object. Run alongside the YOLO hazard nodes (yolo_hazard_pkg), the
Isaac camera bridge (depth + camera_info), and move_group.

Scenario selection: pass `scenario:=replan` (default) or `scenario:=stop_resume`
on the command line. The two profiles invert the xy_margin and stability-
filter polarities because the underlying race is different:

  replan (Phase 2b — persistent parked hazard):
    NEGATIVE xy_margin shrinks the planning box ~2 cm inside the real bottle
    so the global re-plan has clean start state even after execution drift
    nudges the fingertip a few mm into the bottle's actual volume. Box
    position is LATCHED on first publish (large stability threshold) so the
    parked hazard is one stable scene update for the whole demo.

  stop_resume (Phase 2a — transient flythrough):
    POSITIVE xy_margin inflates the planning box ~5 cm beyond the real box
    so ForwardTrajectory halts the arm a clear, visible distance from the
    moving hazard rather than only the last centimetre before contact. Box
    position TRACKS the moving hazard frame-to-frame (small stability
    threshold) so the local sees the obstacle leave the trajectory the
    moment the box clears the workspace.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


_REPLAN_PARAMS = {
    "xy_margin": -0.04,
    "default_obstacle_height": 0.22,
    "stable_position_threshold": 0.05,
}

_STOP_RESUME_PARAMS = {
    # Inflate the box so ForwardTrajectory halts the arm with a visible
    # safety buffer to the moving hazard. The hold is naturally long enough
    # (local_planner.yaml runs at 2 Hz) to wait out the flythrough.
    "xy_margin": 0.08,
    "default_obstacle_height": 0.30,
    # Moving obstacle — keep the planning scene tracking it closely so the
    # box leaves the scene the moment YOLO loses detection.
    "stable_position_threshold": 0.01,
    # Hold the collision_object in the planning scene for this long after
    # the last YOLO detection. Empirically the ForwardTrajectory abort
    # threshold is ~1 stuck iter at 2 Hz (~500 ms total budget), so the
    # collision_object lifetime (= transit + clear_timeout) must stay
    # under that. With box transit ~200 ms at -2.5 m/s, 0.15 s puts total
    # in-scene time ≈ 350 ms — leaves ~150 ms abort margin while still
    # giving the halt a visible duration.
    "clear_timeout_sec": 0.15,
}


def _build_node(context, *args, **kwargs):
    scenario = LaunchConfiguration("scenario").perform(context)
    if scenario == "stop_resume":
        params = _STOP_RESUME_PARAMS
    elif scenario == "replan":
        params = _REPLAN_PARAMS
    else:
        raise RuntimeError(
            f"hazard_collision_injector: unknown scenario '{scenario}' "
            "(expected 'replan' or 'stop_resume')"
        )
    return [
        Node(
            package="moveit_isaac_bridge_pkg",
            executable="hazard_collision_injector_node",
            name="hazard_collision_injector_node",
            output="screen",
            parameters=[params],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "scenario",
                default_value="replan",
                choices=["replan", "stop_resume"],
                description=(
                    "Which hazard scenario the injector is configured for — "
                    "match this to the manager_logic in hybrid_planning.launch.py."
                ),
            ),
            OpaqueFunction(function=_build_node),
        ]
    )
