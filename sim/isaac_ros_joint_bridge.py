import omni.graph.core as og
import usdrt.Sdf
from isaacsim.core.utils import extensions


def enable_ros2_bridge_extension() -> None:
    extensions.enable_extension("isaacsim.ros2.bridge")


def create_ros2_joint_graph(
    articulation_path: str = "/Franka",
    graph_path: str = "/World/ROS/FrankaJointGraph",
    joint_state_topic: str = "/joint_states",
    joint_command_topic: str = "/joint_command",
):
    """Bridge Isaac Sim articulation joint state I/O to ROS 2 topics."""
    enable_ros2_bridge_extension()

    keys = og.Controller.Keys
    graph, _, _, _ = og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "execution",
        },
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
            ],
            keys.SET_VALUES: [
                ("PublishJointState.inputs:topicName", joint_state_topic),
                ("SubscribeJointState.inputs:topicName", joint_command_topic),
                ("ArticulationController.inputs:robotPath", articulation_path),
                ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path(articulation_path)]),
            ],
        },
    )

    og.Controller.evaluate_sync(graph)
    return {
        "graph": graph,
        "articulation_path": articulation_path,
        "joint_state_topic": joint_state_topic,
        "joint_command_topic": joint_command_topic,
    }
