from dataclasses import dataclass

import omni.graph.core as og
import omni.syntheticdata
import usdrt.Sdf
from isaacsim.core.utils import extensions


@dataclass
class RosCameraBridgeConfig:
    camera_path: str
    graph_path: str
    frame_id: str
    width: int = 640
    height: int = 480
    rgb_topic: str = "/camera/image_raw"
    depth_topic: str = "/rgbd_camera/depth_image"
    camera_info_topic: str = "/rgbd_camera/camera_info"
    rgb_step: int = 1
    depth_step: int = 1
    camera_info_step: int = 1


def enable_ros2_bridge_extension() -> None:
    extensions.enable_extension("isaacsim.ros2.bridge")


def create_ros2_camera_graph(config: RosCameraBridgeConfig):
    """Create a ROS2 camera publish graph for an existing Isaac Sim camera prim."""
    enable_ros2_bridge_extension()

    keys = og.Controller.Keys
    graph, _, _, _ = og.Controller.edit(
        {
            "graph_path": config.graph_path,
            "evaluator_name": "execution",
        },
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("cameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("cameraHelperInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                ("cameraHelperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
                ("CreateRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
                ("CreateRenderProduct.outputs:renderProductPath", "cameraHelperInfo.inputs:renderProductPath"),
                ("CreateRenderProduct.outputs:renderProductPath", "cameraHelperDepth.inputs:renderProductPath"),
            ],
            keys.SET_VALUES: [
                ("CreateRenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path(config.camera_path)]),
                ("CreateRenderProduct.inputs:width", int(config.width)),
                ("CreateRenderProduct.inputs:height", int(config.height)),
                ("cameraHelperRgb.inputs:frameId", config.frame_id),
                ("cameraHelperRgb.inputs:topicName", config.rgb_topic),
                ("cameraHelperRgb.inputs:type", "rgb"),
                ("cameraHelperInfo.inputs:frameId", config.frame_id),
                ("cameraHelperInfo.inputs:topicName", config.camera_info_topic),
                ("cameraHelperDepth.inputs:frameId", config.frame_id),
                ("cameraHelperDepth.inputs:topicName", config.depth_topic),
                ("cameraHelperDepth.inputs:type", "depth"),
            ],
        },
    )

    og.Controller.evaluate_sync(graph)

    render_product_path = og.Controller.attribute(
        f"{config.graph_path}/CreateRenderProduct.outputs:renderProductPath"
    ).get()

    return {
        "graph": graph,
        "render_product_path": render_product_path,
        "gate_paths": _set_publish_gate_steps(
            render_product_path=render_product_path,
            rgb_step=config.rgb_step,
            depth_step=config.depth_step,
            camera_info_step=config.camera_info_step,
        ),
    }


def _set_publish_gate_steps(render_product_path: str, rgb_step: int, depth_step: int, camera_info_step: int):
    import omni.syntheticdata._syntheticdata as sd

    rv_rgb = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
    rv_depth = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.DistanceToImagePlane.name)

    rgb_gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv_rgb + "IsaacSimulationGate",
        render_product_path,
    )
    depth_gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv_depth + "IsaacSimulationGate",
        render_product_path,
    )
    camera_info_gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        "PostProcessDispatch" + "IsaacSimulationGate",
        render_product_path,
    )

    _set_gate_step_if_available(rgb_gate_path, rgb_step)
    _set_gate_step_if_available(depth_gate_path, depth_step)
    _set_gate_step_if_available(camera_info_gate_path, camera_info_step)

    return {
        "rgb": rgb_gate_path,
        "depth": depth_gate_path,
        "camera_info": camera_info_gate_path,
    }


def _set_gate_step_if_available(gate_path: str, step: int) -> None:
    try:
        og.Controller.attribute(gate_path + ".inputs:step").set(max(1, int(step)))
    except Exception:
        # Some render product combinations do not expose the expected
        # IsaacSimulationGate path immediately. Do not fail scene setup here.
        pass


def build_top_view_bridge(camera_path: str):
    return create_ros2_camera_graph(
        RosCameraBridgeConfig(
            camera_path=camera_path,
            graph_path="/World/ROS/TopViewCameraGraph",
            frame_id="top_view_camera",
            width=640,
            height=480,
            rgb_topic="/camera/image_raw",
            depth_topic="/rgbd_camera/depth_image",
            camera_info_topic="/rgbd_camera/camera_info",
            rgb_step=1,
            depth_step=1,
            camera_info_step=1,
        )
    )


def build_ee_view_bridge(camera_path: str):
    return create_ros2_camera_graph(
        RosCameraBridgeConfig(
            camera_path=camera_path,
            graph_path="/World/ROS/EEViewCameraGraph",
            frame_id="ee_view_camera",
            width=640,
            height=480,
            rgb_topic="/ee_camera/image_raw",
            depth_topic="/ee_rgbd_camera/depth_image",
            camera_info_topic="/ee_rgbd_camera/camera_info",
            rgb_step=1,
            depth_step=1,
            camera_info_step=1,
        )
    )
