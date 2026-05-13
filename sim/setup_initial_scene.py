import asyncio
import os
import sys
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.timeline
import omni.ui as ui
import omni.usd
from isaacsim.sensors.camera import Camera
from isaacsim.sensors.camera import SingleViewDepthSensorAsset
from isaacsim.core.utils.stage import open_stage
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.viewport.window import ViewportWindow, get_viewport_window_instances
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics


SIM_DIR = Path(__file__).resolve().parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from isaac_ros_camera_bridge import build_ee_view_bridge, build_top_view_bridge
from isaac_ros_joint_bridge import create_ros2_joint_graph

PROJECT_ROOT = SIM_DIR.parent
DOWNLOADS_DIR = Path(os.environ.get("ROBOT_CAPSTONE_DOWNLOADS_DIR", Path.home() / "Downloads")).expanduser()
XR_CONTENT_ROOT = Path(
    os.environ.get("ROBOT_CAPSTONE_XR_CONTENT_ROOT", DOWNLOADS_DIR / "XR_Content_NVD@10010")
).expanduser()
IMPORTED_ASSETS_DIR = SIM_DIR / "assets" / "imported"
ISAACSIM_ROOT = PROJECT_ROOT / "isaacsim"

SOURCE_STAGE = XR_CONTENT_ROOT / "Assets" / "XR" / "Stages" / "robot_capstone.usd"
OUTPUT_STAGE = XR_CONTENT_ROOT / "Assets" / "XR" / "Stages" / "robot_capstone_scene.usd"

APPLE_ASSET = IMPORTED_ASSETS_DIR / "Apple.usd"
USE_APPLE_MESH = True
USE_GLASS_MESH = True
RED_BALL_ASSET = IMPORTED_ASSETS_DIR / "Red_Ball.usd"
BLUE_CUBE_ASSET = ISAACSIM_ROOT / "extscache" / "omni.warp.core-1.8.2+lx64" / "warp" / "examples" / "assets" / "cube.usd"
BOOK_ASSET = IMPORTED_ASSETS_DIR / "Book_brown.usd"
GLASS_ASSET = XR_CONTENT_ROOT / "Assets" / "XR" / "Stages" / "Indoor" / "Modern_House" / "SubUSDs" / "P_Glassware_Short.usd"
BEDSIDE_TABLE_POSITION = np.array([3.3, -1.79, -0.73])
BEDSIDE_TABLE_ROTATION_DEG = np.array([0.0, 0.0, 25.0])
APPLE_TRANSLATE = np.array([-2.43, 3.18, 0.66])
APPLE_ROTATION_DEG = np.array([90.0, 0.0, 0.0])
APPLE_VISUAL_TRANSLATE = np.array([-4.691566, -83.639191, 65.429489])
APPLE_COLLIDER_TRANSLATE = np.array([0.0, 0.0, 1.85])
GLASS_TRANSLATE = np.array([-2.23, 3.03, 0.71])
GLASS_ROTATION_DEG = np.array([0.0, 0.0, 0.0])
GLASS_COLLIDER_TRANSLATE = np.array([0.0, 0.0, 4.5])
RED_BALL_TRANSLATE = np.array([-1.98, 2.97, 0.88])
RED_BALL_ROTATION_DEG = np.array([-90.0, 0.0, 0.0])
RED_BALL_VISUAL_TRANSLATE = np.array([2.981419, -1.258126, -0.150547])
RED_BALL_COLLIDER_TRANSLATE = np.array([0.0, 0.0, 0.02])
BLUE_CUBE_TRANSLATE = np.array([-2.41, 3.00, 0.77])
BLUE_CUBE_ROTATION_DEG = np.array([0.0, 0.0, 0.0])
BLUE_CUBE_COLLIDER_TRANSLATE = np.array([0.0, 0.0, 0.0])
BOOK_TRANSLATE = np.array([-2.11, 2.92, 0.748])
BOOK_ROTATION_DEG = np.array([90.0, 0.0, -68.0])
BOOK_COLLIDER_TRANSLATE = np.array([0.0, 0.0, 0.0])
APPLE_SCALE = np.array([0.001, 0.001, 0.001])
GLASS_SCALE = np.array([0.02, 0.02, 0.02])
RED_BALL_SCALE = np.array([0.05, 0.05, 0.05])
BLUE_CUBE_SCALE = np.array([0.05, 0.05, 0.05])
BOOK_SCALE = np.array([0.08, 0.08, 0.08])
TOP_CAMERA_POSITION = np.array([0.0, 0.2, 2.0])
TOP_CAMERA_ROTATION_DEG = np.array([0.0, 0.0, 0.0])
TOP_CAMERA_FOCAL_LENGTH_MM = 4.0
EE_CAMERA_PATH = "/Franka/panda_hand/EEViewCameraMount/CameraRig/CameraFrame/EEViewCamera"
TOP_CAMERA_PATH = "/World/TopViewCamera"
CAMERA_SENSOR_SCOPE = "/World/CameraSensors"
EE_DEPTH_SCOPE = f"{CAMERA_SENSOR_SCOPE}/EEViewDepth"
TOP_DEPTH_SCOPE = f"{CAMERA_SENSOR_SCOPE}/TopViewDepth"
EE_VIEWPORT_NAME = "EE View"
TOP_VIEWPORT_NAME = "Top View"
EE_VIEWPORT_RESOLUTION = (640, 480)
TOP_VIEWPORT_RESOLUTION = (640, 480)
WRIST_MOUNT_CANDIDATES = ["panda_hand", "panda_link7", "panda_link6"]
EE_MOUNT_FALLBACK_CANDIDATES = ["panda_hand", "gripper_center", "tool0", "ee_link", "right_gripper"]
EE_CAMERA_MOUNT_TRANSLATE = np.array([0.000, 0.0, 0.030])
EE_CAMERA_LOCAL_TRANSLATE = np.array([0.095, 0.0, -0.030])
EE_CAMERA_LOCAL_ROTATION_DEG = np.array([-160.0, 0.0, 90.0])
TABLETOP_OBJECT_PATHS = {
    "Apple": "/World/CapstoneAdditions/TabletopItems/Apple",
    "Glass": "/World/CapstoneAdditions/TabletopItems/Glass",
    "RedBall": "/World/CapstoneAdditions/TabletopItems/RedBall",
    "BlueCube": "/World/CapstoneAdditions/TabletopItems/BlueCube",
    "Book": "/World/CapstoneAdditions/TabletopItems/Book",
}

DEPTH_OVERLAY = None
TOP_VIEW_ROS_BRIDGE = None
EE_VIEW_ROS_BRIDGE = None
FRANKA_JOINT_ROS_BRIDGE = None


def set_xform(prim, translate=None, rotate_xyz_deg=None, scale=None):
    xformable = UsdGeom.Xformable(prim)
    ordered_ops = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}
    if translate is not None:
        (ordered_ops.get("xformOp:translate") or xformable.AddTranslateOp()).Set(Gf.Vec3d(*map(float, translate)))
    if rotate_xyz_deg is not None:
        (ordered_ops.get("xformOp:rotateXYZ") or xformable.AddRotateXYZOp()).Set(Gf.Vec3f(*map(float, rotate_xyz_deg)))
    if scale is not None:
        (ordered_ops.get("xformOp:scale") or xformable.AddScaleOp()).Set(Gf.Vec3f(*map(float, scale)))


def define_xform(stage, path, translate=None, rotate_xyz_deg=None, scale=None):
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    set_xform(prim, translate=translate, rotate_xyz_deg=rotate_xyz_deg, scale=scale)
    return prim


def add_visual_reference(stage, path, asset_path, translate=None, rotate_xyz_deg=None, scale=None):
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().ClearReferences()
    prim.GetReferences().AddReference(str(asset_path))
    set_xform(prim, translate=translate, rotate_xyz_deg=rotate_xyz_deg, scale=scale)
    return prim


def set_display_color(prim, rgb):
    gprim = UsdGeom.Gprim(prim)
    if gprim:
        gprim.CreateDisplayColorAttr([Gf.Vec3f(*map(float, rgb))])


def set_descendant_display_color(prim, rgb):
    for child in Usd.PrimRange(prim):
        if child == prim:
            continue
        set_display_color(child, rgb)


def iter_collision_prims(root_prim):
    supported = {"Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone"}
    for prim in Usd.PrimRange(root_prim):
        if prim.GetTypeName() in supported:
            yield prim


def apply_static_collider(root_prim, approximation="convexHull"):
    for prim in iter_collision_prims(root_prim):
        UsdPhysics.CollisionAPI.Apply(prim)
        if prim.GetTypeName() == "Mesh":
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision.CreateApproximationAttr().Set(approximation)


def create_dynamic_body_root(stage, path, translate, mass):
    root = define_xform(stage, path, translate=translate)
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid_body.CreateRigidBodyEnabledAttr(True)
    rigid_body.CreateStartsAsleepAttr(True)
    physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    physx_rigid_body.CreateDisableGravityAttr(False)
    physx_rigid_body.CreateAngularDampingAttr(0.2)
    physx_rigid_body.CreateLinearDampingAttr(0.05)
    physx_rigid_body.CreateSleepThresholdAttr(0.0)
    physx_rigid_body.CreateStabilizationThresholdAttr(0.0)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(mass))
    return root


def build_box_collider(stage, path, size, translate=None, rotate_xyz_deg=None):
    prim = UsdGeom.Cube.Define(stage, path).GetPrim()
    UsdGeom.Cube(prim).CreateSizeAttr(1.0)
    set_xform(prim, translate=translate, rotate_xyz_deg=rotate_xyz_deg, scale=size)
    apply_static_collider(prim, approximation="boundingCube")
    UsdGeom.Imageable(prim).MakeInvisible()
    return prim


def build_sphere_collider(stage, path, radius, translate=None):
    prim = UsdGeom.Sphere.Define(stage, path).GetPrim()
    UsdGeom.Sphere(prim).CreateRadiusAttr(float(radius))
    set_xform(prim, translate=translate)
    apply_static_collider(prim, approximation="boundingSphere")
    UsdGeom.Imageable(prim).MakeInvisible()
    return prim


def build_cylinder_collider(stage, path, radius, height, translate=None, rotate_xyz_deg=None):
    prim = UsdGeom.Cylinder.Define(stage, path).GetPrim()
    cylinder = UsdGeom.Cylinder(prim)
    cylinder.CreateRadiusAttr(float(radius))
    cylinder.CreateHeightAttr(float(height))
    set_xform(prim, translate=translate, rotate_xyz_deg=rotate_xyz_deg)
    apply_static_collider(prim, approximation="convexHull")
    UsdGeom.Imageable(prim).MakeInvisible()
    return prim


def build_apple(stage, path):
    root = create_dynamic_body_root(stage, path, APPLE_TRANSLATE, mass=0.12)
    if USE_APPLE_MESH and APPLE_ASSET.exists():
        add_visual_reference(
            stage,
            f"{path}/Visual",
            APPLE_ASSET,
            translate=(APPLE_VISUAL_TRANSLATE * APPLE_SCALE).tolist(),
            rotate_xyz_deg=APPLE_ROTATION_DEG,
            scale=APPLE_SCALE,
        )
    else:
        body = UsdGeom.Sphere.Define(stage, f"{path}/Visual/Body")
        body.CreateRadiusAttr(0.055)
        set_display_color(body.GetPrim(), [0.80, 0.10, 0.08])

        stem = UsdGeom.Cylinder.Define(stage, f"{path}/Visual/Stem")
        stem.CreateRadiusAttr(0.006)
        stem.CreateHeightAttr(0.045)
        set_xform(stem.GetPrim(), translate=[0.0, 0.0, 0.065])
        set_display_color(stem.GetPrim(), [0.35, 0.22, 0.08])

        leaf = UsdGeom.Cube.Define(stage, f"{path}/Visual/Leaf")
        leaf.CreateSizeAttr(1.0)
        set_xform(leaf.GetPrim(), translate=[0.02, 0.0, 0.07], rotate_xyz_deg=[0.0, 22.0, 35.0], scale=[0.018, 0.008, 0.004])
        set_display_color(leaf.GetPrim(), [0.18, 0.45, 0.12])
    build_sphere_collider(
        stage,
        f"{path}/Collider",
        radius=float(77.5 * APPLE_SCALE[0]),
        translate=(APPLE_COLLIDER_TRANSLATE * APPLE_SCALE).tolist(),
    )
    return root


def build_red_ball(stage, path):
    root = create_dynamic_body_root(stage, path, RED_BALL_TRANSLATE, mass=0.08)
    if RED_BALL_ASSET.exists():
        add_visual_reference(
            stage,
            f"{path}/Visual",
            RED_BALL_ASSET,
            translate=(RED_BALL_VISUAL_TRANSLATE * RED_BALL_SCALE).tolist(),
            rotate_xyz_deg=RED_BALL_ROTATION_DEG,
            scale=RED_BALL_SCALE,
        )
    else:
        ball = UsdGeom.Sphere.Define(stage, f"{path}/Visual/Ball")
        ball.CreateRadiusAttr(0.045)
        set_display_color(ball.GetPrim(), [0.90, 0.08, 0.08])
    build_sphere_collider(
        stage,
        f"{path}/Collider",
        radius=float(1.93 * RED_BALL_SCALE[0]),
        translate=(RED_BALL_COLLIDER_TRANSLATE * RED_BALL_SCALE).tolist(),
    )
    return root


def build_glass(stage, path):
    root = create_dynamic_body_root(stage, path, GLASS_TRANSLATE, mass=0.18)
    if USE_GLASS_MESH and GLASS_ASSET.exists():
        add_visual_reference(
            stage,
            f"{path}/Visual",
            GLASS_ASSET,
            rotate_xyz_deg=GLASS_ROTATION_DEG,
            scale=GLASS_SCALE,
        )
    else:
        glass = UsdGeom.Cylinder.Define(stage, f"{path}/Visual/Glass")
        glass.CreateRadiusAttr(0.04)
        glass.CreateHeightAttr(0.12)
        set_display_color(glass.GetPrim(), [0.75, 0.85, 0.95])
    build_cylinder_collider(
        stage,
        f"{path}/Collider",
        radius=float(3.8 * GLASS_SCALE[0]),
        height=float(9.0 * GLASS_SCALE[2]),
        translate=(GLASS_COLLIDER_TRANSLATE * GLASS_SCALE).tolist(),
        rotate_xyz_deg=GLASS_ROTATION_DEG,
    )
    return root


def build_blue_cube(stage, path):
    root = create_dynamic_body_root(stage, path, BLUE_CUBE_TRANSLATE, mass=0.2)
    if BLUE_CUBE_ASSET.exists():
        visual = add_visual_reference(
            stage,
            f"{path}/Visual",
            BLUE_CUBE_ASSET,
            rotate_xyz_deg=BLUE_CUBE_ROTATION_DEG,
            scale=BLUE_CUBE_SCALE,
        )
        set_descendant_display_color(visual, [0.08, 0.22, 0.90])
    build_box_collider(
        stage,
        f"{path}/Collider",
        size=(np.array([2.0, 2.0, 2.0]) * BLUE_CUBE_SCALE).tolist(),
        translate=(BLUE_CUBE_COLLIDER_TRANSLATE * BLUE_CUBE_SCALE).tolist(),
        rotate_xyz_deg=BLUE_CUBE_ROTATION_DEG,
    )
    return root


def build_book(stage, path):
    root = create_dynamic_body_root(stage, path, BOOK_TRANSLATE, mass=0.35)
    set_xform(root, rotate_xyz_deg=BOOK_ROTATION_DEG)
    physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    physx_rigid_body.CreateAngularDampingAttr(2.5)
    physx_rigid_body.CreateLinearDampingAttr(0.3)
    if BOOK_ASSET.exists():
        add_visual_reference(
            stage,
            f"{path}/Visual",
            BOOK_ASSET,
            rotate_xyz_deg=[0.0, 0.0, 0.0],
            scale=BOOK_SCALE,
        )
    else:
        cover = UsdGeom.Cube.Define(stage, f"{path}/Visual/Cover")
        cover.CreateSizeAttr(1.0)
        set_xform(cover.GetPrim(), scale=[0.13, 0.09, 0.012])
        set_display_color(cover.GetPrim(), [0.14, 0.28, 0.62])

        pages = UsdGeom.Cube.Define(stage, f"{path}/Visual/Pages")
        pages.CreateSizeAttr(1.0)
        set_xform(pages.GetPrim(), translate=[0.0, 0.0, 0.005], scale=[0.118, 0.078, 0.009])
        set_display_color(pages.GetPrim(), [0.94, 0.93, 0.88])
    build_box_collider(
        stage,
        f"{path}/Collider",
        size=(np.array([1.49, 0.38, 2.45]) * BOOK_SCALE).tolist(),
        translate=(BOOK_COLLIDER_TRANSLATE * BOOK_SCALE).tolist(),
        rotate_xyz_deg=[0.0, 0.0, 0.0],
    )
    return root


def build_tabletop_items(stage, root_path):
    if stage.GetPrimAtPath(root_path):
        stage.RemovePrim(root_path)
    props_root = define_xform(stage, root_path, translate=BEDSIDE_TABLE_POSITION, rotate_xyz_deg=BEDSIDE_TABLE_ROTATION_DEG)
    build_apple(stage, f"{props_root.GetPath()}/Apple")
    build_glass(stage, f"{props_root.GetPath()}/Glass")
    build_red_ball(stage, f"{props_root.GetPath()}/RedBall")
    build_blue_cube(stage, f"{props_root.GetPath()}/BlueCube")
    build_book(stage, f"{props_root.GetPath()}/Book")
    return props_root


def find_franka_root(stage):
    for prim in stage.Traverse():
        if prim.GetName().lower() == "franka":
            return prim
    return None


def find_descendant_by_candidates(root_prim, candidates):
    if root_prim is None:
        return None
    candidate_names = {name.lower() for name in candidates}
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName().lower() in candidate_names:
            return prim
    return None


def create_camera(stage, path, translate, rotate_xyz_deg=None, focal_length_mm=1.93):
    camera = UsdGeom.Camera.Define(stage, path)
    set_xform(camera.GetPrim(), translate=translate, rotate_xyz_deg=rotate_xyz_deg)
    camera.CreateFocalLengthAttr(float(focal_length_mm))
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.02, 1000.0))
    camera.CreateHorizontalApertureAttr(3.84)
    camera.CreateVerticalApertureAttr(2.16)
    return camera.GetPrim()


def get_ee_mount_prim(franka_root):
    return find_descendant_by_candidates(franka_root, WRIST_MOUNT_CANDIDATES) or find_descendant_by_candidates(
        franka_root, EE_MOUNT_FALLBACK_CANDIDATES
    )


def deactivate_legacy_ee_cameras(stage):
    legacy_paths = [
        "/Franka/panda_link6/camera_mount/Realsense/RSD455",
        "/Franka/panda_link6/realsense_d435",
        "/Franka/panda_hand/realsense_d435",
        "/World/CapstoneAdditions/EEViewCamera",
    ]
    for path in legacy_paths:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            prim.SetActive(False)


def create_ee_camera(stage):
    deactivate_legacy_ee_cameras(stage)
    franka_root = find_franka_root(stage)
    pose_source = get_ee_mount_prim(franka_root) if franka_root else None
    if pose_source is None:
        pose_source = stage.GetPrimAtPath("/Franka/panda_link7")
    if pose_source is None or not pose_source.IsValid():
        pose_source = stage.GetPrimAtPath("/Franka/panda_link6")

    parent_path = str(pose_source.GetPath())
    mount_path = f"{parent_path}/EEViewCameraMount"
    if stage.GetPrimAtPath(mount_path):
        stage.RemovePrim(mount_path)
    mount = define_xform(
        stage,
        mount_path,
        translate=EE_CAMERA_MOUNT_TRANSLATE,
    )
    rig = define_xform(
        stage,
        f"{mount.GetPath()}/CameraRig",
        rotate_xyz_deg=EE_CAMERA_LOCAL_ROTATION_DEG,
    )
    camera_frame = define_xform(
        stage,
        f"{rig.GetPath()}/CameraFrame",
        translate=EE_CAMERA_LOCAL_TRANSLATE,
    )

    return create_camera(
        stage,
        f"{camera_frame.GetPath()}/EEViewCamera",
        [0.0, 0.0, 0.0],
        focal_length_mm=1.1,
    )


def force_perspective_view():
    viewport = get_active_viewport()
    if viewport is None:
        return
    try:
        viewport.camera_path = "/OmniverseKit_Persp"
    except Exception:
        pass


def attach_depth_sensor_template(stage, camera_path, scope_path, baseline_mm=None):
    if stage.GetPrimAtPath(scope_path):
        stage.RemovePrim(scope_path)
    stage.DefinePrim(scope_path, "Scope")
    kwargs = {}
    if baseline_mm is not None:
        kwargs["omni:rtx:post:depthSensor:baselineMM"] = baseline_mm
    try:
        SingleViewDepthSensorAsset.add_template_render_product(
            parent_prim_path=scope_path,
            camera_prim_path=camera_path,
            **kwargs,
        )
    except Exception:
        pass


def get_viewport_window_by_name(name):
    for window in get_viewport_window_instances():
        if getattr(window, "name", None) == name:
            return window
    return None


def ensure_viewport_window(name, camera_path, resolution):
    window = get_viewport_window_by_name(name)
    if window is None:
        window = ViewportWindow(name=name, width=resolution[0], height=resolution[1])
    viewport_api = getattr(window, "viewport_api", None)
    if viewport_api is not None:
        try:
            viewport_api.camera_path = camera_path
        except Exception:
            pass
        try:
            viewport_api.set_active_camera(camera_path)
        except Exception:
            pass
        try:
            viewport_api.set_texture_resolution(resolution)
        except Exception:
            pass
    return window


def bind_custom_viewports(ee_camera_path, top_camera_path):
    ensure_viewport_window(EE_VIEWPORT_NAME, ee_camera_path, EE_VIEWPORT_RESOLUTION)
    ensure_viewport_window(TOP_VIEWPORT_NAME, top_camera_path, TOP_VIEWPORT_RESOLUTION)


def save_current_stage(stage):
    root_layer = stage.GetRootLayer()
    root_path = Path(root_layer.realPath or root_layer.identifier)
    target_path = OUTPUT_STAGE.resolve()

    if not root_layer.Save():
        raise RuntimeError(f"Failed to save stage in place: {root_path}")

    if root_path.resolve() != target_path:
        if not root_layer.Export(str(OUTPUT_STAGE)):
            raise RuntimeError(f"Failed to export stage: {OUTPUT_STAGE}")


class TabletopDepthOverlay:
    def __init__(self, stage):
        self._stage = stage
        self._timeline = omni.timeline.get_timeline_interface()
        self._frame_counter = 0
        self._labels = {}
        self._window = None
        self._update_subscription = None
        self._ee_camera = None
        self._top_camera = None
        self._build_ui()
        self._initialize_cameras()
        self._update_subscription = (
            omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._on_update)
        )

    def destroy(self):
        self._update_subscription = None
        self._ee_camera = None
        self._top_camera = None
        self._window = None
        self._labels = {}

    def _build_ui(self):
        self._window = ui.Window("Tabletop Depth Monitor", width=360, height=230)
        with self._window.frame:
            with ui.VStack(spacing=6):
                ui.Label("Depth updates while the simulation is playing.")
                for name in TABLETOP_OBJECT_PATHS:
                    label = ui.Label(f"{name}: top=-- m | ee=-- m")
                    self._labels[name] = label

    def _initialize_camera(self, prim_path, resolution, name):
        camera = Camera(
            prim_path=prim_path,
            name=name,
            resolution=resolution,
        )
        camera.initialize(attach_rgb_annotator=False)
        camera.add_distance_to_camera_to_frame()
        return camera

    def _initialize_cameras(self):
        self._ee_camera = self._initialize_camera(EE_CAMERA_PATH, EE_VIEWPORT_RESOLUTION, "ee_depth_monitor")
        self._top_camera = self._initialize_camera(TOP_CAMERA_PATH, TOP_VIEWPORT_RESOLUTION, "top_depth_monitor")

    def _get_world_position(self, prim_path):
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = transform.ExtractTranslation()
        return np.array([float(translation[0]), float(translation[1]), float(translation[2])], dtype=np.float32)

    def _sample_depth_at_world_point(self, camera, world_position):
        if world_position is None:
            return None
        frame = camera.get_current_frame()
        depth = frame.get("distance_to_camera")
        if depth is None:
            return None
        depth = np.asarray(depth)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        if depth.ndim != 2:
            return None

        image_coords = camera.get_image_coords_from_world_points(np.asarray([world_position], dtype=np.float32))
        if image_coords is None or len(image_coords) == 0:
            return None
        u, v = image_coords[0]
        u = int(round(float(u)))
        v = int(round(float(v)))
        height, width = depth.shape
        if u < 0 or v < 0 or u >= width or v >= height:
            return None

        value = float(depth[v, u])
        if not np.isfinite(value) or value <= 0.0:
            return None
        return value

    def _format_depth(self, value):
        if value is None:
            return "--"
        return f"{value:.3f}"

    def _update_labels(self):
        for name, prim_path in TABLETOP_OBJECT_PATHS.items():
            world_position = self._get_world_position(prim_path)
            top_depth = self._sample_depth_at_world_point(self._top_camera, world_position)
            ee_depth = self._sample_depth_at_world_point(self._ee_camera, world_position)
            self._labels[name].text = (
                f"{name}: top={self._format_depth(top_depth)} m | ee={self._format_depth(ee_depth)} m"
            )

    def _on_update(self, _event):
        if not self._timeline.is_playing():
            return
        self._frame_counter += 1
        if self._frame_counter % 3 != 0:
            return
        self._update_labels()


def apply_scene():
    global TOP_VIEW_ROS_BRIDGE, EE_VIEW_ROS_BRIDGE, FRANKA_JOINT_ROS_BRIDGE
    stage = omni.usd.get_context().get_stage()
    additions_root = define_xform(stage, "/World/CapstoneAdditions")
    bed_prim = stage.GetPrimAtPath(f"{additions_root.GetPath()}/HospitalBed")
    if bed_prim and bed_prim.IsValid():
        bed_prim.SetActive(False)
    table_path = f"{additions_root.GetPath()}/BedsideTable"
    table_prim = stage.GetPrimAtPath(table_path)
    if table_prim and table_prim.IsValid():
        table_prim.SetActive(False)
    build_tabletop_items(stage, f"{additions_root.GetPath()}/TabletopItems")
    ee_camera = create_ee_camera(stage)
    top_camera = create_camera(
        stage,
        TOP_CAMERA_PATH,
        TOP_CAMERA_POSITION,
        TOP_CAMERA_ROTATION_DEG,
        TOP_CAMERA_FOCAL_LENGTH_MM,
    )
    attach_depth_sensor_template(stage, str(ee_camera.GetPath()), EE_DEPTH_SCOPE, baseline_mm=42)
    attach_depth_sensor_template(stage, str(top_camera.GetPath()), TOP_DEPTH_SCOPE)
    force_perspective_view()
    bind_custom_viewports(str(ee_camera.GetPath()), str(top_camera.GetPath()))
    EE_VIEW_ROS_BRIDGE = build_ee_view_bridge(str(ee_camera.GetPath()))
    TOP_VIEW_ROS_BRIDGE = build_top_view_bridge(str(top_camera.GetPath()))
    FRANKA_JOINT_ROS_BRIDGE = create_ros2_joint_graph(
        articulation_path="/Franka",
        graph_path="/World/ROS/FrankaJointGraph",
        joint_state_topic="/joint_states",
        joint_command_topic="/joint_command",
    )


async def main():
    global DEPTH_OVERLAY
    app = omni.kit.app.get_app()
    for _ in range(180):
        await app.next_update_async()
    if not open_stage(str(SOURCE_STAGE)):
        raise RuntimeError(f"Failed to open stage: {SOURCE_STAGE}")
    for _ in range(60):
        await app.next_update_async()
    force_perspective_view()
    apply_scene()
    for _ in range(60):
        await app.next_update_async()
    force_perspective_view()
    bind_custom_viewports(EE_CAMERA_PATH, TOP_CAMERA_PATH)
    stage = omni.usd.get_context().get_stage()
    save_current_stage(stage)
    if DEPTH_OVERLAY is not None:
        DEPTH_OVERLAY.destroy()
    DEPTH_OVERLAY = TabletopDepthOverlay(stage)
    print(f"Saved: {OUTPUT_STAGE}")


asyncio.ensure_future(main())
