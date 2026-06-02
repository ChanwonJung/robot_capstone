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
BOOK_ASSET = IMPORTED_ASSETS_DIR / "Book_brown.usd"
BASKET_ASSET = IMPORTED_ASSETS_DIR / "Basket.usd"
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
RED_BALL_TRANSLATE = np.array([-1.85, 2.97, 0.88])
RED_BALL_ROTATION_DEG = np.array([-90.0, 0.0, 0.0])
RED_BALL_VISUAL_TRANSLATE = np.array([2.981419, -1.258126, -0.150547])
RED_BALL_COLLIDER_TRANSLATE = np.array([0.0, 0.0, 0.02])
BOOK_TRANSLATE = np.array([-2.11, 2.92, 0.8])
BOOK_ROTATION_DEG = np.array([0.0, -90.0, -68.0])
BOOK_COLLIDER_TRANSLATE = np.array([0.0, 0.0, 0.0])
BASKET_TRANSLATE = np.array([-2.05, 2.30, 0.72])
BASKET_ROTATION_DEG = np.array([90.0, 0.0, -25.0])
BASKET_SCALE = np.array([0.17, 0.17, 0.17])
APPLE_SCALE = np.array([0.001, 0.001, 0.001])
GLASS_SCALE = np.array([0.02, 0.02, 0.02])
RED_BALL_SCALE = np.array([0.05, 0.05, 0.05])
BOOK_SCALE = np.array([0.08, 0.08, 0.08])

# Hazard placeholders for Fast Brain testing. Procedural for now; positions
# spawn outside the top-view FOV and have initial velocity so the hazards "fly
# in" toward the workspace when the simulation plays. Tune in viewport, then
# Ctrl+S to persist.
CARDBOX_ASSET = (
    XR_CONTENT_ROOT
    / "Assets" / "XR" / "Stages" / "Indoor" / "Warehouse" / "Containers" / "Cardboard"
    / "Cardbox_C2.usd"
)
# Top-view camera FOV at z=0.5 is roughly x∈[-0.7,0.7], y∈[-0.2,0.6]
# (focal=4mm, 3.84mm aperture, camera at z=2.0). Spawning hazards at
# |x|≈2.5 or |y|≈2.5 keeps them clearly outside the frame at t=0 and gives
# ~4-5s of fly-in time at v=0.4 m/s. Gravity disabled so they cruise straight
# at constant altitude through the robot's reaching path (z≈0.5).
HAZARD_BOX_TRANSLATE = np.array([2.50, -0.09, 0.35])
HAZARD_BOX_ROTATION_DEG = np.array([0.0, 0.0, 30.0])
# Cardbox_C2 USD is authored in centimeters (metersPerUnit=0.01) while the
# parent stage is in meters. USD does NOT auto-rescale references, so the
# visual scale bakes in the 0.01 unit factor.
# native ≈ 0.51 m × 0.30 = 0.15 m box at scale=0.003.
HAZARD_BOX_SCALE = np.array([0.003, 0.003, 0.003])
HAZARD_BOX_FALLBACK_SIZE = np.array([0.14, 0.14, 0.14])
# Box flight speed at /hazard/launch_bottle trigger. Override via
# ROBOT_CAPSTONE_BOX_VX when paired with HAZARD_OBJECT=box for the
# transient stop+resume demo (faster crossing = cleaner flythrough).
HAZARD_BOX_LINEAR_VELOCITY = np.array([
    float(os.environ.get("ROBOT_CAPSTONE_BOX_VX", "-0.4")),
    0.0, 0.0,
])

# pet_bottle hazard. Uses real PET USD captured into the v3 dataset so train
# and inference share the same render. Spawn / velocity mirror HazardBox so
# the top-view sees an off-frame approach then a base/glass crossing.
HAZARD_BOTTLE_ASSET = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/DigitalTwin/Assets/Warehouse/Storage/Bottles/Plastic/"
    "NaturalBostonRound_A/NaturalBostonRoundBottle_A02_PR_NVD_01.usd"
)
# x=0.85: just outside the top-view FOV edge (~0.7) so it's off-frame at rest but
# enters the workspace ~0.5 s after the launch trigger (fired when the arm starts).
# Appears-in-FOV delay ≈ (x - 0.7) / |velocity| s. Pull toward 0.78 to appear even
# sooner, push out for later. (Stay under the stage wall that blocked 4.5+.)
# Y / Z are env-controllable so the bottle can be aligned with the target
# lane (e.g. book at y=-0.16) without code edits — useful for tuning the
# replan demo so the parked bottle sits exactly across the arm's reach path.
HAZARD_BOTTLE_TRANSLATE = np.array([
    0.85,
    float(os.environ.get("ROBOT_CAPSTONE_BOTTLE_SPAWN_Y", "-0.09")),
    float(os.environ.get("ROBOT_CAPSTONE_BOTTLE_SPAWN_Z", "0.35")),
])
HAZARD_BOTTLE_ROTATION_DEG = np.array([0.0, 0.0, 0.0])
# Warehouse Bottles authored in centimeters (metersPerUnit=0.01) like Cardbox.
# Bake the unit factor + final-size scale into one factor. Tune in viewport
# if the bottle reads too small/large after a fresh import.
HAZARD_BOTTLE_SCALE = np.array([0.01, 0.01, 0.01])
HAZARD_BOTTLE_RADIUS = 0.035    # collider approx (real PET ~6cm diameter)
HAZARD_BOTTLE_HEIGHT = 0.22     # collider approx
# Velocity applied to the bottle when /hazard/launch_bottle fires (it sits still
# until then). From x=1.0 at -0.3 m/s it reaches the arm's goal-x (~0.71) in ~1 s
# and crosses the workspace over the next ~2 s — i.e. during the grasp motion.
# Tune magnitude to match how fast you want the hazard to cross.
# Slower (-0.3) than the original -0.9 so "park" mode stops it precisely: at high
# speed the per-frame park check overshoots PARK_X by 10-20 cm. -0.3 keeps the
# per-frame travel small so the bottle halts near PARK_X. Still flies in well
# within the (very slow) arm motion.
#
# Override via ROBOT_CAPSTONE_BOTTLE_VX. For the EE-detected replan demo the
# bottle should be SLOW (e.g. -0.15) so the EE camera has time to see it,
# the injector to publish /collision_object, and the global planner to issue
# a re-plan before the bottle reaches PARK_X.
HAZARD_BOTTLE_LINEAR_VELOCITY = np.array([
    float(os.environ.get("ROBOT_CAPSTONE_BOTTLE_VX", "-0.3")),
    0.0, 0.0,
])

# Hazard scenario mode (env ROBOT_CAPSTONE_HAZARD_MODE):
#   "flythrough" (default) — bottle flies straight through the workspace and
#                            exits: a TRANSIENT hazard for the stop+resume demo.
#   "park"                 — bottle flies in, then HALTS in front of the arm and
#                            stays put: a PERSISTENT hazard for the avoidance
#                            (replan) demo. Top YOLO keeps detecting it, so the
#                            injected collision object persists and the global
#                            re-plan routes the arm around it.
HAZARD_BOTTLE_MODE = os.environ.get("ROBOT_CAPSTONE_HAZARD_MODE", "flythrough").strip().lower()
# Which hazard asset gets spawned + launched (env ROBOT_CAPSTONE_HAZARD_OBJECT):
#   "bottle" (default) — pet_bottle USD, used for v3 dataset / runtime hazard.
#   "box"              — cardbox USD, spawned at the bottle's position with the
#                        bottle's flight params so capture parity holds.
HAZARD_OBJECT = os.environ.get("ROBOT_CAPSTONE_HAZARD_OBJECT", "bottle").strip().lower()
# In "park" mode, zero the bottle's velocity once its world x drops to this value.
# It flies from x=0.85 toward -x, so stopping near the arm's goal-x leaves it
# sitting in the reaching path. Tune via env ROBOT_CAPSTONE_HAZARD_PARK_X so it
# actually blocks the path the arm takes to the goal.
HAZARD_BOTTLE_PARK_X = float(os.environ.get("ROBOT_CAPSTONE_HAZARD_PARK_X", "0.45"))
# Auto-trigger the hazard launch the first time the arm starts moving instead
# of waiting for a manual `ros2 topic pub /hazard/launch_bottle`. The launcher
# subscribes to /joint_states and, AFTER an arming delay (so MoveIt hybrid
# startup and any home-pose settling don't trip the trigger), fires once when
# any joint exceeds the threshold from the latched reference pose.
#
# Tuning notes:
#  - AUTO_TRIGGER_RAD: 0.10 rad ≈ 5.7° per joint. Real goal-directed motion
#    blows past this; Isaac physics jitter and hybrid startup wiggle do not.
#  - AUTO_ARM_SEC: 5 s comfortably swallows the hybrid container init + the
#    first MoveAction "settle to current state" wiggle.
HAZARD_AUTO_LAUNCH = os.environ.get("ROBOT_CAPSTONE_HAZARD_AUTO_LAUNCH", "0").strip() == "1"
HAZARD_AUTO_TRIGGER_RAD = float(os.environ.get("ROBOT_CAPSTONE_AUTO_TRIGGER_RAD", "0.10"))
HAZARD_AUTO_ARM_SEC = float(os.environ.get("ROBOT_CAPSTONE_AUTO_ARM_SEC", "5.0"))

# Capture-only static humans for hand/forearm dataset (NOT hazards — no rigid
# body, no motion). NVIDIA 4.2 People catalog has ~8 characters total. Pick
# whichever reads least uncanny + has exposed forearms; swap the URL below.
PEOPLE_ROOT = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/4.2/Isaac/People/Characters"
)
CAPTURE_HUMAN_ASSET_BUSINESS_FEMALE = f"{PEOPLE_ROOT}/F_Business_02/F_Business_02.usd"
CAPTURE_HUMAN_ASSET_CONSTRUCTION_NEW = (
    f"{PEOPLE_ROOT}/original_male_adult_construction_05_new/male_adult_construction_05_new.usd"
)
CAPTURE_HUMAN_ASSET_CONSTRUCTION_02 = (
    f"{PEOPLE_ROOT}/original_male_adult_construction_02/male_adult_construction_02.usd"
)
CAPTURE_HUMAN_ASSET_FEMALE_MEDICAL = f"{PEOPLE_ROOT}/F_Medical_01/F_Medical_01.usd"
CAPTURE_HUMAN_ASSET_MALE_MEDICAL = f"{PEOPLE_ROOT}/M_Medical_01/M_Medical_01.usd"
CAPTURE_HUMAN_ASSET_POLICE_FEMALE = (
    f"{PEOPLE_ROOT}/original_female_adult_police_01/female_adult_police_01.usd"
)
# Active character for capture. Swap to any of the above constants.
CAPTURE_HUMAN_ASSET = CAPTURE_HUMAN_ASSET_BUSINESS_FEMALE
# Stage is Z-up but the 4.2 People assets are authored Y-up. Additionally
# the character's /Root prim has a baked xformOp:rotateXYZ = (-90, 0, 0)
# inside the referenced USD. Our outer rotate composes onto that, so to land
# USD +Y (head) at stage +Z (up) we need net X = +90  ⇒  outer X = +180.
# Verified by simulating wrapper composition with bbox probe:
#   outer (90, 0, *) gives upright (mesh +Z up in stage). outer (180,*,*) lays flat.
# Feet at character-local z=-0.12 after upright rotation, so translate_z = floor(-0.73) - (-0.12) = -0.61.
CAPTURE_HUMAN_TRANSLATE = np.array([0.40, 0.20, -0.61])     # CAPTURE: inside top-camera FOV (x[-0.7,0.7] y[-0.2,0.6]); ignore table clip for capture phase
# CAPTURE_HUMAN_TRANSLATE = np.array([0.47, -1.10, -0.61])  # SCENARIO: next to basket, table-side standing (out of top-FOV — for hazard phase only)
CAPTURE_HUMAN_ROTATION_DEG = np.array([90.0, 0.0, 0.0])     # X+90° → upright; tweak the Z (yaw) to face the robot base
CAPTURE_HUMAN_SCALE = np.array([1.0, 1.0, 1.0])             # NVIDIA People assets are authored in meters

HAZARD_ARM_TRANSLATE = np.array([0.0, 2.50, 0.55])
HAZARD_ARM_ROTATION_DEG = np.array([90.0, 0.0, 0.0])
HAZARD_FOREARM_RADIUS = 0.045
HAZARD_FOREARM_LENGTH = 0.28
HAZARD_HAND_RADIUS = 0.06
HAZARD_ARM_LINEAR_VELOCITY = np.array([0.0, -0.4, 0.0])

HAZARD_OBJECT_PATHS = {
    "HazardBox": "/World/CapstoneAdditions/Hazards/HazardBox",
    "HazardBottle": "/World/CapstoneAdditions/Hazards/HazardBottle",
    "HazardArm": "/World/CapstoneAdditions/Hazards/HazardArm",
}

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
    "Book": "/World/CapstoneAdditions/TabletopItems/Book",
    "Basket": "/World/CapstoneAdditions/TabletopItems/Basket",
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


def create_kinematic_body_root(stage, path, translate, rotate_xyz_deg=None):
    """Kinematic rigid body — no gravity, scriptable transform."""
    root = define_xform(stage, path, translate=translate, rotate_xyz_deg=rotate_xyz_deg)
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid_body.CreateRigidBodyEnabledAttr(True)
    rigid_body.CreateKinematicEnabledAttr(True)
    physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    physx_rigid_body.CreateDisableGravityAttr(True)
    return root


def apply_initial_motion(root_prim, linear_velocity):
    """Mark a dynamic body awake, disable gravity, and set initial velocity.

    Gravity is disabled so hazards drift across the workspace at constant
    altitude instead of arcing down before they reach the target zone.
    """
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    rigid_body.CreateStartsAsleepAttr(False)
    rigid_body.CreateVelocityAttr(Gf.Vec3f(*[float(v) for v in linear_velocity]))
    physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
    physx_rigid_body.CreateDisableGravityAttr(True)


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


def build_basket(stage, path):
    root = define_xform(
        stage,
        path,
        translate=BASKET_TRANSLATE,
        rotate_xyz_deg=BASKET_ROTATION_DEG,
    )
    if BASKET_ASSET.exists():
        add_visual_reference(
            stage,
            f"{path}/Visual",
            BASKET_ASSET,
            scale=BASKET_SCALE,
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
    build_book(stage, f"{props_root.GetPath()}/Book")
    build_basket(stage, f"{props_root.GetPath()}/Basket")
    return props_root


def build_hazard_box(stage, path, translate=None, rotation_deg=None, initial_velocity=None):
    """small_box hazard — uses Cardbox_C2 USD from XR Content if available.

    Spawn/motion can be overridden so the box can stand in for the bottle in
    capture mode (ROBOT_CAPSTONE_HAZARD_OBJECT=box): same spawn pose, zero
    initial velocity, and the /hazard/launch_bottle trigger gives it flight.
    """
    if translate is None:
        translate = HAZARD_BOX_TRANSLATE
    if rotation_deg is None:
        rotation_deg = HAZARD_BOX_ROTATION_DEG
    if initial_velocity is None:
        initial_velocity = HAZARD_BOX_LINEAR_VELOCITY
    root = create_dynamic_body_root(stage, path, translate, mass=0.4)
    set_xform(root, rotate_xyz_deg=rotation_deg)
    if CARDBOX_ASSET.exists():
        add_visual_reference(
            stage,
            f"{path}/Visual",
            CARDBOX_ASSET,
            scale=HAZARD_BOX_SCALE,
        )
    else:
        body = UsdGeom.Cube.Define(stage, f"{path}/Visual/Body").GetPrim()
        UsdGeom.Cube(body).CreateSizeAttr(1.0)
        set_xform(body, scale=HAZARD_BOX_FALLBACK_SIZE)
        set_display_color(body, [0.72, 0.55, 0.32])
    build_box_collider(
        stage,
        f"{path}/Collider",
        size=HAZARD_BOX_FALLBACK_SIZE.tolist(),
    )
    apply_initial_motion(root, initial_velocity)
    return root


def build_hazard_bottle(stage, path):
    """pet_bottle hazard — references the NaturalBostonRound PET USD used to
    build the v3 capture dataset, so train and inference share the same render.
    """
    root = create_dynamic_body_root(stage, path, HAZARD_BOTTLE_TRANSLATE, mass=0.25)
    set_xform(root, rotate_xyz_deg=HAZARD_BOTTLE_ROTATION_DEG)
    add_visual_reference(
        stage,
        f"{path}/Visual",
        HAZARD_BOTTLE_ASSET,
        scale=HAZARD_BOTTLE_SCALE,
    )
    build_cylinder_collider(
        stage,
        f"{path}/Collider",
        radius=HAZARD_BOTTLE_RADIUS,
        height=HAZARD_BOTTLE_HEIGHT,
    )
    # Stationary (gravity off, zero velocity) until /hazard/launch_bottle fires —
    # the bottle then gets HAZARD_BOTTLE_LINEAR_VELOCITY so it flies in synced with
    # the arm motion. See _bottle_launch_loop / _apply_bottle_launch_velocity.
    apply_initial_motion(root, np.zeros(3))
    return root


def build_hazard_arm(stage, path):
    """hand/forearm placeholder — dynamic so it flies in with initial velocity."""
    root = create_dynamic_body_root(stage, path, HAZARD_ARM_TRANSLATE, mass=0.6)
    set_xform(root, rotate_xyz_deg=HAZARD_ARM_ROTATION_DEG)
    forearm = UsdGeom.Cylinder.Define(stage, f"{path}/Visual/Forearm").GetPrim()
    UsdGeom.Cylinder(forearm).CreateRadiusAttr(float(HAZARD_FOREARM_RADIUS))
    UsdGeom.Cylinder(forearm).CreateHeightAttr(float(HAZARD_FOREARM_LENGTH))
    set_display_color(forearm, [0.92, 0.76, 0.66])
    hand = UsdGeom.Sphere.Define(stage, f"{path}/Visual/Hand").GetPrim()
    UsdGeom.Sphere(hand).CreateRadiusAttr(float(HAZARD_HAND_RADIUS))
    hand_offset_z = HAZARD_FOREARM_LENGTH * 0.5 + HAZARD_HAND_RADIUS * 0.6
    set_xform(hand, translate=[0.0, 0.0, hand_offset_z])
    set_display_color(hand, [0.96, 0.82, 0.72])
    build_cylinder_collider(
        stage,
        f"{path}/ColliderForearm",
        radius=HAZARD_FOREARM_RADIUS,
        height=HAZARD_FOREARM_LENGTH,
    )
    build_sphere_collider(
        stage,
        f"{path}/ColliderHand",
        radius=HAZARD_HAND_RADIUS,
        translate=[0.0, 0.0, hand_offset_z],
    )
    apply_initial_motion(root, HAZARD_ARM_LINEAR_VELOCITY)
    return root


def build_capture_humans(stage, root_path):
    """Static human prim(s) for hand/forearm capture. Visual reference only —
    no rigid body, no collider, no motion. Move/rotate/scale in the viewport
    (G/R/S keys or gizmos) while capturing top + ee shots. Swap the URL
    constant to cycle characters. Remove this call once dataset is complete.

    Wrapper-Xform pattern: outer rotate lives on Subject, the asset reference
    on /Body. This way Subject's outer rotation COMPOSES with the asset's
    baked /Root rotation instead of overriding it (the hazard builders do the
    same).
    """
    if stage.GetPrimAtPath(root_path):
        stage.RemovePrim(root_path)
    humans_root = define_xform(stage, root_path)
    subject = define_xform(
        stage,
        f"{humans_root.GetPath()}/Subject",
        translate=CAPTURE_HUMAN_TRANSLATE,
        rotate_xyz_deg=CAPTURE_HUMAN_ROTATION_DEG,
        scale=CAPTURE_HUMAN_SCALE,
    )
    add_visual_reference(
        stage,
        f"{subject.GetPath()}/Body",
        CAPTURE_HUMAN_ASSET,
    )
    return humans_root


def build_hazards(stage, root_path):
    if stage.GetPrimAtPath(root_path):
        stage.RemovePrim(root_path)
    hazards_root = define_xform(stage, root_path)
    # Active hazard chosen by ROBOT_CAPSTONE_HAZARD_OBJECT (bottle | box). Both
    # spawn stationary at the bottle's pose and respond to /hazard/launch_bottle
    # so the capture helper / hybrid demos stay identical across assets.
    if HAZARD_OBJECT == "box":
        build_hazard_box(
            stage,
            f"{hazards_root.GetPath()}/HazardBox",
            translate=HAZARD_BOTTLE_TRANSLATE,
            rotation_deg=HAZARD_BOTTLE_ROTATION_DEG,
            initial_velocity=np.zeros(3),
        )
    else:
        build_hazard_bottle(stage, f"{hazards_root.GetPath()}/HazardBottle")
    # build_hazard_arm(stage, f"{hazards_root.GetPath()}/HazardArm")
    return hazards_root


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
    # === Mode toggle ===
    # Capture mode  : `build_capture_humans` ON, `build_hazards` OFF
    # Hazard mode   : `build_capture_humans` OFF, `build_hazards` ON (default flight scenario)
    # build_capture_humans(stage, f"{additions_root.GetPath()}/CaptureHumans")
    build_hazards(stage, f"{additions_root.GetPath()}/Hazards")
    ee_camera = create_ee_camera(stage)
    top_camera = create_camera(
        stage,
        TOP_CAMERA_PATH,
        TOP_CAMERA_POSITION,
        TOP_CAMERA_ROTATION_DEG,
        TOP_CAMERA_FOCAL_LENGTH_MM,
    )
    attach_depth_sensor_template(stage, str(ee_camera.GetPath()), EE_DEPTH_SCOPE, baseline_mm=42)
    attach_depth_sensor_template(stage, str(top_camera.GetPath()), TOP_DEPTH_SCOPE, baseline_mm=42)
    force_perspective_view()
    bind_custom_viewports(str(ee_camera.GetPath()), str(top_camera.GetPath()))
    EE_VIEW_ROS_BRIDGE = build_ee_view_bridge(str(ee_camera.GetPath()))
    TOP_VIEW_ROS_BRIDGE = build_top_view_bridge(str(top_camera.GetPath()))
    FRANKA_JOINT_ROS_BRIDGE = create_ros2_joint_graph(
        articulation_path="/Franka",
        graph_path="/World/ROS/FrankaJointGraph",
        # Isaac publishes raw (sim-time) joint states here; joint_state_restamp_node
        # re-stamps them to wall time and republishes on /joint_states, which the
        # MoveIt stack (incl. moveit_cpp's hard-coded 'joint_states') consumes.
        joint_state_topic="/joint_states_isaac",
        joint_command_topic="/joint_command",
    )


# Single cached rigid-prim view for the hazard bottle. Creating a fresh
# RigidPrim/SingleRigidPrim view every frame RE-INITIALISES the physics view and
# zeroes the bottle's velocity (so it never flies). Create the view ONCE and
# reuse it for launch / pose-read / stop.
_BOTTLE_RB = {"view": None, "kind": None}


def _get_bottle_rb():
    """Create (once) and return the cached (view, kind) for the active hazard prim."""
    if _BOTTLE_RB["view"] is not None:
        return _BOTTLE_RB["view"], _BOTTLE_RB["kind"]
    path = HAZARD_OBJECT_PATHS["HazardBox" if HAZARD_OBJECT == "box" else "HazardBottle"]
    try:
        from isaacsim.core.prims import RigidPrim
        _BOTTLE_RB["view"], _BOTTLE_RB["kind"] = RigidPrim(path), "multi"
        return _BOTTLE_RB["view"], _BOTTLE_RB["kind"]
    except Exception as exc:
        print(f"[hazard] RigidPrim create failed: {exc}")
    try:
        from isaacsim.core.prims import SingleRigidPrim
        _BOTTLE_RB["view"], _BOTTLE_RB["kind"] = SingleRigidPrim(path), "single"
        return _BOTTLE_RB["view"], _BOTTLE_RB["kind"]
    except Exception as exc:
        print(f"[hazard] SingleRigidPrim create failed: {exc}")
    return None, None


def _set_bottle_velocity(v):
    """Set the bottle's linear velocity at runtime via the cached view."""
    rb, kind = _get_bottle_rb()
    if rb is None:
        return False
    try:
        if kind == "multi":
            rb.set_velocities(
                np.array([[float(v[0]), float(v[1]), float(v[2]), 0.0, 0.0, 0.0]], dtype=np.float32))
        else:
            rb.set_linear_velocity(np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float32))
        return True
    except Exception as exc:
        print(f"[hazard] set_bottle_velocity failed: {exc}")
        return False


def _apply_bottle_launch_velocity():
    """Give the stationary hazard prim (bottle or box) its flight velocity.

    Picks the velocity vector for the currently-active HAZARD_OBJECT so the
    box / bottle scenarios can be tuned independently. Previously this always
    applied the bottle velocity, which broke the "box flythrough + bottle park"
    pairing once the two scenarios diverged in target speed.
    """
    v = HAZARD_BOX_LINEAR_VELOCITY if HAZARD_OBJECT == "box" else HAZARD_BOTTLE_LINEAR_VELOCITY
    if _set_bottle_velocity(v):
        print(f"[hazard] {HAZARD_OBJECT} launched v={v.tolist()}")
    else:
        print(f"[hazard] ERROR: could not apply {HAZARD_OBJECT} velocity")


def _stop_bottle():
    """Zero the bottle's velocity so it halts in place (gravity off -> stays put)."""
    return _set_bottle_velocity((0.0, 0.0, 0.0))


def _get_bottle_x():
    """Read the bottle's current world x at runtime (None if unavailable)."""
    rb, kind = _get_bottle_rb()
    if rb is None:
        return None
    try:
        if kind == "multi":
            positions, _ = rb.get_world_poses()
            return float(positions[0][0])
        pos, _ = rb.get_world_pose()
        return float(pos[0])
    except Exception:
        return None


async def _bottle_launch_loop():
    """Launch the stationary bottle when /hazard/launch_bottle is received."""
    try:
        import rclpy
        from std_msgs.msg import Empty as _Empty
    except Exception as exc:
        print(f"[hazard] rclpy unavailable in Isaac python; bottle trigger disabled: {exc}")
        return
    try:
        if not rclpy.ok():
            rclpy.init()
    except Exception as exc:
        print(f"[hazard] rclpy.init failed: {exc}")
        return
    node = rclpy.create_node("hazard_bottle_launcher")
    pending = {"go": False}
    node.create_subscription(_Empty, "/hazard/launch_bottle", lambda _m: pending.__setitem__("go", True), 10)
    app = omni.kit.app.get_app()
    state = {"launched": False, "parked": False}

    # Auto-launch on arm motion: subscribe to /joint_states and fire once any
    # joint exceeds HAZARD_AUTO_TRIGGER_RAD from the latched reference pose.
    # During the AUTO_ARM_SEC arming window the reference pose is continuously
    # refreshed so that hybrid startup wiggle / settling don't bake themselves
    # in as the trigger baseline. Coexists with the manual
    # /hazard/launch_bottle path — whichever fires first wins.
    import time as _time
    auto_state = {"initial": None, "arm_at": None}
    if HAZARD_AUTO_LAUNCH:
        try:
            from sensor_msgs.msg import JointState as _JointState
        except Exception as _exc:
            print(f"[hazard] auto-launch disabled — JointState import failed: {_exc}")
        else:
            def _js_cb(msg):
                if state["launched"]:
                    return
                pos = list(msg.position)
                now = _time.monotonic()
                if auto_state["arm_at"] is None:
                    auto_state["arm_at"] = now + HAZARD_AUTO_ARM_SEC
                # Arming window: keep refreshing the reference pose so startup
                # twitching gets absorbed instead of being treated as motion.
                if now < auto_state["arm_at"]:
                    auto_state["initial"] = pos
                    return
                try:
                    max_delta = max(abs(p - p0) for p, p0 in zip(pos, auto_state["initial"]))
                except Exception:
                    return
                if max_delta > HAZARD_AUTO_TRIGGER_RAD:
                    pending["go"] = True
                    print(f"[hazard] auto-launch fired — max joint Δ={max_delta:.3f} rad "
                          f"(threshold={HAZARD_AUTO_TRIGGER_RAD})")
            node.create_subscription(_JointState, "/joint_states", _js_cb, 10)
            print(f"[hazard] auto-launch armed — Δ>{HAZARD_AUTO_TRIGGER_RAD} rad "
                  f"after {HAZARD_AUTO_ARM_SEC}s warm-up on /joint_states")

    print(f"[hazard] {HAZARD_OBJECT} launcher ready (mode={HAZARD_BOTTLE_MODE}, "
          f"auto={'on' if HAZARD_AUTO_LAUNCH else 'off'}) — "
          "trigger: /hazard/launch_bottle")
    while True:
        try:
            rclpy.spin_once(node, timeout_sec=0.0)
        except Exception:
            pass
        if pending["go"]:
            pending["go"] = False
            _apply_bottle_launch_velocity()
            state["launched"] = True
            state["parked"] = False
        # "park" mode: once the flying bottle reaches PARK_X, halt it so it stays
        # in the arm's path as a persistent obstacle (avoidance / replan demo).
        if HAZARD_BOTTLE_MODE == "park" and state["launched"] and not state["parked"]:
            x = _get_bottle_x()
            if x is not None and x <= HAZARD_BOTTLE_PARK_X:
                if _stop_bottle():
                    state["parked"] = True
                    print(f"[hazard] bottle PARKED at x≈{x:.2f} — persistent obstacle")
        await app.next_update_async()


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
    asyncio.ensure_future(_bottle_launch_loop())


asyncio.ensure_future(main())
