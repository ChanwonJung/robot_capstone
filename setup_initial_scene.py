import asyncio
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.usd
from isaacsim.core.utils.stage import open_stage
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, Usd, UsdGeom


SOURCE_STAGE = Path("/home/chanwonjung/Downloads/XR_Content_NVD@10010/Assets/XR/Stages/robot_capstone.usd")
OUTPUT_STAGE = Path("/home/chanwonjung/Downloads/XR_Content_NVD@10010/Assets/XR/Stages/robot_capstone_hospital.usd")

BEDSIDE_TABLE_ASSET = Path(
    "/home/chanwonjung/Downloads/XR_Content_NVD@10010/Assets/XR/Stages/Indoor/Modern_House/SubUSDs/Roxana_RoundEndTable.usd"
)
APPLE_ASSET = Path("/home/chanwonjung/robot_capstone/assets/imported/Apple.usd")
USE_APPLE_MESH = True
USE_GLASS_MESH = True
RED_BALL_ASSET = Path("/home/chanwonjung/robot_capstone/assets/imported/Red_Ball.usd")
BLUE_CUBE_ASSET = Path(
    "/home/chanwonjung/robot_capstone/isaacsim/extscache/omni.warp.core-1.8.2+lx64/warp/examples/assets/cube.usd"
)
BOOK_ASSET = Path("/home/chanwonjung/robot_capstone/assets/imported/Book.usd")
GLASS_ASSET = Path(
    "/home/chanwonjung/Downloads/XR_Content_NVD@10010/Assets/XR/Stages/Indoor/Modern_House/SubUSDs/P_Glassware_Short.usd"
)
BEDSIDE_TABLE_POSITION = np.array([3.3, -1.69, -0.73])
BEDSIDE_TABLE_ROTATION_DEG = np.array([0.0, 0.0, 25.0])
BEDSIDE_TABLE_SCALE = np.array([0.01, 0.01, 0.01])
APPLE_TRANSLATE = np.array([-2.5, 3.08, 0.86])
APPLE_ROTATION_DEG = np.array([0.0, 0.0, 0.0])
GLASS_TRANSLATE = np.array([-2.3, 3.0, 0.7])
GLASS_ROTATION_DEG = np.array([0.0, 0.0, 0.0])
RED_BALL_TRANSLATE = np.array([-1.84, 2.80, 0.87])
RED_BALL_ROTATION_DEG = np.array([-90.0, 0.0, 0.0])
BLUE_CUBE_TRANSLATE = np.array([-2.0, 3.2, 0.78])
BLUE_CUBE_ROTATION_DEG = np.array([0.0, 0.0, 0.0])
BOOK_TRANSLATE = np.array([-1.6, 3.0, 0.75])
BOOK_ROTATION_DEG = np.array([0.0, 0.0, -68.0])
APPLE_SCALE = np.array([0.001, 0.001, 0.001])
GLASS_SCALE = np.array([0.02, 0.02, 0.02])
RED_BALL_SCALE = np.array([0.05, 0.05, 0.05])
BLUE_CUBE_SCALE = np.array([0.05, 0.05, 0.05])
BOOK_SCALE = np.array([0.25, 0.25, 0.25])
TOP_CAMERA_POSITION = np.array([0.0, 0.0, 4.8])
TOP_CAMERA_ROTATION_DEG = np.array([0.0, 90.0, -90.0])
END_EFFECTOR_CANDIDATES = ["panda_hand", "gripper_center", "tool0", "ee_link", "right_gripper", "panda_link6"]
CAMERA_MOUNT_CANDIDATES = ["camera_mount", "realsense", "camera"]
REALSENSE_LOCAL_TRANSLATE = np.array([0.06, 0.0, 0.08])
# The camera must move with the gripper during play, so it is attached under the
# ee hierarchy. USD Camera optical axis is typically different from the Franka ee
# forward axis, so apply a fixed local correction to align the camera view with
# the gripper's forward direction.
CAMERA_TO_EE_CORRECTION = Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), -90.0).GetQuat()


def set_xform(prim, translate=None, rotate_xyz_deg=None, scale=None, orient_quat=None):
    xformable = UsdGeom.Xformable(prim)
    ordered_ops = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}
    if translate is not None:
        (ordered_ops.get("xformOp:translate") or xformable.AddTranslateOp()).Set(Gf.Vec3d(*map(float, translate)))
    if orient_quat is not None:
        quatf = Gf.Quatf(float(orient_quat.GetReal()), Gf.Vec3f(*map(float, orient_quat.GetImaginary())))
        (ordered_ops.get("xformOp:orient") or xformable.AddOrientOp()).Set(quatf)
    if rotate_xyz_deg is not None:
        (ordered_ops.get("xformOp:rotateXYZ") or xformable.AddRotateXYZOp()).Set(Gf.Vec3f(*map(float, rotate_xyz_deg)))
    if scale is not None:
        (ordered_ops.get("xformOp:scale") or xformable.AddScaleOp()).Set(Gf.Vec3f(*map(float, scale)))


def define_xform(stage, path, translate=None, rotate_xyz_deg=None, scale=None, orient_quat=None):
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    set_xform(prim, translate=translate, rotate_xyz_deg=rotate_xyz_deg, scale=scale, orient_quat=orient_quat)
    return prim


def add_reference(stage, path, asset_path, translate=None, rotate_xyz_deg=None, scale=None):
    if stage.GetPrimAtPath(path):
        stage.RemovePrim(path)
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


def build_apple(stage, path):
    if USE_APPLE_MESH and APPLE_ASSET.exists():
        return add_reference(stage, path, APPLE_ASSET, APPLE_TRANSLATE, APPLE_ROTATION_DEG, APPLE_SCALE)

    root = define_xform(stage, path, translate=APPLE_TRANSLATE, rotate_xyz_deg=APPLE_ROTATION_DEG)

    body = UsdGeom.Sphere.Define(stage, f"{path}/Body")
    body.CreateRadiusAttr(0.055)
    set_display_color(body.GetPrim(), [0.80, 0.10, 0.08])

    stem = UsdGeom.Cylinder.Define(stage, f"{path}/Stem")
    stem.CreateRadiusAttr(0.006)
    stem.CreateHeightAttr(0.045)
    set_xform(stem.GetPrim(), translate=[0.0, 0.0, 0.065])
    set_display_color(stem.GetPrim(), [0.35, 0.22, 0.08])

    leaf = UsdGeom.Cube.Define(stage, f"{path}/Leaf")
    leaf.CreateSizeAttr(1.0)
    set_xform(leaf.GetPrim(), translate=[0.02, 0.0, 0.07], rotate_xyz_deg=[0.0, 22.0, 35.0], scale=[0.018, 0.008, 0.004])
    set_display_color(leaf.GetPrim(), [0.18, 0.45, 0.12])
    return root


def build_red_ball(stage, path):
    if RED_BALL_ASSET.exists():
        return add_reference(stage, path, RED_BALL_ASSET, RED_BALL_TRANSLATE, RED_BALL_ROTATION_DEG, RED_BALL_SCALE)

    ball = UsdGeom.Sphere.Define(stage, path)
    ball.CreateRadiusAttr(0.045)
    set_xform(ball.GetPrim(), translate=RED_BALL_TRANSLATE, rotate_xyz_deg=RED_BALL_ROTATION_DEG, scale=RED_BALL_SCALE)
    set_display_color(ball.GetPrim(), [0.90, 0.08, 0.08])
    return ball.GetPrim()


def build_glass(stage, path):
    if USE_GLASS_MESH and GLASS_ASSET.exists():
        return add_reference(stage, path, GLASS_ASSET, GLASS_TRANSLATE, GLASS_ROTATION_DEG, GLASS_SCALE)

    glass = UsdGeom.Cylinder.Define(stage, path)
    glass.CreateRadiusAttr(0.04)
    glass.CreateHeightAttr(0.12)
    set_xform(glass.GetPrim(), translate=GLASS_TRANSLATE, rotate_xyz_deg=GLASS_ROTATION_DEG)
    set_display_color(glass.GetPrim(), [0.75, 0.85, 0.95])
    return glass.GetPrim()


def build_blue_cube(stage, path):
    if BLUE_CUBE_ASSET.exists():
        prim = add_reference(stage, path, BLUE_CUBE_ASSET, BLUE_CUBE_TRANSLATE, BLUE_CUBE_ROTATION_DEG, BLUE_CUBE_SCALE)
        set_descendant_display_color(prim, [0.08, 0.22, 0.90])
        return prim
    return None


def build_book(stage, path):
    if BOOK_ASSET.exists():
        return add_reference(stage, path, BOOK_ASSET, BOOK_TRANSLATE, BOOK_ROTATION_DEG, BOOK_SCALE)

    root = define_xform(stage, path, translate=BOOK_TRANSLATE, rotate_xyz_deg=BOOK_ROTATION_DEG)

    cover = UsdGeom.Cube.Define(stage, f"{path}/Cover")
    cover.CreateSizeAttr(1.0)
    set_xform(cover.GetPrim(), scale=[0.13, 0.09, 0.012])
    set_display_color(cover.GetPrim(), [0.14, 0.28, 0.62])

    pages = UsdGeom.Cube.Define(stage, f"{path}/Pages")
    pages.CreateSizeAttr(1.0)
    set_xform(pages.GetPrim(), translate=[0.0, 0.0, 0.005], scale=[0.118, 0.078, 0.009])
    set_display_color(pages.GetPrim(), [0.94, 0.93, 0.88])
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


def create_camera(stage, path, translate, rotate_xyz_deg=None, focal_length_mm=1.93, orient_quat=None):
    camera = UsdGeom.Camera.Define(stage, path)
    set_xform(camera.GetPrim(), translate=translate, rotate_xyz_deg=rotate_xyz_deg, orient_quat=orient_quat)
    camera.CreateFocalLengthAttr(float(focal_length_mm))
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
    camera.CreateHorizontalApertureAttr(3.84)
    camera.CreateVerticalApertureAttr(2.16)
    return camera.GetPrim()


def get_mount_or_ee_prim(franka_root):
    ee_prim = None
    if franka_root is None:
        return None

    for prim in Usd.PrimRange(franka_root):
        if prim.GetName().lower() in END_EFFECTOR_CANDIDATES:
            ee_prim = prim
            break

    if ee_prim is None:
        return None

    for prim in Usd.PrimRange(ee_prim):
        if prim.GetName().lower() in CAMERA_MOUNT_CANDIDATES and prim.GetTypeName() in ("Xform", ""):
            return prim

    return ee_prim


def deactivate_legacy_realsense(stage):
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


def create_realsense_camera(stage):
    deactivate_legacy_realsense(stage)
    franka_root = find_franka_root(stage)
    pose_source = get_mount_or_ee_prim(franka_root) if franka_root else None
    if pose_source is None:
        pose_source = stage.GetPrimAtPath("/Franka/panda_link6")

    parent_path = str(pose_source.GetPath())
    mount = define_xform(
        stage,
        f"{parent_path}/EEViewCameraMount",
        translate=REALSENSE_LOCAL_TRANSLATE,
        orient_quat=CAMERA_TO_EE_CORRECTION,
    )

    return create_camera(
        stage,
        f"{mount.GetPath()}/EEViewCamera",
        [0.0, 0.0, 0.0],
        focal_length_mm=1.93,
        orient_quat=Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0)),
    )


def force_perspective_view():
    viewport = get_active_viewport()
    if viewport is None:
        return
    try:
        viewport.camera_path = "/OmniverseKit_Persp"
    except Exception:
        pass


def save_current_stage(stage):
    root_layer = stage.GetRootLayer()
    root_path = Path(root_layer.realPath or root_layer.identifier)
    target_path = OUTPUT_STAGE.resolve()

    if root_path.resolve() == target_path:
        if not root_layer.Save():
            raise RuntimeError(f"Failed to save stage in place: {target_path}")
    else:
        if not root_layer.Export(str(OUTPUT_STAGE)):
            raise RuntimeError(f"Failed to export stage: {OUTPUT_STAGE}")


def apply_scene():
    stage = omni.usd.get_context().get_stage()
    additions_root = define_xform(stage, "/World/CapstoneAdditions")
    bed_prim = stage.GetPrimAtPath(f"{additions_root.GetPath()}/HospitalBed")
    if bed_prim and bed_prim.IsValid():
        bed_prim.SetActive(False)
    table_path = f"{additions_root.GetPath()}/BedsideTable"
    if BEDSIDE_TABLE_ASSET.exists():
        add_reference(
            stage,
            table_path,
            BEDSIDE_TABLE_ASSET,
            BEDSIDE_TABLE_POSITION,
            BEDSIDE_TABLE_ROTATION_DEG,
            BEDSIDE_TABLE_SCALE,
        )
    build_tabletop_items(stage, f"{additions_root.GetPath()}/TabletopItems")
    create_realsense_camera(stage)
    create_camera(stage, "/World/TopViewCamera", TOP_CAMERA_POSITION, TOP_CAMERA_ROTATION_DEG, 2.5)
    force_perspective_view()


async def main():
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
    stage = omni.usd.get_context().get_stage()
    save_current_stage(stage)
    print(f"Saved: {OUTPUT_STAGE}")


asyncio.ensure_future(main())
