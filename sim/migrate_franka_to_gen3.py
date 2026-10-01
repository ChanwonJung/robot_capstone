#!/usr/bin/env python3
"""migrate_franka_to_gen3.py — SOURCE_STAGE(sim/scenes/robot_capstone.usd)의 /Franka
서브트리를 제거하고 Kinova Gen3(+Robotiq 2F-85) USD 레퍼런스로 교체한다.

오프라인 pxr 스크립트로만 편집한다 — Isaac Sim GUI에서 라이브 Save는 절대 금지
(2026-07-26 씬 손상 전례, robot_capstone.usd.bak-badSave-20260726 참고).

run_capstone_scene.sh가 매 실행마다 이 파일을 STAGES_DIR(~/Downloads/XR_Content_.../
Assets/XR/Stages/robot_capstone.usd)로 덮어쓰므로, 이 레포 파일이 진짜 소스다.

Franka 트랜스폼(translate/orient/scale)은 실행 중인 Isaac Sim 세션에서
Script Editor로 /Franka의 xformOp를 직접 덤프해서 얻은 값 — 테이블/top 카메라 등
모든 씬 좌표가 이 기준점에 맞춰져 있으므로 그대로 재사용해서 베이스 포즈를 보존한다.

실행: isaacsim/python.sh sim/migrate_franka_to_gen3.py
"""
import datetime
import shutil
from pathlib import Path

from pxr import Gf, Usd, UsdGeom

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE_PATH = REPO_ROOT / "sim" / "scenes" / "robot_capstone.usd"
GEN3_ASSET = (
    REPO_ROOT
    / "sim"
    / "assets"
    / "imported"
    / "kinova_gen3_robotiq_2f85"
    / "gen3_robotiq_2f_85"
    / "gen3_robotiq_2f_85.usda"
)

# /Franka에서 그대로 가져온 값 — Kinova도 같은 베이스 포즈에 놓는다.
FRANKA_TRANSLATE = Gf.Vec3d(0.0, -0.64, 0.0)
FRANKA_ORIENT = Gf.Quatf(0.7071067811865476, 0.0, 0.0, 0.7071067811865475)  # (w, x, y, z) — Z축 90도
FRANKA_SCALE = Gf.Vec3f(1.0, 1.0, 1.0)

KINOVA_PRIM_PATH = "/Kinova"

# setup_initial_scene.py 의 EE_CAMERA_PATH 와 반드시 동일해야 한다 — 저기서
# create_ee_camera() 가 실제로 이 경로에 카메라를 짓는다.
EE_CAMERA_PATH = (
    f"{KINOVA_PRIM_PATH}/Geometry/world/base_link/shoulder_link/half_arm_1_link/half_arm_2_link/"
    "forearm_link/spherical_wrist_1_link/spherical_wrist_2_link/bracelet_link/"
    "end_effector_link/robotiq_85_base_link/EEViewCameraMount/CameraRig/CameraFrame/EEViewCamera"
)

# XR 씬 저작 툴이 애초에 박아둔 별도 OmniGraph — create_ros2_joint_graph() 가
# Python 에서 새로 만드는 /World/ROS/KinovaJointGraph 와는 완전히 무관하고,
# 토픽명도 다르다(isaac_joint_states/isaac_joint_commands, 언더스코어+슬래시 없음
# — 저희 파이프라인은 /joint_states_isaac, /joint_command 를 씀). Franka 시절엔
# /Franka 가 유효해서 조용히 아무 기능도 안 하며 켜져만 있었는데, 로봇을 지우면
# "/Franka is not valid" 에러로 시끄러워진다. 순수 죽은 코드라 통째로 지운다.
LEGACY_ACTION_GRAPH_PATH = "/ActionGraph"

# 뷰포트 렌더텍스처가 구 EE 카메라를 직접 relationship 으로 물고 있던 잔재.
LEGACY_CAMERA_REL_TARGETS = [
    ("/Render/OmniverseKit/HydraTextures/Replicator", "camera"),
    ("/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_1", "camera"),
]


def backup():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = STAGE_PATH.parent / f"robot_capstone.usd.bak-pre_gen3_migration-{ts}"
    shutil.copy2(STAGE_PATH, dst)
    print(f"[backup] {dst}")

    desktop_dir = Path.home() / "Desktop" / f"capstone_backup_{ts[:8]}"
    desktop_dir.mkdir(parents=True, exist_ok=True)
    desktop_dst = desktop_dir / STAGE_PATH.name
    shutil.copy2(STAGE_PATH, desktop_dst)
    print(f"[backup] {desktop_dst}")


def main():
    if not STAGE_PATH.exists():
        raise FileNotFoundError(STAGE_PATH)
    if not GEN3_ASSET.exists():
        raise FileNotFoundError(GEN3_ASSET)

    backup()

    stage = Usd.Stage.Open(str(STAGE_PATH))
    if not stage:
        raise RuntimeError(f"failed to open stage: {STAGE_PATH}")

    franka_prim = stage.GetPrimAtPath("/Franka")
    if franka_prim.IsValid():
        stage.RemovePrim("/Franka")
        print("[migrate] removed /Franka subtree")
    else:
        print("[migrate] WARNING: /Franka not found in stage — nothing removed")

    kinova_prim = stage.DefinePrim(KINOVA_PRIM_PATH, "Xform")
    kinova_prim.GetReferences().AddReference(str(GEN3_ASSET))
    print(f"[migrate] referenced {GEN3_ASSET} at {KINOVA_PRIM_PATH}")

    xform = UsdGeom.Xformable(kinova_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(FRANKA_TRANSLATE)
    xform.AddOrientOp().Set(FRANKA_ORIENT)
    xform.AddScaleOp().Set(FRANKA_SCALE)
    print(f"[migrate] set {KINOVA_PRIM_PATH} transform: "
          f"translate={FRANKA_TRANSLATE} orient={FRANKA_ORIENT} scale={FRANKA_SCALE}")

    ag = stage.GetPrimAtPath(LEGACY_ACTION_GRAPH_PATH)
    if ag.IsValid():
        stage.RemovePrim(LEGACY_ACTION_GRAPH_PATH)
        print(f"[migrate] removed vestigial {LEGACY_ACTION_GRAPH_PATH}")

    ee_camera_target = Sdf.Path(EE_CAMERA_PATH)
    for prim_path, rel_name in LEGACY_CAMERA_REL_TARGETS:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        rel = prim.GetRelationship(rel_name)
        if rel:
            rel.SetTargets([ee_camera_target])
            print(f"[migrate] retargeted {prim_path}.{rel_name} -> {ee_camera_target}")

    stage.GetRootLayer().Save()
    print(f"[migrate] saved {STAGE_PATH}")


if __name__ == "__main__":
    main()
