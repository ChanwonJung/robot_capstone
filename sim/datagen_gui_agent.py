"""datagen_gui_agent.py — GUI Isaac 안에서 도는 데이터 생성 에이전트.

파일 명령(cmd.json)을 받아 컵을 옮기고 머티리얼을 투명/opaque 로 토글한 뒤 ack 를 쓴다.
실제 DIP 캡처/저장은 ROS 쪽 오케스트레이터가 한다(핸드셰이크).

사용법 — Isaac Sim Script Editor 에 아래 한 줄 붙여넣고 실행
(<REPO> = 이 저장소 루트의 절대경로. Script Editor 는 cwd 가 달라 상대경로가 안 먹는다):
    exec(open('<REPO>/sim/datagen_gui_agent.py').read())

종료: cmd.json 에 done 명령이 오거나, Script Editor 에서 datagen_stop() 호출.
"""
import os
import json
import asyncio

import numpy as np
import omni.usd
import omni.kit.app
from pxr import Usd, UsdShade, UsdGeom, UsdPhysics, Sdf, Gf

IPC = "/tmp/datagen_ipc"
os.makedirs(IPC, exist_ok=True)
CMD = os.path.join(IPC, "cmd.json")
ACK = os.path.join(IPC, "ack.txt")

CUP = "/World/CapstoneAdditions/TabletopItems/Glass"
CUPV = CUP + "/Visual"
MAT = CUP + "/OpaqueGTMat"
EE_CAM = "/Franka/panda_hand/EEViewCameraMount/CameraRig/CameraFrame/EEViewCamera"
WARMUP = 60         # 투명 캡처용 — RGB 는 temporal(TAA/DLSS) 누적이라 텔레포트/토글 직후
                    # 캡처하면 잔상이 겹친다. TAA 수렴까지 충분히 기다림.
WARMUP_OPAQUE = 12  # opaque 캡처용 — depth(geometry AOV)만 쓰므로 TAA 수렴 불필요.
                    # material 이 depth 에 반영되는 몇 프레임이면 충분 → 시간 절반.

_stage = omni.usd.get_context().get_stage()


def _ensure_mat():
    if _stage.GetPrimAtPath(MAT).IsValid():
        return UsdShade.Material.Get(_stage, MAT)
    mat = UsdShade.Material.Define(_stage, MAT)
    sh = UsdShade.Shader.Define(_stage, MAT + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.55, 0.6, 0.7))
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


_MATERIAL = _ensure_mat()


def _find_glass_mat():
    """참조된 원본 유리 머티리얼 prim 을 찾는다(OpaqueGTMat 제외). 투명 복원용 rebind 대상."""
    root = _stage.GetPrimAtPath(CUP)
    for p in Usd.PrimRange(root):
        if p.IsA(UsdShade.Material) and p.GetPath().pathString != MAT:
            return UsdShade.Material(p)
    return None


_GLASS = _find_glass_mat()
print(f"[agent] 유리 머티리얼: {_GLASS.GetPath() if _GLASS else '못찾음(unbind fallback)'}")


def _targets():
    root = _stage.GetPrimAtPath(CUPV)
    ts = [root]
    for p in Usd.PrimRange(root):
        if p.IsA(UsdGeom.Mesh) or p.IsA(UsdGeom.Gprim):
            ts.append(p)
    return ts


_TS = _targets()


def _set_opaque(on):
    # ★ 언바인드는 GUI 에서 투명 복원이 안 된다(잔재). 원본 유리를 명시적으로 rebind 한다.
    m = _MATERIAL if on else _GLASS
    for p in _TS:
        b = UsdShade.MaterialBindingAPI.Apply(p)
        if m is not None:
            b.Bind(m, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        else:
            b.UnbindDirectBinding()


# 컵을 KINEMATIC 으로 만들어 물리가 못 건드리게 하고, setup 의 resting 포즈(GLASS_TRANSLATE
# 로컬 → world)에 정확히 고정한다. dynamic 이면 텔레포트 후 물리/sleep 이 떠있게 만들지만
# kinematic 은 놓은 자리에 그대로 있어(프레시 씬과 동일 = 붙은 상태).
GLASS_TRANSLATE = (-2.23, 3.03, 0.733)  # setup_initial_scene.py 컵 로컬 resting translate
_GLASS_LOCAL = np.array(GLASS_TRANSLATE, dtype=np.float64)
_cup_prim = None
_PINV3 = None   # 부모 local->world 3x3 의 역행렬 (world 이동 → 로컬 이동)
try:
    from isaacsim.core.prims import SingleRigidPrim  # noqa: F401
    _cup_prim = _stage.GetPrimAtPath(CUP)
    # KINEMATIC — 물리가 못 건드려 완벽히 안착. 이동은 USD 로컬 translate 로(아래).
    UsdPhysics.RigidBodyAPI.Apply(_cup_prim).CreateKinematicEnabledAttr().Set(True)
    _cup_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*GLASS_TRANSLATE))
    # 부모(props_root) local->world 3x3 — world 평면 이동(dx,dy)을 로컬 delta 로 바꾸는 데 필요.
    _P = UsdGeom.Xformable(_cup_prim.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    _M3 = np.array([[_P[i][j] for j in range(3)] for i in range(3)], dtype=np.float64)
    _PINV3 = np.linalg.inv(_M3)
    _wt = UsdGeom.Xformable(_cup_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    print(f"[agent] cup KINEMATIC + USD-move 준비. resting world={np.array([_wt[0],_wt[1],_wt[2]])}")
except Exception as e:  # noqa: BLE001
    _cup_prim = None
    print(f"[agent] cup init 실패: {e}  (컵 이동 비활성, dx/dy 무시)")


def _move_cup(dx, dy):
    if _cup_prim is None or _PINV3 is None:
        return
    # world 평면 (dx,dy,0) 이동 → 로컬 delta (row-vector 규약: local = world @ inv(M3)).
    # kinematic 컵은 USD 로컬 translate 를 따라 이동하며 Z(=resting)는 그대로 유지.
    local_delta = np.array([dx, dy, 0.0]) @ _PINV3
    new_local = _GLASS_LOCAL + local_delta
    _cup_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*new_local))


# 카메라 지터 — EE 카메라 로컬에 namespaced rotate/translate op 추가(팔 안 건드림).
# 재로드 시 이미 있으면 재사용. 프레임마다 소량 회전/이동 → 뷰포인트 다양화.
def _cam_ops():
    xf = UsdGeom.Xformable(_stage.GetPrimAtPath(EE_CAM))
    rot = tr = None
    for op in xf.GetOrderedXformOps():
        n = op.GetOpName()
        if 'camjit' in n and 'rotateXYZ' in n:
            rot = op
        elif 'camjit' in n and 'translate' in n:
            tr = op
    if rot is None:
        rot = xf.AddRotateXYZOp(opSuffix='camjit')
    if tr is None:
        tr = xf.AddTranslateOp(opSuffix='camjit')
    return rot, tr


try:
    _CAM_ROT, _CAM_TR = _cam_ops()
    _CAM_ROT.Set(Gf.Vec3f(0, 0, 0))
    _CAM_TR.Set(Gf.Vec3d(0, 0, 0))
    print(f'[agent] 카메라 지터 op 준비: {EE_CAM}')
except Exception as e:  # noqa: BLE001
    _CAM_ROT = _CAM_TR = None
    print(f'[agent] 카메라 지터 준비 실패(카메라 고정): {e}')


def _set_camera(crx, cry, crz, ctx, cty, ctz):
    if _CAM_ROT is None:
        return
    _CAM_ROT.Set(Gf.Vec3f(float(crx), float(cry), float(crz)))
    _CAM_TR.Set(Gf.Vec3d(float(ctx), float(cty), float(ctz)))


_STOP = {"v": False}


def datagen_stop():
    _STOP["v"] = True


async def _loop():
    app = omni.kit.app.get_app()
    last = -1
    _set_opaque(False)
    print(f"[agent] datagen 에이전트 시작 — IPC={IPC}")
    while not _STOP["v"]:
        await app.next_update_async()
        try:
            if not os.path.exists(CMD):
                continue
            c = json.load(open(CMD))
        except Exception:  # noqa: BLE001
            continue
        if int(c.get("seq", -1)) <= last:
            continue
        if c.get("done"):
            break
        is_opaque = bool(c.get("opaque", False))
        # 컵 이동은 투명(프레임 시작) 에만 — 안착시킨 뒤 opaque 때 재이동하면 다시 떠버린다.
        if not is_opaque:
            _move_cup(float(c.get("dx", 0.0)), float(c.get("dy", 0.0)))
        _set_camera(c.get("crx", 0), c.get("cry", 0), c.get("crz", 0),
                    c.get("ctx", 0), c.get("cty", 0), c.get("ctz", 0))
        _set_opaque(is_opaque)
        wu = WARMUP_OPAQUE if is_opaque else WARMUP  # opaque(depth만) 은 짧게
        for _ in range(wu):
            await app.next_update_async()
        last = int(c["seq"])
        with open(ACK, "w") as f:
            f.write(str(last))
    _set_opaque(False)
    print("[agent] datagen 에이전트 종료 (컵 투명 복원)")


asyncio.ensure_future(_loop())
print("[agent] loaded. 중단: datagen_stop()")
