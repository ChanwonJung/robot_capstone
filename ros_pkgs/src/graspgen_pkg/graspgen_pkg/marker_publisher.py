"""
marker_publisher.py — RViz MarkerArray builders for grasp candidates.

Namespace 'vgn_grasps' matches the existing rviz config (GraspMarkers display).
Per-grasp visualization: Franka Panda gripper shape (4 CUBE markers).
  - left finger / right finger / palm / wrist
Gripper frame convention (matches GraspGen output):
  Z = rot.apply([0,0,1]) → away from object (toward gripper body)
  X = rot.apply([1,0,0]) → finger spread direction
"""
from __future__ import annotations

import numpy as np
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

_NS = 'vgn_grasps'

# Franka Panda gripper geometry (metres)
_FINGER_LEN = 0.060   # finger length along Z (from fingertip toward palm)
_FINGER_W   = 0.020   # finger width in X (spread direction)
_FINGER_D   = 0.018   # finger depth in Y
_PALM_LEN   = 0.028   # palm thickness in Z
_WRIST_LEN  = 0.030   # wrist stub thickness in Z


def _cube(frame, stamp, mid, center, quat, sx, sy, sz, color):
    m = Marker()
    m.header.frame_id     = frame
    m.header.stamp        = stamp
    m.ns                  = _NS
    m.id                  = mid
    m.type                = Marker.CUBE
    m.action              = Marker.ADD
    m.pose.position.x     = float(center[0])
    m.pose.position.y     = float(center[1])
    m.pose.position.z     = float(center[2])
    m.pose.orientation.x  = float(quat[0])
    m.pose.orientation.y  = float(quat[1])
    m.pose.orientation.z  = float(quat[2])
    m.pose.orientation.w  = float(quat[3])
    m.scale               = Vector3(x=float(sx), y=float(sy), z=float(sz))
    m.color               = color
    return m


def build_grasp_markers(
    candidates: list[dict],
    frame: str,
    stamp,
    gripper_width: float,
) -> tuple[MarkerArray, MarkerArray]:
    """Build (clear_ma, markers_ma) for publishing to /grasp_markers.

    Returns a DELETEALL MarkerArray first, then the populated one so the caller
    can publish both in sequence to avoid stale markers.
    """
    from scipy.spatial.transform import Rotation as Rot

    clear_m              = Marker()
    clear_m.header.frame_id = frame
    clear_m.header.stamp    = stamp
    clear_m.ns              = _NS
    clear_m.action          = Marker.DELETEALL
    clear_ma                = MarkerArray(markers=[clear_m])

    markers: list[Marker] = []
    for i, c in enumerate(candidates):
        q     = float(c['quality'])
        color = ColorRGBA(r=0.0, g=max(0.0, min(1.0, 0.6 * q)), b=1.0, a=0.85)

        pos   = np.array(c['position'], dtype=float)
        quat  = np.array(c['quaternion'], dtype=float)  # [qx, qy, qz, qw]
        rot   = Rot.from_quat(quat)
        width = float(c.get('width', gripper_width))

        # Gripper frame axes in world frame
        z_vec = rot.apply([0.0, 0.0, 1.0])   # toward wrist (away from object)
        x_vec = rot.apply([1.0, 0.0, 0.0])   # finger spread direction

        # Finger centers: fingertips at pos, fingers extend in +z_vec direction
        finger_offset_x = x_vec * (width / 2 + _FINGER_W / 2)
        finger_center_z = z_vec * (_FINGER_LEN / 2)
        left_center  = pos + finger_center_z + finger_offset_x
        right_center = pos + finger_center_z - finger_offset_x

        # Palm: connects both fingers, further along z_vec
        palm_w      = width + 2 * _FINGER_W
        palm_center = pos + z_vec * (_FINGER_LEN + _PALM_LEN / 2)

        # Wrist stub
        wrist_center = pos + z_vec * (_FINGER_LEN + _PALM_LEN + _WRIST_LEN / 2)

        base = i * 4
        # scale: (X=finger-spread, Y=depth, Z=along-approach) — matches gripper orientation
        markers.append(_cube(frame, stamp, base + 1,
                             left_center,  quat, _FINGER_W, _FINGER_D, _FINGER_LEN, color))
        markers.append(_cube(frame, stamp, base + 2,
                             right_center, quat, _FINGER_W, _FINGER_D, _FINGER_LEN, color))
        markers.append(_cube(frame, stamp, base + 3,
                             palm_center,  quat, palm_w, _FINGER_D, _PALM_LEN, color))
        markers.append(_cube(frame, stamp, base + 4,
                             wrist_center, quat, 0.040, _FINGER_D + 0.010, _WRIST_LEN,
                             ColorRGBA(r=0.0, g=max(0.0, min(1.0, 0.6 * q)), b=1.0, a=0.5)))

    markers_ma         = MarkerArray()
    markers_ma.markers = markers
    return clear_ma, markers_ma


def build_target_cloud_msg(pts: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """Build a minimal XYZ-only PointCloud2 for /graspgen/target_cloud debug topic."""
    data                = pts.astype(np.float32).tobytes()
    msg                 = PointCloud2()
    msg.header.stamp    = stamp
    msg.header.frame_id = frame_id
    msg.height          = 1
    msg.width           = len(pts)
    msg.is_dense        = True
    msg.is_bigendian    = False
    msg.point_step      = 12
    msg.row_step        = 12 * len(pts)
    msg.fields          = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = data
    return msg
