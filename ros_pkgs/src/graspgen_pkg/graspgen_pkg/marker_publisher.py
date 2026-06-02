"""
marker_publisher.py — RViz MarkerArray builders for grasp candidates.

Namespace 'vgn_grasps' matches the existing rviz config (GraspMarkers display).
Per-grasp visualization: ARROW (approach dir) + SPHERE (grasp point) + CYLINDER (jaw width).
"""
from __future__ import annotations

import numpy as np
from geometry_msgs.msg import Point, Vector3
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

_NS = 'vgn_grasps'


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

    clear_m             = Marker()
    clear_m.header.frame_id = frame
    clear_m.header.stamp    = stamp
    clear_m.ns              = _NS
    clear_m.action          = Marker.DELETEALL
    clear_ma                = MarkerArray(markers=[clear_m])

    markers: list[Marker] = []
    for i, c in enumerate(candidates):
        q       = float(c['quality'])
        color   = ColorRGBA(r=0.0, g=max(0.0, min(1.0, 0.6 * q)), b=1.0, a=1.0)
        pos     = c['position']
        quat    = c['quaternion']
        rot     = Rot.from_quat(quat)
        approach = rot.apply([0.0, 0.0, -1.0])
        d        = 0.20  # arrow shaft length (m)

        arrow             = Marker()
        arrow.header.frame_id = frame
        arrow.header.stamp    = stamp
        arrow.ns     = _NS
        arrow.id     = i * 3 + 1
        arrow.type   = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.points = [
            Point(x=pos[0] - approach[0] * d,
                  y=pos[1] - approach[1] * d,
                  z=pos[2] - approach[2] * d),
            Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
        ]
        arrow.scale = Vector3(x=0.02, y=0.04, z=0.0)
        arrow.color = color
        markers.append(arrow)

        sphere             = Marker()
        sphere.header.frame_id = frame
        sphere.header.stamp    = stamp
        sphere.ns     = _NS
        sphere.id     = i * 3 + 2
        sphere.type   = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
        sphere.scale  = Vector3(x=0.03, y=0.03, z=0.03)
        sphere.color  = color
        markers.append(sphere)

        width     = float(c.get('width', gripper_width))
        gripper_x = rot.apply([1.0, 0.0, 0.0])
        z_axis    = np.array([0.0, 0.0, 1.0])
        cross     = np.cross(z_axis, gripper_x)
        cross_n   = np.linalg.norm(cross)
        if cross_n > 1e-6:
            axis   = cross / cross_n
            angle  = float(np.arccos(np.clip(np.dot(z_axis, gripper_x), -1.0, 1.0)))
            q_ring = Rot.from_rotvec(axis * angle).as_quat()
        else:
            q_ring = np.array([0.0, 0.0, 0.0, 1.0])

        ring             = Marker()
        ring.header.frame_id = frame
        ring.header.stamp    = stamp
        ring.ns     = _NS
        ring.id     = i * 3 + 3
        ring.type   = Marker.CYLINDER
        ring.action = Marker.ADD
        ring.pose.position    = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
        ring.pose.orientation.x = float(q_ring[0])
        ring.pose.orientation.y = float(q_ring[1])
        ring.pose.orientation.z = float(q_ring[2])
        ring.pose.orientation.w = float(q_ring[3])
        ring.scale = Vector3(x=width, y=width, z=0.005)
        ring.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        markers.append(ring)

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
