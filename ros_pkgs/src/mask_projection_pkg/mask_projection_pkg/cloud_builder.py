"""
cloud_builder.py

Assembles a sensor_msgs/PointCloud2 from labeled point data.
No ROS math here — just message packing.

Point layout (20 bytes, 4-byte aligned):
  offset  0 : x        float32
  offset  4 : y        float32
  offset  8 : z        float32
  offset 12 : rgb      float32  packed uint32 reinterpreted as float32
                                (r<<16 | g<<8 | b) — PCL/RViz2 convention
                                RViz2 RGB8 transformer requires this exact field name + type
  offset 16 : category uint8
  padding    : 3 bytes
  point_step = 20
"""
from __future__ import annotations

from typing import List

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from .label_mapper import CategoryPoints


_POINT_STEP: int = 20

_FIELDS: List[PointField] = [
    PointField(name='x',        offset=0,  datatype=PointField.FLOAT32, count=1),
    PointField(name='y',        offset=4,  datatype=PointField.FLOAT32, count=1),
    PointField(name='z',        offset=8,  datatype=PointField.FLOAT32, count=1),
    PointField(name='rgb',      offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name='category', offset=16, datatype=PointField.UINT8,   count=1),
]


def build_pointcloud2(
    header:          Header,
    category_points: List[CategoryPoints],
) -> PointCloud2:
    """
    Merge all CategoryPoints groups into a single PointCloud2 message.
    Returns an empty cloud (width=0) if category_points is empty.
    """
    if not category_points:
        return _empty_cloud(header)

    all_pts  = np.concatenate([cp.points     for cp in category_points], axis=0)  # (N,3)
    all_col  = np.concatenate([cp.colors     for cp in category_points], axis=0)  # (N,3) uint8
    all_cats = np.concatenate([cp.categories for cp in category_points], axis=0)  # (N,) uint8
    N = len(all_pts)

    # pack RGB into float32 (standard PCL convention required by RViz2 RGB8)
    r = all_col[:, 0].astype(np.uint32)
    g = all_col[:, 1].astype(np.uint32)
    b = all_col[:, 2].astype(np.uint32)
    rgb_f32 = ((r << 16) | (g << 8) | b).view(np.float32)

    dt = np.dtype([
        ('x',        np.float32),
        ('y',        np.float32),
        ('z',        np.float32),
        ('rgb',      np.float32),
        ('category', np.uint8),
        ('_pad',     np.uint8, (3,)),
    ])
    arr          = np.zeros(N, dtype=dt)
    arr['x']     = all_pts[:, 0]
    arr['y']     = all_pts[:, 1]
    arr['z']     = all_pts[:, 2]
    arr['rgb']   = rgb_f32
    arr['category'] = all_cats

    msg              = PointCloud2()
    msg.header       = header
    msg.height       = 1
    msg.width        = N
    msg.fields       = _FIELDS
    msg.is_bigendian = False
    msg.point_step   = _POINT_STEP
    msg.row_step     = _POINT_STEP * N
    msg.is_dense     = True
    msg.data         = arr.tobytes()
    return msg


def _empty_cloud(header: Header) -> PointCloud2:
    msg              = PointCloud2()
    msg.header       = header
    msg.height       = 1
    msg.width        = 0
    msg.fields       = _FIELDS
    msg.is_bigendian = False
    msg.point_step   = _POINT_STEP
    msg.row_step     = 0
    msg.is_dense     = True
    msg.data         = b''
    return msg
