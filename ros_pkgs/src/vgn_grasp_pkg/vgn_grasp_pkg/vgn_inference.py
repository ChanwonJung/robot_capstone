"""vgn_inference.py — VGN network loading and inference helpers (ROS-free)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# ── VGN library path (external/vgn submodule) ────────────────────────────────
_WS = Path(os.environ.get('GSAM_WS', str(Path.home() / 'gsam_ws')))
_VGN_SRC = _WS / 'external' / 'vgn' / 'src'
if _VGN_SRC.exists() and str(_VGN_SRC) not in sys.path:
    sys.path.insert(0, str(_VGN_SRC))

try:
    import torch
    from scipy import ndimage as _ndimage
    from vgn.networks import ConvNet
    from vgn.utils.transform import Transform as _VgnTransform, Rotation as _VgnRotation

    def load_network(path, device):
        net = ConvNet().to(device)
        net.load_state_dict(torch.load(path, map_location=device))
        return net

    class Grasp:
        def __init__(self, pose, width):
            self.pose  = pose
            self.width = width

    def vgn_predict(tsdf_vol, net, device):
        assert tsdf_vol.shape == (1, 40, 40, 40)
        x = torch.from_numpy(tsdf_vol).unsqueeze(0).to(device)
        with torch.no_grad():
            qual_vol, rot_vol, width_vol = net(x)
        return (qual_vol.cpu().squeeze().numpy(),
                rot_vol.cpu().squeeze().numpy(),
                width_vol.cpu().squeeze().numpy())

    def vgn_process(tsdf_vol, qual_vol, rot_vol, width_vol,
                    gaussian_filter_sigma=1.0, min_width=1.33, max_width=9.33):
        tsdf_vol = tsdf_vol.squeeze()
        qual_vol = _ndimage.gaussian_filter(qual_vol, sigma=gaussian_filter_sigma,
                                            mode='nearest')
        outside  = tsdf_vol > 0.5
        inside   = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < 0.5)
        valid    = _ndimage.morphology.binary_dilation(
                       outside, iterations=2, mask=np.logical_not(inside))
        qual_vol[valid == False] = 0.0  # noqa: E712
        qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0
        return qual_vol, rot_vol, width_vol

    def vgn_select(qual_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4):
        qual_vol = qual_vol.copy()
        qual_vol[qual_vol < threshold] = 0.0
        max_vol  = _ndimage.maximum_filter(qual_vol, size=max_filter_size)
        qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
        grasps, scores = [], []
        for idx in np.argwhere(np.where(qual_vol, 1.0, 0.0)):
            i, j, k = idx
            score = qual_vol[i, j, k]
            ori   = _VgnRotation.from_quat(rot_vol[:, i, j, k])
            pos   = np.array([i, j, k], dtype=np.float64)
            grasps.append(Grasp(_VgnTransform(ori, pos), width_vol[i, j, k]))
            scores.append(score)
        return grasps, scores

    def from_voxel_coordinates(grasp, voxel_size):
        grasp.pose.translation = grasp.pose.translation * voxel_size
        grasp.width = grasp.width * voxel_size
        return grasp

    VGN_OK = True
    VGN_IMPORT_ERROR = ''

except ImportError as _e:
    VGN_OK = False
    VGN_IMPORT_ERROR = str(_e)
    load_network = None
    torch = None
