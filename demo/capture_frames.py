#!/usr/bin/env python3
"""Snapshot both cameras — RGB, depth, and intrinsics — into demo/resources/.

One shot, no trigger topic: run it and it grabs the next frame from each stream,
writes the files, and exits.  Intended for building offline test fixtures for
the Slow Brain demos.

    source launch_env.bash
    python3 demo/capture_frames.py --prefix scene1

Requires Isaac Sim running AND THE TIMELINE PLAYING — the OmniGraph ROS 2
publishers only tick while the sim is playing, so a paused sim looks exactly
like a broken one.  The timeout message says so.

For each camera it writes:
    <prefix>_ee_rgb.png       bgr8, straight to disk
    <prefix>_ee_depth.npy     float32 METRES — the lossless one, use this
    <prefix>_ee_depth.png     8-bit visualisation only, NOT for computation
    <prefix>_ee_info.json     K, distortion, width/height

Depth gets two files on purpose.  The PNG is normalised to 0-255 for eyeballing
and throws away absolute scale; anything that needs real distances (projection,
grasp tests) must read the .npy.  Saving only a PNG is how you end up with
centroids that are self-consistent and metres wrong.

Contrast with sim/image_capture_helper.py, which is a different tool: that one
watches ONE camera and saves a burst of frames on a std_msgs/Empty trigger, for
YOLO training data.  This one is a single synchronised-enough snapshot of
everything, for VLM fixtures.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

# Isaac publishes the two views under inconsistent namespaces — RGB on
# /camera and /ee_camera, but depth and camera_info on /rgbd_camera and
# /ee_rgbd_camera.  See sim/isaac_ros_camera_bridge.py.
DEFAULT_STREAMS = {
    "ee": {
        "rgb": "/ee_camera/image_raw",
        "depth": "/ee_rgbd_camera/depth_image",
        "info": "/ee_rgbd_camera/camera_info",
    },
    "top": {
        "rgb": "/camera/image_raw",
        "depth": "/rgbd_camera/depth_image",
        "info": "/rgbd_camera/camera_info",
    },
}


class FrameGrabber(Node):
    """Keeps the first message seen on each topic, then reports done."""

    def __init__(self, streams: dict, want_depth: bool, want_info: bool) -> None:
        super().__init__("demo_frame_grabber")
        self._bridge = CvBridge()
        self._got: dict[str, object] = {}
        self._wanted: list[str] = []

        for view, topics in streams.items():
            self._sub(f"{view}_rgb", topics["rgb"], Image)
            if want_depth:
                self._sub(f"{view}_depth", topics["depth"], Image)
            if want_info:
                self._sub(f"{view}_info", topics["info"], CameraInfo)

    def _sub(self, key: str, topic: str, msg_type) -> None:
        self._wanted.append(key)
        # Depth 1: we only ever want the newest frame, and a deep queue would
        # just hand us a stale one from before the sim started playing.
        self.create_subscription(
            msg_type, topic, lambda m, k=key: self._store(k, m), 1)
        self.get_logger().info(f"waiting on {topic}")

    def _store(self, key: str, msg) -> None:
        if key not in self._got:
            self._got[key] = msg
            self.get_logger().info(f"got {key}  ({len(self._got)}/{len(self._wanted)})")

    @property
    def complete(self) -> bool:
        return len(self._got) == len(self._wanted)

    @property
    def missing(self) -> list[str]:
        return [k for k in self._wanted if k not in self._got]

    def to_bgr(self, msg: Image) -> np.ndarray:
        return self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def to_depth(self, msg: Image) -> np.ndarray:
        """Depth as float32 metres, whatever the wire encoding was."""
        raw = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        arr = np.asarray(raw)
        if arr.dtype == np.uint16:
            # 16UC1 is millimetres by ROS convention.
            return arr.astype(np.float32) / 1000.0
        return arr.astype(np.float32)


def depth_preview(depth_m: np.ndarray) -> np.ndarray:
    """8-bit false-colour view. Invalid (0/NaN/inf) pixels render black."""
    valid = np.isfinite(depth_m) & (depth_m > 0)
    out = np.zeros(depth_m.shape, dtype=np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth_m[valid], [2, 98])
        if hi > lo:
            norm = np.clip((depth_m - lo) / (hi - lo), 0, 1)
            out[valid] = (norm[valid] * 255).astype(np.uint8)
    return cv2.applyColorMap(out, cv2.COLORMAP_TURBO) * valid[..., None]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="capture",
                    help="filename prefix, e.g. --prefix scene1")
    ap.add_argument("--out-dir",
                    default=str(Path(__file__).resolve().parent / "resources"))
    ap.add_argument("--views", default="ee,top",
                    help="comma-separated subset of: ee, top")
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--no-info", action="store_true")
    args, ros_argv = ap.parse_known_args(argv)

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    unknown = [v for v in views if v not in DEFAULT_STREAMS]
    if unknown:
        print(f"unknown view(s): {unknown}. choose from {list(DEFAULT_STREAMS)}",
              file=sys.stderr)
        return 2
    streams = {v: DEFAULT_STREAMS[v] for v in views}

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init(args=ros_argv)
    node = FrameGrabber(streams, not args.no_depth, not args.no_info)

    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and not node.complete and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)

        if not node.complete:
            print(f"\nTIMEOUT after {args.timeout:.0f}s — never saw: "
                  f"{node.missing}", file=sys.stderr)
            print("\nMost likely the simulation is not PLAYING. Isaac's ROS 2",
                  file=sys.stderr)
            print("publishers only tick while the timeline runs — press Play.",
                  file=sys.stderr)
            print("Otherwise check:  ros2 topic list | grep camera", file=sys.stderr)
            return 1

        print()
        written = []
        for view in views:
            rgb = node.to_bgr(node._got[f"{view}_rgb"])
            p = out_dir / f"{args.prefix}_{view}_rgb.png"
            cv2.imwrite(str(p), rgb)
            written.append((p, f"{rgb.shape[1]}x{rgb.shape[0]} bgr8"))

            if not args.no_depth:
                d = node.to_depth(node._got[f"{view}_depth"])
                pn = out_dir / f"{args.prefix}_{view}_depth.npy"
                np.save(pn, d)
                valid = np.isfinite(d) & (d > 0)
                rng = (f"{d[valid].min():.3f}-{d[valid].max():.3f} m"
                       if valid.any() else "NO VALID PIXELS")
                written.append((pn, f"{d.shape[1]}x{d.shape[0]} float32, {rng}"))

                pp = out_dir / f"{args.prefix}_{view}_depth.png"
                cv2.imwrite(str(pp), depth_preview(d))
                written.append((pp, "preview only — not for computation"))

                # The mask must match the DEPTH resolution, not the RGB one.
                # Flag the mismatch here rather than letting mask_projection
                # discover it as an IndexError.
                if d.shape[:2] != rgb.shape[:2]:
                    print(f"  NOTE [{view}] depth {d.shape[1]}x{d.shape[0]} != "
                          f"rgb {rgb.shape[1]}x{rgb.shape[0]} — sam_a100 must "
                          "resize masks to the depth size")

            if not args.no_info:
                info = node._got[f"{view}_info"]
                pi = out_dir / f"{args.prefix}_{view}_info.json"
                pi.write_text(json.dumps({
                    "width": info.width, "height": info.height,
                    "K": list(info.k), "D": list(info.d),
                    "distortion_model": info.distortion_model,
                    "frame_id": info.header.frame_id,
                }, indent=2))
                written.append((pi, f"fx={info.k[0]:.1f} fy={info.k[4]:.1f}"))

        print("=" * 62)
        for p, note in written:
            print(f"  {p.name:<34} {note}")
        print("=" * 62)
        print(f"\n{len(written)} files → {out_dir}")
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
