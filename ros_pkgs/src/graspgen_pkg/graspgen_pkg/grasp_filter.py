"""
grasp_filter.py — Client-side grasp candidate filters.

The GraspGen paper recommends publishing ~100 candidates as a "goal set"
to the downstream motion planner. Their planner of choice (cuRobo) is
goal-set-aware and picks a reachable goal natively. Our MoveIt + BT
pipeline is single-goal sequential, so we filter on the client side to
shrink the published pool down to candidates the BT can usefully attempt.

Filter stack (cheap → expensive):
    1. top_down_filter        — drop side grasps; keep approach within
                                `angle_threshold_deg` of straight down.
                                Cheap (one quat decode per candidate).
                                Tuned for packed-scene tabletop where side
                                grasps risk collision with neighbours.
    2. ik_feasibility_filter  — drop grasps unreachable by Panda IK using
                                MoveIt's /compute_ik service. Strong
                                filter — eliminates goals the BT could
                                never reach regardless of motion planner.
                                Expensive (~50ms per candidate); run on
                                the small set left after filter 1.
    3. confidence_top_n       — final cap so the published pool stays
                                BT-friendly.

All functions are pure (or wrap a ROS service); they operate on a list of
candidate dicts in the canonical schema:
    {'position': [x,y,z], 'quaternion': [x,y,z,w], 'quality': float,
     'width': float, 'frame': str}
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as Rot


# ── Filter 1: approach-angle (top-down preference) ─────────────────────────

def top_down_filter(
    candidates: list[dict],
    angle_threshold_deg: float,
    logger=None,
) -> list[dict]:
    """Keep candidates whose gripper approach (+Z in candidate frame) is
    within `angle_threshold_deg` of straight down.

    Candidates are expected to be already in panda_link0 (or world) frame
    with +Z up. A grasp's "approach" is the gripper local +Z direction —
    after the X-flip in graspgen_node, this points from the wrist toward
    the fingertips, i.e. toward the object.

    For a top-down grasp:
        gripper +Z (world) · (0, 0, -1) ≈ 1   (almost straight down)
    The dot product equals R[2, 2] (the z-component of gripper +Z in the
    candidate frame), negated. We keep candidates with R[2, 2] ≤ -cos(θ).

    A threshold of 45° accepts grasps from above plus moderate side
    angles. 30° is stricter (closer to pure vertical). 90° disables.

    If `logger` is provided, the distribution of R[2,2] is reported once
    per call. Useful for diagnosing convention mismatches (e.g. when our
    flip_orientation_x setting disagrees with the gripper's actual frame
    after publication, the sign of R[2,2] inverts and the filter drops
    every candidate).
    """
    if not candidates or angle_threshold_deg >= 90.0:
        return candidates

    cos_threshold = -float(np.cos(np.radians(angle_threshold_deg)))

    if logger is not None:
        zs = np.array([Rot.from_quat(c['quaternion']).as_matrix()[2, 2]
                       for c in candidates], dtype=np.float32)
        logger.info(
            f'top_down R[2,2] stats: '
            f'min={zs.min():+.2f} max={zs.max():+.2f} '
            f'median={float(np.median(zs)):+.2f} '
            f'(threshold ≤ {cos_threshold:+.2f}, '
            f'i.e. gripper +Z pointing down)')

    kept: list[dict] = []
    for c in candidates:
        R = Rot.from_quat(c['quaternion']).as_matrix()
        # R[2, 2] = z-component of gripper +Z in candidate frame.
        # < 0 means pointing down; <= cos_threshold means within angle.
        if R[2, 2] <= cos_threshold:
            kept.append(c)
    return kept


# ── Filter 3: final confidence cap ──────────────────────────────────────────

def confidence_top_n(candidates: list[dict], n: int) -> list[dict]:
    """Sort by `quality` descending, truncate to top `n`. `n <= 0` → no cap."""
    if n <= 0 or len(candidates) <= n:
        return sorted(candidates, key=lambda c: c['quality'], reverse=True)
    return sorted(candidates, key=lambda c: c['quality'], reverse=True)[:n]


# ── Filter 2: IK feasibility (MoveIt /compute_ik) ───────────────────────────

class IKFeasibilityChecker:
    """MoveIt /compute_ik wrapper for per-candidate reachability checks.

    Why: a grasp pose can pass quality/angle filters yet be kinematically
    unreachable by Panda (joint limits, link self-collision). Sending such
    a goal to MoveIt either fails planning or — worse — drives the hybrid
    planner into the SIGABRT race we have already seen. Pre-checking IK
    eliminates those before they hit the BT.

    Threading: this client uses async service calls + polled futures, so
    the host node MUST be spinning on a MultiThreadedExecutor with the IK
    client on a ReentrantCallbackGroup. With SingleThreadedExecutor the
    poll loop blocks the same thread that would dispatch the response
    callback → deadlock.

    Failure mode: if the service is unavailable, `filter` returns the
    input unchanged (fail-open). We trade a stricter filter for the
    ability to keep running when MoveIt isn't up yet.
    """

    def __init__(
        self,
        node,
        service_name: str,
        planning_group: str,
        ee_link: str,
        frame_id: str,
        per_call_timeout_sec: float,
        callback_group=None,
    ) -> None:
        from rclpy.callback_groups import ReentrantCallbackGroup
        from moveit_msgs.srv import GetPositionIK

        self._GetPositionIK = GetPositionIK
        self._node          = node
        self._group         = planning_group
        self._link          = ee_link
        self._frame         = frame_id
        self._timeout       = float(per_call_timeout_sec)
        self._available     = False

        cb_group = callback_group or ReentrantCallbackGroup()
        self._client = node.create_client(
            GetPositionIK, service_name, callback_group=cb_group)

    def wait_for_service(self, timeout_sec: float) -> bool:
        """Block at most `timeout_sec` for the IK service to appear.
        Returns True if available."""
        self._available = self._client.wait_for_service(timeout_sec=timeout_sec)
        return self._available

    def filter(self, candidates: list[dict]) -> tuple[list[dict], dict]:
        """Return (kept, stats). `stats` reports {'checked', 'kept',
        'failed', 'service_down'} so the caller can log the breakdown."""
        stats = {
            'checked':      0,
            'kept':         0,
            'failed':       0,
            'service_down': False,
        }

        if not candidates:
            return candidates, stats

        # Re-check service availability cheaply (0.1s) — MoveIt may have
        # come up after node init.
        if not self._available:
            self._available = self._client.wait_for_service(timeout_sec=0.1)
        if not self._available:
            stats['service_down'] = True
            return candidates, stats  # fail-open

        kept: list[dict] = []
        for c in candidates:
            stats['checked'] += 1
            if self._check_one(c):
                stats['kept'] += 1
                kept.append(c)
            else:
                stats['failed'] += 1
        return kept, stats

    def _check_one(self, c: dict) -> bool:
        from geometry_msgs.msg import PoseStamped
        import rclpy

        req = self._GetPositionIK.Request()
        req.ik_request.group_name      = self._group
        req.ik_request.ik_link_name    = self._link
        req.ik_request.avoid_collisions = True
        # ros2 duration: split seconds + nanoseconds
        secs            = int(self._timeout)
        req.ik_request.timeout.sec     = secs
        req.ik_request.timeout.nanosec = int((self._timeout - secs) * 1e9)

        ps = PoseStamped()
        ps.header.frame_id  = self._frame
        ps.pose.position.x  = float(c['position'][0])
        ps.pose.position.y  = float(c['position'][1])
        ps.pose.position.z  = float(c['position'][2])
        ps.pose.orientation.x = float(c['quaternion'][0])
        ps.pose.orientation.y = float(c['quaternion'][1])
        ps.pose.orientation.z = float(c['quaternion'][2])
        ps.pose.orientation.w = float(c['quaternion'][3])
        req.ik_request.pose_stamped = ps

        future = self._client.call_async(req)
        # Poll-wait — safe because the service callback runs on a
        # different thread of the MultiThreadedExecutor (Reentrant group).
        deadline = self._node.get_clock().now().nanoseconds + int((self._timeout + 0.5) * 1e9)
        while not future.done():
            now = self._node.get_clock().now().nanoseconds
            if now >= deadline:
                return False
            import time as _time
            _time.sleep(0.005)

        resp = future.result()
        if resp is None:
            return False
        # MoveItErrorCodes.SUCCESS = 1
        return resp.error_code.val == 1
