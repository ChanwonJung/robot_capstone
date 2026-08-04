"""ROS 2 node: EE camera frame + instruction -> grounded detections.

Wiring only.  All VLM logic lives in qwen_call.py, all contract shape in
qwen_schema.py.  Keep it that way — this file should stay boring.

Subscribes
  /user_instruction            std_msgs/String    TRIGGER
  <image_topic>                sensor_msgs/Image  cached (latest frame wins)
  /bt/replan_request           std_msgs/Empty     re-TRIGGER with cached instruction

Publishes
  /qwen/labeled_detections     std_msgs/String    JSON array, latched
  /qwen/grounding_result       std_msgs/String    JSON object, latched
  /qwen/source_image           sensor_msgs/Image  the exact frame the VLM saw, latched
  /slow_brain/status           std_msgs/String    diagnostic state, latched
  /slow_brain/question         std_msgs/String    clarification request, latched
  /slow_brain/question_image   sensor_msgs/Image  numbered candidate boxes, latched

Three outcomes per scan, and only the first publishes a scene:
  ok          -> detections + grounding + source_image
  needs_input -> question + question_image; the next /user_instruction is
                 treated as the ANSWER and merged with the original command
  failed      -> nothing; the previous latched scan stands

Why /qwen/source_image exists: sam_a100 must segment the SAME frame the boxes
were computed on.  If it grabbed the live camera topic instead, any motion
between the two calls misaligns every box against the pixels, silently.  The
frame is captured once under the lock in _dispatch and reused for both the VLM
call and the republish, so the two can never diverge.

It is republished unmodified — same message object, same header.stamp — so a
downstream mask can be correlated back to its source frame.  Note the VLM sees
a JPEG-compressed copy (see qwen_call.encode_image) while this topic carries
the original, so boxes and masks are computed on very slightly different pixels.

Publish order: detections, then grounding, then source_image LAST.  sam_a100
should TRIGGER on /qwen/source_image and CACHE /qwen/labeled_detections — ROS 2
guarantees no ordering across topics, so a consumer triggering on the first
publish here can fire before the rest arrive.  Triggering on the last publish
is the same trick GSAM uses when it publishes its mask last for the projector.

This node does not publish the mask at all — sam_a100 publishes /sam/mask_image,
and that causality is what satisfies mask_projection_pkg's requirement that
detections arrive before the mask.  If you ever merge the two nodes, publish the
mask LAST by hand.

The trigger is inverted relative to the legacy qwen_pkg: there, GSAM detections
triggered and the instruction was cached.  Here the instruction triggers and
the image is cached, because there is no detector upstream any more.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, String

from .qwen_call import ground
from .qwen_schema import build_labeled_detections, to_json

# bt_pkg subscribes /qwen/grounding_result with plain volatile QoS, and
# mask_projection_pkg subscribes /qwen/labeled_detections the same way.  A
# TRANSIENT_LOCAL publisher is compatible with a VOLATILE subscriber, so
# latching here costs nothing and rescues late-starting consumers.
LATCHED = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class QwenBridgeNode(Node):

    def __init__(self) -> None:
        super().__init__("qwen_bridge_node")

        self.declare_parameter("vllm_endpoint_url", "http://localhost:8000/v1")
        self.declare_parameter("model_name", "qwen35-local")
        self.declare_parameter("image_topic", "/ee_camera/image_raw")
        # Test mode: when set, this file is used as THE frame and image_topic is
        # not subscribed at all. Lets the whole ROS path run without Isaac Sim.
        self.declare_parameter("image_path", "")
        self.declare_parameter("instruction", "")
        self.declare_parameter("bbox_convention", "absolute")
        self.declare_parameter("timeout_sec", 120.0)
        self.declare_parameter("max_tokens", 2048)
        self.declare_parameter("detections_topic", "/qwen/labeled_detections")
        self.declare_parameter("grounding_topic", "/qwen/grounding_result")
        self.declare_parameter("source_image_topic", "/qwen/source_image")
        self.declare_parameter("status_topic", "/slow_brain/status")
        self.declare_parameter("question_topic", "/slow_brain/question")
        self.declare_parameter("question_image_topic", "/slow_brain/question_image")
        # Where the annotated candidate view is written, and whether to pop it
        # open in the desktop image viewer.
        self.declare_parameter("question_image_path",
                               "/tmp/slow_brain_question.png")
        self.declare_parameter("open_question_window", True)
        # Guard against a model that stays ambiguous no matter what it is told.
        # 'redo' resets the count, since that is a genuinely new observation.
        self.declare_parameter("max_clarification_rounds", 3)

        self._endpoint = self.get_parameter("vllm_endpoint_url").value
        self._model = self.get_parameter("model_name").value
        self._bbox_convention = self.get_parameter("bbox_convention").value
        self._timeout = float(self.get_parameter("timeout_sec").value)
        self._max_tokens = int(self.get_parameter("max_tokens").value)

        seed = self.get_parameter("instruction").value
        self._instruction: str | None = seed or None
        self._frame: Image | None = None
        self._lock = threading.Lock()
        self._busy = False

        self._bridge = CvBridge()

        self.create_subscription(
            String, "/user_instruction", self._instruction_cb, 10)
        self.create_subscription(
            Empty, "/bt/replan_request", self._replan_cb, 10)

        image_path = self.get_parameter("image_path").value
        if image_path:
            self._load_static_frame(image_path)
        else:
            self.create_subscription(
                Image, self.get_parameter("image_topic").value, self._image_cb, 1)

        self._det_pub = self.create_publisher(
            String, self.get_parameter("detections_topic").value, LATCHED)
        self._grounding_pub = self.create_publisher(
            String, self.get_parameter("grounding_topic").value, LATCHED)
        self._image_pub = self.create_publisher(
            Image, self.get_parameter("source_image_topic").value, LATCHED)
        # Observability only — never let anything gate on this. A failed scan is
        # otherwise invisible unless someone is watching this node's console.
        self._status_pub = self.create_publisher(
            String, self.get_parameter("status_topic").value, LATCHED)
        # Clarification: the question, plus a rendered view of the candidates so
        # the operator answers against a picture instead of guessing what the
        # robot sees.
        self._question_pub = self.create_publisher(
            String, self.get_parameter("question_topic").value, LATCHED)
        self._question_image_pub = self.create_publisher(
            Image, self.get_parameter("question_image_topic").value, LATCHED)
        self._scan_id = 0
        # Set while waiting for the operator to disambiguate. The next
        # /user_instruction is treated as an ANSWER to this, not a new command.
        self._pending_question: str | None = None
        self._pending_instruction: str | None = None
        self._clarify_rounds = 0
        self._question_path = self.get_parameter("question_image_path").value
        self._open_window = bool(self.get_parameter("open_question_window").value)
        self._max_rounds = int(self.get_parameter("max_clarification_rounds").value)
        self._viewer_proc = None

        image_src = ("file:" + image_path if image_path
                     else self.get_parameter("image_topic").value)
        self.get_logger().info(
            f"qwen_bridge ready — endpoint={self._endpoint} model={self._model} "
            f"image={image_src}"
            + (f" seed_instruction={self._instruction!r}" if self._instruction else "")
        )

        # Offline one-shot: a static frame plus a seeded instruction has nothing
        # left to wait for, so run once by itself. Deferred by a timer rather
        # than called inline so publishers are connected before we publish.
        # With a live camera we still wait for /user_instruction, because the
        # seeded instruction would otherwise fire against whatever frame
        # happened to arrive first.
        self._autorun_timer = None
        if image_path and self._instruction:
            self._autorun_timer = self.create_timer(1.0, self._autorun_once)

    def _autorun_once(self) -> None:
        self._autorun_timer.cancel()
        self.get_logger().info("offline one-shot — grounding seeded instruction")
        self._dispatch(self._instruction)

    # ── status ───────────────────────────────────────────────────────────────

    def _status(self, state: str, reason: str = "", **extra) -> None:
        """Publish a machine-readable pipeline state.

        state: "running" | "ok" | "failed"
        Diagnostic only. Consumers may display or log it; nothing may block on
        it, because a stalled publisher would then stall the pipeline.
        """
        payload = {"scan_id": self._scan_id, "stage": "qwen_grounding",
                   "state": state, "reason": reason, **extra}
        self._status_pub.publish(String(data=json.dumps(payload)))

    def _publish_question_image(self, src: Image, image_bgr, candidates) -> None:
        """Show the operator EXACTLY the frame the VLM saw, candidates numbered.

        Three deliveries, cheapest first, because none of them is guaranteed to
        be available: a ROS topic (rqt_image_view), a file on disk, and a
        desktop viewer window.  All failures are non-fatal — a missing preview
        must never abort a scan.
        """
        try:
            vis = image_bgr.copy()
            for n, c in enumerate(candidates, start=1):
                x1, y1, x2, y2 = (int(v) for v in c["bbox_xyxy"])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
                label = f"{n}. {c['label']}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                ty = max(th + 4, y1 - 4)
                # Filled plate behind the text — labels vanish against a busy
                # workspace otherwise.
                cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 4, ty + 2),
                              (0, 200, 255), -1)
                cv2.putText(vis, label, (x1 + 2, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
                            cv2.LINE_AA)

            msg = self._bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            msg.header = src.header
            self._question_image_pub.publish(msg)

            if self._question_path:
                cv2.imwrite(self._question_path, vis)
                self.get_logger().info(f"candidate view → {self._question_path}")
                if self._open_window:
                    self._open_viewer(self._question_path)
        except Exception as exc:  # noqa: BLE001 — a preview must never break the scan
            self.get_logger().warn(f"could not render question image: {exc}")

    def _open_viewer(self, path: str) -> None:
        """Pop the candidate view open in whatever image viewer exists.

        Replaces the previous window rather than stacking one per round.
        Headless machines have no viewer and no DISPLAY; that is fine, the topic
        and the file are still there.
        """
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            self.get_logger().info(
                "no display — view the candidates via /slow_brain/question_image")
            return

        viewer = next((v for v in ("eog", "xdg-open", "feh", "display")
                       if shutil.which(v)), None)
        if viewer is None:
            self.get_logger().info(
                "no image viewer found (tried eog, xdg-open, feh, display)")
            return

        try:
            if self._viewer_proc is not None and self._viewer_proc.poll() is None:
                self._viewer_proc.terminate()
            self._viewer_proc = subprocess.Popen(
                [viewer, path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"could not open viewer: {exc}")

    # ── static test frame ────────────────────────────────────────────────────

    def _load_static_frame(self, path: str) -> None:
        """Use a file on disk as the camera frame (offline testing)."""
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            raise RuntimeError(
                f"image_path is set but the file could not be read: {path!r}")

        msg = self._bridge.cv2_to_imgmsg(image_bgr, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "ee_camera_static"
        self._frame = msg

        h, w = image_bgr.shape[:2]
        self.get_logger().warn(
            f"TEST MODE — using static image {path} ({w}x{h}); "
            "image_topic is NOT subscribed")

    # ── callbacks ────────────────────────────────────────────────────────────

    def _image_cb(self, msg: Image) -> None:
        with self._lock:
            self._frame = msg

    def _instruction_cb(self, msg: String) -> None:
        text = msg.data.strip()
        if not text:
            return

        with self._lock:
            pending_q = self._pending_question
            original = self._pending_instruction

        if pending_q is not None and original is not None:
            if text.lower() in ("redo", "retry", "again"):
                # The operator repositioned the scene rather than describing it.
                # Re-run the ORIGINAL instruction unchanged — the new geometry is
                # the disambiguation. _dispatch always reads the latest cached
                # frame, so this automatically grounds a freshly captured image.
                self.get_logger().info(
                    "REDO — re-grounding the original instruction on a new frame")
                with self._lock:
                    self._pending_question = None
                    self._pending_instruction = None
                    self._clarify_rounds = 0     # new scene, fresh budget
                    self._instruction = original
                self._dispatch(original)
                return

            with self._lock:
                self._clarify_rounds += 1
                rounds = self._clarify_rounds
            if rounds > self._max_rounds:
                self.get_logger().error(
                    f"still ambiguous after {self._max_rounds} clarifications — "
                    "giving up. Reposition the scene and type 'redo', or issue a "
                    "new instruction.")
                with self._lock:
                    self._pending_question = None
                    self._pending_instruction = None
                    self._clarify_rounds = 0
                self._status("failed", "clarification_exhausted")
                return

            # Merge rather than replace: the answer ("the red one") is
            # meaningless without the original command. Re-grounding with both
            # keeps this to a single-turn call — no dialogue state in qwen_call.
            # The frame is re-read at dispatch, so the operator may also have
            # nudged the scene while answering.
            merged = (f"{original}\n\n"
                      f"Earlier you asked: {pending_q}\n"
                      f"The operator answered: {text}\n"
                      "Use that answer to pick exactly one TARGET.")
            self.get_logger().info(
                f"clarification {rounds}/{self._max_rounds}: {text!r}")
            with self._lock:
                self._pending_question = None
                self._pending_instruction = None
                self._instruction = merged
            self._dispatch(merged)
            return

        with self._lock:
            self._instruction = text
            self._clarify_rounds = 0
        self.get_logger().info(f"instruction: {text!r}")
        self._dispatch(text)

    def _replan_cb(self, _msg: Empty) -> None:
        """BT asked for a fresh scan after exhausting its grasp candidates."""
        with self._lock:
            instruction = self._instruction
        if instruction is None:
            self.get_logger().warn("replan requested but no instruction cached")
            return
        self.get_logger().info("replan requested — re-grounding cached instruction")
        self._dispatch(instruction)

    # ── worker ───────────────────────────────────────────────────────────────

    def _dispatch(self, instruction: str) -> None:
        with self._lock:
            if self._busy:
                self.get_logger().warn("VLM call already in flight — request dropped")
                return
            frame = self._frame
            if frame is None:
                self.get_logger().warn(
                    "no camera frame cached yet — is the EE image topic publishing?")
                return
            self._busy = True

        threading.Thread(
            target=self._run, args=(instruction, frame), daemon=True).start()

    def _run(self, instruction: str, frame: Image) -> None:
        t0 = time.monotonic()
        self._scan_id += 1
        self._status("running", instruction=instruction)
        try:
            image_bgr = self._bridge.imgmsg_to_cv2(frame, desired_encoding="bgr8")

            objects, grounding, meta = ground(
                image_bgr,
                instruction,
                endpoint_url=self._endpoint,
                model=self._model,
                bbox_convention=self._bbox_convention,
                timeout_sec=self._timeout,
                max_tokens=self._max_tokens,
            )

            # AMBIGUOUS: several objects match equally. Ask rather than guess —
            # publishing one of N arbitrary candidates is worse than stopping,
            # because the arm would confidently grasp the wrong thing.
            if meta.get("ambiguous"):
                question = meta["question"]
                cands = meta["target_candidates"]
                with self._lock:
                    self._pending_question = question
                    self._pending_instruction = instruction
                self.get_logger().warn(
                    f"AMBIGUOUS — {len(cands)} candidates: "
                    f"{[c['label'] for c in cands]}. Asking the operator.")
                self.get_logger().warn(f"QUESTION: {question}")
                self._question_pub.publish(String(data=json.dumps({
                    "scan_id": self._scan_id,
                    "question": question,
                    "candidates": cands,
                    "image_path": self._question_path,
                    "options": [
                        "Describe which one you mean "
                        "(e.g. 'the left one', 'the red one')",
                        "Move the object you want into clear view, "
                        "then type 'redo' to re-scan",
                    ],
                })))
                self._publish_question_image(frame, image_bgr, cands)
                self._status("needs_input", "ambiguous_target",
                             question=question,
                             candidates=[c["label"] for c in cands])
                return

            # FATAL: without a target there is nothing to grasp, and the
            # projector would emit no "target" key at all. Publishing a
            # target-less scene would stall the BT with no explanation, so fail
            # loudly and leave the previous latched scan in place.
            if not meta["has_target"]:
                self.get_logger().error(
                    "no TARGET found — nothing to grasp. Not publishing. "
                    f"Detected: {[d.label for d in objects]}")
                self._status("failed", "no_target",
                             detected=[d.label for d in objects])
                return

            # NON-FATAL: a pick-only instruction ("pick up the book") has no
            # destination by design, and a named-but-unsegmented destination is
            # recoverable too. Publish the target either way — the BT can pick
            # and hold. Only the place phase is unavailable.
            if meta.get("pick_only"):
                self.get_logger().info(
                    "pick-only instruction — no destination. Publishing target "
                    "only; the place phase will be skipped.")
            elif not meta["has_destination"]:
                self.get_logger().warn(
                    "destination named but not segmentable — publishing target "
                    "only; the place phase will have no pose.")
            if meta["warning"]:
                self.get_logger().warn(meta["warning"])

            # source_image is published LAST and is the intended trigger for
            # sam_a100 — same pattern as GSAM publishing the mask last to
            # trigger the projector. ROS 2 gives no cross-topic ordering
            # guarantee, so a consumer that triggers on the FIRST publish here
            # can fire before the later ones arrive. Cache detections, trigger
            # on the image.
            self._det_pub.publish(
                String(data=json.dumps(build_labeled_detections(objects))))
            # exclude_none: a None destination must OMIT the key, not emit null.
            # bt_pkg does j.contains("destination") then .value() on it, which
            # would throw inside its parser on a null.
            self._grounding_pub.publish(
                String(data=to_json(grounding, exclude_none=True)))
            self._image_pub.publish(frame)

            dest = grounding.destination
            dest_desc = (
                f"dest={dest.reference_label!r} type={dest.type} "
                f"relation={dest.relation or '-'} region={dest.region or '-'}"
                if dest is not None else "dest=<none, pick-only>")
            self.get_logger().info(
                f"grounded in {time.monotonic() - t0:.1f}s — "
                f"{meta['n_objects']} objects, "
                f"target={grounding.target_label!r} {dest_desc}")
            self._status(
                "ok",
                target=grounding.target_label,
                destination=(dest.reference_label if dest else None),
                pick_only=bool(meta.get("pick_only")),
                n_objects=meta["n_objects"],
                confidence=grounding.confidence,
                latency_s=round(time.monotonic() - t0, 2),
            )
        except Exception as exc:  # noqa: BLE001 — a failed scan must not kill the node
            self.get_logger().error(f"grounding failed after "
                                    f"{time.monotonic() - t0:.1f}s: {exc}")
            self._status("failed", str(exc)[:200])
        finally:
            with self._lock:
                self._busy = False


def main(args=None) -> None:
    rclpy.init(args=args)
    node = QwenBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
