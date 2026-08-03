"""Prompt the operator for instructions on stdin, publish to /user_instruction.

Also surfaces clarification questions from qwen_bridge: when the Slow Brain
cannot decide between several candidate targets it publishes a question, which
is printed here.  The operator's next line is treated as the ANSWER — the bridge
merges it with the original instruction and re-grounds.

Runs the blocking input() loop on a daemon thread so rclpy keeps spinning.

``ros2 launch`` does not forward stdin to child processes, so the launch file
wraps this node in an xterm.  Running it directly with ``ros2 run`` works fine
in an interactive terminal.
"""
from __future__ import annotations

import json
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

# Latched: qwen_bridge may start after the operator has already typed a
# command, and we would rather it pick up the last instruction than sit idle.
LATCHED = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class InstructionPromptNode(Node):

    def __init__(self) -> None:
        super().__init__("instruction_prompt_node")

        self.declare_parameter("topic", "/user_instruction")
        self.declare_parameter("question_topic", "/slow_brain/question")
        topic = self.get_parameter("topic").value

        self._pub = self.create_publisher(String, topic, LATCHED)
        self.create_subscription(
            String, self.get_parameter("question_topic").value,
            self._question_cb, 10)
        self.get_logger().info(f"Publishing instructions to {topic}")

        threading.Thread(target=self._input_loop, daemon=True).start()

    def _question_cb(self, msg: String) -> None:
        """Print a clarification request above the waiting prompt."""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        cands = data.get("candidates", [])
        print("\n" + "!" * 64)
        print("  AMBIGUOUS — the robot found several possible targets")
        print()
        print(f"  {data.get('question', '')}")
        if cands:
            print()
            for n, c in enumerate(cands, start=1):
                print(f"    {n}. {c.get('label', '?')}")
        img = data.get("image_path")
        if img:
            print("\n  A window should have opened showing the frame it used.")
            print(f"  If not: {img}   (or topic /slow_brain/question_image)")
        print()
        print("  You can either:")
        for opt in data.get("options", []):
            print(f"    - {opt}")
        print()
        print("  Anything you type is added to your original command; 'redo'")
        print("  re-scans with a fresh camera frame instead.")
        print("!" * 64)
        print("instruction> ", end="", flush=True)

    def _input_loop(self) -> None:
        print("\n" + "=" * 60)
        print("  Slow Brain — type a manipulation instruction, then Enter.")
        print("  Example: put the book in the box")
        print("  Ctrl-D or Ctrl-C to quit.")
        print("=" * 60 + "\n")

        while rclpy.ok():
            try:
                text = input("instruction> ").strip()
            except (EOFError, KeyboardInterrupt):
                self.get_logger().info("stdin closed — instruction prompt exiting")
                return

            if not text:
                continue

            self._pub.publish(String(data=text))
            self.get_logger().info(f"published: {text!r}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InstructionPromptNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
