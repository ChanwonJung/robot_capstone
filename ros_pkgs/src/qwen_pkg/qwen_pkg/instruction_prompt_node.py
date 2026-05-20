"""
instruction_prompt_node.py

Prompts the user for a natural-language robot command on stdin and publishes
each entry to /user_instruction (std_msgs/String).  Runs the input() loop on
a daemon thread so rclpy.spin() is not blocked.
"""
from __future__ import annotations

import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class InstructionPromptNode(Node):

    def __init__(self) -> None:
        super().__init__("instruction_prompt_node")
        self._pub = self.create_publisher(String, "/user_instruction", 10)
        self._thread = threading.Thread(target=self._prompt_loop, daemon=True)
        self._thread.start()
        self.get_logger().info(
            "InstructionPromptNode ready — type a command and press Enter"
        )

    def _prompt_loop(self) -> None:
        while rclpy.ok():
            try:
                text = input("\n[robot command] > ").strip()
            except EOFError:
                break
            if not text:
                continue
            self._pub.publish(String(data=text))
            self.get_logger().info(f"Published: '{text}'")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InstructionPromptNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
