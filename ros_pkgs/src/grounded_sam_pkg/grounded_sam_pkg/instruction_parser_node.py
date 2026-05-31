"""InstructionParserNode — 자연어 instruction에서 명사 추출 후 /dino_prompt 발행."""

import os

from google import genai
from google.genai import types
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

_SYSTEM_PROMPT = (
    "Extract all object noun phrases from the user instruction. "
    "Output ONLY a comma-separated list of noun phrases, nothing else. "
    "Example input: 'Grab the yellow banana next to the laptop and put it in the brown box' "
    "Example output: yellow banana, laptop, brown box"
)


class InstructionParserNode(Node):
    def __init__(self):
        super().__init__("instruction_parser_node")

        self.declare_parameter("api_key", "")

        api_key = self.get_parameter("api_key").value or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Gemini API key not set. "
                "Pass via param 'api_key' or env var GEMINI_API_KEY."
            )

        self._client = genai.Client(api_key=api_key)
        self._config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
        )

        self._sub = self.create_subscription(
            String, "/instruction", self._instruction_cb, 10
        )
        self._pub = self.create_publisher(String, "/dino_prompt", 10)

        self.get_logger().info("Ready — waiting for /instruction")

    def _instruction_cb(self, msg: String) -> None:
        instruction = msg.data.strip()
        if not instruction:
            return

        self.get_logger().info(f"Parsing: '{instruction}'")
        try:
            response = self._client.models.generate_content(
                model="gemini-2.5-flash",
                contents=instruction,
                config=self._config,
            )
            nouns = response.text.strip()
        except Exception as e:
            self.get_logger().error(f"Gemini API error: {e}")
            return

        if not nouns:
            self.get_logger().warn("Gemini 빈 응답 — /dino_prompt 발행 생략")
            return

        out = String()
        out.data = nouns
        self._pub.publish(out)
        self.get_logger().info(f"Published /dino_prompt: '{nouns}'")


def main(args=None):
    rclpy.init(args=args)
    node = InstructionParserNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
