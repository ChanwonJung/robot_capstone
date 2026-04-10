#!/usr/bin/env python3
"""
테스트용 이미지 퍼블리셔 노드.
test_inference.launch.py 와 함께 사용.

이미지 경로와 발행 속도는 아래 설정 블록에서 수정.
"""
# =============================================
# 여기만 수정
IMAGE_PATH = "/home/parksanghyun/Pictures/Screenshots/cafe_ocean.png"
HZ = 0.2   # 발행 주기 (Hz). CPU 추론이 느리므로 낮게 유지 (0.2 = 5초마다)
TOPIC = "/camera/image_raw"
# =============================================

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from sensor_msgs.msg import Image

# TRANSIENT_LOCAL = latched publisher
# 늦게 구독해도 마지막 메시지를 받을 수 있음 (모델 로딩 시간 무관)
LATCHED_QOS = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


class TestImagePublisher(Node):
    def __init__(self):
        super().__init__("test_image_publisher")

        image_bgr = cv2.imread(IMAGE_PATH)
        if image_bgr is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

        self.bridge = CvBridge()
        self.msg = self.bridge.cv2_to_imgmsg(image_bgr, encoding="bgr8")
        self.pub = self.create_publisher(Image, TOPIC, qos_profile=LATCHED_QOS)

        # 구독자가 준비될 때까지 잠깐 대기 후 한 번만 발행
        self.timer = self.create_timer(1.0, self._publish_once)

        self.get_logger().info(f"1초 후 이미지 1장 발행: {IMAGE_PATH} → {TOPIC}")

    def _publish_once(self):
        self.timer.cancel()
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.msg)
        self.get_logger().info("이미지 발행 완료. 추론 결과를 기다리는 중...")


def main(args=None):
    rclpy.init(args=args)
    node = TestImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
