"""
YOLO 推理微服务
从 Kafka 消费视频帧，运行 YOLO-Pose 检测，将关键点结果发送到 Kafka Topic: rehab.poses
支持 NVIDIA GPU (CUDA) 加速
"""

import os
import sys
import json
import base64
import signal
import logging
import time
import cv2
import numpy as np

sys.path.insert(0, '/app/src')
from rehab_pose.kafka_utils import (
    KafkaProducerWrapper, KafkaConsumerWrapper,
    FrameMessage, PoseMessage,
    TOPIC_FRAMES, TOPIC_POSES, wait_for_kafka,
)
from rehab_pose.config import KEYPOINT_NAMES, YOLO_CONF_THRESHOLD
from rehab_pose.pose_detector import PoseDetector

logging.basicConfig(level=logging.INFO, format='[YOLOInference] %(message)s')
logger = logging.getLogger(__name__)

running = True


def signal_handler(sig, frame):
    global running
    logger.info("收到停止信号，正在关闭...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def base64_to_frame(b64_str: str) -> np.ndarray:
    """将 Base64 JPEG 解码为 OpenCV 帧"""
    img_bytes = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def main():
    global running

    # 等待 Kafka 就绪
    if not wait_for_kafka():
        logger.error("无法连接 Kafka，退出")
        sys.exit(1)

    # 初始化 Kafka
    producer = KafkaProducerWrapper()
    consumer = KafkaConsumerWrapper(
        topics=[TOPIC_FRAMES],
        group_id="yolo-inference-group",
    )

    # 初始化 YOLO 模型
    logger.info("正在加载 YOLO-Pose 模型...")
    detector = PoseDetector()
    detector.load_model()
    logger.info("YOLO-Pose 模型加载完成")

    # 统计
    processed_count = 0
    start_time = time.time()

    def process_frame(topic: str, data: dict):
        """处理单帧：YOLO 推理 → 发送结果"""
        nonlocal processed_count

        try:
            # 解码帧
            frame = base64_to_frame(data["image_base64"])
            if frame is None:
                return

            # YOLO 推理
            kpts = detector.extract_keypoints(frame)

            if kpts is not None:
                # 构建关键点字典
                keypoints_dict = {}
                for i, name in enumerate(KEYPOINT_NAMES):
                    keypoints_dict[name] = {
                        "x": float(kpts[i, 0]),
                        "y": float(kpts[i, 1]),
                        "conf": float(kpts[i, 2]),
                    }

                pose_msg = PoseMessage(
                    frame_id=data["frame_id"],
                    timestamp_ms=data["timestamp_ms"],
                    keypoints=keypoints_dict,
                    session_id=data.get("session_id", ""),
                    detected=True,
                )
            else:
                # 未检测到人体
                pose_msg = PoseMessage(
                    frame_id=data["frame_id"],
                    timestamp_ms=data["timestamp_ms"],
                    keypoints={},
                    session_id=data.get("session_id", ""),
                    detected=False,
                )

            # 发送结果
            producer.produce(TOPIC_POSES, pose_msg, key=data.get("session_id", ""))

            processed_count += 1
            if processed_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = processed_count / elapsed if elapsed > 0 else 0
                logger.info(f"已处理 {processed_count} 帧, FPS={fps:.1f}")

        except Exception as e:
            logger.error(f"处理帧失败: {e}")

    # 启动消费
    logger.info("YOLO 推理服务已启动，等待视频帧...")
    consumer.start(process_frame)

    # 清理
    producer.close()
    logger.info(f"YOLO 推理服务已停止，共处理 {processed_count} 帧")


if __name__ == "__main__":
    main()
