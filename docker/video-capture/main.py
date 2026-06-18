"""
视频采集微服务
从摄像头采集视频帧，Base64 编码后发送到 Kafka Topic: rehab.frames
"""

import os
import sys
import time
import json
import base64
import uuid
import signal
import logging
import cv2
import numpy as np

sys.path.insert(0, '/app/src')
from rehab_pose.kafka_utils import (
    KafkaProducerWrapper, FrameMessage,
    TOPIC_FRAMES, wait_for_kafka,
)

logging.basicConfig(level=logging.INFO, format='[VideoCapture] %(message)s')
logger = logging.getLogger(__name__)

# 配置
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")  # 0=默认摄像头，或 RTSP URL
CAPTURE_FPS = int(os.getenv("CAPTURE_FPS", "15"))
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))

# 全局状态
running = True
session_id = ""
action_name = ""
patient_id = ""


def signal_handler(sig, frame):
    global running
    logger.info("收到停止信号，正在关闭...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def frame_to_base64(frame: np.ndarray, quality: int = 80) -> str:
    """将 OpenCV 帧编码为 Base64 JPEG"""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')


def main():
    global session_id, action_name, patient_id

    # 等待 Kafka 就绪
    if not wait_for_kafka():
        logger.error("无法连接 Kafka，退出")
        sys.exit(1)

    # 初始化 Kafka Producer
    producer = KafkaProducerWrapper()

    # 初始化摄像头
    cam_source = int(CAMERA_SOURCE) if CAMERA_SOURCE.isdigit() else CAMERA_SOURCE
    cap = cv2.VideoCapture(cam_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        logger.error(f"无法打开摄像头: {CAMERA_SOURCE}")
        sys.exit(1)

    logger.info(f"摄像头已打开: {CAMERA_SOURCE}, 分辨率: {FRAME_WIDTH}x{FRAME_HEIGHT}")

    # 启动控制台命令监听线程（接收前端控制指令）
    import threading

    def listen_commands():
        """监听 stdin 命令（用于前端控制开始/停止录制）"""
        global session_id, action_name, patient_id
        for line in sys.stdin:
            try:
                cmd = json.loads(line.strip())
                if cmd.get("type") == "start":
                    session_id = cmd.get("session_id", str(uuid.uuid4()))
                    action_name = cmd.get("action_name", "")
                    patient_id = cmd.get("patient_id", "")
                    logger.info(f"开始采集: session={session_id}, action={action_name}")
                elif cmd.get("type") == "stop":
                    session_id = ""
                    logger.info("停止采集")
            except json.JSONDecodeError:
                pass

    cmd_thread = threading.Thread(target=listen_commands, daemon=True)
    cmd_thread.start()

    # 主循环：采集帧并发送到 Kafka
    frame_count = 0
    interval = 1.0 / CAPTURE_FPS

    logger.info(f"视频采集服务已启动，FPS={CAPTURE_FPS}")

    while running:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            logger.warning("读取帧失败，重试...")
            time.sleep(0.1)
            continue

        # 只有在录制状态时才发送帧到 Kafka
        if session_id:
            ts_ms = int(time.time() * 1000)
            img_b64 = frame_to_base64(frame, JPEG_QUALITY)

            msg = FrameMessage(
                frame_id=frame_count,
                timestamp_ms=ts_ms,
                image_base64=img_b64,
                width=frame.shape[1],
                height=frame.shape[0],
                session_id=session_id,
                action_name=action_name,
                patient_id=patient_id,
            )

            producer.produce(TOPIC_FRAMES, msg, key=session_id)
            frame_count += 1

            if frame_count % 30 == 0:
                logger.info(f"已采集 {frame_count} 帧")

        # 控制帧率
        elapsed = time.time() - loop_start
        sleep_time = max(0, interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    # 清理
    cap.release()
    producer.close()
    logger.info(f"视频采集服务已停止，共采集 {frame_count} 帧")


if __name__ == "__main__":
    main()
