"""
姿态对比微服务
从 Kafka 消费 pose 结果，缓存帧序列，执行 DTW 对比，发送结果到 Kafka Topic: rehab.comparisons
"""

import os
import sys
import json
import signal
import logging
import time
from collections import defaultdict

sys.path.insert(0, '/app/src')
from rehab_pose.kafka_utils import (
    KafkaProducerWrapper, KafkaConsumerWrapper,
    PoseMessage, ComparisonMessage,
    TOPIC_POSES, TOPIC_COMPARISONS, wait_for_kafka,
)
from rehab_pose.keypoints import PoseFrame, Keypoint2D, EvaluationResult
from rehab_pose.pose_comparator import compare_poses
from rehab_pose.database import (
    init_database, load_standard_action, save_rehab_log,
)

logging.basicConfig(level=logging.INFO, format='[CompareService] %(message)s')
logger = logging.getLogger(__name__)

running = True


def signal_handler(sig, frame):
    global running
    logger.info("收到停止信号，正在关闭...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# 帧缓存：按 session_id 缓存 pose 帧
frame_cache: dict[str, list[PoseFrame]] = defaultdict(list)
session_meta: dict[str, dict] = {}


def main():
    global running

    # 等待 Kafka 就绪
    if not wait_for_kafka():
        logger.error("无法连接 Kafka，退出")
        sys.exit(1)

    # 初始化数据库
    init_database()

    # 初始化 Kafka
    producer = KafkaProducerWrapper()
    consumer = KafkaConsumerWrapper(
        topics=[TOPIC_POSES],
        group_id="compare-service-group",
    )

    def process_pose(topic: str, data: dict):
        """处理 pose 结果：缓存帧，当收到结束信号时执行对比"""
        session_id = data.get("session_id", "")
        if not session_id:
            return

        # 检查是否是结束信号
        if data.get("type") == "session_end":
            logger.info(f"会话 {session_id} 结束，开始姿态对比...")
            execute_comparison(session_id, producer)
            return

        # 缓存 pose 帧
        if data.get("detected", False) and data.get("keypoints"):
            keypoints = {}
            for name, kp_data in data["keypoints"].items():
                keypoints[name] = Keypoint2D(
                    x=kp_data["x"],
                    y=kp_data["y"],
                    conf=kp_data["conf"],
                )

            pose_frame = PoseFrame(
                frame_id=data["frame_id"],
                timestamp_ms=data["timestamp_ms"],
                keypoints=keypoints,
            )
            frame_cache[session_id].append(pose_frame)

            # 保存会话元数据
            if session_id not in session_meta:
                session_meta[session_id] = {
                    "action_name": data.get("action_name", ""),
                    "patient_id": data.get("patient_id", ""),
                }

    def execute_comparison(session_id: str, producer: KafkaProducerWrapper):
        """执行姿态对比并发送结果"""
        frames = frame_cache.get(session_id, [])
        meta = session_meta.get(session_id, {})
        action_name = meta.get("action_name", "")
        patient_id = meta.get("patient_id", "")

        if not frames:
            logger.warning(f"会话 {session_id} 无有效帧，跳过对比")
            cleanup_session(session_id)
            return

        # 加载标准动作
        # 注意：这里需要 action_id，实际中应该通过 session 传递
        # 简化处理：通过 action_name 查找
        from rehab_pose.database import list_standard_actions
        actions = list_standard_actions()
        std_action = None
        for a in actions:
            if a["action_name"] == action_name:
                std_action = load_standard_action(a["action_id"])
                break

        if std_action is None:
            logger.warning(f"未找到标准动作: {action_name}")
            cleanup_session(session_id)
            return

        # 执行对比
        result = compare_poses(std_action.frames, frames)

        if result:
            # 发送对比结果
            comp_msg = ComparisonMessage(
                session_id=session_id,
                action_name=action_name,
                patient_id=patient_id,
                similarity_score=result.similarity_score,
                angle_deviations=result.angle_deviations,
                avg_keypoint_offset=result.avg_keypoint_offset,
                rom_comparison=result.rom_comparison,
                frame_count=len(frames),
            )
            producer.produce(TOPIC_COMPARISONS, comp_msg, key=session_id)
            logger.info(f"对比完成: session={session_id}, score={result.similarity_score}")
        else:
            logger.warning(f"对比失败: session={session_id}")

        cleanup_session(session_id)

    def cleanup_session(session_id: str):
        """清理会话缓存"""
        frame_cache.pop(session_id, None)
        session_meta.pop(session_id, None)

    # 启动消费
    logger.info("姿态对比服务已启动，等待 pose 数据...")
    consumer.start(process_pose)

    # 清理
    producer.close()
    logger.info("姿态对比服务已停止")


if __name__ == "__main__":
    main()
