"""Kafka 通信工具模块 —— 封装 Producer/Consumer，统一消息格式"""

import os
import json
import time
import logging
from typing import Optional, Callable, Any
from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# ── Kafka 配置 ──────────────────────────────────────────────

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

# Topic 常量
TOPIC_FRAMES = "rehab.frames"           # 视频帧
TOPIC_POSES = "rehab.poses"             # 姿态检测结果
TOPIC_COMPARISONS = "rehab.comparisons"  # 姿态对比结果
TOPIC_REPORTS = "rehab.reports"          # AI 评估报告


# ── 消息格式 ────────────────────────────────────────────────

@dataclass
class FrameMessage:
    """视频帧消息"""
    frame_id: int
    timestamp_ms: int
    image_base64: str          # Base64 编码的 JPEG 图像
    width: int = 0
    height: int = 0
    session_id: str = ""       # 评估会话 ID
    action_name: str = ""      # 动作名称
    patient_id: str = ""       # 患者 ID


@dataclass
class PoseMessage:
    """姿态检测结果消息"""
    frame_id: int
    timestamp_ms: int
    keypoints: dict            # 17 关键点 {name: {x, y, conf}}
    session_id: str = ""
    detected: bool = True


@dataclass
class ComparisonMessage:
    """姿态对比结果消息"""
    session_id: str
    action_name: str
    patient_id: str
    similarity_score: float
    angle_deviations: dict
    avg_keypoint_offset: float
    rom_comparison: dict
    frame_count: int = 0


@dataclass
class ReportMessage:
    """AI 评估报告消息"""
    session_id: str
    action_name: str
    patient_id: str
    similarity_score: float
    grade: str
    report: dict               # 完整 AI 报告
    log_id: int = 0
    angle_deviations: dict = None
    avg_keypoint_offset: float = 0.0
    rom_comparison: dict = None
    frame_count: int = 0


# ── Kafka Producer ──────────────────────────────────────────

class KafkaProducerWrapper:
    """Kafka 生产者封装"""

    def __init__(self, bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS):
        self.conf = {
            'bootstrap.servers': bootstrap_servers,
            'client.id': 'rehab-producer',
            'acks': 'all',
            'retries': 3,
            'linger.ms': 5,
        }
        self._producer = Producer(self.conf)
        self._delivery_reports = {}

    def _delivery_callback(self, err, msg):
        """消息投递回调"""
        if err:
            logger.error(f"消息投递失败: {err}")
        else:
            logger.debug(f"消息已投递: {msg.topic()} [{msg.partition()}] @ {msg.offset()}")

    def produce(self, topic: str, value: Any, key: str = None):
        """
        发送消息到指定 Topic。

        Args:
            topic: Kafka Topic 名称
            value: 消息内容（dataclass 或 dict）
            key: 消息键（用于分区）
        """
        if hasattr(value, '__dataclass_fields__'):
            data = asdict(value)
        elif isinstance(value, dict):
            data = value
        else:
            data = {"value": str(value)}

        payload = json.dumps(data, ensure_ascii=False).encode('utf-8')
        key_bytes = key.encode('utf-8') if key else None

        try:
            self._producer.produce(
                topic=topic,
                value=payload,
                key=key_bytes,
                callback=self._delivery_callback,
            )
            self._producer.poll(0)  # 触发回调
        except KafkaException as e:
            logger.error(f"Kafka 发送失败: {e}")

    def flush(self, timeout: float = 10.0):
        """等待所有消息发送完成"""
        self._producer.flush(timeout)

    def close(self):
        """关闭 Producer"""
        self.flush()
        logger.info("Kafka Producer 已关闭")


# ── Kafka Consumer ──────────────────────────────────────────

class KafkaConsumerWrapper:
    """Kafka 消费者封装"""

    def __init__(
        self,
        topics: list[str],
        group_id: str,
        bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
    ):
        self.conf = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'latest',
            'enable.auto.commit': True,
            'session.timeout.ms': 30000,
        }
        self._consumer = Consumer(self.conf)
        self._topics = topics
        self._running = False

    def start(self, callback: Callable[[str, dict], None]):
        """
        开始消费消息。

        Args:
            callback: 回调函数 (topic, message_dict) -> None
        """
        self._consumer.subscribe(self._topics)
        self._running = True
        logger.info(f"Kafka Consumer 已启动，监听: {self._topics}")

        while self._running:
            msg = self._consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error(f"Consumer 错误: {msg.error()}")
                continue

            try:
                topic = msg.topic()
                data = json.loads(msg.value().decode('utf-8'))
                callback(topic, data)
            except json.JSONDecodeError as e:
                logger.error(f"消息解析失败: {e}")
            except Exception as e:
                logger.error(f"处理消息异常: {e}")

    def stop(self):
        """停止消费"""
        self._running = False
        self._consumer.close()
        logger.info("Kafka Consumer 已停止")


# ── 工具函数 ────────────────────────────────────────────────

def wait_for_kafka(bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
                   max_retries: int = 30,
                   retry_interval: float = 2.0) -> bool:
    """等待 Kafka 服务就绪"""
    from confluent_kafka.admin import AdminClient

    for i in range(max_retries):
        try:
            admin = AdminClient({'bootstrap.servers': bootstrap_servers})
            metadata = admin.list_topics(timeout=5)
            if metadata.brokers:
                logger.info(f"Kafka 已就绪，发现 {len(metadata.brokers)} 个 Broker")
                return True
        except Exception:
            pass
        logger.info(f"等待 Kafka 就绪... ({i+1}/{max_retries})")
        time.sleep(retry_interval)

    logger.error("Kafka 连接超时")
    return False
