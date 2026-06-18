"""
AI 评估微服务
从 Kafka 消费对比结果，调用 LLM 生成评估报告，发送到 Kafka Topic: rehab.reports
"""

import os
import sys
import json
import signal
import logging
import time

sys.path.insert(0, '/app/src')
from rehab_pose.kafka_utils import (
    KafkaProducerWrapper, KafkaConsumerWrapper,
    ComparisonMessage, ReportMessage,
    TOPIC_COMPARISONS, TOPIC_REPORTS, wait_for_kafka,
)
from rehab_pose.keypoints import EvaluationResult
from rehab_pose.ai_evaluator import AIEvaluator
from rehab_pose.database import (
    init_database, save_rehab_log,
)

logging.basicConfig(level=logging.INFO, format='[AIEvaluator] %(message)s')
logger = logging.getLogger(__name__)

running = True


def signal_handler(sig, frame):
    global running
    logger.info("收到停止信号，正在关闭...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


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
        topics=[TOPIC_COMPARISONS],
        group_id="ai-evaluator-group",
    )

    # 初始化 AI 评估引擎
    logger.info("正在初始化 AI 评估引擎...")
    evaluator = AIEvaluator()
    logger.info("AI 评估引擎已就绪")

    def process_comparison(topic: str, data: dict):
        """处理对比结果：调用 LLM 生成评估报告"""
        session_id = data.get("session_id", "")
        action_name = data.get("action_name", "")
        patient_id = data.get("patient_id", "")

        logger.info(f"收到对比结果: session={session_id}, score={data.get('similarity_score')}")

        try:
            # 构建 EvaluationResult
            result = EvaluationResult(
                action_name=action_name,
                angle_deviations=data.get("angle_deviations", {}),
                avg_keypoint_offset=data.get("avg_keypoint_offset", 0.0),
                rom_comparison=data.get("rom_comparison", {}),
                similarity_score=data.get("similarity_score", 0.0),
            )

            # 调用 AI 评估
            logger.info(f"正在生成 AI 评估报告: {action_name}")
            report = evaluator.evaluate(result, action_name)

            if report is None:
                report = {
                    "overall_score": data.get("similarity_score", 0),
                    "grade": "未知",
                    "joint_analysis": [],
                    "problems": [],
                    "suggestions": ["AI 评估服务暂时不可用"],
                    "risk_warnings": [],
                    "note": "AI 评估失败，返回基础评分",
                }

            # 保存到数据库
            log_id = save_rehab_log(
                patient_id=patient_id,
                action_name=action_name,
                similarity_score=data.get("similarity_score", 0.0),
                angle_deviations=data.get("angle_deviations", {}),
                avg_keypoint_offset=data.get("avg_keypoint_offset", 0.0),
                rom_comparison=data.get("rom_comparison", {}),
                ai_report=json.dumps(report, ensure_ascii=False),
                rag_sources=report.get("rag_sources", []),
                duration_s=0.0,
            )

            # 发送报告到 Kafka
            grade = report.get("grade", "N/A")
            report_msg = ReportMessage(
                session_id=session_id,
                action_name=action_name,
                patient_id=patient_id,
                similarity_score=data.get("similarity_score", 0.0),
                grade=grade,
                report=report,
                log_id=log_id,
                angle_deviations=data.get("angle_deviations", {}),
                avg_keypoint_offset=data.get("avg_keypoint_offset", 0.0),
                rom_comparison=data.get("rom_comparison", {}),
                frame_count=data.get("frame_count", 0),
            )

            producer.produce(TOPIC_REPORTS, report_msg, key=session_id)
            logger.info(f"AI 评估报告已发送: session={session_id}, grade={grade}, log_id={log_id}")

        except Exception as e:
            logger.error(f"AI 评估失败: {e}")
            # 发送错误报告
            error_report = ReportMessage(
                session_id=session_id,
                action_name=action_name,
                patient_id=patient_id,
                similarity_score=data.get("similarity_score", 0.0),
                grade="错误",
                report={"error": str(e)},
                log_id=0,
            )
            producer.produce(TOPIC_REPORTS, error_report, key=session_id)

    # 启动消费
    logger.info("AI 评估服务已启动，等待对比结果...")
    consumer.start(process_comparison)

    # 清理
    producer.close()
    logger.info("AI 评估服务已停止")


if __name__ == "__main__":
    main()
