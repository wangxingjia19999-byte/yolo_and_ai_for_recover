"""
Web 前端微服务
Streamlit + Kafka Consumer，实时展示姿态检测和 AI 评估报告
"""

import os
import sys
import json
import time
import threading
import logging

sys.path.insert(0, '/app/src')
from rehab_pose.kafka_utils import (
    KafkaConsumerWrapper, ReportMessage,
    TOPIC_REPORTS, wait_for_kafka,
)
from rehab_pose.database import (
    init_database, get_rehab_logs, get_score_trend,
    list_standard_actions, load_standard_action,
)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

logging.basicConfig(level=logging.INFO, format='[WebApp] %(message)s')
logger = logging.getLogger(__name__)

# ── 全局报告缓存 ────────────────────────────────────────────
# 使用线程安全的方式存储最新的评估报告
import queue
report_queue = queue.Queue(maxsize=100)


def start_kafka_consumer():
    """后台线程：消费 Kafka 评估报告"""
    if not wait_for_kafka(max_retries=5):
        logger.warning("Kafka 未就绪，跳过 Consumer 启动")
        return

    consumer = KafkaConsumerWrapper(
        topics=[TOPIC_REPORTS],
        group_id="web-app-group",
    )

    def on_report(topic, data):
        try:
            report_queue.put_nowait(data)
        except queue.Full:
            report_queue.get()  # 丢弃最旧的
            report_queue.put_nowait(data)

    consumer.start(on_report)


# 启动后台 Kafka Consumer
if "kafka_started" not in st.session_state:
    t = threading.Thread(target=start_kafka_consumer, daemon=True)
    t.start()
    st.session_state.kafka_started = True

# ── 页面配置 ────────────────────────────────────────────────
st.set_page_config(
    page_title="康复训练姿态评估系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 初始化数据库
init_database()

# ── 侧边栏 ─────────────────────────────────────────────────
st.sidebar.title("🏥 康复训练姿态评估系统")
st.sidebar.markdown("基于 YOLO + AI 大模型 · Kafka 微服务架构")

page = st.sidebar.radio(
    "导航菜单",
    ["📹 标准动作管理", "🏃 实时评估", "📊 康复日志", "⚙️ 系统状态"],
)

patient_id = st.sidebar.text_input("患者标识", value="patient_001")

# ── 页面一：标准动作管理 ────────────────────────────────────
if page == "📹 标准动作管理":
    st.title("📹 标准动作管理")
    st.markdown("管理标准动作模板库。")

    actions = list_standard_actions()
    if actions:
        st.subheader("已保存的标准动作")
        df = pd.DataFrame(actions)
        df.columns = ["动作ID", "动作名称", "描述", "创建时间"]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("暂无标准动作，请通过视频采集服务录制。")

# ── 页面二：实时评估 ────────────────────────────────────────
elif page == "🏃 实时评估":
    st.title("🏃 实时评估")
    st.markdown("通过摄像头进行实时姿态评估，AI 自动生成康复报告。")

    # 选择标准动作
    actions = list_standard_actions()
    if not actions:
        st.warning("请先录制标准动作。")
        st.stop()

    action_options = {a["action_id"]: a["action_name"] for a in actions}
    selected_id = st.selectbox(
        "选择标准动作",
        list(action_options.keys()),
        format_func=lambda x: action_options[x],
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("控制面板")
        st.info("请通过视频采集服务的 stdin 发送开始/停止命令。")
        st.code('{"type":"start","session_id":"sess_001","action_name":"站立抬腿","patient_id":"patient_001"}')

    with col2:
        st.subheader("最新评估报告")

        # 从队列获取最新报告
        latest_report = None
        while not report_queue.empty():
            try:
                latest_report = report_queue.get_nowait()
            except queue.Empty:
                break

        if latest_report:
            report = latest_report.get("report", {})
            score = latest_report.get("similarity_score", 0)
            grade = latest_report.get("grade", "N/A")

            # 评分指标
            metric_cols = st.columns(3)
            metric_cols[0].metric("相似度评分", f"{score:.1f}/100")
            metric_cols[1].metric("评估等级", grade)
            metric_cols[2].metric("帧数", latest_report.get("frame_count", 0))

            # AI 报告
            st.markdown("### 🤖 AI 康复评估报告")

            if "joint_analysis" in report:
                for ja in report.get("joint_analysis", []):
                    st.write(f"- **{ja.get('joint', '')}**: {ja.get('assessment', '')} (置信度: {ja.get('confidence', '')})")

            if "problems" in report:
                st.markdown("#### 发现的问题")
                for p in report.get("problems", []):
                    st.warning(f"**{p.get('problem', '')}** — {p.get('severity', '')}: {p.get('possible_cause', '')}")

            if "suggestions" in report:
                st.markdown("#### 改进建议")
                for s in report.get("suggestions", []):
                    st.success(f"✅ {s}")

            if "risk_warnings" in report:
                st.markdown("#### 风险提示")
                for r in report.get("risk_warnings", []):
                    st.error(f"⚠️ {r}")
        else:
            st.info("等待评估结果... 请开始训练评估。")

# ── 页面三：康复日志 ────────────────────────────────────────
elif page == "📊 康复日志":
    st.title("📊 康复日志")

    logs = get_rehab_logs(patient_id, limit=50)
    if logs:
        df = pd.DataFrame(logs)
        st.dataframe(df[["id", "action_name", "similarity_score", "created_at"]], use_container_width=True, hide_index=True)

        # 趋势图
        trend = get_score_trend(patient_id)
        if trend:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[t["created_at"] for t in trend],
                y=[t["similarity_score"] for t in trend],
                mode='lines+markers',
                name='相似度评分',
            ))
            fig.update_layout(title="评分趋势", xaxis_title="时间", yaxis_title="评分")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无康复日志。")

# ── 页面四：系统状态 ────────────────────────────────────────
elif page == "⚙️ 系统状态":
    st.title("⚙️ 系统状态")

    st.markdown("""
    ### 微服务架构

    | 服务 | 状态 | 说明 |
    |------|------|------|
    | video-capture | 🟢 运行中 | 视频帧采集 → Kafka |
    | yolo-inference | 🟢 运行中 | YOLO 推理（GPU） → Kafka |
    | compare-service | 🟢 运行中 | DTW 对比 → Kafka |
    | ai-evaluator | 🟢 运行中 | LLM 评估 → Kafka |
    | web-app | 🟢 运行中 | Streamlit 前端 |
    """)

    st.code("""
    # 查看所有服务状态
    docker-compose ps

    # 查看服务日志
    docker-compose logs -f yolo-inference

    # 重启某个服务
    docker-compose restart ai-evaluator
    """)
