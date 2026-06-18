"""Streamlit 前端界面 —— 康复训练姿态评估系统"""

import sys
import time
import json
import tempfile
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rehab_pose.config import (
    KEYPOINT_NAMES, JOINT_ANGLE_DEFS,
    RECORD_FPS, RECORD_DURATION_SEC, RECORD_COUNTDOWN_SEC,
    COMPARISON_WEIGHTS,
)
from rehab_pose.keypoints import (
    PoseFrame, StandardAction, Keypoint2D,
    compute_all_joint_angles, keypoints_to_dict,
)
from rehab_pose.pose_detector import PoseDetector
from rehab_pose.pose_comparator import compare_poses
from rehab_pose.database import (
    init_database, save_standard_action, load_standard_action,
    list_standard_actions, delete_standard_action,
    save_rehab_log, get_rehab_logs, get_score_trend, get_rehab_log,
)
from rehab_pose.ai_evaluator import AIEvaluator
from rehab_pose.visualization import draw_pose_overlay, draw_welcome_screen
from rehab_pose.rehab_logger import (
    generate_score_trend_chart, format_report_for_display,
)


# ── 页面配置 ───────────────────────────────────────────────
st.set_page_config(
    page_title="康复训练姿态评估系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 初始化 ─────────────────────────────────────────────────
@st.cache_resource
def init_system():
    """初始化数据库和模型（缓存避免重复加载）"""
    init_database()
    detector = PoseDetector()
    detector.load_model()
    return detector


# ── 侧边栏导航 ─────────────────────────────────────────────
st.sidebar.title("🏥 康复训练姿态评估系统")
st.sidebar.markdown("基于 YOLO + AI 大模型")

page = st.sidebar.radio(
    "导航菜单",
    ["📹 标准动作录制", "🏃 患者训练评估", "📊 康复日志", "⚙️ 系统设置"],
)

patient_id = st.sidebar.text_input("患者标识", value="patient_001")

# 摄像头选择
camera_index = st.sidebar.selectbox("摄像头", [0, 1, 2, 3], index=0,
    format_func=lambda x: f"Camera {x}" + (" (默认)" if x == 0 else " (USB)"))

# ── 加载全局资源 ──────────────────────────────────────────
detector = init_system()


# ═══════════════════════════════════════════════════════════
# 页面一：标准动作录制
# ═══════════════════════════════════════════════════════════
if page == "📹 标准动作录制":
    st.title("📹 标准动作录制")
    st.markdown("康复医师在此录制标准示范动作，形成评估基准模板。")

    col1, col2 = st.columns([2, 1])

    with col2:
        action_name = st.text_input("动作名称", placeholder="例如：站立抬右腿")
        action_desc = st.text_area("动作描述", placeholder="描述该动作的要领和评估重点")
        record_duration = st.slider("录制时长（秒）", 3, 15, RECORD_DURATION_SEC)

        if st.button("🎬 开始录制", type="primary", use_container_width=True):
            st.session_state.recording = True
            st.session_state.record_frames = []
            st.session_state.record_started = False

        if st.button("⏹️ 停止并保存", type="secondary", use_container_width=True):
            if st.session_state.get("record_frames"):
                action = StandardAction(
                    action_id=f"act_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    action_name=action_name,
                    description=action_desc,
                    frames=st.session_state.record_frames,
                    angle_sequences={},
                    created_at=datetime.now().isoformat(),
                )
                save_standard_action(action)
                st.success(f"✅ 标准动作「{action_name}」已保存！共 {len(action.frames)} 帧")
                st.session_state.recording = False
            else:
                st.warning("没有录制数据，请先开始录制。")

        # 已保存的标准动作列表
        st.markdown("---")
        st.subheader("📋 已保存的标准动作")
        actions = list_standard_actions()
        if actions:
            selected_action = st.selectbox(
                "选择动作模板",
                [a["action_id"] for a in actions],
                format_func=lambda x: f"{dict(actions)[x]['action_name']}" if False else next(
                    (a["action_name"] for a in actions if a["action_id"] == x), x
                ),
            )
        else:
            st.info("暂未保存任何标准动作")

    with col1:
        cap_placeholder = st.empty()

        if st.session_state.get("recording"):
            cap = cv2.VideoCapture(camera_index)
            frame_placeholder = st.empty()
            status_placeholder = st.empty()

            # 倒计时
            if not st.session_state.get("record_started"):
                for countdown in range(RECORD_COUNTDOWN_SEC, 0, -1):
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.putText(
                            frame, f"Starting... {countdown}",
                            (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3,
                        )
                        frame_placeholder.image(frame, channels="BGR", use_container_width=True)
                    time.sleep(1)
                st.session_state.record_started = True

            # 录制（多线程：一个线程采集+推理，主线程显示）
            frame_queue = Queue(maxsize=2)
            stop_event = threading.Event()

            def capture_and_detect():
                """后台线程：采集视频帧 + YOLO 推理"""
                f_count = 0
                s_time = time.time()
                while time.time() - s_time < record_duration and not stop_event.is_set():
                    ret, frm = cap.read()
                    if not ret:
                        break
                    ts = int((time.time() - s_time) * 1000)
                    pose_frame, kpts = detector.extract_pose_frame(frm, f_count, ts)
                    frame_queue.put((frm, pose_frame, kpts, f_count))
                    f_count += 1
                frame_queue.put(None)  # 结束信号

            capture_thread = threading.Thread(target=capture_and_detect, daemon=True)
            capture_thread.start()

            start_time = time.time()
            frame_count = 0
            while True:
                item = frame_queue.get()
                if item is None:
                    break
                frame, pose_frame, kpts, frame_count = item

                if pose_frame:
                    st.session_state.record_frames.append(pose_frame)
                    if kpts is not None:
                        frame = draw_pose_overlay(frame, kpts)
                        cv2.putText(frame, f"Recording: {frame_count} frames",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                frame_placeholder.image(frame, channels="BGR", use_container_width=True)
                status_placeholder.progress(
                    min((time.time() - start_time) / record_duration, 1.0),
                    f"录制中... {frame_count} 帧"
                )

            stop_event.set()
            capture_thread.join(timeout=2)

            cap.release()
            status_placeholder.success(f"录制完成！共 {len(st.session_state.record_frames)} 帧有效数据")

            # 自动保存录制数据
            if st.session_state.get("record_frames") and action_name:
                action = StandardAction(
                    action_id=f"act_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    action_name=action_name,
                    description=action_desc,
                    frames=st.session_state.record_frames,
                    angle_sequences={},
                    created_at=datetime.now().isoformat(),
                )
                save_standard_action(action)
                st.success(f"✅ 标准动作「{action_name}」已自动保存！")
                st.session_state.recording = False
                st.rerun()
        else:
            cap_placeholder.image(draw_welcome_screen(), channels="BGR", use_container_width=True)


# ═══════════════════════════════════════════════════════════
# 页面二：患者训练评估
# ═══════════════════════════════════════════════════════════
elif page == "🏃 患者训练评估":
    st.title("🏃 患者训练评估")
    st.markdown("患者模仿标准动作，系统实时对比并生成 AI 评估报告。")

    # 选择标准动作模板
    actions = list_standard_actions()
    action_options = {a["action_id"]: a["action_name"] for a in actions}

    col1, col2 = st.columns([2, 1])

    with col2:
        if action_options:
            selected_action_id = st.selectbox(
                "选择标准动作模板",
                list(action_options.keys()),
                format_func=lambda x: action_options[x],
            )
        else:
            st.warning("请先录制标准动作模板")
            selected_action_id = None

        train_duration = st.slider("训练评估时长（秒）", 3, 15, RECORD_DURATION_SEC)

        if selected_action_id and st.button("▶️ 开始评估", type="primary", use_container_width=True):
            st.session_state.evaluating = True
            st.session_state.eval_started = False
            st.session_state.patient_frames = []

    with col1:
        if st.session_state.get("evaluating") and selected_action_id:
            # 加载标准动作
            std_action = load_standard_action(selected_action_id)
            if std_action is None:
                st.error("无法加载标准动作模板")
            else:
                st.info(f"标准动作：{std_action.action_name} | 共 {len(std_action.frames)} 帧")

                cap = cv2.VideoCapture(camera_index)
                frame_placeholder = st.empty()
                status_placeholder = st.empty()
                progress_placeholder = st.empty()

                # 倒计时
                if not st.session_state.get("eval_started"):
                    for countdown in range(RECORD_COUNTDOWN_SEC, 0, -1):
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.putText(
                                frame, f"Evaluating... {countdown}",
                                (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3,
                            )
                            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
                        time.sleep(1)
                    st.session_state.eval_started = True

                # 评估录制（多线程：后台推理，主线程显示）
                frame_queue = Queue(maxsize=2)
                stop_event = threading.Event()

                def eval_capture_and_detect():
                    """后台线程：采集 + YOLO 推理"""
                    f_count = 0
                    s_time = time.time()
                    while time.time() - s_time < train_duration and not stop_event.is_set():
                        ret, frm = cap.read()
                        if not ret:
                            break
                        ts = int((time.time() - s_time) * 1000)
                        pose_frame, kpts = detector.extract_pose_frame(frm, f_count, ts)
                        frame_queue.put((frm, pose_frame, kpts, f_count))
                        f_count += 1
                    frame_queue.put(None)

                eval_thread = threading.Thread(target=eval_capture_and_detect, daemon=True)
                eval_thread.start()

                start_time = time.time()
                frame_count = 0
                while True:
                    item = frame_queue.get()
                    if item is None:
                        break
                    frame, pose_frame, kpts, frame_count = item

                    if pose_frame:
                        st.session_state.patient_frames.append(pose_frame)
                        # 可视化 - 叠加标准动作关键点（使用缓存的 kpts，不再重复推理）
                        if kpts is not None and std_action.frames:
                            progress = min(frame_count / max(len(std_action.frames), 1), 1.0)
                            ref_idx = min(int(progress * len(std_action.frames)), len(std_action.frames) - 1)
                            ref_frame = std_action.frames[ref_idx]
                            ref_kpts = np.zeros((17, 3), dtype=np.float32)
                            for i, name in enumerate(KEYPOINT_NAMES):
                                pt = ref_frame.get_point(name)
                                if pt:
                                    ref_kpts[i] = [pt[0], pt[1], 1.0]
                            frame = draw_pose_overlay(frame, kpts, ref_kpts, draw_angles=True)

                    progress_placeholder.progress(
                        min((time.time() - start_time) / train_duration, 1.0),
                        f"评估中... {frame_count} 帧 | 患者 {patient_id}"
                    )
                    frame_placeholder.image(frame, channels="BGR", use_container_width=True)

                stop_event.set()
                eval_thread.join(timeout=2)
                cap.release()
                st.session_state.evaluating = False

                # ── 执行姿态对比 ──
                if st.session_state.patient_frames:
                    status_placeholder.info("正在进行姿态对比分析...")
                    result = compare_poses(std_action.frames, st.session_state.patient_frames)

                    if result:
                        result.action_name = std_action.action_name
                        status_placeholder.success(
                            f"姿态对比完成！相似度评分：{result.similarity_score}/100"
                        )

                        # ── AI 评估 ──
                        report = None
                        try:
                            with st.spinner("AI is generating rehabilitation report..."):
                                evaluator = AIEvaluator()
                                report = evaluator.evaluate(result, std_action.action_name)
                        except Exception as e:
                            st.warning(f"AI evaluation failed: {e}")
                            report = None

                        if report is None:
                            # Fallback: generate basic report without AI
                            score = result.similarity_score
                            grade = "Excellent" if score >= 90 else "Good" if score >= 75 else "Fair" if score >= 60 else "Poor"
                            report = {
                                "overall_score": score,
                                "grade": grade,
                                "joint_analysis": [
                                    {"joint": j, "deviation": d, "assessment": "Auto-evaluated", "confidence": "Medium"}
                                    for j, d in result.angle_deviations.items()
                                ],
                                "problems": [],
                                "suggestions": ["AI service unavailable - this is a basic auto-evaluation"],
                                "risk_warnings": [],
                                "rag_sources": [],
                                "note": "Offline auto-evaluation (AI model not connected)",
                            }

                        # 保存康复日志
                        log_id = save_rehab_log(
                            patient_id=patient_id,
                            action_name=std_action.action_name,
                            similarity_score=result.similarity_score,
                            angle_deviations=result.angle_deviations,
                            avg_keypoint_offset=result.avg_keypoint_offset,
                            rom_comparison=result.rom_comparison,
                            ai_report=json.dumps(report, ensure_ascii=False),
                            rag_sources=report.get("rag_sources", []),
                            duration_s=train_duration,
                        )

                        # ── 展示评估结果 ──
                        st.markdown("---")
                        st.subheader("📊 评估结果")

                        # 评分仪表盘
                        metric_cols = st.columns(4)
                        with metric_cols[0]:
                            st.metric("相似度评分", f"{result.similarity_score:.1f}/100")
                        with metric_cols[1]:
                            grade = report.get("grade", "N/A")
                            st.metric("评估等级", grade)
                        with metric_cols[2]:
                            st.metric("有效帧数", len(st.session_state.patient_frames))
                        with metric_cols[3]:
                            st.metric("日志ID", log_id)

                        # 关节偏差表格
                        st.markdown("### 关节角度偏差")
                        angle_df = pd.DataFrame([
                            {"关节": k.replace("left_", "左").replace("right_", "右")
                                        .replace("_", " "), "偏差(°)": f"{v:.1f}"}
                            for k, v in result.angle_deviations.items()
                        ])
                        st.dataframe(angle_df, use_container_width=True, hide_index=True)

                        # AI 评估报告
                        st.markdown("### 🤖 AI 康复评估报告")
                        report_md = format_report_for_display(report)
                        st.markdown(report_md)

                        st.success(f"✅ 评估完成，日志已保存（ID: {log_id}）")
                    else:
                        st.error("姿态对比失败，请重试。")
                else:
                    st.warning("未检测到有效姿态数据，请确保身体完整出现在摄像头中。")


# ═══════════════════════════════════════════════════════════
# 页面三：康复日志
# ═══════════════════════════════════════════════════════════
elif page == "📊 康复日志":
    st.title("📊 康复日志")
    st.markdown(f"患者 **{patient_id}** 的历史康复记录与趋势分析。")

    tab1, tab2, tab3 = st.tabs(["📈 评分趋势", "📋 历史记录", "📝 报告详情"])

    with tab1:
        st.plotly_chart(
            generate_score_trend_chart(patient_id),
            use_container_width=True,
        )

    with tab2:
        logs = get_rehab_logs(patient_id, limit=50)
        if logs:
            logs_df = pd.DataFrame([{
                "日志ID": log["id"],
                "动作": log["action_name"],
                "评分": log["similarity_score"],
                "时间": log["created_at"],
                "耗时(s)": log["duration_s"],
            } for log in logs])
            st.dataframe(logs_df, use_container_width=True, hide_index=True)
        else:
            st.info("暂无训练记录")

    with tab3:
        log_id_input = st.number_input("输入日志ID查看详情", min_value=1, value=1, step=1)
        if st.button("🔍 查看报告"):
            log = get_rehab_log(log_id_input)
            if log:
                st.markdown(f"### 评估报告 — {log['action_name']}")
                st.markdown(f"- 评分：{log['similarity_score']}/100")
                st.markdown(f"- 时间：{log['created_at']}")
                st.markdown(f"- 耗时：{log['duration_s']}秒")
                try:
                    report = json.loads(log["ai_report"])
                    st.markdown(format_report_for_display(report))
                except (json.JSONDecodeError, TypeError):
                    st.text(log.get("ai_report", "无报告内容"))
            else:
                st.warning(f"未找到日志 ID={log_id_input}")


# ═══════════════════════════════════════════════════════════
# 页面四：系统设置
# ═══════════════════════════════════════════════════════════
elif page == "⚙️ 系统设置":
    st.title("⚙️ 系统设置")

    st.subheader("姿态对比权重调整")
    st.markdown("调整多维度评分权重，影响相似度计算。")

    w_angle = st.slider("关节角度偏差权重", 0.0, 1.0, COMPARISON_WEIGHTS["angle_deviation"], 0.05)
    w_kp = st.slider("关键点空间偏移权重", 0.0, 1.0, COMPARISON_WEIGHTS["keypoint_offset"], 0.05)
    w_rom = st.slider("动作幅度差异权重", 0.0, 1.0, COMPARISON_WEIGHTS["rom_difference"], 0.05)

    # 归一化确保总和为 1
    total = w_angle + w_kp + w_rom
    if total > 0:
        st.info(f"实际权重：角度={w_angle/total:.2f}，关键点={w_kp/total:.2f}，幅度={w_rom/total:.2f}")

    st.markdown("---")
    st.subheader("🎯 关节活动度参考值 (ROM)")
    rom_data = pd.DataFrame([
        {"关节": "肘关节", "正常范围": "0°-145°"},
        {"关节": "膝关节", "正常范围": "0°-135°"},
        {"关节": "肩关节", "正常范围": "0°-180°"},
        {"关节": "髋关节", "正常范围": "0°-120°"},
    ])
    st.table(rom_data)

    st.markdown("---")
    st.subheader("📋 标准动作管理")
    actions = list_standard_actions()
    if actions:
        for action in actions:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"**{action['action_name']}** ({action['action_id']})")
                st.caption(f"创建于 {action['created_at']}")
            with col_b:
                if st.button("🗑️ 删除", key=f"del_{action['action_id']}"):
                    delete_standard_action(action["action_id"])
                    st.rerun()
    else:
        st.info("暂无标准动作模板")


# ── 页脚 ──────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>宁波大学计算机系\n\n"
    "《计算机系统综合实习》课程设计\n\n"
    "基于 YOLO26-Pose + LLM + RAG\n\n"
    "© 2026 王兴伽</small>",
    unsafe_allow_html=True,
)
