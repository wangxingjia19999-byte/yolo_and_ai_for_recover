"""FastAPI 后端服务 —— 康复训练姿态评估系统 API"""

import sys
import json
import io
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from rehab_pose.config import RECORD_DURATION_SEC, RECORD_FPS
from rehab_pose.keypoints import (
    PoseFrame, StandardAction, Keypoint2D, EvaluationResult,
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

# ── 应用初始化 ─────────────────────────────────────────────
app = FastAPI(
    title="康复训练姿态评估系统 API",
    description="基于 YOLO 姿态估计与 AI 大模型的康复训练姿态评估系统",
    version="1.0.0",
)

# 全局实例
detector: Optional[PoseDetector] = None
evaluator: Optional[AIEvaluator] = None


@app.on_event("startup")
async def startup():
    global detector, evaluator
    init_database()
    detector = PoseDetector()
    detector.load_model()
    evaluator = AIEvaluator()
    print("🚀 API 服务已启动")


# ── 请求/响应模型 ──────────────────────────────────────────

class ActionCreate(BaseModel):
    action_name: str
    description: str = ""


class EvaluationRequest(BaseModel):
    patient_id: str
    action_id: str


class AngleData(BaseModel):
    joint: str
    value: Optional[float]


# ── 标准动作管理 API ───────────────────────────────────────

@app.get("/actions")
async def api_list_actions():
    """获取所有标准动作模板列表"""
    actions = list_standard_actions()
    return {"actions": actions, "count": len(actions)}


@app.get("/actions/{action_id}")
async def api_get_action(action_id: str):
    """获取指定标准动作模板详情"""
    action = load_standard_action(action_id)
    if action is None:
        raise HTTPException(status_code=404, detail="动作模板不存在")
    return {
        "action_id": action.action_id,
        "action_name": action.action_name,
        "description": action.description,
        "frame_count": len(action.frames),
        "duration_ms": action.frames[-1].timestamp_ms if action.frames else 0,
        "created_at": action.created_at,
    }


@app.delete("/actions/{action_id}")
async def api_delete_action(action_id: str):
    """删除标准动作模板"""
    if delete_standard_action(action_id):
        return {"message": f"动作模板 {action_id} 已删除"}
    raise HTTPException(status_code=404, detail="动作模板不存在")


# ── 姿态检测 API ───────────────────────────────────────────

@app.post("/pose/detect")
async def api_detect_pose(file: UploadFile = File(...)):
    """
    上传单张图片，返回人体17个关键点坐标。
    用于单帧姿态检测测试。
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="检测器未就绪")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="无法解码图片")

    kpts = detector.extract_keypoints(frame)

    if kpts is None:
        return {"detected": False, "message": "未检测到人体"}

    keypoints_detail = []
    for i, name in enumerate([
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]):
        keypoints_detail.append({
            "name": name,
            "x": round(float(kpts[i, 0]), 4),
            "y": round(float(kpts[i, 1]), 4),
            "confidence": round(float(kpts[i, 2]), 4),
        })

    # 计算关节角度
    kp_dict = {kp["name"]: (kp["x"], kp["y"]) for kp in keypoints_detail}
    angles = compute_all_joint_angles(kp_dict)

    return {
        "detected": True,
        "image_size": {"width": frame.shape[1], "height": frame.shape[0]},
        "keypoints": keypoints_detail,
        "joint_angles": {
            k: round(v, 2) if v is not None else None
            for k, v in angles.items()
        },
    }


# ── 姿态对比 API ───────────────────────────────────────────

@app.post("/evaluate")
async def api_evaluate(req: EvaluationRequest):
    """
    对患者动作与标准动作进行对比评估。
    需配合视频采集使用，此接口接收已录制的患者动作帧数据。
    """
    # 加载标准动作
    std_action = load_standard_action(req.action_id)
    if std_action is None:
        raise HTTPException(status_code=404, detail="标准动作模板不存在")

    raise HTTPException(
        status_code=501,
        detail="此接口需要通过 WebSocket 或流式 API 传输患者帧数据。"
               "请使用 Streamlit 前端进行完整评估，或通过 /evaluate/batch 接口提交批量帧数据。",
    )


# ── 康复日志 API ───────────────────────────────────────────

@app.get("/logs/{patient_id}")
async def api_get_logs(
    patient_id: str,
    action_name: str = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    """查询康复日志"""
    logs = get_rehab_logs(patient_id, action_name, limit)
    return {"logs": logs, "count": len(logs)}


@app.get("/logs/{patient_id}/trend")
async def api_get_trend(
    patient_id: str,
    action_name: str = Query(None),
):
    """获取评分趋势数据"""
    trend = get_score_trend(patient_id, action_name)
    return {"trend": trend, "count": len(trend)}


@app.get("/log/{log_id}")
async def api_get_log_detail(log_id: int):
    """获取单条康复日志详情"""
    log = get_rehab_log(log_id)
    if log is None:
        raise HTTPException(status_code=404, detail="日志不存在")

    # 解析 JSON 字段
    for field in ["angle_deviations", "rom_comparison", "rag_sources"]:
        if isinstance(log.get(field), str):
            try:
                log[field] = json.loads(log[field])
            except json.JSONDecodeError:
                pass

    return log


# ── 健康检查 ───────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "detector_ready": detector is not None and detector.is_ready,
        "evaluator_ready": evaluator is not None,
        "version": "1.0.0",
    }


# ── 启动入口 ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    from rehab_pose.config import API_HOST, API_PORT

    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=True)
