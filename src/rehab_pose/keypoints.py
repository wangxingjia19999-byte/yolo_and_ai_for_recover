"""关键点数据结构与关节角度计算模块"""

import numpy as np
from typing import Optional
from pydantic import BaseModel, Field

from .config import KEYPOINT_NAMES, JOINT_ANGLE_DEFS


# ── 关键点数据模型 ─────────────────────────────────────────

class Keypoint2D(BaseModel):
    """单个关键点 (x, y, confidence)"""
    x: float = Field(..., ge=0.0, le=1.0, description="归一化x坐标 [0,1]")
    y: float = Field(..., ge=0.0, le=1.0, description="归一化y坐标 [0,1]")
    conf: float = Field(..., ge=0.0, le=1.0, description="检测置信度")


class PoseFrame(BaseModel):
    """单帧姿态数据"""
    frame_id: int
    timestamp_ms: int
    keypoints: dict[str, Keypoint2D]

    def get_point(self, name: str) -> Optional[tuple[float, float]]:
        """获取关键点坐标"""
        kp = self.keypoints.get(name)
        if kp is None or kp.conf <= 0:
            return None
        return (kp.x, kp.y)


class StandardAction(BaseModel):
    """标准动作模板"""
    action_id: str
    action_name: str
    description: str = ""
    frames: list[PoseFrame] = []
    angle_sequences: dict[str, list[float]] = Field(default_factory=dict)
    created_at: str = ""


class EvaluationResult(BaseModel):
    """姿态评估结果"""
    action_name: str
    angle_deviations: dict[str, float]     # 各关节角度偏差
    avg_keypoint_offset: float             # 关键点平均偏移量
    rom_comparison: dict[str, dict]        # 动作幅度对比
    similarity_score: float                # 整体相似度评分 (0-100)


# ── 关节角度计算 ───────────────────────────────────────────

def calculate_angle(a: tuple[float, float],
                    b: tuple[float, float],
                    c: tuple[float, float]) -> float:
    """
    计算三点形成的角度（以B为顶点）。
    公式：θ = arccos(BA·BC / (|BA| * |BC|))

    Args:
        a: 第一个关键点坐标 (x, y)
        b: 顶点坐标 (x, y)
        c: 第三个关键点坐标 (x, y)

    Returns:
        角度值（度），范围 [0, 180]
    """
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])

    dot_product = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cos_theta = np.clip(dot_product / (norm_ba * norm_bc), -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    return float(np.degrees(theta_rad))


def compute_all_joint_angles(keypoints: dict[str, tuple[float, float]]) -> dict[str, float]:
    """
    计算所有关节角度。

    Args:
        keypoints: 关键点字典，{名称: (x, y)}

    Returns:
        关节角度字典，{关节名: 角度(度)}
    """
    angles = {}
    for joint_name, (kp_a, kp_b, kp_c) in JOINT_ANGLE_DEFS.items():
        pt_a = keypoints.get(kp_a)
        pt_b = keypoints.get(kp_b)
        pt_c = keypoints.get(kp_c)

        if pt_a is None or pt_b is None or pt_c is None:
            angles[joint_name] = None
        else:
            angles[joint_name] = calculate_angle(pt_a, pt_b, pt_c)

    return angles


def compute_torso_tilt_angle(keypoints: dict[str, tuple[float, float]]) -> Optional[float]:
    """
    计算躯干倾斜角。
    躯干倾斜角 = 双肩中点与双髋中点连线与垂线之间的夹角。

    Args:
        keypoints: 关键点字典

    Returns:
        躯干倾斜角（度），左侧正值
    """
    ls = keypoints.get("left_shoulder")
    rs = keypoints.get("right_shoulder")
    lh = keypoints.get("left_hip")
    rh = keypoints.get("right_hip")

    if not all([ls, rs, lh, rh]):
        return None

    shoulder_mid = np.array([(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2])
    hip_mid = np.array([(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2])
    torso_vec = hip_mid - shoulder_mid  # 从上到下
    vertical = np.array([0, 1])  # 垂线方向（向下为正）

    dot = np.dot(torso_vec, vertical)
    norm_torso = np.linalg.norm(torso_vec)
    if norm_torso == 0:
        return 0.0

    cos_a = np.clip(dot / norm_torso, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def keypoints_to_dict(pose_frame: PoseFrame) -> dict[str, tuple[float, float]]:
    """将 PoseFrame 中的关键点转换为简单坐标字典"""
    result = {}
    for name in KEYPOINT_NAMES:
        pt = pose_frame.get_point(name)
        if pt is not None:
            result[name] = pt
    return result


def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    自适应归一化策略：
    1. 以双髋中点为原点平移
    2. 以肩髋距离为参考尺度缩放

    Args:
        keypoints: shape [17, 3] — (x, y, conf)

    Returns:
        归一化后的关键点
    """
    kpts = keypoints.copy()

    # 获取左右髋索引 (11, 12)
    lh_idx, rh_idx = 11, 12
    ls_idx, rs_idx = 5, 6

    # 双髋中点为原点
    hip_center_x = (kpts[lh_idx, 0] + kpts[rh_idx, 0]) / 2
    hip_center_y = (kpts[lh_idx, 1] + kpts[rh_idx, 1]) / 2

    if hip_center_x == 0 or np.isnan(hip_center_x):
        return kpts

    # 平移归一化
    kpts[:, 0] -= hip_center_x
    kpts[:, 1] -= hip_center_y

    # 缩放归一化（以肩髋距离为参考尺度）
    shoulder_mid_x = (kpts[ls_idx, 0] + kpts[rs_idx, 0]) / 2
    shoulder_mid_y = (kpts[ls_idx, 1] + kpts[rs_idx, 1]) / 2
    hip_mid = np.array([0.0, 0.0])  # 已平移到原点
    shoulder_mid = np.array([shoulder_mid_x, shoulder_mid_y])
    torso_length = np.linalg.norm(shoulder_mid - hip_mid)

    if torso_length > 0.01:
        kpts[:, :2] /= torso_length
        # 恢复髋中点位置
        kpts[:, 0] -= hip_mid[0] / torso_length
        kpts[:, 1] -= hip_mid[1] / torso_length

    return kpts
