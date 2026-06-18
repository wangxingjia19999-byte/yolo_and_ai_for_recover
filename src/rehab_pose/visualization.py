"""姿态可视化模块 —— 骨骼叠加、关键点标记、角度标注"""

import cv2
import numpy as np
from typing import Optional

from .config import (
    KEYPOINT_NAMES,
    SKELETON_CONNECTIONS,
    SKELETON_COLORS,
    JOINT_ANGLE_DEFS,
)
from .keypoints import compute_all_joint_angles


def _get_side(name: str) -> str:
    """判断关键点的左右侧"""
    if name.startswith("left_"):
        return "left"
    elif name.startswith("right_"):
        return "right"
    return "center"


def draw_pose_overlay(
    frame: np.ndarray,
    keypoints: np.ndarray,  # [17, 3]
    reference_keypoints: Optional[np.ndarray] = None,  # 标准动作关键点（半透明叠加）
    draw_angles: bool = True,
    draw_labels: bool = True,
) -> np.ndarray:
    """
    在视频帧上叠加姿态可视化图层。

    Args:
        frame: BGR 视频帧
        keypoints: 当前帧关键点 [17, 3] (x_像素, y_像素, conf)
        reference_keypoints: 参考关键点（标准模板）
        draw_angles: 是否绘制角度标注
        draw_labels: 是否绘制关键点标签

    Returns:
        叠加后的视频帧
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ── 可选：叠加参考关键点（半透明） ──
    if reference_keypoints is not None:
        ref_overlay = frame.copy()
        ref_kpts_denorm = _denormalize_to_pixel(reference_keypoints, h, w)
        _draw_skeleton(ref_overlay, ref_kpts_denorm, alpha=0.35)
        overlay = cv2.addWeighted(overlay, 0.7, ref_overlay, 0.3, 0)

    # ── 绘制当前关键点的骨架 ──
    kpts_pixel = _denormalize_to_pixel(keypoints, h, w)
    _draw_skeleton(overlay, kpts_pixel)
    _draw_keypoints(overlay, kpts_pixel, draw_labels)

    # ── 绘制关节角度标注 ──
    if draw_angles:
        kpts_denorm = {name: (float(keypoints[i, 0]) * w, float(keypoints[i, 1]) * h)
                       for i, name in enumerate(KEYPOINT_NAMES)}
        kpts_norm = {name: (float(keypoints[i, 0]), float(keypoints[i, 1]))
                     for i, name in enumerate(KEYPOINT_NAMES)
                     if keypoints[i, 2] > 0.5}
        angles = compute_all_joint_angles(kpts_norm)
        _draw_angle_annotations(overlay, angles, kpts_denorm)

    return overlay


def _denormalize_to_pixel(kpts: np.ndarray, h: int, w: int) -> np.ndarray:
    """将归一化关键点转换为像素坐标"""
    arr = kpts.copy()
    arr[:, 0] *= w
    arr[:, 1] *= h
    return arr


def _draw_skeleton(frame: np.ndarray, kpts_pixel: np.ndarray, alpha: float = 1.0):
    """绘制人体骨架连线"""
    for conn in SKELETON_CONNECTIONS:
        i1 = KEYPOINT_NAMES.index(conn[0])
        i2 = KEYPOINT_NAMES.index(conn[1])

        pt1 = kpts_pixel[i1]
        pt2 = kpts_pixel[i2]

        if pt1[2] < 0.5 or pt2[2] < 0.5:
            continue

        side = _get_side(conn[0])
        color = SKELETON_COLORS.get(side, (0, 255, 255))

        if alpha < 1.0:
            color = tuple(int(c * alpha) for c in color)

        cv2.line(
            frame,
            (int(pt1[0]), int(pt1[1])),
            (int(pt2[0]), int(pt2[1])),
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_keypoints(frame: np.ndarray, kpts_pixel: np.ndarray, draw_labels: bool):
    """绘制关键点标记"""
    for i, name in enumerate(KEYPOINT_NAMES):
        x, y, conf = kpts_pixel[i]

        if conf < 0.5:
            continue

        # 颜色根据置信度从绿（高）渐变到红（低）
        green = int(255 * conf)
        red = int(255 * (1 - conf))
        color = (0, green, red)

        # 头部关键点用不同颜色
        if name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
            color = (255, 255, 0)

        cv2.circle(frame, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 255), 1, cv2.LINE_AA)

        if draw_labels and name in ("left_shoulder", "right_shoulder",
                                     "left_hip", "right_hip",
                                     "left_knee", "right_knee"):
            cv2.putText(frame, name.replace("_", " "),
                        (int(x) + 8, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)


def _draw_angle_annotations(frame: np.ndarray,
                            angles: dict[str, Optional[float]],
                            kpts_pixel: dict[str, tuple[float, float]]):
    """绘制关节角度标注"""
    for joint_name, (kp_a, kp_b, kp_c) in JOINT_ANGLE_DEFS.items():
        angle = angles.get(joint_name)
        if angle is None:
            continue

        pt_b = kpts_pixel.get(kp_b)
        if pt_b is None:
            continue

        text = f"{angle:.0f}°"
        side = _get_side(joint_name)
        color = SKELETON_COLORS.get(side, (255, 255, 255))

        cv2.putText(
            frame, text,
            (int(pt_b[0]) + 10, int(pt_b[1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA,
        )


def draw_pose_comparison(
    frame: np.ndarray,
    patient_kpts: np.ndarray,  # [17, 3]
    standard_kpts: np.ndarray,  # [17, 3]
) -> np.ndarray:
    """
    并排显示患者动作与标准动作的对比视图。
    适用于 Streamlit 展示。

    Returns:
        拼接后的对比图像
    """
    h, w = frame.shape[:2]

    # 患者侧
    patient_view = frame.copy()
    patient_view = draw_pose_overlay(patient_view, patient_kpts)

    # 标准侧（灰底 + 标准关键点叠加）
    standard_view = np.full((h, w, 3), 50, dtype=np.uint8)
    standard_view = draw_pose_overlay(standard_view, standard_kpts,
                                       draw_angles=True, draw_labels=True)
    cv2.putText(standard_view, "Standard Template", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(patient_view, "Patient (Live)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return np.hstack([patient_view, standard_view])


def draw_welcome_screen(size: tuple = (640, 480)) -> np.ndarray:
    """绘制欢迎/等待画面"""
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cv2.putText(img, "Rehabilitation Pose Assessment System",
                (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(img, "Press 'Start' to begin recording",
                (120, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img
