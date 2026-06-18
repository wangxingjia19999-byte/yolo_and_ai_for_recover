"""YOLO 姿态检测模块"""

import numpy as np
import cv2
from typing import Optional
from ultralytics import YOLO

from .config import (
    YOLO_MODEL_NAME,
    YOLO_INPUT_SIZE,
    YOLO_CONF_THRESHOLD,
    KEYPOINT_NAMES,
)
from .keypoints import PoseFrame, Keypoint2D, normalize_keypoints


class PoseDetector:
    """
    YOLO姿态检测器。
    基于YOLO-Pose模型从视频帧中实时提取人体17个关键点坐标。
    """

    def __init__(self, model_name: str = YOLO_MODEL_NAME):
        self.model_name = model_name
        self.model: Optional[YOLO] = None
        self._initialized = False

    def load_model(self) -> None:
        """加载YOLO-Pose模型并执行预热推理"""
        print(f"[PoseDetector] 正在加载模型: {self.model_name}")
        self.model = YOLO(self.model_name)

        # 预热推理（消除首次推理冷启动延迟）
        dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy_input, verbose=False)
        self._initialized = True
        print("[PoseDetector] 模型加载完成，预热推理已执行")

    def extract_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        从视频帧中提取人体关键点。

        Args:
            frame: BGR视频帧 (H, W, 3)

        Returns:
            关键点数组 [17, 3] (x_normalized, y_normalized, confidence)
            如果未检测到人体，返回 None
        """
        if not self._initialized:
            self.load_model()

        results = self.model(frame, verbose=False)[0]

        if results.keypoints is None or results.keypoints.data.shape[0] == 0:
            return None

        # 获取关键点数据 shape: [N, 17, 3]
        kpts = results.keypoints.data.cpu().numpy()

        # 如果检测到多人，取置信度最高的人体
        if len(kpts) > 1:
            confs = kpts[:, :, 2].mean(axis=1)
            best_idx = int(confs.argmax())
            kpts = kpts[best_idx:best_idx + 1]

        kpts = kpts[0]  # [17, 3]

        # 过滤低置信度关键点
        low_conf_mask = kpts[:, 2] < YOLO_CONF_THRESHOLD
        if low_conf_mask.all():
            return None

        # 归一化坐标
        h, w = frame.shape[:2]
        kpts[:, 0] /= w  # x 归一化
        kpts[:, 1] /= h  # y 归一化

        return kpts

    def extract_pose_frame(self, frame: np.ndarray,
                           frame_id: int,
                           timestamp_ms: int) -> tuple[Optional[PoseFrame], Optional[np.ndarray]]:
        """
        从视频帧提取并构造 PoseFrame，同时返回原始关键点用于可视化。

        Args:
            frame: BGR视频帧
            frame_id: 帧序号
            timestamp_ms: 时间戳（毫秒）

        Returns:
            (PoseFrame, keypoints_array) 元组，如果未检测到人体返回 (None, None)
        """
        kpts = self.extract_keypoints(frame)
        if kpts is None:
            return None, None

        keypoints_dict = {}
        for i, name in enumerate(KEYPOINT_NAMES):
            keypoints_dict[name] = Keypoint2D(
                x=float(kpts[i, 0]),
                y=float(kpts[i, 1]),
                conf=float(kpts[i, 2]),
            )

        pose_frame = PoseFrame(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            keypoints=keypoints_dict,
        )
        return pose_frame, kpts

    def get_bounding_box(self, kpts: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        """
        从关键点计算人体边界框（像素坐标）。
        返回 (x1, y1, x2, y2)，即 (左上x, 左上y, 右下x, 右下y)。
        """
        valid = kpts[kpts[:, 2] > YOLO_CONF_THRESHOLD]
        if len(valid) == 0:
            return None

        x_min, y_min = valid[:, 0].min(), valid[:, 1].min()
        x_max, y_max = valid[:, 0].max(), valid[:, 1].max()
        return (int(x_min), int(y_min), int(x_max), int(y_max))

    @property
    def is_ready(self) -> bool:
        return self._initialized and self.model is not None
