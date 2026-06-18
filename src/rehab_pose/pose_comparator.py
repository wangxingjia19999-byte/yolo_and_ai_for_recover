"""姿态对比模块 —— DTW帧对齐 + 多维度偏差计算"""

import numpy as np
from typing import Optional

from .config import JOINT_ANGLE_DEFS, COMPARISON_WEIGHTS, KEYPOINT_NAMES
from .keypoints import (
    PoseFrame, EvaluationResult, keypoints_to_dict,
    compute_all_joint_angles, calculate_angle,
)


def _frames_to_angle_sequence(frames: list[PoseFrame]) -> np.ndarray:
    """
    将姿态帧序列转换为关节角度序列矩阵。

    Args:
        frames: PoseFrame 列表

    Returns:
        角度序列矩阵 [N_frames, N_joints]
    """
    seq = []
    for frame in frames:
        kp_dict = keypoints_to_dict(frame)
        angles = compute_all_joint_angles(kp_dict)
        angle_list = [angles.get(j, 0.0) or 0.0 for j in JOINT_ANGLE_DEFS]
        seq.append(angle_list)
    return np.array(seq)


def _dtw_distance_matrix(seq_a: np.ndarray, seq_b: np.ndarray) -> np.ndarray:
    """
    计算两条序列的 DTW 累积距离矩阵。

    Args:
        seq_a: 序列 A [M, D]
        seq_b: 序列 B [N, D]

    Returns:
        DTW 累积距离矩阵 [M, N]
    """
    M, N = seq_a.shape[0], seq_b.shape[0]
    dtw = np.full((M, N), np.inf)

    # 计算逐对欧氏距离矩阵
    for i in range(M):
        for j in range(N):
            dist = np.sqrt(np.sum((seq_a[i] - seq_b[j]) ** 2))
            if i == 0 and j == 0:
                dtw[i, j] = dist
            elif i == 0:
                dtw[i, j] = dist + dtw[i, j - 1]
            elif j == 0:
                dtw[i, j] = dist + dtw[i - 1, j]
            else:
                dtw[i, j] = dist + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return dtw


def _dtw_backtrack(dtw: np.ndarray) -> list[tuple[int, int]]:
    """
    从 DTW 累积距离矩阵回溯最优对齐路径。

    Args:
        dtw: 累积距离矩阵 [M, N]

    Returns:
        对齐路径 [(i, j), ...]
    """
    M, N = dtw.shape
    i, j = M - 1, N - 1
    path = [(i, j)]

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = [
                (i - 1, j, dtw[i - 1, j]),
                (i, j - 1, dtw[i, j - 1]),
                (i - 1, j - 1, dtw[i - 1, j - 1]),
            ]
            i, j, _ = min(candidates, key=lambda x: x[2])

        path.append((i, j))

    path.reverse()
    return path


def dtw_align_frames(standard_angle_seq: np.ndarray,
                     patient_angle_seq: np.ndarray) -> list[tuple[int, int]]:
    """
    使用动态时间规整（DTW）对齐标准动作与患者动作的帧序列。

    算法复杂度 O(M*N)，其中 M、N 分别为两条序列的帧数。
    对于康复动作评估场景（通常 50-150 帧），此复杂度完全可以接受。

    Args:
        standard_angle_seq: 标准角度序列 [N_std, N_joints]
        patient_angle_seq: 患者角度序列 [N_pt, N_joints]

    Returns:
        帧对齐对列表 [(standard_frame_idx, patient_frame_idx), ...]
    """
    if standard_angle_seq.shape[0] == 0 or patient_angle_seq.shape[0] == 0:
        return []

    dtw_matrix = _dtw_distance_matrix(standard_angle_seq, patient_angle_seq)
    path = _dtw_backtrack(dtw_matrix)
    return path


def compute_angle_deviation(std_frames: list[PoseFrame],
                            pt_frames: list[PoseFrame],
                            alignment_path: list[tuple[int, int]]) -> dict[str, float]:
    """
    计算各关节角度偏差（维度一）。

    ∆θ_k = (1/N) * Σ|θ_pt(k) - θ_std(k)|

    Returns:
        {关节名: 平均角度偏差(度)}
    """
    if not alignment_path:
        return {}

    deviations = {joint: [] for joint in JOINT_ANGLE_DEFS}

    for std_idx, pt_idx in alignment_path:
        if std_idx >= len(std_frames) or pt_idx >= len(pt_frames):
            continue

        std_kp = keypoints_to_dict(std_frames[std_idx])
        pt_kp = keypoints_to_dict(pt_frames[pt_idx])

        std_angles = compute_all_joint_angles(std_kp)
        pt_angles = compute_all_joint_angles(pt_kp)

        for joint in JOINT_ANGLE_DEFS:
            sa = std_angles.get(joint)
            pa = pt_angles.get(joint)
            if sa is not None and pa is not None:
                deviations[joint].append(abs(pa - sa))

    return {
        joint: float(np.mean(vals)) if vals else 0.0
        for joint, vals in deviations.items()
    }


def compute_keypoint_offset(std_frames: list[PoseFrame],
                            pt_frames: list[PoseFrame],
                            alignment_path: list[tuple[int, int]]) -> dict[str, float]:
    """
    计算每个关键点的归一化欧氏距离偏差（维度二）。

    ∆d_kp = (1/N) * Σ√((x_pt - x_std)² + (y_pt - y_std)²)

    Returns:
        {关键点名: 平均偏移量}
    """
    if not alignment_path:
        return {}

    offsets = {name: [] for name in KEYPOINT_NAMES}

    for std_idx, pt_idx in alignment_path:
        if std_idx >= len(std_frames) or pt_idx >= len(pt_frames):
            continue

        for name in KEYPOINT_NAMES:
            sp = std_frames[std_idx].get_point(name)
            pp = pt_frames[pt_idx].get_point(name)
            if sp is not None and pp is not None:
                dist = np.sqrt((pp[0] - sp[0]) ** 2 + (pp[1] - sp[1]) ** 2)
                offsets[name].append(dist)

    return {
        name: float(np.mean(vals)) if vals else 0.0
        for name, vals in offsets.items()
    }


def compute_rom_difference(std_frames: list[PoseFrame],
                           pt_frames: list[PoseFrame],
                           alignment_path: list[tuple[int, int]]) -> dict[str, dict]:
    """
    计算动作幅度差异（维度三）。

    ∆ROM_k = |(max(θ_pt) - min(θ_pt)) - (max(θ_std) - min(θ_std))|

    Returns:
        {关节名: {"std_range": float, "patient_range": float, "difference": float}}
    """
    if not alignment_path:
        return {}

    # 提取标准动作的角度序列
    std_angles_seq = {j: [] for j in JOINT_ANGLE_DEFS}
    for std_idx, _ in alignment_path:
        if std_idx < len(std_frames):
            kp = keypoints_to_dict(std_frames[std_idx])
            angles = compute_all_joint_angles(kp)
            for j in JOINT_ANGLE_DEFS:
                v = angles.get(j)
                if v is not None:
                    std_angles_seq[j].append(v)

    # 提取患者动作的角度序列
    pt_angles_seq = {j: [] for j in JOINT_ANGLE_DEFS}
    for _, pt_idx in alignment_path:
        if pt_idx < len(pt_frames):
            kp = keypoints_to_dict(pt_frames[pt_idx])
            angles = compute_all_joint_angles(kp)
            for j in JOINT_ANGLE_DEFS:
                v = angles.get(j)
                if v is not None:
                    pt_angles_seq[j].append(v)

    result = {}
    for joint in JOINT_ANGLE_DEFS:
        sv = std_angles_seq.get(joint, [])
        pv = pt_angles_seq.get(joint, [])
        std_range = float(max(sv) - min(sv)) if len(sv) > 1 else 0.0
        pt_range = float(max(pv) - min(pv)) if len(pv) > 1 else 0.0
        result[joint] = {
            "std_range": round(std_range, 2),
            "patient_range": round(pt_range, 2),
            "difference": round(abs(pt_range - std_range), 2),
        }

    return result


def compute_similarity_score(angle_deviations: dict[str, float],
                             keypoint_offsets: dict[str, float],
                             rom_diff: dict[str, dict],
                             weights: dict = None) -> float:
    """
    综合计算整体相似度评分（维度四）。

    S_overall = 100 - (w1 * ∆θ_avg + w2 * ∆d_avg + w3 * ∆ROM_avg)

    Args:
        angle_deviations: 角度偏差字典
        keypoint_offsets: 关键点偏移字典
        rom_diff: 动作幅度差异字典
        weights: 权重字典，默认使用 COMPARISON_WEIGHTS

    Returns:
        相似度评分 (0-100)
    """
    if weights is None:
        weights = COMPARISON_WEIGHTS

    # 平均角度偏差
    avg_angle_dev = (
        np.mean(list(angle_deviations.values()))
        if angle_deviations else 0.0
    )

    # 平均关键点偏移
    avg_kp_offset = (
        np.mean(list(keypoint_offsets.values()))
        if keypoint_offsets else 0.0
    )

    # 平均ROM差异
    rom_diffs = [v["difference"] for v in rom_diff.values()]
    avg_rom_diff = np.mean(rom_diffs) if rom_diffs else 0.0

    # 综合评分
    score = 100.0 - (
        weights["angle_deviation"] * avg_angle_dev +
        weights["keypoint_offset"] * avg_kp_offset * 100 +  # 归一化偏移需放大
        weights["rom_difference"] * avg_rom_diff
    )

    return max(0.0, min(100.0, score))


def compare_poses(std_frames: list[PoseFrame],
                  pt_frames: list[PoseFrame]) -> Optional[EvaluationResult]:
    """
    完整姿态对比流程：
    1. DTW 帧对齐
    2. 角度偏差计算
    3. 关键点偏移计算
    4. 动作幅度差异计算
    5. 整体相似度评分

    Args:
        std_frames: 标准动作帧序列
        pt_frames: 患者动作帧序列

    Returns:
        EvaluationResult 评估结果，失败返回 None
    """
    if not std_frames or not pt_frames:
        return None

    # Step 1: 转换为角度序列并进行 DTW 对齐
    std_angle_seq = _frames_to_angle_sequence(std_frames)
    pt_angle_seq = _frames_to_angle_sequence(pt_frames)

    if std_angle_seq.shape[0] == 0 or pt_angle_seq.shape[0] == 0:
        return None

    alignment = dtw_align_frames(std_angle_seq, pt_angle_seq)

    if not alignment:
        return None

    # Step 2-4: 多维度偏差计算
    angle_devs = compute_angle_deviation(std_frames, pt_frames, alignment)
    kp_offsets = compute_keypoint_offset(std_frames, pt_frames, alignment)
    rom_diff = compute_rom_difference(std_frames, pt_frames, alignment)

    # Step 5: 整体相似度评分
    avg_kp_offset = float(np.mean(list(kp_offsets.values()))) if kp_offsets else 0.0
    score = compute_similarity_score(angle_devs, kp_offsets, rom_diff)

    return EvaluationResult(
        action_name="",
        angle_deviations=angle_devs,
        avg_keypoint_offset=round(avg_kp_offset, 4),
        rom_comparison=rom_diff,
        similarity_score=round(score, 1),
    )
