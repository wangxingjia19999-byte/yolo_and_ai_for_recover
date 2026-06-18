"""数据层 —— 支持 SQLite（本地开发）和 PostgreSQL（Docker 部署）"""

import os
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import DB_PATH
from .keypoints import PoseFrame, StandardAction, Keypoint2D

# 数据库类型：从环境变量读取，支持 sqlite / postgresql
DATABASE_URL = os.getenv("DATABASE_URL", "")
IS_POSTGRES = DATABASE_URL.startswith("postgresql://")


def _get_connection():
    """获取数据库连接（自动适配 SQLite / PostgreSQL）"""
    if IS_POSTGRES:
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    else:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn


def _execute(conn, sql, params=None):
    """执行 SQL，返回 cursor"""
    cursor = conn.cursor()
    cursor.execute(sql, params or ())
    return cursor


def _fetchall(cursor):
    """获取所有结果，返回 list[dict]"""
    if IS_POSTGRES:
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]
    else:
        return [dict(row) for row in cursor.fetchall()]


def _fetchone(cursor):
    """获取单条结果，返回 dict 或 None"""
    row = cursor.fetchone()
    if row is None:
        return None
    if IS_POSTGRES:
        cols = [desc[0] for desc in cursor.description]
        return dict(zip(cols, row))
    else:
        return dict(row)


def init_database() -> None:
    """初始化数据库表结构（自动适配 SQLite / PostgreSQL）"""
    conn = _get_connection()
    cursor = conn.cursor()

    if IS_POSTGRES:
        # PostgreSQL 使用 init-db.sql 初始化，此处仅确认连接
        cursor.execute("SELECT 1")
        conn.commit()
        conn.close()
        print(f"[Database] PostgreSQL 已连接: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}")
    else:
        # SQLite 建表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS standard_actions (
                action_id TEXT PRIMARY KEY,
                action_name TEXT NOT NULL,
                description TEXT DEFAULT '',
                frames_json TEXT NOT NULL,
                angle_sequences_json TEXT DEFAULT '{}',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rehab_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                action_name TEXT NOT NULL,
                similarity_score REAL,
                angle_deviations TEXT,
                avg_keypoint_offset REAL,
                rom_comparison TEXT,
                ai_report TEXT,
                ai_confidence TEXT,
                rag_sources TEXT,
                duration_s REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        print(f"[Database] SQLite 已初始化: {DB_PATH}")


# ── 标准动作 CRUD ──────────────────────────────────────────

def _placeholder():
    """返回参数占位符"""
    return "%s" if IS_POSTGRES else "?"


def save_standard_action(action: StandardAction) -> None:
    """保存标准动作模板"""
    conn = _get_connection()
    frames_data = [f.model_dump() for f in action.frames]
    p = _placeholder()

    if IS_POSTGRES:
        sql = f"""INSERT INTO standard_actions
                  (action_id, action_name, description, frames_json, angle_sequences_json, created_at)
                  VALUES ({p}, {p}, {p}, {p}, {p}, {p})
                  ON CONFLICT (action_id) DO UPDATE SET
                    action_name=EXCLUDED.action_name,
                    description=EXCLUDED.description,
                    frames_json=EXCLUDED.frames_json,
                    angle_sequences_json=EXCLUDED.angle_sequences_json,
                    created_at=EXCLUDED.created_at"""
    else:
        sql = f"""INSERT OR REPLACE INTO standard_actions
                  (action_id, action_name, description, frames_json, angle_sequences_json, created_at)
                  VALUES ({p}, {p}, {p}, {p}, {p}, {p})"""

    _execute(conn, sql, (
        action.action_id,
        action.action_name,
        action.description,
        json.dumps(frames_data, ensure_ascii=False),
        json.dumps(action.angle_sequences, ensure_ascii=False),
        action.created_at or datetime.now().isoformat(),
    ))
    conn.commit()
    conn.close()


def load_standard_action(action_id: str) -> Optional[StandardAction]:
    """加载标准动作模板"""
    conn = _get_connection()
    p = _placeholder()
    cursor = _execute(conn, f"SELECT * FROM standard_actions WHERE action_id = {p}", (action_id,))
    row = _fetchone(cursor)
    conn.close()

    if row is None:
        return None

    frames_data = json.loads(row["frames_json"])
    frames = [PoseFrame(
        frame_id=f["frame_id"],
        timestamp_ms=f["timestamp_ms"],
        keypoints={k: Keypoint2D(**v) for k, v in f["keypoints"].items()},
    ) for f in frames_data]

    return StandardAction(
        action_id=row["action_id"],
        action_name=row["action_name"],
        description=row["description"] or "",
        frames=frames,
        angle_sequences=json.loads(row["angle_sequences_json"] or "{}"),
        created_at=str(row["created_at"] or ""),
    )


def list_standard_actions() -> list[dict]:
    """列出所有标准动作模板（仅摘要信息）"""
    conn = _get_connection()
    cursor = _execute(conn, "SELECT action_id, action_name, description, created_at FROM standard_actions ORDER BY created_at DESC")
    rows = _fetchall(cursor)
    conn.close()
    return rows


def delete_standard_action(action_id: str) -> bool:
    """删除标准动作模板"""
    conn = _get_connection()
    p = _placeholder()
    cursor = _execute(conn, f"DELETE FROM standard_actions WHERE action_id = {p}", (action_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


# ── 康复日志 CRUD ──────────────────────────────────────────

def save_rehab_log(
    patient_id: str,
    action_name: str,
    similarity_score: float,
    angle_deviations: dict,
    avg_keypoint_offset: float,
    rom_comparison: dict,
    ai_report: str = "",
    ai_confidence: str = "",
    rag_sources: list[str] = None,
    duration_s: float = 0.0,
) -> int:
    """保存康复日志，返回日志ID"""
    conn = _get_connection()
    p = _placeholder()

    if IS_POSTGRES:
        sql = f"""INSERT INTO rehab_logs
                  (patient_id, action_name, similarity_score, angle_deviations,
                   avg_keypoint_offset, rom_comparison, ai_report, ai_confidence,
                   rag_sources, duration_s, created_at)
                  VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p})
                  RETURNING id"""
    else:
        sql = f"""INSERT INTO rehab_logs
                  (patient_id, action_name, similarity_score, angle_deviations,
                   avg_keypoint_offset, rom_comparison, ai_report, ai_confidence,
                   rag_sources, duration_s, created_at)
                  VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p})"""

    params = (
        patient_id,
        action_name,
        similarity_score,
        json.dumps(angle_deviations, ensure_ascii=False),
        avg_keypoint_offset,
        json.dumps(rom_comparison, ensure_ascii=False),
        ai_report,
        ai_confidence,
        json.dumps(rag_sources or [], ensure_ascii=False),
        duration_s,
        datetime.now().isoformat(),
    )

    cursor = _execute(conn, sql, params)

    if IS_POSTGRES:
        log_id = cursor.fetchone()[0]
    else:
        log_id = cursor.lastrowid

    conn.commit()
    conn.close()
    return log_id


def get_rehab_logs(patient_id: str,
                   action_name: str = None,
                   limit: int = 50) -> list[dict]:
    """查询康复日志"""
    conn = _get_connection()
    p = _placeholder()
    query = f"SELECT * FROM rehab_logs WHERE patient_id = {p}"
    params = [patient_id]

    if action_name:
        query += f" AND action_name = {p}"
        params.append(action_name)

    query += f" ORDER BY created_at DESC LIMIT {p}"
    params.append(limit)

    cursor = _execute(conn, query, params)
    rows = _fetchall(cursor)
    conn.close()
    return rows


def get_rehab_log(log_id: int) -> Optional[dict]:
    """获取单条康复日志"""
    conn = _get_connection()
    p = _placeholder()
    cursor = _execute(conn, f"SELECT * FROM rehab_logs WHERE id = {p}", (log_id,))
    row = _fetchone(cursor)
    conn.close()
    return row


def get_score_trend(patient_id: str, action_name: str = None) -> list[dict]:
    """
    获取评分趋势数据。

    Returns:
        [{"created_at": "...", "similarity_score": float, "action_name": str}, ...]
    """
    conn = _get_connection()
    p = _placeholder()
    query = f"""SELECT created_at, similarity_score, action_name
               FROM rehab_logs WHERE patient_id = {p}"""
    params = [patient_id]

    if action_name:
        query += f" AND action_name = {p}"
        params.append(action_name)

    query += " ORDER BY created_at ASC"
    cursor = _execute(conn, query, params)
    rows = _fetchall(cursor)
    conn.close()
    return rows
