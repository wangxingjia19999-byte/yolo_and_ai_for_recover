-- 康复训练姿态评估系统 - PostgreSQL 初始化脚本

-- 标准动作模板表
CREATE TABLE IF NOT EXISTS standard_actions (
    action_id TEXT PRIMARY KEY,
    action_name TEXT NOT NULL,
    description TEXT DEFAULT '',
    frames_json TEXT NOT NULL,
    angle_sequences_json TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 康复日志表
CREATE TABLE IF NOT EXISTS rehab_logs (
    id SERIAL PRIMARY KEY,
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_rehab_logs_patient ON rehab_logs(patient_id);
CREATE INDEX IF NOT EXISTS idx_rehab_logs_action ON rehab_logs(action_name);
CREATE INDEX IF NOT EXISTS idx_rehab_logs_created ON rehab_logs(created_at);
