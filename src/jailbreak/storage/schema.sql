PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS objectives (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    category TEXT NOT NULL,
    judge_rubric_override TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS techniques (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    family TEXT NOT NULL,
    description TEXT NOT NULL,
    exemplars_json TEXT NOT NULL DEFAULT '[]',
    source TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    objective_id TEXT NOT NULL,
    parent_ids_json TEXT NOT NULL DEFAULT '[]',
    technique_ids_json TEXT NOT NULL DEFAULT '[]',
    op TEXT NOT NULL CHECK (op IN ('mutate', 'crossover', 'seed')),
    strategy TEXT NOT NULL DEFAULT 'single_turn'
        CHECK (strategy IN ('single_turn', 'scripted_multi_turn', 'adaptive_multi_turn')),
    prompt TEXT NOT NULL,
    target_model TEXT NOT NULL,
    response TEXT NOT NULL,
    transcript_json TEXT,
    n_turns INTEGER NOT NULL DEFAULT 1,
    judge_score REAL NOT NULL,
    judge_success INTEGER NOT NULL,
    judge_rationale TEXT NOT NULL,
    refusal_signals_json TEXT NOT NULL DEFAULT '[]',
    success_turn INTEGER,
    features_json TEXT NOT NULL,
    embedding BLOB,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    latency_ms INTEGER NOT NULL DEFAULT 0,
    token_usage_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (objective_id) REFERENCES objectives(id)
);

CREATE INDEX IF NOT EXISTS idx_attempts_objective_success
    ON attempts(objective_id, judge_success);
CREATE INDEX IF NOT EXISTS idx_attempts_created_at
    ON attempts(created_at);

CREATE TABLE IF NOT EXISTS archive_cells (
    cell_key TEXT PRIMARY KEY,
    attempt_id INTEGER NOT NULL,
    score REAL NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (attempt_id) REFERENCES attempts(id)
);

CREATE TABLE IF NOT EXISTS reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iter INTEGER NOT NULL,
    summary TEXT NOT NULL,
    top_families_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_hash TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at TEXT,
    iters_completed INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS proposed_objectives (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    category TEXT NOT NULL,
    judge_rubric_override TEXT,
    rationale TEXT NOT NULL,
    proposed_at TEXT NOT NULL DEFAULT (datetime('now')),
    proposed_at_iter INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'rejected', 'blocked_by_ethics')),
    status_reason TEXT,
    reviewed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_proposed_objectives_status
    ON proposed_objectives(status);
