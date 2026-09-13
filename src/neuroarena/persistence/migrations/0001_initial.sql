CREATE TABLE models (
    model_id TEXT PRIMARY KEY,
    backend TEXT NOT NULL,
    observation_space TEXT NOT NULL,
    action_space TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    track_id TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL,
    starting_generation INTEGER NOT NULL
);

CREATE TABLE generation_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    generation INTEGER NOT NULL,
    best_fitness REAL NOT NULL,
    mean_fitness REAL NOT NULL,
    worst_fitness REAL NOT NULL,
    population_size INTEGER NOT NULL,
    champion_metrics TEXT NOT NULL,
    sim_time REAL NOT NULL,
    wall_time REAL NOT NULL
);

CREATE TABLE settings_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    generation INTEGER NOT NULL,
    changed_at TEXT NOT NULL,
    diff TEXT NOT NULL
);

CREATE TABLE checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    kind TEXT NOT NULL,
    generation INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    schema_version INTEGER NOT NULL
);

CREATE INDEX idx_generation_stats_model ON generation_stats(model_id, generation);
CREATE INDEX idx_settings_history_model ON settings_history(model_id, generation);
CREATE INDEX idx_checkpoints_model_kind ON checkpoints(model_id, kind, generation);
