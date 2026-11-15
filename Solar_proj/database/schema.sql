CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    voltage REAL NOT NULL,
    current REAL NOT NULL,
    power REAL NOT NULL,
    energy_wh REAL NOT NULL,
    source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    filename TEXT NOT NULL,
    condition TEXT NOT NULL,
    confidence REAL NOT NULL,
    affected_area REAL NOT NULL,
    source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cleanings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    reason TEXT NOT NULL,
    mode TEXT NOT NULL,
    duration_seconds REAL NOT NULL,
    status TEXT NOT NULL,
    before_condition TEXT NOT NULL,
    after_condition TEXT NOT NULL,
    before_area REAL NOT NULL,
    after_area REAL NOT NULL,
    source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    description TEXT NOT NULL,
    acknowledged INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL
);
