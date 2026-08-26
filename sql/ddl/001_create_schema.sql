PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS user_behavior (
    time TEXT NOT NULL,
    user_id INTEGER NOT NULL CHECK (user_id > 0),
    item_id INTEGER NOT NULL CHECK (item_id > 0),
    item_category INTEGER NOT NULL CHECK (item_category > 0),
    behavior_type INTEGER NOT NULL CHECK (behavior_type IN (1, 2, 3, 4))
);
