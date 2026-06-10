CREATE TABLE matches (
    match_id          INTEGER PRIMARY KEY,
    date              TEXT,
    tournament        TEXT,
    tournament_tier   TEXT,
    hunter_player     TEXT,
    survivor1_player  TEXT,
    survivor2_player  TEXT,
    survivor3_player  TEXT,
    survivor4_player  TEXT,
    map_name          TEXT,
    n_escaped         INTEGER,
    hunter_wins       INTEGER,
    source_file       TEXT,
    source_row        INTEGER
);
CREATE TABLE players (
    canonical_id TEXT PRIMARY KEY,
    known_roles  TEXT,
    first_seen   TEXT,
    last_seen    TEXT,
    n_games      INTEGER
);
CREATE INDEX idx_matches_date            ON matches(date);
CREATE INDEX idx_matches_hunter_player   ON matches(hunter_player);
CREATE INDEX idx_matches_tournament      ON matches(tournament);
CREATE INDEX idx_matches_tournament_tier ON matches(tournament_tier);
CREATE INDEX idx_matches_map_name        ON matches(map_name);
