# Identity V Esports Analysis — Project Handoff

## Project Overview

A statistical analysis of competitive Identity V esports across 6 years of tournament data (IVL, IJL, COA from 2020-2025). The goal is to build a quant-style multi-component analysis demonstrating mixed effects modeling, shrinkage estimation, variance decomposition, and time-varying skill estimation. Designed as a portfolio project for quantitative trading/research internship recruiting (Jane Street Insight, summer 2027 QT/QR roles).

### About Identity V

Asymmetric 1v4 game: one player as the Hunter, four as Survivors. A match consists of two halves (the team that played Hunter swaps roles in the second half). Outcomes are scored by remaining generators (gens). Pro play uses a multi-phase ban/pick draft of characters and maps.

### The Four Models

1. **Bradley-Terry skill ratings (player-level)** — Latent skill parameter per (player, role). Roster changes handled automatically since ratings attach to players, not teams. MLE via `scipy.optimize` with L2 regularization for sparse players.

2. **Mixed effects logistic regression** — Outcome modeled with random effects for player skill, fixed effects for map, draft features, era. Era × character and era × map interactions capture meta evolution. Random effects provide shrinkage for sparse players (statistical analog of portfolio covariance shrinkage).

3. **Variance decomposition** — Decomposes outcome variance into player skill, draft, map, era, and residual components. The big interpretive output of the project: "what fraction of competitive outcomes is skill vs draft vs noise?"

4. **Hunter efficiency metric** — Per-player residual performance after subtracting character × map population baseline. Analog to alpha in factor models. Restricted to 2024秋 onward due to data availability.

### Evaluation Strategy

Temporal cross-validation only (`sklearn.TimeSeriesSplit`). Random splits would leak future into training, which is the standard lookahead bias trap in quant. Metrics: log-loss (primary, measures probability calibration), accuracy, AUC-ROC, and calibration curves.

### Quant Framing (For Interviews and README)

- Player skill ratings ↔ alpha estimation
- Random effects shrinkage ↔ portfolio covariance shrinkage
- Variance decomposition ↔ factor model attribution
- Era interactions ↔ alpha decay / regime change
- Recency-weighted estimation ↔ exponential decay in trading signals
- Temporal cross-validation ↔ backtest methodology, no lookahead bias

---

## Data Architecture

### Source Files (in `data/raw/`)

```
2020-2023.xlsx                      # Legacy file, year/tournament as separate tabs
2024IVL夏季赛常规赛.xlsx              # Modern files: one season per file
2024IVL夏季赛季后赛.xlsx
2024IVL秋季赛常规赛.xlsx
2024IVL秋季赛季后赛.xlsx
2025IVL[夏季/秋季][常规/季后].xlsx     # 8 files total for IVL
2024IJL夏季赛常规赛.xlsx              # IJL files
2024IJL秋季赛季后赛.xlsx
2025IJL[夏季/秋季][常规/季后].xlsx     # 6 files for IJL
COA8 全球总决赛小组赛.xlsx            # COA main events
COA8 全球总决赛淘汰赛.xlsx
COA9 全球总决赛小组赛.xlsx
COA9 全球总决赛淘汰赛.xlsx
```

**Excluded:** `2025IVS.xlsx`, `COA8 日本赛区预选赛.xlsx`, `COA9 日本赛区预选赛.xlsx` (regional qualifiers — different competitive population, small sample, would add noise).

**Missing:** COA7 — no data available, documented in README.

### Format Differences

**Legacy file (2020-2023.xlsx):**
- Multiple tabs per file: `2020原始`, `COA4`, `2021原始`, `COA5`, `2022原始`, `COA6`, `2023原始`
- Also has `曾用id` tab (player alias history — must be loaded for player ID normalization)
- Header row 0
- Each tab combines game-level data and player-level data in ONE sheet

**Modern files (2024+):**
- One file per tournament season
- Multiple sheets: `原始数据` (game-level), `赛后数据` (player-level stats), `对局数据` (in-game events, not yet used)
- Header row varies (usually 0 or 1) — auto-detected
- 2024 IVL/IJL summer + select fall files lack hunter aggregate stats in 赛后数据 (no hunter ID, hits, knockdowns) — only 14 modern files have full hunter data

### Schema Evolution Across Years

| Concept | 2020 | 2021 | 2022 | 2023 | 2024+ |
|---|---|---|---|---|---|
| Game halves (上/下) | ❌ | ✓ | ✓ | ✓ | ✓ |
| Survivor team identifier | ❌ | ✓ | ✓ | ✓ | ✓ |
| Per-player rescues | ❌ | ✓ | ✓ | ✓ | ✓ |
| Board breaks per player | ❌ | ❌ | ❌ | ✓ | ✓ |
| Heals per player | ❌ | ❌ | ❌ | ✓ | ✓ |
| Hunter aggregate stats | ❌ | ❌ | ❌ | ❌ | partial |

**Modeling implication:**
- Bradley-Terry skill: use all 6 years
- Mixed effects variance decomp: use 2021+ (need game halves)
- Hunter efficiency metric: use 2024 fall onward only (14 modern files)

### Column Naming Inconsistency

Three generations of column naming exist; the normalization dictionary unifies all of them:

| Concept | 2020-2022 | 2023 | 2024 | 2025/COA9 |
|---|---|---|---|---|
| Survivor ID | `求生者1ID` | `求生者1ID` | `求生者1ID` | `人ID1` |
| Survivor character | `使用角色` | `角色` | `角色` | `角色` |
| Repairs | `修机进度` | `修机进度` | `修机进度` | `修机` |
| Rescues | `救人数` | `救人数` | `救人数` | `救人` |
| Heals | (none) | `治疗数` | `治疗数` | `治疗` |
| Board breaks | (none) | `砸板命中` | `砸板命中` | `砸板` |
| Harassment | `牵制时长` | `牵制时长` | `牵制时长` | `牵制` |
| Hunter ID (赛后) | n/a | n/a | `屠名` (partial) | `屠ID` |

### Tricky Quirks (Context-Dependent Column Meanings)

- **`角色` (no suffix)** in legacy `原始数据`: MVP's character (junk). In modern `赛后数据`: survivor1's character.
- **`角色.4`** in legacy: MVP character variant (junk). In modern `赛后数据`: hunter's character.
- **`场次`**: match number, often only filled for first half of a series (~64% null is expected, not a bug — can be reconstructed from date + teams later).
- **`总逃脱` and `逃生数`**: same data, both columns kept by data entry for QC. Use one, flag the other as `_qc_`.

The legacy reader pre-renames `角色` and `角色.4` to `_qc_mvp_character_*` BEFORE normalization to handle this.

### Player ID Issues

- ~231 unique player IDs across modern data
- Capitalization is inconsistent — must `.lower().strip()` all IDs
- Some likely typo duplicates flagged by fuzzy match (e.g. `ppicha` vs `pipicha`)
- Players occasionally change IDs between seasons — `曾用id` tab in legacy file tracks this (not yet parsed)
- Most players are role-specialized; a few dual-role (e.g. `ppxia`) — give them separate ratings per role: index by `(player_id, role)` tuple

### Template Row Filtering

Excel files have 700-1632 rows per sheet but most are pre-formatted empty template rows with default values like `0` and `0.0`. These can't be filtered with `dropna(how='all')` since they contain those default zeros. Filter using a "must be populated" anchor column. The `_drop_template_rows` helper tries `hunter_team`, `hunter_player`, `survivor1_player`, `home_team` in order.

After filtering, expect roughly:
- Legacy raw tabs: 300-500 rows each
- Modern raw sheets: 200-250 rows each (a season has ~100-120 game halves typically)
- Total dataset: ~4000-5000 game halves across all sources

---

## What's Done

A Jupyter exploration notebook `notebooks/01_explore.ipynb` containing:

1. File inventory and existence checks
2. **Canonical column dictionary** (~140 entries) mapping Chinese variants to standardized English names — this is the most important artifact built so far
3. Auto-detecting header reader functions (`read_legacy_tab`, `read_modern_sheet`)
4. Template-row filtering using anchor columns
5. Per-file and cross-file schema audits
6. Column collision/duplicate detection
7. Missing-value analysis
8. Player ID audit with fuzzy duplicate detection
9. Map and character coverage analysis

The notebook is exploratory and inline. It needs to be migrated into proper modules.

### Known Working State

- All 18 modern files + legacy file load and normalize cleanly
- No remaining unresolved column collisions
- Hunter character data: 0% missing in 原始数据 sheets
- Survivor data: <5% missing across all years where columns exist
- Hunter aggregate stats: missing for 4 early 2024 files (expected)
- ~3960 game halves combined modern + ~1500 legacy (after template filtering)

---

## Project Structure Setup

The project root is already initialized as a git repo with a virtual environment:

```
idv-analysis/
├── venv/                  # Python virtual environment (activated)
├── data/
│   ├── raw/              # All .xlsx files (read-only — DO NOT MODIFY)
│   └── processed/        # Cleaned outputs go here
├── notebooks/
│   └── 01_explore.ipynb  # Exploration notebook (existing)
├── src/                  # Empty — needs to be built
├── outputs/              # For charts and reports
└── README.md
```

Installed packages: `pandas openpyxl numpy scipy statsmodels scikit-learn matplotlib seaborn plotly jupyter`.

---

## Next Steps (In Order)

### Phase 1 — Promote Notebook Code Into Modules

The exploration notebook has working code that needs to be productionized. Create:

**`src/ingest.py`** — File reading and column normalization:
- The `COLUMN_ALIASES` dictionary (currently ~140 entries, will grow)
- `detect_header_row(path, sheet_name, markers)`
- `normalize_columns(df, alias_map)`
- `read_legacy_tab(tab_name)` — with the `角色`/`角色.4` pre-rename quirk
- `read_modern_sheet(filename, sheet_name)`
- `_drop_template_rows(df)` with multi-column anchor fallback
- File registry constants (`LEGACY_FILE`, `IVL_FILES`, `IJL_FILES`, `COA_FILES`)

**`src/players.py`** — Player ID normalization (NEW, not yet implemented):
- Parse the `曾用id` tab from `data/raw/2020-2023.xlsx`
- Build `ALIAS_MAP: Dict[str, str]` mapping old IDs to canonical IDs
- Support manual overrides for post-2023 ID changes
- `normalize_player_id(raw_id)` function applied at ingestion time
- Apply to all 5 player ID columns (`hunter_player`, `survivor1_player`, ..., `survivor4_player`)
- Handle case sensitivity (lowercase everything)

**`src/db.py`** — SQLite database operations (NEW):

Schema design:
```sql
CREATE TABLE games (
    game_id INTEGER PRIMARY KEY,
    tournament TEXT,         -- e.g. 'IVL', 'IJL', 'COA8'
    season TEXT,             -- e.g. '2025夏季常规'
    date TEXT,
    half TEXT,               -- '上' or '下', NULL for 2020
    home_team TEXT,
    away_team TEXT,
    hunter_team TEXT,
    survivor_team TEXT,
    hunter_player TEXT,      -- canonical ID
    hunter_character TEXT,
    map_name TEXT,
    winner_side TEXT,        -- '屠' or '人'
    gens_remaining INTEGER,
    source_file TEXT         -- for traceability
);

CREATE TABLE player_stats (
    game_id INTEGER REFERENCES games(game_id),
    slot INTEGER,            -- 1-4 for survivors, 0 for hunter
    player_id TEXT,          -- canonical
    role TEXT,               -- 'hunter' or 'survivor'
    character TEXT,
    repairs REAL,
    rescues INTEGER,
    heals REAL,
    boards REAL,
    harassment REAL,
    result TEXT,             -- escaped/eliminated/etc
    PRIMARY KEY (game_id, slot)
);

CREATE TABLE players (
    canonical_id TEXT PRIMARY KEY,
    known_roles TEXT,        -- 'hunter' | 'survivor' | 'both'
    first_seen TEXT,
    last_seen TEXT
);
```

Build the panel dataset by:
1. Reading every legacy tab and modern raw sheet
2. Pivoting legacy survivor columns from wide-per-slot format to long
3. Joining modern raw sheets with their corresponding 赛后数据 sheets on (date, half, hunter_team, survivor_team)
4. Writing to SQLite with proper indexes

### Phase 2 — Feature Engineering

**`src/features.py`** — Compute analysis features from the database:
- Encode categorical variables (maps as dummies, characters as dummies)
- Build era buckets (year-level for character/map interactions)
- Compute "main pick" indicator (fraction of a player's games on this character)
- Aggregate player stats for the efficiency metric

### Phase 3 — Models

**`src/models/bradley_terry.py`** — Player skill ratings:
- Implement from scratch using `scipy.optimize.minimize` with L-BFGS-B
- L2 regularization for sparse players
- Player parameters indexed by `(player_id, role)` tuple
- Asymmetric match strength function: $S_{hunter\_side} = \beta^H_{hunter} + \frac{w}{4}\sum \beta^S_{survivor_k}$, with $w$ estimated jointly

**`src/models/mixed_effects.py`** — The variance decomposition model:
- Use `statsmodels.MixedLM` (linear mixed model — note GLMM limitation in README, since outcomes are binary)
- Random effects: hunter player, four survivor players
- Fixed effects: map, character, era, character × era interactions
- Extract variance components after fit

**`src/models/efficiency.py`** — Hunter efficiency metric:
- Population baseline by (character, map)
- Per-player z-scored residuals
- Aggregate per player with minimum-games threshold (probably 15-20)
- Restricted to 2024秋 onward

**`src/eval.py`** — Evaluation:
- Temporal cross-validation using `TimeSeriesSplit`
- Log-loss (primary), accuracy, AUC-ROC
- Calibration curves
- Separate metrics for sparse vs well-represented players

### Phase 4 — Outputs and Writeup

Generate:
- Player skill rating leaderboards (top hunters, top survivors)
- Character meta evolution timelines
- Variance decomposition bar chart (% explained by each factor)
- Calibration plots
- README writeup with quant framing for interview talking points

---

## Important Things To Avoid

1. **Don't try to use random train-test split.** Temporal CV only.
2. **Don't drop the 2020 data** when adding the survivor_team filter — 2020 lacks `survivor_team`. The `_drop_template_rows` function specifically uses `hunter_team` as primary anchor for this reason.
3. **Don't try to fit GLMM with statsmodels** — its mixed model is linear only. Note the linearization as a limitation in the README. If time permits, can use `pymer4` (R wrapper) for proper logistic mixed models.
4. **Don't try to fit player-by-character interactions** — too many sparse parameters. Use the "main pick" indicator instead.
5. **Don't include IVT or Japan qualifier files** — different competitive population, would introduce noise.
6. **Don't forget the `_qc_` columns** — drop any column prefixed `_qc_` or `_` before modeling.
7. **Don't trust `header=0` or `header=1` universally** — always use `detect_header_row`. Different files use different rows.

---

## Timeline Constraint

User starts PROMYS counselor program on **June 21** which is intensive and immersive. Effective working time before PROMYS: ~3 weeks (current date June 2). Goal: have a working pipeline + baseline Bradley-Terry model + preliminary variance decomposition by June 21. Refinement happens during PROMYS in limited free time. Full project completion target: end of July.

Application deadline for Jane Street Insight is June 14 — the user is applying without this project on the resume (existing resume is already strong enough). This project is for the **summer 2027 main recruiting cycle** starting July 1.

---

## Suggested First Action For Claude Code

Start by examining the existing exploration notebook (`notebooks/01_explore.ipynb`) to ground yourself in what's been built, then create `src/ingest.py` by extracting and cleaning up the relevant functions and constants. Confirm it works by loading a sample file end-to-end and printing a clean dataframe.

After that, tackle player alias parsing from the `曾用id` tab — this is the biggest unfinished foundational task before the database can be built.
