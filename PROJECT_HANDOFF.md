# Identity V Skill Decay Analysis — Project Handoff

## The Research Question

**How quickly does information about competitive skill become obsolete in a high-frequency esports environment?**

Operationalized: find the optimal exponential time-decay rate for weighting historical match outcomes when predicting future outcomes. The half-life of this decay is the answer.

This is structurally identical to the **alpha decay** problem in quantitative trading — every signal a fund discovers stops working at some rate, and measuring that rate is one of the central problems in systematic strategy research. The project is framed throughout as a methodologically-rigorous analog to that problem, with explicit framing in the README aimed at quant trading/research recruiters.

### Primary Deliverable

A single plot: out-of-sample predictive log-loss vs decay half-life across a swept range, with an identified optimum and a confidence band. The headline finding is a sentence like *"Skill information has an optimal half-life of X months for predicting Identity V pro match outcomes."*

### Author Context

User is a Brown sophomore (Math-CS, 4.0 GPA), applying to main 2027 quant recruiting opening July 1. PROMYS counselor starts June 21 — limited time during. **Effective focused work time before PROMYS: roughly 3 weeks (today is early June).** Project should be presentable on GitHub with clean code and a strong README by end of June.

---

## Project Scope (Carefully Constrained)

### The Three Components

**Component 1 — Time-weighted Bradley-Terry skill ratings.** A modified Bradley-Terry MLE with exponential weights on historical matches. Player-level ratings indexed by `(player_id, role)` tuple. L2 regularization for sparse players.

**Component 2 — Half-life sweep with temporal cross-validation.** Sweep decay half-life over a log-spaced range. For each candidate, fit Bradley-Terry on the training period and evaluate log-loss on a strictly-future validation period. Plot the curve, identify the optimum.

**Component 3 (stretch).** Decompose the decay rate across populations: role (hunter vs survivor), competitive tier (IVL vs IJL vs COA), and time (has the meta sped up?). Each is a self-contained additional analysis.

### What's Explicitly Out Of Scope

Do **not** build any of these unless every other component is complete and time remains:

- Mixed effects / variance decomposition model
- Hunter efficiency metric
- Era × character interactions
- Map effects, draft effects, player playstyle analysis
- Any per-survivor stats (repairs, rescues, heals, etc.)

These were in an earlier project plan and got cut for scope. The user has 3 focused weeks — depth over breadth.

---

## Minimal Data Schema

The core insight that scoped this project: the decay analysis only needs match outcomes, not detailed stats. **Use only 6 fields per game half:**

```sql
CREATE TABLE matches (
    match_id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,             -- ISO format YYYY-MM-DD
    tournament TEXT,                -- 'IVL_2024_summer_regular', 'COA9_groups', etc
    hunter_player TEXT NOT NULL,    -- canonical ID
    survivor1_player TEXT,
    survivor2_player TEXT,
    survivor3_player TEXT,
    survivor4_player TEXT,
    hunter_wins INTEGER NOT NULL,   -- 1 if hunter won, 0 if survivor side won
    
    -- Debugging metadata, not used in modeling
    source_file TEXT,
    source_row INTEGER
);

CREATE TABLE players (
    canonical_id TEXT PRIMARY KEY,
    known_roles TEXT,               -- 'hunter' | 'survivor' | 'both'
    first_seen TEXT,
    last_seen TEXT,
    n_games INTEGER
);
```

That's it. No player_stats table, no games-vs-player_stats join, no map/character columns. Only what the model uses.

### Why This Scope Matters

Carrying 80 columns "in case you need them" costs debugging cycles even when unused. The full alias dictionary (~140 entries) maps many columns we don't need — that's fine, the alias dictionary stays intact for future extensions, but **only extract the 6 modeling columns into SQLite**.

If the user later wants map effects or character analysis, the raw Excel files are preserved and re-extraction is straightforward. Don't optimize for hypothetical future needs.

### Source Coverage

For the matches table, use the modern `原始数据` sheets AND legacy raw tabs. Skip the modern `赛后数据` sheets entirely — they're redundant for outcome data and add complexity (cross-sheet joins, inconsistent hunter stat availability) we don't need.

### Draws

A small fraction of matches end in `平` (draws). Drop them for modeling — Bradley-Terry assumes binary outcomes. Document the drop count in the writeup.

---

## Data Architecture Reference

### Source Files (in `data/raw/`)

```
2020-2023.xlsx                      # Legacy file, year/tournament as tabs
2024IVL夏季赛常规赛.xlsx              # Modern files, one per season
2024IVL夏季赛季后赛.xlsx
2024IVL秋季赛常规赛.xlsx
2024IVL秋季赛季后赛.xlsx
2025IVL[夏季/秋季][常规/季后].xlsx     # 8 files for IVL
2024IJL夏季赛常规赛.xlsx              # IJL files
2024IJL秋季赛季后赛.xlsx
2025IJL[夏季/秋季][常规/季后].xlsx     # 6 files for IJL
COA8 全球总决赛小组赛.xlsx            # COA international events
COA8 全球总决赛淘汰赛.xlsx
COA9 全球总决赛小组赛.xlsx
COA9 全球总决赛淘汰赛.xlsx
```

**Excluded:** `2025IVS.xlsx`, both Japan qualifier files (regional, small sample, different competitive population).

**Missing:** COA7 — no data available. Document in README.

### Legacy file structure

- Tabs: `2020原始`, `2021原始`, `2022原始`, `2023原始`, `COA4`, `COA5`, `COA6`
- Also contains `曾用id` tab — player alias history. **Critical: must be parsed for player ID normalization.**
- Header row 0
- Each tab combines game-level and player-level data into one sheet

### Modern file structure

- Multiple sheets, but **only `原始数据` is used** for this project
- Header row varies; auto-detect with marker columns
- One file per tournament season

### Column Naming Inconsistency

Three generations of column naming exist. The alias dictionary unifies all of them.

| Concept | 2020-2022 | 2023 | 2024 | 2025/COA9 |
|---|---|---|---|---|
| Survivor ID | `求生者1ID` | `求生者1ID` | `求生者1ID` | `人ID1` |
| Winner | `胜利方` | `胜利方` | `胜利方` | `胜利方` |
| Hunter player | `屠名` | `屠名` | `屠名` | `屠名` |
| Date | various | various | from `月`+`日` cols | from `月`+`日` cols |

The hunter and winner columns are consistent across all years. The survivor ID columns are the only tricky one for our minimal schema.

### Template Row Filtering

Excel files have 700-1632 rows per sheet but most are pre-formatted empty templates with default values (`0`, `0.0`). These can't be filtered with `dropna(how='all')` since they contain those defaults. Filter using a "must be populated" anchor column — `hunter_team` works for all sheets, falling back to `hunter_player`, `survivor1_player`, `home_team`.

After filtering, expect roughly 4000-5000 game halves total across all sources.

### Player IDs

- ~231 unique IDs across modern data, more across legacy
- Capitalization inconsistent — must `.lower().strip()` everything
- Players change IDs between seasons — the `曾用id` tab tracks this
- A few dual-role players (ppxia is one) — use `(player_id, role)` tuples as keys
- The `曾用id` tab parsing is **the most important unfinished foundational task**

---

## What's Already Done

A Jupyter exploration notebook `notebooks/01_explore.ipynb` exists with working code for:

1. File inventory and existence checks
2. **Canonical column dictionary** (~140 entries, Chinese → English standardization)
3. Auto-detecting header reader functions (`read_legacy_tab`, `read_modern_sheet`)
4. Template-row filtering using anchor columns
5. Per-file and cross-file schema audits
6. Column collision/duplicate detection
7. Missing-value analysis
8. Player ID audit with fuzzy duplicate detection

### Known Working State

- All 18 modern files + legacy file load and normalize cleanly
- No unresolved column collisions
- Hunter character data: 0% missing in `原始数据` sheets
- Survivor data: <5% missing across all years
- ~3960 game halves combined modern + ~1500 legacy after template filtering

### Quirks Encountered During Exploration

- `角色` (no suffix) means survivor1's character in modern files but MVP's character in legacy. Legacy reader pre-renames it to `_qc_mvp_character` before normalization.
- `角色.4` means hunter character in modern `赛后数据` but MVP variant in legacy. Same pre-rename solution.
- `总逃脱` and `逃生数` are the same data — both in some files. One mapped to canonical, other to `_qc_`.
- 2024 summer + early fall files lack hunter aggregate stats in `赛后数据`. Irrelevant for this project since we don't use that sheet.

---

## Project Structure

```
idv-analysis/
├── venv/                  # Python venv (activated)
├── data/
│   ├── raw/              # All .xlsx files, READ ONLY
│   └── processed/        # SQLite db goes here
├── notebooks/
│   ├── 01_explore.ipynb  # Exploration (reference only, don't extend)
│   └── 02_modeling.ipynb # Working notebook for model development
├── src/
│   ├── ingest.py         # File reading + column normalization (TO BUILD)
│   ├── players.py        # Player ID canonicalization (TO BUILD)
│   ├── db.py             # SQLite schema and population (TO BUILD)
│   ├── bradley_terry.py  # Time-weighted BT model (TO BUILD)
│   └── eval.py           # Temporal CV evaluation (TO BUILD)
├── outputs/              # Plots, results
├── PROJECT_HANDOFF.md    # This file
└── README.md             # Final writeup
```

Installed: `pandas openpyxl numpy scipy statsmodels scikit-learn matplotlib seaborn plotly jupyter`.

---

## Build Plan (In Order)

### Phase 1 — Foundation (target: ~5 days)

Extract working code from the exploration notebook into proper modules. **Do not skip ahead to modeling until this is solid.**

**`src/ingest.py`:**
- The `COLUMN_ALIASES` dictionary
- `detect_header_row(path, sheet_name, markers)`
- `normalize_columns(df, alias_map)`
- `read_legacy_tab(tab_name)` — with `角色`/`角色.4` pre-rename
- `read_modern_sheet(filename, sheet_name)`
- `_drop_template_rows(df)` with multi-column anchor fallback
- File registry constants

**`src/players.py` — NEW, critical foundational task:**
- Parse the `曾用id` tab from `data/raw/2020-2023.xlsx`
- Inspect the table's structure first (likely wide format: canonical ID in col 0, aliases in subsequent cols, but verify)
- Build `ALIAS_MAP: Dict[str, str]` mapping every historical ID to its canonical form
- Add a manual `MANUAL_ALIASES` dict for post-2023 ID changes the legacy table doesn't cover (start with `{'ppicha': 'pipicha'}` based on the fuzzy match findings)
- `normalize_player_id(raw_id)` function that lowercases, strips, and looks up
- Apply this to all 5 ID columns at ingestion time, before any downstream processing

**`src/db.py`:**
- Schema as documented above
- `build_database()` function that:
  1. Reads every legacy raw tab and every modern `原始数据` sheet
  2. For each row, extracts the 6 modeling fields + 2 metadata fields
  3. Normalizes player IDs using `players.normalize_player_id`
  4. Computes `hunter_wins` from `winner_side`: `1` if `'屠'`, `0` if `'人'`, drop draws (`'平'`)
  5. Parses dates: for modern files use `月` and `日` columns plus year from filename; for legacy use the `date` column directly
  6. Constructs tournament identifier from filename / source tab name
  7. Inserts into SQLite
- `build_players_table()` — separate function that scans the matches table and produces the players registry

Validate by querying: total match count, count per tournament, count of unique players, date range, hunter win rate. Expected hunter win rate is in the 45-55% range based on earlier exploration.

### Phase 2 — Baseline Bradley-Terry (target: ~4 days)

**`src/bradley_terry.py`:**

```python
def fit_bradley_terry(
    matches: pd.DataFrame,         # columns: date, hunter_player, 
                                   #          survivor[1-4]_player, hunter_wins
    half_life_days: float = None,  # None = no time weighting
    reference_date: datetime = None,
    l2_lambda: float = 1.0,
    survivor_weight_w: float = 1.0  # weight on avg survivor skill on hunter side
) -> Dict[Tuple[str, str], float]:  # (player_id, role) -> beta
    """
    Fit time-weighted Bradley-Terry MLE via L-BFGS-B.
    Returns mapping from (player_id, role) tuple to log-skill parameter.
    """
```

The match strength function is asymmetric due to the 1v4 structure:

$$S_{hunter\_side} = \beta^H_{hunter\_player} + w \cdot \overline{\beta^S_{survivor\_team}}$$
$$P(\text{hunter wins}) = \sigma(S_{hunter\_side} - \overline{\beta^S_{survivor\_team}})$$

where $\overline{\beta^S}$ averages over the four survivor players' ratings. For v1, fix `w=1`. Estimating `w` jointly is a defensible extension if time permits.

Identifiability constraint: fix one player's $\beta$ to 0 (or work in zero-mean parameterization). Standard trick.

Sanity check: fit without time weighting, print top 10 hunter ratings and top 10 survivor ratings. They should be recognizable strong players from the user's domain knowledge.

### Phase 3 — Half-Life Sweep (target: ~5 days)

**`src/eval.py`:**

```python
def sweep_half_lives(
    matches: pd.DataFrame,
    half_lives_days: List[float],
    n_splits: int = 5,
    l2_lambda: float = 1.0
) -> pd.DataFrame:
    """
    For each candidate half-life, run temporal CV.
    Returns dataframe with columns: half_life, fold, train_size, 
                                    test_size, log_loss, accuracy, auc
    """
```

Critical methodology points:
- Use `sklearn.model_selection.TimeSeriesSplit` — NEVER random k-fold on time-ordered data
- Sort matches by date before splitting
- For each fold: fit BT on training rows, predict on validation rows, compute log-loss
- Cold-start handling: validation matches where any player has zero training appearances should either be dropped from the metric or get a fallback prediction (uniform priors). Document this choice.

Candidate half-lives: log-spaced from 30 days to 1825 days (5 years), roughly 15-20 points. Tighter sampling near the expected optimum (probably somewhere in the 6-month to 2-year range, but don't assume — let the data speak).

Plot result with matplotlib: x-axis log half-life, y-axis mean log-loss, error bars = standard error across folds. Mark the minimum.

### Phase 4 — Stretch Analyses (target: ~3 days)

Only if Phases 1-3 are clean and complete.

**Role asymmetry.** Fit BT for hunter ratings and survivor ratings separately. Sweep half-life independently for each. Compare the two optimal half-lives. Expected finding direction (hypothesis, not assumption): hunter half-life is shorter because hunter skill is more character-tied and characters get patched.

**Tier comparison.** Three separate sweeps: IVL only, IJL only, COA only. Compare optimal half-lives. Smallest sample size will be COA — confidence intervals will be wide; report them honestly.

**Temporal stability.** Two sweeps: matches from 2020-2022 only, matches from 2023-2025 only. Has the optimum shifted? A shorter optimum in recent years would suggest the meta is moving faster.

### Phase 5 — Writeup (target: ~3 days)

**`README.md`** with:

- One-sentence headline finding: *"Skill information in competitive Identity V has an optimal half-life of X months for predicting future match outcomes."*
- Brief background: what the game is, what the data is, why the question matters
- **Explicit quant framing**: alpha decay analogy, temporal cross-validation as backtest methodology, shrinkage via L2 regularization as analogous to portfolio covariance shrinkage
- Methodology section explaining time-weighted MLE, why log-loss, why temporal CV
- The headline plot (half-life sweep curve)
- Stretch analyses if done, each with their own plot
- Limitations section: linearized binary outcome treatment, draws dropped, regional/tier coverage gaps, cold-start handling
- Reproducibility: how to run the pipeline end to end

---

## Interview Talking Points To Embed In The Project

These are sentences the user should be able to say naturally in interviews. Build the project so these statements are accurate descriptions of what was done:

- *"I implemented time-weighted maximum likelihood Bradley-Terry from scratch, using L-BFGS-B optimization with L2 regularization for sparse players."*
- *"I used log-loss as the evaluation metric because it's a strictly proper scoring rule and matches the likelihood being optimized."*
- *"I used `TimeSeriesSplit` temporal cross-validation to avoid lookahead bias — random k-fold would have leaked future matches into training, which is the standard pitfall in backtesting trading strategies."*
- *"The decay rate parameter is structurally analogous to alpha decay in quantitative trading — I'm measuring how fast historical information becomes stale."*
- *"L2 regularization on player ratings is equivalent to a Gaussian prior, providing shrinkage for sparse players in the same way one would shrink a sample covariance matrix in portfolio construction."*

---

## Things To Avoid

1. **Do not use random train-test split.** Temporal CV only. This is the single most important methodological point.
2. **Do not try to fit GLMM or mixed effects models.** Cut from scope. Linear/logistic Bradley-Terry only.
3. **Do not include 2025IVS or Japan qualifier files** — different population.
4. **Do not extract more columns than the schema specifies.** Resist the temptation to "carry along" map names or characters "in case." Six columns plus metadata.
5. **Do not extend the exploration notebook.** Use a fresh `02_modeling.ipynb` for development and put production code in `src/`.
6. **Do not skip the `曾用id` parsing.** Player ID continuity is foundational. If IDs aren't normalized, the time decay analysis is contaminated by aliasing.
7. **Do not pre-emptively optimize for stretch goals.** Get Phases 1-3 done end to end first, then revisit.
8. **Do not trust `header=0` or `header=1` universally.** Always use `detect_header_row`.

---

## Suggested First Action

1. Read `notebooks/01_explore.ipynb` end to end to understand what's been built and what conventions are established
2. Extract code into `src/ingest.py` and verify it works by loading one legacy tab and one modern sheet
3. Then tackle `src/players.py` — start by inspecting the `曾用id` tab structure with a simple `pd.read_excel(...).head(20)` to see its layout, then build the alias map

After those two modules are working, building `src/db.py` is straightforward — it's mostly orchestration code that calls the ingestion and player normalization functions. Once the database exists, modeling can begin.
