/* i18n.js — English / Chinese translations */

'use strict';

const TRANSLATIONS = {
  en: {
    // header
    title:    'IDV Match Predictor',
    subtitle: 'Bradley-Terry ordinal rating model · IVL / IJL / COA',
    lang_toggle: '中文',

    // tabs
    tab_single:      'Single Half',
    tab_box:         'BoX Series',
    tab_leaderboard: 'Leaderboard',

    // single half panel
    select_players: 'Select Players',
    hunter_label:   'Hunter',
    survivors_label: 'Survivors (4)',
    predict_btn:    'Predict',
    prediction_title: 'Prediction',
    select_prompt:  'Select a hunter and 4 survivors, then click Predict.',

    // box series panel
    format_label:       'Format',
    predict_series_btn: 'Predict Series',

    // half breakdown headings (plain text parts)
    half1_prefix: 'Half 1 — ',
    half1_verb:   ' hunts ',
    half2_prefix: 'Half 2 — ',
    half2_verb:   ' hunts ',

    // leaderboard
    search_player_ph: 'Search player…',
    hunters_btn:      'Hunters',
    survivors_btn:    'Survivors',
    min_games_label:  '30+ games only',
    col_rank:      '#',
    col_player:    'Player',
    col_role:      'Role',
    col_rating:    'Rating β',
    col_games:     'Games',
    col_last_seen: 'Last Seen',

    // search placeholders
    search_hunter_ph: 'Search hunter…',
    surv1_ph: 'Survivor 1',
    surv2_ph: 'Survivor 2',
    surv3_ph: 'Survivor 3',
    surv4_ph: 'Survivor 4',

    // alerts
    alert_hunter:     'Please select a hunter.',
    alert_survivors:  'Please select all 4 survivors.',
    alert_both_hunters:   'Please select hunters for both teams.',
    alert_both_survivors: 'Please select all 4 survivors for both teams.',

    // outcome labels (bar chart / table)
    n0: '0 esc (hunter crush)',
    n1: '1 esc (hunter win)',
    n2: '2 esc (draw)',
    n3: '3 esc (surv win)',
    n4: '4 esc (surv crush)',
    outcome_col: 'Outcome',
    prob_col:    'Probability',
    hunter_wins_row:   'Hunter wins (0–1 esc)',
    draw_row:          'Draw (2 esc)',
    survivors_win_row: 'Survivors win (3–4 esc)',
    most_likely_score: 'Most likely score:',

    // single-half result summary
    survivors_favoured: 'Survivors favoured',
    hunter_favoured:    'Hunter favoured',
    eta_line:     'η =',
    exp_escapes:  'Expected escapes:',
    most_likely_result: 'Most likely result:',

    // half score labels (halfScoreLabel)
    survivors_win: 'Survivors win',
    hunter_wins:   'Hunter wins',
    draw:          'Draw',
    escape_s:  'escapes',   // plural
    escape_1:  'escape',    // singular

    // series prediction panel
    predicted_round_score: 'Predicted round score:',
    per_round_odds:        'Per-round odds:',
    series_suffix:         'series:',
    tb_prefix:    'Tiebreaker (if needed):',
    tb_score:     'higher avg score',
    tb_equal:     'Tiebreaker: equal expected score (coin-flip)',
    tb_rule_note: 'Score tie after all rounds → extra round; extra round draw → avg margin decides.',

    // team labels
    team_a: 'Team A',
    team_b: 'Team B',

    // simulation
    simulated:     'Simulated',
    round_prefix:  'Rd ',
    round_suffix:  '',
    extra_label:   'Extra',
    h1_label:      'H1:',
    h2_label:      'H2:',
    esc_abbr:      'esc',
    series_winner_label: 'Series winner:',
    resimulate:    '↺ Re-simulate',

    // tie notes (constructed in simulateSeries)
    tie_note_score:      'Rounds tied',
    tie_note_wins_on:    'wins on score',
    tie_note_extra_pre:  'Rounds & score tied — extra round →',
    tie_note_extra_wins: 'wins',
    tie_note_extra_draw: 'Extra round draw — margin tiebreaker →',

    // leaderboard footer
    lb_showing: 'Showing',
    lb_of:      'of',
    lb_hunters:   'hunters',
    lb_survivors: 'survivors',

    // role display in table
    role_hunter:   'hunter',
    role_survivor: 'survivor',
  },

  zh: {
    title:    'IDV 比赛预测',
    subtitle: 'Bradley-Terry 序列评分模型 · IVL / IJL / COA',
    lang_toggle: 'English',

    tab_single:      '单局',
    tab_box:         'BoX 系列赛',
    tab_leaderboard: '排行榜',

    select_players:   '选择选手',
    hunter_label:     '监管者',
    survivors_label:  '求生者（4人）',
    predict_btn:      '预测',
    prediction_title: '预测结果',
    select_prompt:    '选择一位监管者和4位求生者，然后点击预测。',

    format_label:       '赛制',
    predict_series_btn: '预测系列赛',

    half1_prefix: '上半局 — ',
    half1_verb:   ' 担任监管者对阵 ',
    half2_prefix: '下半局 — ',
    half2_verb:   ' 担任监管者对阵 ',

    search_player_ph: '搜索选手…',
    hunters_btn:      '监管者',
    survivors_btn:    '求生者',
    min_games_label:  '仅30场起',
    col_rank:      '#',
    col_player:    '选手',
    col_role:      '职位',
    col_rating:    '评分 β',
    col_games:     '场数',
    col_last_seen: '最近活跃',

    search_hunter_ph: '搜索监管者…',
    surv1_ph: '求生者 1',
    surv2_ph: '求生者 2',
    surv3_ph: '求生者 3',
    surv4_ph: '求生者 4',

    alert_hunter:         '请选择监管者。',
    alert_survivors:      '请选择4位求生者。',
    alert_both_hunters:   '请为两队各选择监管者。',
    alert_both_survivors: '请为两队各选择4位求生者。',

    n0: '0人出门（监管者完封）',
    n1: '1人出门（监管者胜）',
    n2: '2人出门（平局）',
    n3: '3人出门（求生者胜）',
    n4: '4人出门（求生者完封）',
    outcome_col: '结果',
    prob_col:    '概率',
    hunter_wins_row:   '监管者胜（0–1人出门）',
    draw_row:          '平局（2人出门）',
    survivors_win_row: '求生者胜（3–4人出门）',
    most_likely_score: '最可能比分：',

    survivors_favoured: '求生者占优',
    hunter_favoured:    '监管者占优',
    eta_line:     'η =',
    exp_escapes:  '预期出门人数：',
    most_likely_result: '最可能结果：',

    survivors_win: '求生者胜',
    hunter_wins:   '监管者胜',
    draw:          '平局',
    escape_s:  '人出门',
    escape_1:  '人出门',

    predicted_round_score: '预测单局比分：',
    per_round_odds:        '单局胜率：',
    series_suffix:         '系列赛：',
    tb_prefix:    '加赛决出（如需要）：',
    tb_score:     '平均分更高',
    tb_equal:     '加赛：预期分相同（抽签）',
    tb_rule_note: '所有局打完平分 → 加赛；加赛平局 → 差分决定。',

    team_a: 'A队',
    team_b: 'B队',

    simulated:     '模拟',
    round_prefix:  '第',
    round_suffix:  '局',
    extra_label:   '加赛',
    h1_label:      '上半局：',
    h2_label:      '下半局：',
    esc_abbr:      '出门',
    series_winner_label: '系列赛胜者：',
    resimulate:    '↺ 重新模拟',

    tie_note_score:      '局数相同',
    tie_note_wins_on:    '以分数胜出',
    tie_note_extra_pre:  '局数和分数均相同 — 加赛 →',
    tie_note_extra_wins: '胜',
    tie_note_extra_draw: '加赛平局 — 差分决定 →',

    lb_showing:   '显示',
    lb_of:        '/',
    lb_hunters:   '位监管者',
    lb_survivors: '位求生者',

    role_hunter:   '监管者',
    role_survivor: '求生者',
  },
};

window.LANG = 'en';

function t(key) {
  return (TRANSLATIONS[window.LANG] || TRANSLATIONS.en)[key] || key;
}

function setLang(lang) {
  window.LANG = lang;

  // Static text nodes
  document.querySelectorAll('[data-i18n]').forEach(el => {
    el.textContent = t(el.dataset.i18n);
  });

  // Input placeholders
  document.querySelectorAll('[data-i18n-ph]').forEach(el => {
    el.placeholder = t(el.dataset.i18nPh);
  });

  // Lang toggle button label
  const btn = document.getElementById('lang-toggle');
  if (btn) btn.textContent = t('lang_toggle');

  // Half breakdown headings (contain colored spans, need innerHTML)
  const ta = `<span class="team-a-label">${t('team_a')}</span>`;
  const tb = `<span class="team-b-label">${t('team_b')}</span>`;
  const h1el = document.getElementById('half1-heading');
  const h2el = document.getElementById('half2-heading');
  if (h1el) h1el.innerHTML = t('half1_prefix') + ta + t('half1_verb') + tb;
  if (h2el) h2el.innerHTML = t('half2_prefix') + tb + t('half2_verb') + ta;

  // <html lang> attribute
  document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
}
