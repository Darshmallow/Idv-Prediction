/* app.js — IDV Match Predictor */

'use strict';

/* ── Model math ──────────────────────────────────────────────── */

const θ = MODEL.thresholds;   // [θ1, θ2, θ3, θ4]

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

/** P(n_escaped = 0..4) given η = mean(surv_β) − hunter_β */
function halfProbs(eta) {
  const u = θ.map(t => sigmoid(t - eta));
  return [
    u[0],
    u[1] - u[0],
    u[2] - u[1],
    u[3] - u[2],
    1 - u[3],
  ].map(p => Math.max(0, p));
}

function computeEta(hunterBeta, survivorBetas) {
  const validBetas = survivorBetas.filter(b => b !== null);
  if (validBetas.length === 0) return 0;
  return validBetas.reduce((a, b) => a + b, 0) / validBetas.length - hunterBeta;
}

// IDV competitive scoring per half (survivor pts : hunter pts):
// 0 esc → 0:5 | 1 esc → 1:3 | 2 esc → 2:2 | 3 esc → 3:1 | 4 esc → 5:0
const HALF_SCORE = [
  { surv: 0, hunt: 5 },
  { surv: 1, hunt: 3 },
  { surv: 2, hunt: 2 },
  { surv: 3, hunt: 1 },
  { surv: 5, hunt: 0 },
];

/**
 * Per-round result distribution.
 * probs1[k] = P(Team B escapes k) in Half 1 (Team A hunts Team B).
 * probs2[k] = P(Team A escapes k) in Half 2 (Team B hunts Team A).
 * Round score:
 *   Team A = HALF_SCORE[n1].hunt + HALF_SCORE[n2].surv
 *   Team B = HALF_SCORE[n1].surv + HALF_SCORE[n2].hunt
 */
function roundResult(probs1, probs2) {
  let pA = 0, pB = 0, pD = 0, expSA = 0, expSB = 0;
  for (let n1 = 0; n1 <= 4; n1++) {
    for (let n2 = 0; n2 <= 4; n2++) {
      const p  = probs1[n1] * probs2[n2];
      const sA = HALF_SCORE[n1].hunt + HALF_SCORE[n2].surv;
      const sB = HALF_SCORE[n1].surv + HALF_SCORE[n2].hunt;
      expSA += p * sA; expSB += p * sB;
      if (sA > sB) pA += p; else if (sB > sA) pB += p; else pD += p;
    }
  }
  return { pA, pB, pD, expSA, expSB };
}

/**
 * DP-based series win probability for a Bo(numRounds) series.
 *
 * Rules:
 *  - Play all numRounds rounds.
 *  - Winner = team with more round wins.
 *  - Tie on round wins → winner = team with higher cumulative score.
 *  - Tie on both → extra round: winner of extra round wins; if extra
 *    round draws → winner = team with better expected margin (from model).
 */
function computeSeriesProb(probs1, probs2, numRounds) {
  // Build compact round distribution (25 entries)
  const dist = [];
  for (let n1 = 0; n1 <= 4; n1++) {
    for (let n2 = 0; n2 <= 4; n2++) {
      const p  = probs1[n1] * probs2[n2];
      const sA = HALF_SCORE[n1].hunt + HALF_SCORE[n2].surv;
      const sB = HALF_SCORE[n1].surv + HALF_SCORE[n2].hunt;
      dist.push({ p, wA: sA > sB ? 1 : 0, wB: sB > sA ? 1 : 0, diff: sA - sB });
    }
  }

  // DP over (winsA, winsB, cumScoreDiff)
  // cumScoreDiff ranges from -10*numRounds to +10*numRounds
  const maxD = 10 * numRounds, sz = 2 * maxD + 1, off = maxD;
  let dp = Array.from({length: numRounds + 1}, () =>
    Array.from({length: numRounds + 1}, () => new Float64Array(sz))
  );
  dp[0][0][off] = 1.0;

  for (let r = 0; r < numRounds; r++) {
    const nx = Array.from({length: numRounds + 1}, () =>
      Array.from({length: numRounds + 1}, () => new Float64Array(sz))
    );
    for (let wA = 0; wA <= numRounds; wA++)
      for (let wB = 0; wB <= numRounds; wB++) {
        const layer = dp[wA][wB];
        for (const {p, wA: rA, wB: rB, diff} of dist)
          for (let d = 0; d < sz; d++) {
            if (!layer[d]) continue;
            const nd = d + diff;
            if (nd >= 0 && nd < sz) nx[wA + rA][wB + rB][nd] += layer[d] * p;
          }
      }
    dp = nx;
  }

  // Resolve outcomes; extra-round tiebreaker uses expected margin
  const rnd = roundResult(probs1, probs2);
  const eps = 1e-9;
  const pMarginA = rnd.expSA > rnd.expSB + eps ? 1 : rnd.expSA < rnd.expSB - eps ? 0 : 0.5;
  let pA_wins = 0;

  for (let wA = 0; wA <= numRounds; wA++)
    for (let wB = 0; wB <= numRounds; wB++) {
      const layer = dp[wA][wB];
      for (let d = 0; d < sz; d++) {
        if (!layer[d]) continue;
        const prob = layer[d], diff = d - off;
        if (wA > wB) { pA_wins += prob; }
        else if (wA < wB) { /* B wins */ }
        else {
          if (diff > 0) pA_wins += prob;
          else if (diff < 0) { /* B wins */ }
          else {
            // Fully tied → extra round, then margin tiebreaker
            pA_wins += prob * (rnd.pA + rnd.pD * pMarginA);
          }
        }
      }
    }
  return { pA: pA_wins, pB: 1 - pA_wins };
}

function pct(p) { return (p * 100).toFixed(1) + '%'; }
function pctRaw(p) { return (p * 100).toFixed(1); }
function fmt2(x) { return (x >= 0 ? '+' : '') + x.toFixed(2); }

/** Sample one outcome from a probability distribution. */
function sampleFromDist(probs) {
  let r = Math.random(), cum = 0;
  for (let i = 0; i < probs.length; i++) {
    cum += probs[i];
    if (r <= cum) return i;
  }
  return probs.length - 1;
}

/**
 * Simulate one full BoX series by drawing from the half distributions.
 * Returns { rounds, seriesWinner, tieNote }.
 * rounds = [{r, n1, n2, sA, sB, winner, extra?}]
 */
function simulateSeries(probs1, probs2, numRounds, rnd) {
  const rounds = [];
  let cumsA = 0, cumsB = 0, winsA = 0, winsB = 0;
  const winsNeeded = Math.ceil(numRounds / 2);

  for (let r = 1; r <= numRounds; r++) {
    const n1 = sampleFromDist(probs1);
    const n2 = sampleFromDist(probs2);
    const sA = HALF_SCORE[n1].hunt + HALF_SCORE[n2].surv;
    const sB = HALF_SCORE[n1].surv + HALF_SCORE[n2].hunt;
    const w  = sA > sB ? 'A' : sB > sA ? 'B' : 'D';
    if (w === 'A') winsA++; else if (w === 'B') winsB++;
    cumsA += sA; cumsB += sB;
    rounds.push({ r, n1, n2, sA, sB, winner: w });
    if (winsA >= winsNeeded || winsB >= winsNeeded) break;
  }

  let seriesWinner, tieNote = null;

  if (winsA > winsB) {
    seriesWinner = 'A';
  } else if (winsB > winsA) {
    seriesWinner = 'B';
  } else if (cumsA !== cumsB) {
    seriesWinner = cumsA > cumsB ? 'A' : 'B';
    const [hi, lo] = cumsA > cumsB ? [cumsA, cumsB] : [cumsB, cumsA];
    tieNote = `Rounds tied — ${seriesWinner === 'A' ? 'Team A' : 'Team B'} wins on score (${hi}–${lo})`;
  } else {
    // Extra round
    const n1e = sampleFromDist(probs1);
    const n2e = sampleFromDist(probs2);
    const sAe = HALF_SCORE[n1e].hunt + HALF_SCORE[n2e].surv;
    const sBe = HALF_SCORE[n1e].surv + HALF_SCORE[n2e].hunt;
    const we  = sAe > sBe ? 'A' : sBe > sAe ? 'B' : 'D';
    rounds.push({ r: 'X', n1: n1e, n2: n2e, sA: sAe, sB: sBe, winner: we, extra: true });

    if (we !== 'D') {
      seriesWinner = we;
      tieNote = `Rounds & score tied — extra round → Team ${we} wins (${sAe}–${sBe})`;
    } else {
      seriesWinner = rnd.expSA >= rnd.expSB ? 'A' : 'B';
      tieNote = `Extra round draw — margin tiebreaker → Team ${seriesWinner}`;
    }
  }

  return { rounds, seriesWinner, tieNote, cumsA, cumsB };
}

/** Return the most likely n (argmax P) for a half distribution. */
function predictedN(probs) {
  return probs.indexOf(Math.max(...probs));
}

/** Format a half outcome as "Survivors X – Hunter Y (n escapes)" */
function halfScoreLabel(n) {
  const s = HALF_SCORE[n];
  const chip = `<span class="score-chip">${s.surv} – ${s.hunt}</span>`;
  const esc = `(${n} escape${n === 1 ? '' : 's'})`;
  if (s.surv === s.hunt) return `Draw &nbsp;${chip}&nbsp; ${esc}`;
  if (s.surv > s.hunt)  return `Survivors win &nbsp;${chip}&nbsp; ${esc}`;
  return `Hunter wins &nbsp;${chip}&nbsp; ${esc}`;
}

/** Render a simulation result into #box-simulation. */
function renderSimulation(sim) {
  const fmt = seriesRounds === 3 ? 'Bo3' : seriesRounds === 5 ? 'Bo5' : 'Bo7';
  const winEl = sim.seriesWinner === 'A'
    ? `<span class="team-a-label">Team A</span>`
    : `<span class="team-b-label">Team B</span>`;

  const roundsHtml = sim.rounds.map(r => {
    const label = r.extra ? 'Extra' : `Rd ${r.r}`;
    const wClass = r.winner === 'A' ? 'team-a-label' : r.winner === 'B' ? 'team-b-label' : 'sim-draw';
    const wText  = r.winner === 'A' ? 'Team A' : r.winner === 'B' ? 'Team B' : 'Draw';
    return `
      <div class="sim-round${r.extra ? ' sim-extra' : ''}">
        <div class="sim-round-label">${label}</div>
        <div class="sim-score">
          <span class="team-a-label">${r.sA}</span>
          <span class="sim-dash"> – </span>
          <span class="team-b-label">${r.sB}</span>
        </div>
        <div class="sim-round-winner ${wClass}">${wText}</div>
        <div class="sim-halves">H1: ${r.n1} esc &nbsp;·&nbsp; H2: ${r.n2} esc</div>
      </div>`;
  }).join('');

  document.getElementById('box-simulation').innerHTML = `
    <div class="sim-header">Simulated ${fmt}</div>
    <div class="sim-rounds">${roundsHtml}</div>
    ${sim.tieNote ? `<div class="sim-tie-note">${sim.tieNote}</div>` : ''}
    <div class="sim-result">Series winner: ${winEl}</div>
    <button class="sim-resim-btn" id="resim-btn">↺ Re-simulate</button>`;

  document.getElementById('resim-btn').addEventListener('click', () => {
    renderSimulation(simulateSeries(_simProbs1, _simProbs2, seriesRounds, _simRnd));
  });
  document.getElementById('box-sim-wrap').classList.remove('hidden');
}

// Stored for re-simulate
let _simProbs1, _simProbs2, _simRnd;

/* ── Lookup helpers ──────────────────────────────────────────── */

const hunterMap   = new Map(MODEL.hunters.map(h => [h.id, h]));
const survivorMap = new Map(MODEL.survivors.map(s => [s.id, s]));

function lookupBeta(id, role) {
  const m = role === 'hunter' ? hunterMap : survivorMap;
  return m.has(id) ? m.get(id).beta : null;
}

/* ── Searchable select ───────────────────────────────────────── */

function initSearchableSelect(container, pool, onSelect) {
  const input    = container.querySelector('.ss-input');
  const dropdown = container.querySelector('.ss-dropdown');
  let selectedId = null;

  function renderOptions(query) {
    const q   = query.toLowerCase();
    const hit = pool.filter(p => p.id.toLowerCase().includes(q)).slice(0, 80);
    dropdown.innerHTML = '';
    hit.forEach(p => {
      const div   = document.createElement('div');
      div.className = 'ss-option';
      div.innerHTML = `<span class="op-name">${p.id}</span><span class="op-meta">β ${p.beta >= 0 ? '+' : ''}${p.beta.toFixed(2)} · ${p.nGames}g</span>`;
      div.addEventListener('mousedown', e => {
        e.preventDefault();
        input.value = p.id;
        input.classList.add('selected');
        selectedId = p.id;
        dropdown.classList.remove('open');
        onSelect(p.id, p.beta);
      });
      dropdown.appendChild(div);
    });
    const hasFocus = document.activeElement === input || input.matches(':focus');
    dropdown.classList.toggle('open', hit.length > 0 && hasFocus);
  }

  input.addEventListener('focus', () => renderOptions(input.value));
  input.addEventListener('input', () => {
    input.classList.remove('selected');
    selectedId = null;
    onSelect(null, null);
    renderOptions(input.value);
  });
  input.addEventListener('blur', () => {
    setTimeout(() => dropdown.classList.remove('open'), 150);
  });

  function getSelected() { return selectedId; }
  function getSelectedBeta() {
    if (!selectedId) return null;
    const p = pool.find(x => x.id === selectedId);
    return p ? p.beta : null;
  }
  function clear() {
    input.value = '';
    input.classList.remove('selected');
    selectedId = null;
    dropdown.classList.remove('open');
  }

  return { getSelected, getSelectedBeta, clear };
}

/* ── Prob bars renderer ──────────────────────────────────────── */

const LABELS = ['0 esc (hunter crush)', '1 esc (hunter win)', '2 esc (draw)', '3 esc (surv win)', '4 esc (surv crush)'];
const BAR_CLASSES = ['bar-n0', 'bar-n1', 'bar-n2', 'bar-n3', 'bar-n4'];

function renderBars(containerId, probs) {
  const el  = document.getElementById(containerId);
  el.innerHTML = '';
  const max = Math.max(...probs, 0.01);
  probs.forEach((p, i) => {
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
      <span class="bar-label">n=${i}</span>
      <div class="bar-outer">
        <div class="bar-inner ${BAR_CLASSES[i]}" style="width:${(p / max * 100).toFixed(1)}%"></div>
      </div>
      <span class="bar-pct">${pct(p)}</span>`;
    el.appendChild(row);
  });
}

function renderTable(tableId, probs) {
  renderTableWithScore(tableId, probs, predictedN(probs), true);
}

function renderTableWithScore(tableId, probs, n_pred, showPredicted) {
  const el = document.getElementById(tableId);
  const maxIdx = probs.indexOf(Math.max(...probs));
  const rows = probs.map((p, i) => {
    const hi = i === maxIdx ? ' class="highlight"' : '';
    return `<tr><td${hi}>${LABELS[i]}</td><td${hi} style="text-align:right">${pct(p)}</td></tr>`;
  });
  const pSurv = probs[3] + probs[4];
  const pHunt = probs[0] + probs[1];
  const s = HALF_SCORE[n_pred];
  const scoreRow = `
    <tr><td colspan="2" style="padding-top:8px;border-top:1px solid var(--border)"></td></tr>
    <tr>
      <td><span class="pred-label">Most likely score:</span></td>
      <td style="text-align:right">${halfScoreLabel(n_pred)}</td>
    </tr>`;
  el.innerHTML = `
    <thead><tr><th>Outcome</th><th style="text-align:right">Probability</th></tr></thead>
    <tbody>
      ${rows.join('')}
      <tr><td colspan="2" style="padding-top:8px;border-top:1px solid var(--border)"></td></tr>
      <tr><td style="color:var(--accent-b)">Hunter wins (0–1 esc)</td><td style="text-align:right;color:var(--accent-b);font-weight:600">${pct(pHunt)}</td></tr>
      <tr><td style="color:var(--muted)">Draw (2 esc)</td><td style="text-align:right;color:var(--muted);font-weight:600">${pct(probs[2])}</td></tr>
      <tr><td style="color:var(--green)">Survivors win (3–4 esc)</td><td style="text-align:right;color:var(--green);font-weight:600">${pct(pSurv)}</td></tr>
      ${scoreRow}
    </tbody>`;
}

/* ── SINGLE HALF ─────────────────────────────────────────────── */

const singleHunterSS = initSearchableSelect(
  document.getElementById('ss-hunter-single'),
  MODEL.hunters,
  () => {}
);

const singleSurvSS = Array.from(
  document.querySelectorAll('#single-survivors .searchable-select')
).map(el => initSearchableSelect(el, MODEL.survivors, () => {}));

document.getElementById('predict-single').addEventListener('click', () => {
  const hunterId = singleHunterSS.getSelected();
  const hunterBeta = singleHunterSS.getSelectedBeta();
  if (!hunterId || hunterBeta === null) {
    alert('Please select a hunter.'); return;
  }
  const survBetas = singleSurvSS.map(ss => ss.getSelectedBeta());
  const validBetas = survBetas.filter(b => b !== null);
  if (validBetas.length < 4) {
    alert('Please select all 4 survivors.'); return;
  }

  const eta   = computeEta(hunterBeta, validBetas);
  const probs = halfProbs(eta);
  const pSurv = probs[3] + probs[4];
  const pHunt = probs[0] + probs[1];
  const expN  = probs.reduce((s, p, i) => s + p * i, 0);

  const n_pred = predictedN(probs);
  const s_pred = HALF_SCORE[n_pred];

  const summaryEl = document.getElementById('single-summary');
  const winnerClass = pSurv > pHunt ? 'surv-win' : 'hunt-win';
  const winnerText  = pSurv > pHunt ? `Survivors favoured (${pct(pSurv)})` : `Hunter favoured (${pct(pHunt)})`;
  summaryEl.innerHTML = `
    <div>η = <span class="eta-val">${fmt2(eta)}</span> &nbsp;·&nbsp; Expected escapes: <span class="eta-val">${expN.toFixed(2)}</span></div>
    <div class="${winnerClass}" style="margin-top:6px;font-size:1rem">${winnerText}</div>
    <div class="predicted-result" style="margin-top:10px">
      <span class="pred-label">Most likely result:</span> ${halfScoreLabel(n_pred)}
    </div>`;

  renderBars('single-bars', probs);
  renderTable('single-table', probs);

  document.querySelector('#single-result .placeholder').classList.add('hidden');
  document.querySelector('#single-result .result-content').classList.remove('hidden');
});

/* ── BOX SERIES ──────────────────────────────────────────────── */

const boxHunterASS = initSearchableSelect(document.getElementById('ss-hunter-a'), MODEL.hunters, () => {});
const boxHunterBSS = initSearchableSelect(document.getElementById('ss-hunter-b'), MODEL.hunters, () => {});

const boxSurvASS = Array.from(
  document.querySelectorAll('#box-survivors-a .searchable-select')
).map(el => initSearchableSelect(el, MODEL.survivors, () => {}));

const boxSurvBSS = Array.from(
  document.querySelectorAll('#box-survivors-b .searchable-select')
).map(el => initSearchableSelect(el, MODEL.survivors, () => {}));

let seriesRounds = 3;   // total rounds in series (3 = Bo3, 5 = Bo5, 7 = Bo7)

document.querySelectorAll('.fmt-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.fmt-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    seriesRounds = parseInt(btn.dataset.rounds);
  });
});

document.getElementById('predict-box').addEventListener('click', () => {
  const hunterABeta = boxHunterASS.getSelectedBeta();
  const hunterBBeta = boxHunterBSS.getSelectedBeta();
  if (hunterABeta === null || hunterBBeta === null) {
    alert('Please select hunters for both teams.'); return;
  }
  const survABetas = boxSurvASS.map(ss => ss.getSelectedBeta()).filter(b => b !== null);
  const survBBetas = boxSurvBSS.map(ss => ss.getSelectedBeta()).filter(b => b !== null);
  if (survABetas.length < 4 || survBBetas.length < 4) {
    alert('Please select all 4 survivors for both teams.'); return;
  }

  // Half 1: Team A hunts → Team B's survivors face Team A's hunter
  const eta1   = computeEta(hunterABeta, survBBetas);
  const probs1 = halfProbs(eta1);   // P(Team B escapes k)

  // Half 2: Team B hunts → Team A's survivors face Team B's hunter
  const eta2   = computeEta(hunterBBeta, survABetas);
  const probs2 = halfProbs(eta2);   // P(Team A escapes k)

  // Per-round win probabilities and expected scores
  const rnd = roundResult(probs1, probs2);

  // Series win probability via DP with tiebreaker rules
  const seriesResult = computeSeriesProb(probs1, probs2, seriesRounds);
  const pASeries = seriesResult.pA, pBSeries = seriesResult.pB;

  // Predicted (argmax) outcome for each half → predicted round score
  const n1_pred = predictedN(probs1);   // Team B escapes in Half 1
  const n2_pred = predictedN(probs2);   // Team A escapes in Half 2
  const scoreA = HALF_SCORE[n1_pred].hunt + HALF_SCORE[n2_pred].surv;
  const scoreB = HALF_SCORE[n1_pred].surv + HALF_SCORE[n2_pred].hunt;
  const roundWinner = scoreA > scoreB ? `<span class="team-a-label">Team A</span>` :
                      scoreB > scoreA ? `<span class="team-b-label">Team B</span>` : 'Draw';
  const roundScoreLabel =
    `<span class="team-a-label">${scoreA}</span> – <span class="team-b-label">${scoreB}</span> &nbsp;(${roundWinner})`;

  // Tiebreaker note
  const tbNote = rnd.expSA > rnd.expSB
    ? `Tiebreaker (if needed): <span class="team-a-label">Team A</span> (higher avg score ${rnd.expSA.toFixed(1)} vs ${rnd.expSB.toFixed(1)})`
    : rnd.expSB > rnd.expSA
    ? `Tiebreaker (if needed): <span class="team-b-label">Team B</span> (higher avg score ${rnd.expSB.toFixed(1)} vs ${rnd.expSA.toFixed(1)})`
    : `Tiebreaker: equal expected score (coin-flip)`;

  const fmt = seriesRounds === 3 ? 'Bo3' : seriesRounds === 5 ? 'Bo5' : 'Bo7';

  // Series bar
  document.getElementById('series-fill-a').style.width = pctRaw(pASeries) + '%';
  document.getElementById('series-fill-b').style.width = pctRaw(pBSeries) + '%';
  document.getElementById('series-pct-a').textContent  = `Team A ${pct(pASeries)}`;
  document.getElementById('series-pct-b').textContent  = `Team B ${pct(pBSeries)}`;

  document.getElementById('game-probs').innerHTML = `
    <div><strong>Predicted round score:</strong> ${roundScoreLabel}</div>
    <div style="margin-top:6px"><strong>Per-round odds:</strong> Team A ${pct(rnd.pA)} · Draw ${pct(rnd.pD)} · Team B ${pct(rnd.pB)}</div>
    <div><strong>${fmt} series:</strong> Team A <strong style="color:var(--accent-a)">${pct(pASeries)}</strong> · Team B <strong style="color:var(--accent-b)">${pct(pBSeries)}</strong></div>
    <div style="font-size:.78rem;margin-top:6px;color:var(--text-dim)">${tbNote}</div>
    <div style="font-size:.78rem;margin-top:2px;color:var(--text-dim)">Score tie after all rounds → extra round; extra round draw → avg margin decides.</div>`;

  document.getElementById('box-result-summary').classList.remove('hidden');

  // Simulation
  _simProbs1 = probs1;
  _simProbs2 = probs2;
  _simRnd    = rnd;
  renderSimulation(simulateSeries(probs1, probs2, seriesRounds, rnd));

  // Half breakdowns with predicted score
  renderBars('half1-bars', probs1);
  renderTableWithScore('half1-table', probs1, n1_pred, false);
  renderBars('half2-bars', probs2);
  renderTableWithScore('half2-table', probs2, n2_pred, false);
  document.getElementById('box-halves').classList.remove('hidden');
});

/* ── LEADERBOARD ─────────────────────────────────────────────── */

const allPlayers = [
  ...MODEL.hunters.map(p  => ({ ...p, role: 'hunter' }))
    .sort((a, b) => b.beta - a.beta).map((p, i) => ({ ...p, rank: i + 1 })),
  ...MODEL.survivors.map(p => ({ ...p, role: 'survivor' }))
    .sort((a, b) => b.beta - a.beta).map((p, i) => ({ ...p, rank: i + 1 })),
];

let lbFilter   = 'hunter';
let lbSort     = { col: 'beta', dir: 'desc' };
let lbQuery    = '';
let lbMinGames = true;

function lbGetFiltered() {
  return allPlayers.filter(p => {
    const roleOk  = p.role === lbFilter;
    const queryOk = !lbQuery || p.id.toLowerCase().includes(lbQuery.toLowerCase());
    const gamesOk = !lbMinGames || p.nGames > 30;
    return roleOk && queryOk && gamesOk;
  });
}

function lbSorted(rows) {
  const { col, dir } = lbSort;
  return [...rows].sort((a, b) => {
    let va = a[col], vb = b[col];
    if (typeof va === 'string') va = va.toLowerCase(), vb = vb.toLowerCase();
    if (va < vb) return dir === 'asc' ? -1 : 1;
    if (va > vb) return dir === 'asc' ? 1 : -1;
    return 0;
  });
}

function renderLeaderboard() {
  const rows     = lbSorted(lbGetFiltered());
  const tbody    = document.getElementById('lb-body');
  const fragment = document.createDocumentFragment();

  rows.forEach(p => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="td-rank">${p.rank}</td>
      <td class="td-name">${p.id}</td>
      <td class="td-role-${p.role}">${p.role}</td>
      <td class="td-beta">${p.beta >= 0 ? '+' : ''}${p.beta.toFixed(4)}</td>
      <td class="td-games">${p.nGames}</td>
      <td class="td-last">${p.lastSeen}</td>`;
    fragment.appendChild(tr);
  });

  tbody.innerHTML = '';
  tbody.appendChild(fragment);

  const roleTotal = allPlayers.filter(p => p.role === lbFilter).length;
  document.getElementById('lb-footer').textContent =
    `Showing ${rows.length} of ${roleTotal} ${lbFilter}s`;

  // Update sort indicators
  document.querySelectorAll('.lb-table th').forEach(th => {
    th.classList.remove('active', 'asc', 'desc');
    if (th.dataset.col === lbSort.col) {
      th.classList.add('active', lbSort.dir);
    }
  });
}

document.getElementById('lb-search').addEventListener('input', e => {
  lbQuery = e.target.value;
  renderLeaderboard();
});

document.getElementById('lb-min-games').addEventListener('change', e => {
  lbMinGames = e.target.checked;
  renderLeaderboard();
});

document.querySelectorAll('.role-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.role-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    lbFilter = btn.dataset.role;
    renderLeaderboard();
  });
});

document.querySelectorAll('.lb-table th[data-col]').forEach(th => {
  th.addEventListener('click', () => {
    const col = th.dataset.col;
    if (lbSort.col === col) {
      lbSort.dir = lbSort.dir === 'asc' ? 'desc' : 'asc';
    } else {
      lbSort = { col, dir: col === 'beta' || col === 'nGames' ? 'desc' : 'asc' };
    }
    renderLeaderboard();
  });
});

renderLeaderboard();

/* ── Tabs ────────────────────────────────────────────────────── */

document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
  });
});
