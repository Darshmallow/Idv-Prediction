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

/** Convolve two half distributions into P(A wins game), P(draw), P(B wins game).
 *  probsA = P(Team A escapes k), probsB = P(Team B escapes k) */
function gameResult(probsA, probsB) {
  let pA = 0, pDraw = 0, pB = 0;
  for (let a = 0; a <= 4; a++) {
    for (let b = 0; b <= 4; b++) {
      const p = probsA[a] * probsB[b];
      if (a > b)      pA    += p;
      else if (a < b) pB    += p;
      else            pDraw += p;
    }
  }
  return { pA, pDraw, pB };
}

function binom(n, k) {
  if (k < 0 || k > n) return 0;
  if (k === 0 || k === n) return 1;
  let r = 1;
  for (let i = 0; i < k; i++) r *= (n - i) / (i + 1);
  return r;
}

/** P(Team A wins a best-of-(2N-1) series), excluding draws (draws replay). */
function seriesProb(pA, pB, N) {
  const tot = pA + pB;
  if (tot < 1e-12) return 0.5;
  const pAe = pA / tot, pBe = pB / tot;
  let pASeries = 0;
  for (let k = 0; k < N; k++) {
    pASeries += binom(N + k - 1, k) * Math.pow(pAe, N) * Math.pow(pBe, k);
  }
  return pASeries;
}

function pct(p) { return (p * 100).toFixed(1) + '%'; }
function pctRaw(p) { return (p * 100).toFixed(1); }
function fmt2(x) { return (x >= 0 ? '+' : '') + x.toFixed(2); }

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
  const el = document.getElementById(tableId);
  const maxIdx = probs.indexOf(Math.max(...probs));
  const rows = probs.map((p, i) => {
    const hi = i === maxIdx ? ' class="highlight"' : '';
    return `<tr><td${hi}>${LABELS[i]}</td><td${hi} style="text-align:right">${pct(p)}</td></tr>`;
  });
  const pSurv = probs[3] + probs[4];
  const pHunt = probs[0] + probs[1];
  el.innerHTML = `
    <thead><tr><th>Outcome</th><th style="text-align:right">Probability</th></tr></thead>
    <tbody>
      ${rows.join('')}
      <tr><td colspan="2" style="padding-top:8px;border-top:1px solid var(--border)"></td></tr>
      <tr><td style="color:var(--accent-b)">Hunter wins (0–1 esc)</td><td style="text-align:right;color:var(--accent-b);font-weight:600">${pct(pHunt)}</td></tr>
      <tr><td style="color:var(--muted)">Draw (2 esc)</td><td style="text-align:right;color:var(--muted);font-weight:600">${pct(probs[2])}</td></tr>
      <tr><td style="color:var(--green)">Survivors win (3–4 esc)</td><td style="text-align:right;color:var(--green);font-weight:600">${pct(pSurv)}</td></tr>
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

  const summaryEl = document.getElementById('single-summary');
  const winnerClass = pSurv > pHunt ? 'surv-win' : 'hunt-win';
  const winnerText  = pSurv > pHunt ? `Survivors favoured (${pct(pSurv)})` : `Hunter favoured (${pct(pHunt)})`;
  summaryEl.innerHTML = `
    <div>η = <span class="eta-val">${fmt2(eta)}</span> &nbsp;·&nbsp; Expected escapes: <span class="eta-val">${expN.toFixed(2)}</span></div>
    <div class="${winnerClass}" style="margin-top:6px;font-size:1rem">${winnerText}</div>`;

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

let seriesN = 2;   // wins needed (2 = Bo3, 3 = Bo5, 4 = Bo7)

document.querySelectorAll('.fmt-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.fmt-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    seriesN = parseInt(btn.dataset.n);
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

  // Half 1: Team A hunts → Team B survives against Team A's hunter
  const eta1   = computeEta(hunterABeta, survBBetas);
  const probs1 = halfProbs(eta1);   // P(Team B escapes k)

  // Half 2: Team B hunts → Team A survives against Team B's hunter
  const eta2   = computeEta(hunterBBeta, survABetas);
  const probs2 = halfProbs(eta2);   // P(Team A escapes k)

  // Game: Team A wins if n_A > n_B
  const g     = gameResult(probs2, probs1);  // (probsA, probsB)
  const pASeries = seriesProb(g.pA, g.pB, seriesN);
  const pBSeries = 1 - pASeries;

  // Series bar
  document.getElementById('series-fill-a').style.width = pctRaw(pASeries) + '%';
  document.getElementById('series-fill-b').style.width = pctRaw(pBSeries) + '%';
  document.getElementById('series-pct-a').textContent  = `Team A ${pct(pASeries)}`;
  document.getElementById('series-pct-b').textContent  = `Team B ${pct(pBSeries)}`;

  const fmt = seriesN === 2 ? 'Bo3' : seriesN === 3 ? 'Bo5' : 'Bo7';
  document.getElementById('game-probs').innerHTML = `
    <div><strong>Per-game:</strong> Team A ${pct(g.pA)} · Draw ${pct(g.pDraw)} · Team B ${pct(g.pB)}</div>
    <div><strong>${fmt} series:</strong> Team A <strong style="color:var(--accent-a)">${pct(pASeries)}</strong> · Team B <strong style="color:var(--accent-b)">${pct(pBSeries)}</strong></div>
    <div style="font-size:.78rem;margin-top:4px;color:var(--text-dim)">Draws replay; draw probability excluded from series calculation.</div>`;

  document.getElementById('box-result-summary').classList.remove('hidden');

  // Half breakdowns
  renderBars('half1-bars', probs1);
  renderTable('half1-table', probs1);
  renderBars('half2-bars', probs2);
  renderTable('half2-table', probs2);
  document.getElementById('box-halves').classList.remove('hidden');
});

/* ── LEADERBOARD ─────────────────────────────────────────────── */

const allPlayers = [
  ...MODEL.hunters.map(p  => ({ ...p, role: 'hunter' })),
  ...MODEL.survivors.map(p => ({ ...p, role: 'survivor' })),
].sort((a, b) => b.beta - a.beta)
 .map((p, i) => ({ ...p, rank: i + 1 }));

let lbFilter = 'all';
let lbSort   = { col: 'beta', dir: 'desc' };
let lbQuery  = '';

function lbGetFiltered() {
  return allPlayers.filter(p => {
    const roleOk  = lbFilter === 'all' || p.role === lbFilter;
    const queryOk = !lbQuery || p.id.toLowerCase().includes(lbQuery.toLowerCase());
    return roleOk && queryOk;
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

  document.getElementById('lb-footer').textContent =
    `Showing ${rows.length} of ${allPlayers.length} players`;

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
