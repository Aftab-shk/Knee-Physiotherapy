/*
 * progress-view.js - renders a progress payload into charts.
 * ==========================================================
 *
 * Shared by progress.html and share.html. Both draw the same things from the
 * same shape of data; the only difference is how they fetch it - one as the
 * signed-in patient, the other through a share token. Keeping the drawing in one
 * place is what stops a clinician's view quietly diverging from the patient's.
 *
 * Usage:
 *   <link rel="stylesheet" href="assets/progress-view.css">
 *   <div id="progress-root"></div>
 *   <script src="assets/progress-view.js"></script>
 *   ProgressView.mount(document.getElementById('progress-root'));
 *   ProgressView.render(payload);          // a /me/progress or /share/{t} body
 *
 * mount() injects the markup the drawing functions expect, so the element IDs
 * below are an internal detail of this file rather than a contract each page has
 * to remember to honour.
 */
(function (global) {
  'use strict';

  // No wrapper of its own: the page supplies the host (#content) and shows or
  // hides it. A wrapper here once carried id="content" class="hidden" too, which
  // duplicated the page's id - getElementById found the outer one, unhid it, and
  // left every chart inside the inner one hidden for good.
  const MARKUP = `
        <div class="kpis">
          <div class="kpi">
            <div class="kpi-label">Latest flexion</div>
            <div id="kpi-latest"></div>
            <div class="kpi-foot" id="kpi-latest-foot"></div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Best in range</div>
            <div id="kpi-best"></div>
            <div class="kpi-foot" id="kpi-best-foot"></div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Current streak</div>
            <div id="kpi-streak"></div>
            <div class="kpi-foot" id="kpi-streak-foot"></div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Sessions</div>
            <div id="kpi-sessions"></div>
            <div class="kpi-foot" id="kpi-sessions-foot"></div>
          </div>
          <!-- Hidden until a questionnaire has been answered. A "No data" tile
               for something nobody has been offered yet reads as broken. -->
          <div class="kpi hidden" id="kpi-koos-tile">
            <div class="kpi-label">How it feels</div>
            <div id="kpi-koos"></div>
            <div class="kpi-foot" id="kpi-koos-foot"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-head">
            <div class="card-title">Range of motion</div>
            <div class="card-note" id="rom-note"></div>
          </div>
          <div class="card-sub">
            The furthest your knee bent each day, against the safe limit set from your X-ray.
            Shown one exercise at a time: a straight-leg hold and a squat ask different things of the knee,
            so their angles are not comparable.
          </div>
          <div class="ex-picker" id="ex-picker" role="group" aria-label="Exercise"></div>
          <div class="chart-wrap" id="rom-wrap">
            <svg class="chart" id="rom-chart" role="img" aria-labelledby="rom-desc"></svg>
            <div class="sr-only" id="rom-desc"></div>
            <div class="tip" id="rom-tip"></div>
          </div>
          <div class="legend" id="rom-legend"></div>
        </div>

        <!-- The patient's own verdict. Next to range of motion on purpose: the
             two answer different questions, and the interesting cases are the
             ones where they disagree. -->
        <div class="card hidden" id="koos-card">
          <div class="card-head">
            <div class="card-title">How your knee feels</div>
            <div class="card-note" id="koos-note"></div>
          </div>
          <div class="card-sub">
            Your own answers to the KOOS-JR questionnaire, scored 0 to 100, the same scale
            a joint registry uses, where higher is better. This is the half of recovery that
            bending further does not measure.
          </div>
          <div class="ex-picker" id="koos-picker" role="group" aria-label="Knee"></div>
          <div class="chart-wrap" id="koos-wrap">
            <svg class="chart" id="koos-chart" role="img" aria-labelledby="koos-desc"></svg>
            <div class="sr-only" id="koos-desc"></div>
            <div class="tip" id="koos-tip"></div>
          </div>
          <div class="legend" id="koos-legend"></div>
        </div>

        <div class="card">
          <div class="card-head">
            <div class="card-title">Consistency</div>
            <div class="card-note" id="adherence-note"></div>
          </div>
          <div class="card-sub">Every day in range. Darker means more sessions that day.</div>
          <div class="chart-wrap" id="heat-wrap">
            <svg class="chart" id="heat-chart" role="img" aria-labelledby="heat-desc"></svg>
            <div class="sr-only" id="heat-desc"></div>
            <div class="tip" id="heat-tip"></div>
          </div>
          <div class="legend">
            <div class="legend-item">
              <span class="legend-swatch" style="background:var(--heat-0);border:1px solid var(--border)"></span>
              <span>No session</span>
            </div>
            <div class="legend-item legend-scale">
              <span>Fewer</span>
              <span class="legend-swatch" style="background:var(--heat-1)"></span>
              <span class="legend-swatch" style="background:var(--heat-2)"></span>
              <span class="legend-swatch" style="background:var(--heat-3)"></span>
              <span>More</span>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-head">
            <div class="card-title">By exercise</div>
            <div class="card-note">Most practised first</div>
          </div>
          <div class="card-sub">Where your effort went, and where you went past the limit.</div>
          <div class="tbl-wrap">
            <table>
              <thead>
                <tr>
                  <th>Exercise</th>
                  <th class="num">Sessions</th>
                  <th class="num">Sets</th>
                  <th class="num">Reps</th>
                  <th class="num">Best angle</th>
                  <th class="num">Over limit</th>
                </tr>
              </thead>
              <tbody id="ex-body"></tbody>
            </table>
          </div>
        </div>
`;

  // ── Module state and helpers ─────────────────────────────────────────────
  // These came out of progress.html with the drawing code that uses them. The
  // chart's chosen exercise is view state, not page state: both pages that show
  // this view need it, and neither should have to declare it.
  const $ = (id) => document.getElementById(id);
  const deg = (v) => `${Math.round(v)}°`;
  const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;

  let latest = null;          // the payload most recently rendered
  let romExercise = null;     // which series the range-of-motion chart is showing
  let koosSide = null;        // which knee the questionnaire chart is showing

  /** Inject the view's markup into `root`. Call once, before render(). */
  function mount(root) {
    if (!root) throw new Error('ProgressView.mount needs an element');
    root.innerHTML = MARKUP;
  }

    // A plain "2026-09-12" is a calendar day and is read as one, never through
    // UTC - new Date('2026-09-12') would land on the 11th west of Greenwich.
    // A full timestamp (share links, created_at) is a moment, so it is placed
    // in the reader's own day. Splitting a timestamp on "-" as if it were a day
    // is what printed "Invalid Date" on the share page.
    function toDay(value) {
      const s = String(value);
      if (s.length > 10) {
        const t = new Date(s);
        return new Date(t.getFullYear(), t.getMonth(), t.getDate());
      }
      const [y, m, d] = s.split('-').map(Number);
      return new Date(y, m - 1, d);
    }

    function shortDate(iso) {
      return toDay(iso).toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
    }

    function longDate(iso) {
      return toDay(iso).toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short' });
    }

    // Every value that reaches the DOM goes through textContent. The bearer
    // token lives in localStorage on this origin, so an innerHTML path fed by
    // API data would be an account-compromise bug, not a cosmetic one.
    function el(tag, cls, text) {
      const node = document.createElement(tag);
      if (cls) node.className = cls;
      if (text !== undefined) node.textContent = text;
      return node;
    }

    const svgEl = (tag, attrs = {}) => {
      const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
      for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
      return node;
    };

    // A sprite icon. The page supplies the <symbol>.
    const icon = (id) => {
      const svg = svgEl('svg', { class: 'i', 'aria-hidden': 'true' });
      svg.appendChild(svgEl('use', { href: `#${id}` }));
      return svg;
    };

    function statValue(target, value, unit, emptyText) {
      target.textContent = '';
      if (value === null || value === undefined) {
        target.appendChild(el('div', 'kpi-none', emptyText));
        return;
      }
      const wrap = el('div', 'kpi-value', String(value));
      if (unit) {
        const u = el('span', 'kpi-unit', unit);
        wrap.appendChild(u);
      }
      target.appendChild(wrap);
    }

    // ── Range of motion: a single series, so no legend box is needed for it -
    //    the card title names it. The legend below only explains the reference
    //    line and the breach marker, which are not series.
    function drawRom(points) {
      const svg = $('rom-chart');
      svg.textContent = '';

      const W = 900, H = 260;
      const pad = { top: 18, right: 18, bottom: 30, left: 42 };
      svg.setAttribute('viewBox', `0 0 ${W} ${H}`);

      $('rom-legend').textContent = '';

      if (!points.length) {
        const note = svgEl('text', { x: W / 2, y: H / 2, 'text-anchor': 'middle', class: 'axis-text' });
        note.textContent = 'No verified measurements in this range';
        svg.appendChild(note);
        return;
      }

      const innerW = W - pad.left - pad.right;
      const innerH = H - pad.top - pad.bottom;

      const limits = points.map(p => p.angle_limit);
      const maxY = Math.max(...points.map(p => p.peak_flexion_deg), ...limits) * 1.12;
      const yMax = Math.ceil(maxY / 15) * 15;

      const x = (i) => pad.left + (points.length === 1 ? innerW / 2 : (i / (points.length - 1)) * innerW);
      const y = (v) => pad.top + innerH - (v / yMax) * innerH;

      // Recessive grid, drawn first so marks sit on top of it.
      for (let t = 0; t <= yMax; t += 15) {
        svg.appendChild(svgEl('line', { class: 'grid-line', x1: pad.left, x2: W - pad.right, y1: y(t), y2: y(t) }));
        const label = svgEl('text', { class: 'axis-text', x: pad.left - 8, y: y(t) + 3.5, 'text-anchor': 'end' });
        label.textContent = `${t}°`;
        svg.appendChild(label);
      }

      // The safe ceiling. A threshold, not a series - dashed and in axis ink so
      // it never competes with the measurement.
      const limitPath = points.map((p, i) => `${i ? 'L' : 'M'}${x(i)},${y(p.angle_limit)}`).join(' ');
      svg.appendChild(svgEl('path', { class: 'limit-line', d: limitPath }));
      const limitTag = svgEl('text', {
        class: 'limit-label', x: W - pad.right, y: y(points[points.length - 1].angle_limit) - 7, 'text-anchor': 'end',
      });
      limitTag.textContent = `Safe limit ${points[points.length - 1].angle_limit}°`;
      svg.appendChild(limitTag);

      // Area then line: the fill gives the trend weight without a second hue.
      const linePath = points.map((p, i) => `${i ? 'L' : 'M'}${x(i)},${y(p.peak_flexion_deg)}`).join(' ');
      svg.appendChild(svgEl('path', {
        class: 'rom-area',
        d: `${linePath} L${x(points.length - 1)},${y(0)} L${x(0)},${y(0)} Z`,
      }));
      svg.appendChild(svgEl('path', { class: 'rom-line', d: linePath }));

      points.forEach((p, i) => {
        const over = p.peak_flexion_deg > p.angle_limit;
        svg.appendChild(svgEl('circle', {
          class: over ? 'rom-dot-breach' : 'rom-dot',
          cx: x(i), cy: y(p.peak_flexion_deg), r: over ? 5 : 4,
        }));
      });

      // Date axis: first, middle and last only. A label under every point is
      // unreadable at 90 days and tells the reader nothing extra.
      const ticks = points.length <= 2 ? points.map((_, i) => i)
        : [0, Math.floor((points.length - 1) / 2), points.length - 1];
      for (const i of new Set(ticks)) {
        const t = svgEl('text', { class: 'axis-text', x: x(i), y: H - 10, 'text-anchor': 'middle' });
        t.textContent = shortDate(points[i].date);
        svg.appendChild(t);
      }

      // Hover layer: a crosshair and a tooltip, with a hit area far wider than
      // the 4px dot so it is usable with a trackpad.
      const cross = svgEl('line', { class: 'crosshair', y1: pad.top, y2: pad.top + innerH, opacity: 0 });
      svg.appendChild(cross);

      const tip = $('rom-tip');
      const hit = svgEl('rect', { class: 'hit', x: pad.left, y: pad.top, width: innerW, height: innerH });
      svg.appendChild(hit);

      const wrap = $('rom-wrap');
      hit.addEventListener('pointermove', (evt) => {
        const box = svg.getBoundingClientRect();
        const px = ((evt.clientX - box.left) / box.width) * W;
        let best = 0, bestDist = Infinity;
        points.forEach((_, i) => {
          const d = Math.abs(x(i) - px);
          if (d < bestDist) { bestDist = d; best = i; }
        });
        const p = points[best];

        cross.setAttribute('x1', x(best));
        cross.setAttribute('x2', x(best));
        cross.setAttribute('opacity', 1);

        tip.textContent = '';
        tip.appendChild(el('div', 'tip-title', longDate(p.date)));
        const row = el('div', 'tip-row');
        row.appendChild(el('span', null, 'Furthest bend'));
        row.appendChild(el('span', 'tip-val', deg(p.peak_flexion_deg)));
        tip.appendChild(row);
        const row2 = el('div', 'tip-row');
        row2.appendChild(el('span', null, 'Safe limit'));
        row2.appendChild(el('span', 'tip-val', deg(p.angle_limit)));
        tip.appendChild(row2);
        if (p.sessions > 1) tip.appendChild(el('div', 'tip-title', plural(p.sessions, 'session', 'sessions')));
        if (p.peak_flexion_deg > p.angle_limit) {
          const warn = el('div', 'tip-title');
          warn.append(icon('i-alert'), ' past the safe limit');
          tip.appendChild(warn);
        }

        const wrapBox = wrap.getBoundingClientRect();
        tip.style.left = `${(x(best) / W) * box.width}px`;
        tip.style.top = `${((y(p.peak_flexion_deg) / H) * box.height) - 12 + (box.top - wrapBox.top)}px`;
        tip.classList.add('on');
      });

      hit.addEventListener('pointerleave', () => {
        cross.setAttribute('opacity', 0);
        tip.classList.remove('on');
      });

      $('rom-desc').textContent =
        `Line chart of the furthest knee flexion each day over ${(latest && latest.range_days) || 90} days, ` +
        `from ${deg(points[0].peak_flexion_deg)} on ${points[0].date} to ` +
        `${deg(points[points.length - 1].peak_flexion_deg)} on ${points[points.length - 1].date}, ` +
        `against a safe limit of ${points[points.length - 1].angle_limit} degrees.`;

      // Legend: explains the two non-series marks. Identity is never colour
      // alone - each swatch is labelled.
      const legend = $('rom-legend');
      legend.textContent = '';
      const items = [
        ['Daily best', 'var(--data)', false],
        ['Past the safe limit', 'var(--breach)', false],
        ['Safe limit', 'var(--axis)', true],
      ];
      for (const [label, colour, dashed] of items) {
        const item = el('div', 'legend-item');
        const sw = el('span', 'legend-swatch');
        sw.style.background = dashed ? 'transparent' : colour;
        if (dashed) { sw.style.borderTop = `2px dashed ${colour}`; sw.style.height = '0'; sw.style.borderRadius = '0'; }
        item.appendChild(sw);
        item.appendChild(el('span', null, label));
        legend.appendChild(item);
      }
    }

    // ── How the knee feels: KOOS-JR over time ──────────────────────────────
    //
    // Drawn against a fixed 0-100 axis rather than one scaled to the data. The
    // scale is the instrument's, not this chart's: a series running 44 to 51
    // auto-scaled would show a dramatic climb, when what actually happened is
    // seven points on a hundred-point scale - inside what the questionnaire can
    // reliably tell apart.
    //
    // The reference line is the patient's own first score, which is what every
    // later one is read against. Same visual treatment as the safe-limit line on
    // the range-of-motion chart: dashed, in axis ink, never competing with the
    // measurement.
    function drawOutcome(series) {
      const svg = $('koos-chart');
      svg.textContent = '';
      $('koos-legend').textContent = '';

      const points = series.points;
      const W = 900, H = 260;
      const pad = { top: 18, right: 18, bottom: 30, left: 42 };
      svg.setAttribute('viewBox', `0 0 ${W} ${H}`);

      const innerW = W - pad.left - pad.right;
      const innerH = H - pad.top - pad.bottom;

      const x = (i) => pad.left + (points.length === 1 ? innerW / 2 : (i / (points.length - 1)) * innerW);
      const y = (v) => pad.top + innerH - (v / 100) * innerH;

      for (let t = 0; t <= 100; t += 25) {
        svg.appendChild(svgEl('line', { class: 'grid-line', x1: pad.left, x2: W - pad.right, y1: y(t), y2: y(t) }));
        const label = svgEl('text', { class: 'axis-text', x: pad.left - 8, y: y(t) + 3.5, 'text-anchor': 'end' });
        label.textContent = String(t);
        svg.appendChild(label);
      }

      // Where they started. Only worth drawing once there is something to
      // compare against it.
      if (points.length > 1) {
        svg.appendChild(svgEl('line', {
          class: 'limit-line', x1: pad.left, x2: W - pad.right, y1: y(series.baseline), y2: y(series.baseline),
        }));
        const tag = svgEl('text', {
          class: 'limit-label', x: W - pad.right, y: y(series.baseline) - 7, 'text-anchor': 'end',
        });
        tag.textContent = `Where you started · ${Math.round(series.baseline)}`;
        svg.appendChild(tag);
      }

      const linePath = points.map((p, i) => `${i ? 'L' : 'M'}${x(i)},${y(p.interval_score)}`).join(' ');
      svg.appendChild(svgEl('path', {
        class: 'rom-area',
        d: `${linePath} L${x(points.length - 1)},${y(0)} L${x(0)},${y(0)} Z`,
      }));
      svg.appendChild(svgEl('path', { class: 'rom-line', d: linePath }));

      points.forEach((p, i) => {
        svg.appendChild(svgEl('circle', { class: 'rom-dot', cx: x(i), cy: y(p.interval_score), r: 4 }));
      });

      const ticks = points.length <= 2 ? points.map((_, i) => i)
        : [0, Math.floor((points.length - 1) / 2), points.length - 1];
      for (const i of new Set(ticks)) {
        const t = svgEl('text', { class: 'axis-text', x: x(i), y: H - 10, 'text-anchor': 'middle' });
        t.textContent = shortDate(points[i].date);
        svg.appendChild(t);
      }

      const cross = svgEl('line', { class: 'crosshair', y1: pad.top, y2: pad.top + innerH, opacity: 0 });
      svg.appendChild(cross);

      const tip = $('koos-tip');
      const hit = svgEl('rect', { class: 'hit', x: pad.left, y: pad.top, width: innerW, height: innerH });
      svg.appendChild(hit);

      const wrap = $('koos-wrap');
      hit.addEventListener('pointermove', (evt) => {
        const box = svg.getBoundingClientRect();
        const px = ((evt.clientX - box.left) / box.width) * W;
        let best = 0, bestDist = Infinity;
        points.forEach((_, i) => {
          const d = Math.abs(x(i) - px);
          if (d < bestDist) { bestDist = d; best = i; }
        });
        const p = points[best];

        cross.setAttribute('x1', x(best));
        cross.setAttribute('x2', x(best));
        cross.setAttribute('opacity', 1);

        tip.textContent = '';
        tip.appendChild(el('div', 'tip-title', longDate(p.date)));
        const row = el('div', 'tip-row');
        row.appendChild(el('span', null, 'Score'));
        row.appendChild(el('span', 'tip-val', `${Math.round(p.interval_score)} / 100`));
        tip.appendChild(row);
        if (best > 0) {
          const move = p.interval_score - series.baseline;
          const row2 = el('div', 'tip-row');
          row2.appendChild(el('span', null, 'Since you started'));
          row2.appendChild(el('span', 'tip-val', `${move >= 0 ? '+' : '−'}${Math.abs(Math.round(move))}`));
          tip.appendChild(row2);
        }
        if (p.weeks_post_op !== null && p.weeks_post_op !== undefined) {
          tip.appendChild(el('div', 'tip-title', `week ${p.weeks_post_op} after surgery`));
        }

        const wrapBox = wrap.getBoundingClientRect();
        tip.style.left = `${(x(best) / W) * box.width}px`;
        tip.style.top = `${((y(p.interval_score) / H) * box.height) - 12 + (box.top - wrapBox.top)}px`;
        tip.classList.add('on');
      });

      hit.addEventListener('pointerleave', () => {
        cross.setAttribute('opacity', 0);
        tip.classList.remove('on');
      });

      $('koos-desc').textContent = points.length === 1
        ? `KOOS-JR score of ${Math.round(series.latest)} out of 100, recorded on ${points[0].date}. `
          + 'Higher is better. One answer so far, so there is no trend yet.'
        : `Line chart of KOOS-JR scores out of 100, where higher is better: `
          + `${Math.round(series.baseline)} on ${points[0].date} rising or falling to `
          + `${Math.round(series.latest)} on ${points[points.length - 1].date}. `
          + series.change_from_baseline.summary;

      const legend = $('koos-legend');
      const items = [['Your score', 'var(--data)', false]];
      if (points.length > 1) items.push(['Where you started', 'var(--axis)', true]);
      for (const [label, colour, dashed] of items) {
        const item = el('div', 'legend-item');
        const sw = el('span', 'legend-swatch');
        sw.style.background = dashed ? 'transparent' : colour;
        if (dashed) { sw.style.borderTop = `2px dashed ${colour}`; sw.style.height = '0'; sw.style.borderRadius = '0'; }
        item.appendChild(sw);
        item.appendChild(el('span', null, label));
        legend.appendChild(item);
      }
    }

    function drawOutcomePicker(seriesList) {
      const card = $('koos-card');
      const picker = $('koos-picker');
      picker.textContent = '';

      if (!seriesList.length) {
        card.classList.add('hidden');
        return;
      }
      card.classList.remove('hidden');

      // One knee needs no chooser. Two do - KOOS-JR asks about "your knee",
      // singular, and someone with two bad knees has two different answers.
      if (seriesList.length > 1) {
        if (!seriesList.some(s => s.knee_side === koosSide)) koosSide = seriesList[0].knee_side;
        for (const s of seriesList) {
          const btn = el('button', 'ex-btn', `${s.knee_side} knee`);
          btn.type = 'button';
          btn.setAttribute('aria-pressed', String(s.knee_side === koosSide));
          btn.addEventListener('click', () => {
            koosSide = s.knee_side;
            drawOutcomePicker(seriesList);
          });
          picker.appendChild(btn);
        }
      } else {
        koosSide = seriesList[0].knee_side;
      }

      const chosen = seriesList.find(s => s.knee_side === koosSide);
      $('koos-note').textContent = chosen.count === 1
        ? 'first answer'
        : `${plural(chosen.count, 'answer', 'answers')} · ${chosen.change_from_baseline.summary.toLowerCase()}`;
      drawOutcome(chosen);
    }

    // ── Consistency: a calendar heatmap. Sequential, one hue, three steps. ──
    function drawHeat(adherence, days) {
      const svg = $('heat-chart');
      svg.textContent = '';

      const byDate = new Map(adherence.map(d => [d.date, d]));
      const today = new Date();
      const cells = [];
      for (let i = days - 1; i >= 0; i--) {
        const d = new Date(today);
        d.setDate(d.getDate() - i);
        const iso = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
        cells.push({ iso, dow: d.getDay(), entry: byDate.get(iso) || null });
      }

      // Pad to a whole week so columns line up under the weekday labels.
      const lead = cells[0].dow;
      const weeks = Math.ceil((lead + cells.length) / 7);

      const CELL = 13, GAP = 3, LEFT = 26, TOP = 14;
      const W = LEFT + weeks * (CELL + GAP);
      const H = TOP + 7 * (CELL + GAP) + 4;
      svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
      svg.style.maxWidth = `${Math.max(W, 460)}px`;

      for (const [i, label] of [[1, 'Mon'], [3, 'Wed'], [5, 'Fri']]) {
        const t = svgEl('text', { class: 'heat-label', x: 0, y: TOP + i * (CELL + GAP) + CELL - 2.5 });
        t.textContent = label;
        svg.appendChild(t);
      }

      const shade = (n) => (n >= 3 ? 'var(--heat-3)' : n === 2 ? 'var(--heat-2)' : n === 1 ? 'var(--heat-1)' : 'var(--heat-0)');
      const tip = $('heat-tip');
      const wrap = $('heat-wrap');
      let lastMonth = null;
      let lastLabelCol = -2;
      let lastLabel = null;

      cells.forEach((cell, i) => {
        const slot = lead + i;
        const col = Math.floor(slot / 7);
        const row = slot % 7;
        const n = cell.entry ? cell.entry.sessions : 0;

        const rect = svgEl('rect', {
          class: 'heat-cell',
          x: LEFT + col * (CELL + GAP),
          y: TOP + row * (CELL + GAP),
          width: CELL, height: CELL,
          fill: shade(n),
        });
        if (n === 0) { rect.setAttribute('stroke', 'var(--border)'); rect.setAttribute('stroke-width', '0.75'); }
        svg.appendChild(rect);

        // Label the column a month starts in, wherever in that column it falls.
        // Keying off the row meant a month beginning on a Thursday was silently
        // never labelled.
        const month = cell.iso.slice(0, 7);
        if (month !== lastMonth) {
          lastMonth = month;
          if (col > lastLabelCol) {
            // A three-letter month is wider than one column, so a month that
            // starts in the very next column would print over the last label
            // ("JunJul"). The earlier one is only a stub of a few days at the
            // edge of the range, so it gives way.
            if (lastLabel && col === lastLabelCol + 1) lastLabel.remove();
            lastLabelCol = col;
            const t = svgEl('text', { class: 'heat-label', x: LEFT + col * (CELL + GAP), y: 8 });
            t.textContent = new Date(`${cell.iso}T00:00:00`).toLocaleDateString(undefined, { month: 'short' });
            svg.appendChild(t);
            lastLabel = t;
          }
        }

        rect.addEventListener('pointerenter', () => {
          tip.textContent = '';
          tip.appendChild(el('div', 'tip-title', longDate(cell.iso)));
          if (!cell.entry) {
            tip.appendChild(el('div', null, 'No session'));
          } else {
            const r1 = el('div', 'tip-row');
            r1.appendChild(el('span', null, 'Sessions'));
            r1.appendChild(el('span', 'tip-val', String(cell.entry.sessions)));
            tip.appendChild(r1);
            const r2 = el('div', 'tip-row');
            r2.appendChild(el('span', null, 'Sets'));
            r2.appendChild(el('span', 'tip-val', String(cell.entry.sets)));
            tip.appendChild(r2);
            if (cell.entry.completed_sessions < cell.entry.sessions) {
              tip.appendChild(el('div', 'tip-title', 'Not every set finished'));
            }
          }
          const box = svg.getBoundingClientRect();
          const wrapBox = wrap.getBoundingClientRect();
          const sx = box.width / W, sy = box.height / H;
          tip.style.left = `${(LEFT + col * (CELL + GAP) + CELL / 2) * sx}px`;
          tip.style.top = `${(TOP + row * (CELL + GAP)) * sy - 8 + (box.top - wrapBox.top)}px`;
          tip.classList.add('on');
        });
        rect.addEventListener('pointerleave', () => tip.classList.remove('on'));
      });

      const active = adherence.length;
      $('heat-desc').textContent =
        `Calendar heatmap of exercise sessions over ${days} days: ${plural(active, 'active day', 'active days')}.`;
    }

    // ── By exercise ────────────────────────────────────────────────────────
    function drawTable(rows) {
      const body = $('ex-body');
      body.textContent = '';
      const maxSessions = Math.max(...rows.map(r => r.sessions), 1);

      for (const r of rows) {
        const tr = el('tr');

        const name = el('td');
        name.appendChild(el('div', 'ex-name', r.exercise_name));
        tr.appendChild(name);

        const sess = el('td', 'num');
        const cell = el('div');
        cell.style.display = 'flex';
        cell.style.alignItems = 'center';
        cell.style.gap = '9px';
        cell.style.justifyContent = 'flex-end';
        cell.appendChild(el('span', null, String(r.sessions)));
        const track = el('span', 'bar-track');
        const fill = el('span', 'bar-fill');
        fill.style.width = `${(r.sessions / maxSessions) * 100}%`;
        track.appendChild(fill);
        cell.appendChild(track);
        sess.appendChild(cell);
        tr.appendChild(sess);

        tr.appendChild(el('td', 'num', String(r.sets)));
        tr.appendChild(el('td', 'num', String(r.reps)));

        const best = el('td', 'num');
        if (r.peak_flexion_deg === null) {
          const q = el('span', 'quiet', '-');
          q.title = 'No verified measurement: the camera angle was not confirmed';
          best.appendChild(q);
        } else {
          best.textContent = deg(r.peak_flexion_deg);
        }
        tr.appendChild(best);

        const breach = el('td', 'num');
        if (r.breach_count === 0) {
          breach.appendChild(el('span', 'quiet', 'None'));
        } else {
          // Status ships with a label, never colour alone.
          const chip = el('span', 'breach-chip');
          chip.appendChild(icon('i-alert'));
          chip.appendChild(el('span', null, `${r.breach_count}× · ${r.breach_seconds.toFixed(1)}s`));
          breach.appendChild(chip);
        }
        tr.appendChild(breach);

        body.appendChild(tr);
      }
    }

    function drawPicker(series) {
      const picker = $('ex-picker');
      picker.textContent = '';

      if (!series.length) {
        romExercise = null;
        $('rom-note').textContent = '';
        drawRom([]);
        return;
      }

      // A single exercise needs no chooser - the card title already names it.
      if (series.length > 1) {
        if (!series.some(r => r.exercise_name === romExercise)) romExercise = series[0].exercise_name;
        for (const r of series) {
          const btn = el('button', 'ex-btn', r.exercise_name);
          btn.type = 'button';
          btn.setAttribute('aria-pressed', String(r.exercise_name === romExercise));
          btn.addEventListener('click', () => {
            romExercise = r.exercise_name;
            drawPicker(series);
          });
          picker.appendChild(btn);
        }
      } else {
        romExercise = series[0].exercise_name;
      }

      const chosen = series.find(r => r.exercise_name === romExercise);
      $('rom-note').textContent =
        `${chosen.exercise_name} · ${plural(chosen.days_measured, 'day measured', 'days measured')}`;
      drawRom(chosen.points);
    }

    // ── Render ─────────────────────────────────────────────────────────────
    function render(data) {
      latest = data;
      const s = data.summary;

      statValue($('kpi-latest'), s.latest_flexion_deg === null ? null : Math.round(s.latest_flexion_deg), '°', 'No data');
      // Named, because "latest flexion" means nothing without knowing which
      // exercise produced it.
      $('kpi-latest-foot').textContent = s.primary_exercise || 'nothing measured yet';

      statValue($('kpi-best'), s.best_flexion_deg === null ? null : Math.round(s.best_flexion_deg), '°', 'No data');
      $('kpi-best-foot').textContent = s.best_flexion_deg === null
        ? 'needs a verified camera angle'
        : `over ${plural(data.range_days, 'day', 'days')}`;

      statValue($('kpi-streak'), s.current_streak_days, s.current_streak_days === 1 ? ' day' : ' days', '-');
      $('kpi-streak-foot').textContent = s.longest_streak_days > s.current_streak_days
        ? `best run ${plural(s.longest_streak_days, 'day', 'days')}`
        : 'your best run so far';

      statValue($('kpi-sessions'), s.sessions, '', '0');
      $('kpi-sessions-foot').textContent = `${plural(s.active_days, 'active day', 'active days')} · ${plural(s.sets, 'set', 'sets')}`;

      // The patient's own verdict. Shown only once there is one - a "No data"
      // tile for a questionnaire nobody has been offered reads as broken.
      const outcomes = data.outcome_measures || [];
      const koosTile = $('kpi-koos-tile');
      if (s.latest_outcome_score === null || s.latest_outcome_score === undefined) {
        koosTile.classList.add('hidden');
      } else {
        koosTile.classList.remove('hidden');
        statValue($('kpi-koos'), Math.round(s.latest_outcome_score), '/100', 'No data');

        // The band, not the number again. "58" means nothing to someone reading
        // it for the first time; "fair" is the part they can act on.
        const newest = outcomes.length
          ? outcomes.reduce((a, b) => (Date.parse(a.latest_at) > Date.parse(b.latest_at) ? a : b))
          : null;
        const moved = newest && newest.count > 1 ? newest.change_from_baseline : null;

        let foot = s.outcome_band;
        if (!moved) {
          foot += ' · your first answer';
        } else if (moved.direction === 'unchanged') {
          // The server has already decided this is inside the questionnaire's
          // own noise. Rendering it as "+3" would invite a conclusion the
          // instrument cannot support.
          foot += ' · about where you started';
        } else {
          foot += ` · ${Math.abs(Math.round(moved.delta))} `
            + `${moved.direction === 'declined' ? 'lower' : 'higher'} than when you started`;
        }
        $('kpi-koos-foot').textContent = foot;
      }

      drawPicker(data.rom_by_exercise);
      drawOutcomePicker(outcomes);
      $('adherence-note').textContent = `${plural(s.active_days, 'active day', 'active days')} of ${data.range_days}`;

      const unverified = $('unverified-notice');
      if (s.unverified_sessions > 0) {
        $('unverified-text').textContent =
          `${plural(s.unverified_sessions, 'session was', 'sessions were')} recorded with the camera angle ` +
          `unverified. They count towards your consistency, but their angles are left out of the range-of-motion chart. ` +
          `A knee filmed square-on to the camera reads much straighter than it really is, so including them ` +
          `would show progress you have not actually made.`;
        unverified.classList.remove('hidden');
      } else {
        unverified.classList.add('hidden');
      }

      drawHeat(data.adherence, data.range_days);
      drawTable(data.by_exercise);
    }


  /**
   * Re-render whatever was last rendered.
   *
   * Used on resize: the tooltips and hit areas are positioned from the rendered
   * box, so a stale layout puts them in the wrong place. The payload lives in
   * this module, so the redraw has to as well.
   */
  function redraw() {
    if (latest) render(latest);
  }

  global.ProgressView = { mount, render, redraw, shortDate, longDate };
})(typeof window !== 'undefined' ? window : globalThis);
