/* TickTock demo front end. Plain JS, hand-drawn SVG charts, no build step. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const S = {
    info: null, dates: [], ticker: "NVDA", side: "buy", dist: 2.0, target: null, asof: null,
    model: null, ref: null, history: null, pred: null, playing: null, lockTarget: false,
  };
  const SVGNS = "http://www.w3.org/2000/svg";
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  // ---------- formatting ----------
  const pct = (p) => (p == null ? "–" : p < 0.005 ? "<1%" : p > 0.995 ? ">99%" : `${Math.round(p * 100)}%`);
  const acc = (p) => (p == null ? "–" : `${(p * 100).toFixed(1)}%`);
  const usd = (x) => `$${Number(x).toFixed(2)}`;
  const H = () => S.info.horizon;
  function days(x, short = false) {
    if (x == null) return short ? `${H()}+` : `${H()}+ days`;
    if (x < 1) return short ? "<1" : "next session";
    const n = Math.max(1, Math.round(x));
    return short ? String(n) : `${n} day${n === 1 ? "" : "s"}`;
  }
  function rangeText(lo, hi) {
    if (lo == null) return `Not within ${H()} days`;
    if (hi == null) return `${lo < 1 ? 1 : Math.round(lo)}–${H()}+ days`;
    const a = lo < 1 ? 1 : Math.round(lo), b = Math.max(a, Math.round(hi));
    return a === b ? `${a} day${a === 1 ? "" : "s"}` : `${a}–${b} days`;
  }

  // ---------- data ----------
  async function api(path, params) {
    const q = params ? "?" + new URLSearchParams(params) : "";
    const r = await fetch(`/api/${path}${q}`);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    return r.json();
  }

  let reqId = 0;
  async function refresh({ keep = "dist" } = {}) {
    const id = ++reqId;
    document.querySelector(".grid").classList.add("loading");
    try {
      const hist = await api("history", { ticker: S.ticker, as_of: S.asof });
      if (id !== reqId) return;
      S.history = hist;
      S.ref = hist.past[hist.past.length - 1].close;
      const sgn = S.side === "buy" ? -1 : 1;
      if (keep === "dist" || S.target == null) {
        S.target = +(S.ref * (1 + sgn * S.dist / 100)).toFixed(2);
      } else {
        S.dist = Math.max(0, sgn * (S.target / S.ref - 1) * 100);
      }
      syncInputs();
      const pred = await api("predict", { ticker: S.ticker, side: S.side, target: S.target, as_of: S.asof, model: S.model });
      if (id !== reqId) return;
      S.pred = pred;
      $("error").hidden = true;
      render();
    } catch (e) {
      $("error").textContent = `Could not load forecast: ${e.message}`;
      $("error").hidden = false;
      stopPlay();
    } finally {
      if (id === reqId) document.querySelector(".grid").classList.remove("loading");
    }
  }

  function syncInputs() {
    $("target").value = S.target.toFixed(2);
    $("dist").value = Math.min(12, Math.max(0.1, S.dist)).toFixed(1);
    $("dist-label").textContent = `${S.side === "buy" ? "−" : "+"}${S.dist.toFixed(2)}% (${usd(S.target)})`;
    $("asof").value = S.history ? S.history.as_of : S.asof;
  }

  // ---------- render ----------
  function render() {
    const p = S.pred;
    $("hero-ticker").textContent = `${p.side === "buy" ? "Buy" : "Sell"} ${p.ticker}`;
    $("hero-ref").textContent = `last close ${usd(p.reference_price)} on ${p.as_of} · target ${usd(p.target)} (${p.distance_pct > 0 ? "+" : ""}${p.distance_pct.toFixed(2)}%)`;
    if (p.marketable) {
      $("hero-range").textContent = "Fills now";
      $("hero-sub").textContent = "Your price is at or through the market.";
    } else {
      $("hero-range").textContent = rangeText(p.q10, p.q90);
      const half = rangeText(p.q25, p.q75);
      const med = p.q50 == null ? `more than ${H()} trading days` : days(p.q50);
      const when = p.calendar.q50 ? ` (around ${p.calendar.q50})` : "";
      $("hero-sub").textContent = `Typical wait ${med}${when}. 50% range: ${half}. Trading days.`;
    }
    $("briefing").textContent = p.briefing;
    const tiles = [["1", "next session"], ["5", "within 5 days"], ["10", "within 10 days"], ["20", "within 20 days"]]
      .map(([k, l]) => `<div class="tile"><div class="v">${pct(p.p_within[k])}</div><div class="k">${l}</div></div>`);
    tiles.push(`<div class="tile"><div class="v">${pct(p.p_beyond_horizon)}</div><div class="k">not within ${H()} days</div></div>`);
    $("tiles").innerHTML = tiles.join("");
    const t = p.model_test;
    $("model-acc").innerHTML = t ? `<span class="acc-big">${acc(t.accuracy)}</span>
      <span><b>accuracy of this model</b> on ${t.n.toLocaleString()} held-out test orders (${S.info.splits.test}): right/wrong calls on "fills within 1, 5, 10, 20 days".
      Always guessing the common answer scores ${acc(t.baseline)}. The real wait landed inside its 80% range ${acc(t.range80_hit)} of the time.</span>
      ${p.in_sample ? `<span class="warn-note">This replay date is inside the training period, so the model has seen it.</span>` : ""}` : "";

    renderGauges(p);
    renderActual(p);
    renderCdf(p);
    renderTradeoff(p);
    renderPrice(p);
    renderDrivers(p);
    renderRisk(p);
  }

  function gaugeHTML(title, valueText, level, frac, ticks, color) {
    const f = Math.max(0, Math.min(1, frac));
    return `<div class="row"><span>${title}</span><span class="status"><span class="dot" style="background:${color}"></span>${level}</span></div>
      <div class="bar"><div class="needle" style="left:${(f * 100).toFixed(1)}%"></div></div>
      <div class="ticks">${ticks.map((t) => `<span>${t}</span>`).join("")}</div>
      <div class="muted small" style="margin-top:4px">${valueText}</div>`;
  }

  function renderGauges(p) {
    const t = p.traffic;
    const tColor = t.level === "busy" ? css("--series-2") : t.level === "quiet" ? css("--muted") : css("--accent");
    $("traffic-gauge").innerHTML = gaugeHTML(`${p.ticker} trading traffic`,
      `Last 5 days' volume is ${t.ratio.toFixed(2)}× this stock's usual (higher than ${t.percentile.toFixed(0)}% of its past days).`,
      t.level, t.percentile / 100, ["quiet", "normal", "busy"], tColor);
    const m = p.market;
    const mColor = m.level === "stressed" ? css("--bad") : m.level === "calm" ? css("--good") : css("--accent");
    $("stress-gauge").innerHTML = gaugeHTML("Whole-market stress", `Market volatility z-score ${m.stress.toFixed(2)} vs its own history${m.fomc_next10 ? " · Fed meeting in the next 10 sessions" : ""}.`,
      m.level, (m.stress + 2) / 6, ["calm", "normal", "stressed"], mColor);
  }

  function renderActual(p) {
    const a = p.actual;
    if (!a) { $("actual").innerHTML = "No later data: this is a live-style forecast."; return; }
    if (!$("reveal").checked) {
      $("actual").innerHTML = `Replay: what really happened is hidden. Tick <b>Reveal</b> on the price chart to check the forecast.`;
      return;
    }
    if (a.filled) {
      const inside = p.q10 != null && a.days >= Math.floor(p.q10) && (p.q90 == null || a.days <= Math.ceil(p.q90));
      $("actual").innerHTML = `Actually filled after <b>${a.days} trading day${a.days === 1 ? "" : "s"}</b> (${a.date}). ${inside ? "Inside" : "Outside"} the 80% range.`;
    } else {
      $("actual").innerHTML = `Did <b>not</b> fill in the ${a.observed_days} trading days observed.`;
    }
  }

  // ---------- svg helpers ----------
  const cw = (id) => Math.max(300, Math.round($(id).clientWidth || 760));
  function svg(el, w, h) {
    el.innerHTML = "";
    const s = document.createElementNS(SVGNS, "svg");
    s.setAttribute("viewBox", `0 0 ${w} ${h}`);
    s.setAttribute("role", "img");
    el.appendChild(s);
    return s;
  }
  function add(parent, tag, attrs, text) {
    const e = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    parent.appendChild(e);
    return e;
  }
  function scale(d0, d1, r0, r1) {
    const f = (x) => r0 + ((x - d0) / (d1 - d0 || 1)) * (r1 - r0);
    f.inv = (y) => d0 + ((y - r0) / (r1 - r0 || 1)) * (d1 - d0);
    return f;
  }
  function niceTicks(lo, hi, n = 5) {
    const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) || step0;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function axes(s, x, y, xt, yt, fx, fy, M, W, Hh) {
    const g = add(s, "g", { class: "axis" });
    yt.forEach((v) => {
      add(g, "line", { class: "gridline", x1: M.l, x2: W - M.r, y1: y(v), y2: y(v) });
      add(g, "text", { x: M.l - 8, y: y(v) + 4, "text-anchor": "end" }, fy(v));
    });
    xt.forEach((v) => add(g, "text", { x: x(v), y: Hh - M.b + 16, "text-anchor": "middle" }, fx(v)));
    add(g, "line", { class: "baseline", x1: M.l, x2: W - M.r, y1: Hh - M.b, y2: Hh - M.b });
  }
  const tip = $("tooltip");
  function showTip(evt, html) {
    tip.innerHTML = html; tip.hidden = false;
    const pad = 14, r = tip.getBoundingClientRect();
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - pad;
    if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - pad;
    tip.style.left = `${x}px`; tip.style.top = `${y}px`;
  }
  const hideTip = () => { tip.hidden = true; };
  function overlay(s, M, W, Hh, onMove) {
    const r = add(s, "rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: Hh - M.t - M.b, fill: "transparent" });
    const toLocal = (evt) => {
      const pt = s.createSVGPoint(); pt.x = evt.clientX; pt.y = evt.clientY;
      return pt.matrixTransform(s.getScreenCTM().inverse());
    };
    r.addEventListener("pointermove", (e) => onMove(e, toLocal(e)));
    r.addEventListener("pointerleave", () => { hideTip(); onMove(null); });
    return r;
  }

  // ---------- charts ----------
  function renderCdf(p) {
    const W = cw("cdf-chart"), Hh = 260, M = { l: 44, r: W < 500 ? 70 : 90, t: 12, b: 28 };
    const s = svg($("cdf-chart"), W, Hh);
    const n = p.cdf.length - 1;
    const x = scale(0, n, M.l, W - M.r), y = scale(0, 1, Hh - M.b, M.t);
    axes(s, x, y, niceTicks(0, n, 8), [0, 0.25, 0.5, 0.75, 1], (v) => v, (v) => `${v * 100}%`, M, W, Hh);
    const c1 = css("--series-1");
    // 80% / 50% bands on the time axis
    const band = (lo, hi, op) => {
      if (lo == null) return;
      const a = x(Math.max(0, lo)), b = x(hi == null ? n : Math.min(n, hi));
      add(s, "rect", { x: a, y: M.t, width: Math.max(1, b - a), height: Hh - M.t - M.b, fill: c1, opacity: op });
    };
    band(p.q10, p.q90, 0.07); band(p.q25, p.q75, 0.1);
    const pts = p.cdf.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`);
    add(s, "path", { d: `M${pts.join("L")}L${x(n)},${y(0)}L${x(0)},${y(0)}Z`, fill: c1, opacity: 0.08 });
    add(s, "path", { d: `M${pts.join("L")}`, fill: "none", stroke: c1, "stroke-width": 2, "stroke-linejoin": "round" });
    add(s, "text", { class: "direct-label", x: x(n) + 6, y: y(p.cdf[n]) + 4 }, `${pct(p.cdf[n])} by day ${n}`);
    if (p.q50 != null) {
      add(s, "line", { x1: x(p.q50), x2: x(p.q50), y1: y(0.5), y2: Hh - M.b, stroke: css("--text-2"), "stroke-dasharray": "3 3" });
      add(s, "text", { class: "direct-label", x: x(p.q50) + 5, y: y(0.5) - 6 }, `median ${days(p.q50)}`);
    }
    if ($("reveal").checked && p.actual && p.actual.filled) {
      const ax = x(p.actual.days);
      add(s, "line", { x1: ax, x2: ax, y1: M.t, y2: Hh - M.b, stroke: css("--series-2"), "stroke-width": 2 });
      add(s, "text", { class: "direct-label", x: ax + 5, y: M.t + 12 }, `actual fill: day ${p.actual.days}`);
    }
    add(s, "text", { class: "lbl-text", x: W - M.r, y: Hh - 2, "text-anchor": "end" }, "trading days");
    const cross = add(s, "line", { y1: M.t, y2: Hh - M.b, stroke: css("--muted"), visibility: "hidden" });
    const dot = add(s, "circle", { r: 4.5, fill: c1, stroke: css("--surface"), "stroke-width": 2, visibility: "hidden" });
    overlay(s, M, W, Hh, (e, pt) => {
      if (!e) { cross.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); return; }
      const i = Math.max(0, Math.min(n, Math.round(x.inv(pt.x))));
      cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
      dot.setAttribute("cx", x(i)); dot.setAttribute("cy", y(p.cdf[i])); dot.setAttribute("visibility", "visible");
      showTip(e, `<div class="t">Day ${i}</div><b>${pct(p.cdf[i])}</b> chance filled by then`);
    });
  }

  function renderTradeoff(p) {
    const W = cw("tradeoff-chart"), Hh = 270, M = { l: 44, r: 20, t: 14, b: 40 };
    const s = svg($("tradeoff-chart"), W, Hh);
    const t = p.tradeoff, cap = H();
    const xs = t.map((d) => d.distance_pct);
    const x = scale(Math.min(...xs), Math.max(...xs), M.l, W - M.r), y = scale(0, cap, Hh - M.b, M.t);
    const capd = (v) => (v == null ? cap : Math.min(cap, v));
    axes(s, x, y, (W < 500 ? [1, 2, 4, 6, 8, 10] : [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).filter((v) => v <= Math.max(...xs)), niceTicks(0, cap, 4),
      (v) => `${p.side === "buy" ? "−" : "+"}${v}%`, (v) => (v === cap ? `${v}+` : v), M, W, Hh);
    const c1 = css("--series-1");
    const top = t.map((d) => `${x(d.distance_pct)},${y(capd(d.q90))}`);
    const bot = t.slice().reverse().map((d) => `${x(d.distance_pct)},${y(capd(d.q10))}`);
    add(s, "path", { d: `M${top.join("L")}L${bot.join("L")}Z`, fill: c1, opacity: 0.15 });
    add(s, "path", { d: `M${t.map((d) => `${x(d.distance_pct)},${y(capd(d.q50))}`).join("L")}`, fill: "none", stroke: c1, "stroke-width": 2 });
    add(s, "text", { class: "lbl-text", x: (M.l + W - M.r) / 2, y: Hh - 4, "text-anchor": "middle" }, `limit price distance from last close (${p.side === "buy" ? "below" : "above"})`);
    add(s, "text", { class: "lbl-text", x: M.l + 4, y: M.t + 2 }, "wait (trading days)");
    // current order marker
    const cx = Math.abs(p.distance_pct);
    if (!p.marketable && cx >= Math.min(...xs) && cx <= Math.max(...xs)) {
      add(s, "line", { x1: x(cx), x2: x(cx), y1: M.t, y2: Hh - M.b, stroke: css("--text-2"), "stroke-dasharray": "3 3" });
      add(s, "circle", { cx: x(cx), cy: y(capd(p.q50)), r: 5, fill: css("--series-2"), stroke: css("--surface"), "stroke-width": 2 });
      add(s, "text", { class: "direct-label", x: x(cx) + 7, y: y(capd(p.q50)) - 8 }, `your order ${usd(p.target)}`);
    }
    const dot = add(s, "circle", { r: 4.5, fill: c1, stroke: css("--surface"), "stroke-width": 2, visibility: "hidden" });
    overlay(s, M, W, Hh, (e, pt) => {
      if (!e) { dot.setAttribute("visibility", "hidden"); return; }
      const v = x.inv(pt.x);
      const d = t.reduce((a, b) => (Math.abs(b.distance_pct - v) < Math.abs(a.distance_pct - v) ? b : a));
      dot.setAttribute("cx", x(d.distance_pct)); dot.setAttribute("cy", y(capd(d.q50))); dot.setAttribute("visibility", "visible");
      showTip(e, `<div class="t">${usd(d.price)} (${p.side === "buy" ? "−" : "+"}${d.distance_pct}%)</div>
        median <b>${days(d.q50)}</b><br>80% range ${rangeText(d.q10, d.q90)}<br>${pct(d.p10)} within 10 days`);
    });
    // click to pick a price
    s.style.cursor = "pointer";
    s.addEventListener("click", (e) => {
      const pt = s.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
      const v = x.inv(pt.matrixTransform(s.getScreenCTM().inverse()).x);
      const d = t.reduce((a, b) => (Math.abs(b.distance_pct - v) < Math.abs(a.distance_pct - v) ? b : a));
      S.dist = d.distance_pct; refresh({ keep: "dist" });
    });
  }

  function renderPrice(p) {
    const h = S.history; if (!h) return;
    const reveal = $("reveal").checked;
    const rows = reveal ? h.past.concat(h.future) : h.past;
    const W = cw("price-chart"), Hh = 260, M = { l: 54, r: W < 500 ? 80 : 96, t: 12, b: 28 };
    const s = svg($("price-chart"), W, Hh);
    const total = h.past.length + h.future.length;
    const x = scale(0, Math.max(total - 1, 1), M.l, W - M.r);
    const lows = rows.map((r) => r.low).concat([p.target]), highs = rows.map((r) => r.high).concat([p.target]);
    const lo = Math.min(...lows), hi = Math.max(...highs), padv = (hi - lo) * 0.06;
    const y = scale(lo - padv, hi + padv, Hh - M.b, M.t);
    const iAsof = h.past.length - 1;
    const xt = [0, Math.floor(iAsof / 2), iAsof].concat(reveal && h.future.length ? [total - 1] : []);
    const lab = (i) => (rows[i] ? rows[i].date.slice(5) : "");
    axes(s, x, y, xt, niceTicks(lo - padv, hi + padv, 4), lab, (v) => `$${v.toFixed(v < 100 ? 1 : 0)}`, M, W, Hh);
    const muted = css("--muted"), c1 = css("--series-1"), c2 = css("--series-2");
    rows.forEach((r, i) => add(s, "line", { x1: x(i), x2: x(i), y1: y(r.low), y2: y(r.high), stroke: muted, "stroke-width": 1.5, opacity: 0.55 }));
    const line = (arr, off, color) => add(s, "path", { d: `M${arr.map((r, i) => `${x(i + off).toFixed(1)},${y(r.close).toFixed(1)}`).join("L")}`, fill: "none", stroke: color, "stroke-width": 2 });
    line(h.past, 0, c1);
    if (reveal && h.future.length) line([h.past[iAsof]].concat(h.future), iAsof, c2);
    add(s, "line", { x1: x(iAsof), x2: x(iAsof), y1: M.t, y2: Hh - M.b, stroke: css("--text-2"), "stroke-dasharray": "2 3" });
    add(s, "text", { class: "direct-label", x: x(iAsof) - 4, y: M.t + 10, "text-anchor": "end" }, "order placed");
    add(s, "line", { x1: x(iAsof), x2: W - M.r, y1: y(p.target), y2: y(p.target), stroke: css("--text"), "stroke-width": 1.5, "stroke-dasharray": "5 4" });
    add(s, "text", { class: "direct-label", x: W - M.r + 6, y: y(p.target) + 4 }, `target ${usd(p.target)}`);
    if (reveal && p.actual && p.actual.filled) {
      const i = iAsof + p.actual.days;
      add(s, "circle", { cx: x(i), cy: y(p.target), r: 6, fill: c2, stroke: css("--surface"), "stroke-width": 2 });
      add(s, "text", { class: "direct-label", x: x(i) + 8, y: y(p.target) - 8 }, `filled ${p.actual.date}`);
    }
    if (reveal && h.future.length) add(s, "text", { class: "direct-label", x: W - M.r + 6, y: y(h.future[h.future.length - 1].close) + 4 }, "what happened");
    overlay(s, M, W, Hh, (e, pt) => {
      if (!e) return;
      const i = Math.max(0, Math.min(rows.length - 1, Math.round(x.inv(pt.x))));
      const r = rows[i];
      showTip(e, `<div class="t">${r.date}${i > iAsof ? " · after order" : ""}</div>close <b>${usd(r.close)}</b><br>low ${usd(r.low)} · high ${usd(r.high)}`);
    });
  }

  function renderDrivers(p) {
    const d = p.drivers || [];
    const maxAbs = Math.max(0.2, ...d.map((x) => Math.abs(x.log_effect)));
    $("drivers").innerHTML = d.map((x) => {
      const w = (Math.abs(x.log_effect) / maxAbs) * 50;
      const longer = x.log_effect > 0;
      const style = longer ? `left:50%;width:${w}%;background:${css("--series-2")}` : `left:${50 - w}%;width:${w}%;background:${css("--series-1")}`;
      return `<div class="drv"><span>${x.driver}</span><span class="track"><span class="mid"></span><span class="fill" style="${style}"></span></span><span class="num">×${x.multiplier.toFixed(2)}</span></div>`;
    }).join("") + `<p class="muted small" style="margin-top:8px"><span style="color:${css("--series-2")}">■</span> makes the wait longer · <span style="color:${css("--series-1")}">■</span> makes it shorter</p>`;
  }

  function renderRisk(p) {
    const r = p.risk;
    const name = (t) => t.replace(/_/g, " ").replace("mna", "M&A").replace("ai tech", "AI / tech");
    const mult = r.wait_multiplier != null
      ? `<p class="small">Combined effect of risk themes and sector on the typical wait: <b>×${r.wait_multiplier.toFixed(2)}</b> (${r.wait_multiplier >= 1 ? "longer" : "shorter"} than an average company).</p>` : "";
    const other = r.other_year ? `<div class="risk-year">${r.other_year.year}</div><div class="risk-text muted">${r.other_year.text}</div>` : "";
    $("risk").innerHTML = `<div class="risk-year">${r.year} · ${r.sector.replace(/_/g, " ")} · macro exposure ${r.macro_exposure}/4</div>
      <div class="risk-text">${r.text}</div>
      <div class="chips">${r.themes.map((t) => `<span class="chip" title="matched: ${t.phrases.join(", ")}">${name(t.theme)}</span>`).join("")}</div>
      ${mult}${other}`;
  }

  async function renderScorecard() {
    const sc = await api("scorecard");
    const label = Object.fromEntries(S.info.models.map((m) => [m.id, m.label]));
    const sp = S.info.splits;
    $("score-test-title").textContent = `Full scorecard on the test period, ${sp.test} (held out, scored once)`;
    $("score-tune-title").textContent = `Tune set, ${sp.tune}, used for model choices`;
    const test = sc.test_scorecard || [];
    const order = test.slice().sort((a, b) => b.accuracy - a.accuracy);
    const n = test.length ? test[0].n : 0;
    $("acc-intro").textContent = `Each model was trained on ${sp.train}, then scored once on ${n.toLocaleString()} orders from ${sp.test} that it never saw. Accuracy = share of correct yes/no answers to "will this order fill within 1, 5, 10 and 20 trading days?" (the model says yes when its probability is at least 50%).`;
    $("acc-cards").innerHTML = order.map((r, i) => `<div class="acc-card ${i === 0 ? "best" : ""}">
        <div class="acc-name">${label[r.model] || r.model}${i === 0 ? ' <span class="pill">most accurate</span>' : ""}</div>
        <div class="acc-big">${acc(r.accuracy)}</div>
        <div class="acc-bar"><div style="width:${(r.accuracy * 100).toFixed(1)}%"></div><i style="left:${(r.accuracy_base * 100).toFixed(1)}%" title="baseline"></i></div>
        <div class="small muted">+${((r.accuracy - r.accuracy_base) * 100).toFixed(1)} pts over the ${acc(r.accuracy_base)} baseline</div>
        <div class="acc-days small">${[1, 5, 10, 20].map((t) => `<span>≤${t}d <b>${acc(r[`acc@${t}d`])}</b></span>`).join("")}</div>
        <div class="small">80% range hit <b>${acc(r["cov80%"])}</b> <span class="muted">(target 80%)</span></div>
      </div>`).join("") || "<p>Run the pipeline first.</p>";
    $("acc-foot").textContent = "Baseline: always give the most common answer for each question (the tick on each bar). Range hit should be close to its target; a model that always predicted very wide ranges would score near 100% there and be useless.";
    const tbl = (rows) => {
      if (!rows) return "<tr><td>Run the pipeline first.</td></tr>";
      const best = Math.max(...rows.map((r) => r.accuracy));
      return `<tr><th>Model</th><th>Accuracy</th><th>Baseline</th><th>Acc ≤1d</th><th>Acc ≤5d</th><th>Acc ≤10d</th><th>Acc ≤20d</th><th>50% range hit</th><th>80% range hit</th><th>Brier ≤5d</th><th>Brier ≤20d</th><th>Censored LL</th><th>ms / pred</th></tr>` +
        rows.map((r) => `<tr class="${r.accuracy === best ? "sel" : ""}"><td>${label[r.model] || r.model}</td><td>${acc(r.accuracy)}</td><td>${acc(r.accuracy_base)}</td>
          <td>${acc(r["acc@1d"])}</td><td>${acc(r["acc@5d"])}</td><td>${acc(r["acc@10d"])}</td><td>${acc(r["acc@20d"])}</td><td>${acc(r["cov50%"])}</td><td>${acc(r["cov80%"])}</td>
          <td>${r["brier@5d"].toFixed(4)}</td><td>${r["brier@20d"].toFixed(4)}</td><td>${r.censored_ll.toFixed(4)}</td><td>${r.ms_per_pred.toFixed(4)}</td></tr>`).join("");
    };
    $("score-test").innerHTML = tbl(sc.test_scorecard);
    $("score-tune").innerHTML = tbl(sc.tune_scorecard);
  }

  // ---------- models vs reality ----------
  const C = { data: null, init: false };
  const MODEL_COLORS = ["--m1", "--m2", "--m3", "--m4", "--m5"];
  const modelColor = (name) => css(MODEL_COLORS[S.info.models.findIndex((m) => m.id === name) % 5]);
  const modelLabel = (name) => (S.info.models.find((m) => m.id === name) || { label: name }).label;
  const pts = (v) => (v == null ? "–" : `${(v * 100).toFixed(1)} pts`);

  async function initReality() {
    if (C.init) return loadReality();
    C.init = true;
    $("cmp-ticker").innerHTML = `<option value="ALL">All 20 stocks</option>` +
      S.info.tickers.map((t) => `<option value="${t.ticker}">${t.ticker}</option>`).join("");
    $("cmp-ticker").value = S.ticker;
    $("cmp-dist").innerHTML = `<option value="ALL">All distances (0.5% to 8%)</option>` +
      ["0.5", "1", "2", "3", "5", "8"].map((d) => `<option value="${d}">${d}% from the close</option>`).join("");
    ["cmp-ticker", "cmp-side", "cmp-dist"].forEach((id) => $(id).addEventListener("change", loadReality));
    await loadReality();
  }

  async function loadReality() {
    try {
      C.data = await api("comparison", { ticker: $("cmp-ticker").value, side: $("cmp-side").value, dist: $("cmp-dist").value });
      $("error").hidden = true;
      renderReality();
    } catch (e) {
      $("error").textContent = `Could not load comparison: ${e.message}`; $("error").hidden = false;
    }
  }

  function renderReality() {
    const d = C.data;
    if (!d) return;
    const who = d.ticker === "ALL" ? "all 20 stocks" : d.ticker;
    $("cmp-note").textContent = `Every hypothetical limit order for ${who} placed between ${d.period[0]} and ${d.period[1]}. The models never saw these orders: they were trained on 2006–2021 data only.`;
    $("cmp-n").textContent = `${d.curve.n.toLocaleString()} orders`;
    const models = Object.keys(d.curve.models);
    $("cmp-legend").innerHTML = `<span><i class="actual" style="background:${css("--text")}"></i><b>Real outcome</b></span>` +
      models.map((m) => `<span><i style="background:${modelColor(m)}"></i>${modelLabel(m)}</span>`).join("");
    $("cmp-tl-title").textContent = `Month by month: chance of filling within ${d.timeline_days} trading days`;
    renderCmpCurve(d, models);
    renderCmpTimeline(d, models);
    renderCmpPrice(d);
    const rows = models.map((m) => ({ m, c: d.curve.models[m], t: d.timeline.models[m] }))
      .sort((a, b) => a.c.mean_gap - b.c.mean_gap);
    $("cmp-table").innerHTML = `<tr><th>Model</th><th>Fill curve: avg gap</th><th>Biggest gap</th><th>Monthly ≤${d.timeline_days}d: avg gap</th><th>Held-out accuracy</th></tr>` +
      rows.map((r, i) => `<tr class="${i === 0 ? "sel" : ""}"><td><span class="swatch" style="background:${modelColor(r.m)}"></span>${modelLabel(r.m)}${i === 0 ? " · closest" : ""}</td>
        <td>${pts(r.c.mean_gap)}</td><td>${pts(r.c.max_gap)} on day ${r.c.max_gap_day}</td><td>${pts(r.t.mean_gap)}</td>
        <td>${acc((S.info.models.find((x) => x.id === r.m) || {}).test?.accuracy)}</td></tr>`).join("");
  }

  function renderCmpCurve(d, models) {
    const W = cw("cmp-curve"), Hh = 300, M = { l: 44, r: 16, t: 12, b: 28 };
    const s = svg($("cmp-curve"), W, Hh);
    const n = d.horizon;
    const all = [d.curve.actual, ...models.map((m) => d.curve.models[m].curve)];
    const ymax = Math.min(1, Math.ceil(Math.max(...all.flat()) * 10 + 0.5) / 10);
    const x = scale(0, n, M.l, W - M.r), y = scale(0, ymax, Hh - M.b, M.t);
    axes(s, x, y, niceTicks(0, n, 8), niceTicks(0, ymax, 5), (v) => v, (v) => `${Math.round(v * 100)}%`, M, W, Hh);
    const line = (arr, color, wdt) => add(s, "path", { d: "M" + arr.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("L"),
      fill: "none", stroke: color, "stroke-width": wdt, "stroke-linejoin": "round" });
    models.forEach((m) => line(d.curve.models[m].curve, modelColor(m), 1.8));
    line(d.curve.actual, css("--text"), 3.2);
    add(s, "text", { class: "lbl-text", x: W - M.r, y: Hh - 2, "text-anchor": "end" }, "trading days after the order");
    const cross = add(s, "line", { y1: M.t, y2: Hh - M.b, stroke: css("--muted"), visibility: "hidden" });
    overlay(s, M, W, Hh, (e, pt) => {
      if (!e) { cross.setAttribute("visibility", "hidden"); return; }
      const i = Math.max(1, Math.min(n, Math.round(x.inv(pt.x))));
      cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
      const a = d.curve.actual[i];
      showTip(e, `<div class="t">Filled by day ${i}</div><b>Real: ${pct(a)}</b><br>` + models.map((m) => {
        const v = d.curve.models[m].curve[i], g = (v - a) * 100;
        return `<span class="swatch" style="background:${modelColor(m)}"></span>${modelLabel(m)}: ${pct(v)} <span class="muted">(${g >= 0 ? "+" : ""}${g.toFixed(1)} pts)</span>`;
      }).join("<br>"));
    });
  }

  function renderCmpTimeline(d, models) {
    const W = cw("cmp-timeline"), Hh = 260, M = { l: 44, r: 16, t: 12, b: 28 };
    const s = svg($("cmp-timeline"), W, Hh);
    const months = d.months, n = months.length - 1;
    const series = [d.timeline.actual, ...models.map((m) => d.timeline.models[m].series)].flat().filter((v) => v != null);
    const lo = Math.max(0, Math.floor(Math.min(...series) * 10) / 10), hi = Math.min(1, Math.ceil(Math.max(...series) * 10) / 10);
    const x = scale(0, n, M.l, W - M.r), y = scale(lo, hi, Hh - M.b, M.t);
    const yearTicks = months.map((m, i) => [m, i]).filter(([m]) => m.endsWith("-01")).map(([, i]) => i);
    axes(s, x, y, yearTicks, niceTicks(lo, hi, 5), (i) => months[i].slice(0, 4), (v) => `${Math.round(v * 100)}%`, M, W, Hh);
    const line = (arr, color, wdt) => {
      let dstr = "", pen = false;
      arr.forEach((v, i) => { if (v == null) { pen = false; return; } dstr += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`; pen = true; });
      add(s, "path", { d: dstr, fill: "none", stroke: color, "stroke-width": wdt, "stroke-linejoin": "round" });
    };
    models.forEach((m) => line(d.timeline.models[m].series, modelColor(m), 1.5));
    line(d.timeline.actual, css("--text"), 2.8);
    const cross = add(s, "line", { y1: M.t, y2: Hh - M.b, stroke: css("--muted"), visibility: "hidden" });
    overlay(s, M, W, Hh, (e, pt) => {
      if (!e) { cross.setAttribute("visibility", "hidden"); return; }
      const i = Math.max(0, Math.min(n, Math.round(x.inv(pt.x))));
      cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
      const a = d.timeline.actual[i];
      showTip(e, `<div class="t">${months[i]} · ${d.timeline.n[i]} orders</div><b>Real: ${pct(a)}</b> filled within ${d.timeline_days} days<br>` +
        models.map((m) => `<span class="swatch" style="background:${modelColor(m)}"></span>${modelLabel(m)}: ${pct(d.timeline.models[m].series[i])}`).join("<br>"));
    });
  }

  function renderCmpPrice(d) {
    const el = $("cmp-price");
    if (!d.price) { el.innerHTML = `<p class="muted small">Pick a single stock to see its price below the timeline.</p>`; return; }
    const W = cw("cmp-price"), Hh = 130, M = { l: 44, r: 16, t: 18, b: 24 };
    const s = svg(el, W, Hh);
    const v = d.price, n = v.length - 1, ok = v.filter((p) => p != null);
    const x = scale(0, n, M.l, W - M.r), y = scale(Math.min(...ok) * 0.95, Math.max(...ok) * 1.05, Hh - M.b, M.t);
    const yearTicks = d.months.map((m, i) => [m, i]).filter(([m]) => m.endsWith("-01")).map(([, i]) => i);
    axes(s, x, y, yearTicks, niceTicks(Math.min(...ok) * 0.95, Math.max(...ok) * 1.05, 3), (i) => d.months[i].slice(0, 4), (p) => `$${Math.round(p)}`, M, W, Hh);
    let dstr = "";
    v.forEach((p, i) => { if (p != null) dstr += `${dstr ? "L" : "M"}${x(i).toFixed(1)},${y(p).toFixed(1)}`; });
    add(s, "path", { d: dstr, fill: "none", stroke: css("--muted"), "stroke-width": 1.6 });
    add(s, "text", { class: "direct-label", x: M.l, y: 11 }, `${d.ticker} month-end price, for context`);
  }

  // ---------- replay ----------
  function stepDay(delta) {
    const cur = S.history ? S.history.as_of : S.asof;
    let i = S.dates.indexOf(cur);
    if (i < 0) i = S.dates.findIndex((d) => d > cur);
    const j = Math.max(0, Math.min(S.dates.length - 1, i + delta));
    if (j === i) { stopPlay(); return false; }
    S.asof = S.dates[j];
    return true;
  }
  function stopPlay() {
    if (S.playing) clearInterval(S.playing);
    S.playing = null; S.lockTarget = false;
    $("play").classList.remove("on"); $("play").innerHTML = "&#9654; Replay";
  }
  function startPlay() {
    S.lockTarget = true;
    $("play").classList.add("on"); $("play").innerHTML = "&#10074;&#10074; Pause";
    S.playing = setInterval(async () => {
      if (!stepDay(1)) return;
      await refresh({ keep: "target" });
      if (S.pred && S.pred.marketable) stopPlay();
    }, 1100);
  }

  // ---------- wiring ----------
  let deb;
  const debounce = (fn, ms = 140) => { clearTimeout(deb); deb = setTimeout(fn, ms); };

  async function init() {
    S.info = await api("info");
    S.dates = await api("dates");
    S.asof = S.info.dates.default;
    S.model = S.info.default_model;
    $("ticker").innerHTML = S.info.tickers.map((t) => `<option value="${t.ticker}">${t.ticker} · ${t.sector.replace(/_/g, " ")}</option>`).join("");
    $("ticker").value = S.ticker;
    $("model").innerHTML = S.info.models.slice().reverse().map((m) => `<option value="${m.id}">${m.label}${m.test ? ` · ${acc(m.test.accuracy)} accurate` : ""}</option>`).join("");
    $("model").value = S.model;
    $("asof").min = S.info.dates.first; $("asof").max = S.info.dates.last;

    $("ticker").addEventListener("change", (e) => { stopPlay(); S.ticker = e.target.value; refresh(); });
    $("model").addEventListener("change", (e) => { S.model = e.target.value; refresh({ keep: "target" }); });
    $("dist").addEventListener("input", (e) => { stopPlay(); S.dist = +e.target.value; S.target = null; syncDistLabel(); debounce(() => refresh({ keep: "dist" })); });
    $("target").addEventListener("change", (e) => { stopPlay(); S.target = +e.target.value; refresh({ keep: "target" }); });
    $("asof").addEventListener("change", (e) => { stopPlay(); if (e.target.value) { S.asof = e.target.value; refresh({ keep: "dist" }); } });
    $("prev-day").addEventListener("click", () => { stopPlay(); if (stepDay(-1)) refresh({ keep: "target" }); });
    $("next-day").addEventListener("click", () => { stopPlay(); if (stepDay(1)) refresh({ keep: "target" }); });
    $("play").addEventListener("click", () => (S.playing ? stopPlay() : startPlay()));
    $("reveal").addEventListener("change", () => S.pred && render());
    document.querySelectorAll(".seg-btn").forEach((b) => b.addEventListener("click", () => {
      stopPlay();
      document.querySelectorAll(".seg-btn").forEach((x) => x.classList.toggle("active", x === b));
      S.side = b.dataset.side; S.target = null; refresh({ keep: "dist" });
    }));
    document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
      document.querySelectorAll(".view").forEach((v) => { v.hidden = v.id !== `view-${b.dataset.view}`; });
      if (b.dataset.view === "scorecard") renderScorecard();
      if (b.dataset.view === "reality") initReality();
    }));
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (S.pred) render(); renderReality(); });
    let rz;
    window.addEventListener("resize", () => { hideTip(); clearTimeout(rz); rz = setTimeout(() => { if (S.pred) render(); if (!$("view-reality").hidden) renderReality(); }, 150); });
    await refresh();
  }
  function syncDistLabel() {
    if (!S.ref) return;
    const sgn = S.side === "buy" ? -1 : 1;
    $("dist-label").textContent = `${sgn < 0 ? "−" : "+"}${S.dist.toFixed(2)}% (${usd(S.ref * (1 + sgn * S.dist / 100))})`;
  }

  init().catch((e) => { $("error").textContent = `Could not start: ${e.message}`; $("error").hidden = false; });
})();
