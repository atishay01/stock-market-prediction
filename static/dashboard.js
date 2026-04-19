let historyChart = null;
let backtestChart = null;
let featureChart = null;

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).error || msg; } catch {}
    throw new Error(msg);
  }
  return r.json();
}

function renderLine(canvasId, existing, labels, datasets) {
  if (existing) existing.destroy();
  const ctx = document.getElementById(canvasId).getContext("2d");
  return new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { color: "#c1d1ea" } } },
      scales: {
        x: { ticks: { color: "#8ea0bd", maxTicksLimit: 10 }, grid: { color: "#253049" } },
        y: { ticks: { color: "#8ea0bd" }, grid: { color: "#253049" } },
      },
    },
  });
}

function renderBadges(bt) {
  const liftClass = bt.lift_vs_persistence_pct >= 0 ? "good" : "bad";
  const dirClass = bt.directional_accuracy >= 50 ? "good" : "bad";
  document.getElementById("backtest-badge").innerHTML = [
    `<span class="badge">${bt.ticker} · last <strong>${bt.n_days}</strong> days</span>`,
    `<span class="badge">RF MAPE <strong>${bt.rf_mape}%</strong></span>`,
    `<span class="badge">Persistence MAPE <strong>${bt.persistence_mape}%</strong></span>`,
    `<span class="badge ${liftClass}">Lift vs persistence <strong>${bt.lift_vs_persistence_pct >= 0 ? "+" : ""}${bt.lift_vs_persistence_pct}%</strong></span>`,
    `<span class="badge ${dirClass}">Directional acc <strong>${bt.directional_accuracy}%</strong></span>`,
  ].join(" ");
}

async function loadCharts(ticker) {
  try {
    const hist = await fetchJSON(`/api/history?ticker=${encodeURIComponent(ticker)}`);
    historyChart = renderLine("history-chart", historyChart, hist.dates, [
      { label: `${hist.ticker} Close`, data: hist.close, borderColor: "#38d9a9",
        backgroundColor: "#38d9a933", tension: 0.25, pointRadius: 0, borderWidth: 2 },
    ]);

    const bt = await fetchJSON(`/api/backtest?ticker=${encodeURIComponent(ticker)}`);
    renderBadges(bt);
    backtestChart = renderLine("backtest-chart", backtestChart, bt.dates, [
      { label: "Actual next close", data: bt.actual, borderColor: "#4dabf7",
        backgroundColor: "transparent", tension: 0.2, pointRadius: 0, borderWidth: 2 },
      { label: "Predicted (RF)", data: bt.predicted, borderColor: "#fab005",
        borderDash: [6, 4], backgroundColor: "transparent", tension: 0.2,
        pointRadius: 0, borderWidth: 2 },
      { label: "Persistence baseline (yesterday)", data: bt.persistence,
        borderColor: "#9aa7bf", borderDash: [2, 3], backgroundColor: "transparent",
        tension: 0.2, pointRadius: 0, borderWidth: 1.5 },
    ]);
  } catch (e) {
    console.error("chart load error:", e);
  }
}

async function loadFeatureImportance() {
  try {
    const m = await fetchJSON("/api/metrics");
    const fi = (m.random_forest && m.random_forest.feature_importance) || [];
    const top = fi.slice(0, 10);
    if (!top.length) return;
    if (featureChart) featureChart.destroy();
    const ctx = document.getElementById("feature-importance-chart").getContext("2d");
    featureChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: top.map(x => x.feature),
        datasets: [{
          label: "Importance",
          data: top.map(x => x.importance),
          backgroundColor: "#38d9a9aa",
          borderColor: "#38d9a9",
          borderWidth: 1,
        }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#8ea0bd" }, grid: { color: "#253049" } },
          y: { ticks: { color: "#c1d1ea" }, grid: { color: "#253049" } },
        },
      },
    });
  } catch (e) {
    console.error("feature importance load error:", e);
  }
}

function currentTicker() {
  return (document.getElementById("ticker").value.trim() || "AAPL").toUpperCase();
}

document.querySelectorAll(".chip").forEach((btn) => {
  btn.addEventListener("click", (ev) => {
    ev.preventDefault();
    const t = btn.getAttribute("data-ticker");
    document.getElementById("ticker").value = t;
    loadCharts(t);
  });
});

document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const ticker = currentTicker();
  const headline = document.getElementById("headline").value;
  const box = document.getElementById("predict-result");
  box.textContent = `Fetching ${ticker} and running models ...`;
  try {
    const r = await fetchJSON("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker, headline }),
    });
    const sign = (v) => (v >= 0 ? "+" : "");
    const lines = [
      `Ticker:                        ${r.ticker}`,
      `Last close (${r.last_date}):   $${r.last_close}`,
      `Next trading date:             ${r.next_trading_date}`,
      `VADER sentiment score:         ${r.sentiment_score}`,
      ``,
      `Random Forest predicted close: $${r.rf_prediction}   (${sign(r.rf_change_pct)}${r.rf_change_pct}%)`,
    ];
    if (r.lstm_prediction !== null && r.lstm_prediction !== undefined) {
      lines.push(`LSTM predicted close:          $${r.lstm_prediction}   (${sign(r.lstm_change_pct)}${r.lstm_change_pct}%)`);
    } else {
      lines.push(`LSTM:                          disabled in this deployment (RF-only mode)`);
    }
    box.textContent = lines.join("\n");
    loadCharts(ticker);
  } catch (err) {
    box.textContent = "Error: " + err.message;
  }
});

(async function boot() {
  await loadFeatureImportance();
  await loadCharts(currentTicker());
})();
