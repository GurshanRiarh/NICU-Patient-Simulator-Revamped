const COLORS = {
  hr: "#ff729b",
  spo2: "#58e3c2",
  rr: "#62a8ff",
  temp: "#f6c96a",
  brady: "rgba(255,114,155,0.14)",
  desat: "rgba(88,227,194,0.12)",
  apnea: "rgba(180,155,255,0.16)",
};

const CONDITION_LABELS = {
  stable_preterm: "Stable preterm",
  bradycardia: "Recurrent bradycardia",
  tachycardia: "Tachycardia",
  apnea_of_prematurity: "Apnoea of prematurity",
  rds: "Respiratory distress syndrome",
  sepsis: "Late-onset sepsis",
  nec: "Necrotising enterocolitis",
};

const $ = (id) => document.getElementById(id);
const form = $("simForm");
let chart = null;
let currentData = null;
let animationTimer = null;
let toastTimer = null;

function fmtTime(iso) {
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function fmtShiftTime(data, minute) {
  if (!data.times || !data.times.length) return "--:--";
  const start = new Date(data.times[0]);
  if (Number.isNaN(start.getTime())) return "--:--";
  const offset = Number(minute) || 0;
  return new Date(start.getTime() + offset * 60000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 2800);
}

function formPayload() {
  const seedValue = $("seed").value.trim();
  return {
    patient_id: "SIM-" + String(Math.floor(1000 + Math.random() * 8999)),
    name: $("name").value.trim() || "Baby Doe",
    gestational_age_weeks: Number($("ga").value),
    birth_weight_g: Number($("weight").value),
    postnatal_age_days: Number($("age").value),
    sex: $("sex").value,
    condition: $("condition").value,
    severity: Number($("severity").value),
    shift_hours: Number($("hours").value),
    interval_minutes: Number($("interval").value),
    cohort: "auto",
    care_events: $("care").checked,
    closed_loop: $("closed").checked,
    seed: seedValue === "" ? null : Number(seedValue),
  };
}

function updateRangeLabels() {
  $("gaValue").textContent = Number($("ga").value).toFixed(1);
  $("weightValue").textContent = Number($("weight").value).toFixed(0);
  $("severityValue").textContent = Number($("severity").value).toFixed(2);
}

function updatePatient(data) {
  const c = data.config;
  $("patientName").textContent = c.name;
  $("patientId").textContent = c.patient_id;
  $("patientDemo").textContent =
    `${Number(c.gestational_age_weeks).toFixed(0)}w · ${(c.birth_weight_g / 1000).toFixed(2)} kg · day ${c.postnatal_age_days}`;
  $("conditionChip").textContent = CONDITION_LABELS[c.condition] || c.condition;
  $("cohortLabel").textContent = c.cohort === "pics" ? "PICS" : "PreMo";
  $("shiftLabel").textContent = `${c.shift_hours} h`;
  $("modelLabel").textContent = data.model.model || "NeoFlow";
  $("modelStatus").textContent = `${data.model.model || "NeoFlow"} · ${data.model.gpu ? "GPU" : "CUDA"}`;
  if (data.model.gpu) $("metricDevice").textContent = data.model.gpu;
}

function computeStats(data, upto) {
  const hr = data.series.hr.slice(0, upto);
  const rr = data.series.rr.slice(0, upto);
  const spo2 = data.series.spo2.slice(0, upto);
  const temp = data.series.temp.slice(0, upto);
  const stability = data.stability.slice(0, upto);
  const mean = (a) => a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;
  const interventions = data.interventions.filter((i) => i.timestamp < upto).length;
  const hrMean = mean(hr), rrMean = mean(rr), spo2Mean = mean(spo2), tempMean = mean(temp);
  $("statHr").textContent = Math.round(hrMean);
  $("statHrSub").textContent = `${Math.round(Math.min(...hr))}–${Math.round(Math.max(...hr))} bpm range`;
  $("statSpo2").textContent = spo2Mean.toFixed(1);
  $("statSpo2Sub").textContent = `nadir ${Math.round(Math.min(...spo2))}%`;
  $("statRr").textContent = Math.round(rrMean);
  $("statRrSub").textContent = `${Math.round(Math.min(...rr))}–${Math.round(Math.max(...rr))} /min`;
  $("statTemp").textContent = tempMean.toFixed(1);
  $("statTempSub").textContent = "thermoregulated";
  $("statStability").textContent = Math.round(mean(stability));
  $("statStabilitySub").textContent = mean(stability) > 85 ? "stable trajectory" : "monitor closely";
  $("statInterventions").textContent = interventions;
  $("statInterventionsSub").textContent = `${data.events.length} clinical events`;
  $("playStatus").textContent = upto >= data.times.length ? "Complete" : `Live · ${fmtTime(data.times[Math.max(0, upto - 1)])}`;
}

function eventAreas(data, upto, kind) {
  const interval = data.config.interval_minutes || 5;
  const labels = data.times.map(fmtTime);
  const areas = [];
  for (const e of data.events) {
    let group = "care";
    if (e.type.includes("brady")) group = "brady";
    else if (e.type.includes("desat")) group = "desat";
    else if (e.type === "apnea") group = "apnea";
    if (kind === "hr" && group !== "brady") continue;
    if (kind === "spo2" && !["desat", "apnea"].includes(group)) continue;
    const start = Math.max(0, Math.round(e.start_minute / interval));
    const end = Math.min(upto - 1, Math.round((e.start_minute + e.duration_minutes) / interval));
    if (start >= upto || end < 0) continue;
    areas.push([
      { xAxis: labels[Math.min(start, labels.length - 1)], itemStyle: { color: COLORS[group] } },
      { xAxis: labels[Math.max(0, Math.min(end, labels.length - 1))] },
    ]);
  }
  return areas;
}

function makeSeries(data, upto) {
  const n = upto;
  return [
    {
      name: "Heart rate", type: "line", xAxisIndex: 0, yAxisIndex: 0,
      data: data.series.hr.slice(0, n), smooth: 0.32, symbol: "none",
      lineStyle: { color: COLORS.hr, width: 1.9 },
      areaStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: "rgba(255,114,155,0.24)" },
          { offset: 1, color: "rgba(255,114,155,0)" },
        ]),
      },
      markArea: { silent: true, data: eventAreas(data, n, "hr") },
    },
    {
      name: "SpO₂", type: "line", xAxisIndex: 1, yAxisIndex: 1,
      data: data.series.spo2.slice(0, n), smooth: 0.3, symbol: "none",
      lineStyle: { color: COLORS.spo2, width: 1.9 },
      areaStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: "rgba(88,227,194,0.22)" },
          { offset: 1, color: "rgba(88,227,194,0)" },
        ]),
      },
      markArea: { silent: true, data: eventAreas(data, n, "spo2") },
    },
    {
      name: "Respiratory rate", type: "line", xAxisIndex: 2, yAxisIndex: 2,
      data: data.series.rr.slice(0, n), smooth: 0.3, symbol: "none",
      lineStyle: { color: COLORS.rr, width: 1.7 },
    },
    {
      name: "Temperature", type: "line", xAxisIndex: 3, yAxisIndex: 3,
      data: data.series.temp.slice(0, n), smooth: 0.45, symbol: "none",
      lineStyle: { color: COLORS.temp, width: 1.7 },
      areaStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: "rgba(246,201,106,0.13)" },
          { offset: 1, color: "rgba(246,201,106,0)" },
        ]),
      },
    },
  ];
}

function baseOption(data) {
  const labels = data.times.map(fmtTime);
  const names = ["Heart rate (bpm)", "SpO₂ (%)", "Resp. rate (/min)", "Temperature (°C)"];
  const ranges = [[40, 230], [55, 101], [0, 95], [34.6, 39.6]];
  const colors = [COLORS.hr, COLORS.spo2, COLORS.rr, COLORS.temp];
  const grids = [0, 1, 2, 3].map((i) => ({
    left: 66, right: 26, top: `${6 + i * 23.5}%`, height: "16%",
  }));
  const xAxes = [0, 1, 2, 3].map((i) => ({
    type: "category", gridIndex: i, data: labels, boundaryGap: false,
    axisLine: { lineStyle: { color: "rgba(137,177,220,0.18)" } },
    axisTick: { show: false },
    axisLabel: {
      show: i === 3, color: "#6f86a2", fontSize: 9.5,
      hideOverlap: true, margin: 10,
    },
    axisPointer: { label: { show: i === 3 } },
  }));
  const yAxes = [0, 1, 2, 3].map((i) => ({
    type: "value", gridIndex: i, name: names[i], min: ranges[i][0], max: ranges[i][1],
    nameLocation: "middle", nameGap: 43,
    nameTextStyle: { color: "#6f86a2", fontSize: 9.5, align: "center" },
    axisLabel: { color: "#5c728c", fontSize: 9 },
    splitLine: { lineStyle: { color: "rgba(137,177,220,0.07)" } },
    axisLine: { show: false }, axisTick: { show: false },
  }));
  return {
    backgroundColor: "transparent",
    animation: false,
    grid: grids,
    xAxis: xAxes,
    yAxis: yAxes,
    axisPointer: { link: [{ xAxisIndex: "all" }], lineStyle: { color: "rgba(200,230,255,0.32)", type: "dashed" } },
    tooltip: {
      trigger: "axis", backgroundColor: "rgba(8,18,31,0.96)",
      borderColor: "rgba(137,177,220,0.22)", textStyle: { color: "#dbe9f8", fontSize: 11 },
      axisPointer: { type: "cross", crossStyle: { color: "rgba(200,230,255,0.25)" } },
      formatter: (params) => {
        if (!params.length) return "";
        const t = params[0].axisValue;
        const rows = params.map((p) => `<div style="display:flex;justify-content:space-between;gap:18px"><span>${p.marker}${p.seriesName}</span><b>${Number(p.value).toFixed(1)}</b></div>`).join("");
        return `<div style="font-weight:600;margin-bottom:5px">${t}</div>${rows}`;
      },
    },
    dataZoom: [
      { type: "inside", xAxisIndex: [0, 1, 2, 3], filterMode: "none" },
    ],
    series: makeSeries(data, 0),
  };
}

function initChart() {
  if (!chart) chart = echarts.init($("vitalsChart"), null, { renderer: "canvas" });
}

function renderChart(data, upto, reset = false) {
  initChart();
  if (reset) {
    chart.setOption(baseOption(data), { notMerge: true });
    chart.setOption({ series: makeSeries(data, upto) });
  } else {
    chart.setOption({ series: makeSeries(data, upto) });
  }
}

function renderTimeline(data) {
  const box = $("eventTimeline");
  const events = data.events.slice().sort((a, b) => a.start_minute - b.start_minute);
  $("eventCount").textContent = `${events.length} events`;
  if (!events.length) {
    box.innerHTML = '<div class="empty">No clinical events in this shift.</div>';
    return;
  }
  box.innerHTML = events.slice(0, 120).map((e) => {
    let cls = "care";
    if (e.type.includes("brady")) cls = "brady";
    else if (e.type.includes("desat")) cls = "desat";
    else if (e.type === "apnea") cls = "apnea";
    return `<div class="event-row">
      <span class="event-time">${fmtShiftTime(data, e.start_minute)}</span>
      <span class="event-dot ${cls}"></span>
      <span class="event-label">${e.label || e.type}</span>
      <span class="event-duration">${e.duration_minutes.toFixed(0)} min</span>
    </div>`;
  }).join("");
}

function renderInterventions(data) {
  const body = $("interventionBody");
  const items = data.interventions.slice().sort((a, b) => a.minute - b.minute);
  $("interventionCount").textContent = `${items.length} orders`;
  if (!items.length) {
    body.innerHTML = '<tr><td colspan="3" class="empty-cell">No interventions in this shift.</td></tr>';
    return;
  }
  body.innerHTML = items.map((i) => {
    const time = fmtTime(data.times[Math.min(i.timestamp, data.times.length - 1)]);
    return `<tr><td>${time}</td><td>${i.type}</td><td>${i.detail}</td></tr>`;
  }).join("");
}

function animate(data) {
  clearInterval(animationTimer);
  let upto = 0;
  const total = data.times.length;
  const stride = Math.max(1, Math.round(total / 95));
  computeStats(data, 1);
  renderChart(data, 1, true);
  animationTimer = setInterval(() => {
    upto = Math.min(total, upto + stride);
    renderChart(data, upto);
    computeStats(data, upto);
    if (upto >= total) {
      clearInterval(animationTimer);
      chart.resize();
    }
  }, 34);
}

async function runSimulation() {
  const btn = $("generateBtn");
  btn.disabled = true;
  $("playStatus").textContent = "Generating…";
  try {
    const response = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formPayload()),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }
    const data = await response.json();
    currentData = data;
    updatePatient(data);
    renderTimeline(data);
    renderInterventions(data);
    animate(data);
    showToast(`Generated ${data.times.length} timesteps with ${data.model.model || "NeoFlow"}`);
  } catch (error) {
    $("playStatus").textContent = "Failed";
    showToast(`Simulation failed: ${error.message}`);
  } finally {
    btn.disabled = false;
  }
}

async function loadPresets() {
  try {
    const response = await fetch("/api/presets");
    const payload = await response.json();
    const grid = $("presetGrid");
    grid.innerHTML = payload.presets.map((p) => `<button type="button" class="preset" data-id="${p.id}">${p.label}</button>`).join("");
    grid.querySelectorAll(".preset").forEach((button, index) => {
      button.addEventListener("click", () => {
        const p = payload.presets[index];
        $("ga").value = p.gestational_age_weeks;
        $("weight").value = p.birth_weight_g;
        $("age").value = p.postnatal_age_days;
        $("sex").value = p.sex;
        $("condition").value = p.condition;
        $("severity").value = p.severity;
        grid.querySelectorAll(".preset").forEach((b) => b.classList.remove("active"));
        button.classList.add("active");
        updateRangeLabels();
        runSimulation();
      });
    });
  } catch {
    showToast("Could not load presets");
  }
}

async function downloadExport(format) {
  try {
    const response = await fetch(`/api/export?format=${format}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formPayload()),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `neosim_${format}.${format === "csv" ? "zip" : "json"}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    showToast(`Exported ${format.toUpperCase()}`);
  } catch (error) {
    showToast(`Export failed: ${error.message}`);
  }
}

function wire() {
  ["ga", "weight", "severity"].forEach((id) => $(id).addEventListener("input", updateRangeLabels));
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    runSimulation();
  });
  $("exportJson").addEventListener("click", () => downloadExport("json"));
  $("exportCsv").addEventListener("click", () => downloadExport("csv"));
  window.addEventListener("resize", () => chart && chart.resize());
  updateRangeLabels();
  loadPresets().then(() => {
    const stable = document.querySelector('.preset[data-id="stable"]');
    if (stable) stable.classList.add("active");
    runSimulation();
  });
}

wire();
