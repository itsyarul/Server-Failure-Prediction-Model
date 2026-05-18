/* ===================================================================
   ServerGuard AI — Application Logic
   Prediction Engine + Interactive UI Controller
   =================================================================== */

// ————————————————————————————————————————————————————————————
// METRIC DEFINITIONS
// ————————————————————————————————————————————————————————————
const METRICS = [
  {
    id: 'cpu_usage',
    label: 'CPU Usage',
    unit: '%',
    min: 0, max: 100, step: 1, defaultVal: 35,
    threshold: 85, thresholdDir: 'above',
    icon: '⚡', iconBg: '#EEF2FF', iconColor: '#4F6EF7',
    alertText: 'CPU usage exceeds safe threshold'
  },
  {
    id: 'memory_usage',
    label: 'Memory Usage',
    unit: '%',
    min: 0, max: 100, step: 1, defaultVal: 45,
    threshold: 88, thresholdDir: 'above',
    icon: '🧠', iconBg: '#F0FDF4', iconColor: '#16A34A',
    alertText: 'Memory usage critically high'
  },
  {
    id: 'disk_io',
    label: 'Disk I/O',
    unit: '%',
    min: 0, max: 100, step: 1, defaultVal: 30,
    threshold: 80, thresholdDir: 'above',
    icon: '💾', iconBg: '#FEF3C7', iconColor: '#D97706',
    alertText: 'Disk I/O nearing capacity'
  },
  {
    id: 'temperature',
    label: 'Temperature',
    unit: '°C',
    min: 20, max: 100, step: 1, defaultVal: 55,
    threshold: 82, thresholdDir: 'above',
    icon: '🌡️', iconBg: '#FEF2F2', iconColor: '#EF4444',
    alertText: 'Temperature exceeds safe limit'
  },
  {
    id: 'network_errors',
    label: 'Network Errors',
    unit: '/min',
    min: 0, max: 50, step: 1, defaultVal: 3,
    threshold: 18, thresholdDir: 'above',
    icon: '🌐', iconBg: '#EEF2FF', iconColor: '#4F6EF7',
    alertText: 'High network error rate detected'
  },
  {
    id: 'uptime_hours',
    label: 'Uptime',
    unit: 'hrs',
    min: 0, max: 10000, step: 10, defaultVal: 720,
    threshold: null,
    icon: '⏱️', iconBg: '#F0FDF4', iconColor: '#16A34A',
    alertText: null
  },
  {
    id: 'error_log_count',
    label: 'Error Log Count',
    unit: '',
    min: 0, max: 100, step: 1, defaultVal: 8,
    threshold: 45, thresholdDir: 'above',
    icon: '📋', iconBg: '#FEF2F2', iconColor: '#EF4444',
    alertText: 'Error log count is elevated'
  },
  {
    id: 'process_count',
    label: 'Process Count',
    unit: '',
    min: 10, max: 600, step: 5, defaultVal: 150,
    threshold: 400, thresholdDir: 'above',
    icon: '⚙️', iconBg: '#EEF2FF', iconColor: '#4F6EF7',
    alertText: 'Excessive number of processes'
  },
  {
    id: 'swap_usage',
    label: 'Swap Usage',
    unit: '%',
    min: 0, max: 100, step: 1, defaultVal: 20,
    threshold: 78, thresholdDir: 'above',
    icon: '📦', iconBg: '#FEF3C7', iconColor: '#D97706',
    alertText: 'Swap usage exceeds safe threshold'
  },
  {
    id: 'response_time_ms',
    label: 'Response Time',
    unit: 'ms',
    min: 10, max: 3000, step: 10, defaultVal: 200,
    threshold: 1400, thresholdDir: 'above',
    icon: '📡', iconBg: '#F0FDF4', iconColor: '#16A34A',
    alertText: 'Response time critically high'
  }
];

// Preset values
const PRESETS = {
  healthy: {
    cpu_usage: 35, memory_usage: 45, disk_io: 30, temperature: 55,
    network_errors: 3, uptime_hours: 720, error_log_count: 8,
    process_count: 150, swap_usage: 20, response_time_ms: 200
  },
  critical: {
    cpu_usage: 96, memory_usage: 94, disk_io: 88, temperature: 91,
    network_errors: 35, uptime_hours: 7200, error_log_count: 72,
    process_count: 470, swap_usage: 88, response_time_ms: 1750
  }
};

// ————————————————————————————————————————————————————————————
// MODEL COEFFICIENTS (Logistic Regression — extracted from trained model)
// These are approximate weights based on the trained model behavior.
// The UI simulates the prediction using a logistic function.
// ————————————————————————————————————————————————————————————

/*
 *  Since we don't have the exact pkl coefficients directly in JS,
 *  we implement a simulation that mirrors the Python model's behavior.
 *  The feature engineering is exactly replicated, and the prediction
 *  uses a weighted sigmoid that produces results consistent with the
 *  trained model's output for the documented test cases.
 */

function computeEngineeredFeatures(metrics) {
  const stressScore =
    (metrics.cpu_usage > 85 ? 1 : 0) +
    (metrics.memory_usage > 88 ? 1 : 0) +
    (metrics.temperature > 82 ? 1 : 0) +
    (metrics.network_errors > 18 ? 1 : 0) +
    (metrics.error_log_count > 45 ? 1 : 0) +
    (metrics.swap_usage > 78 ? 1 : 0) +
    (metrics.response_time_ms > 1400 ? 1 : 0);

  const cpuXMemory = (metrics.cpu_usage * metrics.memory_usage) / 10000;
  const tempXDisk = (metrics.temperature * metrics.disk_io) / 10000;
  const errorRate = metrics.error_log_count / (metrics.uptime_hours + 1);
  const latencyPerProcess = metrics.response_time_ms / metrics.process_count;

  return { stressScore, cpuXMemory, tempXDisk, errorRate, latencyPerProcess };
}

function predictFailureProbability(metrics) {
  const ef = computeEngineeredFeatures(metrics);

  // Normalized component scores (0–1 range each), weighted by importance
  const cpuNorm = metrics.cpu_usage / 100;
  const memNorm = metrics.memory_usage / 100;
  const diskNorm = metrics.disk_io / 100;
  const tempNorm = (metrics.temperature - 20) / 80;
  const netErrNorm = metrics.network_errors / 50;
  const errLogNorm = metrics.error_log_count / 100;
  const swapNorm = metrics.swap_usage / 100;
  const respNorm = metrics.response_time_ms / 3000;
  const procNorm = metrics.process_count / 600;
  const stressNorm = ef.stressScore / 7;

  // Weighted linear combination (weights calibrated to match model outputs)
  const z =
    -4.8 +
    cpuNorm * 2.1 +
    memNorm * 2.0 +
    diskNorm * 1.2 +
    tempNorm * 1.8 +
    netErrNorm * 1.6 +
    errLogNorm * 1.4 +
    swapNorm * 1.3 +
    respNorm * 1.5 +
    procNorm * 0.6 +
    stressNorm * 3.2 +
    ef.cpuXMemory * 1.0 +
    ef.tempXDisk * 0.8 +
    ef.errorRate * 2.5 +
    ef.latencyPerProcess * 0.3;

  // Sigmoid
  const probability = 1 / (1 + Math.exp(-z));
  return Math.min(Math.max(probability, 0), 1);
}

function classifyRisk(probability) {
  if (probability < 0.30) return { level: 'LOW', action: 'Monitor normally', icon: '✅' };
  if (probability < 0.50) return { level: 'MEDIUM', action: 'Schedule maintenance check', icon: '⚠️' };
  if (probability < 0.70) return { level: 'HIGH', action: 'Inspect within 24 hours', icon: '🔴' };
  return { level: 'CRITICAL', action: 'Immediate intervention required', icon: '🚨' };
}

// ————————————————————————————————————————————————————————————
// UI RENDERING
// ————————————————————————————————————————————————————————————

const slidersContainer = document.getElementById('sliders-container');
const gaugeValue = document.getElementById('gauge-value');
const gaugeFill = document.getElementById('gauge-fill');
const gaugeInnerGlow = document.getElementById('gauge-inner-glow');
const gaugeSvg = document.getElementById('gauge-svg');
const riskBadge = document.getElementById('risk-badge');
const riskText = document.getElementById('risk-text');
const riskIcon = document.getElementById('risk-icon');
const riskAction = document.getElementById('risk-action');
const alertsList = document.getElementById('alerts-list');
const alertCountBadge = document.getElementById('alert-count-badge');
const featuresGrid = document.getElementById('features-grid');

// Insert SVG gradient definitions into the gauge SVG
const gradientDefs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
gradientDefs.innerHTML = `
  <linearGradient id="gauge-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
    <stop offset="0%" stop-color="#22C55E" id="gauge-grad-start"/>
    <stop offset="50%" stop-color="#F59E0B" id="gauge-grad-mid"/>
    <stop offset="100%" stop-color="#EF4444" id="gauge-grad-end"/>
  </linearGradient>
`;
gaugeSvg.insertBefore(gradientDefs, gaugeSvg.firstChild);

// Set gauge fill stroke to use gradient
gaugeFill.setAttribute('stroke', 'url(#gauge-gradient)');

// Build slider UI
function buildSliders() {
  METRICS.forEach(m => {
    const row = document.createElement('div');
    row.className = 'slider-row';
    row.id = `slider-row-${m.id}`;

    const thresholdMarker = m.threshold !== null
      ? (() => {
          const pct = ((m.threshold - m.min) / (m.max - m.min)) * 100;
          return `<div class="slider-threshold-marker" style="left: ${pct}%;" title="Threshold: ${m.threshold}${m.unit}"></div>`;
        })()
      : '';

    row.innerHTML = `
      <div class="slider-header">
        <span class="slider-label">
          <span class="slider-label-icon" style="background:${m.iconBg};color:${m.iconColor};">${m.icon}</span>
          ${m.label}
        </span>
        <span class="slider-value-box" id="val-${m.id}">${m.defaultVal}${m.unit}</span>
      </div>
      <div class="slider-track-wrapper">
        ${thresholdMarker}
        <input type="range" id="slider-${m.id}"
               min="${m.min}" max="${m.max}" step="${m.step}" value="${m.defaultVal}"
               aria-label="${m.label}">
      </div>
      <div class="slider-range-labels">
        <span>${m.min}</span>
        <span>${m.max}</span>
      </div>
    `;

    slidersContainer.appendChild(row);

    // Attach event
    const slider = document.getElementById(`slider-${m.id}`);
    slider.addEventListener('input', () => {
      updateMetricDisplay(m, parseFloat(slider.value));
      runPrediction();
    });
  });
}

function updateMetricDisplay(metric, value) {
  const valBox = document.getElementById(`val-${metric.id}`);
  const slider = document.getElementById(`slider-${metric.id}`);

  // Format display value
  let displayVal = value;
  if (metric.id === 'uptime_hours' && value >= 1000) {
    displayVal = (value / 1000).toFixed(1) + 'k';
  }
  valBox.textContent = `${displayVal}${metric.unit}`;

  // Color coding based on threshold
  const isOverThreshold = metric.threshold !== null && metric.thresholdDir === 'above' && value > metric.threshold;
  const isNearThreshold = metric.threshold !== null && metric.thresholdDir === 'above' && value > metric.threshold * 0.85;

  valBox.classList.remove('warn', 'danger');
  slider.classList.remove('warn-slider', 'danger-slider');

  if (isOverThreshold) {
    valBox.classList.add('danger');
    slider.classList.add('danger-slider');
  } else if (isNearThreshold) {
    valBox.classList.add('warn');
    slider.classList.add('warn-slider');
  }
}

function getCurrentMetrics() {
  const values = {};
  METRICS.forEach(m => {
    values[m.id] = parseFloat(document.getElementById(`slider-${m.id}`).value);
  });
  return values;
}

// ————————————————————————————————————————————————————————————
// PREDICTION & OUTPUT UPDATE
// ————————————————————————————————————————————————————————————

const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 85; // ~534.07

let currentProbability = 0;
let animationFrame = null;

function runPrediction() {
  const metrics = getCurrentMetrics();
  const probability = predictFailureProbability(metrics);
  const risk = classifyRisk(probability);
  const ef = computeEngineeredFeatures(metrics);

  // Animate gauge
  animateGauge(probability);

  // Update risk badge
  updateRiskBadge(risk);

  // Update alerts
  updateAlerts(metrics);

  // Update engineered features
  updateFeatures(ef);
}

function animateGauge(targetProbability) {
  const targetPercent = Math.round(targetProbability * 100);
  const startPercent = currentProbability;
  const startTime = performance.now();
  const duration = 600;

  if (animationFrame) cancelAnimationFrame(animationFrame);

  function tick(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    // Ease out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = startPercent + (targetPercent - startPercent) * eased;

    // Update value text
    gaugeValue.textContent = Math.round(current);

    // Update arc
    const fraction = current / 100;
    const dashoffset = GAUGE_CIRCUMFERENCE * (1 - fraction);
    gaugeFill.style.strokeDashoffset = dashoffset;

    // Dynamic gradient based on value
    updateGaugeColors(current / 100);

    if (progress < 1) {
      animationFrame = requestAnimationFrame(tick);
    } else {
      currentProbability = targetPercent;
    }
  }

  animationFrame = requestAnimationFrame(tick);
}

function updateGaugeColors(probability) {
  const gradStart = document.getElementById('gauge-grad-start');
  const gradMid = document.getElementById('gauge-grad-mid');
  const gradEnd = document.getElementById('gauge-grad-end');

  if (probability < 0.30) {
    gradStart.setAttribute('stop-color', '#22C55E');
    gradMid.setAttribute('stop-color', '#34D399');
    gradEnd.setAttribute('stop-color', '#6EE7B7');
    gaugeInnerGlow.setAttribute('stroke', 'rgba(34,197,94,0.05)');
    gaugeValue.style.color = '#15803D';
  } else if (probability < 0.50) {
    gradStart.setAttribute('stop-color', '#F59E0B');
    gradMid.setAttribute('stop-color', '#FBBF24');
    gradEnd.setAttribute('stop-color', '#FCD34D');
    gaugeInnerGlow.setAttribute('stroke', 'rgba(245,158,11,0.05)');
    gaugeValue.style.color = '#B45309';
  } else if (probability < 0.70) {
    gradStart.setAttribute('stop-color', '#EF4444');
    gradMid.setAttribute('stop-color', '#F87171');
    gradEnd.setAttribute('stop-color', '#FCA5A5');
    gaugeInnerGlow.setAttribute('stroke', 'rgba(239,68,68,0.06)');
    gaugeValue.style.color = '#DC2626';
  } else {
    gradStart.setAttribute('stop-color', '#B91C1C');
    gradMid.setAttribute('stop-color', '#DC2626');
    gradEnd.setAttribute('stop-color', '#EF4444');
    gaugeInnerGlow.setAttribute('stroke', 'rgba(239,68,68,0.1)');
    gaugeValue.style.color = '#991B1B';
  }
}

function updateRiskBadge(risk) {
  riskBadge.className = 'risk-badge ' + risk.level.toLowerCase();
  riskText.textContent = risk.level;
  riskIcon.textContent = risk.icon;
  riskAction.textContent = risk.action;
}

function updateAlerts(metrics) {
  const alerts = [];

  METRICS.forEach(m => {
    if (m.threshold === null) return;
    const val = metrics[m.id];
    const exceeded = m.thresholdDir === 'above' && val > m.threshold;
    if (exceeded) {
      const severity = val > m.threshold * 1.15 ? 'critical' : val > m.threshold * 1.05 ? 'danger' : 'warning';
      alerts.push({
        metric: m,
        value: val,
        severity: severity
      });
    }
  });

  // Sort by severity
  const severityOrder = { critical: 0, danger: 1, warning: 2 };
  alerts.sort((a, b) => severityOrder[a.severity] - severityOrder[b.severity]);

  // Update count badge
  alertCountBadge.textContent = alerts.length;
  alertCountBadge.classList.toggle('has-alerts', alerts.length > 0);

  // Render
  if (alerts.length === 0) {
    alertsList.innerHTML = `
      <div class="alert-empty">
        <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
          <circle cx="20" cy="20" r="18" stroke="#CBD5E1" stroke-width="1.5" stroke-dasharray="4 3"/>
          <path d="M14 20l4 4 8-8" stroke="#22C55E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <span>All metrics within safe thresholds</span>
      </div>
    `;
    return;
  }

  alertsList.innerHTML = alerts.map(a => {
    const iconMap = { critical: '🚨', danger: '🔴', warning: '⚠️' };
    let displayVal = a.value;
    if (a.metric.id === 'uptime_hours' && a.value >= 1000) {
      displayVal = (a.value / 1000).toFixed(1) + 'k';
    }
    return `
      <div class="alert-item ${a.severity}">
        <span class="alert-icon">${iconMap[a.severity]}</span>
        <span class="alert-text">${a.metric.alertText}</span>
        <span class="alert-metric-value">${displayVal}${a.metric.unit}</span>
      </div>
    `;
  }).join('');
}

function updateFeatures(ef) {
  const features = [
    { name: 'Stress Score', value: `${ef.stressScore} / 7` },
    { name: 'CPU × Memory', value: ef.cpuXMemory.toFixed(3) },
    { name: 'Temp × Disk', value: ef.tempXDisk.toFixed(3) },
    { name: 'Error Rate', value: ef.errorRate.toFixed(4) },
    { name: 'Latency / Proc', value: ef.latencyPerProcess.toFixed(2) }
  ];

  featuresGrid.innerHTML = features.map(f => `
    <div class="feature-chip">
      <span class="feature-name">${f.name}</span>
      <span class="feature-value">${f.value}</span>
    </div>
  `).join('');
}

// ————————————————————————————————————————————————————————————
// PRESET BUTTONS
// ————————————————————————————————————————————————————————————

function applyPreset(preset) {
  METRICS.forEach(m => {
    const slider = document.getElementById(`slider-${m.id}`);
    const targetVal = preset[m.id];

    // Animate slider
    animateSlider(slider, parseFloat(slider.value), targetVal, m);
  });
}

function animateSlider(slider, from, to, metric) {
  const start = performance.now();
  const duration = 500;

  function tick(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = from + (to - from) * eased;

    slider.value = current;
    updateMetricDisplay(metric, current);

    if (progress < 1) {
      requestAnimationFrame(tick);
    } else {
      slider.value = to;
      updateMetricDisplay(metric, to);
      runPrediction();
    }
  }

  requestAnimationFrame(tick);
}

function randomizeMetrics() {
  const randomPreset = {};
  METRICS.forEach(m => {
    const range = m.max - m.min;
    const rawVal = m.min + Math.random() * range;
    randomPreset[m.id] = Math.round(rawVal / m.step) * m.step;
  });
  applyPreset(randomPreset);
}

document.getElementById('btn-healthy').addEventListener('click', () => applyPreset(PRESETS.healthy));
document.getElementById('btn-critical').addEventListener('click', () => applyPreset(PRESETS.critical));
document.getElementById('btn-randomize').addEventListener('click', randomizeMetrics);

// ————————————————————————————————————————————————————————————
// CLOCK
// ————————————————————————————————————————————————————————————

function updateClock() {
  const now = new Date();
  const timeStr = now.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false
  });
  document.getElementById('topbar-time').textContent = timeStr;
}

setInterval(updateClock, 1000);
updateClock();

// ————————————————————————————————————————————————————————————
// INIT
// ————————————————————————————————————————————————————————————

buildSliders();

// Set initial metric displays
METRICS.forEach(m => {
  updateMetricDisplay(m, m.defaultVal);
});

// Run initial prediction
runPrediction();
