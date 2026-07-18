import "./style.css";
import { MetricChart } from "./dashboard/chart";
import { PredictionHeatmap } from "./dashboard/heatmap";
import { DEMO_CONFIG, FULL_CONFIG, type GrokkingConfig } from "./ml/config";
import type { MetricPoint, TrainingPhase, WorkerCommand, WorkerMessage } from "./protocol";

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("Missing #app element");

app.innerHTML = `
  <main class="app-shell">
    <header class="topbar">
      <div class="brand"><div class="brand-mark"></div><div><h1>Grokking Observatory</h1><p>modular addition · browser-native jax-js</p></div></div>
      <div class="top-status"><span id="status-dot" class="status-dot"></span><span id="status-text">idle · no device</span></div>
    </header>

    <section class="panel control-panel">
      <div class="field"><label for="preset">Experiment</label><select id="preset"><option value="demo">Quick demo</option><option value="full">Full reproduction</option></select></div>
      <div class="field"><label for="backend">Backend</label><select id="backend"><option value="auto">Auto / WebGPU</option><option value="webgpu">WebGPU</option><option value="wasm">Wasm</option></select></div>
      <div class="field"><label for="modulus">Modulus p</label><input id="modulus" type="number" min="5" max="257" /></div>
      <div class="field"><label for="epochs">Epochs</label><input id="epochs" type="number" min="1" max="100000" /></div>
      <div class="field"><label for="learning-rate">Learning rate</label><input id="learning-rate" type="number" min="0.000001" max="1" step="0.0001" /></div>
      <div class="field"><label for="evaluate-every">Evaluate every</label><input id="evaluate-every" type="number" min="1" max="10000" /></div>
      <div class="actions">
        <button id="start" class="btn btn-primary">Start run</button>
        <button id="pause" class="btn" disabled>Pause</button>
        <button id="reset" class="btn" disabled>Reset</button>
        <button id="export" class="btn" disabled>Export</button>
      </div>
    </section>

    <div id="error" class="error-banner"></div>

    <section class="hero-grid">
      <article class="panel phase-panel">
        <div><span class="eyebrow">Learning phase</span><h2 id="phase" class="phase-name">Ready</h2><p id="phase-copy" class="phase-copy">Configure an experiment and watch memorization give way to generalization.</p></div>
        <div id="phase-track" class="phase-track"><span class="phase-segment"></span><span class="phase-segment"></span><span class="phase-segment"></span><span class="phase-segment"></span></div>
      </article>
      <article class="panel metrics-panel">
        <div class="metric-card"><div class="metric-label">Epoch</div><div id="epoch-value" class="metric-value">0</div><div id="epoch-sub" class="metric-sub">of 0</div></div>
        <div class="metric-card"><div class="metric-label">Throughput</div><div id="speed-value" class="metric-value">—</div><div id="speed-sub" class="metric-sub">epochs / second</div></div>
        <div class="metric-card"><div class="metric-label">Train accuracy</div><div id="train-value" class="metric-value violet">—</div><div id="train-loss" class="metric-sub">loss —</div></div>
        <div class="metric-card"><div class="metric-label">Test accuracy</div><div id="test-value" class="metric-value mint">—</div><div id="test-loss" class="metric-sub">loss —</div></div>
      </article>
    </section>

    <section class="dashboard-grid">
      <div class="charts">
        <article class="panel chart-panel"><div class="panel-head"><span class="panel-title">Cross-entropy loss · log/log</span><div class="legend"><span>train</span><span>test</span></div></div><canvas id="loss-chart" class="chart"></canvas></article>
        <article class="panel chart-panel"><div class="panel-head"><span class="panel-title">Accuracy · log epoch</span><div class="legend"><span>train</span><span>test</span></div></div><canvas id="accuracy-chart" class="chart"></canvas></article>
      </div>
      <article class="panel heatmap-panel">
        <div class="panel-head"><span class="panel-title">Prediction field</span><span id="heatmap-epoch" class="eyebrow">not evaluated</span></div>
        <div class="heatmap-wrap"><canvas id="heatmap" class="heatmap"></canvas><div id="heatmap-tooltip" class="heatmap-tooltip"></div></div>
        <div class="heatmap-key"><span class="key-item"><i class="swatch" style="background:#8b6ee9"></i>train · correct</span><span class="key-item"><i class="swatch" style="background:#21cdb0"></i>test · correct</span><span class="key-item"><i class="swatch" style="background:#d15b81"></i>train · wrong</span><span class="key-item"><i class="swatch" style="background:#4a2941"></i>test · wrong</span></div>
        <div id="run-note" class="run-note">Each pixel is one ordered pair (a, b). The field turns mint when the model discovers the modular rule beyond its training examples.</div>
      </article>
    </section>
  </main>`;

function element<T extends HTMLElement>(selector: string): T {
  const result = document.querySelector<T>(selector);
  if (!result) throw new Error(`Missing element ${selector}`);
  return result;
}

const controls = {
  preset: element<HTMLSelectElement>("#preset"),
  backend: element<HTMLSelectElement>("#backend"),
  p: element<HTMLInputElement>("#modulus"),
  epochs: element<HTMLInputElement>("#epochs"),
  learningRate: element<HTMLInputElement>("#learning-rate"),
  evaluateEvery: element<HTMLInputElement>("#evaluate-every"),
  start: element<HTMLButtonElement>("#start"),
  pause: element<HTMLButtonElement>("#pause"),
  reset: element<HTMLButtonElement>("#reset"),
  export: element<HTMLButtonElement>("#export"),
};

const lossChart = new MetricChart(element<HTMLCanvasElement>("#loss-chart"), "loss");
const accuracyChart = new MetricChart(element<HTMLCanvasElement>("#accuracy-chart"), "accuracy");
const heatmap = new PredictionHeatmap(
  element<HTMLCanvasElement>("#heatmap"),
  element<HTMLElement>("#heatmap-tooltip"),
);

let worker: Worker | undefined;
let runningConfig: GrokkingConfig | undefined;
let points: MetricPoint[] = [];
let workerState: "idle" | "initializing" | "running" | "paused" | "complete" = "idle";

const phaseDescriptions: Record<TrainingPhase | "ready", string> = {
  ready: "Configure an experiment and watch memorization give way to generalization.",
  initializing: "Loading the numerical backend and compiling the first differentiated graph.",
  memorizing: "Training accuracy is climbing as the model fits the examples it has seen.",
  plateau: "Training is solved, but held-out examples remain mysterious. Keep watching.",
  grokking: "The hidden modular structure is surfacing across held-out combinations.",
  generalized: "The transformer has discovered the modular rule, not merely the training set.",
};

function applyPreset(config: GrokkingConfig): void {
  controls.p.value = String(config.p);
  controls.epochs.value = String(config.totalEpochs);
  controls.learningRate.value = String(config.learningRate);
  controls.evaluateEvery.value = String(config.evaluateEvery);
}

function readConfig(): GrokkingConfig {
  const base = controls.preset.value === "full" ? FULL_CONFIG : DEMO_CONFIG;
  return {
    ...base,
    p: Number(controls.p.value),
    totalEpochs: Number(controls.epochs.value),
    learningRate: Number(controls.learningRate.value),
    evaluateEvery: Number(controls.evaluateEvery.value),
    backend: controls.backend.value as GrokkingConfig["backend"],
  };
}

function setText(selector: string, value: string): void {
  element<HTMLElement>(selector).textContent = value;
}

function setPhase(phase: TrainingPhase | "ready"): void {
  setText("#phase", phase);
  setText("#phase-copy", phaseDescriptions[phase]);
  const order: Array<TrainingPhase> = ["memorizing", "plateau", "grokking", "generalized"];
  const activeIndex = phase === "ready" || phase === "initializing" ? -1 : order.indexOf(phase);
  document.querySelectorAll(".phase-segment").forEach((segment, index) => {
    segment.classList.toggle("active", index <= activeIndex);
  });
}

function setStatus(label: string, active: boolean): void {
  setText("#status-text", label);
  element("#status-dot").classList.toggle("running", active);
}

function setError(message = ""): void {
  const banner = element("#error");
  banner.textContent = message;
  banner.classList.toggle("visible", Boolean(message));
}

function updateButtons(): void {
  const active = workerState === "initializing" || workerState === "running" || workerState === "paused";
  controls.start.disabled = active;
  controls.pause.disabled = !active || workerState === "initializing";
  controls.pause.textContent = workerState === "paused" ? "Resume" : "Pause";
  controls.reset.disabled = !worker;
  controls.export.disabled = points.length === 0;
  for (const control of [controls.preset, controls.backend, controls.p, controls.epochs, controls.learningRate, controls.evaluateEvery]) {
    control.disabled = active;
  }
}

function updateMetric(point: MetricPoint, phase: TrainingPhase): void {
  points.push(point);
  const config = runningConfig!;
  setText("#epoch-value", point.epoch.toLocaleString());
  setText("#epoch-sub", `of ${config.totalEpochs.toLocaleString()} · ${formatDuration(point.elapsedSeconds)}`);
  setText("#speed-value", point.epochsPerSecond.toFixed(point.epochsPerSecond < 10 ? 1 : 0));
  setText("#train-value", formatPercent(point.trainAccuracy));
  setText("#train-loss", `loss ${formatLoss(point.trainLoss)}`);
  if (point.testAccuracy !== undefined) setText("#test-value", formatPercent(point.testAccuracy));
  if (point.testLoss !== undefined) setText("#test-loss", `loss ${formatLoss(point.testLoss)}`);
  setPhase(phase);
  lossChart.setData(points, config.totalEpochs);
  accuracyChart.setData(points, config.totalEpochs);
}

function handleMessage(message: WorkerMessage): void {
  if (message.type === "ready") {
    const chunked = message.trainChunkSize < message.trainSize;
    setStatus(
      `${message.selectedDevice}${message.selectedDevice === "wasm" ? message.wasmSharedMemory ? " · shared memory" : " · bounded memory" : ""}${chunked ? ` · ${message.trainChunkSize}-sample chunks` : ""} · ${message.trainSize.toLocaleString()} train / ${message.testSize.toLocaleString()} test`,
      true,
    );
    const backendNote = chunked
      ? `Wasm is accumulating an exact full-batch gradient in ${message.trainChunkSize}-example chunks to stay within browser memory.`
      : `Backend ${message.selectedDevice} is executing each optimizer step as one full batch.`;
    const securityNote =
      message.selectedDevice === "wasm" && !message.secureContext
        ? " WebGPU is unavailable over LAN HTTP; use HTTPS or localhost to enable it."
        : "";
    setText(
      "#run-note",
      `${backendNote}${securityNote} Mint pixels are correct held-out predictions.`,
    );
    return;
  }
  if (message.type === "metrics") {
    updateMetric(message.point, message.phase);
    return;
  }
  if (message.type === "heatmap") {
    heatmap.setData(message.predictions, message.labels, message.trainMask, message.p);
    setText("#heatmap-epoch", `epoch ${message.epoch.toLocaleString()}`);
    return;
  }
  if (message.type === "state") {
    if (message.state === "running") workerState = "running";
    if (message.state === "paused") workerState = "paused";
    if (message.state === "complete") {
      workerState = "complete";
      setStatus("complete · run retained locally", false);
    }
    if (message.state === "stopped") workerState = "idle";
    updateButtons();
    return;
  }
  workerState = "idle";
  setError(`${message.message}${message.stack ? `\n${message.stack}` : ""}`);
  setStatus("error · inspect details", false);
  updateButtons();
}

function start(): void {
  reset(false);
  setError();
  runningConfig = readConfig();
  points = [];
  workerState = "initializing";
  lossChart.setData([], runningConfig.totalEpochs);
  accuracyChart.setData([], runningConfig.totalEpochs);
  heatmap.clear();
  setText("#epoch-sub", `of ${runningConfig.totalEpochs.toLocaleString()}`);
  setText("#train-value", "—");
  setText("#test-value", "—");
  setText("#train-loss", "loss —");
  setText("#test-loss", "loss —");
  setText("#heatmap-epoch", "not evaluated");
  setPhase("initializing");
  setStatus("initializing · selecting backend", true);

  worker = new Worker(new URL("./ml/training.worker.ts", import.meta.url), { type: "module" });
  worker.onmessage = (event: MessageEvent<WorkerMessage>) => handleMessage(event.data);
  worker.onerror = (event) => {
    setError(event.message);
    workerState = "idle";
    updateButtons();
  };
  const command: WorkerCommand = { type: "start", config: runningConfig };
  worker.postMessage(command);
  updateButtons();
}

function reset(clearVisuals = true): void {
  if (worker) {
    const stop: WorkerCommand = { type: "stop" };
    worker.postMessage(stop);
    worker.terminate();
    worker = undefined;
  }
  workerState = "idle";
  if (clearVisuals) {
    points = [];
    runningConfig = undefined;
    lossChart.setData([], 1);
    accuracyChart.setData([], 1);
    heatmap.clear();
    setText("#epoch-value", "0");
    setText("#epoch-sub", "of 0");
    setText("#speed-value", "—");
    setText("#train-value", "—");
    setText("#test-value", "—");
    setText("#train-loss", "loss —");
    setText("#test-loss", "loss —");
    setText("#heatmap-epoch", "not evaluated");
    setPhase("ready");
    setStatus("idle · no device", false);
    setError();
  }
  updateButtons();
}

function exportRun(): void {
  if (!runningConfig || points.length === 0) return;
  const blob = new Blob([JSON.stringify({ config: runningConfig, metrics: points }, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `grokking-${runningConfig.p}-${new Date().toISOString().replaceAll(":", "-")}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(value > 0.99 ? 2 : 1)}%`;
}

function formatLoss(value: number): string {
  return value < 0.001 ? value.toExponential(2) : value.toFixed(4);
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${Math.round(seconds % 60)}s`;
}

controls.preset.addEventListener("change", () => {
  applyPreset(controls.preset.value === "full" ? FULL_CONFIG : DEMO_CONFIG);
});
controls.start.addEventListener("click", start);
controls.pause.addEventListener("click", () => {
  if (!worker) return;
  const command: WorkerCommand = { type: workerState === "paused" ? "resume" : "pause" };
  worker.postMessage(command);
});
controls.reset.addEventListener("click", () => reset());
controls.export.addEventListener("click", exportRun);

applyPreset(DEMO_CONFIG);
updateButtons();
