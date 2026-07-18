import type { MetricPoint } from "../protocol";

type ChartKind = "loss" | "accuracy";

const COLORS = {
  train: "#a78bfa",
  test: "#35e6c5",
  grid: "rgba(166, 183, 218, 0.12)",
  text: "#8290ad",
};

export class MetricChart {
  private points: MetricPoint[] = [];
  private totalEpochs = 1;
  private readonly context: CanvasRenderingContext2D;
  private readonly observer: ResizeObserver;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly kind: ChartKind,
  ) {
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Canvas 2D is unavailable");
    this.context = context;
    this.observer = new ResizeObserver(() => this.draw());
    this.observer.observe(canvas);
  }

  setData(points: MetricPoint[], totalEpochs: number): void {
    this.points = points;
    this.totalEpochs = Math.max(1, totalEpochs);
    this.draw();
  }

  private resize(): { width: number; height: number } {
    const rect = this.canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(rect.width * ratio));
    const height = Math.max(1, Math.round(rect.height * ratio));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { width: rect.width, height: rect.height };
  }

  private draw(): void {
    const { width, height } = this.resize();
    const ctx = this.context;
    ctx.clearRect(0, 0, width, height);

    const frame = { left: 48, right: width - 18, top: 18, bottom: height - 30 };
    const plotWidth = Math.max(1, frame.right - frame.left);
    const plotHeight = Math.max(1, frame.bottom - frame.top);

    ctx.strokeStyle = COLORS.grid;
    ctx.fillStyle = COLORS.text;
    ctx.font = "11px Inter, system-ui, sans-serif";
    ctx.lineWidth = 1;

    for (let index = 0; index <= 4; index += 1) {
      const y = frame.top + (plotHeight * index) / 4;
      ctx.beginPath();
      ctx.moveTo(frame.left, y);
      ctx.lineTo(frame.right, y);
      ctx.stroke();
      const label = this.kind === "accuracy" ? `${Math.round((1 - index / 4) * 100)}%` : "";
      if (label) ctx.fillText(label, 8, y + 4);
    }

    const xTicks = [1, 10, 100, 1_000, 10_000, this.totalEpochs]
      .filter((value, index, values) => value <= this.totalEpochs && values.indexOf(value) === index);
    for (const epoch of xTicks) {
      const x = this.x(epoch, frame.left, plotWidth);
      ctx.beginPath();
      ctx.moveTo(x, frame.top);
      ctx.lineTo(x, frame.bottom);
      ctx.stroke();
      ctx.fillText(formatEpoch(epoch), x - 8, height - 9);
    }

    if (this.points.length === 0) {
      ctx.fillStyle = "rgba(166, 183, 218, 0.55)";
      ctx.font = "13px Inter, system-ui, sans-serif";
      ctx.fillText("Waiting for the first compiled step…", frame.left + 14, frame.top + 28);
      return;
    }

    let lossMin = 1e-5;
    let lossMax = 10;
    if (this.kind === "loss") {
      const values = this.points.flatMap((point) =>
        point.testLoss === undefined ? [point.trainLoss] : [point.trainLoss, point.testLoss],
      );
      lossMin = Math.max(1e-9, Math.min(...values.filter((value) => value > 0)) * 0.5);
      lossMax = Math.max(...values, 1) * 1.25;
      ctx.fillText(lossMax.toExponential(0), 5, frame.top + 4);
      ctx.fillText(lossMin.toExponential(0), 5, frame.bottom + 4);
    }

    this.drawSeries("train", frame, plotWidth, plotHeight, lossMin, lossMax);
    this.drawSeries("test", frame, plotWidth, plotHeight, lossMin, lossMax);
  }

  private x(epoch: number, left: number, width: number): number {
    return left + (Math.log10(Math.max(1, epoch)) / Math.log10(Math.max(10, this.totalEpochs))) * width;
  }

  private y(value: number, top: number, height: number, lossMin: number, lossMax: number): number {
    if (this.kind === "accuracy") return top + (1 - Math.max(0, Math.min(1, value))) * height;
    const low = Math.log10(lossMin);
    const high = Math.log10(lossMax);
    const position = (Math.log10(Math.max(lossMin, value)) - low) / Math.max(1e-9, high - low);
    return top + (1 - position) * height;
  }

  private drawSeries(
    series: "train" | "test",
    frame: { left: number; right: number; top: number; bottom: number },
    width: number,
    height: number,
    lossMin: number,
    lossMax: number,
  ): void {
    const ctx = this.context;
    const valueOf = (point: MetricPoint): number | undefined => {
      if (this.kind === "loss") return series === "train" ? point.trainLoss : point.testLoss;
      return series === "train" ? point.trainAccuracy : point.testAccuracy;
    };

    ctx.beginPath();
    ctx.strokeStyle = COLORS[series];
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    let started = false;
    for (const point of this.points) {
      const value = valueOf(point);
      if (value === undefined || !Number.isFinite(value)) continue;
      const x = this.x(point.epoch, frame.left, width);
      const y = this.y(value, frame.top, height, lossMin, lossMax);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  }
}

function formatEpoch(epoch: number): string {
  if (epoch >= 1_000) return `${Math.round(epoch / 1_000)}k`;
  return String(epoch);
}
