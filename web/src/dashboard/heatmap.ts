export class PredictionHeatmap {
  private predictions: Int32Array<ArrayBufferLike> = new Int32Array();
  private labels: Int32Array<ArrayBufferLike> = new Int32Array();
  private trainMask: Uint8Array<ArrayBufferLike> = new Uint8Array();
  private p = 0;
  private readonly context: CanvasRenderingContext2D;
  private readonly observer: ResizeObserver;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly tooltip: HTMLElement,
  ) {
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Canvas 2D is unavailable");
    this.context = context;
    this.observer = new ResizeObserver(() => this.draw());
    this.observer.observe(canvas);
    canvas.addEventListener("pointermove", (event) => this.showTooltip(event));
    canvas.addEventListener("pointerleave", () => this.tooltip.classList.remove("visible"));
  }

  setData(
    predictions: Int32Array,
    labels: Int32Array,
    trainMask: Uint8Array,
    p: number,
  ): void {
    this.predictions = predictions;
    this.labels = labels;
    this.trainMask = trainMask;
    this.p = p;
    this.draw();
  }

  clear(): void {
    this.predictions = new Int32Array();
    this.p = 0;
    this.draw();
  }

  private resize(): { width: number; height: number; ratio: number } {
    const rect = this.canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    if (this.canvas.width !== Math.round(width * ratio) || this.canvas.height !== Math.round(height * ratio)) {
      this.canvas.width = Math.round(width * ratio);
      this.canvas.height = Math.round(height * ratio);
    }
    this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { width, height, ratio };
  }

  private draw(): void {
    const { width, height } = this.resize();
    const ctx = this.context;
    ctx.clearRect(0, 0, width, height);

    if (this.p === 0) {
      const gradient = ctx.createLinearGradient(0, 0, width, height);
      gradient.addColorStop(0, "rgba(167, 139, 250, 0.08)");
      gradient.addColorStop(1, "rgba(53, 230, 197, 0.04)");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = "rgba(166, 183, 218, 0.55)";
      ctx.font = "13px Inter, system-ui, sans-serif";
      ctx.fillText("Prediction field appears after evaluation", 20, 32);
      return;
    }

    const cellWidth = width / this.p;
    const cellHeight = height / this.p;
    for (let a = 0; a < this.p; a += 1) {
      for (let b = 0; b < this.p; b += 1) {
        const index = a * this.p + b;
        const correct = this.predictions[index] === this.labels[index];
        const training = this.trainMask[index] === 1;
        ctx.fillStyle = correct
          ? training
            ? "#8b6ee9"
            : "#21cdb0"
          : training
            ? "#d15b81"
            : "#4a2941";
        ctx.fillRect(
          Math.floor(b * cellWidth),
          Math.floor(a * cellHeight),
          Math.ceil(cellWidth + 0.3),
          Math.ceil(cellHeight + 0.3),
        );
      }
    }
  }

  private showTooltip(event: PointerEvent): void {
    if (this.p === 0) return;
    const rect = this.canvas.getBoundingClientRect();
    const b = Math.min(this.p - 1, Math.floor(((event.clientX - rect.left) / rect.width) * this.p));
    const a = Math.min(this.p - 1, Math.floor(((event.clientY - rect.top) / rect.height) * this.p));
    const index = a * this.p + b;
    const prediction = this.predictions[index];
    const label = this.labels[index];
    this.tooltip.innerHTML = `<strong>${a} + ${b} mod ${this.p}</strong><span>pred ${prediction} · true ${label} · ${this.trainMask[index] ? "train" : "test"}</span>`;
    this.tooltip.style.left = `${event.clientX - rect.left + 14}px`;
    this.tooltip.style.top = `${event.clientY - rect.top + 14}px`;
    this.tooltip.classList.add("visible");
  }
}
