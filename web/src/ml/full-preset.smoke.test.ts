import { defaultDevice, init } from "@jax-js/jax";
import { describe, expect, it } from "vitest";
import type { MetricPoint } from "../protocol";
import { FULL_CONFIG } from "./config";
import { buildDataset } from "./dataset";
import { Trainer } from "./trainer";

describe("full reproduction Wasm smoke test", () => {
  it("completes repeated accumulated full-batch epochs without overflowing memory", async () => {
    await init("wasm");
    defaultDevice("wasm");
    const config = {
      ...FULL_CONFIG,
      totalEpochs: 5,
      metricEvery: 1,
      evaluateEvery: 100,
      backend: "wasm" as const,
    };
    const metrics: MetricPoint[] = [];
    const heatmaps: Int32Array[] = [];
    const trainer = new Trainer(config, buildDataset(config), "wasm", {
      onMetrics(point) {
        metrics.push(point);
      },
      onHeatmap(_epoch, predictions) {
        heatmaps.push(predictions);
      },
      onState() {},
    });

    await trainer.run();

    expect(metrics).toHaveLength(5);
    expect(Number.isFinite(metrics[0]!.trainLoss)).toBe(true);
    expect(Number.isFinite(metrics[0]!.testLoss)).toBe(true);
    expect(heatmaps).toHaveLength(1);
    expect(heatmaps[0]).toHaveLength(config.p * config.p);
  }, 120_000);
});
