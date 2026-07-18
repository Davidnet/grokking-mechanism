import { defaultDevice, init } from "@jax-js/jax";
import { beforeAll, describe, expect, it } from "vitest";
import type { MetricPoint } from "../protocol";
import { DEMO_CONFIG, type GrokkingConfig } from "./config";
import { buildDataset } from "./dataset";
import { resolveTrainChunkSize, Trainer } from "./trainer";

const config: GrokkingConfig = {
  ...DEMO_CONFIG,
  p: 23,
  fractionTrain: 0.6,
  dModel: 8,
  numHeads: 2,
  dHead: 4,
  dMlp: 16,
  totalEpochs: 3,
  metricEvery: 1,
  evaluateEvery: 1,
};

describe("training runtime", () => {
  beforeAll(async () => {
    await init("wasm");
    defaultDevice("wasm");
  });

  it("updates, evaluates, reports, and disposes a complete tiny run", async () => {
    const metrics: MetricPoint[] = [];
    const states: string[] = [];
    const heatmapEpochs: number[] = [];
    const dataset = buildDataset(config);
    expect(dataset.trainSize).toBeGreaterThan(resolveTrainChunkSize("wasm", dataset.trainSize));
    const trainer = new Trainer(config, dataset, "wasm", {
      onMetrics(point) {
        metrics.push(point);
      },
      onHeatmap(epoch, predictions) {
        heatmapEpochs.push(epoch);
        expect(predictions).toHaveLength(config.p * config.p);
      },
      onState(state) {
        states.push(state);
      },
    });

    await trainer.run();

    expect(metrics.map((point) => point.epoch)).toEqual([1, 2, 3]);
    expect(metrics.every((point) => Number.isFinite(point.trainLoss))).toBe(true);
    expect(metrics.every((point) => Number.isFinite(point.testLoss))).toBe(true);
    expect(metrics.at(-1)!.trainLoss).toBeLessThan(metrics[0]!.trainLoss);
    expect(heatmapEpochs).toEqual([1, 2, 3]);
    expect(states).toEqual(["running", "complete"]);
  });
});
