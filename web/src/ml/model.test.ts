import { defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { beforeAll, describe, expect, it } from "vitest";
import { DEMO_CONFIG, type GrokkingConfig } from "./config";
import { createModelFunctions, initializeParams } from "./model";

const config: GrokkingConfig = {
  ...DEMO_CONFIG,
  p: 7,
  dModel: 8,
  numHeads: 2,
  dHead: 4,
  dMlp: 16,
  totalEpochs: 2,
};

describe("jax-js grokking model", () => {
  beforeAll(async () => {
    await init("wasm");
    defaultDevice("wasm");
  });

  it("differentiates a whole batch through lookup and causal attention", async () => {
    const params = initializeParams(config);
    const functions = createModelFunctions(config);
    const tokens = np.array(
      new Int32Array([0, 1, 7, 2, 2, 7, 4, 6, 7, 1, 0, 7]),
      { shape: [4, 3], dtype: np.int32 },
    );
    const labels = np.array(new Int32Array([1, 4, 3, 1]), { dtype: np.int32 });

    const [[loss, accuracy], gradients] = functions.valueAndGradient(
      tree.ref(params),
      tokens.ref,
      labels.ref,
    );
    const [lossData, accuracyData] = await Promise.all([loss.data(), accuracy.data()]);

    expect(Number(lossData[0])).toBeGreaterThan(0);
    expect(Number.isFinite(Number(lossData[0]))).toBe(true);
    expect(Number(accuracyData[0])).toBeGreaterThanOrEqual(0);
    expect(Number(accuracyData[0])).toBeLessThanOrEqual(1);
    expect(gradients.tokenEmbedding.shape).toEqual([8, 8]);
    expect(gradients.wQuery.shape).toEqual([8, 8]);

    tree.dispose(gradients);
    tree.dispose(params);
    tokens.dispose();
    labels.dispose();
    functions.dispose();
  });

  it("reports fractional accuracy for a mixed-correctness batch", async () => {
    const params = tree.map(
      (array: np.Array) => array.mul(0),
      initializeParams(config),
    ) as ReturnType<typeof initializeParams>;
    const functions = createModelFunctions(config);
    const tokens = np.array(
      new Int32Array([0, 0, 7, 1, 1, 7, 2, 2, 7, 3, 3, 7]),
      { shape: [4, 3], dtype: np.int32 },
    );
    const labels = np.array(new Int32Array([0, 1, 2, 3]), { dtype: np.int32 });

    const [loss, accuracy] = functions.evaluate(tree.ref(params), tokens.ref, labels.ref);
    const [lossData, accuracyData] = await Promise.all([loss.data(), accuracy.data()]);

    expect(Number(lossData[0])).toBeCloseTo(Math.log(config.p), 5);
    expect(Number(accuracyData[0])).toBeCloseTo(0.25, 6);

    tree.dispose(params);
    tokens.dispose();
    labels.dispose();
    functions.dispose();
  });
});
