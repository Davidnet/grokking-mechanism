import { describe, expect, it } from "vitest";
import { buildDataset, shuffledIndices } from "./dataset";

describe("modular addition dataset", () => {
  it("builds every ordered pair with the equality token", () => {
    const dataset = buildDataset({ p: 5, fractionTrain: 0.4, seed: 7 });

    expect(dataset.allTokens).toHaveLength(75);
    expect(dataset.allLabels).toHaveLength(25);
    expect(Array.from(dataset.allTokens.slice(0, 6))).toEqual([0, 0, 5, 0, 1, 5]);
    expect(Array.from(dataset.allLabels.slice(0, 7))).toEqual([0, 1, 2, 3, 4, 1, 2]);
    expect(dataset.trainSize).toBe(10);
    expect(dataset.testSize).toBe(15);
    expect(dataset.trainMask.reduce((sum, value) => sum + value, 0)).toBe(10);
  });

  it("produces a deterministic seeded permutation", () => {
    expect(Array.from(shuffledIndices(12, 42))).toEqual(
      Array.from(shuffledIndices(12, 42)),
    );
    expect(Array.from(shuffledIndices(12, 42))).not.toEqual(
      Array.from(shuffledIndices(12, 43)),
    );
  });
});
