import type { GrokkingConfig } from "./config";

export interface DatasetSplit {
  allTokens: Int32Array;
  allLabels: Int32Array;
  trainTokens: Int32Array;
  trainLabels: Int32Array;
  testTokens: Int32Array;
  testLabels: Int32Array;
  trainMask: Uint8Array;
  trainSize: number;
  testSize: number;
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}

export function shuffledIndices(size: number, seed: number): Int32Array {
  const indices = Int32Array.from({ length: size }, (_, index) => index);
  const random = mulberry32(seed);
  for (let index = size - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    const value = indices[index]!;
    indices[index] = indices[swapIndex]!;
    indices[swapIndex] = value;
  }
  return indices;
}

export function buildDataset(config: Pick<GrokkingConfig, "p" | "fractionTrain" | "seed">): DatasetSplit {
  const size = config.p * config.p;
  const allTokens = new Int32Array(size * 3);
  const allLabels = new Int32Array(size);

  for (let a = 0; a < config.p; a += 1) {
    for (let b = 0; b < config.p; b += 1) {
      const index = a * config.p + b;
      allTokens[index * 3] = a;
      allTokens[index * 3 + 1] = b;
      allTokens[index * 3 + 2] = config.p;
      allLabels[index] = (a + b) % config.p;
    }
  }

  const permutation = shuffledIndices(size, config.seed);
  const trainSize = Math.floor(size * config.fractionTrain);
  const testSize = size - trainSize;
  const trainTokens = new Int32Array(trainSize * 3);
  const trainLabels = new Int32Array(trainSize);
  const testTokens = new Int32Array(testSize * 3);
  const testLabels = new Int32Array(testSize);
  const trainMask = new Uint8Array(size);

  for (let splitIndex = 0; splitIndex < size; splitIndex += 1) {
    const sourceIndex = permutation[splitIndex]!;
    const isTrain = splitIndex < trainSize;
    const destinationIndex = isTrain ? splitIndex : splitIndex - trainSize;
    const destinationTokens = isTrain ? trainTokens : testTokens;
    const destinationLabels = isTrain ? trainLabels : testLabels;

    destinationTokens[destinationIndex * 3] = allTokens[sourceIndex * 3]!;
    destinationTokens[destinationIndex * 3 + 1] = allTokens[sourceIndex * 3 + 1]!;
    destinationTokens[destinationIndex * 3 + 2] = allTokens[sourceIndex * 3 + 2]!;
    destinationLabels[destinationIndex] = allLabels[sourceIndex]!;
    if (isTrain) trainMask[sourceIndex] = 1;
  }

  return {
    allTokens,
    allLabels,
    trainTokens,
    trainLabels,
    testTokens,
    testLabels,
    trainMask,
    trainSize,
    testSize,
  };
}
