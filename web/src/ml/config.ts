export type BackendPreference = "auto" | "webgpu" | "wasm";

export interface GrokkingConfig {
  p: number;
  fractionTrain: number;
  seed: number;
  dModel: number;
  numHeads: number;
  dHead: number;
  dMlp: number;
  learningRate: number;
  beta1: number;
  beta2: number;
  weightDecay: number;
  totalEpochs: number;
  metricEvery: number;
  evaluateEvery: number;
  backend: BackendPreference;
}

export const FULL_CONFIG: GrokkingConfig = {
  p: 113,
  fractionTrain: 0.3,
  seed: 0,
  dModel: 128,
  numHeads: 4,
  dHead: 32,
  dMlp: 512,
  learningRate: 1e-3,
  beta1: 0.9,
  beta2: 0.98,
  weightDecay: 1.0,
  totalEpochs: 40_000,
  metricEvery: 10,
  evaluateEvery: 100,
  backend: "auto",
};

export const DEMO_CONFIG: GrokkingConfig = {
  ...FULL_CONFIG,
  p: 31,
  fractionTrain: 0.4,
  dModel: 64,
  dHead: 16,
  dMlp: 128,
  totalEpochs: 8_000,
  metricEvery: 5,
  evaluateEvery: 50,
};

export function validateConfig(config: GrokkingConfig): void {
  if (!Number.isInteger(config.p) || config.p < 5) {
    throw new Error("p must be an integer of at least 5");
  }
  if (config.fractionTrain <= 0 || config.fractionTrain >= 1) {
    throw new Error("fractionTrain must be between 0 and 1");
  }
  if (config.dModel !== config.numHeads * config.dHead) {
    throw new Error("dModel must equal numHeads * dHead");
  }
  if (config.totalEpochs < 1 || config.metricEvery < 1 || config.evaluateEvery < 1) {
    throw new Error("epoch and reporting intervals must be positive");
  }
}
