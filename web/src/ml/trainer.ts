import {
  blockUntilReady,
  numpy as np,
  tree,
  type Device,
} from "@jax-js/jax";
import { adamw, applyUpdates, type GradientTransformation, type OptState } from "@jax-js/optax";
import type { MetricPoint, TrainingPhase } from "../protocol";
import type { GrokkingConfig } from "./config";
import type { DatasetSplit } from "./dataset";
import {
  createModelFunctions,
  disposeParams,
  initializeParams,
  type ModelFunctions,
  type ModelParams,
} from "./model";

interface DeviceDataset {
  trainTokens: np.Array;
  trainLabels: np.Array;
  testTokens: np.Array;
  testLabels: np.Array;
  allTokens: np.Array;
}

const WASM_TRAIN_CHUNK_SIZE = 128;
const WASM_EVALUATION_CHUNK_SIZE = 256;

export function resolveTrainChunkSize(device: Device, trainSize: number): number {
  return device === "wasm" ? Math.min(WASM_TRAIN_CHUNK_SIZE, trainSize) : trainSize;
}

function resolveEvaluationChunkSize(device: Device, size: number): number {
  return device === "wasm" ? Math.min(WASM_EVALUATION_CHUNK_SIZE, size) : size;
}

function scaleGradient(gradients: ModelParams, weight: number): ModelParams {
  return tree.map((array: np.Array) => array.mul(weight), gradients) as ModelParams;
}

function addGradients(left: ModelParams, right: ModelParams): ModelParams {
  return tree.map(
    (leftArray: np.Array, rightArray: np.Array) => leftArray.add(rightArray),
    left,
    right,
  ) as ModelParams;
}

function addWeightedScalar(
  accumulated: np.Array | undefined,
  value: np.Array,
  weight: number,
): np.Array {
  const weighted = value.mul(weight);
  return accumulated ? accumulated.add(weighted) : weighted;
}

export interface TrainerCallbacks {
  onMetrics(point: MetricPoint, phase: TrainingPhase): void;
  onHeatmap(epoch: number, predictions: Int32Array): void;
  onState(state: "running" | "paused" | "complete" | "stopped"): void;
}

function uploadDataset(dataset: DatasetSplit, device: Device): DeviceDataset {
  return {
    trainTokens: np.array(dataset.trainTokens, {
      shape: [dataset.trainSize, 3],
      dtype: np.int32,
      device,
    }),
    trainLabels: np.array(dataset.trainLabels, { dtype: np.int32, device }),
    testTokens: np.array(dataset.testTokens, {
      shape: [dataset.testSize, 3],
      dtype: np.int32,
      device,
    }),
    testLabels: np.array(dataset.testLabels, { dtype: np.int32, device }),
    allTokens: np.array(dataset.allTokens, {
      shape: [dataset.allLabels.length, 3],
      dtype: np.int32,
      device,
    }),
  };
}

async function scalar(array: np.Array): Promise<number> {
  const data = await array.data();
  const value = data[0];
  if (value === undefined) throw new Error("Expected a scalar array");
  return Number(value);
}

function classifyPhase(trainAccuracy: number, testAccuracy: number | undefined): TrainingPhase {
  if (trainAccuracy < 0.95) return "memorizing";
  if (testAccuracy === undefined || testAccuracy < 0.2) return "plateau";
  if (testAccuracy < 0.9) return "grokking";
  return "generalized";
}

function yieldToWorker(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

export class Trainer {
  private params: ModelParams | undefined;
  private optimizerState: OptState | undefined;
  private optimizer: GradientTransformation | undefined;
  private functions: ModelFunctions | undefined;
  private arrays: DeviceDataset | undefined;
  private paused = false;
  private stopped = false;
  private resumeWaiters: Array<() => void> = [];

  constructor(
    private readonly config: GrokkingConfig,
    private readonly dataset: DatasetSplit,
    private readonly device: Device,
    private readonly callbacks: TrainerCallbacks,
  ) {}

  pause(): void {
    if (this.stopped || this.paused) return;
    this.paused = true;
    this.callbacks.onState("paused");
  }

  resume(): void {
    if (this.stopped || !this.paused) return;
    this.paused = false;
    for (const resolve of this.resumeWaiters.splice(0)) resolve();
    this.callbacks.onState("running");
  }

  stop(): void {
    this.stopped = true;
    this.paused = false;
    for (const resolve of this.resumeWaiters.splice(0)) resolve();
  }

  private async waitWhilePaused(): Promise<void> {
    if (!this.paused) return;
    await new Promise<void>((resolve) => this.resumeWaiters.push(resolve));
  }

  private fullBatchGradient(
    functions: ModelFunctions,
    params: ModelParams,
    arrays: DeviceDataset,
  ): [[np.Array, np.Array], ModelParams] {
    const total = this.dataset.trainSize;
    const chunkSize = resolveTrainChunkSize(this.device, total);
    let accumulatedLoss: np.Array | undefined;
    let accumulatedAccuracy: np.Array | undefined;
    let accumulatedGradients: ModelParams | undefined;

    for (let start = 0; start < total; start += chunkSize) {
      const end = Math.min(total, start + chunkSize);
      const count = end - start;
      const weight = count / total;
      const tokens = arrays.trainTokens.ref.slice([start, end], []);
      const labels = arrays.trainLabels.ref.slice([start, end]);
      const result = functions.valueAndGradient(tree.ref(params), tokens, labels);
      const [[loss, accuracy], gradients]: [[np.Array, np.Array], ModelParams] = result;

      accumulatedLoss = addWeightedScalar(accumulatedLoss, loss, weight);
      accumulatedAccuracy = addWeightedScalar(accumulatedAccuracy, accuracy, weight);
      const weightedGradients = scaleGradient(gradients, weight);
      accumulatedGradients = accumulatedGradients
        ? addGradients(accumulatedGradients, weightedGradients)
        : weightedGradients;
    }

    if (!accumulatedLoss || !accumulatedAccuracy || !accumulatedGradients) {
      throw new Error("Cannot compute a gradient for an empty training split");
    }
    return [[accumulatedLoss, accumulatedAccuracy], accumulatedGradients];
  }

  private evaluateBatched(
    functions: ModelFunctions,
    params: ModelParams,
    tokens: np.Array,
    labels: np.Array,
    total: number,
  ): [np.Array, np.Array] {
    const chunkSize = resolveEvaluationChunkSize(this.device, total);
    let accumulatedLoss: np.Array | undefined;
    let accumulatedAccuracy: np.Array | undefined;

    for (let start = 0; start < total; start += chunkSize) {
      const end = Math.min(total, start + chunkSize);
      const weight = (end - start) / total;
      const chunkTokens = tokens.ref.slice([start, end], []);
      const chunkLabels = labels.ref.slice([start, end]);
      const [loss, accuracy] = functions.evaluate(
        tree.ref(params),
        chunkTokens,
        chunkLabels,
      );
      accumulatedLoss = addWeightedScalar(accumulatedLoss, loss, weight);
      accumulatedAccuracy = addWeightedScalar(accumulatedAccuracy, accuracy, weight);
    }

    if (!accumulatedLoss || !accumulatedAccuracy) {
      throw new Error("Cannot evaluate an empty split");
    }
    return [accumulatedLoss, accumulatedAccuracy];
  }

  private async predictBatched(
    functions: ModelFunctions,
    params: ModelParams,
    tokens: np.Array,
    total: number,
  ): Promise<Int32Array> {
    const chunkSize = resolveEvaluationChunkSize(this.device, total);
    const predictions = new Int32Array(total);
    for (let start = 0; start < total; start += chunkSize) {
      const end = Math.min(total, start + chunkSize);
      const chunkTokens = tokens.ref.slice([start, end], []);
      const chunkPredictions = functions.predict(tree.ref(params), chunkTokens);
      const predictionData = await chunkPredictions.data();
      predictions.set(predictionData, start);
    }
    return predictions;
  }

  async run(): Promise<void> {
    const arrays = uploadDataset(this.dataset, this.device);
    let params = initializeParams(this.config);
    const functions = createModelFunctions(this.config);
    const optimizer = adamw(this.config.learningRate, {
      b1: this.config.beta1,
      b2: this.config.beta2,
      weightDecay: this.config.weightDecay,
    });
    let optimizerState = optimizer.init(tree.ref(params));
    this.arrays = arrays;
    this.params = params;
    this.functions = functions;
    this.optimizer = optimizer;
    this.optimizerState = optimizerState;
    this.callbacks.onState("running");

    const startedAt = performance.now();
    let lastTrainLoss = Number.NaN;
    let lastTrainAccuracy = 0;
    let lastTestLoss: number | undefined;
    let lastTestAccuracy: number | undefined;

    try {
      for (let epoch = 1; epoch <= this.config.totalEpochs; epoch += 1) {
        await this.waitWhilePaused();
        if (this.stopped) break;

        const [[loss, accuracy], gradients] = this.fullBatchGradient(
          functions,
          params,
          arrays,
        );
        const [updates, nextOptimizerState]: [ModelParams, OptState] = optimizer.update(
          gradients,
          optimizerState,
          tree.ref(params),
        );
        optimizerState = nextOptimizerState;
        params = applyUpdates(params, updates);
        this.optimizerState = optimizerState;
        this.params = params;

        const shouldEvaluate = epoch === 1 || epoch % this.config.evaluateEvery === 0;
        const shouldReport =
          epoch === 1 || epoch % this.config.metricEvery === 0 || shouldEvaluate;

        if (shouldReport) {
          [lastTrainLoss, lastTrainAccuracy] = await Promise.all([scalar(loss), scalar(accuracy)]);
        } else {
          loss.dispose();
          accuracy.dispose();
        }

        if (shouldEvaluate) {
          const [testLoss, testAccuracy] = this.evaluateBatched(
            functions,
            params,
            arrays.testTokens,
            arrays.testLabels,
            this.dataset.testSize,
          );
          [lastTestLoss, lastTestAccuracy] = await Promise.all([
            scalar(testLoss),
            scalar(testAccuracy),
          ]);

          const predictionCopy = await this.predictBatched(
            functions,
            params,
            arrays.allTokens,
            this.dataset.allLabels.length,
          );
          this.callbacks.onHeatmap(epoch, predictionCopy);
        }

        if (shouldReport) {
          const elapsedSeconds = (performance.now() - startedAt) / 1_000;
          const point: MetricPoint = {
            epoch,
            trainLoss: lastTrainLoss,
            trainAccuracy: lastTrainAccuracy,
            elapsedSeconds,
            epochsPerSecond: epoch / Math.max(elapsedSeconds, 1e-6),
          };
          if (shouldEvaluate) {
            point.testLoss = lastTestLoss;
            point.testAccuracy = lastTestAccuracy;
          }
          this.callbacks.onMetrics(
            point,
            classifyPhase(lastTrainAccuracy, lastTestAccuracy),
          );
          await blockUntilReady(params);
          await yieldToWorker();
        }
      }

      this.callbacks.onState(this.stopped ? "stopped" : "complete");
    } finally {
      this.dispose();
    }
  }

  dispose(): void {
    this.functions?.dispose();
    this.functions = undefined;
    disposeParams(this.params);
    this.params = undefined;
    if (this.optimizerState) tree.dispose(this.optimizerState);
    this.optimizerState = undefined;
    if (this.arrays) tree.dispose(this.arrays);
    this.arrays = undefined;
  }
}
