import { jit, nn, numpy as np, random, tree, valueAndGrad, type OwnedFunction } from "@jax-js/jax";
import type { GrokkingConfig } from "./config";

export type ModelParams = {
  tokenEmbedding: np.Array;
  positionEmbedding: np.Array;
  wQuery: np.Array;
  wKey: np.Array;
  wValue: np.Array;
  wAttentionOut: np.Array;
  wMlpIn: np.Array;
  wMlpOut: np.Array;
  wUnembed: np.Array;
};

export interface ModelFunctions {
  valueAndGradient: OwnedFunction<
    (params: ModelParams, tokens: np.Array, labels: np.Array) => [[np.Array, np.Array], ModelParams]
  >;
  evaluate: OwnedFunction<
    (params: ModelParams, tokens: np.Array, labels: np.Array) => [np.Array, np.Array]
  >;
  predict: OwnedFunction<(params: ModelParams, tokens: np.Array) => np.Array>;
  dispose(): void;
}

function normalMatrix(seed: number, shape: number[], scale: number): np.Array {
  return random.normal(random.key(seed), shape).mul(scale);
}

function oneHot(indices: np.Array, classes: number): np.Array {
  const expanded = np.expandDims(indices, -1);
  const classIds = np.arange(classes, undefined, undefined, { dtype: np.int32 });
  return np.equal(expanded, classIds).astype(np.float32);
}

function accuracyFromLogits(logits: np.Array, labels: np.Array): np.Array {
  // jax-js preserves the input dtype for mean(), so averaging booleans would
  // cast every non-zero fraction back to true. Cast first to retain the ratio.
  return np.equal(np.argmax(logits, -1), labels).astype(np.float32).mean();
}

export function initializeParams(config: GrokkingConfig): ModelParams {
  let seed = config.seed * 101 + 17;
  const nextSeed = () => {
    seed += 1;
    return seed;
  };

  return {
    tokenEmbedding: normalMatrix(nextSeed(), [config.p + 1, config.dModel], 1 / Math.sqrt(config.p + 1)),
    positionEmbedding: normalMatrix(nextSeed(), [3, config.dModel], 1 / Math.sqrt(3)),
    wQuery: normalMatrix(nextSeed(), [config.dModel, config.dModel], 1 / Math.sqrt(config.dModel)),
    wKey: normalMatrix(nextSeed(), [config.dModel, config.dModel], 1 / Math.sqrt(config.dModel)),
    wValue: normalMatrix(nextSeed(), [config.dModel, config.dModel], 1 / Math.sqrt(config.dModel)),
    wAttentionOut: normalMatrix(nextSeed(), [config.dModel, config.dModel], 1 / Math.sqrt(config.dModel)),
    wMlpIn: normalMatrix(nextSeed(), [config.dModel, config.dMlp], 1 / Math.sqrt(config.dModel)),
    wMlpOut: normalMatrix(nextSeed(), [config.dMlp, config.dModel], 1 / Math.sqrt(config.dMlp)),
    wUnembed: normalMatrix(nextSeed(), [config.dModel, config.p], 1 / Math.sqrt(config.dModel)),
  };
}

export function forward(config: GrokkingConfig, params: ModelParams, tokens: np.Array): np.Array {
  // Avoid gather-based `take` and `nn.oneHot` in differentiated graphs. The
  // equality-based encoding stays fully batched without tracing gather indices.
  const tokenVectors = np.matmul(oneHot(tokens, config.p + 1), params.tokenEmbedding);
  const positions = np.arange(3, undefined, undefined, { dtype: np.int32 });
  const positionVectors = np.matmul(oneHot(positions, 3), params.positionEmbedding);
  const input = tokenVectors.add(positionVectors);

  const query = np
    .matmul(input.ref, params.wQuery)
    .reshape([-1, 3, config.numHeads, config.dHead]);
  const key = np
    .matmul(input.ref, params.wKey)
    .reshape([-1, 3, config.numHeads, config.dHead]);
  const value = np
    .matmul(input.ref, params.wValue)
    .reshape([-1, 3, config.numHeads, config.dHead]);

  const attended = nn
    .dotProductAttention(query, key, value, { isCausal: true })
    .reshape([-1, 3, config.dModel]);
  const afterAttention = input.add(np.matmul(attended, params.wAttentionOut));
  const hidden = nn.relu(np.matmul(afterAttention.ref, params.wMlpIn));
  const afterMlp = afterAttention.add(np.matmul(hidden, params.wMlpOut));
  const finalPosition = afterMlp.slice([], 2, []);
  return np.matmul(finalPosition, params.wUnembed);
}

function lossAndAccuracy(
  config: GrokkingConfig,
  params: ModelParams,
  tokens: np.Array,
  labels: np.Array,
): [np.Array, np.Array] {
  const logits = forward(config, params, tokens);
  const targets = oneHot(labels.ref, config.p);
  const loss = nn.logSoftmax(logits.ref, -1).mul(targets).sum(-1).neg().mean();
  const accuracy = accuracyFromLogits(logits, labels);
  return [loss, accuracy];
}

export function createModelFunctions(config: GrokkingConfig): ModelFunctions {
  const objective = (params: ModelParams, tokens: np.Array, labels: np.Array) =>
    lossAndAccuracy(config, params, tokens, labels);
  const differentiated = valueAndGrad(objective, { hasAux: true });
  const valueAndGradient = jit(differentiated) as ModelFunctions["valueAndGradient"];

  const evaluate = jit((params: ModelParams, tokens: np.Array, labels: np.Array) => {
    const logits = forward(config, params, tokens);
    const targets = oneHot(labels.ref, config.p);
    const loss = nn.logSoftmax(logits.ref, -1).mul(targets).sum(-1).neg().mean();
    const accuracy = accuracyFromLogits(logits, labels);
    return [loss, accuracy];
  }) as ModelFunctions["evaluate"];

  const predict = jit((params: ModelParams, tokens: np.Array) =>
    np.argmax(forward(config, params, tokens), -1),
  ) as ModelFunctions["predict"];

  return {
    valueAndGradient,
    evaluate,
    predict,
    dispose() {
      valueAndGradient.dispose();
      evaluate.dispose();
      predict.dispose();
    },
  };
}

export function disposeParams(params: ModelParams | undefined): void {
  if (params) tree.dispose(params);
}
