/// <reference lib="webworker" />

import { defaultDevice, init, type Device } from "@jax-js/jax";
import type { WorkerCommand, WorkerMessage } from "../protocol";
import { validateConfig } from "./config";
import { buildDataset } from "./dataset";
import { resolveTrainChunkSize, Trainer } from "./trainer";

const worker = self as DedicatedWorkerGlobalScope;
let trainer: Trainer | undefined;
let availableDevices: Device[] = [];

function send(message: WorkerMessage, transfer: Transferable[] = []): void {
  worker.postMessage(message, transfer);
}

async function selectDevice(preference: "auto" | "webgpu" | "wasm"): Promise<Device> {
  if (availableDevices.length === 0) availableDevices = await init();
  if (preference === "webgpu" && !availableDevices.includes("webgpu")) {
    throw new Error(
      worker.isSecureContext
        ? "WebGPU was requested but is not available in this browser or worker."
        : "WebGPU requires a secure context. Use HTTPS or open the dashboard through localhost.",
    );
  }
  const preferred = preference === "auto" ? "webgpu" : preference;
  const selected: Device = availableDevices.includes(preferred)
    ? preferred
    : availableDevices.includes("wasm")
      ? "wasm"
      : "cpu";
  defaultDevice(selected);
  return selected;
}

async function start(command: Extract<WorkerCommand, { type: "start" }>): Promise<void> {
  if (trainer) throw new Error("A training run is already active");
  validateConfig(command.config);
  const selectedDevice = await selectDevice(command.config.backend);
  const dataset = buildDataset(command.config);

  send({
    type: "ready",
    availableDevices,
    selectedDevice,
    trainSize: dataset.trainSize,
    testSize: dataset.testSize,
    trainChunkSize: resolveTrainChunkSize(selectedDevice, dataset.trainSize),
    secureContext: worker.isSecureContext,
    wasmSharedMemory: typeof SharedArrayBuffer !== "undefined",
  });

  trainer = new Trainer(command.config, dataset, selectedDevice, {
    onMetrics(point, phase) {
      send({ type: "metrics", point, phase });
    },
    onHeatmap(epoch, predictions) {
      const labels = dataset.allLabels.slice();
      const trainMask = dataset.trainMask.slice();
      send(
        {
          type: "heatmap",
          epoch,
          predictions,
          labels,
          trainMask,
          p: command.config.p,
        },
        [predictions.buffer, labels.buffer, trainMask.buffer],
      );
    },
    onState(state) {
      send({ type: "state", state });
    },
  });

  try {
    await trainer.run();
  } finally {
    trainer = undefined;
  }
}

worker.onmessage = (event: MessageEvent<WorkerCommand>) => {
  const command = event.data;
  if (command.type === "pause") {
    trainer?.pause();
    return;
  }
  if (command.type === "resume") {
    trainer?.resume();
    return;
  }
  if (command.type === "stop") {
    trainer?.stop();
    return;
  }

  void start(command).catch((error: unknown) => {
    const normalized = error instanceof Error ? error : new Error(String(error));
    const allocatorFailure = normalized.message.includes("WebAssembly.Memory.grow");
    send({
      type: "error",
      message: allocatorFailure
        ? "The Wasm allocator ran out of addressable memory. Reduce the model size or use WebGPU over HTTPS/localhost."
        : normalized.message,
      stack: normalized.stack,
    });
    trainer?.stop();
    trainer = undefined;
  });
};
