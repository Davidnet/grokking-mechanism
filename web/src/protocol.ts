import type { Device } from "@jax-js/jax";
import type { GrokkingConfig } from "./ml/config";

export type TrainingPhase = "initializing" | "memorizing" | "plateau" | "grokking" | "generalized";

export interface StartCommand {
  type: "start";
  config: GrokkingConfig;
}

export interface PauseCommand {
  type: "pause";
}

export interface ResumeCommand {
  type: "resume";
}

export interface StopCommand {
  type: "stop";
}

export type WorkerCommand = StartCommand | PauseCommand | ResumeCommand | StopCommand;

export interface ReadyMessage {
  type: "ready";
  availableDevices: Device[];
  selectedDevice: Device;
  trainSize: number;
  testSize: number;
  trainChunkSize: number;
  secureContext: boolean;
  wasmSharedMemory: boolean;
}

export interface MetricPoint {
  epoch: number;
  trainLoss: number;
  trainAccuracy: number;
  testLoss?: number;
  testAccuracy?: number;
  elapsedSeconds: number;
  epochsPerSecond: number;
}

export interface MetricsMessage {
  type: "metrics";
  point: MetricPoint;
  phase: TrainingPhase;
}

export interface HeatmapMessage {
  type: "heatmap";
  epoch: number;
  predictions: Int32Array;
  labels: Int32Array;
  trainMask: Uint8Array;
  p: number;
}

export interface StateMessage {
  type: "state";
  state: "running" | "paused" | "complete" | "stopped";
}

export interface ErrorMessage {
  type: "error";
  message: string;
  stack?: string;
}

export type WorkerMessage = ReadyMessage | MetricsMessage | HeatmapMessage | StateMessage | ErrorMessage;
