# Browser Grokking Dashboard Plan

## Goal

Extend this project with an interactive web application that trains the modular-addition transformer locally in the browser using [jax-js](https://github.com/ekzhang/jax-js) and displays the grokking transition in a live dashboard.

The existing Python/JAX implementation remains the reference implementation. The web application will be a separate TypeScript implementation of the same experiment because jax-js cannot directly execute the current Flax NNX classes.

## Recommended architecture

```text
Dashboard UI
    | controls and metric updates
    v
Training Web Worker
    |
    +-- jax-js model and autodiff
    +-- @jax-js/optax AdamW
    +-- WebGPU, falling back to Wasm
```

The preferred design runs training in a dedicated Web Worker so GPU work and metric synchronization do not block dashboard rendering. The initial feasibility spike must confirm WebGPU support in the worker on the target browsers. If that is unreliable, training can run in cooperative chunks on the main thread, yielding to the UI between chunks. Do not enable cross-origin isolation for the current jax-js Wasm fallback: its SharedArrayBuffer backend exceeds the allocator's addressable range on the full preset even with moderate gradient chunks.

No Python web server is required for the first version. The application can be built and deployed as a static site.

## Proposed repository structure

```text
grokking-mechanism/
|-- config.py, data.py, model.py, train.py  # Python reference
|-- mise.toml                              # Reproducible dev toolchain and tasks
|-- scripts/
|   `-- export_web_fixture.py
`-- web/
    |-- package.json
    |-- vite.config.ts
    `-- src/
        |-- ml/
        |   |-- config.ts
        |   |-- dataset.ts
        |   |-- model.ts
        |   |-- trainer.ts
        |   `-- training.worker.ts
        |-- dashboard/
        |   |-- charts.ts
        |   |-- heatmap.ts
        |   `-- controls.ts
        |-- protocol.ts
        `-- main.ts
```

Use Vite and TypeScript for the frontend. Keep the UI framework choice lightweight; it can be vanilla TypeScript, Svelte, or another small framework without changing the ML architecture.

## Development toolchain with mise

Use [mise](https://mise.jdx.dev/) as the repository-level entry point for development tools and common commands. Continue using uv for Python project dependencies, and use mise to provide a pinned Node.js runtime for Vite, TypeScript, npm, and browser tooling.

Add a repository `mise.toml` during the initial frontend setup. Pin Node.js 24 so development does not depend on whichever system Node happens to be installed. npm is included with Node, so a separate global npm installation is unnecessary.

The intended setup flow is:

```bash
mise install
mise exec -- node --version
mise exec -- npm --version
uv sync
```

Run frontend commands through mise, either directly or through mise tasks:

```bash
mise exec -- npm --prefix web install
mise exec -- npm --prefix web run dev
mise exec -- npm --prefix web run test
mise exec -- npm --prefix web run build
```

Define short tasks in `mise.toml` for the normal workflow:

```toml
[tools]
node = "24"

[tasks.web-install]
run = "npm install"
dir = "web"

[tasks.web-dev]
run = "npm run dev"
dir = "web"

[tasks.web-test]
run = "npm run test"
dir = "web"

[tasks.web-build]
run = "npm run build"
dir = "web"

[tasks.python-test]
run = "uv run pytest"
```

This gives contributors a consistent interface:

```bash
mise run web-install
mise run web-dev
mise run web-test
mise run web-build
mise run python-test
```

Commit `mise.toml` and the frontend package lockfile. CI should run `mise install` before invoking the same mise tasks used locally. Avoid installing Vite, TypeScript, test runners, or browser tooling globally; keep them in `web/package.json`.

## Milestone 1: Feasibility spike

Before porting the transformer:

1. Add `mise.toml` with pinned Node.js 24 and the initial frontend tasks.
2. Create the `web/` Vite project using the mise-provided Node/npm toolchain.
3. Install and pin exact versions of `@jax-js/jax` and `@jax-js/optax`.
4. Initialize jax-js and report the available WebGPU and Wasm backends.
5. Validate an embedding lookup and its gradient on both backends. The implemented jax-js 0.1.18 path uses equality-based one-hot matmul because gather-based `take` and `nn.oneHot` fail in the differentiated graph.
6. Confirm that a representative whole-batch forward/loss/gradient graph compiles without using `vmap`.
7. Train a tiny linear or two-layer model in a Web Worker.
8. Verify start, pause, cancel, and reset behavior.
9. Run several thousand steps while monitoring GPU and JavaScript memory.
10. Confirm that the dashboard remains responsive during training.

jax-js uses explicit reference-counted array ownership. Training code must consistently use `.ref` when retaining arrays and dispose of arrays that are no longer needed. A memory soak test is a required part of this milestone.

### Acceptance criteria

- WebGPU training works in a current Chrome/Edge browser.
- Wasm fallback works when WebGPU is unavailable.
- The UI remains interactive during training.
- Repeated start/reset cycles do not cause steadily increasing memory use.

## Milestone 2: Port the experiment

Represent model parameters as a plain TypeScript jax-js tree and implement the model as pure functions.

Port these components:

- A manually managed token-embedding parameter matrix with shape `[p + 1, dModel]` and indexed row lookup.
- A manually managed positional-embedding parameter matrix with shape `[3, dModel]` and indexed row lookup.
- Bias-free Q, K, V, and attention-output projections.
- Four attention heads with head dimension 32.
- Three-token causal attention using jax-js's built-in `dotProductAttention`.
- Attention residual connection.
- Bias-free `128 -> 512 -> 128` ReLU MLP.
- MLP residual connection.
- Final-position unembedding into `p` classes.
- Integer-label softmax cross-entropy.
- Accuracy calculation.
- AdamW with learning rate `1e-3`, `b1=0.9`, `b2=0.98`, and weight decay `1.0`.

Implement a jitted single training step and call it from a JavaScript loop. Do not design around Flax NNX or assume that the Python `nnx.scan` can be translated directly.

Keep the batch dimension explicit throughout the model and use whole-batch `matmul`, `einsum`, embedding lookup, attention, and reductions. Do not apply `vmap` over individual examples: jax-js only partially supports `vmap`, and the model does not need it when every operation is already batched.

jax-js has no `nnx.Embed`-equivalent layer. Embeddings are ordinary learned arrays in the parameter tree, and the forward pass performs lookup explicitly with an equality-based one-hot encoding multiplied by the embedding matrix. Although `take` works for the forward pass, gather-based `take` and `nn.oneHot` fail in the jax-js 0.1.18 differentiated graph; keep this finding covered by the model feasibility test before reconsidering indexed lookup in a future release. Token and position lookup results are both `[batch, 3, dModel]` and are added before attention.

Do not construct the causal mask manually. Reshape the projected Q, K, and V tensors to jax-js's expected `[batch, sequence, heads, headDimension]` layout, call:

```ts
nn.dotProductAttention(q, k, v, { isCausal: true });
```

and reshape the resulting `[batch, 3, 4, 32]` tensor back to `[batch, 3, 128]` before the attention output projection. Reserve the `mask` option for a future experiment that needs an additional custom boolean mask.

### Dataset and split

Dataset generation is easy to reproduce directly in TypeScript:

```text
tokens = [a, b, "="]
label  = (a + b) mod p
```

The current NumPy `default_rng().permutation()` split will not trivially reproduce in JavaScript. For the first version, add `scripts/export_web_fixture.py` to export the existing train/test indices for `p=113`, `seed=0` into a compact browser-readable fixture.

If arbitrary browser seeds become important, adopt one portable PRNG and shuffle implementation in both the Python and TypeScript code.

## Milestone 3: Python/browser parity harness

Exact random initialization is not expected to match automatically across JAX and jax-js. Validate the mathematical implementation with fixed fixtures instead of requiring identical complete training curves.

Add tests for:

- Dataset tokens and modular labels.
- Train/test split fixture loading.
- Token and positional embedding lookup parity, including gradients for repeated token IDs.
- Forward-pass parity with small fixed weights.
- Softmax cross-entropy and accuracy parity.
- One-step gradient parity.
- One-step AdamW update parity.
- Q/K/V head reshaping and parity of built-in causal `dotProductAttention` behavior.
- A reduced `p=13` model that quickly overfits.
- A full-sized run that eventually reaches at least 95% test accuracy.

Small parity fixtures should be JSON or another format both Python and the browser can read easily.

## Milestone 4: Responsive training runtime

The Python loop currently evaluates the entire test set on every epoch. With `p=113` and a 30% training split, that means approximately 3,830 training examples and 8,939 test examples. Full test evaluation at all 40,000 epochs is unnecessarily expensive for an interactive browser run.

Use separate intervals:

- Perform one optimizer update per epoch.
- Collect training metrics every epoch if this does not add a GPU synchronization bottleneck.
- Send training metrics to the UI every 10-25 epochs.
- Evaluate the complete test set every 50-100 epochs.
- Redraw charts at most 10-20 times per second.
- Refresh the prediction heatmap only after a full evaluation.

The worker should send batched messages shaped approximately like:

```ts
type MetricsMessage = {
  type: "metrics";
  epoch: number;
  trainLoss: number;
  trainAccuracy: number;
  testLoss?: number;
  testAccuracy?: number;
  epochsPerSecond: number;
  elapsedSeconds: number;
};
```

Avoid a GPU-to-JavaScript scalar read and a `postMessage` call on every epoch if profiling shows that synchronization is expensive.

### Runtime controls

- Start.
- Pause and resume.
- Reset.
- WebGPU/Wasm backend selection.
- Seed.
- Total epochs.
- Learning rate.
- Test evaluation interval.
- Optional presets for quick demo and full reproduction runs.

## Milestone 5: Live dashboard

The initial dashboard should include:

- Log-scale train and test loss curves.
- Train and test accuracy curves.
- A generalization-gap indicator.
- Current epoch and progress.
- Elapsed time and estimated completion time.
- Epochs per second.
- Active backend and fallback status.
- A phase badge such as `memorizing`, `plateau`, `grokking`, or `generalized`.
- Annotations when test accuracy first crosses 50%, 90%, and 99%.
- A `p x p` heatmap showing correct and incorrect modular-addition predictions.
- A compact view of the experiment configuration.

The heatmap is the main visual differentiator: it should make the sudden transition from memorizing the training split to learning the modular rule visible across all `(a, b)` pairs.

Charts should receive sampled or downsampled data instead of forcing a complete redraw of all 40,000 points after every metric update. Full-resolution metrics may still be retained for export.

## Milestone 6: Persistence and polish

After the end-to-end training experience works:

- Store completed configurations and metrics in IndexedDB.
- Export and import runs as JSON.
- Optionally save browser checkpoints every 1,000 epochs.
- Add a curve overlay comparing a browser run against an existing Python run.
- Add responsive mobile and desktop layouts.
- Add clear unsupported-browser and backend-fallback messages.
- Document the Python and browser workflows in `README.md`.
- Add a static deployment workflow, such as GitHub Pages.

Checkpoint persistence should come after metric persistence. jax-js parameter and optimizer trees are considerably larger and require careful serialization and disposal.

## Performance and reliability risks

### Browser memory leaks

jax-js arrays require explicit ownership management. Mitigate this with helper functions, narrowly scoped training code, reset-cycle tests, and long-running memory soak tests.

### UI stalls

GPU synchronization and converting device arrays to JavaScript can block. Batch metric reads, limit chart redraw frequency, and keep training in a worker when supported.

### Different learning curves

Python and browser initialization and random-number semantics may differ. Use parity fixtures to verify the math and statistical acceptance criteria to verify grokking.

### Browser compatibility

WebGPU support and performance vary. Treat current Chrome/Edge WebGPU as the primary target, provide Wasm fallback, and show the selected backend prominently.

### Long full-sized runs

The existing run first reaches 90% held-out accuracy at roughly epoch 10,762. Provide a reduced quick-demo preset during development and retain the 40,000-epoch configuration as the full reproduction preset. On Wasm, preserve full-batch optimizer semantics by accumulating weighted gradients over bounded chunks before the single AdamW update; also chunk evaluation and heatmap inference to avoid the jax-js allocator's signed 32-bit growth overflow near 2 GiB.

## Definition of done for the first release

- Training runs entirely in the browser through jax-js.
- WebGPU is used when available and Wasm fallback functions correctly.
- The UI remains responsive during a full run.
- Start, pause, resume, and reset work without accumulating array memory.
- The curves visibly reproduce memorization followed by grokking.
- The modular-addition heatmap changes as generalization emerges.
- The browser model passes the small-model Python parity tests.
- Metrics can be exported.
- The application builds into a static deployable site.

## Recommended implementation order

1. Add the mise-managed Node/npm toolchain and common tasks.
2. WebGPU/Wasm and worker feasibility spike.
3. Small jax-js model with memory tests.
4. Dataset, transformer, loss, and AdamW port.
5. Python/browser parity fixtures and tests.
6. Batched worker metric protocol.
7. Loss and accuracy dashboard.
8. Modular-addition prediction heatmap.
9. Persistence, comparison overlays, documentation, and deployment.

## References

- [jax-js repository and tutorial](https://github.com/ekzhang/jax-js)
- [jax-js browser MNIST training demo](https://jax-js.com/mnist)
- [jax-js compatibility table](https://github.com/ekzhang/jax-js/blob/main/FEATURES.md)
- [jax-js Optax package](https://github.com/ekzhang/jax-js/tree/main/packages/optax)
