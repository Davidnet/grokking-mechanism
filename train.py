import json
import time
from pathlib import Path

import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

import data
from config import GrokkingConfig
from model import GrokkingTransformer


def _loss_and_acc(logits, labels):
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    acc = (logits.argmax(-1) == labels).mean()
    return loss, acc


def make_run_chunk(cfg: GrokkingConfig):
    @nnx.jit(static_argnames=("length",))
    def run_chunk(model, optimizer, train_toks, train_lbls, test_toks, test_lbls, length):
        def step(carry):
            model, optimizer = carry

            def loss_fn(model):
                return _loss_and_acc(model(train_toks), train_lbls)

            (train_loss, train_acc), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
            optimizer.update(model, grads)
            test_loss, test_acc = _loss_and_acc(model(test_toks), test_lbls)
            return (model, optimizer), (train_loss, train_acc, test_loss, test_acc)

        return nnx.scan(
            step, length=length, in_axes=nnx.Carry, out_axes=(nnx.Carry, 0)
        )((model, optimizer))

    return run_chunk


def train(cfg: GrokkingConfig, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_json_dict(), indent=2))

    tokens, labels = data.build_dataset(cfg.p)
    train_idx, test_idx = data.split(len(tokens), cfg.seed, cfg.frac_train)
    np.savez(run_dir / "data_split.npz", seed=cfg.seed, train_idx=train_idx, test_idx=test_idx)

    train_tokens, train_labels = tokens[train_idx], labels[train_idx]
    test_tokens, test_labels = tokens[test_idx], labels[test_idx]

    model = GrokkingTransformer(cfg, rngs=nnx.Rngs(cfg.seed))
    tx = optax.adamw(
        learning_rate=cfg.learning_rate,
        b1=cfg.beta1,
        b2=cfg.beta2,
        weight_decay=cfg.weight_decay,
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    run_chunk = make_run_chunk(cfg)

    n_chunks = cfg.total_epochs // cfg.checkpoint_every
    assert n_chunks * cfg.checkpoint_every == cfg.total_epochs

    options = ocp.CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, create=True)
    mgr = ocp.CheckpointManager((run_dir / "checkpoints").resolve(), options=options)

    metrics = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    t_start = time.time()
    for chunk in range(n_chunks):
        (model, optimizer), (tr_loss, tr_acc, te_loss, te_acc) = run_chunk(
            model, optimizer, train_tokens, train_labels, test_tokens, test_labels,
            length=cfg.checkpoint_every,
        )
        metrics["train_loss"].append(np.asarray(tr_loss))
        metrics["train_acc"].append(np.asarray(tr_acc))
        metrics["test_loss"].append(np.asarray(te_loss))
        metrics["test_acc"].append(np.asarray(te_acc))

        _, params = nnx.split(model, nnx.Param)
        mgr.save(chunk, args=ocp.args.StandardSave(params))

        if chunk == 0 or (chunk + 1) % 20 == 0 or chunk == n_chunks - 1:
            epoch = (chunk + 1) * cfg.checkpoint_every
            elapsed = time.time() - t_start
            print(
                f"epoch {epoch:>6d}/{cfg.total_epochs} "
                f"train_loss={tr_loss[-1]:.4f} train_acc={tr_acc[-1]:.4f} "
                f"test_loss={te_loss[-1]:.4f} test_acc={te_acc[-1]:.4f} "
                f"elapsed={elapsed:.1f}s"
            )

    mgr.wait_until_finished()

    epochs = np.arange(1, cfg.total_epochs + 1)
    np.savez(
        run_dir / "metrics.npz",
        epoch=epochs,
        train_loss=np.concatenate(metrics["train_loss"]),
        train_acc=np.concatenate(metrics["train_acc"]),
        test_loss=np.concatenate(metrics["test_loss"]),
        test_acc=np.concatenate(metrics["test_acc"]),
    )
    print(f"done in {time.time() - t_start:.1f}s -> {run_dir}")
    return run_dir


def load_checkpoint(cfg: GrokkingConfig, run_dir: Path, step: int):
    """Restore the model's Param state from a saved checkpoint (step = chunk index,
    i.e. epoch // checkpoint_every - 1)."""
    options = ocp.CheckpointManagerOptions(read_only=True, create=False)
    mgr = ocp.CheckpointManager((run_dir / "checkpoints").resolve(), options=options)
    abstract_model = nnx.eval_shape(lambda: GrokkingTransformer(cfg, rngs=nnx.Rngs(0)))
    graphdef, abstract_params = nnx.split(abstract_model, nnx.Param)
    restored_params = mgr.restore(step, args=ocp.args.StandardRestore(abstract_params))
    return nnx.merge(graphdef, restored_params)
