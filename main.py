import argparse
from datetime import datetime
from pathlib import Path

from config import GrokkingConfig

RUNS_DIR = Path(__file__).parent / "runs"


def latest_run_dir() -> Path:
    runs = sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()) if RUNS_DIR.exists() else []
    if not runs:
        raise SystemExit(f"no runs found under {RUNS_DIR}")
    return runs[-1]


def cmd_train(args):
    import train as train_module

    cfg = GrokkingConfig(
        seed=args.seed,
        total_epochs=args.epochs,
        checkpoint_every=args.checkpoint_every,
    )
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / run_id
    train_module.train(cfg, run_dir)


def cmd_plot(args):
    import plot as plot_module

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    plot_module.plot(run_dir)


def main():
    parser = argparse.ArgumentParser(description="Grokking on modular addition (a + b) mod p")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train and checkpoint a run")
    p_train.add_argument("--epochs", type=int, default=GrokkingConfig.total_epochs)
    p_train.add_argument("--checkpoint-every", type=int, default=GrokkingConfig.checkpoint_every)
    p_train.add_argument("--seed", type=int, default=GrokkingConfig.seed)
    p_train.add_argument("--run-id", type=str, default=None)
    p_train.set_defaults(func=cmd_train)

    p_plot = sub.add_parser("plot", help="plot the training curve for a run")
    p_plot.add_argument("--run-dir", type=str, default=None, help="defaults to the most recent run")
    p_plot.set_defaults(func=cmd_plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
