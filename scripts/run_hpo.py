"""Hyperparameter optimisation for Lenia world models using Optuna.

Each trial trains a model for a short number of epochs (``hpo.hpo_epochs`` in
config.yaml) and reports the final validation loss to the study.  Bad trials
are pruned early by a MedianPruner.

Usage:
    # Basic run — 50 trials, results only in memory
    python scripts/run_hpo.py --setup pixel
    python scripts/run_hpo.py --setup jepa --n-trials 30

    # Persistent study — survives interruptions, can be resumed
    python scripts/run_hpo.py --setup pixel \\
        --storage sqlite:///experiments/hpo.db \\
        --study-name pixel_hpo

    # Resume an existing study (same --storage + --study-name)
    python scripts/run_hpo.py --setup pixel \\
        --storage sqlite:///experiments/hpo.db \\
        --study-name pixel_hpo --n-trials 20

    # Parallel trials (requires a shared persistent storage backend)
    python scripts/run_hpo.py --setup jepa \\
        --storage sqlite:///experiments/hpo.db \\
        --n-jobs 4

    # Quick smoke-test with dummy data
    python scripts/run_hpo.py --setup pixel --dummy --n-trials 3 --hpo-epochs 2

Notes:
    - Best hyperparameters are saved to experiments/hpo_{setup}_best_params.yaml
      after the study finishes.
    - Per-trial TensorBoard logs land in experiments/hpo_{setup}_trial_XXXX/.
    - Set ``num_workers: 0`` in config.yaml when using ``--n-jobs > 1`` to avoid
      DataLoader fork conflicts.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

import optuna
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_single import SEED, train_trial


# ─── Search space ─────────────────────────────────────────────────────────────

def _suggest_hyperparams(trial: optuna.Trial, setup_type: str, ss: dict) -> dict:
    """Ask the trial to suggest values for all tuneable hyperparameters.

    The search space is read from config.yaml under ``hpo.search_space``.
    Defaults are used when a key is absent from the config.

    Args:
        trial: Active Optuna trial.
        setup_type: "pixel" or "jepa".
        ss: The ``hpo.search_space`` sub-dict from config.yaml.

    Returns:
        Dict mapping hyperparameter names to their suggested values.
    """
    overrides: dict = {}

    # ── Common to both setups ─────────────────────────────────────────────────
    overrides["embed_dim"] = trial.suggest_categorical(
        "embed_dim", ss.get("embed_dim", [64, 128, 256])
    )
    overrides["learning_rate"] = trial.suggest_float(
        "learning_rate", *ss.get("learning_rate", [1e-4, 1e-2]), log=True
    )
    overrides["weight_decay"] = trial.suggest_float(
        "weight_decay", *ss.get("weight_decay", [1e-6, 1e-2]), log=True
    )
    overrides["batch_size"] = trial.suggest_categorical(
        "batch_size", ss.get("batch_size", [32, 64, 128])
    )

    # ── JEPA-only ─────────────────────────────────────────────────────────────
    if setup_type in ("jepa", "patch_jepa"):
        overrides["ema_momentum"] = trial.suggest_float(
            "ema_momentum", *ss.get("ema_momentum", [0.90, 0.999])
        )
        overrides["predictor_hidden_dim"] = trial.suggest_categorical(
            "predictor_hidden_dim", ss.get("predictor_hidden_dim", [128, 256, 512])
        )

        # Variance regularisation (anti-collapse).
        # Always on for JEPA — collapse produces artificially low loss that fools Optuna.
        overrides["use_variance_reg"] = True
        overrides["var_reg_weight"] = trial.suggest_float(
            "var_reg_weight", *ss.get("var_reg_weight", [0.1, 10.0]), log=True
        )

        # Covariance regularisation (anti-redundancy). Always on for same reason.
        overrides["use_covariance_reg"] = True
        overrides["cov_reg_weight"] = trial.suggest_float(
            "cov_reg_weight", *ss.get("cov_reg_weight", [1e-3, 0.1]), log=True
        )

    return overrides


# ─── Optuna objective ─────────────────────────────────────────────────────────

def build_objective(setup_type: str, cfg: dict, hpo_epochs: int, dummy: bool):
    """Return an Optuna objective closure for the given setup.

    Args:
        setup_type: "pixel" or "jepa".
        cfg: Full config dict.
        hpo_epochs: Number of training epochs per trial.
        dummy: Whether to use in-memory random data.
    """
    ss = cfg.get("hpo", {}).get("search_space", {})

    def objective(trial: optuna.Trial) -> float:
        overrides = _suggest_hyperparams(trial, setup_type, ss)
        log_dir = str(
            Path("experiments") / f"hpo_{setup_type}_trial_{trial.number:04d}"
        )
        return train_trial(
            setup_type=setup_type,
            cfg=cfg,
            overrides=overrides,
            log_dir=log_dir,
            num_epochs=hpo_epochs,
            seed=SEED,
            trial=trial,
            dummy=dummy,
        )

    return objective


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimisation for Lenia world models (Optuna).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--setup",
        required=True,
        choices=["pixel", "jepa", "patch_jepa"],
        help="Training paradigm to optimise.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml. Default: config.yaml",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of Optuna trials. Overrides hpo.n_trials in config.",
    )
    parser.add_argument(
        "--hpo-epochs",
        type=int,
        default=None,
        help="Training epochs per trial. Overrides hpo.hpo_epochs in config.",
    )
    parser.add_argument(
        "--study-name",
        default=None,
        help="Optuna study name. Overrides hpo.study_name in config.",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help=(
            "Optuna storage URL for study persistence, e.g. "
            "sqlite:///experiments/hpo.db  "
            "Overrides hpo.storage in config."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Number of parallel trials. Values > 1 require a shared persistent "
            "storage backend and num_workers: 0 in config.yaml."
        ),
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use in-memory random data instead of loading from disk (for testing).",
    )
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent.parent / config_path
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    hpo_cfg = cfg.get("hpo", {})

    n_trials: int = args.n_trials or hpo_cfg.get("n_trials", 50)
    hpo_epochs: int = args.hpo_epochs or hpo_cfg.get("hpo_epochs", 20)
    study_name: str = (
        args.study_name or hpo_cfg.get("study_name") or f"lenia_{args.setup}_hpo"
    )
    storage: Optional[str] = args.storage or hpo_cfg.get("storage") or None
    direction: str = hpo_cfg.get("direction", "minimize")

    pruner_name: str = hpo_cfg.get("pruner", "median")
    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        if pruner_name == "median"
        else optuna.pruners.NopPruner()
    )

    # Silence Optuna's per-trial INFO logs so our epoch prints are readable
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        pruner=pruner,
        load_if_exists=True,
    )

    print(f"Study: {study_name}")
    print(f"Setup: {args.setup}  |  {n_trials} trials  |  {hpo_epochs} epochs/trial")
    if storage:
        print(f"Storage: {storage}")
    if args.dummy:
        print("Using dummy (randomly generated) data.")

    study.optimize(
        build_objective(args.setup, cfg, hpo_epochs, args.dummy),
        n_trials=n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )

    # ── Report results ────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\nBest trial: #{best.number}  val_loss={best.value:.6f}")
    print("  Hyperparameters:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # ── Save best params to YAML ──────────────────────────────────────────────
    out_dir = Path("experiments")
    out_dir.mkdir(exist_ok=True)
    best_params_path = out_dir / f"hpo_{args.setup}_best_params.yaml"
    with open(best_params_path, "w") as f:
        yaml.dump(
            {"best_val_loss": best.value, "params": best.params},
            f,
            default_flow_style=False,
            sort_keys=True,
        )
    print(f"\nBest params saved to {best_params_path}")


if __name__ == "__main__":
    main()
