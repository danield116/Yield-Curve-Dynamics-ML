"""Main ablation runner for full model comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_run import evaluate_run
from evaluation.metrics import save_scorecard
from training.train_stage_a import load_config
from training.train_stage_b import ABLATION_PRESETS
from visualization.plot_curves import plot_horizon_curve, plot_scorecard_bar


def _run_script(script_rel: str, extra_args: list[str] | None = None) -> None:
    script = PROJECT_ROOT / script_rel
    cmd = [sys.executable, str(script)] + (extra_args or [])
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_ablation_suite(
    config: dict,
    *,
    train_stage_a: bool = False,
    train_stage_b: bool = False,
    skip_existing: bool = True,
) -> list[dict]:
    """Train (optional) and evaluate all Stage B ablations + baselines."""
    training_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation", {})
    model_cfg = config.get("model", {})

    config_path = eval_cfg.get("config_path", str(PROJECT_ROOT / "config" / "default.yaml"))
    config_arg = ["--config", config_path]

    stage_a_ckpt = Path(training_cfg.get("checkpoint_dir", "reports/checkpoints/stage_a"))
    variant = model_cfg.get("stage_a_variant", "student_t_cvae")
    stage_a_file = stage_a_ckpt / f"stage_a_{variant}_best.pt"

    if train_stage_a or (not stage_a_file.exists() and not skip_existing):
        _run_script("training/train_stage_a.py", config_arg)
    elif not stage_a_file.exists():
        raise FileNotFoundError(f"Stage A checkpoint missing: {stage_a_file}")

    stage_b_dir = Path(training_cfg.get("checkpoint_dir_stage_b", "reports/checkpoints/stage_b"))
    for ablation in ABLATION_PRESETS:
        ckpt = stage_b_dir / f"stage_b_{ablation}_best.pt"
        if train_stage_b or (not ckpt.exists() and not skip_existing):
            _run_script("training/train_stage_b.py", config_arg + ["--ablation", ablation])
        elif not ckpt.exists():
            print(f"Warning: missing Stage B checkpoint for {ablation}; skipping training.")

    rows = evaluate_run(config)
    output_dir = Path(eval_cfg.get("report_dir", "reports/comparison"))
    tenors = config.get("data", {}).get("tenors")
    save_scorecard(rows, output_dir, tenors=tenors)

    horizons = eval_cfg.get("horizons", [1, 5, 21, 63])
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    h1_rows = [r for r in rows if r.get("horizon") == horizons[0]]
    plot_scorecard_bar(h1_rows, metric="curve_rmse", output_path=figures_dir / f"curve_rmse_by_model_h{horizons[0]}.png")

    for ablation in ABLATION_PRESETS:
        model_name = f"stage_b_{ablation}"
        if any(r["model"] == model_name for r in rows):
            plot_horizon_curve(
                rows,
                model=model_name,
                metric="curve_rmse",
                output_path=figures_dir / f"{model_name}_rmse_vs_horizon.png",
            )

    summary_path = output_dir / "summary.json"
    summary = [
        {
            "model": r["model"],
            "horizon": r["horizon"],
            "curve_rmse": r["curve_rmse"],
            "latent_rmse": r.get("latent_rmse"),
            "segments": r.get("segments"),
            "arbitrage": r.get("arbitrage"),
        }
        for r in rows
    ]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote comparison summary to: {summary_path.resolve()}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full model comparison experiment.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--train-stage-a", action="store_true")
    parser.add_argument("--train-stage-b", action="store_true")
    parser.add_argument("--force-train", action="store_true", help="Retrain even if checkpoints exist.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    run_ablation_suite(
        config,
        train_stage_a=args.train_stage_a,
        train_stage_b=args.train_stage_b,
        skip_existing=not args.force_train,
    )


if __name__ == "__main__":
    main()
