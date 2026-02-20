from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import download_openbb_prices, load_csv_prices
from .model import HMMConfig
from .pipeline import run_regime_pipeline
from .walk_forward import run_walk_forward_regime_analysis



def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run HMM market regime detection pipeline.")

    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Path to input CSV.")
    source.add_argument("--symbol", help="Ticker symbol for OpenBB download, e.g. SPY.")

    p.add_argument("--start-date", help="OpenBB start date (YYYY-MM-DD). Required with --symbol.")
    p.add_argument("--end-date", help="OpenBB end date (YYYY-MM-DD). Required with --symbol.")
    p.add_argument("--provider", default="yfinance", help="OpenBB provider (default: yfinance).")
    p.add_argument(
        "--save-downloaded-csv",
        default=None,
        help="Optional path to persist downloaded OpenBB prices as CSV.",
    )

    p.add_argument("--output-dir", required=True, help="Directory for output artifacts.")
    p.add_argument("--datetime-col", default="datetime", help="Datetime column in CSV.")
    p.add_argument("--close-col", default="close", help="Close column in CSV/OpenBB response.")

    p.add_argument("--n-states", type=int, default=3, help="Number of latent HMM states.")
    p.add_argument("--n-iter", type=int, default=300, help="Maximum EM iterations.")
    p.add_argument("--tol", type=float, default=1e-4, help="EM convergence tolerance.")
    p.add_argument("--random-state", type=int, default=42, help="Random seed.")

    p.add_argument(
        "--wf-train-bars",
        type=int,
        default=None,
        help="Walk-forward train window bars. Set together with --wf-test-bars to enable.",
    )
    p.add_argument(
        "--wf-test-bars",
        type=int,
        default=None,
        help="Walk-forward test window bars. Set together with --wf-train-bars to enable.",
    )
    p.add_argument(
        "--wf-step-bars",
        type=int,
        default=None,
        help="Walk-forward step bars (defaults to test bars).",
    )
    p.add_argument(
        "--wf-min-test-bars",
        type=int,
        default=20,
        help="Minimum test bars for the final fold (default: 20).",
    )

    return p



def _load_prices_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> pd.DataFrame:
    if args.input is not None:
        return load_csv_prices(
            args.input,
            datetime_col=args.datetime_col,
            close_col=args.close_col,
        )

    if args.start_date is None or args.end_date is None:
        parser.error("--start-date and --end-date are required when using --symbol.")

    prices = download_openbb_prices(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        provider=args.provider,
        close_col=args.close_col,
    )

    if args.save_downloaded_csv:
        out = Path(args.save_downloaded_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(out, index=True)
        print(f"Saved downloaded prices: {out}")

    return prices



def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    wf_enabled = (args.wf_train_bars is not None) or (args.wf_test_bars is not None)
    if wf_enabled and (args.wf_train_bars is None or args.wf_test_bars is None):
        parser.error("Set both --wf-train-bars and --wf-test-bars to enable walk-forward mode.")

    prices = _load_prices_from_args(args, parser)

    cfg = HMMConfig(
        n_states=args.n_states,
        n_iter=args.n_iter,
        tol=args.tol,
        random_state=args.random_state,
    )

    result = run_regime_pipeline(prices, close_col=args.close_col, hmm_config=cfg)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    states_df = pd.concat(
        [
            result.close.reindex(result.states.index).rename("close"),
            result.log_returns.rename("log_ret_1"),
            result.states,
            result.regime_labels,
            result.state_probabilities,
        ],
        axis=1,
    )

    states_path = out_dir / "states.csv"
    transition_path = out_dir / "transition_matrix.csv"
    durations_path = out_dir / "duration_stats.csv"
    summary_path = out_dir / "return_summary.csv"

    states_df.to_csv(states_path, index=True)
    result.transition_matrix.to_csv(transition_path, index=True)
    result.duration_stats.to_csv(durations_path, index=True)
    result.return_summary.to_csv(summary_path, index=True)

    print("Saved:")
    print(f"- {states_path}")
    print(f"- {transition_path}")
    print(f"- {durations_path}")
    print(f"- {summary_path}")
    print("State labels:")
    for state, label in sorted(result.state_label_map.items()):
        print(f"- state {state}: {label}")

    if wf_enabled:
        wf = run_walk_forward_regime_analysis(
            prices,
            close_col=args.close_col,
            hmm_config=cfg,
            train_bars=args.wf_train_bars,
            test_bars=args.wf_test_bars,
            step_bars=args.wf_step_bars,
            min_test_bars=args.wf_min_test_bars,
        )

        wf_folds_path = out_dir / "walk_forward_folds.csv"
        wf_trans_drift_path = out_dir / "drift_transition.csv"
        wf_flip_path = out_dir / "drift_label_flip.csv"
        wf_ll_path = out_dir / "drift_loglik.csv"

        wf.folds.to_csv(wf_folds_path, index=True)
        wf.drift_transition.to_csv(wf_trans_drift_path, index=True)
        wf.drift_label_flip.to_csv(wf_flip_path, index=True)
        wf.drift_loglik.to_csv(wf_ll_path, index=True)

        print("Walk-forward diagnostics saved:")
        print(f"- {wf_folds_path}")
        print(f"- {wf_trans_drift_path}")
        print(f"- {wf_flip_path}")
        print(f"- {wf_ll_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
