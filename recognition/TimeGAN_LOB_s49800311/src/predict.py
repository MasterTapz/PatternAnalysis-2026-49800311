"""Load trained generators, sample synthetic order books and audit them against the test hour.

For every run given with --run this script:
  1. rebuilds the exact data pipeline stored in the checkpoint and loads best.pt
  2. generates synthetic windows and decodes them into order books (prices in
     dollars, sizes in shares), printing one example book
  3. scores them against the held-out test hour (15:00 to 15:50):
       invalid books     crossed or locked, broken ladder, negative size
       KL divergence     spreads (ticks) and mid moves (half-ticks), target <= 0.1
       SSIM              depth heatmaps (20 levels x time, log volume), target > 0.6
       volatility        autocorrelation of squared returns, lags 1 to 20
       price path        share of steps where the mid does not move, and how far
                         the mid wanders from its starting value within a window
       diversity         spread of samples and distance to the nearest training window
     with the same statistics for real training windows as a reference
  4. saves an autopsy figure of five rule-selected synthetic windows next to
     their closest real window, and an autocorrelation plot

With several runs, it also writes a comparison table and a combined
autocorrelation plot. With --export-assets DIR it then copies the figures
the README embeds out of the runs folder into DIR and draws the ladder-rate
figure; this needs no model, so it also works without --run.
The analysis itself lives in helpers/audit.py and the figures in
helpers/plotting.py. NumPy and scikit-image are used there for SSIM and
plots, which the spec allows on the prediction side.

Examples (from the project folder, the one holding src/):
  python src/predict.py --data-dir ../../../../data/LOBSTER --run runs/structured_timegan_s0
  python src/predict.py --data-dir ../../../../data/LOBSTER --run runs/structured_timegan_s0 runs/structured_rgan_s0
  python src/predict.py --export-assets assets
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from helpers.audit import RunAudit, audit_run, comparison_table, print_example_book, print_summary
from helpers.checkpoints import load_run
from helpers.plotting import export_readme_assets, plot_autopsy, plot_volatility_acf, plot_volatility_comparison


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None, help="Folder holding the LOBSTER CSVs (needed with --run)")
    p.add_argument("--run", nargs="+", default=[], help="One or more run folders produced by train.py")
    p.add_argument("--n-samples", type=int, default=1024)
    p.add_argument("--ssim-pool", type=int, default=256, help="Synthetic and training windows used for nearest-match SSIM")
    p.add_argument("--out-dir", default="runs/comparison", help="Where the multi-run comparison goes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--export-assets", default=None, metavar="DIR",
                   help="Copy the README figures out of --runs-dir into DIR and draw the ladder-rate figure")
    p.add_argument("--runs-dir", default="runs", help="Folder holding every run, read by --export-assets")
    args = p.parse_args(argv)
    if not args.run and args.export_assets is None:
        p.error("give --run, --export-assets or both")
    if args.run and args.data_dir is None:
        p.error("--run needs --data-dir to rebuild the data pipeline")
    return args


def audit_and_save(run_dir: Path, args: argparse.Namespace) -> RunAudit:
    """Audit one run and write its predict/ folder: autopsy.png, volatility_acf.png and summary.json."""
    model, splits, state = load_run(run_dir, args.data_dir, args.device)
    print(f"\n=== {run_dir.name}: {type(model).__name__}, {splits.representation} encoding, "
          f"checkpoint step {state['step']} ===")
    audit = audit_run(model, splits, run_dir.name, state["step"], n_samples=args.n_samples,
                      ssim_pool=args.ssim_pool, seed=args.seed, device=args.device)
    print_example_book(audit.fake_books[0])

    out_dir = run_dir / "predict"
    out_dir.mkdir(exist_ok=True)
    plot_autopsy(audit.cases, audit.fake_books, audit.real_books, audit.nearest, audit.data_range,
                 f"{run_dir.name}: five rule-selected synthetic windows vs their closest real test window",
                 out_dir / "autopsy.png")
    plot_volatility_acf(audit.acf, run_dir.name, out_dir / "volatility_acf.png")
    (out_dir / "summary.json").write_text(json.dumps(audit.summary, indent=1))
    print_summary(audit.summary)
    return audit


def write_comparison(audits: list[RunAudit], out_dir: Path) -> None:
    """comparison.md and the combined volatility plot over several runs; the real references come from the first run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "comparison.md"
    # Written as UTF-8 so the superscript in the header survives on Windows.
    table_path.write_text(comparison_table([a.summary for a in audits]), encoding="utf-8")
    reference = audits[0].acf
    plot_volatility_comparison(reference["real_train"], reference["real_test"],
                               {a.summary["run"]: a.acf["synthetic"] for a in audits},
                               out_dir / "volatility_acf_comparison.png")
    print(f"\nComparison written to {table_path}\n")
    print(table_path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    audits = [audit_and_save(Path(r), args) for r in args.run]
    if len(audits) > 1:
        write_comparison(audits, Path(args.out_dir))
    if args.export_assets is not None:
        # Runs after the audit, so a combined call exports the figures it has just redrawn.
        for path in export_readme_assets(Path(args.runs_dir), Path(args.export_assets)):
            print(f"{path}  {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
