"""Command-line interface: ``onsetwindow parameters.param [--out DIR]``."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .model import Trajectory, WindowResult, risk_by_age, simulate, window_of_opportunity
from .params import ParameterError, Parameters, load_parameters

DISCLAIMER = (
    "Research prototype. Parameters are illustrative unless estimated from data; "
    "outputs are not a clinical risk estimate."
)


def _fmt(v: Optional[float], nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def _write_trajectory(path: Path, traj: Trajectory) -> None:
    keys = list(traj.contributions)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["age", "S", "theta", "pi", "pibar", "latent_m"] + [f"contrib_{k}" for k in keys])
        for n, age in enumerate(traj.ages):
            w.writerow([f"{age:.4f}", f"{traj.S[n]:.6f}", f"{traj.theta:.6f}", f"{traj.pi[n]:.6f}",
                        f"{traj.pibar[n]:.6f}", f"{traj.latent[n]:.6f}"]
                       + [f"{traj.contributions[k][n]:.6f}" for k in keys])


def _write_risk(path: Path, rows, horizon: float, traj: Trajectory) -> None:
    keys = list(traj.contributions)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["age", "burden_S", "pi", "pibar", f"risk_next_{horizon:g}y"]
                   + [f"contrib_{k}" for k in keys])
        for r in rows:
            w.writerow([f"{r.age:g}", f"{r.burden:.4f}", f"{r.pi:.4f}", f"{r.pibar:.4f}",
                        "" if r.horizon_risk is None else f"{r.horizon_risk:.4f}"]
                       + [f"{traj.at(traj.contributions[k], r.age):.4f}" for k in keys])


def _write_window(path: Path, results: List[WindowResult]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "scale", "start_age", "T_new", "delta_T", "censored", "in_window"])
        for res in results:
            for pt in res.points:
                w.writerow([res.target, res.scale, f"{pt.start_age:g}",
                            "" if pt.T_new is None else f"{pt.T_new:.3f}",
                            f"{pt.delta_T:.3f}", int(pt.censored), int(pt.in_window)])


def run(params: Parameters, out_dir: Optional[Path], quiet: bool = False) -> dict:
    traj = simulate(params)
    rows = risk_by_age(traj, params.report_ages, params.risk_horizon)
    windows = window_of_opportunity(params, traj)

    summary = {
        "individual_id": params.individual_id,
        "sex": params.sex,
        "threshold_theta": traj.theta,
        "transition_age_T": traj.T,
        "constitutive_share_rho": traj.rho,
        "domain_weights": traj.weights,
        "domain_onset_ages": traj.onsets,
        "risk_by_age": [r.__dict__ for r in rows],
        "windows": [
            {"target": w.target, "scale": w.scale, "delta": w.delta,
             "intervals": [list(iv) for iv in w.intervals]}
            for w in windows
        ],
        "note": DISCLAIMER,
        "parameter_file": params.source,
        "version": __version__,
    }

    if not quiet:
        print(f"onsetwindow {__version__}  |  {DISCLAIMER}")
        print(f"Individual: {params.individual_id}   sex: {params.sex}   "
              f"environment E: {params.environment:g}")
        print(f"Threshold theta_i: {traj.theta:.3f}   transition age T_i: {_fmt(traj.T, 1)}   "
              f"rho_i: {_fmt(traj.rho)}")
        print()
        h = params.risk_horizon
        print(f"{'age':>5} {'S(t)':>8} {'pi(t)':>8} {'pibar(t)':>9} {f'next {h:g}y':>10}")
        for r in rows:
            print(f"{r.age:5g} {r.burden:8.3f} {r.pi:8.3f} {r.pibar:9.3f} "
                  f"{'' if r.horizon_risk is None else f'{r.horizon_risk:.3f}':>10}")
        if params.window is not None:
            print()
            if traj.T is None:
                print("Window of opportunity: not defined (no transition before age_max).")
            for w in windows:
                ivs = ", ".join(f"{a:g}-{b:g}" for a, b in w.intervals) or "empty"
                print(f"W_i({w.target}) [exposure x{w.scale:g}, delta={w.delta:g} y]: start ages {ivs}")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_trajectory(out_dir / "trajectory.csv", traj)
        _write_risk(out_dir / "risk_by_age.csv", rows, params.risk_horizon, traj)
        if windows:
            _write_window(out_dir / "window.csv", windows)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        if not quiet:
            print(f"\nResults written to {out_dir}/")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="onsetwindow",
        description="Individual risk trajectory under the accumulating-burden model.",
    )
    ap.add_argument("param_file", help="path to a .param file")
    ap.add_argument("--out", type=Path, default=None, help="directory for CSV/JSON outputs")
    ap.add_argument("--quiet", action="store_true", help="suppress console output")
    ap.add_argument("--version", action="version", version=f"onsetwindow {__version__}")
    args = ap.parse_args(argv)
    try:
        params = load_parameters(args.param_file)
    except ParameterError as exc:
        print(f"Parameter error: {exc}", file=sys.stderr)
        return 2
    run(params, args.out, args.quiet)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
