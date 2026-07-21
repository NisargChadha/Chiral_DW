from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "Plots" / "plot_taige_orbital_magnetization_convergence.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("orbital_magnetization_plot", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_orbital_magnetization_convergence_plot_renders(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    remote = []
    for point, offset in (("vbm", 0.7), ("midgap", 0.0), ("cbm", -0.7)):
        for cutoff in range(7):
            remote.append(
                {
                    "chemical_potential_point": point,
                    "n_remote_bands_per_valley": cutoff,
                    "orbital_magnetization_mu_b_per_cell": offset + 0.02 * cutoff,
                }
            )
    _write_csv(results / "remote_convergence.csv", remote)
    _write_csv(
        results / "hf_active_space_convergence.csv",
        [
            {
                "chemical_potential_point": "midgap",
                "n_hf_bands_per_valley": n,
                "active_remote_mixing_lambda": 0.4 * (n - 2),
            }
            for n in (2, 3, 4)
        ],
    )
    _write_csv(
        results / "matched_cutoff_comparison.csv",
        [
            {
                "n_total_bands_per_valley": n,
                "frozen_magnetization_mu_b_per_cell": 0.25 + 0.01 * (n - 2),
                "hf_magnetization_mu_b_per_cell": 0.25 + 0.012 * (n - 2),
            }
            for n in (2, 3, 4)
        ],
    )
    benchmarks = []
    for stage, scale in (
        ("density_vertices", 100.0),
        ("exchange_backend", 20.0),
        ("vp_hf_solve", 2.0),
    ):
        for n in (2, 3, 4):
            benchmarks.append(
                {
                    "stage": stage,
                    "n_active_bands_per_valley": n,
                    "elapsed_seconds_median": scale * (n - 1),
                }
            )
    _write_csv(results / "benchmarks.csv", benchmarks)

    module = _load_module()
    png, pdf = module.make_figure(results, tmp_path / "figure")
    assert png.exists() and png.stat().st_size > 10_000
    assert pdf.exists() and pdf.stat().st_size > 1_000
