#!/usr/bin/env python3
"""Plot VP and IVC HF mesh spectra for the ideal epsilon=2 conjugate AC point."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


for thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_variable, "1")

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chiral_dw.ac.projected import build_ac_projected_bundle  # noqa: E402
from chiral_dw.config import (  # noqa: E402
    ACProjectedHFParams,
    ContinuumGridParams,
    ContinuumHFParams,
    ContinuumInteractionParams,
    FirstShellACParams,
)
from chiral_dw.continuum import build_symmetric_hf_references  # noqa: E402


FIG_DIR = ROOT / "Plots" / "figures"
OUTPUT_STEM = "ac_eps2_b1u1_zero_nk15_hf_bands"
COULOMB_MEV_NM = 1439.96454784255
HBAR2_OVER_2ME_MEV_NM2 = 38.0998212

FIGURE = {
    "size": (12.2, 6.2),
    "dpi": 280,
    "subplots_adjust": {
        "left": 0.085,
        "right": 0.98,
        "bottom": 0.19,
        "top": 0.76,
        "wspace": 0.18,
    },
}

FONTS = {
    "base": 12,
    "title": 16,
    "axis_label": 18,
    "tick_label": 14,
    "legend": 13,
    "annotation": 12,
}

COLORS = {
    "red": "#FD4C55",
    "teal": "#378d94",
    "grey": "0.25",
    "axis": "0.18",
    "grid": "0.70",
    "teal_fill": "#77b5b6",
}


class ACHFPlotParams(BaseModel):
    """Reproducible parameters for the epsilon=2 ideal AC HF spectrum."""

    model_config = ConfigDict(frozen=True)

    n_k: int = Field(default=15, ge=2)
    n_ll: int = Field(default=6, ge=1)
    b1: float = 0.0
    u1: float = 0.0
    epsilon: float = Field(default=2.0, gt=0.0)
    interaction_multiplier: float = Field(default=0.1, ge=0.0)
    gate_distance_nm: float = Field(default=30.0, gt=0.0)
    smear_length_nm: float = Field(default=0.347, ge=0.0)
    theta_deg: float = Field(default=3.5, gt=0.0)
    a0_angstrom: float = Field(default=3.47, gt=0.0)
    m_eff: float = Field(default=0.62, gt=0.0)
    local_field_cutoff: int = Field(default=1, ge=0)
    workers: int = Field(default=4, ge=1)

    @property
    def moire_length_nm(self) -> float:
        theta = np.deg2rad(self.theta_deg)
        return float(self.a0_angstrom / (20.0 * np.sin(0.5 * theta)))

    @property
    def moire_cell_area_nm2(self) -> float:
        return float(np.sqrt(3.0) * self.moire_length_nm**2 / 2.0)

    @property
    def landau_level_spacing_mev(self) -> float:
        magnetic_length_sq_nm2 = self.moire_cell_area_nm2 / (2.0 * np.pi)
        return float(
            2.0 * HBAR2_OVER_2ME_MEV_NM2 / (self.m_eff * magnetic_length_sq_nm2)
        )

    @property
    def characteristic_coulomb_mev(self) -> float:
        return float(
            self.interaction_multiplier
            * COULOMB_MEV_NM
            / (self.epsilon * self.moire_length_nm)
        )


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["PT Serif Caption", "PT Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "mathtext.rm": "serif",
            "font.size": FONTS["base"],
            "axes.titlesize": FONTS["title"],
            "axes.labelsize": FONTS["axis_label"],
            "xtick.labelsize": FONTS["tick_label"],
            "ytick.labelsize": FONTS["tick_label"],
            "legend.fontsize": FONTS["legend"],
            "axes.edgecolor": COLORS["axis"],
            "axes.linewidth": 1.15,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.transparent": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _box_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["axis"])
        spine.set_linewidth(1.15)


def _mesh_path(
    n_k: int,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str], list[tuple[int, int]]]:
    """Return a discrete Gamma-M-kappa-Gamma tour on the solved square mesh."""

    anchors = [(0, 0), (n_k // 2, 0), (n_k // 3, n_k // 3), (0, 0)]
    labels = [r"$\Gamma$", r"$M$", r"$\kappa$", r"$\Gamma$"]
    coords: list[tuple[int, int]] = []
    ticks = [0]
    for start, end in zip(anchors[:-1], anchors[1:]):
        di = end[0] - start[0]
        dj = end[1] - start[1]
        steps = max(abs(di), abs(dj))
        for step in range(steps + 1):
            if coords and step == 0:
                continue
            fraction = 0.0 if steps == 0 else step / steps
            coords.append(
                (
                    int(round(start[0] + fraction * di)) % n_k,
                    int(round(start[1] + fraction * dj)) % n_k,
                )
            )
        ticks.append(len(coords) - 1)

    b1 = np.asarray([1.0, 0.0])
    b2 = np.asarray([0.5, np.sqrt(3.0) / 2.0])
    distance = np.zeros(len(coords), dtype=float)
    for idx in range(1, len(coords)):
        previous = coords[idx - 1][0] / n_k * b1 + coords[idx - 1][1] / n_k * b2
        current = coords[idx][0] / n_k * b1 + coords[idx][1] / n_k * b2
        distance[idx] = distance[idx - 1] + float(np.linalg.norm(current - previous))
    flat = np.asarray([i * n_k + j for i, j in coords], dtype=int)
    return flat, distance, ticks, labels, coords


def _spectrum_diagnostics(eigenvalues_mev: np.ndarray) -> dict[str, float]:
    """Return fixed-one-per-k gap diagnostics for a two-band HF spectrum."""

    values = np.asarray(eigenvalues_mev, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("expected a two-band spectrum with shape (n_k^2, 2)")
    direct = values[:, 1] - values[:, 0]
    occupied_max = float(np.max(values[:, 0]))
    empty_min = float(np.min(values[:, 1]))
    return {
        "occupied_band_max_mev": occupied_max,
        "empty_band_min_mev": empty_min,
        "direct_gap_min_mev": float(np.min(direct)),
        "indirect_gap_mev": float(empty_min - occupied_max),
        "occupied_bandwidth_mev": float(np.ptp(values[:, 0])),
        "empty_bandwidth_mev": float(np.ptp(values[:, 1])),
    }


def _build_params(plot_params: ACHFPlotParams) -> ACProjectedHFParams:
    return ACProjectedHFParams(
        grid=ContinuumGridParams(n_k=plot_params.n_k),
        ac=FirstShellACParams(
            b1=plot_params.b1,
            u1=plot_params.u1,
            n_ll=plot_params.n_ll,
        ),
        interaction=ContinuumInteractionParams(
            coulomb_kind="dual_gate",
            v0=plot_params.interaction_multiplier,
            q_mesh="full",
            q_shell=1,
            local_field_cutoff=plot_params.local_field_cutoff,
            epsilon=plot_params.epsilon,
            gate_distance_nm=plot_params.gate_distance_nm,
            smear_length_nm=plot_params.smear_length_nm,
            vertex_workers=plot_params.workers,
            exchange_workers=plot_params.workers,
        ),
        hf=ContinuumHFParams(
            n_occ_per_k=1,
            max_iter=800,
            min_iter=2,
            mixing_method="oda",
            mixing=0.45,
            tolerance=1e-8,
            energy_tolerance=1e-10,
            final_residual_tolerance=1e-7,
            random_seed=1,
        ),
        active_band=0,
        band_diagnostics_n_k=plot_params.n_k,
        moire_length_nm=plot_params.moire_length_nm,
        energy_unit_mev=plot_params.landau_level_spacing_mev,
    )


def _compute_spectra(
    params: ACHFPlotParams,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    bundle = build_ac_projected_bundle(_build_params(params))
    references = build_symmetric_hf_references(bundle, _build_params(params).hf)
    spectra: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for key, result in (("vp", references.vp_plus), ("ivc", references.ivc)):
        values = np.linalg.eigvalsh(result.H_hf) * params.landau_level_spacing_mev
        spectra[key] = np.asarray(values, dtype=float)
        metadata[key] = {
            **_spectrum_diagnostics(values),
            "converged": bool(result.converged),
            "n_iter": int(result.n_iter),
            "aufbau_residual_norm": float(result.diagnostics.aufbau_residual_norm),
            "energy_per_cell_mev": float(
                result.energy
                * params.landau_level_spacing_mev
                / bundle.backend.n_blocks
            ),
        }
    return spectra, metadata


def _write_cache(
    params: ACHFPlotParams,
    spectra: dict[str, np.ndarray],
    metadata: dict[str, dict[str, Any]],
) -> tuple[Path, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = FIG_DIR / f"{OUTPUT_STEM}_spectra.npz"
    json_path = FIG_DIR / f"{OUTPUT_STEM}_metadata.json"
    np.savez_compressed(
        npz_path,
        vp_eigenvalues_mev=spectra["vp"],
        ivc_eigenvalues_mev=spectra["ivc"],
    )
    payload = {
        "params": params.model_dump(mode="json"),
        "derived_scales": {
            "moire_length_nm": params.moire_length_nm,
            "moire_cell_area_nm2": params.moire_cell_area_nm2,
            "landau_level_spacing_mev": params.landau_level_spacing_mev,
            "characteristic_coulomb_mev": params.characteristic_coulomb_mev,
            "characteristic_coulomb_to_ll_ratio": (
                params.characteristic_coulomb_mev / params.landau_level_spacing_mev
            ),
        },
        "spectra": metadata,
        "energy_zero": "each panel is shifted by its occupied-band maximum",
        "momentum_path": "discrete Gamma-M-kappa-Gamma tour on the solved nk x nk mesh",
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    return npz_path, json_path


def _load_cache() -> dict[str, np.ndarray]:
    path = FIG_DIR / f"{OUTPUT_STEM}_spectra.npz"
    data = np.load(path)
    return {
        "vp": np.asarray(data["vp_eigenvalues_mev"], dtype=float),
        "ivc": np.asarray(data["ivc_eigenvalues_mev"], dtype=float),
    }


def _write_plot_csv(
    params: ACHFPlotParams,
    spectra: dict[str, np.ndarray],
) -> Path:
    flat, distance, _ticks, _labels, coords = _mesh_path(params.n_k)
    path = FIG_DIR / f"{OUTPUT_STEM}.csv"
    fieldnames = [
        "state",
        "path_index",
        "k_i",
        "k_j",
        "path_distance",
        "band",
        "occupation",
        "energy_relative_vbm_mev",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for state, values in spectra.items():
            shift = float(np.max(values[:, 0]))
            shifted = values - shift
            for path_index, (block, coord, x_value) in enumerate(
                zip(flat, coords, distance, strict=True)
            ):
                for band in range(values.shape[1]):
                    writer.writerow(
                        {
                            "state": state,
                            "path_index": path_index,
                            "k_i": coord[0],
                            "k_j": coord[1],
                            "path_distance": float(x_value),
                            "band": band,
                            "occupation": 1 if band == 0 else 0,
                            "energy_relative_vbm_mev": float(shifted[block, band]),
                        }
                    )
    return path


def _plot(
    params: ACHFPlotParams,
    spectra: dict[str, np.ndarray],
) -> tuple[Path, Path]:
    _apply_style()
    flat, distance, ticks, labels, _coords = _mesh_path(params.n_k)
    diagnostics = {
        state: _spectrum_diagnostics(values) for state, values in spectra.items()
    }
    shifted = {
        state: values - float(np.max(values[:, 0]))
        for state, values in spectra.items()
    }
    all_path_values = np.concatenate([values[flat].reshape(-1) for values in shifted.values()])
    span = float(np.ptp(all_path_values))
    padding = max(0.08 * span, 0.8)
    y_limits = (
        float(np.min(all_path_values) - padding),
        float(np.max(all_path_values) + padding),
    )

    fig, axes = plt.subplots(1, 2, figsize=FIGURE["size"], sharex=True, sharey=True)
    for panel, (state, title) in enumerate(
        (("vp", r"VP (${\rm VP}_\pm$ spectra coincide)"), ("ivc", "IVC"))
    ):
        ax = axes[panel]
        values = shifted[state]
        diag = diagnostics[state]
        ax.plot(
            distance,
            values[flat, 0],
            color=COLORS["red"],
            linewidth=2.2,
            label="occupied HF band",
        )
        ax.plot(
            distance,
            values[flat, 1],
            color=COLORS["teal"],
            linewidth=2.2,
            label="empty HF band",
        )
        ax.axhspan(
            0.0,
            diag["indirect_gap_mev"],
            color=COLORS["teal_fill"],
            alpha=0.15,
            zorder=-2,
        )
        ax.axhline(0.0, color=COLORS["grey"], linestyle=":", linewidth=1.1)
        for tick in ticks:
            ax.axvline(distance[tick], color="0.82", linewidth=0.85, zorder=-1)
        ax.set_xticks([distance[tick] for tick in ticks], labels)
        ax.set_ylim(*y_limits)
        ax.set_title(
            title
            + "\n"
            + rf"$\Delta_{{\rm ind}}={diag['indirect_gap_mev']:.3f}$ meV, "
            + rf"$\Delta_{{\rm dir}}^{{\rm min}}={diag['direct_gap_min_mev']:.3f}$ meV",
            fontsize=FONTS["title"],
            pad=9,
        )
        ax.set_xlabel("momentum path", fontsize=FONTS["axis_label"])
        ax.grid(axis="y")
        ax.set_box_aspect(0.92)
        _box_axes(ax)

    axes[0].set_ylabel(
        r"$E_{\rm HF}-E_{\rm occ}^{\rm max}$ (meV)",
        fontsize=FONTS["axis_label"],
    )
    fig.suptitle(
        r"Conjugate AC HF bands: "
        + rf"$\epsilon={params.epsilon:g}$, "
        + rf"$B_1=U_1=0$, $n_k={params.n_k}$",
        fontsize=18,
        y=0.97,
    )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(**FIGURE["subplots_adjust"])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{OUTPUT_STEM}.png"
    pdf = FIG_DIR / f"{OUTPUT_STEM}.pdf"
    fig.savefig(png, dpi=FIGURE["dpi"], bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return png, pdf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    params = ACHFPlotParams(workers=args.workers)
    cache_path = FIG_DIR / f"{OUTPUT_STEM}_spectra.npz"
    metadata_path = FIG_DIR / f"{OUTPUT_STEM}_metadata.json"
    if args.force_recompute or not cache_path.exists() or not metadata_path.exists():
        spectra, metadata = _compute_spectra(params)
        _write_cache(params, spectra, metadata)
    else:
        spectra = _load_cache()

    csv_path = _write_plot_csv(params, spectra)
    png, pdf = _plot(params, spectra)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {cache_path}")
    print(f"Wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
