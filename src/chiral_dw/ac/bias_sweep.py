"""Old-compatible conjugate-AC C3-bias sweeps."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chiral_dw.ac.nonideal import NonIdealACLLModel
from chiral_dw.ac.workflow import run_ac_cg_workflow
from chiral_dw.config import (
    ACResponseWorkflowParams,
    ConjugateACBiasSweepParams,
    ConjugateACBiasSweepSummary,
    DomainWallParams,
    FirstShellACParams,
    GatedInteractionParams,
    MomentumGridParams,
    PhysicalCoulombACPreset,
    ResponseParams,
    SourceInterpolationParams,
)
from chiral_dw.response import projector_errors

LL_GAP = 1.0


@dataclass(frozen=True)
class BandPathData:
    """Single-particle K/K' active-band dispersion along a high-symmetry path."""

    k_distance: np.ndarray
    ticks: np.ndarray
    labels: tuple[str, ...]
    up: np.ndarray
    down: np.ndarray
    max_split: float


@dataclass(frozen=True)
class ConjugateACBiasSweepResult:
    """In-memory result for a conjugate-AC bias sweep."""

    params: ConjugateACBiasSweepParams
    bias_values: np.ndarray
    theta: np.ndarray
    kappa: np.ndarray
    band_path: np.ndarray
    band_up: np.ndarray
    band_down: np.ndarray
    rows: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]
    summary: ConjugateACBiasSweepSummary


def physical_coulomb_interaction(interaction_shell: int) -> GatedInteractionParams:
    """Return the old-reference physical-Coulomb AC interaction preset."""
    return PhysicalCoulombACPreset().interaction_params(interaction_shell=interaction_shell)


def _high_symmetry_path(
    model: NonIdealACLLModel,
    n_segment: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    b1 = model.fields.G_shell[0]
    b2 = model.fields.G_shell[1]
    gamma = np.zeros(2)
    kappa = (2.0 * b1 + b2) / 3.0
    m_point = 0.5 * b1
    nodes = (gamma, kappa, m_point, gamma)
    labels = ("Gamma", "kappa", "m", "Gamma")
    points: list[np.ndarray] = []
    distances: list[float] = []
    ticks = [0.0]
    total = 0.0
    for start, end in zip(nodes[:-1], nodes[1:]):
        for s in np.linspace(0.0, 1.0, int(n_segment), endpoint=False):
            k = (1.0 - s) * start + s * end
            if points:
                total += float(np.linalg.norm(k - points[-1]))
            points.append(k)
            distances.append(total)
        total += float(np.linalg.norm(end - points[-1]))
        points.append(end)
        distances.append(total)
        ticks.append(total)
    return np.asarray(points), np.asarray(distances), np.asarray(ticks), labels


def active_band_path(
    ac_params: FirstShellACParams,
    *,
    n_segment: int,
    active_band: int = 0,
) -> BandPathData:
    """Return K and K' active-band dispersions along the old high-symmetry path."""
    model = NonIdealACLLModel(ac_params)
    k_path, k_distance, ticks, labels = _high_symmetry_path(model, n_segment)
    up = np.zeros(len(k_path), dtype=float)
    down = np.zeros_like(up)
    for idx, k in enumerate(k_path):
        up[idx] = model.solve(k, active_band=active_band).eigenvalues[active_band]
        down[idx] = model.solve(-k, active_band=active_band).eigenvalues[active_band]
    return BandPathData(
        k_distance=k_distance,
        ticks=ticks,
        labels=labels,
        up=up,
        down=down,
        max_split=float(np.max(np.abs(up - down))),
    )


def ac_params_for_bias(
    params: ConjugateACBiasSweepParams,
    value: float,
) -> FirstShellACParams:
    """Build first-shell AC parameters for one scalar or magnetic C3 bias."""
    if params.sweep_parameter == "u1_c3":
        return FirstShellACParams(
            b1=params.b1,
            u1=params.u1,
            b1_c3=params.b1_c3_fixed,
            u1_c3=float(value),
            n_ll=params.n_ll,
        )
    return FirstShellACParams(
        b1=params.b1,
        u1=params.u1,
        b1_c3=float(value),
        u1_c3=params.u1_c3_fixed,
        n_ll=params.n_ll,
    )


def _workflow_params(
    params: ConjugateACBiasSweepParams,
    ac_params: FirstShellACParams,
) -> ACResponseWorkflowParams:
    return ACResponseWorkflowParams(
        grid=params.grid,
        ac=ac_params,
        response=params.response,
        source=params.source,
        interaction=params.interaction,
        domain_wall=params.domain_wall,
    )


def _mean_nz(projectors: np.ndarray) -> float:
    arr = np.asarray(projectors, dtype=complex)
    return float(np.mean(np.real(arr[..., 0, 0] - arr[..., 1, 1])))


def _ivc_coherence(projectors: np.ndarray) -> float:
    arr = np.asarray(projectors, dtype=complex)
    return float(np.mean(np.abs(arr[..., 0, 1]) ** 2))


def _midpoint_index(theta: np.ndarray) -> int:
    return int(np.argmin(np.abs(np.asarray(theta, dtype=float) - 0.5 * np.pi)))


def kappa_midpoint_abs(theta: np.ndarray, kappa: np.ndarray) -> float:
    return float(abs(np.interp(0.5 * np.pi, theta, kappa)))


def kappa_odd_residual(theta: np.ndarray, kappa: np.ndarray) -> float:
    theta_arr = np.asarray(theta, dtype=float)
    kappa_arr = np.asarray(kappa, dtype=float)
    reflected = np.interp(np.pi - theta_arr, theta_arr, kappa_arr)
    return float(np.max(np.abs(kappa_arr + reflected)))


def _row_for_bias(
    params: ConjugateACBiasSweepParams,
    value: float,
    workflow_result: Any,
    band_path: BandPathData,
) -> dict[str, Any]:
    theta = workflow_result.theta
    mid = _midpoint_index(theta)
    P_mid = workflow_result.projectors[mid]
    row: dict[str, Any] = {
        params.sweep_parameter: float(value),
        "cG": float(workflow_result.response.cG),
        "max_k_kprime_dispersion_split": float(band_path.max_split),
        "vp_energy": float(workflow_result.energy_total[0]),
        "ivc_energy": float(workflow_result.energy_total[mid]),
        "vp_gap": float(workflow_result.gaps[0]),
        "ivc_gap": float(workflow_result.gaps[mid]),
        "vp_converged": True,
        "ivc_converged": True,
        "ivc_coherence": _ivc_coherence(P_mid),
        "ivc_Nz": _mean_nz(P_mid),
        "Kappa_midpoint_abs": kappa_midpoint_abs(theta, workflow_result.response.K),
        "Kappa_odd_residual": kappa_odd_residual(theta, workflow_result.response.K),
        "gap_min": float(np.min(workflow_result.gaps)),
        "projector_hermiticity_error": projector_errors(workflow_result.projectors)["hermiticity"],
        "projector_idempotency_error": projector_errors(workflow_result.projectors)["idempotency"],
    }
    if params.sweep_parameter == "u1_c3":
        row["u1_c3_over_ll_gap"] = float(value / LL_GAP)
    return row


def _metadata(params: ConjugateACBiasSweepParams) -> dict[str, Any]:
    interaction_matching = "physical_coulomb" if params.use_physical_coulomb else "dimensionless_input"
    base: dict[str, Any] = {
        "units": "omega_c",
        "sweep_parameter": params.sweep_parameter,
        "n_ll": int(params.n_ll),
        "active_band": int(params.active_band),
        "interaction_matching": interaction_matching,
        "V0_dimensionless": float(params.interaction.v0),
        "gate_distance_dimensionless": float(params.interaction.gate_distance),
        "n_phi_check": int(params.n_phi_check),
        "cg_phi_step": float(params.cg_phi_step),
    }
    if params.sweep_parameter == "u1_c3":
        base.update(
            {
                "LL_gap": LL_GAP,
                "projection": "lowest_landau_level",
                "u1_c3_min": float(params.bias_min),
                "u1_c3_max": float(params.bias_max),
                "n_u1_c3": int(params.n_bias),
            }
        )
    else:
        base.update(
            {
                "projection": "finite_ll_lowest_active_band",
                "b1_fixed": float(params.b1),
                "u1_c3_fixed": float(params.u1_c3_fixed),
                "b1_c3_min": float(params.bias_min),
                "b1_c3_max": float(params.bias_max),
                "n_b1_c3": int(params.n_bias),
                "b1_c3_strict_lll_matrix_element_zero": bool(params.n_ll == 1),
            }
        )
    return base


def run_conjugate_ac_bias_sweep(
    params: ConjugateACBiasSweepParams,
    *,
    write_outputs: bool = False,
    write_plots: bool | None = None,
) -> ConjugateACBiasSweepResult:
    """Run a scalar or magnetic C3 conjugate-AC bias sweep."""
    bias_values = np.linspace(params.bias_min, params.bias_max, params.n_bias)
    rows: list[dict[str, Any]] = []
    kappa_rows: list[np.ndarray] = []
    band_up_rows: list[np.ndarray] = []
    band_down_rows: list[np.ndarray] = []
    theta_ref: np.ndarray | None = None
    band_path_ref: np.ndarray | None = None

    for value in bias_values:
        ac_params = ac_params_for_bias(params, float(value))
        workflow_result = run_ac_cg_workflow(_workflow_params(params, ac_params))
        band_path = active_band_path(
            ac_params,
            n_segment=params.dispersion_points,
            active_band=params.active_band,
        )
        theta_ref = workflow_result.theta
        band_path_ref = band_path.k_distance
        rows.append(_row_for_bias(params, float(value), workflow_result, band_path))
        kappa_rows.append(np.asarray(workflow_result.response.K, dtype=float))
        band_up_rows.append(band_path.up)
        band_down_rows.append(band_path.down)

    if theta_ref is None or band_path_ref is None:
        raise RuntimeError("bias sweep produced no rows")
    kappa = np.asarray(kappa_rows, dtype=float)
    band_up = np.asarray(band_up_rows, dtype=float)
    band_down = np.asarray(band_down_rows, dtype=float)
    metadata = _metadata(params)
    summary = ConjugateACBiasSweepSummary(
        sweep_parameter=params.sweep_parameter,
        n_bias=params.n_bias,
        cG_min=float(np.min([row["cG"] for row in rows])),
        cG_max=float(np.max([row["cG"] for row in rows])),
        max_dispersion_split=float(np.max([row["max_k_kprime_dispersion_split"] for row in rows])),
        gap_min=float(np.min([row["gap_min"] for row in rows])),
        projection=str(metadata["projection"]),
    )
    result = ConjugateACBiasSweepResult(
        params=params,
        bias_values=bias_values,
        theta=theta_ref,
        kappa=kappa,
        band_path=band_path_ref,
        band_up=band_up,
        band_down=band_down,
        rows=tuple(rows),
        metadata=metadata,
        summary=summary,
    )
    if write_outputs:
        write_conjugate_ac_bias_sweep_outputs(
            result,
            write_plots=params.write_plots if write_plots is None else write_plots,
        )
    return result


def _prefix(params: ConjugateACBiasSweepParams) -> str:
    return "u1c3_cg_sweep" if params.sweep_parameter == "u1_c3" else "b1c3_cg_sweep"


def _csv_fieldnames(params: ConjugateACBiasSweepParams) -> list[str]:
    common = [
        "cG",
        "max_k_kprime_dispersion_split",
        "vp_energy",
        "ivc_energy",
        "vp_gap",
        "ivc_gap",
        "vp_converged",
        "ivc_converged",
        "ivc_coherence",
        "ivc_Nz",
        "Kappa_midpoint_abs",
        "Kappa_odd_residual",
    ]
    if params.sweep_parameter == "u1_c3":
        return ["u1_c3", "u1_c3_over_ll_gap", *common]
    return ["b1_c3", *common]


def _physical_coulomb_match(params: ConjugateACBiasSweepParams) -> dict[str, Any] | None:
    if not params.use_physical_coulomb:
        return None
    preset = PhysicalCoulombACPreset()
    return preset.model_dump(mode="json")


def write_conjugate_ac_bias_sweep_outputs(
    result: ConjugateACBiasSweepResult,
    *,
    write_plots: bool = True,
) -> None:
    """Write old-compatible CSV/JSON/NPZ and optional plot artifacts."""
    out_dir = Path(result.params.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = _prefix(result.params)
    _write_csv(out_dir / f"{prefix}.csv", result)
    _write_json(out_dir / f"{prefix}.json", result)
    _write_npz(out_dir / f"{prefix}.npz", result)
    if write_plots:
        _write_plots(out_dir, result)


def _write_csv(path: Path, result: ConjugateACBiasSweepResult) -> None:
    fieldnames = _csv_fieldnames(result.params)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result.rows)


def _write_json(path: Path, result: ConjugateACBiasSweepResult) -> None:
    payload: dict[str, Any] = {
        "metadata": result.metadata,
        "params": result.params.model_dump(mode="json"),
        "summary": result.summary.model_dump(mode="json"),
        "rows": list(result.rows),
    }
    match = _physical_coulomb_match(result.params)
    if match is not None:
        payload["physical_coulomb_match"] = match
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _write_npz(path: Path, result: ConjugateACBiasSweepResult) -> None:
    arrays: dict[str, Any] = {
        "cG": np.asarray([row["cG"] for row in result.rows], dtype=float),
        "max_k_kprime_dispersion_split": np.asarray(
            [row["max_k_kprime_dispersion_split"] for row in result.rows],
            dtype=float,
        ),
        "Kappa_theta_centers": result.theta,
        "Kappa": result.kappa,
        "band_path_distance": result.band_path,
    }
    if result.params.sweep_parameter == "u1_c3":
        arrays.update(
            {
                "u1_c3": result.bias_values,
                "u1_c3_over_ll_gap": result.bias_values / LL_GAP,
                "lll_band_up": result.band_up,
                "lll_band_down": result.band_down,
                "LL_gap": np.array(LL_GAP),
            }
        )
    else:
        arrays.update(
            {
                "b1_c3": result.bias_values,
                "lowest_band_up": result.band_up,
                "lowest_band_down": result.band_down,
            }
        )
    np.savez_compressed(path, **arrays)


def _write_plots(out_dir: Path, result: ConjugateACBiasSweepResult) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if result.params.sweep_parameter == "u1_c3":
        x = result.bias_values / LL_GAP
        label = "u1_c3 / omega_c"
        c_name = "cG_vs_u1c3.png"
        k_name = "Kappa_theta_vs_u1c3.png"
        split_name = "dispersion_split_vs_u1c3.png"
        band_min = "lll_band_structure_u1c3_min.png"
        band_max = "lll_band_structure_u1c3_max.png"
    else:
        x = result.bias_values
        label = "b1_c3"
        c_name = "cG_vs_b1c3.png"
        k_name = "Kappa_theta_vs_b1c3.png"
        split_name = "dispersion_split_vs_b1c3.png"
        band_min = "lowest_band_structure_b1c3_min.png"
        band_max = "lowest_band_structure_b1c3_max.png"

    _plot_scalar_series(out_dir / c_name, x, [row["cG"] for row in result.rows], label, "cG")
    _plot_scalar_series(
        out_dir / split_name,
        x,
        [row["max_k_kprime_dispersion_split"] for row in result.rows],
        label,
        "max |E_K - E_Kprime| / omega_c",
    )
    _plot_kappa(out_dir / k_name, x, result.theta, result.kappa, label)
    _plot_band_structure(
        out_dir / band_min,
        result.band_path,
        result.band_up[0],
        result.band_down[0],
        result.params.sweep_parameter,
        float(result.bias_values[0]),
    )
    _plot_band_structure(
        out_dir / band_max,
        result.band_path,
        result.band_up[-1],
        result.band_down[-1],
        result.params.sweep_parameter,
        float(result.bias_values[-1]),
    )
    plt.close("all")


def _plot_scalar_series(path: Path, x: np.ndarray, y: list[Any], xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    ax.plot(x, np.asarray(y, dtype=float), marker="o", linewidth=1.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_kappa(
    path: Path,
    x: np.ndarray,
    theta: np.ndarray,
    kappa: np.ndarray,
    xlabel: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    denom = max(float(x[-1] - x[0]), 1e-14)
    for idx, value in enumerate(x):
        ax.plot(theta / np.pi, kappa[idx], color=cmap(float((value - x[0]) / denom)), linewidth=1.2)
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.axvline(0.5, color="0.55", linewidth=0.9, linestyle="--")
    ax.set_xlabel("theta / pi")
    ax.set_ylabel("Kappa(theta)")
    ax.grid(True, alpha=0.25)
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(float(x[0]), float(x[-1]))), ax=ax, label=xlabel)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_band_structure(
    path: Path,
    k_distance: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    sweep_parameter: str,
    bias_value: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.plot(k_distance, up, linewidth=1.7, label="K active band")
    ax.plot(k_distance, down, linewidth=1.4, linestyle="--", label="Kprime active band")
    ax.set_xlabel("high-symmetry path")
    ax.set_ylabel("energy / omega_c")
    ax.set_title(f"{sweep_parameter}={bias_value:.6g}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-dir", default="results/conjugate_ac_bias_sweep")
    parser.add_argument("--b1", type=float, default=0.0)
    parser.add_argument("--u1", type=float, default=0.0)
    parser.add_argument("--n-ll", type=int, default=1)
    parser.add_argument("--active-band", type=int, default=0)
    parser.add_argument("--n-k", type=int, default=7)
    parser.add_argument("--interaction-shell", type=int, default=2)
    parser.add_argument("--V0", type=float, default=1.0)
    parser.add_argument("--gate-distance", type=float, default=2.0)
    parser.add_argument("--use-physical-coulomb", action="store_true")
    parser.add_argument("--epsilon", type=float, default=35.0)
    parser.add_argument("--gate-distance-nm", type=float, default=30.0)
    parser.add_argument("--theta-deg", type=float, default=3.9)
    parser.add_argument("--lattice-constant-A", type=float, default=3.52)
    parser.add_argument("--m-star-ratio", type=float, default=0.6)
    parser.add_argument("--theta-min", type=float, default=0.0)
    parser.add_argument("--theta-max", type=float, default=float(np.pi))
    parser.add_argument("--n-theta", type=int, default=41)
    parser.add_argument("--n-phi-check", type=int, default=3)
    parser.add_argument("--cg-phi-step", type=float, default=0.2)
    parser.add_argument("--dispersion-points", type=int, default=80)
    parser.add_argument("--source-scale", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--energy-tol", type=float, default=1e-9)
    parser.add_argument("--projector-tol", type=float, default=1e-7)
    parser.add_argument("--min-gap-tol", type=float, default=1e-10)
    parser.add_argument("--plot-all-band-structures", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _params_from_args(args: argparse.Namespace, sweep_parameter: str) -> ConjugateACBiasSweepParams:
    interaction = (
        physical_coulomb_interaction(args.interaction_shell)
        if args.use_physical_coulomb
        else GatedInteractionParams(
            v0=args.V0,
            gate_distance=args.gate_distance,
            interaction_shell=args.interaction_shell,
        )
    )
    if sweep_parameter == "u1_c3":
        bias_min = float(args.u1_c3_min)
        bias_max = float(args.u1_c3_max_frac_ll_gap * LL_GAP)
        n_bias = int(args.n_u1_c3)
        b1_c3_fixed = 0.0
        u1_c3_fixed = 0.0
    else:
        bias_min = float(args.b1_c3_min)
        bias_max = float(args.b1_c3_max)
        n_bias = int(args.n_b1_c3)
        b1_c3_fixed = 0.0
        u1_c3_fixed = float(args.u1_c3)
    return ConjugateACBiasSweepParams(
        sweep_parameter=sweep_parameter,  # type: ignore[arg-type]
        output_dir=args.output_dir,
        b1=args.b1,
        u1=args.u1,
        b1_c3_fixed=b1_c3_fixed,
        u1_c3_fixed=u1_c3_fixed,
        bias_min=bias_min,
        bias_max=bias_max,
        n_bias=n_bias,
        n_ll=args.n_ll,
        active_band=args.active_band,
        grid=MomentumGridParams(n_k=args.n_k),
        response=ResponseParams(
            n_theta=args.n_theta,
            theta_min=args.theta_min,
            theta_max=args.theta_max,
        ),
        source=SourceInterpolationParams(source_scale=args.source_scale),
        interaction=interaction,
        domain_wall=DomainWallParams(),
        dispersion_points=args.dispersion_points,
        n_phi_check=args.n_phi_check,
        cg_phi_step=args.cg_phi_step,
        max_iter=args.max_iter,
        energy_tol=args.energy_tol,
        projector_tol=args.projector_tol,
        min_gap_tol=args.min_gap_tol,
        use_physical_coulomb=args.use_physical_coulomb,
        write_plots=not args.no_plots,
    )


def run_lll_u1c3_sweep_console(argv: list[str] | None = None) -> None:
    parser = _base_parser("Sweep cG versus strict-LLL scalar C3 AC bias.")
    parser.set_defaults(output_dir="results/conjugate_ac_lll_u1c3_cg_sweep")
    parser.add_argument("--u1-c3-min", type=float, default=0.0)
    parser.add_argument("--u1-c3-max-frac-ll-gap", type=float, default=0.2)
    parser.add_argument("--n-u1-c3", type=int, default=11)
    args = parser.parse_args(argv)
    if args.n_ll != 1:
        raise ValueError("the LLL u1_c3 compatibility script requires --n-ll 1")
    result = run_conjugate_ac_bias_sweep(
        _params_from_args(args, "u1_c3"),
        write_outputs=True,
    )
    if not args.quiet:
        print(f"wrote {len(result.rows)} u1_c3 rows to {args.output_dir}")


def run_b1c3_sweep_console(argv: list[str] | None = None) -> None:
    parser = _base_parser("Sweep cG versus finite-LL magnetic C3 AC bias.")
    parser.set_defaults(n_ll=5, output_dir="results/conjugate_ac_b1c3_cg_sweep")
    parser.add_argument("--b1-c3-min", type=float, default=0.0)
    parser.add_argument("--b1-c3-max", type=float, default=0.2)
    parser.add_argument("--n-b1-c3", type=int, default=11)
    parser.add_argument("--u1-c3", type=float, default=0.0)
    args = parser.parse_args(argv)
    result = run_conjugate_ac_bias_sweep(
        _params_from_args(args, "b1_c3"),
        write_outputs=True,
    )
    if not args.quiet:
        print(f"wrote {len(result.rows)} b1_c3 rows to {args.output_dir}")
