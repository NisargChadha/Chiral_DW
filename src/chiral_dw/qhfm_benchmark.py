"""Same-Chern QHFM real-space charge normalization benchmark."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chiral_dw.ac.nonideal import NonIdealACLLModel
from chiral_dw.artifacts import RunArtifact, RunManifest
from chiral_dw.config import QHFMChargeBenchmarkParams, QHFMChargeSummary


def _unit(v: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(norm < eps):
        raise ValueError("cannot normalize a vector with near-zero norm")
    return v / norm


@dataclass(frozen=True)
class QHFMProjectorSolution:
    """Occupied spinors and projectors in the two-band same-Chern basis."""

    k_points: np.ndarray
    k_fractional: np.ndarray
    r_fractional: np.ndarray
    coefficients: np.ndarray
    spinors: np.ndarray
    flavor_projectors: np.ndarray
    pseudospin_field: np.ndarray
    eigenvalues: np.ndarray
    gaps: np.ndarray

    @property
    def spin_expectation(self) -> np.ndarray:
        p = self.flavor_projectors
        nx = 2.0 * np.real(p[..., 0, 1])
        ny = -2.0 * np.imag(p[..., 0, 1])
        nz = np.real(p[..., 0, 0] - p[..., 1, 1])
        return np.stack([nx, ny, nz], axis=-1)

    def k_dependence_measure(self) -> float:
        n = self.spin_expectation
        mean = np.mean(n, axis=(0, 1), keepdims=True)
        return float(np.max(np.linalg.norm(n - mean, axis=-1)))


@dataclass(frozen=True)
class QHFMChargeBenchmarkResult:
    """In-memory and artifact summary for the same-Chern QHFM benchmark."""

    params: QHFMChargeBenchmarkParams
    solution: QHFMProjectorSolution
    curvature_components: dict[str, np.ndarray]
    rho_top: np.ndarray
    q_sk: np.ndarray
    summary: QHFMChargeSummary
    manifest: RunManifest | None = None


class SameChernQHFMTrial:
    """Trial Hamiltonian h(k,r)=epsilon(k)I+d(k,r).sigma for same-Chern AC copies."""

    def __init__(self, params: QHFMChargeBenchmarkParams | None = None) -> None:
        self.params = params or QHFMChargeBenchmarkParams()
        self.model = NonIdealACLLModel(self.params.ac)
        self.b1 = self.model.fields.G_shell[0]
        self.b2 = self.model.fields.G_shell[1]

    def fractional_k_grid(self, extended: bool = False) -> np.ndarray:
        n = int(self.params.grid.n_k)
        stop = n + 1 if extended else n
        pts = np.arange(stop, dtype=float) / n
        uu, vv = np.meshgrid(pts, pts, indexing="ij")
        return np.stack([uu, vv], axis=-1)

    def k_grid(self, extended: bool = False) -> np.ndarray:
        frac = self.fractional_k_grid(extended=extended)
        return frac[..., 0, None] * self.b1 + frac[..., 1, None] * self.b2

    def fractional_r_grid(self, extended: bool = False) -> np.ndarray:
        n = int(self.params.real_space.n_r)
        stop = n + 1 if extended else n
        pts = np.arange(stop, dtype=float) / n
        uu, vv = np.meshgrid(pts, pts, indexing="ij")
        return np.stack([uu, vv], axis=-1)

    def form_factors(self, k_fractional: np.ndarray) -> np.ndarray:
        u = k_fractional[..., 0]
        v = k_fractional[..., 1]
        vals = [
            np.ones_like(u),
            np.cos(2.0 * np.pi * u),
            np.sin(2.0 * np.pi * u),
            np.cos(2.0 * np.pi * v),
            np.sin(2.0 * np.pi * v),
            np.cos(2.0 * np.pi * (u + v)),
            np.sin(2.0 * np.pi * (u + v)),
        ]
        return np.stack(vals[: self.params.n_form_factors], axis=-1)

    def active_band_energies(self, k_points: np.ndarray) -> np.ndarray:
        flat = np.asarray(k_points, dtype=float).reshape(-1, 2)
        energies = [
            self.model.solve(k, active_band=self.params.active_band).eigenvalues[
                self.params.active_band
            ]
            for k in flat
        ]
        return np.asarray(energies, dtype=float).reshape(k_points.shape[:-1])

    def field_from_coefficients(
        self,
        coefficients: np.ndarray,
        k_fractional: np.ndarray,
    ) -> np.ndarray:
        coeffs = np.asarray(coefficients, dtype=float)
        expected = (3, self.params.n_form_factors)
        if coeffs.shape != expected:
            raise ValueError(f"coefficients must have shape {expected}, got {coeffs.shape}")
        forms = self.form_factors(k_fractional)
        return np.einsum("...n,an->...a", forms, coeffs, optimize=True)

    def solve(
        self,
        coefficients: np.ndarray,
        texture_field: np.ndarray | None = None,
        *,
        extended_k: bool = False,
        extended_r: bool = False,
    ) -> QHFMProjectorSolution:
        k_frac = self.fractional_k_grid(extended=extended_k)
        k_points = self.k_grid(extended=extended_k)
        r_frac = self.fractional_r_grid(extended=extended_r)
        d_k = self.field_from_coefficients(coefficients, k_frac)
        d = d_k[:, :, None, None, :]
        if texture_field is None:
            d = np.broadcast_to(d, (*d_k.shape[:2], *r_frac.shape[:2], 3)).copy()
        else:
            texture = np.asarray(texture_field, dtype=float)
            if texture.shape != (*r_frac.shape[:-1], 3):
                raise ValueError(
                    "texture_field must have shape "
                    f"{(*r_frac.shape[:-1], 3)}, got {texture.shape}"
                )
            d = d + texture[None, None, :, :, :]

        eps = self.active_band_energies(k_points)[:, :, None, None]
        h = np.zeros((*d.shape[:-1], 2, 2), dtype=complex)
        h[..., 0, 0] = eps + d[..., 2]
        h[..., 1, 1] = eps - d[..., 2]
        h[..., 0, 1] = d[..., 0] - 1j * d[..., 1]
        h[..., 1, 0] = d[..., 0] + 1j * d[..., 1]

        vals, vecs = np.linalg.eigh(h)
        spinors = vecs[..., :, 0]
        projectors = spinors[..., :, None] * spinors[..., None, :].conj()
        return QHFMProjectorSolution(
            k_points=k_points,
            k_fractional=k_frac,
            r_fractional=r_frac,
            coefficients=np.asarray(coefficients, dtype=float).copy(),
            spinors=spinors,
            flavor_projectors=projectors,
            pseudospin_field=d,
            eigenvalues=vals,
            gaps=vals[..., 1] - vals[..., 0],
        )

    def full_projector(self, k: np.ndarray, flavor_projector: np.ndarray) -> np.ndarray:
        sol = self.model.solve(k, active_band=self.params.active_band)
        coeffs = sol.eigenvectors[:, self.params.active_band]
        orbital_projector = coeffs[:, None] * coeffs[None, :].conj()
        return np.kron(flavor_projector, orbital_projector)


def periodic_skyrmion_lattice_field(
    r_fractional: np.ndarray,
    mass: float = 0.5,
) -> np.ndarray:
    """Return a smooth periodic skyrmion-lattice unit vector on the real-space torus."""

    u = np.asarray(r_fractional, dtype=float)[..., 0]
    v = np.asarray(r_fractional, dtype=float)[..., 1]
    raw = np.stack(
        [
            np.sin(2.0 * np.pi * u),
            np.sin(2.0 * np.pi * v),
            float(mass) + np.cos(2.0 * np.pi * u) + np.cos(2.0 * np.pi * v),
        ],
        axis=-1,
    )
    return _unit(raw)


class GeneralProjector4DCurvature:
    """Link-variable 4D curvature evaluator for same-Chern QHFM projectors."""

    def __init__(self, trial: SameChernQHFMTrial, solution: QHFMProjectorSolution) -> None:
        self.trial = trial
        self.solution = solution
        if solution.k_points.shape[:2] != (trial.params.grid.n_k + 1, trial.params.grid.n_k + 1):
            raise ValueError("solution must be built with extended_k=True")
        if solution.r_fractional.shape[:2] != (
            trial.params.real_space.n_r + 1,
            trial.params.real_space.n_r + 1,
        ):
            raise ValueError("solution must be built with extended_r=True")
        self._coeff_cache: dict[tuple[int, int], np.ndarray] = {}

    def _orbital_coeffs(self, i: int, j: int) -> np.ndarray:
        key = (i, j)
        cached = self._coeff_cache.get(key)
        if cached is not None:
            return cached
        k = self.solution.k_points[i, j]
        sol = self.trial.model.solve(k, active_band=self.trial.params.active_band)
        coeffs = sol.eigenvectors[:, self.trial.params.active_band]
        self._coeff_cache[key] = coeffs
        return coeffs

    def _overlap(
        self,
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> complex:
        ia, ja, xa, ya = a
        ib, jb, xb, yb = b
        ka = self.solution.k_points[ia, ja]
        kb = self.solution.k_points[ib, jb]
        ca = self._orbital_coeffs(ia, ja)
        cb = self._orbital_coeffs(ib, jb)
        orbital = self.trial.model.state_overlap(ka, ca, kb, cb)
        spin = np.vdot(
            self.solution.spinors[ia, ja, xa, ya],
            self.solution.spinors[ib, jb, xb, yb],
        )
        return complex(orbital * spin)

    def _link(
        self,
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> complex:
        overlap = self._overlap(a, b)
        magnitude = abs(overlap)
        if magnitude < 1e-14:
            return 1.0 + 0.0j
        return overlap / magnitude

    @staticmethod
    def _shift(idx: tuple[int, int, int, int], dim: int) -> tuple[int, int, int, int]:
        vals = list(idx)
        vals[dim] += 1
        return tuple(vals)  # type: ignore[return-value]

    def curvature_phase(self, dim_a: int, dim_b: int) -> np.ndarray:
        """Return plaquette Berry phases for a pair of 4D lattice directions."""

        n_k = int(self.trial.params.grid.n_k)
        n_r = int(self.trial.params.real_space.n_r)
        phases = np.zeros((n_k, n_k, n_r, n_r), dtype=float)
        for i in range(n_k):
            for j in range(n_k):
                for x in range(n_r):
                    for y in range(n_r):
                        idx = (i, j, x, y)
                        ia = self._shift(idx, dim_a)
                        ib = self._shift(idx, dim_b)
                        iab = self._shift(ia, dim_b)
                        product = (
                            self._link(idx, ia)
                            * self._link(ia, iab)
                            * self._link(iab, ib)
                            * self._link(ib, idx)
                        )
                        phases[i, j, x, y] = np.angle(product)
        return phases

    def curvature_components(self) -> dict[str, np.ndarray]:
        return {
            "Fkxky": self.curvature_phase(0, 1),
            "Fxky": self.curvature_phase(2, 1),
            "Fykx": self.curvature_phase(3, 0),
            "Fxkx": self.curvature_phase(2, 0),
            "Fyky": self.curvature_phase(3, 1),
            "Fxy": self.curvature_phase(2, 3),
        }

    def second_chern_density(
        self,
        components: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        c = components or self.curvature_components()
        return (
            c["Fxy"] * c["Fkxky"]
            - c["Fxkx"] * c["Fyky"]
            + c["Fxky"] * c["Fykx"]
        ) / (4.0 * np.pi**2)

    def charge_per_realspace_plaquette(
        self,
        components: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Return the topological charge per real-space plaquette."""

        c2 = self.second_chern_density(components)
        return -np.sum(c2, axis=(0, 1))

    def spin_skyrmion_charge_per_plaquette(
        self,
        components: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        c = components or self.curvature_components()
        return np.mean(c["Fxy"], axis=(0, 1)) / (2.0 * np.pi)


def run_qhfm_charge_benchmark(
    params: QHFMChargeBenchmarkParams | None = None,
    *,
    write_outputs: bool = False,
    write_plots: bool = False,
) -> QHFMChargeBenchmarkResult:
    """Run the same-Chern QHFM charge-density normalization check."""

    benchmark_params = params or QHFMChargeBenchmarkParams()
    trial = SameChernQHFMTrial(benchmark_params)
    coeffs = np.zeros((3, benchmark_params.n_form_factors), dtype=float)
    skyrmion_n = periodic_skyrmion_lattice_field(
        trial.fractional_r_grid(extended=True),
        mass=benchmark_params.skyrmion.mass,
    )
    solution = trial.solve(
        coeffs,
        texture_field=-skyrmion_n,
        extended_k=True,
        extended_r=True,
    )
    evaluator = GeneralProjector4DCurvature(trial, solution)
    components = evaluator.curvature_components()
    rho_top = evaluator.charge_per_realspace_plaquette(components)
    q_sk = evaluator.spin_skyrmion_charge_per_plaquette(components)
    summary = summarize_qhfm_charge(components, rho_top, q_sk)
    result = QHFMChargeBenchmarkResult(
        params=benchmark_params,
        solution=solution,
        curvature_components=components,
        rho_top=rho_top,
        q_sk=q_sk,
        summary=summary,
    )
    if write_outputs:
        manifest = write_qhfm_charge_outputs(result, write_plots=write_plots)
        result = QHFMChargeBenchmarkResult(**{**result.__dict__, "manifest": manifest})
    return result


def summarize_qhfm_charge(
    components: dict[str, np.ndarray],
    rho_top: np.ndarray,
    q_sk: np.ndarray,
    charge_tolerance: float = 5e-3,
) -> QHFMChargeSummary:
    mixed_max = max(
        float(np.max(np.abs(components[name])))
        for name in ("Fxkx", "Fxky", "Fykx", "Fyky")
    )
    orbital_chern = float(np.sum(components["Fkxky"][:, :, 0, 0]) / (2.0 * np.pi))
    target = -np.asarray(q_sk, dtype=float)
    rho = np.asarray(rho_top, dtype=float)
    charge_error = float(np.max(np.abs(rho - target)))
    x = target.reshape(-1)
    y = rho.reshape(-1)
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        slope = float("nan")
        intercept = float("nan")
        correlation = float("nan")
    else:
        slope, intercept = np.polyfit(x, y, deg=1)
        correlation = float(np.corrcoef(x, y)[0, 1])
    return QHFMChargeSummary(
        orbital_chern=orbital_chern,
        mixed_curvature_max=mixed_max,
        charge_error_max=charge_error,
        integrated_charge=float(np.sum(rho)),
        integrated_skyrmion_charge=float(np.sum(q_sk)),
        slope=float(slope),
        intercept=float(intercept),
        correlation=correlation,
        valid_charge_normalization=bool(
            abs(orbital_chern - 1.0) < charge_tolerance
            and charge_error < charge_tolerance
            and mixed_max < charge_tolerance
        ),
    )


def write_qhfm_charge_outputs(
    result: QHFMChargeBenchmarkResult,
    *,
    write_plots: bool = False,
) -> RunManifest:
    out_dir = Path(result.params.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "qhfm_charge_density.csv"
    summary_path = out_dir / "qhfm_charge_summary.json"
    curvature_path = out_dir / "qhfm_curvature_components.npz"
    plot_path = out_dir / "qhfm_charge_maps.png"
    manifest_path = out_dir / "artifact_manifest.json"

    _write_charge_csv(csv_path, result)
    payload = {
        "params": result.params.model_dump(mode="json"),
        "summary": result.summary.model_dump(mode="json"),
        "normalization": (
            "Same-Chern QHFM validation limit. Expected relation is "
            "rho_top = -q_sk with this sign convention."
        ),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    if result.params.write_curvature_npz:
        np.savez_compressed(
            curvature_path,
            rho_top=result.rho_top,
            q_sk=result.q_sk,
            minus_q_sk=-result.q_sk,
            **result.curvature_components,
        )
    if write_plots:
        _write_charge_plot(plot_path, result)

    artifacts = [
        _artifact(csv_path, "charge_density", "table", "QHFM rho_top and skyrmion density map"),
        _artifact(summary_path, "summary", "json", "QHFM charge benchmark scalar summary"),
        _artifact(
            curvature_path,
            "curvature_components",
            "array",
            "Raw 4D curvature components and charge arrays",
            required=bool(result.params.write_curvature_npz),
        ),
        _artifact(plot_path, "charge_maps", "plot", "Optional rho/target/error maps", required=False),
    ]
    manifest = RunManifest.from_artifacts(
        run_id="qhfm_charge_benchmark",
        result_dir=str(out_dir),
        artifacts=artifacts,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    return manifest


def _write_charge_csv(path: Path, result: QHFMChargeBenchmarkResult) -> None:
    centers = (np.arange(result.params.real_space.n_r) + 0.5) / result.params.real_space.n_r
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["u_center", "v_center", "rho_top", "q_sk", "minus_q_sk", "difference"],
        )
        writer.writeheader()
        for i, u in enumerate(centers):
            for j, v in enumerate(centers):
                target = -result.q_sk[i, j]
                writer.writerow(
                    {
                        "u_center": float(u),
                        "v_center": float(v),
                        "rho_top": float(result.rho_top[i, j]),
                        "q_sk": float(result.q_sk[i, j]),
                        "minus_q_sk": float(target),
                        "difference": float(result.rho_top[i, j] - target),
                    }
                )


def _write_charge_plot(path: Path, result: QHFMChargeBenchmarkResult) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = -result.q_sk
    error = result.rho_top - target
    vmax = max(float(np.max(np.abs(result.rho_top))), float(np.max(np.abs(target))), 1e-15)
    err_vmax = max(float(np.max(np.abs(error))), 1e-15)
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), constrained_layout=True)
    panels = [
        (result.rho_top, r"$\rho_{\rm top}$", "RdBu_r", vmax),
        (target, r"$-q_{\rm sk}$", "RdBu_r", vmax),
        (error, r"$\rho_{\rm top}+q_{\rm sk}$", "PuOr", err_vmax),
    ]
    for ax, (data, title, cmap, limit) in zip(axes, panels, strict=True):
        im = ax.imshow(
            data.T,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap=cmap,
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Same-Chern QHFM charge benchmark: "
        f"slope={result.summary.slope:.6f}, "
        f"corr={result.summary.correlation:.6f}, "
        f"max error={result.summary.charge_error_max:.2e}",
        fontsize=11,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _artifact(
    path: Path,
    name: str,
    kind: str,
    description: str,
    required: bool = True,
) -> RunArtifact:
    return RunArtifact(
        name=name,
        path=str(path),
        kind=kind,  # type: ignore[arg-type]
        description=description,
        required=required,
        exists=path.exists(),
        size_bytes=path.stat().st_size if path.exists() else None,
    )


__all__ = [
    "GeneralProjector4DCurvature",
    "QHFMChargeBenchmarkResult",
    "QHFMProjectorSolution",
    "SameChernQHFMTrial",
    "periodic_skyrmion_lattice_field",
    "run_qhfm_charge_benchmark",
    "summarize_qhfm_charge",
    "write_qhfm_charge_outputs",
]
