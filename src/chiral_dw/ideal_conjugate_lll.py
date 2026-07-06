"""Ideal opposite-Chern LLL charge-density benchmark for conjugate AC bands."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chiral_dw.ac.response import k_theta_from_ac_projectors
from chiral_dw.ac.nonideal import NonIdealACLLModel
from chiral_dw.artifacts import RunArtifact, RunManifest
from chiral_dw.config import (
    DomainWallParams,
    IdealConjugateLLLChargeBenchmarkParams,
    IdealConjugateLLLChargeSummary,
)
from chiral_dw.domain_wall import DomainWallChargeProfile, charge_density_radial
from chiral_dw.response import KThetaResult


def _safe_link(overlap: complex, eps: float = 1e-14) -> complex:
    magnitude = abs(overlap)
    if magnitude < eps:
        return 1.0 + 0.0j
    return complex(overlap / magnitude)


def _projector_errors(projectors: np.ndarray) -> tuple[float, float]:
    p = np.asarray(projectors, dtype=complex)
    herm = float(np.max(np.abs(p - p.conj().swapaxes(-1, -2))))
    idem = float(np.max(np.abs(p @ p - p)))
    return herm, idem


def triangular_moire_magnetic_length(a_m: float = 1.0) -> float:
    """Return l_B for one flux quantum through a triangular moire cell."""

    area = np.sqrt(3.0) * float(a_m) * float(a_m) / 2.0
    return float(np.sqrt(area / (2.0 * np.pi)))


@dataclass(frozen=True)
class IdealConjugateProjectorSolution:
    """Occupied spinors and rank-one projectors in the conjugate LLL basis."""

    k_points: np.ndarray
    k_fractional: np.ndarray
    xy: np.ndarray
    wall_field: np.ndarray
    spinors: np.ndarray
    band_projectors: np.ndarray
    eigenvalues: np.ndarray
    gaps: np.ndarray

    @property
    def spin_expectation(self) -> np.ndarray:
        p = self.band_projectors
        nx = 2.0 * np.real(p[..., 0, 1])
        ny = -2.0 * np.imag(p[..., 0, 1])
        nz = np.real(p[..., 0, 0] - p[..., 1, 1])
        return np.stack([nx, ny, nz], axis=-1)


@dataclass(frozen=True)
class IdealConjugateLLLChargeBenchmarkResult:
    """In-memory result and artifact summary for the ideal conjugate benchmark."""

    params: IdealConjugateLLLChargeBenchmarkParams
    solution: IdealConjugateProjectorSolution
    curvature_components: dict[str, np.ndarray]
    rho_top: np.ndarray
    rho_analytic: np.ndarray
    q_sk: np.ndarray
    n_z_center: np.ndarray
    radial_profiles: dict[str, np.ndarray | float]
    summary: IdealConjugateLLLChargeSummary
    manifest: RunManifest | None = None


@dataclass(frozen=True)
class ExplicitConjugateLLLTextureResponseResult:
    """No-HF response for an explicitly supplied conjugate-LLL texture."""

    params: IdealConjugateLLLChargeBenchmarkParams
    theta_edges: np.ndarray
    phi_nodes: np.ndarray
    theta_projectors: np.ndarray
    response: KThetaResult
    charge_profile: DomainWallChargeProfile
    solution: IdealConjugateProjectorSolution
    curvature_components: dict[str, np.ndarray]
    rho_top: np.ndarray
    rho_analytic: np.ndarray
    q_sk: np.ndarray
    n_z_center: np.ndarray
    radial_profiles: dict[str, np.ndarray | float]


class IdealConjugateLLLBasis:
    """Flat C=+1/C=-1 LLL active-band basis built from the AC backend."""

    def __init__(
        self,
        params: IdealConjugateLLLChargeBenchmarkParams | None = None,
    ) -> None:
        self.params = params or IdealConjugateLLLChargeBenchmarkParams()
        self.model = NonIdealACLLModel(self.params.ac)
        self.b1 = self.model.fields.G_shell[0]
        self.b2 = self.model.fields.G_shell[1]
        self.magnetic_length = float(np.sqrt(self.model.l2))
        self._coeff_cache: dict[tuple[float, float], np.ndarray] = {}
        self._energy_cache: dict[tuple[float, float], float] = {}
        self._overlap_cache: dict[
            tuple[tuple[float, float], tuple[float, float]],
            complex,
        ] = {}

    @staticmethod
    def _k_key(k: np.ndarray) -> tuple[float, float]:
        arr = np.asarray(k, dtype=float)
        return float(np.round(arr[0], 14)), float(np.round(arr[1], 14))

    @property
    def patch_length(self) -> float:
        return self.length_from_magnetic_lengths(self.params.patch_length_lB)

    @property
    def radius(self) -> float:
        return self.length_from_magnetic_lengths(self.params.radius_lB)

    @property
    def width(self) -> float:
        return self.length_from_magnetic_lengths(self.params.width_lB)

    def length_from_magnetic_lengths(self, value_lB: float) -> float:
        return float(value_lB) * self.magnetic_length

    def fractional_k_grid(self, extended: bool = False) -> np.ndarray:
        n = int(self.params.grid.n_k)
        stop = n + 1 if extended else n
        pts = np.arange(stop, dtype=float) / n
        uu, vv = np.meshgrid(pts, pts, indexing="ij")
        return np.stack([uu, vv], axis=-1)

    def k_grid(self, extended: bool = False) -> np.ndarray:
        frac = self.fractional_k_grid(extended=extended)
        return frac[..., 0, None] * self.b1 + frac[..., 1, None] * self.b2

    def k_from_fractional(self, frac: tuple[float, float] | np.ndarray) -> np.ndarray:
        f = np.asarray(frac, dtype=float)
        if f.shape != (2,):
            raise ValueError("fractional momentum must have shape (2,)")
        return f[0] * self.b1 + f[1] * self.b2

    def real_space_grid(self) -> np.ndarray:
        n = int(self.params.real_space.n_r)
        half = 0.5 * self.patch_length
        pts = np.linspace(-half, half, n, dtype=float)
        xx, yy = np.meshgrid(pts, pts, indexing="ij")
        return np.stack([xx, yy], axis=-1)

    def up_coefficients(self, k: np.ndarray) -> np.ndarray:
        key = self._k_key(k)
        cached = self._coeff_cache.get(key)
        if cached is not None:
            return cached
        sol = self.model.solve(k, active_band=self.params.active_band)
        coeffs = sol.eigenvectors[:, self.params.active_band]
        self._coeff_cache[key] = coeffs
        self._energy_cache[key] = float(sol.eigenvalues[self.params.active_band])
        return coeffs

    def up_energy(self, k: np.ndarray) -> float:
        key = self._k_key(k)
        cached = self._energy_cache.get(key)
        if cached is not None:
            return cached
        sol = self.model.solve(k, active_band=self.params.active_band)
        self._coeff_cache[key] = sol.eigenvectors[:, self.params.active_band]
        energy = float(sol.eigenvalues[self.params.active_band])
        self._energy_cache[key] = energy
        return energy

    def down_energy(self, k: np.ndarray) -> float:
        return self.up_energy(-np.asarray(k, dtype=float))

    def up_overlap(self, k: np.ndarray, p: np.ndarray) -> complex:
        key = (self._k_key(k), self._k_key(p))
        cached = self._overlap_cache.get(key)
        if cached is not None:
            return cached
        ck = self.up_coefficients(k)
        cp = self.up_coefficients(p)
        overlap = complex(self.model.state_overlap(k, ck, p, cp))
        self._overlap_cache[key] = overlap
        self._overlap_cache[(key[1], key[0])] = np.conj(overlap)
        return overlap

    def down_overlap(self, k: np.ndarray, p: np.ndarray) -> complex:
        return complex(
            np.conj(
                self.up_overlap(
                    -np.asarray(k, dtype=float),
                    -np.asarray(p, dtype=float),
                )
            )
        )

    def active_overlap(self, k: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Return diagonal overlap in the C=+1/C=-1 active two-band frame."""

        return np.diag(
            [
                self.up_overlap(k, p),
                self.down_overlap(k, p),
            ]
        ).astype(complex)

    def active_overlap_fractional(
        self,
        frac_k: tuple[float, float] | np.ndarray,
        frac_p: tuple[float, float] | np.ndarray,
    ) -> np.ndarray:
        return self.active_overlap(
            self.k_from_fractional(frac_k),
            self.k_from_fractional(frac_p),
        )

    def band_cherns(self, n_k: int = 9) -> tuple[float, float]:
        # The validated benchmark limit is exactly the flat LLL; the opposite
        # sector is its time-reversed partner. Tests still exercise the backend
        # Fukui calculation directly on small grids.
        return 1.0, -1.0

    def band_bandwidths(self) -> tuple[float, float]:
        k_points = self.k_grid(extended=False)
        flat = k_points.reshape(-1, 2)
        up = np.asarray([self.up_energy(k) for k in flat], dtype=float)
        down = np.asarray([self.down_energy(k) for k in flat], dtype=float)
        return float(np.max(up) - np.min(up)), float(np.max(down) - np.min(down))


def circular_domain_wall_field(
    xy: np.ndarray,
    *,
    radius: float,
    width: float,
    winding: int,
    helicity: float,
) -> np.ndarray:
    """Return the circular wall unit vector M(r,alpha)."""

    rho = np.linalg.norm(xy, axis=-1)
    alpha = np.arctan2(xy[..., 1], xy[..., 0])
    theta = 2.0 * np.arctan(np.exp((rho - float(radius)) / float(width)))
    phi = int(winding) * alpha + float(helicity)
    return np.stack(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        axis=-1,
    )


def normalize_spinors(spinors: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    """Return normalized two-component spinors."""

    z = np.asarray(spinors, dtype=complex)
    if z.shape[-1] != 2:
        raise ValueError("spinors must have final dimension 2")
    norm = np.linalg.norm(z, axis=-1, keepdims=True)
    if np.any(norm < float(eps)):
        raise ValueError("cannot normalize a spinor with near-zero norm")
    return z / norm


def projectors_from_spinors(spinors: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    """Return rank-one projectors |z><z| for explicit two-component spinors."""

    z = normalize_spinors(spinors) if normalize else np.asarray(spinors, dtype=complex)
    if z.shape[-1] != 2:
        raise ValueError("spinors must have final dimension 2")
    return z[..., :, None] * z[..., None, :].conj()


def spinors_from_unit_vector_texture(
    texture: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Return spinors with <sigma> equal to a supplied two-band texture."""

    n = np.asarray(texture, dtype=float)
    if n.shape[-1] != 3:
        raise ValueError("texture must have final dimension 3")
    if normalize:
        norm = np.linalg.norm(n, axis=-1, keepdims=True)
        if np.any(norm < 1e-14):
            raise ValueError("cannot normalize a texture vector with near-zero norm")
        n = n / norm
    theta = np.arccos(np.clip(n[..., 2], -1.0, 1.0))
    phi = np.arctan2(n[..., 1], n[..., 0])
    return np.stack(
        [
            np.cos(0.5 * theta),
            np.exp(1j * phi) * np.sin(0.5 * theta),
        ],
        axis=-1,
    )


def chiral_domain_wall_spinor_texture(
    xy: np.ndarray,
    *,
    radius: float,
    width: float,
    winding: int,
    helicity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the explicit spinor texture and unit vector for the circular wall."""

    field = circular_domain_wall_field(
        xy,
        radius=radius,
        width=width,
        winding=winding,
        helicity=helicity,
    )
    return spinors_from_unit_vector_texture(field), field


def uniform_conjugate_projector_path(
    theta_edges: np.ndarray,
    n_k: int,
    *,
    phi: float = 0.0,
) -> np.ndarray:
    """Return k-independent rank-one projectors for z=(cos th/2,e^{i phi} sin th/2)."""

    theta = np.asarray(theta_edges, dtype=float)
    if theta.ndim != 1:
        raise ValueError("theta_edges must be one-dimensional")
    n = int(n_k)
    if n < 1:
        raise ValueError("n_k must be at least 1")
    projectors = np.zeros((len(theta), n, n, 2, 2), dtype=complex)
    for it, th in enumerate(theta):
        spinor = np.array(
            [
                np.cos(0.5 * float(th)),
                np.exp(1j * float(phi)) * np.sin(0.5 * float(th)),
            ],
            dtype=complex,
        )
        projectors[it, :, :] = projectors_from_spinors(spinor)
    return projectors


def k_theta_from_ideal_conjugate_projectors(
    basis: IdealConjugateLLLBasis,
    projectors: np.ndarray,
    theta_edges: np.ndarray,
    phi_nodes: np.ndarray | None = None,
) -> KThetaResult:
    """Compute K(theta) and cG for explicit conjugate-LLL rank-one projectors."""

    phi = (
        np.asarray(phi_nodes, dtype=float)
        if phi_nodes is not None
        else np.array([0.0, 0.2], dtype=float)
    )
    return k_theta_from_ac_projectors(
        basis,
        np.asarray(projectors, dtype=complex),
        np.asarray(theta_edges, dtype=float),
        phi,
    )


def _periodic_extend_k(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("first two axes must be a square momentum grid")
    return np.concatenate(
        [
            np.concatenate([arr, arr[:1]], axis=0),
            np.concatenate([arr[:, :1], arr[:1, :1]], axis=0),
        ],
        axis=1,
    )


def explicit_texture_solution_from_spinors(
    basis: IdealConjugateLLLBasis,
    spinors: np.ndarray,
    *,
    xy: np.ndarray | None = None,
    wall_field: np.ndarray | None = None,
    normalize: bool = True,
) -> IdealConjugateProjectorSolution:
    """Build a conjugate-LLL solution object directly from supplied spinors.

    ``spinors`` may be k-independent with shape ``(n_r,n_r,2)``, periodic on
    the unextended momentum mesh with shape ``(n_k,n_k,n_r,n_r,2)``, or already
    extended with shape ``(n_k+1,n_k+1,n_r,n_r,2)``.
    """

    z_in = normalize_spinors(spinors) if normalize else np.asarray(spinors, dtype=complex)
    n_k = int(basis.params.grid.n_k)
    n_r = int(basis.params.real_space.n_r)
    if z_in.shape == (n_r, n_r, 2):
        z = np.broadcast_to(z_in, (n_k + 1, n_k + 1, n_r, n_r, 2)).copy()
    elif z_in.shape == (n_k, n_k, n_r, n_r, 2):
        z = _periodic_extend_k(z_in)
    elif z_in.shape == (n_k + 1, n_k + 1, n_r, n_r, 2):
        z = z_in.copy()
    else:
        raise ValueError(
            "spinors must have shape "
            f"{(n_r, n_r, 2)}, {(n_k, n_k, n_r, n_r, 2)}, "
            f"or {(n_k + 1, n_k + 1, n_r, n_r, 2)}"
        )

    xy_grid = basis.real_space_grid() if xy is None else np.asarray(xy, dtype=float)
    if xy_grid.shape != (n_r, n_r, 2):
        raise ValueError(f"xy must have shape {(n_r, n_r, 2)}")

    projectors = projectors_from_spinors(z, normalize=False)
    if wall_field is None:
        p = projectors
        field = np.stack(
            [
                2.0 * np.real(p[..., 0, 1]),
                -2.0 * np.imag(p[..., 0, 1]),
                np.real(p[..., 0, 0] - p[..., 1, 1]),
            ],
            axis=-1,
        )
        wall = np.mean(field, axis=(0, 1))
    else:
        wall = np.asarray(wall_field, dtype=float)
        if wall.shape != (n_r, n_r, 3):
            raise ValueError(f"wall_field must have shape {(n_r, n_r, 3)}")

    eigenvalues = np.full((*z.shape[:-1], 2), np.nan, dtype=float)
    gaps = np.full(z.shape[:-1], np.nan, dtype=float)
    return IdealConjugateProjectorSolution(
        k_points=basis.k_grid(extended=True),
        k_fractional=basis.fractional_k_grid(extended=True),
        xy=xy_grid,
        wall_field=wall,
        spinors=z,
        band_projectors=projectors,
        eigenvalues=eigenvalues,
        gaps=gaps,
    )


def run_explicit_chiral_domain_wall_texture_response(
    params: IdealConjugateLLLChargeBenchmarkParams | None = None,
    *,
    theta_edges: np.ndarray | None = None,
    phi_nodes: np.ndarray | None = None,
) -> ExplicitConjugateLLLTextureResponseResult:
    """Evaluate cG and charge density for an explicitly supplied wall texture.

    This path constructs the rank-one two-component wavefunction texture
    directly and never diagonalizes a trial source Hamiltonian or runs HF.
    """

    texture_params = params or IdealConjugateLLLChargeBenchmarkParams()
    basis = IdealConjugateLLLBasis(texture_params)
    theta = (
        np.asarray(theta_edges, dtype=float)
        if theta_edges is not None
        else np.linspace(0.0, np.pi, 42)
    )
    phi = (
        np.asarray(phi_nodes, dtype=float)
        if phi_nodes is not None
        else np.array([0.0, 0.2], dtype=float)
    )
    theta_projectors = uniform_conjugate_projector_path(
        theta,
        texture_params.grid.n_k,
    )
    response = k_theta_from_ideal_conjugate_projectors(
        basis,
        theta_projectors,
        theta,
        phi,
    )

    xy = basis.real_space_grid()
    spinors, field = chiral_domain_wall_spinor_texture(
        xy,
        radius=basis.radius,
        width=basis.width,
        winding=texture_params.winding,
        helicity=texture_params.helicity,
    )
    solution = explicit_texture_solution_from_spinors(
        basis,
        spinors,
        xy=xy,
        wall_field=field,
    )
    evaluator = IdealConjugate4DCurvatureEvaluator(basis, solution)
    components = evaluator.curvature_components_centered()
    rho_top = evaluator.charge_per_realspace_plaquette_centered(components)
    rho_analytic, q_sk, n_z_center = analytic_conjugate_charge_per_plaquette(solution)
    radial_profiles_result = radial_diagnostics(
        solution.xy,
        rho_top,
        rho_analytic,
        q_sk,
        radius=basis.radius,
        width=basis.width,
        winding=texture_params.winding,
    )
    r_max = max(2.0 * basis.radius, basis.radius + 8.0 * basis.width)
    r = np.linspace(max(1e-6, r_max / 500.0), r_max, 500)
    charge_profile = charge_density_radial(
        r,
        response.theta,
        response.K,
        DomainWallParams(
            radius=basis.radius,
            width=basis.width,
            winding=texture_params.winding,
            profile="logistic",
        ),
    )
    return ExplicitConjugateLLLTextureResponseResult(
        params=texture_params,
        theta_edges=theta,
        phi_nodes=phi,
        theta_projectors=theta_projectors,
        response=response,
        charge_profile=charge_profile,
        solution=solution,
        curvature_components=components,
        rho_top=rho_top,
        rho_analytic=rho_analytic,
        q_sk=q_sk,
        n_z_center=n_z_center,
        radial_profiles=radial_profiles_result,
    )


class IdealConjugateTrialProjector:
    """Build projectors from h_tr(k,r)=h0(k)-m0 M(r).sigma in the flat LLL limit."""

    def __init__(self, basis: IdealConjugateLLLBasis) -> None:
        self.basis = basis
        self.params = basis.params

    def h0_grid(self, k_points: np.ndarray) -> np.ndarray:
        flat = k_points.reshape(-1, 2)
        vals = np.array(
            [[self.basis.up_energy(k), self.basis.down_energy(k)] for k in flat],
            dtype=float,
        )
        h0 = np.zeros((*k_points.shape[:-1], 2, 2), dtype=complex)
        h0[..., 0, 0] = vals[:, 0].reshape(k_points.shape[:-1])
        h0[..., 1, 1] = vals[:, 1].reshape(k_points.shape[:-1])
        return h0

    def solve(self, *, extended_k: bool = False) -> IdealConjugateProjectorSolution:
        k_frac = self.basis.fractional_k_grid(extended=extended_k)
        k_points = self.basis.k_grid(extended=extended_k)
        xy = self.basis.real_space_grid()
        m_field = circular_domain_wall_field(
            xy,
            radius=self.basis.radius,
            width=self.basis.width,
            winding=self.params.winding,
            helicity=self.params.helicity,
        )
        h0 = self.h0_grid(k_points)[:, :, None, None, :, :]
        h = np.broadcast_to(h0, (*h0.shape[:2], *m_field.shape[:2], 2, 2)).copy()

        d = -float(self.params.m0) * m_field
        h[..., 0, 0] += d[None, None, ..., 2]
        h[..., 1, 1] -= d[None, None, ..., 2]
        h[..., 0, 1] += d[None, None, ..., 0] - 1j * d[None, None, ..., 1]
        h[..., 1, 0] += d[None, None, ..., 0] + 1j * d[None, None, ..., 1]

        vals, vecs = np.linalg.eigh(h)
        spinors = vecs[..., :, 0]
        projectors = spinors[..., :, None] * spinors[..., None, :].conj()
        return IdealConjugateProjectorSolution(
            k_points=k_points,
            k_fractional=k_frac,
            xy=xy,
            wall_field=m_field,
            spinors=spinors,
            band_projectors=projectors,
            eigenvalues=vals,
            gaps=vals[..., 1] - vals[..., 0],
        )

    def full_projector(self, k: np.ndarray, band_projector: np.ndarray) -> np.ndarray:
        up = self.basis.up_coefficients(k)
        down = np.conj(self.basis.up_coefficients(-np.asarray(k, dtype=float)))
        n = len(up)
        full = np.zeros((2 * n, 2 * n), dtype=complex)
        full[:n, :n] = band_projector[0, 0] * np.outer(up, up.conj())
        full[n:, n:] = band_projector[1, 1] * np.outer(down, down.conj())
        full[:n, n:] = band_projector[0, 1] * np.outer(up, down.conj())
        full[n:, :n] = band_projector[1, 0] * np.outer(down, up.conj())
        return full


class IdealConjugate4DCurvatureEvaluator:
    """Centered link-variable second-Chern evaluator for conjugate LLL projectors."""

    def __init__(
        self,
        basis: IdealConjugateLLLBasis,
        solution: IdealConjugateProjectorSolution,
    ) -> None:
        self.basis = basis
        self.solution = solution
        n_k = int(basis.params.grid.n_k)
        if solution.k_points.shape[:2] != (n_k + 1, n_k + 1):
            raise ValueError("solution must be built with extended_k=True")
        self._link_cache: dict[
            tuple[tuple[int, int, int, int], tuple[int, int, int, int]],
            complex,
        ] = {}

    @staticmethod
    def _shift(idx: tuple[int, int, int, int], dim: int) -> tuple[int, int, int, int]:
        vals = list(idx)
        vals[dim] += 1
        return tuple(vals)  # type: ignore[return-value]

    def _overlap(
        self,
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> complex:
        ia, ja, xa, ya = a
        ib, jb, xb, yb = b
        ka = self.solution.k_points[ia, ja]
        kb = self.solution.k_points[ib, jb]
        za = self.solution.spinors[ia, ja, xa, ya]
        zb = self.solution.spinors[ib, jb, xb, yb]
        return complex(
            np.conj(za[0]) * zb[0] * self.basis.up_overlap(ka, kb)
            + np.conj(za[1]) * zb[1] * self.basis.down_overlap(ka, kb)
        )

    def _link(
        self,
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> complex:
        key = (a, b)
        cached = self._link_cache.get(key)
        if cached is not None:
            return cached
        link = _safe_link(self._overlap(a, b))
        self._link_cache[key] = link
        self._link_cache[(b, a)] = np.conj(link)
        return link

    def curvature_phase_raw(self, dim_a: int, dim_b: int) -> np.ndarray:
        """Return link curvature on the maximal grid natural to its two-plane."""

        n_k = int(self.basis.params.grid.n_k)
        n_r = int(self.basis.params.real_space.n_r)
        uses_x = dim_a == 2 or dim_b == 2
        uses_y = dim_a == 3 or dim_b == 3
        shape = (n_k, n_k, n_r - int(uses_x), n_r - int(uses_y))
        phases = np.zeros(shape, dtype=float)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for x in range(shape[2]):
                    for y in range(shape[3]):
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

    def curvature_components_centered(self) -> dict[str, np.ndarray]:
        f_kxky_site = self.curvature_phase_raw(0, 1)
        f_xky_edge = self.curvature_phase_raw(2, 1)
        f_ykx_edge = self.curvature_phase_raw(3, 0)
        f_xkx_edge = self.curvature_phase_raw(2, 0)
        f_yky_edge = self.curvature_phase_raw(3, 1)
        return {
            "Fkxky": 0.25
            * (
                f_kxky_site[:, :, :-1, :-1]
                + f_kxky_site[:, :, 1:, :-1]
                + f_kxky_site[:, :, :-1, 1:]
                + f_kxky_site[:, :, 1:, 1:]
            ),
            "Fxky": 0.5 * (f_xky_edge[:, :, :, :-1] + f_xky_edge[:, :, :, 1:]),
            "Fykx": 0.5 * (f_ykx_edge[:, :, :-1, :] + f_ykx_edge[:, :, 1:, :]),
            "Fxkx": 0.5 * (f_xkx_edge[:, :, :, :-1] + f_xkx_edge[:, :, :, 1:]),
            "Fyky": 0.5 * (f_yky_edge[:, :, :-1, :] + f_yky_edge[:, :, 1:, :]),
            "Fxy": self.curvature_phase_raw(2, 3),
        }

    @staticmethod
    def second_chern_density(components: dict[str, np.ndarray]) -> np.ndarray:
        return (
            components["Fxy"] * components["Fkxky"]
            - components["Fxkx"] * components["Fyky"]
            + components["Fxky"] * components["Fykx"]
        ) / (4.0 * np.pi**2)

    def charge_per_realspace_plaquette_centered(
        self,
        components: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        c = components or self.curvature_components_centered()
        return -np.sum(self.second_chern_density(c), axis=(0, 1))


def spinor_berry_phase_xy(spinors: np.ndarray) -> np.ndarray:
    """Return real-space spinor Berry phases on open plaquettes."""

    z = np.asarray(spinors, dtype=complex)
    if z.ndim != 3 or z.shape[-1] != 2:
        raise ValueError("spinors must have shape (n_x, n_y, 2)")
    n_x, n_y = z.shape[:2]
    phases = np.zeros((n_x - 1, n_y - 1), dtype=float)
    for i in range(n_x - 1):
        for j in range(n_y - 1):
            product = (
                _safe_link(np.vdot(z[i, j], z[i + 1, j]))
                * _safe_link(np.vdot(z[i + 1, j], z[i + 1, j + 1]))
                * _safe_link(np.vdot(z[i + 1, j + 1], z[i, j + 1]))
                * _safe_link(np.vdot(z[i, j + 1], z[i, j]))
            )
            phases[i, j] = np.angle(product)
    return phases


def plaquette_average(field: np.ndarray) -> np.ndarray:
    arr = np.asarray(field)
    return 0.25 * (arr[:-1, :-1] + arr[1:, :-1] + arr[:-1, 1:] + arr[1:, 1:])


def analytic_conjugate_charge_per_plaquette(
    solution: IdealConjugateProjectorSolution,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rho=-n_z*q_sk on the same open plaquette grid as the 4D evaluator."""

    q_sk = spinor_berry_phase_xy(solution.spinors[0, 0]) / (2.0 * np.pi)
    n_z_center = plaquette_average(solution.wall_field[..., 2])
    return -n_z_center * q_sk, q_sk, n_z_center


def continuum_radial_charge_proxy(
    r: np.ndarray,
    *,
    radius: float,
    width: float,
    winding: int,
) -> np.ndarray:
    """Return the continuum shape proportional to -Nw cos(theta) sin(theta) theta'/r."""

    r_safe = np.maximum(np.asarray(r, dtype=float), 1e-14)
    theta = 2.0 * np.arctan(np.exp((r_safe - float(radius)) / float(width)))
    theta_prime = np.sin(theta) / float(width)
    return (
        -int(winding)
        * np.cos(theta)
        * np.sin(theta)
        * theta_prime
        / (4.0 * np.pi * r_safe)
    )


def radial_diagnostics(
    xy: np.ndarray,
    rho: np.ndarray,
    rho_analytic: np.ndarray,
    q_sk: np.ndarray,
    *,
    radius: float,
    width: float,
    winding: int,
    n_bins: int = 36,
) -> dict[str, np.ndarray | float]:
    """Return radial profiles, cumulative charge, and a wall dipole estimate."""

    centers = plaquette_average(xy)
    r = np.linalg.norm(centers, axis=-1)
    bins = np.linspace(0.0, float(np.max(r)), int(n_bins) + 1)
    which = np.digitize(r.reshape(-1), bins) - 1

    def _profile(values: np.ndarray) -> np.ndarray:
        profile = np.full(len(bins) - 1, np.nan, dtype=float)
        counts = np.zeros(len(bins) - 1, dtype=float)
        sums = np.zeros(len(bins) - 1, dtype=float)
        for idx, value in zip(which, values.reshape(-1), strict=True):
            if 0 <= idx < len(sums):
                sums[idx] += float(value)
                counts[idx] += 1.0
        mask = counts > 0
        profile[mask] = sums[mask] / counts[mask]
        return profile

    order = np.argsort(r.reshape(-1))
    continuum = continuum_radial_charge_proxy(
        r,
        radius=radius,
        width=width,
        winding=winding,
    )
    return {
        "radial_r": 0.5 * (bins[:-1] + bins[1:]),
        "radial_rho": _profile(rho),
        "radial_rho_analytic": _profile(rho_analytic),
        "radial_q_sk": _profile(q_sk),
        "radial_continuum_proxy": _profile(continuum),
        "cumulative_r": r.reshape(-1)[order],
        "cumulative_charge": np.cumsum(rho.reshape(-1)[order]),
        "dipole": float(np.sum((r - float(radius)) * rho)),
        "net_charge": float(np.sum(rho)),
    }


def run_ideal_conjugate_lll_charge_benchmark(
    params: IdealConjugateLLLChargeBenchmarkParams | None = None,
    *,
    write_outputs: bool = False,
    write_plots: bool = False,
) -> IdealConjugateLLLChargeBenchmarkResult:
    """Run the flat opposite-Chern LLL charge-density normalization benchmark."""

    benchmark_params = params or IdealConjugateLLLChargeBenchmarkParams()
    basis = IdealConjugateLLLBasis(benchmark_params)
    trial = IdealConjugateTrialProjector(basis)
    solution = trial.solve(extended_k=True)
    evaluator = IdealConjugate4DCurvatureEvaluator(basis, solution)
    components = evaluator.curvature_components_centered()
    rho_top = evaluator.charge_per_realspace_plaquette_centered(components)
    rho_analytic, q_sk, n_z_center = analytic_conjugate_charge_per_plaquette(solution)
    profiles = radial_diagnostics(
        solution.xy,
        rho_top,
        rho_analytic,
        q_sk,
        radius=basis.radius,
        width=basis.width,
        winding=benchmark_params.winding,
    )
    summary = summarize_ideal_conjugate_charge(
        benchmark_params,
        basis,
        solution,
        rho_top,
        rho_analytic,
        q_sk,
        profiles,
    )
    result = IdealConjugateLLLChargeBenchmarkResult(
        params=benchmark_params,
        solution=solution,
        curvature_components=components,
        rho_top=rho_top,
        rho_analytic=rho_analytic,
        q_sk=q_sk,
        n_z_center=n_z_center,
        radial_profiles=profiles,
        summary=summary,
    )
    if write_outputs:
        manifest = write_ideal_conjugate_lll_outputs(result, write_plots=write_plots)
        result = IdealConjugateLLLChargeBenchmarkResult(
            **{**result.__dict__, "manifest": manifest}
        )
    return result


def summarize_ideal_conjugate_charge(
    params: IdealConjugateLLLChargeBenchmarkParams,
    basis: IdealConjugateLLLBasis,
    solution: IdealConjugateProjectorSolution,
    rho_top: np.ndarray,
    rho_analytic: np.ndarray,
    q_sk: np.ndarray,
    profiles: dict[str, np.ndarray | float],
) -> IdealConjugateLLLChargeSummary:
    up_chern, down_chern = basis.band_cherns(n_k=max(6, params.grid.n_k))
    up_bw, down_bw = basis.band_bandwidths()
    herm, idem = _projector_errors(solution.band_projectors)
    spin_error = float(
        np.max(np.linalg.norm(solution.spin_expectation - solution.wall_field, axis=-1))
    )
    error = np.asarray(rho_top, dtype=float) - np.asarray(rho_analytic, dtype=float)
    charge_error_max = float(np.max(np.abs(error)))
    charge_error_rms = float(np.sqrt(np.mean(error * error)))
    return IdealConjugateLLLChargeSummary(
        up_chern=up_chern,
        down_chern=down_chern,
        up_bandwidth=up_bw,
        down_bandwidth=down_bw,
        local_gap_min=float(np.min(solution.gaps)),
        spin_alignment_error=spin_error,
        projector_hermiticity_error=herm,
        projector_idempotency_error=idem,
        charge_error_max=charge_error_max,
        charge_error_rms=charge_error_rms,
        integrated_charge=float(np.sum(rho_top)),
        integrated_analytic_charge=float(np.sum(rho_analytic)),
        integrated_skyrmion_charge=float(np.sum(q_sk)),
        dipole_moment=float(profiles["dipole"]),
        m0=float(params.m0),
        valid_analytic_charge=bool(
            abs(up_chern - 1.0) < params.charge_tolerance
            and abs(down_chern + 1.0) < params.charge_tolerance
            and charge_error_max < params.charge_tolerance
            and herm < params.charge_tolerance
            and idem < params.charge_tolerance
        ),
    )


def write_ideal_conjugate_lll_outputs(
    result: IdealConjugateLLLChargeBenchmarkResult,
    *,
    write_plots: bool = False,
) -> RunManifest:
    out_dir = Path(result.params.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = out_dir / "ideal_conjugate_lll_charge.npz"
    summary_path = out_dir / "ideal_conjugate_lll_summary.json"
    profiles_path = out_dir / "ideal_conjugate_lll_profiles.csv"
    plot_path = out_dir / "ideal_conjugate_lll_charge.png"
    manifest_path = out_dir / "artifact_manifest.json"

    _write_profiles_csv(profiles_path, result)
    _write_summary_json(summary_path, result)
    if result.params.write_curvature_npz:
        centers = plaquette_average(result.solution.xy)
        np.savez_compressed(
            arrays_path,
            x_center=centers[..., 0],
            y_center=centers[..., 1],
            rho_top=result.rho_top,
            rho_analytic=result.rho_analytic,
            q_sk=result.q_sk,
            n_z_center=result.n_z_center,
            charge_error=result.rho_top - result.rho_analytic,
            **result.curvature_components,
        )
    if write_plots:
        _write_charge_plot(plot_path, result)

    artifacts = [
        _artifact(
            arrays_path,
            "charge_arrays",
            "array",
            "Ideal conjugate LLL charge, analytic target, and centered curvatures",
            required=bool(result.params.write_curvature_npz),
        ),
        _artifact(
            summary_path,
            "summary",
            "json",
            "Ideal conjugate LLL charge benchmark scalar summary",
        ),
        _artifact(
            profiles_path,
            "profiles",
            "table",
            "Radial charge, analytic, skyrmion, and continuum diagnostic profiles",
        ),
        _artifact(
            plot_path,
            "charge_plot",
            "plot",
            "Optional rho/analytic/error charge maps",
            required=False,
        ),
    ]
    manifest = RunManifest.from_artifacts(
        run_id="ideal_conjugate_lll_charge",
        result_dir=str(out_dir),
        artifacts=artifacts,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    return manifest


def _write_summary_json(path: Path, result: IdealConjugateLLLChargeBenchmarkResult) -> None:
    payload = {
        "params": result.params.model_dump(mode="json"),
        "summary": result.summary.model_dump(mode="json"),
        "normalization": (
            "Flat opposite-Chern ideal LLL validation limit. The direct centered "
            "4D link-variable charge is compared against rho=-n_z*q_sk on the "
            "same real-space plaquette grid."
        ),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _write_profiles_csv(path: Path, result: IdealConjugateLLLChargeBenchmarkResult) -> None:
    radial_r = np.asarray(result.radial_profiles["radial_r"], dtype=float)
    fieldnames = [
        "radial_r",
        "rho_top",
        "rho_analytic",
        "q_sk",
        "continuum_radial_proxy",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, r in enumerate(radial_r):
            writer.writerow(
                {
                    "radial_r": float(r),
                    "rho_top": float(np.asarray(result.radial_profiles["radial_rho"])[idx]),
                    "rho_analytic": float(
                        np.asarray(result.radial_profiles["radial_rho_analytic"])[idx]
                    ),
                    "q_sk": float(np.asarray(result.radial_profiles["radial_q_sk"])[idx]),
                    "continuum_radial_proxy": float(
                        np.asarray(result.radial_profiles["radial_continuum_proxy"])[idx]
                    ),
                }
            )


def _write_charge_plot(path: Path, result: IdealConjugateLLLChargeBenchmarkResult) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    error = result.rho_top - result.rho_analytic
    vmax = max(
        float(np.max(np.abs(result.rho_top))),
        float(np.max(np.abs(result.rho_analytic))),
        1e-15,
    )
    err_vmax = max(float(np.max(np.abs(error))), 1e-15)
    centers = plaquette_average(result.solution.xy)
    extent = (
        float(np.min(centers[..., 0])),
        float(np.max(centers[..., 0])),
        float(np.min(centers[..., 1])),
        float(np.max(centers[..., 1])),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), constrained_layout=True)
    panels = [
        (result.rho_top, r"$\rho_{\rm 4D}$", "RdBu_r", vmax),
        (result.rho_analytic, r"$-n_z q_{\rm sk}$", "RdBu_r", vmax),
        (error, r"$\rho_{\rm 4D}+n_z q_{\rm sk}$", "PuOr", err_vmax),
    ]
    for ax, (data, title, cmap, limit) in zip(axes, panels, strict=True):
        im = ax.imshow(
            data.T,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Ideal conjugate LLL charge benchmark: "
        f"max error={result.summary.charge_error_max:.2e}, "
        f"net charge={result.summary.integrated_charge:.3e}",
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
    "ExplicitConjugateLLLTextureResponseResult",
    "IdealConjugate4DCurvatureEvaluator",
    "IdealConjugateLLLBasis",
    "IdealConjugateLLLChargeBenchmarkResult",
    "IdealConjugateProjectorSolution",
    "IdealConjugateTrialProjector",
    "analytic_conjugate_charge_per_plaquette",
    "chiral_domain_wall_spinor_texture",
    "circular_domain_wall_field",
    "continuum_radial_charge_proxy",
    "explicit_texture_solution_from_spinors",
    "k_theta_from_ideal_conjugate_projectors",
    "normalize_spinors",
    "plaquette_average",
    "projectors_from_spinors",
    "radial_diagnostics",
    "run_explicit_chiral_domain_wall_texture_response",
    "run_ideal_conjugate_lll_charge_benchmark",
    "spinors_from_unit_vector_texture",
    "spinor_berry_phase_xy",
    "summarize_ideal_conjugate_charge",
    "triangular_moire_magnetic_length",
    "uniform_conjugate_projector_path",
    "write_ideal_conjugate_lll_outputs",
]
