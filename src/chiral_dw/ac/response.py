"""AC-specific projector response with magnetic-Bloch orbital overlaps."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from chiral_dw.ac.nonideal import NonIdealACLLModel
from chiral_dw.ac.sewing import ACReciprocalTransport
from chiral_dw.continuum.models import ContinuumActiveSpace, MomentumGrid, hermitize
from chiral_dw.response import KThetaResult, compute_cG, rotate_projector_phi


def _safe_unit(value: complex, eps: float = 1e-14) -> complex:
    magnitude = abs(value)
    if magnitude < eps:
        return 1.0 + 0.0j
    return complex(value / magnitude)


@dataclass
class ACBandOverlapProvider:
    """Cached orbital overlaps for one AC active band and its T' partner.

    When ``active`` is supplied, mesh and reciprocal-shifted eigenvectors are
    phase-anchored to the exact band frame used to build the HF active space.
    This keeps the orbital overlaps and coefficient-space HF projectors in the
    same gauge on the Brillouin-zone torus.
    """

    model: NonIdealACLLModel
    active_band: int = 0
    active: ContinuumActiveSpace | None = None
    key_decimals: int = 12
    _coeff_cache: dict[tuple[float, float], np.ndarray] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _up_overlap_cache: dict[
        tuple[tuple[float, float], tuple[float, float]],
        complex,
    ] = field(default_factory=dict, init=False, repr=False)

    def _key(self, k: np.ndarray) -> tuple[float, float]:
        arr = np.asarray(k, dtype=float)
        rounded = np.round(arr, decimals=int(self.key_decimals))
        return float(rounded[0]), float(rounded[1])

    def k_from_fractional(self, frac: tuple[float, float] | np.ndarray) -> np.ndarray:
        f = np.asarray(frac, dtype=float)
        b1, b2 = self.model.fields.G_shell[0], self.model.fields.G_shell[1]
        return f[0] * b1 + f[1] * b2

    def _active_mesh_reference(self, k: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the wrapped HF-frame momentum and up-valley vector for ``k``."""

        if self.active is None:
            return None
        if self.active.n_active != 1:
            raise ValueError("AC overlap provider requires one active band per valley")
        if self.active.bands is None:
            raise ValueError("AC active space is missing band metadata")
        if self.active.band_vectors.shape[2] != self.model.n_ll:
            raise ValueError("AC active-space and overlap-model Landau-level dimensions differ")

        b1, b2 = self.model.fields.G_shell[0], self.model.fields.G_shell[1]
        fractional = np.linalg.solve(
            np.column_stack([b1, b2]),
            np.asarray(k, dtype=float),
        )
        mesh_coord = fractional * float(self.active.grid.n_k)
        integer_coord = np.rint(mesh_coord).astype(int)
        if not np.allclose(mesh_coord, integer_coord, atol=1e-9, rtol=0.0):
            return None

        wrapped, _shift = self.active.grid.fold_grid_coord(
            (int(integer_coord[0]), int(integer_coord[1]))
        )
        index = self.active.grid.index_of(wrapped)
        wrapped_fractional = np.asarray(wrapped, dtype=float) / float(self.active.grid.n_k)
        wrapped_k = self.k_from_fractional(wrapped_fractional)
        wrapped_coefficients = np.asarray(
            self.active.band_vectors[index, 0, :, 0],
            dtype=complex,
        )
        return wrapped_k, wrapped_coefficients

    def up_coefficients(self, k: np.ndarray) -> np.ndarray:
        key = self._key(k)
        cached = self._coeff_cache.get(key)
        if cached is not None:
            return cached

        momentum = np.asarray(k, dtype=float)
        reference = self._active_mesh_reference(momentum)
        if reference is None:
            sol = self.model.solve(momentum, active_band=int(self.active_band))
            coeffs = sol.eigenvectors[:, int(self.active_band)]
        else:
            wrapped_k, wrapped_coefficients = reference
            if np.allclose(momentum, wrapped_k, atol=1e-12, rtol=0.0):
                coeffs = wrapped_coefficients.copy()
            else:
                sol = self.model.solve(momentum, active_band=int(self.active_band))
                raw = sol.eigenvectors[:, int(self.active_band)]
                # H(k + G) and H(k) have the same finite-LL coefficient
                # representation.  Stabilize only the eigensolver phase here;
                # the magnetic-Bloch state overlap below must retain its
                # physical reciprocal-boundary phase.
                sewing_overlap = complex(np.vdot(wrapped_coefficients, raw))
                if abs(sewing_overlap) < 1e-14:
                    raise ValueError(
                        "AC reciprocal-boundary sewing overlap is too small to fix the band gauge"
                    )
                coeffs = raw * np.conj(_safe_unit(sewing_overlap))
        self._coeff_cache[key] = coeffs
        return coeffs

    def up_overlap(self, k: np.ndarray, p: np.ndarray) -> complex:
        key = (self._key(k), self._key(p))
        cached = self._up_overlap_cache.get(key)
        if cached is not None:
            return cached
        overlap = complex(
            self.model.state_overlap(
                np.asarray(k, dtype=float),
                self.up_coefficients(k),
                np.asarray(p, dtype=float),
                self.up_coefficients(p),
            )
        )
        self._up_overlap_cache[key] = overlap
        self._up_overlap_cache[(key[1], key[0])] = np.conj(overlap)
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

    def sewn_active_overlap_fractional(
        self,
        frac_k: tuple[float, float] | np.ndarray,
        frac_p: tuple[float, float] | np.ndarray,
    ) -> np.ndarray:
        """Return the raw overlap expressed in both folded active frames.

        The microscopic LL overlap is evaluated at the requested raw
        momenta.  Reciprocal sewing matrices then map its two coefficient
        indices back to the stored fundamental-cell frames.  An active-space
        frame is required so reciprocal images use exactly the HF band gauge.
        """

        if self.active is None:
            raise ValueError("sewn AC overlaps require an active-space band frame")
        raw_k = np.asarray(frac_k, dtype=float)
        raw_p = np.asarray(frac_p, dtype=float)
        up = self._sewn_up_overlap_fractional(raw_k, raw_p)
        down = np.conj(self._sewn_up_overlap_fractional(-raw_k, -raw_p))
        return np.diag([up, down]).astype(complex)

    def _sewn_up_overlap_fractional(
        self,
        raw_k: np.ndarray,
        raw_p: np.ndarray,
    ) -> complex:
        """Return the up-valley overlap in its folded active-band frame."""

        transport = ACReciprocalTransport(self.model)
        folded_k, shift_k = transport.fold_fractional(raw_k)
        folded_p, shift_p = transport.fold_fractional(raw_p)
        sewing_k = transport.valley_phase(
            self.k_from_fractional(folded_k),
            shift_k,
            valley=1,
        )
        sewing_p = transport.valley_phase(
            self.k_from_fractional(folded_p),
            shift_p,
            valley=1,
        )
        raw_overlap = self.up_overlap(
            self.k_from_fractional(raw_k),
            self.k_from_fractional(raw_p),
        )
        return complex(sewing_k * raw_overlap * np.conj(sewing_p))

    def band_cherns(self, n_k: int = 9) -> tuple[float, float]:
        up = self.model.berry_curvature_fukui(
            n_k=int(n_k),
            active_band=int(self.active_band),
        )[2]
        return float(up), float(-up)


def _occupied_spinor(projector: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(hermitize(np.asarray(projector, dtype=complex)))
    return vecs[:, int(np.argmax(vals))]


def _coord_fractional(coord: tuple[int, int], n_k: int) -> tuple[float, float]:
    return float(coord[0]) / float(n_k), float(coord[1]) / float(n_k)


def _spinor_link(
    provider: ACBandOverlapProvider,
    za: np.ndarray,
    zb: np.ndarray,
    coord_a: tuple[int, int],
    coord_b: tuple[int, int],
    n_k: int,
) -> complex:
    overlap = provider.sewn_active_overlap_fractional(
        _coord_fractional(coord_a, n_k),
        _coord_fractional(coord_b, n_k),
    )
    return complex(za.conj() @ overlap @ zb)


class ACProjectorChernDiagnostics(BaseModel):
    """Gauge-covariant lattice-Chern and link-admissibility diagnostics."""

    model_config = ConfigDict(frozen=True)

    chern: float
    integer_residual: float = Field(ge=0.0)
    min_link_magnitude: float = Field(ge=0.0)
    small_link_count: int = Field(ge=0)
    link_tolerance: float = Field(gt=0.0)
    max_abs_plaquette_phase: float = Field(ge=0.0)
    min_branch_margin: float = Field(ge=0.0)
    translated_edge_closure_residual: float = Field(ge=0.0)
    numerically_resolved: bool


def ac_projector_chern_diagnostics(
    provider: ACBandOverlapProvider,
    grid: MomentumGrid,
    P: np.ndarray,
    *,
    link_tolerance: float = 1e-8,
) -> ACProjectorChernDiagnostics:
    """Return a shared-link Chern result on the sewn magnetic-Bloch torus."""

    tolerance = float(link_tolerance)
    if tolerance <= 0.0:
        raise ValueError("link_tolerance must be positive")
    arr = hermitize(np.asarray(P, dtype=complex))
    if arr.shape == (grid.n_k, grid.n_k, 2, 2):
        arr = arr.reshape(grid.size, 2, 2)
    if arr.shape != (grid.size, 2, 2):
        raise ValueError("P must have shape (grid.size,2,2) or (n_k,n_k,2,2)")
    spinors = np.asarray([_occupied_spinor(arr[ik]) for ik in range(grid.size)])
    n = grid.n_k
    links_x = np.empty((n, n), dtype=complex)
    links_y = np.empty((n, n), dtype=complex)
    magnitudes = np.empty((n, n, 2), dtype=float)
    for i in range(n):
        for j in range(n):
            a = (i, j)
            za = spinors[grid.index_of(a)]
            bx = (i + 1, j)
            by = (i, j + 1)
            raw_x = _spinor_link(
                provider,
                za,
                spinors[grid.index_of(bx)],
                a,
                bx,
                n,
            )
            raw_y = _spinor_link(
                provider,
                za,
                spinors[grid.index_of(by)],
                a,
                by,
                n,
            )
            magnitudes[i, j] = abs(raw_x), abs(raw_y)
            links_x[i, j] = _safe_unit(raw_x)
            links_y[i, j] = _safe_unit(raw_y)

    phases = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            product = (
                links_x[i, j]
                * links_y[(i + 1) % n, j]
                * np.conj(links_x[i, (j + 1) % n])
                * np.conj(links_y[i, j])
            )
            phases[i, j] = float(np.angle(product))
    chern = float(np.sum(phases) / (2.0 * np.pi))

    closure = 0.0
    for j in range(n):
        start = np.array([0.0, j / n])
        stop = np.array([0.0, (j + 1) / n])
        baseline = provider.sewn_active_overlap_fractional(start, stop)
        translated = provider.sewn_active_overlap_fractional(
            start + np.array([1.0, 0.0]),
            stop + np.array([1.0, 0.0]),
        )
        closure = max(closure, float(np.max(np.abs(translated - baseline))))
    for i in range(n):
        start = np.array([i / n, 0.0])
        stop = np.array([(i + 1) / n, 0.0])
        baseline = provider.sewn_active_overlap_fractional(start, stop)
        translated = provider.sewn_active_overlap_fractional(
            start + np.array([0.0, 1.0]),
            stop + np.array([0.0, 1.0]),
        )
        closure = max(closure, float(np.max(np.abs(translated - baseline))))

    min_link = float(np.min(magnitudes))
    small_count = int(np.count_nonzero(magnitudes < tolerance))
    max_phase = float(np.max(np.abs(phases)))
    branch_margin = float(max(0.0, np.pi - max_phase))
    integer_residual = float(abs(chern - np.rint(chern)))
    resolved = bool(
        small_count == 0
        and integer_residual < 1e-10
        and closure < 1e-10
    )
    return ACProjectorChernDiagnostics(
        chern=chern,
        integer_residual=integer_residual,
        min_link_magnitude=min_link,
        small_link_count=small_count,
        link_tolerance=tolerance,
        max_abs_plaquette_phase=max_phase,
        min_branch_margin=branch_margin,
        translated_edge_closure_residual=closure,
        numerically_resolved=resolved,
    )


def ac_projector_chern(
    provider: ACBandOverlapProvider,
    grid: MomentumGrid,
    P: np.ndarray,
) -> float:
    """Return the occupied-projector Chern number on the sewn AC torus."""

    return ac_projector_chern_diagnostics(provider, grid, P).chern


def ac_reference_cherns_are_valid(
    cherns: dict[str, float],
    *,
    atol: float = 5e-3,
) -> bool:
    """Return whether VP+/VP-/IVC have their symmetry-required Chern values."""

    expected = {"vp_plus": 1.0, "vp_minus": -1.0, "ivc": 0.0}
    if set(expected) - set(cherns):
        return False
    return all(
        np.isfinite(float(cherns[name]))
        and np.isclose(float(cherns[name]), target, atol=float(atol), rtol=0.0)
        for name, target in expected.items()
    )


def _validate_projector_grid(P_theta: np.ndarray, theta: np.ndarray) -> np.ndarray:
    P = hermitize(np.asarray(P_theta, dtype=complex))
    if P.ndim != 5 or P.shape[-2:] != (2, 2):
        raise ValueError("P_theta must have shape (n_theta,n_k,n_k,2,2)")
    if P.shape[1] != P.shape[2]:
        raise ValueError("momentum grid must be square")
    if P.shape[0] != len(theta):
        raise ValueError("theta_edges length must match P_theta leading dimension")
    return P


def _phi_projector_grid(P_theta: np.ndarray, phi_nodes: np.ndarray) -> np.ndarray:
    n_theta, n_k, _, dim, _ = P_theta.shape
    out = np.zeros((n_k + 1, n_k + 1, n_theta, len(phi_nodes), dim, dim), dtype=complex)
    for i in range(n_k + 1):
        for j in range(n_k + 1):
            base = P_theta[:, i % n_k, j % n_k]
            for ip, phi in enumerate(phi_nodes):
                out[i, j, :, ip] = rotate_projector_phi(base, float(phi))
    return hermitize(out)


def k_theta_from_ac_projectors(
    provider: ACBandOverlapProvider,
    P_theta_edges: np.ndarray,
    theta_edges: np.ndarray,
    phi_nodes: np.ndarray,
) -> KThetaResult:
    """Compute K(theta) and cG from AC projectors using link-variable overlaps."""

    theta = np.asarray(theta_edges, dtype=float)
    phi = np.asarray(phi_nodes, dtype=float)
    P = _validate_projector_grid(P_theta_edges, theta)
    if len(theta) < 2:
        raise ValueError("theta_edges must contain at least two values")
    if len(phi) < 2:
        raise ValueError("phi_nodes must contain at least two values")
    if np.any(np.diff(theta) <= 0.0):
        raise ValueError("theta_edges must be strictly increasing")
    if np.any(np.diff(phi) <= 0.0):
        raise ValueError("phi_nodes must be strictly increasing")

    projectors = _phi_projector_grid(P, phi)
    n_k = P.shape[1]
    n_theta = len(theta)
    n_phi = len(phi)
    overlap_cache: dict[tuple[tuple[int, int], tuple[int, int]], np.ndarray] = {}

    def shifted(idx: tuple[int, int, int, int], dim: int) -> tuple[int, int, int, int]:
        vals = list(idx)
        vals[dim] += 1
        return tuple(vals)  # type: ignore[return-value]

    def projector(idx: tuple[int, int, int, int]) -> np.ndarray:
        i, j, it, ip = idx
        return projectors[i, j, it, ip]

    def orbital_overlap(
        idx_a: tuple[int, int, int, int],
        idx_b: tuple[int, int, int, int],
    ) -> np.ndarray:
        coord_a = (idx_a[0], idx_a[1])
        coord_b = (idx_b[0], idx_b[1])
        key = (coord_a, coord_b)
        cached = overlap_cache.get(key)
        if cached is not None:
            return cached
        overlap = provider.sewn_active_overlap_fractional(
            _coord_fractional(coord_a, n_k),
            _coord_fractional(coord_b, n_k),
        )
        overlap_cache[key] = overlap
        overlap_cache[(coord_b, coord_a)] = overlap.conj().T
        return overlap

    def plaquette_phase(
        idx0: tuple[int, int, int, int],
        idx1: tuple[int, int, int, int],
        idx2: tuple[int, int, int, int],
        idx3: tuple[int, int, int, int],
    ) -> float:
        product = np.trace(
            projector(idx0)
            @ orbital_overlap(idx0, idx1)
            @ projector(idx1)
            @ orbital_overlap(idx1, idx2)
            @ projector(idx2)
            @ orbital_overlap(idx2, idx3)
            @ projector(idx3)
            @ orbital_overlap(idx3, idx0)
        )
        return float(np.angle(product))

    def curvature_phase_raw(dim_a: int, dim_b: int) -> np.ndarray:
        uses_theta = dim_a == 2 or dim_b == 2
        uses_phi = dim_a == 3 or dim_b == 3
        shape = (
            n_k,
            n_k,
            n_theta - int(uses_theta),
            n_phi - int(uses_phi),
        )
        phases = np.zeros(shape, dtype=float)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for it in range(shape[2]):
                    for ip in range(shape[3]):
                        idx = (i, j, it, ip)
                        idx_a = shifted(idx, dim_a)
                        idx_b = shifted(idx, dim_b)
                        idx_ab = shifted(idx_a, dim_b)
                        phases[i, j, it, ip] = plaquette_phase(idx, idx_a, idx_ab, idx_b)
        return phases

    f_k1k2_site = curvature_phase_raw(0, 1)
    f_thk2_edge = curvature_phase_raw(2, 1)
    f_phk1_edge = curvature_phase_raw(3, 0)
    f_thk1_edge = curvature_phase_raw(2, 0)
    f_phk2_edge = curvature_phase_raw(3, 1)
    f_thph = curvature_phase_raw(2, 3)
    components = {
        "Fk1k2": 0.25
        * (
            f_k1k2_site[:, :, :-1, :-1]
            + f_k1k2_site[:, :, 1:, :-1]
            + f_k1k2_site[:, :, :-1, 1:]
            + f_k1k2_site[:, :, 1:, 1:]
        ),
        "Fthk2": 0.5 * (f_thk2_edge[:, :, :, :-1] + f_thk2_edge[:, :, :, 1:]),
        "Fphk1": 0.5 * (f_phk1_edge[:, :, :-1, :] + f_phk1_edge[:, :, 1:, :]),
        "Fthk1": 0.5 * (f_thk1_edge[:, :, :, :-1] + f_thk1_edge[:, :, :, 1:]),
        "Fphk2": 0.5 * (f_phk2_edge[:, :, :-1, :] + f_phk2_edge[:, :, 1:, :]),
        "Fthph": f_thph,
    }
    density = (
        components["Fthph"] * components["Fk1k2"]
        - components["Fthk1"] * components["Fphk2"]
        + components["Fthk2"] * components["Fphk1"]
    ) / (4.0 * np.pi**2)
    theta_centers = 0.5 * (theta[:-1] + theta[1:])
    dtheta = np.diff(theta)
    dphi_total = float(phi[-1] - phi[0])
    K = np.sum(density, axis=(0, 1, 3)) / (dtheta * dphi_total)
    return KThetaResult(theta=theta_centers, K=np.asarray(K, dtype=float), cG=compute_cG(theta_centers, K))


__all__ = [
    "ACBandOverlapProvider",
    "ACProjectorChernDiagnostics",
    "ac_projector_chern",
    "ac_projector_chern_diagnostics",
    "ac_reference_cherns_are_valid",
    "k_theta_from_ac_projectors",
]
