"""Circular chiral domain-wall texture and charge profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chiral_dw.config import DomainWallParams, UnitsParams


@dataclass(frozen=True)
class DomainWallChargeProfile:
    """Radial domain-wall charge data in dimensionless moire units."""

    r: np.ndarray
    theta: np.ndarray
    K_theta: np.ndarray
    rho_dimless: np.ndarray

    def rho_physical(self, units: UnitsParams) -> np.ndarray:
        return self.rho_dimless * units.physical_density_scale


def theta_profile(r: np.ndarray, params: DomainWallParams) -> np.ndarray:
    """Return theta(r)=2 atan exp((r-R)/w)."""
    x = (np.asarray(r, dtype=float) - params.radius) / params.width
    return 2.0 * np.arctan(np.exp(x))


def dtheta_dr(r: np.ndarray, params: DomainWallParams) -> np.ndarray:
    """Return derivative of the logistic/tanh domain-wall theta profile."""
    x = (np.asarray(r, dtype=float) - params.radius) / params.width
    return 1.0 / (params.width * np.cosh(x))


def charge_density_radial(
    r: np.ndarray,
    theta_nodes: np.ndarray,
    K_nodes: np.ndarray,
    params: DomainWallParams,
    r_min: float = 1e-12,
) -> DomainWallChargeProfile:
    """Return rho=Nw dtheta_dr K(theta(r))/r in dimensionless units."""
    rr = np.asarray(r, dtype=float)
    theta_r = theta_profile(rr, params)
    K_r = np.interp(theta_r, np.asarray(theta_nodes, dtype=float), np.asarray(K_nodes, dtype=float))
    rho = params.winding * dtheta_dr(rr, params) * K_r / np.maximum(rr, float(r_min))
    return DomainWallChargeProfile(r=rr, theta=theta_r, K_theta=K_r, rho_dimless=rho)
