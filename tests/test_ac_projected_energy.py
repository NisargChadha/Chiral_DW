import numpy as np

from chiral_dw.ac.energy import ProjectedPhysicalEnergy
from chiral_dw.ac.nonideal import NonIdealACLLModel
from chiral_dw.ac.source import FlavorSourceProjector
from chiral_dw.config import FirstShellACParams, GatedInteractionParams


def _small_source() -> FlavorSourceProjector:
    model = NonIdealACLLModel(FirstShellACParams(b1=0.2, u1=0.1, n_ll=3))
    return FlavorSourceProjector(model, n_k=3, n_vec=(0.0, 0.0, 1.0))


def test_source_projector_is_rank_one_and_gapped():
    source = _small_source()
    result = source.projector_and_diagnostics(m0=0.4)
    P = result.projector

    assert result.min_gap > 0.0
    assert np.allclose(P, P.conj().swapaxes(-1, -2), atol=1e-12)
    assert np.allclose(P @ P, P, atol=1e-12)
    assert np.allclose(np.trace(P, axis1=-2, axis2=-1), 1.0, atol=1e-12)
    assert result.parallel_polarization > 0.0


def test_physical_energy_excludes_source_term():
    source = _small_source()
    energy = ProjectedPhysicalEnergy(
        source,
        GatedInteractionParams(v0=1.0, gate_distance=2.0, interaction_shell=1),
    )
    m0 = 0.7
    P = source.projector(m0)
    breakdown = energy.energy(P)
    polarization, parallel = source.polarization(P)
    source_energy = -m0 * parallel

    assert np.isfinite(source_energy)
    assert abs(source_energy) > 1e-3
    assert np.isclose(breakdown.total, breakdown.band + breakdown.hartree + breakdown.fock)
    assert not np.isclose(breakdown.total, breakdown.band + breakdown.hartree + breakdown.fock + source_energy)
    assert polarization.shape == (3,)


def test_hartree_terms_omit_uniform_q0_contribution():
    source = _small_source()
    energy = ProjectedPhysicalEnergy(
        source,
        GatedInteractionParams(v0=1.0, gate_distance=2.0, interaction_shell=1),
    )

    for _V, up, down in energy.hartree_terms():
        assert up.shape == down.shape == (source.n_total,)
    assert all(np.linalg.norm(G) > 1e-12 for G in energy.reciprocal_images if energy.interaction_value(G) != 0.0)


def test_uniform_fixed_spinor_energy_is_phi_independent():
    source = _small_source()
    energy = ProjectedPhysicalEnergy(
        source,
        GatedInteractionParams(v0=1.0, gate_distance=2.0, interaction_shell=1),
    )
    theta = 0.73
    values = []
    for phi in [0.0, 0.4, 1.7]:
        P = source.fixed_spinor_projector(theta, phi)
        values.append(energy.fock_energy(P))

    assert np.allclose(values, values[0], atol=1e-12)


def test_down_form_factor_matches_time_reversal_definition():
    source = _small_source()
    energy = ProjectedPhysicalEnergy(
        source,
        GatedInteractionParams(v0=1.0, gate_distance=2.0, interaction_shell=1),
    )
    k = source.k_mesh[1]
    q = source.k_mesh[2]
    G = source.b1 - source.b2

    down = energy.active_down_form_factor(k, q, G)
    expected = np.conj(energy.active_up_form_factor(-k, -q, -G))
    assert np.allclose(down, expected, atol=1e-12)
