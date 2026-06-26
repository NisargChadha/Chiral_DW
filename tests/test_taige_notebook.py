import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "taige_continuum_symmetric_hf.ipynb"
SCRIPT = ROOT / "notebooks" / "taige_continuum_symmetric_hf.py"


def test_taige_notebook_json_is_valid_and_paired_with_jupytext_script():
    data = json.loads(NOTEBOOK.read_text())
    assert data["metadata"]["jupytext"]["formats"] == "ipynb,py:percent"
    assert SCRIPT.exists()


def test_taige_notebook_uses_fixed_scale_3x2_projector_diagnostic():
    text = SCRIPT.read_text()
    assert "def plot_projector_diagnostics" in text
    assert "plt.subplots(3, 2" in text
    assert "\"P_KK\", maps[\"P_KK\"], \"viridis\", 0.0, 1.0" in text
    assert "\"|P_KKprime|\", maps[\"P_KKprime_abs\"], \"viridis\", 0.0, 1.0" in text
    assert "\"valley polarization\", maps[\"VP\"], \"coolwarm\", -1.0, 1.0" in text
    assert "\"|IVC|\", maps[\"IVC_abs\"], \"viridis\", 0.0, 0.5 * active_local.n_active" in text


def test_taige_notebook_embeds_projectors_for_charge_response():
    text = SCRIPT.read_text()
    assert "active_basis_frames(active).reshape" in text
    assert "k_theta_from_projectors_with_basis(projectors, theta_nodes, basis_frames)" in text
    assert "SymmetricHFReferences(vp_plus=vp_plus, vp_minus=vp_minus, ivc=ivc" in text


def test_taige_notebook_plots_real_space_charge_density():
    text = SCRIPT.read_text()
    assert "charge_grid_size = 301" in text
    assert "rho_xy = grid_profile.rho_dimless" in text
    assert "charge_density_2d.png" in text
    assert "integrated_charge_2d" in text


def test_taige_notebook_plots_trial_physical_energy_per_cell():
    text = SCRIPT.read_text()
    assert "trial_energy_components = [bundle.backend.energy(P_theta) for P_theta in projectors_flat]" in text
    assert "energy_norm = float(bundle.backend.n_blocks)" in text
    assert "trial_physical_energy_theta.csv" in text
    assert "trial_physical_energy_theta.png" in text
    assert "\"energy_total_per_cell\": trial_energy_total_per_cell" in text


def test_taige_notebook_compares_q0_and_finite_q_ivc_energies():
    text = SCRIPT.read_text()
    assert "ContinuumFiniteQParams" in text
    assert "taige_ivc_minus_q_coord(n_k)" in text
    assert "taige_ivc_minus_half_shift_coord(n_k)" in text
    assert "finite_q_shift_metadata(finite_q, bundle.grid)" in text
    assert "build_continuum_bundle(" in text and "finite_q=finite_q" in text
    assert "\"finite_q_ivc\"" in text
    assert "ivc_q0_vs_finite_q_energy_comparison.csv" in text
    assert "Delta_finite_Q_minus_Q0_per_cell" in text


def test_taige_notebook_exposes_interaction_screening_controls():
    text = SCRIPT.read_text()
    assert "epsilon = 16.7" in text
    assert "gate_distance_nm = 30.0" in text
    assert "smear_length_nm = 0.347" in text
    assert "interaction_strength_scale = 1.0" in text
    assert "epsilon=epsilon" in text
    assert "gate_distance_nm=gate_distance_nm" in text
