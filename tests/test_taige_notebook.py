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
    assert "\"|IVC|\", maps[\"IVC_abs\"], \"viridis\", 0.0, 0.5 * active.n_active" in text


def test_taige_notebook_embeds_projectors_for_charge_response():
    text = SCRIPT.read_text()
    assert "active_basis_frames(active).reshape" in text
    assert "k_theta_from_projectors_with_basis(projectors, theta_nodes, basis_frames)" in text


def test_taige_notebook_plots_real_space_charge_density():
    text = SCRIPT.read_text()
    assert "charge_grid_size = 301" in text
    assert "rho_xy = grid_profile.rho_dimless" in text
    assert "charge_density_2d.png" in text
    assert "charge_rho_xy_dimless=rho_xy" in text


def test_taige_notebook_plots_trial_physical_energy_per_cell():
    text = SCRIPT.read_text()
    assert "trial_energy_components = [bundle.backend.energy(P_theta) for P_theta in projectors_flat]" in text
    assert "energy_norm = float(bundle.backend.n_blocks)" in text
    assert "trial_physical_energy_theta.csv" in text
    assert "trial_physical_energy_theta.png" in text
    assert "trial_energy_total_per_cell=trial_energy_total_per_cell" in text
