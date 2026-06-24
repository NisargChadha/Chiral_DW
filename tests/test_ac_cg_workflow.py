import json

import numpy as np

from chiral_dw.ac.workflow import run_ac_cg_workflow
from chiral_dw.config import (
    ACResponseWorkflowParams,
    DomainWallParams,
    FirstShellACParams,
    GatedInteractionParams,
    MomentumGridParams,
    ResponseParams,
    SourceInterpolationParams,
)


def test_small_ac_cg_workflow_writes_valid_artifacts(tmp_path):
    params = ACResponseWorkflowParams(
        grid=MomentumGridParams(n_k=3),
        ac=FirstShellACParams(b1=0.05, u1=0.02, n_ll=3),
        response=ResponseParams(n_theta=5),
        source=SourceInterpolationParams(source_scale=0.7),
        interaction=GatedInteractionParams(interaction_shell=1),
        domain_wall=DomainWallParams(radius=4.0, width=1.0, winding=1),
        output_dir=str(tmp_path),
    )

    result = run_ac_cg_workflow(params, write_outputs=True, write_plots=False)

    assert result.projectors.shape == (5, 3, 3, 2, 2)
    assert np.all(np.isfinite(result.response.K))
    assert np.isfinite(result.response.cG)
    assert result.summary.cG_dimension == "dimensionless"
    assert result.manifest is not None
    assert result.manifest.passed

    for name in ["projectors.npz", "K_theta.csv", "charge_profile.csv", "summary.json", "artifact_manifest.json"]:
        assert (tmp_path / name).exists()

    payload = json.loads((tmp_path / "summary.json").read_text())
    assert payload["summary"]["cG_dimension"] == "dimensionless"
    assert payload["normalization"].startswith("K(theta), cG")

    data = np.load(tmp_path / "projectors.npz")
    assert np.isfinite(float(data["cG"]))
    assert data["projectors"].shape == result.projectors.shape
