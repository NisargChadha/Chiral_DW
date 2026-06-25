"""Self-contained continuum/Hartree-Fock workflows for Chiral_DW."""

from chiral_dw.continuum.builder import (
    build_active_space,
    build_continuum_bundle,
    build_density_vertices,
)
from chiral_dw.continuum.hf import (
    ContinuumHFBackend,
    compute_hf_diagnostics,
    reference_hamiltonian_diagnostics,
    solve_hf,
)
from chiral_dw.continuum.models import (
    ContinuumActiveSpace,
    ContinuumBundle,
    ContinuumHFDiagnostics,
    ContinuumHFResult,
    ConvexPathDiagnostics,
    DensityVertices,
    MomentumGrid,
    ReferenceHamiltonianDiagnostics,
    SymmetricHFReferences,
)
from chiral_dw.continuum.references import (
    build_symmetric_hf_references,
    convex_weights,
    reference_diagnostics,
    solve_reference_hf,
    symmetric_convex_hamiltonian,
    symmetric_convex_path,
    symmetric_convex_projector,
)
from chiral_dw.continuum.seeds import build_seed, ivc_seed, random_seed, valley_polarized_seed
from chiral_dw.continuum.symmetry import (
    TPrimeConstraint,
    ValleyU1Constraint,
    mesh_inversion_map,
    rotate_valley_u1,
    valley_swap_matrix,
    valley_u1_rotation,
)
from chiral_dw.continuum.workflow import (
    ContinuumSymmetricHFWorkflowResult,
    continuum_theta_nodes,
    run_continuum_symmetric_hf_workflow,
    write_continuum_symmetric_hf_outputs,
)

__all__ = [
    "ContinuumActiveSpace",
    "ContinuumBundle",
    "ContinuumHFBackend",
    "ContinuumHFDiagnostics",
    "ContinuumHFResult",
    "ContinuumSymmetricHFWorkflowResult",
    "ConvexPathDiagnostics",
    "DensityVertices",
    "MomentumGrid",
    "ReferenceHamiltonianDiagnostics",
    "SymmetricHFReferences",
    "TPrimeConstraint",
    "ValleyU1Constraint",
    "build_active_space",
    "build_continuum_bundle",
    "build_density_vertices",
    "build_seed",
    "build_symmetric_hf_references",
    "compute_hf_diagnostics",
    "continuum_theta_nodes",
    "convex_weights",
    "ivc_seed",
    "mesh_inversion_map",
    "random_seed",
    "reference_diagnostics",
    "reference_hamiltonian_diagnostics",
    "rotate_valley_u1",
    "run_continuum_symmetric_hf_workflow",
    "solve_hf",
    "solve_reference_hf",
    "symmetric_convex_hamiltonian",
    "symmetric_convex_path",
    "symmetric_convex_projector",
    "valley_polarized_seed",
    "valley_swap_matrix",
    "valley_u1_rotation",
    "write_continuum_symmetric_hf_outputs",
]
