from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from chiral_dw.config import ContinuumGridParams, ContinuumHFParams
from chiral_dw.continuum import (
    ValleyU1Constraint,
    active_basis_frames,
    build_continuum_bundle,
    build_seed,
    compare_hysteresis_records,
    is_clean_hysteresis_record,
    load_taige_backend_cache,
    save_taige_backend_cache,
    select_lowest_energy_clean_record,
    select_lowest_energy_raw_record,
    select_lowest_energy_record,
    solve_reference_hf,
    taige_backend_cache_path,
    taige_backend_cache_signature,
    taige_interaction_params,
    taige_model_params,
    transport_projector_between_frames,
)


ROOT = Path(__file__).resolve().parents[1]
PRECOMPUTE_SCRIPT = ROOT / "scripts" / "precompute_taige_backend_cache.py"
BRANCH_SCRIPT = ROOT / "scripts" / "scan_taige_ivc_hysteresis_linecut.py"
LEGACY_BRANCH_SCRIPT = ROOT / "scripts" / "scan_taige_ivc_hysteresis_by_theta.py"
MERGE_SCRIPT = ROOT / "scripts" / "merge_taige_ivc_hysteresis_sweep.py"
FINITE_MERGE_SCRIPT = ROOT / "scripts" / "merge_taige_ivc_hysteresis_finite_size.py"
CACHE_JOB = ROOT / "jobs" / "precompute_taige_backend_cache_array.sh"
BRANCH_JOB = ROOT / "jobs" / "scan_taige_ivc_hysteresis_by_theta.sh"
ALL_BRANCH_JOB = ROOT / "jobs" / "scan_taige_ivc_hysteresis_all_linecuts_array.sh"
MERGE_JOB = ROOT / "jobs" / "merge_taige_ivc_hysteresis_sweep.sh"
SUBMIT_JOB = ROOT / "jobs" / "submit_taige_ivc_hysteresis_full_pipeline.sh"
FINITE_MERGE_JOB = ROOT / "jobs" / "merge_taige_ivc_hysteresis_finite_size.sh"
FINITE_SUBMIT_JOB = ROOT / "jobs" / "submit_taige_ivc_hysteresis_finite_size_pipeline.sh"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _tiny_interaction():
    return taige_interaction_params(
        q_mesh="shell",
        q_shell=0,
        local_field_cutoff=0,
        interaction_strength_scale=0.0,
        exchange_scale=0.0,
        hartree_scale=0.0,
        density_vertex_retention="hartree_only",
        density_vertex_layout="auto",
        exchange_representation="auto",
        form_factor_backend="auto",
    )


def _tiny_bundle(*, u_D: float = 0.0):
    return build_continuum_bundle(
        model=taige_model_params(
            theta_deg=3.5,
            u_D=u_D,
            plane_wave_shell=0,
            n_bands=1,
            n_active_bands_per_valley=1,
        ),
        grid=ContinuumGridParams(n_k=2),
        interaction=_tiny_interaction(),
    )


def _tiny_cli_args(output_root: Path) -> list[str]:
    return [
        "--output-root",
        str(output_root),
        "--u-d-min",
        "0",
        "--u-d-max",
        "0.25",
        "--n-u-d",
        "2",
        "--theta-min-deg",
        "3.5",
        "--theta-max-deg",
        "3.5",
        "--n-twist",
        "1",
        "--n-k",
        "2",
        "--plane-wave-shell",
        "0",
        "--n-bands",
        "1",
        "--n-active-bands-per-valley",
        "1",
        "--q-mesh",
        "shell",
        "--q-shell",
        "0",
        "--local-field-cutoff",
        "0",
        "--v0",
        "0.0",
        "--exchange-scale",
        "0.0",
        "--hartree-scale",
        "0.0",
        "--max-iter",
        "1",
        "--min-iter",
        "0",
        "--mixing-method",
        "linear",
    ]


def test_lowest_energy_selection_prefers_clean_converged_then_fallbacks():
    rows = [
        {"run_id": "warn_low", "energy_total_per_cell": -10.0, "converged": True, "warning_flag": True},
        {"run_id": "dirty_clean", "energy_total_per_cell": -5.0, "converged": False, "warning_flag": False},
        {"run_id": "clean_high", "energy_total_per_cell": -2.0, "converged": True, "warning_flag": False},
        {"run_id": "clean_low", "energy_total_per_cell": -3.0, "converged": True, "warning_flag": False},
    ]
    selected, pool = select_lowest_energy_record(rows)
    assert selected["run_id"] == "clean_low"
    assert pool == "clean"
    assert select_lowest_energy_clean_record(rows)["run_id"] == "clean_low"
    assert select_lowest_energy_raw_record(rows)["run_id"] == "warn_low"
    assert is_clean_hysteresis_record(rows[2]) is True

    selected, pool = select_lowest_energy_record(rows[:2])
    assert selected["run_id"] == "warn_low"
    assert pool == "all_unclean_raw_fallback"
    assert select_lowest_energy_clean_record(rows[:2]) is None
    assert is_clean_hysteresis_record(rows[1]) is False

    selected, pool = select_lowest_energy_record([rows[0]])
    assert selected["run_id"] == "warn_low"
    assert pool == "all_unclean_raw_fallback"

    with pytest.raises(ValueError, match="empty"):
        select_lowest_energy_record([])


def test_taige_active_frame_transport_preserves_projector_on_neighboring_displacements():
    left = _tiny_bundle(u_D=0.0)
    right = _tiny_bundle(u_D=0.25)
    P = build_seed("ivc", left.active, n_occ_per_k=1)

    transported, diagnostics = transport_projector_between_frames(
        P,
        active_basis_frames(left.active),
        active_basis_frames(right.active),
        n_occ_per_k=1,
    )

    traces = np.trace(transported, axis1=-2, axis2=-1).real
    assert np.allclose(traces, 1.0)
    assert diagnostics.transported_trace_error < 1e-12
    assert diagnostics.transported_idempotency_error_fro < 1e-12
    assert diagnostics.mean_retained_weight > 0.99
    assert diagnostics.min_retained_weight > 0.99


def test_hysteresis_merge_compares_projectors_with_active_frames_at_same_point():
    bundle = _tiny_bundle(u_D=0.0)
    P = build_seed("ivc", bundle.active, n_occ_per_k=1)
    frames = active_basis_frames(bundle.active)
    up = {
        "u_index": 0,
        "theta_index": 0,
        "u_D_meV": 0.0,
        "theta_deg": 3.5,
        "energy_total_per_cell": -2.0,
        "direct_gap_min": 0.2,
        "ivc_amplitude_block": 0.4,
        "warning_flag": False,
        "cG": 1.25,
        "run_id": "up",
        "projector_path": "up.npz",
    }
    down = {
        **up,
        "energy_total_per_cell": -1.0,
        "direct_gap_min": 0.5,
        "ivc_amplitude_block": 0.1,
        "warning_flag": True,
        "cG": 2.5,
        "run_id": "down",
        "projector_path": "down.npz",
    }

    comparison, overlap = compare_hysteresis_records(
        up=up,
        down=down,
        up_projector=P,
        down_projector=P,
        up_frames=frames,
        down_frames=frames,
        n_occ_per_k=1,
    )

    assert overlap.mean_overlap == pytest.approx(1.0)
    assert comparison.selected_lower_energy_branch == "up"
    assert comparison.high_gap_branch == "down"
    assert comparison.low_gap_branch == "up"
    assert comparison.cG_low_gap == pytest.approx(1.25)
    assert comparison.cG_high_gap == pytest.approx(2.5)
    assert comparison.warning_count == 1


def test_backend_cache_save_load_restores_usable_backend_and_vp_references(tmp_path: Path):
    bundle = _tiny_bundle(u_D=0.0)
    controls = ContinuumHFParams(max_iter=1, min_iter=0, mixing_method="linear")
    constraint = ValleyU1Constraint(bundle.active)
    vp_plus = solve_reference_hf(bundle, "vp_plus", controls, constraint=constraint)
    vp_minus = solve_reference_hf(bundle, "vp_minus", controls, constraint=constraint)
    signature = taige_backend_cache_signature(
        model=bundle.params,
        grid=ContinuumGridParams(n_k=2),
        interaction=bundle.interaction,
    )
    cache_path = taige_backend_cache_path(tmp_path, signature)

    manifest = save_taige_backend_cache(
        cache_path,
        bundle=bundle,
        signature=signature,
        vp_plus=vp_plus,
        vp_minus=vp_minus,
    )
    loaded = load_taige_backend_cache(cache_path)

    assert manifest.has_vp_references is True
    assert loaded.has_vp_references is True
    assert loaded.manifest.cache_hash == manifest.cache_hash
    assert loaded.bundle.backend.exchange_representation == bundle.backend.exchange_representation
    assert loaded.bundle.backend.vertex_layout == bundle.backend.vertex_layout
    assert np.allclose(loaded.bundle.active.h0, bundle.active.h0)

    P = build_seed("ivc", bundle.active, n_occ_per_k=1)
    direct_energy = bundle.backend.energy(P)
    loaded_energy = loaded.bundle.backend.energy(P)
    assert loaded_energy.total == pytest.approx(direct_energy.total)
    assert loaded_energy.one_body == pytest.approx(direct_energy.one_body)
    assert loaded_energy.hartree == pytest.approx(direct_energy.hartree)
    assert loaded_energy.fock == pytest.approx(direct_energy.fock)

    direct_H = bundle.backend.hf_hamiltonian(P)
    loaded_H = loaded.bundle.backend.hf_hamiltonian(P)
    assert np.allclose(loaded_H, direct_H)
    _P_direct, _evals_direct, direct_gap, direct_indirect = bundle.backend.update_density_per_k(direct_H, 1)
    _P_loaded, _evals_loaded, loaded_gap, loaded_indirect = loaded.bundle.backend.update_density_per_k(loaded_H, 1)
    assert loaded_gap == pytest.approx(direct_gap)
    assert loaded_indirect == pytest.approx(direct_indirect)
    assert np.allclose(loaded.vp_plus.P, vp_plus.P)
    assert np.allclose(loaded.vp_minus.H_hf, vp_minus.H_hf)
    assert manifest.vp_hf_chern_rows
    assert loaded.vp_hf_chern_rows
    assert "chern_hf_vpplus_band_0" in loaded.vp_hf_chern_columns


def test_hysteresis_scripts_smoke_resume_and_merge(tmp_path: Path):
    output_root = tmp_path / "hysteresis"
    precompute_cmd = [
        sys.executable,
        str(PRECOMPUTE_SCRIPT),
        *_tiny_cli_args(output_root),
        "--theta-min-deg",
        "3.5",
        "--theta-max-deg",
        "3.55",
        "--n-twist",
        "2",
        "--seed-ordered-weight",
        "1.0",
        "--seed-random-weight",
        "0.0",
    ]
    subprocess.run(precompute_cmd, check=True, timeout=120)

    branch_base = [
        sys.executable,
        str(BRANCH_SCRIPT),
        *_tiny_cli_args(output_root),
        "--theta-min-deg",
        "3.5",
        "--theta-max-deg",
        "3.55",
        "--n-twist",
        "2",
        "--random-seeds",
        "7",
        "--no-include-ordered-seed",
        "--n-theta",
        "3",
        "--require-cache",
    ]
    for task_id in range(8):
        subprocess.run(
            [*branch_base, "--sweep-axis", "both", "--task-id", str(task_id)],
            check=True,
            timeout=120,
        )
    resumed = subprocess.run(
        [*branch_base, "--sweep-axis", "both", "--task-id", "0"],
        check=True,
        timeout=120,
        text=True,
        capture_output=True,
    )
    assert "Resumed checkpoint" in resumed.stdout

    subprocess.run(
        [sys.executable, str(MERGE_SCRIPT), "--output-root", str(output_root)],
        check=True,
        timeout=120,
    )

    branch_rows = _read_csv(output_root / "hysteresis_sweep.csv")
    comparison_rows = _read_csv(output_root / "hysteresis_comparison.csv")
    displacement_rows = _read_csv(output_root / "hysteresis_displacement_comparison.csv")
    twist_rows = _read_csv(output_root / "hysteresis_twist_comparison.csv")
    candidate_rows = _read_csv(output_root / "hysteresis_all_branch_candidates.csv")
    assert len(branch_rows) == 16
    assert len(comparison_rows) == 4
    assert len(displacement_rows) == 4
    assert len(twist_rows) == 4
    assert len(candidate_rows) == 16
    assert {"up", "down"} == {row["direction"] for row in branch_rows}
    assert {"u_D", "theta"} == {row["sweep_axis"] for row in branch_rows}
    assert {
        "cG",
        "cG_diagnostic",
        "cG_warning_flag",
        "texture_valid",
        "projector_path",
        "hit_max_iter",
        "delta_P",
        "delta_energy",
        "commutator_norm",
        "constraint_error",
        "trace_error",
        "clean_branch",
        "trial_theta_csv",
        "vp_plus_energy_per_cell",
        "vp_minus_energy_per_cell",
        "vp_plus_clean",
        "vp_minus_clean",
        "chern_hf_vpplus_band_0",
    } <= set(branch_rows[0])
    assert {
        "lowest_energy_raw_branch",
        "lowest_energy_clean_branch",
        "high_gap_branch",
        "low_gap_branch",
        "lowest_energy_raw_cG",
        "lowest_energy_clean_cG",
        "row_reliability",
    } <= set(comparison_rows[0])
    assert {"energy_up_minus_down", "mean_projector_overlap"} <= set(displacement_rows[0])
    assert {"energy_up_minus_down", "mean_projector_overlap"} <= set(twist_rows[0])
    for row in branch_rows:
        assert Path(row["projector_path"]).exists()
        assert Path(row["point_record_path"]).exists()
    for name in (
        "hysteresis_energy_crossing.csv",
        "hysteresis_gap_jump.csv",
        "hysteresis_overlap_discontinuity.csv",
        "hysteresis_selected_branch_cg.csv",
        "hysteresis_gap_families.csv",
        "hysteresis_trial_theta.csv",
        "hysteresis_selected_trial_theta.csv",
        "hysteresis_vp_chern_numbers.csv",
    ):
        assert (output_root / name).exists()
    assert len(_read_csv(output_root / "hysteresis_trial_theta.csv")) == 16 * 3
    assert len(_read_csv(output_root / "hysteresis_selected_trial_theta.csv")) == 4 * 3
    assert _read_csv(output_root / "hysteresis_vp_chern_numbers.csv")


def test_hysteresis_finite_size_merge_writes_clean_fit_tables(tmp_path: Path):
    output_root = tmp_path / "fs_hysteresis"
    for n_k, cG in [(18, 1.2), (20, 1.1), (22, 1.05), (24, 1.0)]:
        mesh = output_root / f"nk_{n_k:03d}"
        mesh.mkdir(parents=True)
        comparison = {
            "theta_index": 0,
            "u_index": 0,
            "theta_deg": 3.5,
            "u_D_meV": 10.0,
            "lowest_energy_clean_branch": "u_D_up",
            "lowest_energy_clean_cG": cG,
            "lowest_energy_raw_branch": "u_D_up",
            "lowest_energy_raw_cG": cG,
            "lowest_energy_raw_clean": True,
            "row_reliability": "clean",
        }
        with (mesh / "hysteresis_comparison.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(comparison))
            writer.writeheader()
            writer.writerow(comparison)
        candidates = [
            {
                "theta_index": 0,
                "u_index": 0,
                "theta_deg": 3.5,
                "u_D_meV": 10.0,
                "branch_id": "u_D_up",
                "gap_family_label": "small_gap",
                "energy_total_per_cell": -1.0,
                "cG": cG,
                "clean_branch": True,
            },
            {
                "theta_index": 0,
                "u_index": 0,
                "theta_deg": 3.5,
                "u_D_meV": 10.0,
                "branch_id": "theta_down",
                "gap_family_label": "large_gap",
                "energy_total_per_cell": -0.8,
                "cG": cG + 0.5,
                "clean_branch": True,
            },
        ]
        with (mesh / "hysteresis_all_branch_candidates.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(candidates[0]))
            writer.writeheader()
            writer.writerows(candidates)
        (mesh / "hysteresis_selected_trial_theta.csv").write_text(
            "theta_index,u_index,theta_deg,u_D_meV,theta,energy_total_per_cell,cG\n"
            f"0,0,3.5,10.0,0.1,-1.0,{cG}\n"
        )
        (mesh / "hysteresis_vp_chern_numbers.csv").write_text(
            "theta_index,u_index,theta_deg,u_D_meV,reference,band,chern\n"
            "0,0,3.5,10.0,VP+,0,1.0\n"
        )

    subprocess.run(
        [
            sys.executable,
            str(FINITE_MERGE_SCRIPT),
            "--output-root",
            str(output_root),
            "--n-k-list",
            "18,20,22,24",
        ],
        check=True,
    )
    fits = _read_csv(output_root / "hysteresis_finite_size_cg_fits.csv")
    selected = [row for row in fits if row["branch_label"] == "lowest_energy_clean"]
    assert selected
    assert selected[0]["fit_status"] == "fit_ok"
    assert selected[0]["n_clean_finite"] == "4"
    assert (output_root / "hysteresis_finite_size_selected_trial_theta.csv").exists()
    assert (output_root / "hysteresis_finite_size_vp_chern_boundary.csv").exists()


def test_hysteresis_dry_runs_plan_default_cluster_task_counts(tmp_path: Path):
    cache_root = tmp_path / "cache_dry"
    branch_root = tmp_path / "branch_dry"
    twist_root = tmp_path / "twist_dry"
    both_root = tmp_path / "both_dry"
    subprocess.run(
        [sys.executable, str(PRECOMPUTE_SCRIPT), "--output-root", str(cache_root), "--dry-run"],
        check=True,
        timeout=60,
    )
    subprocess.run(
        [sys.executable, str(BRANCH_SCRIPT), "--output-root", str(branch_root), "--dry-run"],
        check=True,
        timeout=60,
    )
    subprocess.run(
        [
            sys.executable,
            str(BRANCH_SCRIPT),
            "--output-root",
            str(twist_root),
            "--sweep-axis",
            "theta",
            "--dry-run",
        ],
        check=True,
        timeout=60,
    )
    subprocess.run(
        [
            sys.executable,
            str(BRANCH_SCRIPT),
            "--output-root",
            str(both_root),
            "--sweep-axis",
            "both",
            "--dry-run",
        ],
        check=True,
        timeout=60,
    )

    cache_plan = json.loads((cache_root / "backend_cache_plan.json").read_text())
    branch_plan = json.loads((branch_root / "hysteresis_branch_plan.json").read_text())
    twist_plan = json.loads((twist_root / "hysteresis_branch_plan.json").read_text())
    both_plan = json.loads((both_root / "hysteresis_branch_plan.json").read_text())
    assert cache_plan["n_tasks"] == 21 * 21
    assert cache_plan["n_selected"] == 21 * 21
    assert branch_plan["n_tasks"] == 21 * 2
    assert twist_plan["n_tasks"] == 21 * 2
    assert both_plan["n_tasks"] == 2 * (21 + 21)
    assert {row["direction"] for row in branch_plan["rows"]} == {"up", "down"}
    assert {row["sweep_axis"] for row in branch_plan["rows"]} == {"u_D"}
    assert {row["sweep_axis"] for row in twist_plan["rows"]} == {"theta"}
    assert {row["sweep_axis"] for row in both_plan["rows"]} == {"u_D", "theta"}
    assert branch_plan["rows"][0]["first_u_D_meV"] == pytest.approx(0.0)
    assert branch_plan["rows"][1]["first_u_D_meV"] == pytest.approx(20.0)
    assert twist_plan["rows"][0]["first_theta_deg"] == pytest.approx(2.0)
    assert twist_plan["rows"][1]["first_theta_deg"] == pytest.approx(4.0)


def test_hysteresis_slurm_wrappers_pass_cluster_defaults():
    subprocess.run(
        [
            "bash",
            "-n",
            str(CACHE_JOB),
            str(BRANCH_JOB),
            str(ALL_BRANCH_JOB),
            str(MERGE_JOB),
            str(SUBMIT_JOB),
            str(FINITE_MERGE_JOB),
            str(FINITE_SUBMIT_JOB),
        ],
        check=True,
    )
    cache_text = CACHE_JOB.read_text()
    branch_text = BRANCH_JOB.read_text()
    all_branch_text = ALL_BRANCH_JOB.read_text()
    merge_text = MERGE_JOB.read_text()
    submit_text = SUBMIT_JOB.read_text()
    finite_merge_text = FINITE_MERGE_JOB.read_text()
    finite_submit_text = FINITE_SUBMIT_JOB.read_text()

    assert "#SBATCH -p serial_requeue" in cache_text
    assert "#SBATCH --array=0-440" in cache_text
    assert "#SBATCH -c 4" in cache_text
    assert "#SBATCH --array=0-41" in branch_text
    assert "#SBATCH --array=0-83" in all_branch_text
    assert 'N_TWIST=${N_TWIST:-"21"}' in cache_text
    assert 'THETA_MAX_DEG=${THETA_MAX_DEG:-"4.0"}' in branch_text
    assert 'N_K=${N_K:-"24"}' in branch_text
    assert 'PLANE_WAVE_SHELL=${PLANE_WAVE_SHELL:-"5"}' in branch_text
    assert 'N_ACTIVE_BANDS_PER_VALLEY=${N_ACTIVE_BANDS_PER_VALLEY:-"2"}' in branch_text
    assert 'Q_MESH=${Q_MESH:-"full"}' in branch_text
    assert 'VERTEX_WORKERS=${VERTEX_WORKERS:-"${SLURM_CPUS_PER_TASK:-1}"}' in cache_text
    assert 'EXCHANGE_WORKERS=${EXCHANGE_WORKERS:-"${SLURM_CPUS_PER_TASK:-1}"}' in branch_text
    assert 'DENSITY_VERTEX_RETENTION=${DENSITY_VERTEX_RETENTION:-"hartree_only"}' in branch_text
    assert 'DENSITY_VERTEX_LAYOUT=${DENSITY_VERTEX_LAYOUT:-"auto"}' in branch_text
    assert 'EXCHANGE_REPRESENTATION=${EXCHANGE_REPRESENTATION:-"auto"}' in branch_text
    assert 'FORM_FACTOR_BACKEND=${FORM_FACTOR_BACKEND:-"auto"}' in branch_text
    assert 'MAX_ITER=${MAX_ITER:-"800"}' in branch_text
    assert 'MAX_ITER=${MAX_ITER:-"800"}' in all_branch_text
    assert 'MAX_ITER=${MAX_ITER:-"800"}' in cache_text
    assert 'N_THETA=${N_THETA:-"81"}' in all_branch_text
    assert 'RANDOM_SEEDS=${RANDOM_SEEDS:-"1,7,13,29,53"}' in branch_text
    assert 'REQUIRE_CACHE=${REQUIRE_CACHE:-"1"}' in branch_text
    assert 'TOTAL_TASKS=$((2 * (N_TWIST + N_U_D)))' in all_branch_text
    assert "scan_taige_ivc_hysteresis_all_linecuts_array.sh" in submit_text
    assert "--dependency=afterok:" in submit_text
    assert "precompute_taige_backend_cache_array.sh" in submit_text
    assert "merge_taige_ivc_hysteresis_sweep.sh" in submit_text
    assert "SLURM_ARRAY_TASK_ID" in cache_text
    assert "--task-id \"$TASK_ID\"" in branch_text
    assert "--sweep-axis both" in all_branch_text
    assert "--sweep-axis u_D" in branch_text
    assert "--density-vertex-retention \"$DENSITY_VERTEX_RETENTION\"" in cache_text
    assert "--density-vertex-layout \"$DENSITY_VERTEX_LAYOUT\"" in branch_text
    assert "--exchange-representation \"$EXCHANGE_REPRESENTATION\"" in branch_text
    assert "--form-factor-backend \"$FORM_FACTOR_BACKEND\"" in branch_text
    for text in (cache_text, branch_text, all_branch_text, merge_text):
        assert "export OMP_NUM_THREADS=1" in text
        assert "export MKL_NUM_THREADS=1" in text
        assert "export OPENBLAS_NUM_THREADS=1" in text
        assert "export NUMEXPR_NUM_THREADS=1" in text
    assert "merge_taige_ivc_hysteresis_finite_size.py" in finite_merge_text
    assert 'N_K_LIST=${N_K_LIST:-"18,20,22,24"}' in finite_submit_text
    assert 'NK_MEMORY_GB_MAP=${NK_MEMORY_GB_MAP:-"18:12,20:16,22:20,24:24"}' in finite_submit_text
    assert "jobs/precompute_taige_backend_cache_array.sh" in finite_submit_text
    assert "jobs/scan_taige_ivc_hysteresis_all_linecuts_array.sh" in finite_submit_text
    assert "jobs/merge_taige_ivc_hysteresis_finite_size.sh" in finite_submit_text
    assert "--export=ALL" in finite_submit_text

    dry = subprocess.run(
        [str(FINITE_SUBMIT_JOB)],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "N_U_D": "21",
            "N_TWIST": "21",
            "N_K_LIST": "18,20,22,24",
        },
    ).stdout
    assert "Dry run task counts: cache=1764 scan=336" in dry
    assert "--mem=12G" in dry
    assert "--mem=16G" in dry
    assert "--mem=20G" in dry
    assert "--mem=24G" in dry
