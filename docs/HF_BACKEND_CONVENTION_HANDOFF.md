# Hartree–Fock backend convention and benchmark handoff

This note collects the convention fixes and validation requirements established
by the `tMoTe2-nu0.3`/Mosaic comparison. It is intended both as a maintenance
record for Chiral_DW and as the implementation checklist for porting the
general pieces into Vidyut.

## Matched physical convention

The validated comparison uses projected holes with the filled electron valence
sea as the zero of energy. In the hole representation,

\[
\delta\rho_e=\rho_e-\rho_{\rm filled}=-\rho_h,
\]

so the interaction is normal ordered relative to the empty-hole vacuum:
`reference_density="zero"`. No filled-active-window self-energy is subtracted.
The uniform charging Hartree term is omitted independently; the uniform
exchange term remains.

The remaining matched controls are:

- full layer-potential difference
  \(D_{\rm external}=2u_D^{\rm Chiral}\);
- zero Gaussian smearing when reproducing the external calculation;
- strict physical interaction disk
  \(\lvert q+G\rvert<3\lvert g_M\rvert\);
- integer-compatible global filling
  \(N=\operatorname{round}(\nu n_k^2)\);
- both signed intervalley sectors
  \(Q=\pm(\kappa_+-\kappa_-)\).

## Finite-\(Q\) active frames

Chiral_DW uses the symmetric active-frame convention

```text
K:      k - Q/2
Kprime: k + Q/2
```

with the torus condition

\[
2h=Q\pmod {n_k}.
\]

The physical \(Q\) is exact whenever `n_k` is divisible by three. Requiring
literal unfolded `Q/2` would incorrectly exclude odd meshes such as 21. A
different modular solution for `h` only relabels the common active-frame
momentum.

The two Chiral_DW sectors are:

| sector | \(Q/n_k\) | preferred \(h/n_k\) |
|---|---:|---:|
| minus | \((1/3,1/3)\) | \((1/6,2/3)\) |
| plus | \((2/3,2/3)\) | \((5/6,1/3)\) |

On an 18x18 mesh this gives:

| sector | `q_coord` | `half_shift_coord` |
|---|---:|---:|
| minus | `(6, 6)` | `(3, 12)` |
| plus | `(12, 12)` | `(15, 6)` |

Both sectors must be solved and retained. The lower finite-\(Q\) IVC energy is
then compared with \(Q=0\). If a finite-\(Q\) frame wins, its VP+, VP-, and IVC
references must all come from that same signed active frame.

Implementation:

- `continuum/taige.py`: signed sector helpers and modular half-shift selection;
- `continuum/workflow.py`: solves both signed frames and records their energies;
- `continuum/sweep.py`: signed diagnostics and artifact metadata;
- `tests/test_taige_continuum.py`: even/odd commensurate meshes, opposite
  sectors, and whole-frame branch selection.

## Physical interaction disk

The candidate reciprocal box and the physical interaction domain are separate
objects. The retained channel is determined from the combined physical
momentum:

\[
\left|\left(q_1/n_k+G_1\right)b_1+
       \left(q_2/n_k+G_2\right)b_2\right|<q_{\rm cut}.
\]

For the external matched policy, set:

```python
taige_interaction_params(
    q_mesh="full",
    local_field_cutoff=4,          # candidate G box
    momentum_transfer_cutoff_km=3.0,
    smear_length_nm=0.0,
)
```

`momentum_transfer_cutoff_km` is measured in units of `|g_M|` and uses a strict
open inequality. When it is omitted, the historical
`local_field_cutoff*sqrt(3)/2` radial convention remains available.

The mask must be C3 closed, inversion closed, and applied consistently to form
factors, interaction weights, Hartree construction, exchange construction, and
fine-mesh HF-band evaluation.

The general radial-domain implementation entered the history in commits
`3f2ab8e` and `a8c96b4`. The explicit physical-radius control was added by the
finite-Q/gauge benchmark cleanup.

## Active-space gauge and index adapter

The two backends use different reciprocal coordinates, valley order, band
gauge, and matrix-index storage. On the 12x12 and 18x18 benchmarks:

\[
(i,j)_{\rm external}\mapsto(j,-i)+(n_k/2,0)\pmod {n_k},
\]

and the valley blocks are swapped. If `Z_k` is the momentum-dependent band
unitary and `R` is the flavor permutation, the density transport used in the
benchmark is

\[
P_{\rm Chiral}(k_{\rm Chiral})
 = Z_k R P_{\rm external}(k_{\rm external})^T R^T Z_k^\dagger.
\]

The transpose is essential: omitting it produced a spurious self-energy
difference of approximately 1.67 meV.

`ActiveSpaceGaugeAdapter` is backend independent and supports:

- an arbitrary momentum permutation;
- an arbitrary flavor permutation;
- a full non-Abelian unitary at every momentum;
- an optional matrix-index transpose;
- exact inverse transport;
- an NPZ serialization payload.

Vidyut should preserve this general API rather than encode the present
diagonal-phase special case. Required tests are unitary validation, Hermiticity
preservation, round-trip transport, non-Abelian blocks, and transpose-enabled
storage conversion.

## Additional solver bug exposed by the benchmark

The constrained global Aufbau dynamic program for `TPrimeConstraint` previously
stored only a one-dimensional parent table. Later records could overwrite
parents needed to reconstruct an earlier optimum, returning a projector with
the wrong integer trace. The corrected implementation stores the dynamic
program by stage and backtracks through the stage dimension.

The regression requires exact requested trace, projector idempotency, and
T-prime symmetry for several global particle counts.

## Numerical acceptance benchmarks

The following comparisons use zero smearing, the strict `3|g_M|` disk,
empty-hole normal ordering, two active bands per valley, and the complete gauge
adapter.

### 12x12, `theta=3.9 deg`, `epsilon=80`, `D=0`, `nu=4/9`

| quantity | accepted discrepancy |
|---|---:|
| energy | \(7.1\times10^{-9}\) meV/cell |
| bare-Hamiltonian spectrum | \(6.1\times10^{-8}\) meV |
| HF self-energy | \(4.5\times10^{-9}\) meV |
| HF Hamiltonian | \(6.1\times10^{-8}\) meV |
| single Aufbau projector | \(7.3\times10^{-11}\) |
| mapped density round trip | \(1.7\times10^{-9}\) |

### 18x18, same physical controls

| quantity | accepted discrepancy |
|---|---:|
| energy | \(7.1\times10^{-9}\) meV/cell |
| bare Hamiltonian | \(6.1\times10^{-8}\) meV |
| HF self-energy | \(8.7\times10^{-9}\) meV |
| HF Hamiltonian | \(6.1\times10^{-8}\) meV |
| single Aufbau projector | \(1.4\times10^{-10}\) |
| mapped density round trip | \(3.6\times10^{-9}\) |

Both signed finite-\(Q\) sectors pass. Their energy ordering reverses between
12x12 and 18x18, which is a regression requirement for always retaining both
signs rather than assuming they are finite-size duplicates.

Machine-readable benchmark artifacts live in:

```text
/Users/nisargchadha/Documents/Hartree-Fock Solver/benchmarks/results/
    active_space_gauge_map_12x12/
    active_space_gauge_map_18x18/
    matched_backend_diagnostics_12x12/
```

Human-readable reports are:

```text
/Users/nisargchadha/Documents/Hartree-Fock Solver/analysis/
    active_space_gauge_map_12x12.md
    active_space_gauge_map_18x18.md
```

## Required regression suite

Before changing any of these conventions, the implementation must continue to
pass:

1. strict radial `q+G` mask equality to `|q+G|<3|g_M|`;
2. C3 and inversion closure of the retained channel domain;
3. exact `Q_plus`/`Q_minus` torus relations on meshes divisible by three;
4. odd-mesh modular half-shift tests, especially `n_k=21`;
5. both signed-sector workflow retention and energy selection;
6. gauge-adapter unitary validation and exact round trip;
7. fixed-density energy, self-energy, HF-Hamiltonian, and Aufbau equivalence;
8. exact integer trace/idempotency/T-prime symmetry after constrained global
   Aufbau;
9. empty-hole vacuum checks `E_int[P=0]=0` and `H_HF[0]=h0`.

The last two backend comparisons are stronger than comparing converged native
seeds: they isolate the functional from seed, constraint, and local-minimum
selection policies.
