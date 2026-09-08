# Configuring and Tuning PROPHET

[PROPHET](prophet.md) has a wide configuration surface: about a dozen `edmfx_*`
keys, four that select the cloud and quadrature treatment, and some fifty
closure parameters in ClimaParams. This page says what each of them does, which
ones are to change, and how to tell whether a change did what you meant. For the
formulation itself, see [PROPHET: Overview and Equations](prophet.md) and
[PROPHET: Closures](prophet_closures.md).

The scheme is named `EDMFX` in the code, so every configuration key and every
diagnostic name below carries `edmf` or `up`/`en` (updraft/environment) rather
than `prophet`.

## Turning PROPHET on

The minimum is one key:

```yaml
turbconv: "prognostic_edmfx"
```

but that alone gives you subdomains that do not feed back on the resolved state.
A working configuration also needs the fluxes, the prognostic turbulence
kinetic energy, and an implicit treatment of diffusion:

```yaml
turbconv: "prognostic_edmfx"
prognostic_tke: true
edmfx_sgs_mass_flux: true          # coherent (mass-flux) SGS fluxes
edmfx_sgs_diffusive_flux: true     # diffusive (K-theory) SGS fluxes
edmfx_vertical_diffusion: true     # apply the diffusive tendency to the drafts too
edmfx_nh_pressure: true            # pressure (form) drag on the drafts
edmfx_filter: true                 # state filters on the draft variables
implicit_diffusion: true
approximate_linear_solve_iters: 2
```

Every physics configuration in `config/model_configs/` sets exactly these; the
exceptions are the two diagnostic cases (`prognostic_edmfx_adv_test_column.yml`
and `prognostic_edmfx_simpleplume_column.yml`), which deliberately switch the
grid-mean feedback off. Start from one of the physics configurations,
`prognostic_edmfx_bomex_column.yml` for a single column or
`prognostic_edmfx_aquaplanet.yml` for a sphere, rather than assembling the list
by hand.

`turbconv: "edonly_edmfx"` selects
[`EDOnlyEDMFX`](@ref ClimaAtmos.EDOnlyEDMFX): the eddy-diffusivity and TKE
machinery with no drafts at all. It is useful as a reference for isolating the
mass-flux contribution, and as a lighter boundary-layer scheme.

## What each flag does

| Key                                   | Default           | Effect                                                                                                                                                                  |
|:------------------------------------- |:----------------- |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `turbconv`                            | `~`               | `prognostic_edmfx` for the full scheme, `edonly_edmfx` for eddy diffusivity only.                                                                                       |
| `updraft_number`                      | `1`               | Number of drafts ``M``. Several code paths (draft microphysics species, tracer entrainment, the pressure closure) currently assume 1.                                   |
| `prognostic_tke`                      | `false`           | Carry ``\rho \kappa_{\mathrm{iso}}`` as a prognostic variable. Required for the mixing-length closure to be meaningful.                                                 |
| `edmfx_sgs_mass_flux`                 | `false`           | Apply the coherent mass-flux part of the SGS fluxes to the grid mean.                                                                                                   |
| `edmfx_sgs_diffusive_flux`            | `false`           | Apply the diffusive part of the SGS fluxes to the grid mean, plus TKE transport and dissipation.                                                                        |
| `edmfx_vertical_diffusion`            | `false`           | Also apply the same *specific* diffusive tendency to each draft scalar.                                                                                                 |
| `edmfx_nh_pressure`                   | `false`           | Include the form-drag term in the draft momentum equation. The virtual-mass buoyancy reduction ``(1-\alpha_b)`` is always applied.                                      |
| `edmfx_filter`                        | `false`           | Run the draft state filters after each stage (see [Regularizations](prophet_numerics.md#Regularizations-and-state-filters)).                                            |
| `edmfx_entr_model`                    | `"Generalized"`   | Entrainment closure: `Generalized` ([`InvZEntrainment`](@ref ClimaAtmos.InvZEntrainment)) or `PiGroups` ([`PiGroupsEntrainment`](@ref ClimaAtmos.PiGroupsEntrainment)). |
| `edmfx_detr_model`                    | `"Generalized"`   | Detrainment closure. Only `Generalized` is implemented.                                                                                                                 |
| `edmfx_scale_blending`                | `"SmoothMinimum"` | How the mixing-length scales are blended; `HardMinimum` is the non-smooth alternative.                                                                                  |
| `edmfx_mse_q_tot_upwinding`           | `"first_order"`   | Vertical reconstruction for the draft ``h_s`` and ``q_t``. `none` is central, `third_order` also available.                                                             |
| `edmfx_tracer_upwinding`              | `"first_order"`   | Vertical reconstruction for the remaining draft tracers.                                                                                                                |
| `edmfx_sgsflux_upwinding`             | `"none"`          | Reconstruction of the SGS mass flux applied to the grid mean.                                                                                                           |
| `edmfx_sgs_horizontal_diffusive_flux` | `false`           | Add the horizontal component of the diffusive SGS flux. See [Horizontal Diffusion](prophet_horizontal_diffusion.md).                                                    |
| `edmfx_horizontal_diffusion`          | `false`           | Apply the grid-mean horizontal diffusion tendencies to the drafts. Requires the flag above.                                                                             |
| `cloud_model`                         | `"quadrature"`    | `grid_scale` evaluates cloud fraction at the mean state; `quadrature` integrates over the SGS distribution; `MLCloud` uses a network.                                   |
| `use_sgs_quadrature`                  | `true`            | Integrate microphysical rates over the SGS distribution rather than at the mean state.                                                                                  |
| `sgs_distribution`                    | `"gaussian"`      | Assumed SGS distribution of ``(T, q_t)``: `gaussian`, `lognormal`, or `mean`.                                                                                           |
| `quadrature_order`                    | `3`               | Gauss–Hermite order per dimension, 1–5.                                                                                                                                 |

The generated [Configuration Options](configuration_options.md) table is the
authoritative list; the help strings there come from
`config/default_configs/default_config.yml`.

Two combinations are rejected at model construction:
`edmfx_sgs_horizontal_diffusive_flux` together with a Smagorinsky–Lilly or
anisotropic-minimum-dissipation LES closure (both supply the same horizontal
diffusion), and `edmfx_horizontal_diffusion` without
`edmfx_sgs_horizontal_diffusive_flux`.

## Microphysics coupling

PROPHET carries whichever water species the grid-mean microphysics model
carries. With `microphysics_model: "0M"`, a draft holds only ``q_t^j`` and its
condensate follows from saturation adjustment; with `"1M"`, it also holds
`q_lcl`, `q_icl`, `q_rai`, `q_sno`, and with `"2M"` the number concentrations
`n_lcl` and `n_rai` as well. Each of those is advected, entrained, sedimented,
and given its own microphysical sources inside the draft.

The draft microphysics and sedimentation paths currently support a single draft
and raise an error for `updraft_number > 1` with non-equilibrium microphysics.

Microphysical rates and cloud fraction are integrated over the SGS distribution
described in [Closures](prophet_closures.md#Variances-and-cloud-fraction). The
practical consequences:

  - `sgs_distribution: "mean"` (or `use_sgs_quadrature: false`) collapses the
    quadrature to the mean state. Use it to isolate the effect of the SGS
    integration, not for production.
  - the number of microphysics evaluations grows as the square of
    `quadrature_order`, and 3 already resolves the leading moments to within a
    few percent for typical variances [LopezGomez2022](@cite).
  - `cloud_model: "grid_scale"` bypasses the variance closure entirely, which is
    the right comparison when diagnosing whether a cloud-cover problem is in the
    variance closure or upstream of it.

## Parameters

Closure parameters live in ClimaParams and are overridden through the `toml:`
key:

```yaml
toml: [toml/prognostic_edmfx_1M.toml]
```

The tuned sets in `toml/` are the starting point:
`prognostic_edmfx.toml` (0M), `prognostic_edmfx_1M.toml`,
`prognostic_edmfx_calibrated.toml`, and case-specific variants such as
`prognostic_edmfx_bomex_pigroup.toml` and `prognostic_edmfx_gcmdriven.toml`.
[Closures](prophet_closures.md#Parameters) maps every symbol in the formulation
onto its ClimaParams name.

If you are tuning by hand rather than calibrating, these are the parameters with
the largest effect, roughly in order:

  - `mixing_length_eddy_viscosity_coefficient` (``c_m``) sets the overall
    magnitude of the diffusive transport, and with it boundary-layer depth and
    cloud base.
  - `mixing_length_Ri_crit` (``\mathrm{Ri}_c``) sets where turbulence shuts
    off in stable stratification, and through
    ``c_d = c_m c_b / \mathrm{Ri}_c`` also the dissipation. It controls the
    sharpness of inversions and the stable-boundary-layer depth.
  - `EDMF_interface_entr_efficiency` (``A``) sets cloud-top entrainment
    across unresolved inversions. Calibrate it against *equilibrium* states (a
    day or longer for trade-cumulus cases), because spin-up snapshots reward
    values that fail at equilibrium.
  - `entr_coeff`, `entr_buoy_coeff`, `entr_inv_tau`,
    `detr_buoy_coeff`, `detr_massflux_vertdiv_coeff` set the mass-flux
    profile: how quickly drafts dilute with height and where they terminate.
  - `diagnostic_covariance_coeff` (``c_\sigma / 2``) and
    `cloud_fraction_eps_rel` control cloud fraction at fixed condensate.
  - `EDMF_min_area`, `EDMF_max_area`, `EDMF_max_surface_area` are
    bounds, not tuning knobs; changing them changes what the limiters do rather
    than the physics.

The calibration workflow under `calibration/` fits these against single-column
targets; see its `README.md`.

## Validation cases

The single-column cases in `config/model_configs/` cover the convective regimes
PROPHET is meant to span, and are the fastest way to see the effect of a change.
[Running Single-Column Cases](single_column.md) explains how to run them; the
regime each one probes is tabulated there.

| Regime                | Configuration                                                                                                                  |
|:--------------------- |:------------------------------------------------------------------------------------------------------------------------------ |
| Shallow cumulus       | `prognostic_edmfx_bomex_column.yml`, `prognostic_edmfx_soares_column.yml`                                                      |
| Stratocumulus         | `prognostic_edmfx_dycoms_rf01_column.yml`, `prognostic_edmfx_dycoms_rf02_column.yml`                                           |
| Precipitating cumulus | `prognostic_edmfx_rico_column.yml`, `prognostic_edmfx_rico_column_2M.yml`                                                      |
| Deep convection       | `prognostic_edmfx_trmm_column.yml`                                                                                             |
| Stable boundary layer | `prognostic_edmfx_gabls_column.yml`                                                                                            |
| Idealized plume       | `prognostic_edmfx_simpleplume_column.yml` (no SGS feedback on the grid mean)                                                   |
| Externally driven     | `prognostic_edmfx_gcmdriven_column.yml`, `prognostic_edmfx_armvaranal_column.yml`, `prognostic_edmfx_tv_era5driven_column.yml` |
| Global                | `prognostic_edmfx_aquaplanet.yml`                                                                                              |

`prognostic_edmfx_adv_test_column.yml` is diagnostic rather than physical: it
switches off all momentum tendencies (`advection_test: true`) to test the
advection of draft variables in isolation. The `*_sparse_autodiff` and
`*_dense_autodiff` variants exercise the automatic-differentiation Jacobians
against the hand-written one, and `prognostic_edmfx_bomex_fixtke_column.yml`
runs BOMEX with `prognostic_tke: false`.

A vertical-resolution sweep on DYCOMS and BOMEX is the discriminating test for
the interface closure of
[Closures](prophet_closures.md#Capping-inversions-as-unresolved-interfaces),
since ``N_{e,\mathrm{eff}}^2`` and ``K_e`` act only when the inversion is
unresolved: cloud cover and inversion height should be insensitive to
``\Delta z``.

## Diagnostics

The PROPHET diagnostics are grouped by subdomain. Updraft variables end in
`up`, environment variables in `en`:

| Group              | Variables                                                                                                                |
|:------------------ |:------------------------------------------------------------------------------------------------------------------------ |
| Draft state        | `arup` (area fraction), `rhoaup`, `waup`, `taup`, `thetaaup`, `haup`, `husup`, `hurup`                                   |
| Draft condensate   | `clwup`, `cliup`, `husraup`, `hussnup`, `cdncup`, `ncraup`                                                               |
| Environment        | `aren`, `rhoaen`, `waen`, `taen`, `thetaaen`, `haen`, `husen`, `huren`, `clwen`, `clien`, `husraen`, `hussnen`, `cdncen` |
| Exchange rates     | `entr`, `turbentr`, `detr`                                                                                               |
| Turbulence         | `tke`, `lmix`, `lmixw`, `lmixtke`, `lmixb`, `edt`, `evu`, `kentr`, `bgrad`, `strain`                                     |
| Horizontal closure | `lmixh`, `edth`, `evuh`                                                                                                  |

The mixing-length components (`lmixw`, `lmixtke`, `lmixb` for the wall,
TKE-balance, and buoyancy scales) are what to look at when the blended `lmix` is
not doing what you expect, and `kentr` separates the interfacial-entrainment
diffusivity from the turbulent `edt`. The
[Available Diagnostics](available_diagnostics.md) page lists units and long
names for all of them.

`config/common_configs/diagnostics_column_progedmf_0M.yml` and
`..._1M.yml` are ready-made diagnostic sets for single-column PROPHET runs,
including the per-subdomain microphysics process rates. Prepend one to a case
configuration:

```bash
julia +1.11 --project=.buildkite .buildkite/ci_driver.jl --config_file config/common_configs/diagnostics_column_progedmf_1M.yml --config_file config/model_configs/prognostic_edmfx_bomex_column.yml --job_id bomex
```

Later `--config_file` arguments win on conflicting keys; see
[Creating Custom Configurations](configuration.md).

## Troubleshooting

**The draft never develops.** Check the surface buoyancy flux. The surface mass
source vanishes with the positive part of ``z_i \overline{w'b'}_s``, so it is
identically zero in a stable boundary layer, by construction, and `arup` should
then sit near `EDMF_min_area`, held there by the area-bounding rate. If the
surface buoyancy flux is positive and `arup` still does not grow, look at `entr`
and `detr` in the lowest cells.

**The draft fills the cell.** `arup` saturating at `EDMF_max_area` means the
area limiters are doing the work rather than the closures. The environment
residual is then reconstructed from a small ``\hat{\rho}^0`` and the blend of
[Environment reconstruction](prophet_numerics.md#Environment-reconstruction)
starts substituting grid-mean values, so subdomain sums stop being exact. Treat
this as a closure problem, not a limiter problem.

**Timestep failures.** The physics configurations in `config/model_configs/`
all set
`implicit_diffusion: true` and `approximate_linear_solve_iters: 2`, because the
diffusive coupling between the grid mean and the drafts is stiff at the
timesteps of interest. Check those first, then that `edmfx_filter: true` is
set.

**Stratocumulus shallows over days in a global run.** Too little interfacial
entrainment to balance large-scale subsidence. The entrainment velocity ``w_e``
is powered by the cell-mean TKE at the face, with a self-quenching feedback and
no direct radiative or evaporative pathway, so
`EDMF_interface_entr_efficiency` is the available knob. Raising it will not fix
weak, moisture-dominated inversions, which lie outside the closure's
[validity domain](prophet_closures.md#Capping-inversions-as-unresolved-interfaces).

**Reproducibility.** Changing any PROPHET parameter or flag changes simulation
output, so the reference counter in `reproducibility_tests/ref_counter.jl` has
to be incremented when the change is intentional; see
`reproducibility_tests/README.md`.
