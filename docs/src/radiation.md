# Radiation

Radiation enters the [governing equations](equations.md) as one term: the
divergence of the net radiative flux in the total-energy equation,

```math
\frac{\partial}{\partial t} (\rho e_{tot}) \supset
  - \nabla \cdot (\rho \boldsymbol{\mathcal{F}}_R) .
```

Everything on this page is about how ``\boldsymbol{\mathcal{F}}_R`` is obtained.
There are three routes: a full correlated-``k`` radiative transfer solve with
[RRTMGP.jl](https://clima.github.io/RRTMGP.jl/stable/), a set of idealized flux
profiles for single-column cases, and the Held–Suarez temperature relaxation,
which replaces radiation altogether in the dry dynamical-core benchmark. The
`rad` configuration key chooses among them.

The radiative transfer itself, and the derivations behind the
correlated-``k`` method, belong to RRTMGP.jl and to
[Pincus2019](@cite). This page states what ClimaAtmos hands to the solver, how
often, and what it does with the answer. For the configuration options, see
[Running with Radiation](radiation_howto.md).

## The radiation callback

Radiative transfer is far too slow to run every timestep, and radiative heating
rates change slowly, so the solve is a callback rather than a tendency. Every
`dt_rad` (6 hours by default), `rrtmgp_solver_callback!`

 1. copies the current column state into the solver's input arrays
    (`update_atmospheric_state!`),
 2. refreshes the insolation (`set_insolation_variables!`) and the surface albedo
    (`set_surface_albedo!`),
 3. calls `RRTMGP.update_fluxes!`, and
 4. copies the net upward flux into `p.radiation.ᶠradiation_flux`, a face field
    [W m⁻²].

Between callbacks, `ᶠradiation_flux` is held fixed. The tendency
`radiation_tendency!` takes its divergence at every stage, so the heating rate a
layer receives is constant over the `dt_rad` interval but the state it acts on
is not. A shorter `dt_rad` resolves the diurnal cycle better and calls the
solver more often; see
[Choosing the cadence](radiation_howto.md#Choosing-the-cadence).

Radiation is always explicit, and the flux divergence is applied with `ᶜdivᵥ`,
so flux convergence heats the layer.

!!! note "TODO: not yet implemented"

    With PROPHET, the same grid-mean heating rate is applied to every draft
    (divided by the draft density). The formulation instead partitions the
    heating rate between subdomains by their cloud fraction; see
    [PROPHET: Overview and Equations](prophet.md#Radiation) for the form and for
    what is missing.

## What RRTMGP is given

RRTMGP works on columns of layers, and it needs the layer state, the boundary
conditions, and the optical properties of whatever absorbs and scatters. The
wrapper in
[RRTMGPInterface.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/radiation/RRTMGPInterface.jl)
builds the solver once, at cache construction, and refreshes its inputs on every
callback.

**Layer state.** Pressure and temperature come from the model's cell centers.
RRTMGP needs them on cell faces too, and reconstructs the face values itself
with the interpolation scheme the solver was built with (`BestFit()`, with
`SameAsInterpolation()` at the bottom face).

**The isothermal boundary layer.** RRTMGP integrates to negligible pressure, but
the model top is at a finite height, 30 km by default and 60 km in the global
production configurations. With `add_isothermal_boundary_layer` (the default),
RRTMGP appends one extra layer above the model top, isothermal and reaching
effectively zero pressure, and fills it internally. Without it, the atmosphere
above the model top is absent from the calculation, and the
top-of-atmosphere fluxes are the fluxes at the model top rather than at the top
of the real atmosphere.

**Spherical geometry.** In a deep atmosphere, the area of a spherical shell
grows with height, so a flux that is uniform per unit area at the surface
carries less energy per unit area aloft. With `deep_atmosphere`, RRTMGP scales
the fluxes by ``(a/r)^2``. This requires the cell heights and the planet radius,
which the wrapper passes only for spherical geometries.

**Boundary conditions.** The surface temperature comes from
`sfc_conditions.T_sfc` (see [Surface Conditions](surface_conditions.md)); the
longwave surface emissivity and the direct and diffuse shortwave albedos are per
band. The albedo is set by the `albedo_model` key and can depend on the solar
zenith angle and the near-surface wind; see
[Ocean Surface Albedo](surface_albedo.md). The incoming solar flux is the
insolation, below. Incoming diffuse shortwave at the top of the atmosphere is
accepted by the constructor but ignored, because RRTMGP.jl does not implement
it.

## Modes

Four modes run RRTMGP, differing in what optical properties they include.

| Mode                                                                                                               | `rad`             | Gases | Clouds | Aerosols |
|:------------------------------------------------------------------------------------------------------------------ |:----------------- |:----- |:------ |:-------- |
| [`GrayRadiation`](@ref ClimaAtmos.RRTMGPInterface.GrayRadiation)                                                   | `gray`            | none  | none   | none     |
| [`ClearSkyRadiation`](@ref ClimaAtmos.RRTMGPInterface.ClearSkyRadiation)                                           | `clearsky`        | yes   | none   | optional |
| [`AllSkyRadiation`](@ref ClimaAtmos.RRTMGPInterface.AllSkyRadiation)                                               | `allsky`          | yes   | yes    | optional |
| [`AllSkyRadiationWithClearSkyDiagnostics`](@ref ClimaAtmos.RRTMGPInterface.AllSkyRadiationWithClearSkyDiagnostics) | `allskywithclear` | yes   | yes    | optional |

**Gray radiation** is a one-band model whose optical depth is a prescribed
function of pressure and latitude, from RRTMGP's
`GrayOpticalThicknessOGorman2008` following [OGorman2008](@cite). It needs no
gas, cloud, or aerosol inputs at all, only temperature and pressure, which makes
it the right choice for dynamical-core work where radiation should be present
but uncomplicated. The `lapse_rate` and `optical_thickness_parameter` arguments
the wrapper passes are validated but never reach the radiative transfer, which
uses RRTMGP's own defaults.

**Clear-sky radiation** is the full correlated-``k`` band model with gases and
no cloud optics. Clouds are absent from the radiation, not from the model: a
clear-sky run of a cloudy atmosphere is a legitimate diagnostic, and the solve
is much faster than all-sky.

**All-sky radiation** adds cloud optics. RRTMGP samples cloud overlap
stochastically (McICA), so all-sky fluxes carry sampling noise and depend on the
random number stream. `radiation_reset_rng_seed` reseeds the generator from the
timestep number before each call, which makes runs and restarts bitwise
reproducible, but it correlates the sampling noise in time. It is off by default
and should stay off for production runs.

**All-sky with clear-sky diagnostics** runs the all-sky solve and additionally
computes what the fluxes would have been with the clouds removed. The difference
is the cloud radiative effect, available through the `*cs` diagnostics
(`rsutcs`, `rlutcs`, and so on). It is the mode to use when cloud radiative
effect is one of the quantities being evaluated.

## Gases

Water vapor and ozone are given as profiles; every other gas is a single global
value by default (`use_global_means_for_well_mixed_gases`), which is what "well
mixed" means here. The full list RRTMGP accepts is `co2`, `n2o`, `co`,
`ch4`, `o2`, `n2`, `ccl4`, `cfc11`, `cfc12`, `cfc22`, `hfc143a`, `hfc125`,
`hfc23`, `hfc32`, `hfc134a`, `cf4`, and `no2`, with fixed values from
ClimaParams.

Ozone and carbon dioxide can instead vary in time, read from files and
interpolated at each radiation call. Ozone defaults to the analytic RCEMIP
profile [Wing2018](@cite) of `idealized_ozone`. See
[Trace Gases](trace_gases.md) for both, and for how to switch them on.

`idealized_h2o` replaces the model's water vapor with a prescribed
relative-humidity profile, ramped up over the first 30 days. This decouples the
radiation from the model's own moisture, which is useful for isolating
circulation responses but is not a physical configuration.

## Clouds

Cloud optics need three things per layer: how much condensate there is, how it
is distributed within the layer, and how large the particles are.

**Water paths and cloud fraction.** The condensate comes from the microphysics
model as the cloud liquid and cloud ice water contents, and the cloud fraction
from the cloud model (see
[PROPHET: Closures](prophet_closures.md#Variances-and-cloud-fraction) and
[Microphysics](microphysics.md)). RRTMGP needs *in-cloud* water paths, so
`update_cloud_properties!` converts the grid-mean contents to in-cloud values by
dividing by the cloud fraction ``f_c``,

```math
\mathrm{LWP} = \frac{\rho \, q_l^{cl} \, \Delta z}{\max(f_c, \epsilon)} ,
```

and likewise for ice. Cloud fraction is passed separately, so the radiation sees
a partially filled layer rather than a uniformly thin cloud. The longwave solve
uses these mean in-cloud paths; the shortwave solve uses them multiplied by the
subgrid inhomogeneity factor ``\chi`` below (`solve_fluxes!`).

**Subgrid inhomogeneity.** Plane-parallel reflectance is concave in optical
depth, so evaluating the solver at the mean in-cloud water path overestimates
it when condensate varies horizontally within the cloud. The
effective-thickness approximation [Cahalan1994](@cite) evaluates the solver at
the geometric mean of the optical depth, which scales the in-cloud water path
by

```math
\chi = \frac{\exp(\mathrm{E}[\ln \tau])}{\mathrm{E}[\tau]} \le 1 .
```

For a Gamma-distributed optical depth with shape ``\nu``, the inverse relative
variance [Barker1996](@cite), ``\chi = \exp(\psi(\nu)) / \nu`` with ``\psi``
the digamma function. With `QuadratureCloud`, ``\nu`` follows from the subgrid
distribution that also sets the cloud fraction. For a Gaussian saturation
excess truncated at ``S > 0`` with ``f_c = \Phi(z)`` and inverse Mills ratio
``r = \phi(z) / \Phi(z)``, the in-cloud condensate has mean ``m = z + r`` and
variance ``v = 1 - z r - r^2``, so its relative variance is ``v / m^2``. It
tends to ``1`` in sparse clouds and to ``0`` in overcast layers. Overcast
stratocumulus, however, has a relative standard deviation of liquid water path
of about 0.4–0.8 [Oreopoulos2005](@cite). An unresolved, independent, unit-mean
inhomogeneity of relative variance ``\sigma_f^2``
(`cloud_condensate_relative_variance_floor`, default 0.25) therefore multiplies
the truncated-Gaussian condensate, giving

```math
\frac{1}{\nu} = \left(1 + \frac{v}{m^2}\right) \left(1 + \sigma_f^2\right) - 1 .
```

``\chi`` then increases monotonically with ``f_c``, from about 0.43 in sparse
clouds through 0.58 at ``f_c = 0.5`` to 0.84 in overcast layers (0.59, 0.73,
and 0.96 for ``\sigma_f^2 = 0``).

Under `PrognosticEDMFX`, the cloud cover combines the truncated-Gaussian cloud
of the environment with binary cloudy updrafts (see
[PROPHET: Closures](prophet_closures.md#Variances-and-cloud-fraction)), and
``\chi`` treats the cell's cloud as the same mixture. The environment
contributes cloudy area ``a_0 f_{c,0}`` with in-cloud mean
``q_{c,0} / f_{c,0}`` and the Gamma shape above; each cloudy updraft ``j``
contributes area ``a_j`` with its own condensate ``q_{c,j}`` and relative
variance ``\sigma_f^2``. ``\chi`` is the larger of two estimates that coincide
for a single component. One is the geometric mean of the mixture, exact for the
effective-thickness approximation because ``\mathrm{E}[\ln \tau]`` and
``\mathrm{E}[\tau]`` are area-weighted sums over the components. The other is
the Gamma factor of the mixture's own mean and variance. Each underestimates
``\chi`` where the other holds. The geometric mean collapses when an updraft
that the binary cover counts as cloudy holds only trace condensate, although
such an updraft is optically clear. The moment-matched Gamma puts a small,
dense updraft core into a heavy tail near zero, whereas the geometric mean
weighs the core by its area, so a dense core beside a thin environment cloud
lowers ``\chi`` below the environment value. Where nearly all of the cover is
trace-condensate updraft area, both estimates are small; those levels hold
almost no water. The
radiation callback evaluates ``\chi`` (`set_sw_cloud_optical_depth_factor!`)
before it passes the cloud properties to RRTMGP. Other cloud models
(`GridScaleCloud`, `MLCloud`) have no in-cloud distribution, and prescribed
ERA5 clouds already come from a different cloud scheme; for all of them
``\chi = 1``. Setting `cloud_inhomogeneity_in_sw_radiation: false` leaves the
interactive clouds unscaled as well (``\chi = 1``), for runs that isolate the
effect of the factor.

``\chi`` applies to all cloud types, not only to the marine boundary-layer
clouds on which the effective-thickness approximation was first derived.
Retrievals find comparable or larger in-cloud variability in broken cumulus,
over land, and in ice and convective clouds [Oreopoulos2005](@cite), and the
truncated-Gaussian part comes from the same subgrid distribution that sets the
cloud fraction in every regime. The default ``\sigma_f^2`` is set from overcast
stratocumulus and is therefore likely too small for ice and convective clouds
at coarse resolution, where the inhomogeneity also grows with the grid spacing.
Ice condensate under non-equilibrium microphysics departs from saturation
equilibrium, so for ice, the truncated-Gaussian shape is a weaker model and the
floor accounts for more of the variability.

The effective-thickness approximation is derived for shortwave reflectance,
and ``\chi`` is applied to the shortwave solve only. Longwave emissivity,
``1 - \exp(-\tau_a)`` in absorption optical depth ``\tau_a``, is linear for
thin clouds and saturates for thick ones, so the shortwave factor would
understate the emissivity of thin cirrus and raise the outgoing longwave
radiation where the cloud top is coldest. For a Gamma distribution, the
emissivity-averaged factor is
``\chi_{\mathrm{LW}} = \nu \ln(1 + \tau_a / \nu) / \tau_a``, which tends to
1 for thin clouds and falls below ``\chi`` only where the cloud is already
nearly black. The longwave therefore uses the mean in-cloud water paths, and
the callback runs the two band solves separately (`solve_fluxes!`), with both
bands sampling the same McICA cloud mask. The COSP simulators sample
homogeneous in-cloud condensate, so their simulated optical depths do not
include ``\chi``.

!!! note "TODO: fold the inhomogeneity into RRTMGP's cloud sampling"

    Scaling the water path corrects the band-mean optics after the fact. The
    consistent treatment is to give the McICA subcolumn generator in RRTMGP
    the in-cloud condensate distribution (the Gamma shape ``\nu`` above), so
    that each subcolumn draws its own condensate and both the shortwave and
    the longwave average over it. That would replace ``\chi`` and the
    band-specific handling in `solve_fluxes!` with one sampled cloud field and
    give the longwave the ``\chi_{\mathrm{LW}}`` behavior above.

**Effective radii.** The liquid effective radius follows the ``1/3``-power law
of [LiuHallett1997](@cite), evaluated with a droplet number concentration
diagnosed from the aerosol loading. That closure, `ml_N_cloud_liquid_droplets`,
is log-linear in the dust, sea-salt, and ammonium-sulfate concentrations and in
the liquid water. The ice effective radius is a constant from the microphysics
parameters.

!!! warning "Known inconsistency"

    `ml_N_cloud_liquid_droplets` was calibrated with a specific humidity [kg
    kg⁻¹] as its liquid-water argument, but `update_cloud_properties!` passes
    the column-integrated liquid water path [kg m⁻²]. The reference value it is
    compared against is the specific-humidity one, so the liquid-water term of
    the closure is evaluated far from where it was fitted. This is recorded in
    the docstring and tracked as a code-side task.

**Where the clouds come from.** Three sources, selected independently of the
mode:

  - [`InteractiveCloudInRadiation`](@ref ClimaAtmos.InteractiveCloudInRadiation)
    (the default): condensate and cloud fraction from the model state, refreshed
    every callback. This is the coupled configuration, in which radiation and
    clouds affect each other. With `QuadratureCloud`, the shortwave in-cloud
    water paths are scaled by the subgrid inhomogeneity factor ``\chi`` above,
    unless `cloud_inhomogeneity_in_sw_radiation: false`.
  - [`PrescribedCloudInRadiation`](@ref ClimaAtmos.PrescribedCloudInRadiation)
    (`prescribe_clouds_in_radiation: true`): cloud liquid, cloud ice, and cloud
    fraction read from an ERA5 monthly climatology, with the year 2010 repeated
    indefinitely. The model's own clouds still form and precipitate; they just
    do not radiate. This breaks the cloud-radiative feedback deliberately, which
    makes it useful for attributing circulation changes.
  - `idealized_clouds: true`: two fixed cloud layers, liquid between 1 and
    1.5 km and ice between 4 and 5 km, prescribed once at construction and never
    updated.

## Aerosols

With `aerosol_radiation: true`, prescribed aerosols contribute to the shortwave
optics. The wrapper passes the column mass densities of each species and, for
the dust and sea-salt bins, fixed radii. The aerosol concentrations themselves
are read from a prescribed dataset through the `prescribed_aerosols` key, which
also feeds the droplet-number closure above and, with two-moment microphysics,
aerosol activation. Enabling `aerosol_radiation` without any aerosol species in
`prescribed_aerosols` raises an error.

## Insolation

The incoming solar flux at the top of the atmosphere is set by the `insolation`
key, which selects one of five models. `set_insolation_variables!` writes the
cosine of the solar zenith angle and the solar irradiance into the solver on
each callback, so the insolation is refreshed on the `dt_rad` cadence like
everything else.

| Model                                                            | `insolation`       | What it gives                                                                                                                                            |
|:---------------------------------------------------------------- |:------------------ |:-------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`IdealizedInsolation`](@ref ClimaAtmos.IdealizedInsolation)     | `idealized`        | Annual-mean insolation as a function of latitude only, with no diurnal cycle [OGorman2008](@cite)                                                        |
| [`TimeVaryingInsolation`](@ref ClimaAtmos.TimeVaryingInsolation) | `timevarying`      | The full orbital calculation at the current date, from [Insolation.jl](https://clima.github.io/Insolation.jl/stable/), with a diurnal and seasonal cycle |
| [`RCEMIPIIInsolation`](@ref ClimaAtmos.RCEMIPIIInsolation)       | `rcemipii`         | Fixed uniform values from the RCEMIP-II protocol [Wing2018](@cite)                                                                                       |
| [`Larcform1Insolation`](@ref ClimaAtmos.Larcform1Insolation)     | `larcform1`        | Perpetual polar night: zero incoming solar flux                                                                                                          |
| [`ExternalTVInsolation`](@ref ClimaAtmos.ExternalTVInsolation)   | `externaldriventv` | Values from a column forcing file, time-varying or constant depending on the file                                                                        |

Removing the diurnal cycle is a large approximation for the boundary layer, but
it lifts the constraint that `dt_rad` be short enough to resolve a day, and it
is the conventional choice for aquaplanet climate runs.
`TimeVaryingInsolation` also accepts an explicit `latitude` and `longitude`,
which pins a single-column run to a real site.

## Idealized radiation for single-column cases

Three modes replace radiative transfer with a prescribed profile tuned to a
specific case. All three are applied at every stage rather than through a
callback, since each is a handful of column integrals, and all three require
moist microphysics.

[`RadiationDYCOMS`](@ref ClimaAtmos.RadiationDYCOMS) (`rad: DYCOMS`) is the
longwave parameterization of [Stevens2005](@cite) for the DYCOMS RF01 and RF02
stratocumulus cases [Ackerman2009](@cite). The net upward flux combines
cloud-top cooling, cloud-base warming, and free-tropospheric warming above the
inversion:

```math
F(z) = F_0 \, e^{-Q(z, \infty)} + F_1 \, e^{-Q(0, z)}
  + \rho_i \, c_{pd} \, D \, \alpha_z
    \left[ \tfrac{1}{4}(z - z_i)^{4/3} + z_i (z - z_i)^{1/3} \right] ,
```

with ``Q(z_1, z_2) = \int \kappa \rho q_l \, dz`` the liquid-water optical path,
``D`` the large-scale divergence, and the last term active only above the
inversion height ``z_i``, taken as the highest level with ``q_t \ge 0.008``
kg kg⁻¹ (or the level closest to ``0.008`` kg kg⁻¹ if the entire column is
drier). With `PrognosticEDMFX`, the same radiative heating rate is applied to
each draft's moist static energy (divided by the draft density). The
implementation differs from the reference in three respects, each documented in
the source: the optical path uses the specific liquid water content rather than
the mixing ratio, the third term uses the dry ``c_{pd}``, and that term is
clipped to zero below ``z_i``, where the reference lets it act. All three match
the original TurbulenceConvection implementation.

[`RadiationISDAC`](@ref ClimaAtmos.RadiationISDAC) (`rad: ISDAC`) is the
two-stream liquid-water-path form used for the ISDAC mixed-phase Arctic
stratocumulus case,

```math
F(z) = F_0 \, e^{-\kappa (\mathrm{LWP}_{z_t} - \mathrm{LWP}_z)}
  + F_1 \, e^{-\kappa \, \mathrm{LWP}_z} ,
```

with ``\mathrm{LWP}_z`` the liquid water path from the surface to ``z`` and
``\mathrm{LWP}_{z_t}`` its value at the domain top. The first term is cloud-top
cooling, the second cloud-base warming.

[`RadiationTRMM_LBA`](@ref ClimaAtmos.RadiationTRMM_LBA) (`rad: TRMM_LBA`) is
the odd one out: it prescribes a heating rate directly, with no flux at all. The
observational profile varies with both height and time of day, and is converted
to an energy tendency with the moist isochoric heat capacity,
``\rho \, c_{vm} \, \partial T / \partial t``. It is used for the TRMM-LBA
deep-convection case [Grabowski2006](@cite).

## Held–Suarez forcing

`rad: held_suarez` performs no radiative transfer. It relaxes temperature toward
a prescribed radiative-equilibrium profile and applies Rayleigh friction to the
low-level winds, the standard dry dynamical-core benchmark of
[HeldSuarez1994](@cite):

```math
\frac{\partial T}{\partial t} \supset -k_T(\phi, \sigma) \, (T - T_{eq}(\phi, p)) ,
\qquad
\frac{\partial \boldsymbol{u}_h}{\partial t} \supset -k_v(\sigma) \, \boldsymbol{u}_h .
```

It occupies the `radiation_mode` slot because it replaces radiation, not because
it is radiation. Two consequences follow: it is applied from
`remaining_tendency!` at every stage rather than from a callback, so `dt_rad` is
ignored; and none of the radiation diagnostics are available. The
equator-to-pole contrast and equatorial equilibrium temperature take different
values for dry and moist microphysics (`ΔT_y_dry`/`ΔT_y_wet` and
`T_equator_dry`/`T_equator_wet`). The relaxation rates, the equilibrium
temperature profile, and the friction ramp are given in
[Forcings and Idealized Cases](forcings.md).

## Diagnostics

The flux diagnostics follow the CMIP short-name convention: `r` for radiation,
then `s` or `l` for shortwave or longwave, then `d` or `u` for downwelling or
upwelling. A bare name is the three-dimensional field, a trailing `t` is the
top-of-atmosphere value, and a trailing `s` is the surface value. A `cs` suffix
marks the clear-sky counterpart, available only in the `allskywithclear` mode.

|                               | Downwelling           | Upwelling                   |
|:----------------------------- |:--------------------- |:--------------------------- |
| Shortwave, 3D / TOA / surface | `rsd`, `rsdt`, `rsds` | `rsu`, `rsut`, `rsus`       |
| Longwave, 3D / TOA / surface  | `rld`, `rlds`         | `rlu`, `rlut`, `rlus`       |
| Clear-sky shortwave           | `rsdcs`, `rsdscs`     | `rsucs`, `rsutcs`, `rsuscs` |
| Clear-sky longwave            | `rldcs`, `rldscs`     | `rlucs`, `rlutcs`           |

Alongside the fluxes, `reffclw` and `reffcli` report the cloud liquid and ice
effective radii the radiation actually used, `od550aer` and `odsc550aer` the
aerosol optical depths, and `clt` and `cltl` the total cloud cover as seen by
the shortwave and longwave McICA sampling respectively. Those last two are not
the model's own cloud fraction: they are what the stochastic overlap sampling
produced, and the shortwave and longwave values need not agree. Requesting any
of these in a mode that does not compute them raises an error naming the
variable and the mode. See
[Available Diagnostics](available_diagnostics.md).

## Where this is implemented

| Component                                       | Source                                                                                                                                                                                                               |
|:----------------------------------------------- |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mode types, solver construction, input plumbing | [radiation/RRTMGPInterface.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/radiation/RRTMGPInterface.jl)                                                                           |
| Per-callback input updates                      | [radiation/update_inputs.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/radiation/update_inputs.jl)                                                                               |
| Cache, tendencies, idealized profiles, ozone    | [radiation/radiation.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/radiation/radiation.jl)                                                                                       |
| Held–Suarez forcing                             | [radiation/held_suarez.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/radiation/held_suarez.jl)                                                                                   |
| The callback, insolation, and albedo updates    | [callbacks/callbacks.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/callbacks/callbacks.jl), [callbacks/get_callbacks.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/callbacks/get_callbacks.jl) |
| Configuration dispatch                          | [config/model_getters.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/config/model_getters.jl)                                                                                                              |
| Diagnostics                                     | [diagnostics/radiation_diagnostics.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/diagnostics/radiation_diagnostics.jl)                                                                                    |
