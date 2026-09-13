# Forcings and Idealized Cases

A single column or a small box does not simulate its own large-scale
environment. The subsidence that caps a stratocumulus deck, the horizontal
advection that dries a trade-wind boundary layer, and the geostrophic wind that
the surface layer turns against all originate outside the domain. Forcings
supply those tendencies as prescribed terms, either from an analytic profile
that defines an idealized case or from a data file.

All the forcings described here are applied explicitly, from
`additional_tendency!`, and none has a configuration key of its own. A forcing
comes from the setup that `initial_condition` names, so choosing a case chooses
its forcings;
the two exceptions are Held–Suarez, selected through the `rad` key, and the
monthly-averaged reanalysis case, selected through `external_forcing`. See
[Setups](setups.md) for the hooks a setup implements and
[Running Single-Column Cases](single_column.md) for how to run the cases.

## Large-scale subsidence

Subsidence advects scalars vertically with a prescribed velocity ``w_{sub}(z)``,
negative for descent. The tendency is written in advective rather than flux
form,

```math
\left. \frac{\partial \chi}{\partial t} \right|_{sub} = -w_{sub} \frac{\partial \chi}{\partial z} ,
```

and discretized as the difference between a flux divergence and ``\chi`` times
the divergence of the velocity, with first-order upwinding of ``\chi``. Writing
it this way makes the scheme consistent for a uniform tracer: both terms cancel
level by level, so ``\chi \equiv 1`` produces no tendency.

The vertical operator carries a zero-flux condition at the top and bottom faces.
For the usual case of inflow through the lid, the two terms cancel there as
well, which is equivalent to holding ``\chi`` at its top-level value in the air
descending into the domain.

What subsides depends on which of the two implementations is active. The
analytic subsidence of the idealized cases applies to the total enthalpy
``h_{tot}``, the total water ``q_t``, and, with non-equilibrium microphysics,
cloud liquid and cloud ice. The `Subsidence()` term of the
[file-driven cases](#Driving-a-column-from-data) reaches only ``h_{tot}`` and
``q_t``. Neither touches rain, snow, the hydrometeor number concentrations, the
P3 rime variables, or the passive tracers.

!!! note "TODO: rain and snow subside only through ``q_t``"

    ``q_t`` includes the precipitating species: water vapor is diagnosed as the
    residual ``q_t - q_l - q_i``, with ``q_l = q_l^{cl} + q_r`` and
    ``q_i = q_i^{cl} + q_s`` (see [Microphysics](microphysics.md)). Subsiding
    ``q_t`` without subsiding ``q_r`` and ``q_s`` therefore attributes their
    vertical advection to vapor. The hydrometeor number concentrations likewise
    do not subside while the corresponding masses do, which changes the mean
    particle mass. Whether to subside the precipitating species, or to subside
    the effective total water ``q_t^{\mathrm{eff}} = q_t - q_r - q_s`` as the
    diffusive terms do, is a code-side question.

## Large-scale horizontal advection

Horizontal advection by the large-scale flow enters as prescribed tendencies of
temperature and total water, ``(\partial T/\partial t)_{adv}`` and
``(\partial q_t/\partial t)_{adv}``, supplied as functions of height and, for
temperature, of the Exner function. Total water takes the tendency directly, and
total energy takes

```math
\left. \frac{\partial \rho e_{tot}}{\partial t} \right|_{adv} =
\rho \left[ c_{vm} \left. \frac{\partial T}{\partial t} \right|_{adv}
  + I_v(T) \left. \frac{\partial q_t}{\partial t} \right|_{adv} \right] ,
```

with ``c_{vm}`` the isochoric heat capacity of the moist mixture and ``I_v`` the
internal energy of water vapor. No geopotential term appears: horizontal
advection moves specific humidity between columns at the same height.

## Column Coriolis and geostrophic forcing

In a single column the horizontal pressure gradient cannot be computed, so it is
prescribed through a geostrophic wind ``\boldsymbol{u}_g(z)`` and the flow is
relaxed toward it by the Coriolis acceleration,

```math
\left. \frac{\partial \boldsymbol{u}_h}{\partial t} \right|_{cor} =
  -f \hat{\boldsymbol{k}} \times (\boldsymbol{u}_h - \boldsymbol{u}_g) .
```

The Coriolis parameter ``f`` and both components of ``\boldsymbol{u}_g`` come
from the setup. A case may set ``f = 0``, which switches the term off while
leaving the geostrophic profile available to other parts of the setup.

## Driving a column from data

Cases driven by GCM output, reanalysis, or a field campaign compose their
forcing from a tuple of terms rather than fixing it in the setup. Four terms are
available, and each declares the variables it needs from the file:

| Term                    | Supplies                                                                                                                        | File variables      |
|:----------------------- |:------------------------------------------------------------------------------------------------------------------------------- |:------------------- |
| `HorizontalAdvection()` | Prescribed ``\partial T/\partial t`` and ``\partial q_t/\partial t`` from horizontal advection                                  | `tntha`, `tnhusha`  |
| `VerticalFluctuation()` | The eddy part of the vertical advective tendency                                                                                | `tntva`, `tnhusva`  |
| `Subsidence()`          | Vertical advection of ``h_{tot}`` and ``q_t`` by the large-scale vertical velocity, acting on the model's own evolving profiles | `wa`                |
| `Nudging(vars...)`      | Relaxation of the named variables toward the file profiles                                                                      | the named variables |

The default composition applies all four, nudging temperature and humidity as
one term and the two wind components as another. The separation matters because
the terms are not interchangeable: `VerticalFluctuation` supplies a fixed
tendency computed from the driving data, whereas `Subsidence` transports
whatever the column currently holds. A dataset that already contains the full
vertical advective tendency should use one or the other, not both.

The first three terms accumulate into a single pair of temperature and moisture
tendencies, which is then converted to the prognostic variables,

```math
\left. \frac{\partial \rho e_{tot}}{\partial t} \right|_{ext} =
\rho \left[ c_{vm} \frac{\partial T}{\partial t}
  + \left( c_{vv} (T - T_0) + L_{v0} - R_v T_0 \right) \frac{\partial q_t}{\partial t} \right] ,
\qquad
\left. \frac{\partial \rho q_t}{\partial t} \right|_{ext} = \rho \frac{\partial q_t}{\partial t} .
```

### Nudging

Nudging relaxes a variable toward the driving profile at a rate
``\tau^{-1}(z)``,

```math
\left. \frac{\partial \psi}{\partial t} \right|_{nudge} = -\frac{\psi - \psi_{data}(z, t)}{\tau(z)} .
```

For scalars, the rate follows a raised cosine in height [Shen2022](@cite), so
that the boundary layer evolves freely while the free troposphere is held near
the driving data:

```math
\tau^{-1}(z) =
\begin{cases}
  0 & z < z_i , \\[1ex]
  \dfrac{1}{2 \tau_r} \left[ 1 - \cos\left( \pi \dfrac{z - z_i}{z_r - z_i} \right) \right] & z_i \le z \le z_r , \\[2ex]
  \dfrac{1}{\tau_r} & z > z_r .
\end{cases}
```

The wind rate is height-independent. The four controlling parameters are

| Parameter              | ClimaParams name                          | Default |
|:---------------------- |:----------------------------------------- |:------- |
| ``\tau_r`` for scalars | `gcmdriven_scalar_relaxation_timescale`   | 86400 s |
| ``\tau`` for momentum  | `gcmdriven_momentum_relaxation_timescale` | 21600 s |
| ``z_i``                | `gcmdriven_relaxation_minimum_height`     | 3000 m  |
| ``z_r``                | `gcmdriven_relaxation_maximum_height`     | 3500 m  |

A term may instead be given a constant timescale, a profile, or a height mask.

### Where the data comes from

All three sources reach the forcing terms through one canonical, CMIP-named
vocabulary, so the terms themselves do not know which source they are reading:

  - **GCM output** at a cfsite, read as time-mean profiles. The subsidence
    velocity is derived from the pressure velocity, and the vertical-fluctuation
    tendency is the eddy part of the vertical advection, obtained by subtracting
    the mean advection from the total on the GCM grid.
  - **Reanalysis**, generated from ERA5 for a given site and date, either as a
    matched trajectory or as a monthly-averaged repeating day.
  - **ARM VARANAL** field-campaign forcing, converted from pressure levels.
    Its own vertical advective tendencies are deliberately discarded in favor of
    the `Subsidence()` term, so that vertical transport acts on the simulated
    profiles rather than on the observed ones.

[Column Datasets](column_datasets_reference.md) documents the file schema, the
per-case defaults, and how to supply your own file;
[Adding a Column Dataset](extending_column_datasets.md) covers new formats.

## Held–Suarez forcing

The Held–Suarez benchmark [HeldSuarez1994](@cite) replaces radiation and
boundary-layer physics with Newtonian relaxation toward a prescribed
equilibrium temperature and Rayleigh friction on the low-level winds. It is the
standard test of a dry dynamical core, and it occupies the radiation slot for
the reasons given in [Radiation](radiation.md).

Both terms are confined near the surface by the same ramp in the vertical
coordinate ``\sigma = p/p_s``,

```math
f_h(\sigma) = \max \left( 0, \frac{\sigma - \sigma_b}{1 - \sigma_b} \right) ,
```

where the surface pressure is reconstructed hydrostatically from the surface
elevation and temperature. The thermal relaxation rate is largest in the
tropical boundary layer,

```math
k_T(\phi, \sigma) = k_a + (k_s - k_a) \, f_h(\sigma) \cos^4\phi ,
\qquad k_a = \frac{1}{40 \, \text{day}} , \quad k_s = \frac{1}{4 \, \text{day}} ,
```

and the equilibrium temperature has the familiar equator-to-pole and static
stability contrasts, with a stratospheric floor,

```math
T_{eq}(\phi, p) = \max \left[ T_{min}, \,
  \left( T_{eq,0} - \Delta T_y \sin^2\phi - \Delta\theta_z \log\frac{p}{p_0} \cos^2\phi \right)
  \left( \frac{p}{p_0} \right)^{\kappa_d} \right] .
```

Friction relaxes the horizontal wind toward rest with
``k_f = 1/\text{day}``,

```math
\left. \frac{\partial \boldsymbol{u}_h}{\partial t} \right|_{HS} = -k_f f_h(\sigma) \, \boldsymbol{u}_h .
```

Two implementation choices are worth knowing. Temperature is diagnosed from the
ideal gas law for dry air, so the forcing is independent of moisture even in a
moist run. The energy tendency converts the temperature relaxation with the
*dry* isochoric heat capacity ``c_{vd}``, consistent with relaxing at constant
density. The equator-to-pole contrast and the equatorial equilibrium temperature
take different values in dry and moist configurations
(`ΔT_y_dry`/`ΔT_y_wet` and `T_equator_dry`/`T_equator_wet`).

## Prescribed radiative forcing

Three cases prescribe a radiative flux profile instead of calling RRTMGP: the
DYCOMS cloud-top cooling profile, its ISDAC counterpart, and the TRMM-LBA
prescribed heating rate. They are selected through the same `rad` key as the
radiative transfer modes and are described in [Radiation](radiation.md).

## Prescribed flow

The Shipway and Hill kinematic driver replaces the momentum equation entirely:
the vertical velocity is prescribed as a half-sine pulse in time, and the
thermodynamic and microphysical variables are advected through it. It is a test
of the microphysics and its transport, not of the dynamics, so it requires flat
topography and the explicit solver.

## What each case forces

| Case                          | Subsidence                               | Horizontal advection         | Coriolis                         | Other                                  |
|:----------------------------- |:---------------------------------------- |:---------------------------- |:-------------------------------- |:-------------------------------------- |
| Bomex                         | yes                                      | yes                          | ``f = 3.76 \times 10^{-5}`` s⁻¹  |                                        |
| Rico                          | yes                                      | yes                          | ``f = 4.5 \times 10^{-5}`` s⁻¹   |                                        |
| DYCOMS RF01, RF02             | ``-Dz``, ``D = 3.75 \times 10^{-6}`` s⁻¹ |                              | ``f = 0``                        | Prescribed radiation                   |
| GABLS                         |                                          |                              | ``f = 1.39 \times 10^{-4}`` s⁻¹  |                                        |
| Larcform1                     |                                          |                              | ``f = 1.432 \times 10^{-4}`` s⁻¹ | Polar-night insolation                 |
| ISDAC                         | yes                                      |                              |                                  | Analytic nudging, prescribed radiation |
| TRMM-LBA                      |                                          |                              |                                  | Prescribed radiative heating           |
| Soares, GATE III, SimplePlume |                                          |                              |                                  | None                                   |
| GCM, ARM VARANAL, reanalysis  | `Subsidence()` term                      | `HorizontalAdvection()` term |                                  | Nudging, vertical fluctuation          |

## Symbols

| Symbol                                | Meaning                                                                       | Units      |
|:------------------------------------- |:----------------------------------------------------------------------------- |:---------- |
| ``c_{vm}``, ``c_{vd}``, ``c_{vv}``    | Isochoric heat capacity of moist air, of dry air, of water vapor              | J kg⁻¹ K⁻¹ |
| ``f``                                 | Coriolis parameter                                                            | s⁻¹        |
| ``f_h``                               | Boundary-layer ramp in ``\sigma``                                             |            |
| ``I_v``                               | Specific internal energy of water vapor                                       | J kg⁻¹     |
| ``k_a``, ``k_s``, ``k_f``             | Held–Suarez free-tropospheric, surface, and friction rates                    | s⁻¹        |
| ``L_{v0}``                            | Latent heat of vaporization at ``T_0``                                        | J kg⁻¹     |
| ``T_{eq}``, ``T_{eq,0}``, ``T_{min}`` | Equilibrium temperature, its equatorial value, its floor                      | K          |
| ``\boldsymbol{u}_g``                  | Geostrophic wind                                                              | m s⁻¹      |
| ``w_{sub}``                           | Prescribed subsidence velocity, negative for descent                          | m s⁻¹      |
| ``z_i``, ``z_r``                      | Lower and upper height of the nudging ramp                                    | m          |
| ``\Delta T_y``, ``\Delta\theta_z``    | Equator-to-pole temperature contrast, vertical potential temperature gradient | K          |
| ``\kappa_d``                          | ``R_d/c_{pd}``                                                                |            |
| ``\sigma``, ``\sigma_b``              | Normalized pressure coordinate and the top of the friction layer              |            |
| ``\tau``, ``\tau_r``                  | Nudging timescale and its free-tropospheric value                             | s          |
| ``\chi``                              | Generic advected scalar                                                       | varies     |

## Where this is implemented

| Concept                                | Source                                                                                                                                                                                                                                    |
|:-------------------------------------- |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Subsidence                             | [forcing/subsidence.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/forcing/subsidence.jl)                                                                                                                  |
| Large-scale advection                  | [forcing/large_scale_advection.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/forcing/large_scale_advection.jl)                                                                                            |
| Composed forcing terms                 | [forcing/forcing_terms.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/forcing/forcing_terms.jl)                                                                                                            |
| External driving, nudging, term caches | [forcing/external_forcing.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/forcing/external_forcing.jl)                                                                                                      |
| Column Coriolis                        | [scm_coriolis.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/scm_coriolis.jl)                                                                                                                              |
| Held–Suarez                            | [radiation/held_suarez.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/radiation/held_suarez.jl)                                                                                                        |
| Case definitions                       | [src/setups/](https://github.com/CliMA/ClimaAtmos.jl/tree/main/src/setups)                                                                                                                                                                |
| Dataset formats                        | [src/column_datasets/](https://github.com/CliMA/ClimaAtmos.jl/tree/main/src/column_datasets)                                                                                                                                              |
| Model types                            | [`ClimaAtmos.AbstractForcing`](@ref), [`ClimaAtmos.LargeScaleSubsidence`](@ref), [`ClimaAtmos.LargeScaleAdvection`](@ref), [`ClimaAtmos.HeldSuarezForcing`](@ref), [`ClimaAtmos.ISDACForcing`](@ref), [`ClimaAtmos.PrescribedFlow`](@ref) |
