# Microphysics

## Source terms

Sources from cloud microphysics ``\mathcal{S}`` represent the transfer of mass
between different water categories such as cloud water, cloud ice or precipitation,
as well as the latent heat release due to phase changes.
The model supports four cloud microphysics and precipitation representations:

  - equilibrium cloud formation coupled with a 0-moment microphysics scheme,
  - nonequilibrium cloud formation coupled with a 1-moment microphysics scheme
    representing both liquid and ice phase precipitation,
  - nonequilibrium cloud formation coupled with a 2-moment microphysics scheme
    representing liquid phase precipitation,
  - nonequilibrium cloud formation coupled with a 2-moment warm-rain scheme
    plus the P3 ice scheme (`microphysics_model: "2MP3"`), which additionally
    prognoses ice number and rime mass and volume.

The equilibrium 0-moment option does not introduce any microphysics variables beyond ``\rho q_{tot}``.
The cloud condensate and phase partitioning are diagnosed using saturation adjustment
and the 0-moment microphysics provides a sink on total water due to precipitation.
Precipitation is immediately removed from the computational domain, except for the part
that evaporates or sublimates below cloud. This evaporation is a model option, set
in the YAML config with `precipitation_evaporation_0m: "Kessler"` (default `~`, off;
the option is only valid with `microphysics_model: "0M"`). When it is on, the
`precipitation_evaporation_*` parameters are read (their ClimaParams defaults, e.g.
`precipitation_evaporation_coefficient = 1.36e-4` [1/s], unless they are set in the
parameter TOML; the run errors if the coefficient is not positive). The 0-moment sweep
of `step_coupled_microphysics_sedimentation!` then scans each column top-down: each
cell removes its condensate excess into the falling flux and takes up the evaporation
of the flux from above, limited by that flux and by wet-bulb cooling. The evaporation
goes to the grid mean, and with `PrognosticEDMFX` it is computed from the environment
state, so the environment takes it up. The scan also carries the energy the falling
precipitation was removed from the air with; evaporated water returns the flux-mean
specific energy of its phase, so precipitation that evaporates entirely leaves no
energy flux at the surface, and the surface water and energy fluxes are the column
integrals of the removal less the evaporation.
With `PrognosticEDMFX`, updraft condensate precipitates on its own timescale,
`precipitation_timescale_updraft_ratio` times `precipitation_timescale`
(the default ratio is 1).
The nonequilibrium 1-moment option expands the state vector by four microphysics tracers:
cloud liquid water, cloud ice, rain and snow ``(q_{liq}, q_{ice}, q_{rai}, q_{sno})``;
in the state vector these are `ρq_lcl`, `ρq_icl`, `ρq_rai`, and `ρq_sno`.
The nonequilibrium 2-moment option expands the state vector by six microphysics tracers:
cloud liquid and cloud ice mass, rain and snow mass, and cloud droplet and rain drop
number concentrations ``(q_{liq}, q_{ice}, q_{rai}, q_{sno}, N_{liq}, N_{rai})``,
the last two carried as `ρn_lcl` and `ρn_rai`.
Ice and snow are carried but their 2-moment microphysical sources are currently zero;
only warm-rain processes are active.

All microphysics mass tracers are part of the working fluid
and are defined as a ratio of the tracer mass over the mass of the working fluid.
The different cloud and precipitation source terms are provided by
[CloudMicrophysics.jl](https://github.com/CliMA/CloudMicrophysics.jl) library
and are defined as the change of mass normalized by the mass of the working fluid.
See the [CloudMicrophysics.jl docs](https://clima.github.io/CloudMicrophysics.jl/dev/)
for more details.

Consider the transition ``x \rightarrow y``, where ``x`` and ``y`` can be any
of the microphysics tracers:

```math
\mathcal{S}_{x \rightarrow y} := \frac{\frac{dm_x}{dt}}{m_{dry} + m_{vap} + m_{liq} + m_{ice} + m_{rai} + m_{sno}}
```

If ``\mathcal{S}_{x \rightarrow y}`` is a sink of ``q_{tot}`` from the 0-moment scheme,
it has a corresponding sink on density and energy:

```math
\frac{d}{dt} \rho =
\frac{d}{dt} \rho q_{tot} =
\rho \mathcal{S}_{x \rightarrow y}
```

```math
\frac{d}{dt} \rho e = \rho \mathcal{S}_{x \rightarrow y} (I_{y} + \Phi)
```

where ``I_{y}`` is the internal energy of the ``y`` phase.

In nonequilibrium cloud formation and the 1-moment and 2-moment schemes,
since all microphysics tracers are part of the working fluid,
microphysics sources do not introduce corresponding sources/sinks of
total water, density or total energy.

!!! note

    In the above derivations, we assume that the volume
    of the working fluid is constant (not the pressure).

### Autoconversion velocity of the PROPHET environment

The `Kessler1M` rain autoconversion blends its threshold and timescale between
quiescent (stratiform) and convective values with the factor
``w_+^2 / (w_+^2 + w_0^2)``, where ``w_+ = \max(w, 0)`` is the upward velocity
of the subdomain for which the rate is evaluated. With `PrognosticEDMFX`, each
updraft takes its own velocity ``w^j`` and the environment its velocity
``w^0``. The environment descends where updrafts rise, so cloud liquid that
updrafts detrain into it converts to rain at the quiescent values. Two
provisional parameters let it convert at convective values: the environment
autoconversion is evaluated at

```math
w^0_{mp} = \max\!\left( w^0, \; f_{neq} \left( c_w w_{up} + w_d \right) \right),
\qquad
w_{up} = \frac{\sum_j a^j \max(w^j, 0)}{\max\!\left( \sum_j a^j, \, a_{\min} \right)},
\qquad
f_{neq} = \frac{\left( q_c^0 - q_{c,eq}^0 \right)_+}{q_c^0} .
```

Here, ``w_{up}`` is the area-weighted mean updraft velocity of the cell, and
``f_{neq}`` is the fraction of the environment cloud condensate
``q_c^0 = q_{liq}^0 + q_{ice}^0`` in excess of the equilibrium condensate
``q_{c,eq}^0`` of the subgrid saturation-excess distribution (see
[PROPHET: Closures](prophet_closures.md)). Cloud liquid relaxes to equilibrium
within its condensation timescale, so an excess persists where detrainment
supplies it: ``f_{neq} \approx 0`` in a stratocumulus deck, and
``f_{neq} \approx 1`` for condensate detrained into a subsaturated cumulus
layer. The weight ``c_w`` (`sgs_env_autoconversion_updraft_velocity_weight`)
scales the updraft velocity, and ``w_d``
(`sgs_env_autoconversion_detrained_velocity`) is the velocity of detrained
condensate that has outlived its drafts. Both are zero by default, which
recovers ``w^0``.

The excess ``q_c^0 - q_{c,eq}^0`` is the vapor deficit of the environment; for
a narrow distribution with a saturated mean, it equals
``q_{sat}^0 - q_{vap}^0``. It is independent of the phase partition of the
condensate, so cloud ice, which relaxes more slowly than cloud liquid, enters
``f_{neq}`` through the total ``q_c^0``. Where the excess is positive, the
vapor is subsaturated with respect to liquid, and cloud liquid evaporates
within its condensation timescale unless detrainment supplies it.

In a descending environment, the blending factor of the autoconversion is
``x^2 / (1 + x^2)`` with ``x = f_{neq} (c_w w_{up} + w_d) / w_0``. It is
quadratic in ``f_{neq}`` for ``x \ll 1`` and reaches one half at
``f_{neq} = w_0 / (c_w w_{up} + w_d)``. The parameters ``c_w`` and ``w_d``
hence act relative to the velocity scale ``w_0``
(`rain_autoconversion_velocity_scale`): values that take
``c_w w_{up} + w_d`` far above ``w_0`` turn the blending into a switch at
small ``f_{neq}``.

## Sedimentation

All microphysics tracers sediment with a bulk (group) sedimentation velocity.
The sedimentation velocity can be parameterized via CloudMicrophysics.jl or
specified as a fixed value for each tracer.

Cloud condensate that fills only part of a cell is more concentrated within the
cloud than in the cell mean, and its particles are larger and fall faster: the
1-moment velocities of cloud liquid (Stokes) and cloud ice (Chen et al., 2022)
take the mean particle size from the specific humidity and the prescribed
number concentration of cloudy air. With the parameter `sgs_cloud_sedimentation_cf_floor` set below
its default of 1, these velocities are evaluated at the in-cloud specific
humidity `q / clamp(CF, c_min, 1)`, where `CF` is the cloud fraction and
`c_min` is the value of the parameter. The velocity multiplies the mean
condensate, so the sedimentation flux is the in-cloud flux weighted by the
cloud fraction. With `PrognosticEDMFX`, `q` and `CF` are those of the
environment; the updrafts sediment cloud condensate at their own mean specific
humidity.

Under SGS quadrature, the one-moment microphysics places a fraction of the rain
and snow on a subgrid precipitation shaft (`sgs_precip_incloud_fraction`,
`sgs_precip_fraction_floor`), and the two species sediment at the mass-weighted
mean of the terminal velocities in the shaft and in the rest of the cell
(`shaft_weighted_terminal_velocity`). Because the terminal velocity increases
with the specific humidity of the species, precipitation concentrated in the
shaft falls faster than the same mass spread over the cell.

Sedimentation is treated implicitly per column via backward-Euler first-order
upwinding in the post-ARK coupled microphysics-sedimentation sweep
(`step_coupled_microphysics_sedimentation!`). Because all tracers are part of
the working fluid, their sedimentation produces corresponding mass and
total-energy fluxes, surface precipitation diagnostics (`surface_rain_flux`,
`surface_snow_flux`, `col_integrated_precip_energy_tendency`), and precipitation
energy and mass exchange for `SlabOceanTemperature`.

!!! note

    We assume that all microphysics tracers are at ambient air temperature.
    A more accurate treatment would assume that sedimenting condensate is
    at the wet-bulb temperature.

## Stability and positivity

Microphysics tracers should remain positive throughout the simulation.
The model numerics, however, may produce errors that lead to the spurious formation
of small negative numbers.
The most common causes of these errors are:

  - spurious oscillations caused by the high order horizontal transport scheme,
  - time integration of microphysics sources at a time step longer than the stability limit,
  - use of hyperdiffusion.

Our strategy is to minimize the effects of those errors.

### Implicit treatment of microphysics sinks

Microphysical processes can produce large negative tendencies (sinks) for
tracer variables. These are handled through a time-averaged formulation in
CloudMicrophysics.jl, in which sink terms are locally linearized and folded
into the time integration scheme; see the [CloudMicrophysics.jl bulk-tendency
documentation](https://clima.github.io/CloudMicrophysics.jl/dev/BulkTendencies/).

### Coupled microphysics and sedimentation

In `step_coupled_microphysics_sedimentation!`
(`coupled_microphysics_sweep.jl`), microphysical phase changes and
sedimentation are solved jointly per column in a top-down sweep. For 1-moment
microphysics, `_combined_microphysics_1m_params` merges `microphysics_1m_params`
and `microphysics_1m_implicit_params` so cloud condensation, sublimation, and
precipitation formation solve together with sedimentation. For 1-moment subgrid-scale
quadrature (`LogNormalQuad`, `GaussQuad`), `_CompactConditionalRule`
(`conditional_nodes.jl`) packs the active conditional intervals into a dense
list of at most eight wet nodes, skipping inactive slots and dry clear-air
branches. On GPUs, 1-moment sweeps pre-scan horizontal columns into active
vertical-extent and SGS buckets via `DataLayouts.IJHMask`, skipping clear dry
columns and grouping active columns for warp-coherent, deterministic column
updates. For 2-moment microphysics, the sweep subcycles at
``\Delta t_{\mathrm{sub,2M}} \le 30\,\mathrm{s}`` when
``\Delta t > 30\,\mathrm{s}``.

### Enforcing physical constraints

Additional corrections are applied through the state-constraint hook
`constrain_state!`, which runs at the cadence set by
`update_constrain_state_every` (every stage by default). Its microphysics and
updraft part is

```julia
enforce_physical_constraints!(Y, p, t, atmos)
```

This function applies local corrective updates to keep prognostic variables
in a physically admissible range.

Currently, this includes:

  - enforcing non-negative condensate masses,
  - rescaling condensate when the total condensate exceeds total moisture,
  - clamping the updraft area fraction to ``[0, 1]``, clipping negative updraft
    vertical velocities, and resetting the updraft scalars to the grid mean
    where the updraft area is negligible (the `edmfx_filter` option),
  - ensuring subdomain consistency ``\rho a^j \chi^j \le \rho \chi``.

These corrections are intended to prevent nonphysical states such as negative
tracer values or condensate mass exceeding the available total moisture.

### Hyperdiffusion

Hyperdiffusion (``\nabla^4`` operator) is a tendency applied
to remove noise buildup at the small scales and improve model stability.
It is more selective than the second-order diffusion operator of
[Diffusion](diffusion.md), damping only the smallest scales of the simulation
while preserving the sharp features of the modeled tracers.

Hyperdiffusion is a higher order derivative operator, and as a result does not guarantee positivity.
Total water (`q_tot_eff = q_tot - q_rai - q_sno` for the 1M and 2M schemes;
the full `q_tot` for 0M and 2MP3) and passive tracers are
hyperdiffused at full strength; the resulting `ρq_tot` mass tendency is
distributed proportionally to the cloud species (`ρq_lcl`, `ρq_icl`, and their
number densities). Rain, snow, and rain number density receive no
hyperdiffusion.

### Diffusion

Physical diffusion smooths tracer gradients, so it also reduces the negative
values and spurious oscillations discussed above. The available schemes, which
species each one diffuses, and how the tendency is split between the implicit
and explicit halves of the timestep are described in
[Diffusion](diffusion.md).

### Non-negativity constraints

Often, diffusion and the limiters described above are not enough to ensure
positivity of the microphysics tracers.
ClimaAtmos supports four additional constraints that can be used to enforce non-negativity of the microphysics tracers.
This is controlled by the `tracer_nonnegativity_method` in the `AtmosWater` struct.
`elementwise_constraint` also comes in a `_qtot` variant that includes `ρq_tot`
itself among the constrained tracers and adds the water it creates to the air
mass and total energy as vapor, so that the three budgets stay consistent.
`vapor_constraint_qtot` is accepted but leaves `ρq_tot` unchanged, since a
negative total water has no vapor to borrow from; `vapor_tendency` and
`vertical_water_borrowing` ignore the suffix with a warning.
The available options are:

  - `TracerNonnegativityElementConstraint`:
    This option enforces non-negativity while conserving the tracer mass within the element.
    It uses the `Limiters.compute_bounds!` and `Limiters.apply_limiter!` functions to redistribute the mass of the tracer within the element
    such that the tracer concentration is non-negative and bounded by the maximum value in the element.
    This method borrows mass from the neighboring nodes within the element to fill the negative holes.
    This method is conservative and does not introduce any source/sink of total water mass.

  - `TracerNonnegativityVaporConstraint`:
    This option enforces non-negativity by borrowing mass from the water vapor.
    If a microphysics tracer ``q_x`` becomes negative at a given node, it is set to zero.
    Since the total water content ``q_{tot}`` is conserved during this operation, and ``q_{tot} = q_{vap} + \sum q_x``,
    setting a negative ``q_x`` to zero implicitly decreases ``q_{vap}``.

    ```math
    q_x = \max(0, q_x) \quad \text{where } q_{tot} > 0
    ```

    (nodes with ``q_{tot} \le 0`` are left unchanged).
    This method is applied instantaneously at the end of each time step (or stage).
    It preserves the total water mass but redistributes it between phases.
    It should be used with caution as it can lead to negative water vapor if the negative hole in ``q_x`` is large, although
    usually the negative values are small and there is plenty of water vapor available.

  - `TracerNonnegativityVaporTendency`:
    This option is similar to `TracerNonnegativityVaporConstraint` in that it borrows mass from water vapor,
    but it does so via a tendency term rather than an instantaneous adjustment.
    It computes a tendency that tends to restore the tracer to zero over the timestep ``\Delta t``.
    The tendency is limited by the available water vapor ``q_{vap}`` to avoid creating negative vapor.

    ```math
    \frac{\partial q_x}{\partial t} = \dots + \mathcal{S}_{fixer}
    ```

    where ``\mathcal{S}_{fixer}`` is positive if ``q_x < 0``.
    This method is less aggressive than the instantaneous constraint and integrates the correction into the time stepping scheme.

  - `TracerNonnegativityVerticalWaterBorrowing`:
    This option redistributes tracer mass vertically within each column using a
    vertical mass-borrowing limiter, filling negative values from levels above
    or below. The species it applies to are set by the
    `vertical_water_borrowing_species` configuration argument; when total water
    is limited, density and total energy are corrected for consistency. This one acts in the timestepper's limiter stage; the other three run from the state-constraint hook.

## Aerosol activation for 2-moment microphysics

Aerosol activation uses functions from the
[CloudMicrophysics.jl](https://github.com/CliMA/CloudMicrophysics.jl) library,
based on the Abdul-Razzak and Ghan (ARG) parameterization
[AbdulRazzakGhan2000](@cite). ARG predicts the number of activated cloud
droplets in a parcel of clear air rising adiabatically. It is traditionally
applied only at cloud base, where the maximum supersaturation typically occurs.

To use ARG locally, without explicitly identifying cloud base,
CloudMicrophysics.jl implements a modified equation for the maximum
supersaturation that accounts for pre-existing liquid and ice particles.
Activation can then be applied inside clouds. So that it occurs only where it
is physically appropriate, we add clipping logic:

  - If the predicted maximum supersaturation is less than the local supersaturation (i.e., supersaturation is decreasing), aerosol activation is not applied.
  - If the predicted number of activated droplets is less than the existing local cloud droplet number concentration, activation is also suppressed.

Droplet activation therefore occurs only in physically meaningful regions —
typically near cloud base — even though the activation routine can be applied
throughout the domain.

## Symbols

| Symbol                            | Meaning                                                                                                   | Units         |
|:--------------------------------- |:--------------------------------------------------------------------------------------------------------- |:------------- |
| ``\mathcal{S}_{x \rightarrow y}`` | Microphysical transfer from category ``x`` to category ``y``, normalized by the mass of the working fluid | s⁻¹           |
| ``\mathcal{S}_{fixer}``           | Restoring tendency of the non-negativity fixer                                                            | s⁻¹           |
| ``m_x``                           | Mass of category ``x``, with ``x`` one of dry air, vapor, liquid, ice, rain, snow                         | kg            |
| ``q_{tot}``, ``q_{vap}``          | Total water and water vapor specific humidity                                                             | kg kg⁻¹       |
| ``q_{liq}``, ``q_{ice}``          | Cloud liquid and cloud ice specific humidity, carried as `ρq_lcl` and `ρq_icl`                            | kg kg⁻¹       |
| ``q_{rai}``, ``q_{sno}``          | Rain and snow specific content, carried as `ρq_rai` and `ρq_sno`                                          | kg kg⁻¹       |
| ``q_x``                           | Any one of the microphysics tracers                                                                       | kg kg⁻¹       |
| ``N_{liq}``, ``N_{rai}``          | Cloud droplet and rain drop number per unit mass of working fluid, carried as `ρn_lcl` and `ρn_rai`       | kg⁻¹          |
| ``I_y``                           | Specific internal energy of phase ``y``                                                                   | J kg⁻¹        |
| ``\Phi``                          | Geopotential                                                                                              | m² s⁻²        |
| ``\rho``, ``\rho e``              | Density and volumetric total energy of moist air                                                          | kg m⁻³, J m⁻³ |
| ``a^j``, ``\chi^j``               | Area fraction and tracer of updraft subdomain ``j``                                                       | varies        |
| ``w^j``, ``w^0``                  | Vertical velocity of updraft ``j`` and of the environment                                                 | m s⁻¹         |
| ``\Delta t``                      | Model timestep                                                                                            | s             |

[Notation and Symbols](notation.md) maps these onto the field names in the code;
there the water species are written ``q_l^{cl}``, ``q_i^{cl}``, ``q_r``, ``q_s``,
``n_l``, and ``n_r``, following the dycore paper.

## Where this is implemented

| Concept                                                    | Source                                                                                                                                                                                                                                                                                             |
|:---------------------------------------------------------- |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Coupled microphysics and sedimentation sweeps              | [coupled_microphysics_sweep.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/microphysics/coupled_microphysics_sweep.jl), [conditional_nodes.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/microphysics/conditional_nodes.jl) |
| CloudMicrophysics.jl wrappers and subgrid-scale quadrature | [microphysics_wrappers.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/microphysics/microphysics_wrappers.jl), [sgs_quadrature.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/microphysics/sgs_quadrature.jl)                 |
| Sedimentation of water and its enthalpy                    | [src/prognostic_equations/water_advection.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/water_advection.jl)                                                                                                                                                        |
| State-constraint hook                                      | [src/prognostic_equations/constrain_state.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/constrain_state.jl)                                                                                                                                                        |
| Vapor-borrowing tendency fixer                             | [src/parameterized_tendencies/microphysics/moisture_fixers.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/microphysics/moisture_fixers.jl)                                                                                                                      |
| Element and vertical-borrowing limiters                    | [src/prognostic_equations/limited_tendencies.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/limited_tendencies.jl)                                                                                                                                                  |
| Tracer hyperdiffusion                                      | [src/prognostic_equations/hyperdiffusion.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/hyperdiffusion.jl)                                                                                                                                                          |
| Model types                                                | [`ClimaAtmos.EquilibriumMicrophysics0M`](@ref), [`ClimaAtmos.NonEquilibriumMicrophysics1M`](@ref), [`ClimaAtmos.NonEquilibriumMicrophysics2M`](@ref), [`ClimaAtmos.NonEquilibriumMicrophysics2MP3`](@ref), [`ClimaAtmos.TracerNonnegativityMethod`](@ref)                                          |

The scheme is selected with the `microphysics_model` configuration key; see
[Configuration Options](configuration_options.md).
