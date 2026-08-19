# Microphysics

## Source terms

Sources from cloud microphysics ``\mathcal{S}`` represent the transfer of mass
between different water categories such as cloud water, cloud ice or precipitation,
as well as the latent heat release due to phase changes.
The model supports four different cloud microphysics and precipitation representations:

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
Precipitation is immediately removed from the computational domain.
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

Considering the transition from
``x \rightarrow y`` where ``x`` and ``y`` can be any of the microphysics tracers

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

    In the above derivations we assume that the volume
    of the working fluid is constant (not the pressure).

## Sedimentation

All microphysics tracers sediment with a bulk (group) sedimentation velocity.
The sedimentation velocity can be parameterized via CloudMicrophysics.jl or
specified as a fixed value for each tracer.

Sedimentation is treated implicitly through a first-order upwinding scheme.
Because all tracers are part of the working fluid, their sedimentation
results in sedimentation terms for density and total energy.

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

Microphysical processes can produce large negative tendencies (sinks) for tracer variables. These tendencies are handled through a time-averaged formulation in CloudMicrophysics.jl, in which sink terms are locally linearized and incorporated into the time integration scheme (see [here](https://clima.github.io/CloudMicrophysics.jl/dev/BulkTendencies/)).

### Enforcing physical constraints

Additional corrections are applied through the state-constraint hook
`constrain_state!`, which runs at the cadence set by
`update_constrain_state_every` (every step by default). Its microphysics and
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
It is more selective than the standard diffusion operator, and applies the damping only
at the smallest scales of the simulation without degrading the sharp features
of the modeled tracers.

Hyperdiffusion is a higher order derivative operator, and as a result does not guarantee positivity.
Total water (`q_tot_eff = q_tot - q_rai - q_sno` for the 1M and 2M schemes;
the full `q_tot` for 0M and 2MP3) and passive tracers are
hyperdiffused at full strength; the resulting `ρq_tot` mass tendency is
distributed proportionally to the cloud species (`ρq_lcl`, `ρq_icl`, and their
number densities). Rain, snow, and rain number density receive no
hyperdiffusion.

### Diffusion

ClimaAtmos provides different horizontal and vertical diffusion schemes that
improve model stability and reduce negative numbers and spurious oscillations.

Horizontal diffusion tendency is based on either the Smagorinsky-Lilly model
[Sridhar2022](@cite) or the Anisotropic Minimum-Dissipation model (AMD) [Akbar2016](@cite)
and is applied explicitly.

Vertical diffusion tendency can be based on either of the above models,
or computed as a function that decays with height and is capped at some value above the tropopause.
Vertical diffusion can be applied implicitly when using `VerticalDiffusion`,
`DecayWithHeightDiffusion`, or the PROPHET diffusive flux; the Smagorinsky-Lilly
and AMD vertical tendencies are always explicit.
With `VerticalDiffusion` or `DecayWithHeightDiffusion`, `q_tot_eff` (defined as
for hyperdiffusion above) is diffused directly and the resulting mass tendency is
distributed proportionally to the cloud species; precipitating species
(`q_rai`, `q_sno`, `n_rai`) receive no diffusion. The Smagorinsky-Lilly and
AMD models apply the full eddy diffusivity to every grid-scale tracer.

### Non-negativity constraints

Often, the diffusion and limiters described above are not enough to ensure positivity of the microphysics tracers.
ClimaAtmos supports four additional constraints that can be used to enforce non-negativity of the microphysics tracers.
This is controlled by the `tracer_nonnegativity_method` in the `AtmosWater` struct.
Each method (except `vertical_water_borrowing`) also comes in a `_qtot` variant,
which additionally constrains `ρq_tot` itself and then restores mass–energy
consistency.
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
    is limited, density and total energy are corrected for consistency. Unlike
    the other three options, this one acts in the timestepper's limiter stage
    rather than in the state-constraint hook.

## Aerosol Activation for 2-Moment Microphysics

Aerosol activation uses functions from the [CloudMicrophysics.jl](https://github.com/CliMA/CloudMicrophysics.jl) library, based on the Abdul-Razzak and Ghan (ARG) parameterization [AbdulRazzakGhan2000](@cite). ARG predicts the number of activated cloud droplets assuming a parcel of clear air rising adiabatically. This formulation is traditionally applied only at cloud base, where the maximum supersaturation typically occurs.

To enable ARG to be used locally (i.e., without explicitly identifying cloud base), CloudMicrophysics.jl implements a modified equation for the maximum supersaturation that accounts for the presence of pre-existing liquid and ice particles. This allows activation to be applied inside clouds. To ensure that activation occurs only where physically appropriate, we apply additional clipping logic:

  - If the predicted maximum supersaturation is less than the local supersaturation (i.e., supersaturation is decreasing), aerosol activation is not applied.
  - If the predicted number of activated droplets is less than the existing local cloud droplet number concentration, activation is also suppressed.

This ensures that droplet activation occurs only in physically meaningful regions—typically near cloud base—even though the activation routine can be applied throughout the domain.
