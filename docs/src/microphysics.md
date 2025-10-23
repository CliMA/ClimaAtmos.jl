# Microphysics

## Source terms

Sources from cloud microphysics ``\mathcal{S}`` represent the transfer of mass
  between different water categories such as cloud water, cloud ice or precipitation,
  as well as the latent heat release due to phase changes.
The model supports three different cloud microphysics and precipitation representations:
  - equilibrium cloud formation coupled with a 0-moment microphysics scheme,
  - nonequilibrium cloud formation coupled with a 1-moment microphysics scheme
    representing both liquid and ice phase precipitation,
  - nonequilibrium cloud formation coupled with a 2-moment microphysics scheme
    representing liquid phase precipitation.

The equilibrium 0-moment option does not introduce any new variables to the state vector.
The cloud condensate and phase partitioning are diagnosed using saturation adjustment
  and the 0-moment microphysics provides a sink on total water due to precipitation.
Precipitation is immediately removed from the computational domain.
The nonequilibrium 1-moment option expands the state vector by four microphysics tracers:
  cloud liquid water, cloud ice, rain and snow ``(q_{liq}, q_{ice}, q_{rai}, q_{sno})``.
The nonequilibrium 2-moment option expands the state vector by four microphysics tracers:
  cloud liquid water and droplet number concentration, rain water and drop number concentration:
   ``(q_{liq}, N_{liq}, q_{rai}, N_{rai})``.

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
If ``\mathcal{S}_{x \rightarrow y}`` is a sink of ``q_{tot}`` from the 0-moment scheme
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

!!! todo
    In the above derivations we are assuming that the volume
    of the working fluid is constant (not the pressure).

## Sedimentation

All microphysics tracers sediment with a bulk (group) sedimentation velocity
  parameterized via [CloudMicrophysics.jl](https://github.com/CliMA/CloudMicrophysics.jl).
Sedimentation is done implicitly through a first-order upwinding scheme.
Because all tracers are part of the working fluid, their sedimentation
  results in sedimentation terms for density and total energy.

!!! todo
    We assume that all microphysics tracers are at ambient air temperature.
    It would be more correct to assume that the microphysics tracers are
    at wet bulb temperature.

## Stability and positivity

Microphysics tracers should remain positive throughout the simulation.
The numerics of the model however, may result in errors that lead to the spurious formation
  of small negative numbers.
Most common causes of those errors are:
  - spurious oscillations caused by the high order horizontal transport scheme,
  - time integration of microphysics sources at time-step that is longer than the stability limit,
  - use of hyperdiffusion.
Our strategy is to minimize the untoward effects of those errors,
  without aiming for strict positivity.

### Limiters

All microphysics source terms are individually limited by the available mass of the source tracer ``x``.
We typically set ``lim_x = \frac{q_x}{a \; dt}`` where ``a > 1``.
The limiter is formulated based on the triangle inequality:
```math
limiter(\mathcal{S}_{x \rightarrow y}, lim_x, lim_y) = \mathcal{S}_{x \rightarrow y} + lim_x - \sqrt{\mathcal{S}_{x \rightarrow y}^2 + lim_x^2}
```
If the source is positive but larger than the available tracer ``lim_x``,
  the tendency is smoothly adjusted.
If, due to numerics, the source tracer is negative, the resulting tendency switches signs and acts
  as a restoring force towards a state where the tracer is positive.
This means that instead of ``x \rightarrow y`` transfer, we now consider ``y \rightarrow x``.
In such cases we limit the tendency by the available mass of the new source tracer ``y``.
All the tendencies passed to the limiter should be positive.
In case the numerics switches the sign of the source term itself,
  we again treat it as an inverted process ``y \rightarrow x``:
``limiter(\mathcal{S}_{x \rightarrow y}, lim_x, lim_y) = -limiter(-\mathcal{S}_{x \rightarrow y}, lim_y, lim_x)``.
Below figure illustrates the behavior of the limiter for positive and negative force
  with the ``x`` tracer being capped at 5 and the ``y`` tracer being capped at 2.

```@example
include("limiter_plots.jl") # hide
```
![] (assets/limiters_plot.png)

### Hyperdiffusion

Hyperdiffusion (``\nabla^4`` operator) is a tendency applied
  in order to remove noise buildup at the small scales and improve the model stability.
It's more selective than standard diffusion operator, and applies the damping only
  at the smallest scales of the simulation without degrading the sharp features
  of the modeled tracers.

Hyperdiffusion is a higher order derivative operator, and as a result does not guarantee positivity.
The user has a choice to opt-in certain microphysics tracers to use hyperdiffusion.
By default hyperdiffusion is applied to total water and cloud tracers,
  but not precipitating tracers.
The magnitude of hyperdiffusion acting on precipitation tracers can be changed by
  adjusting the free parameter `tracer_hyperdiffusion_factor`.

### Diffusion

ClimaAtmos provides different horizontal and vertical diffusion schemes that can be used
  to improve model stability and reduce the negative numbers and spurious oscillations.

Horizontal diffusion tendency is based on either the Smagorinsky-Lilly model
  [Sridhar2022](@cite) or the Anisotropic Minimum-Dissipation model (AMD) [Akbar2016](@cite)
  and is applied explicitly.

Vertical diffusion tendency can be based on either of the above models,
  or computed as a decaying with height function that is capped at some value above the tropopause.
Vertical diffusion can be applied implicitly.
When using the decay with height options (`VerticalDiffusion` or `DecayWithHeight`),
  similar to hyperdiffusion,  diffusion is applied to total water and cloud tracers.
The magnitude of diffusion acting on precipitation tracers can be scaled using the
  `tracer_vertical_diffusion_factor`.
There is no such scaling applied when using the Smagorinsky-Lilly or AMD models.

## Aerosol Activation for 2-Moment Microphysics

Aerosol activation uses functions from the [CloudMicrophysics.jl](https://github.com/CliMA/CloudMicrophysics.jl) library, based on the Abdul-Razzak and Ghan (ARG) parameterization. ARG predicts the number of activated cloud droplets assuming a parcel of clear air rising adiabatically. This formulation is traditionally applied only at cloud base, where the maximum supersaturation typically occurs.

To enable ARG to be used locally (i.e., without explicitly identifying cloud base), CloudMicrophysics.jl implements a modified equation for the maximum supersaturation that accounts for the presence of pre-existing liquid and ice particles. This allows activation to be applied inside clouds. To ensure that activation occurs only where physically appropriate, we apply additional clipping logic:

- If the predicted maximum supersaturation is less than the local supersaturation (i.e., supersaturation is decreasing), aerosol activation is not applied.
- If the predicted number of activated droplets is less than the existing local cloud droplet number concentration, activation is also suppressed.

This ensures that droplet activation occurs only in physically meaningful regions—typically near cloud base—even though the activation routine can be applied throughout the domain.
