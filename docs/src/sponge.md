# Model Top and Sponge Layer

The model top is a rigid lid at a fixed height ``z_t``, with insulating and
free-slip boundary conditions. A rigid lid reflects upward-propagating waves, so
a sponge layer below it absorbs them before they can reflect and contaminate the
solution [Yatunin2026](@cite).

ClimaAtmos provides two independent sponges: a viscous sponge, which applies
horizontal viscous damping to the velocities and most scalars, and Rayleigh
damping, which relaxes velocities toward zero. Both are optional and both are
confined to the upper part of the domain by the same vertical ramp.

## The ramp

Damping has to switch on smoothly, or the onset itself reflects waves. Both
sponges use the ramp

```math
\beta_{\mathrm{sponge}}(z) =
\begin{cases}
  0 & z \le z_d, \\[1ex]
  \sin^2 \left( \dfrac{\pi}{2} \dfrac{z - z_d}{z_t - z_d} \right) & z > z_d,
\end{cases}
```

which rises from zero at the onset height ``z_d`` to one at the model top
[KlempLilly1978](@cite).

The layer should be roughly 1.5 times the vertical wavelength of the waves it
aims to absorb, which is around 2–10 km in the stratosphere
[Durran1983, Klemp2008](@cite). The default onset
height is set by the `zd_rayleigh` and `zd_viscous` parameters.

!!! warning "The onset height is an absolute altitude"

    `zd_rayleigh` and `zd_viscous` set ``z_d`` as an altitude above the surface,
    used as given. A domain with a different top therefore needs its own values:
    set them explicitly whenever `z_max` differs from the configuration the defaults were tuned for.

## Viscous sponge

The viscous sponge applies horizontal viscous damping to the velocities
(grid-mean and updraft vertical velocities), the total energy, the water, and
the passive grid-scale tracers. As in
[hyperdiffusion](hyperdiffusion.md), the precipitating species are exempt:
water is damped through the effective total water
``q_t^{\mathrm{eff}} = q_t - q_{rai} - q_{sno}``, and the resulting mass
tendency is distributed over the suspended cloud species. Turbulence kinetic
energy and the PROPHET subdomain scalars are also exempt; they receive only the
Rayleigh tracer damping described below. For a scalar ``\psi``, the tendency is

```math
\rho S_\psi = \dots + \kappa_{\max} \beta_{\mathrm{sponge}}
  \nabla_h \cdot (\rho \nabla_h \psi),
```

with turbulent diffusivity ``\kappa_{\max}``, default ``10^6`` in units of m²
s⁻¹, set by `kappa_2_sponge`. For the velocity, the same coefficient multiplies
the horizontal vector Laplacian defined in [Hyperdiffusion](hyperdiffusion.md);
horizontal density variations are neglected there, as in the hyperdiffusion
tendency.

Viscous damping of all variables absorbs waves without applying the zonal-mean
torque that Rayleigh damping of horizontal velocities produces
[Shepherd1996](@cite).

Energy is damped through the same two-piece decomposition that
[hyperdiffusion](hyperdiffusion.md) and the vertical diffusion use, rather than
through ``\rho e_{tot}`` directly: a dry-static-energy flux
``\rho \nabla_h s_d``, which applies in every configuration, plus a
water-enthalpy flux ``\rho h_{\mathrm{tot,cl}} \nabla_h q_t^{\mathrm{eff}}``
in moist configurations. Damping the two pieces separately keeps the sponge
consistent with the [enthalpy-based formulation](thermodynamics.md) used
elsewhere in the model.

## Rayleigh damping

Rayleigh damping relaxes a velocity component linearly toward zero,

```math
\boldsymbol{S}_u = \dots
  - \alpha_{\max} \beta_{\mathrm{sponge}}
    (\hat{\boldsymbol{k}} \cdot \boldsymbol{u}) \hat{\boldsymbol{k}} ,
```

applied to the vertical velocity with coefficient ``\alpha_{\max}``, default
1 s⁻¹, set by `alpha_rayleigh_w`.

Rayleigh damping is restricted to the vertical velocity for the same reason the
viscous sponge handles everything else: relaxing horizontal velocities toward
zero applies a zonal-mean torque, which distorts the upper-atmosphere
circulation [Shepherd1996, Jablonowski2011](@cite).

The implementation also offers damping of horizontal velocity and of tracers,
for configurations that need it:

| Coefficient         | Parameter               | Default | Applies to                               |
|:------------------- |:----------------------- | -------:|:---------------------------------------- |
| ``\alpha_w``        | `alpha_rayleigh_w`      | 1.0     | Vertical velocity ``u_3``                |
| ``\alpha_{uh}``     | `alpha_rayleigh_uh`     | 0.0     | Horizontal velocity ``\boldsymbol{u}_h`` |
| ``\alpha_{tracer}`` | `alpha_rayleigh_tracer` | 0.0     | TKE and the PROPHET subdomain scalars    |

Because the latter two default to zero, the default configuration damps only the
vertical velocity, as described above. Some PROPHET configurations enable a
small tracer damping. The tracer coefficient acts only on the turbulence kinetic
energy and on the PROPHET subdomain scalars, and for the latter the relaxation
is toward the grid-mean value rather than toward zero, so only the subgrid-scale
departure is damped.

## Interaction with timestepping

The Rayleigh damping of the vertical velocity is carried in the vertically
implicit part of the tendency split, so its damping timescale does not restrict
the timestep; this is why ``\alpha_{\max}`` of order 1 s⁻¹ works at timesteps
far longer than 1 s. The viscous sponge, which involves horizontal operators,
and the optional Rayleigh terms for horizontal velocity and tracers are
treated explicitly. See
[Discretization and Operators](discretization.md) for the split and
[Implicit Solver](implicit_solver.md) for the solve.

## Where this is implemented

| Concept          | Source                                                                                                                                                            |
|:---------------- |:----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Viscous sponge   | [src/parameterized_tendencies/sponge/viscous_sponge.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/sponge/viscous_sponge.jl)   |
| Rayleigh damping | [src/parameterized_tendencies/sponge/rayleigh_sponge.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/sponge/rayleigh_sponge.jl) |
| Model types      | [`ClimaAtmos.ViscousSponge`](@ref), [`ClimaAtmos.RayleighSponge`](@ref)                                                                                           |

Both sponges are selected with the `viscous_sponge` and `rayleigh_sponge`
configuration keys; see [Configuration Options](configuration_options.md).
