# Conservation Properties

ClimaAtmos conserves total energy, air mass, and water to floating-point
precision, without ad hoc fixers. It does so in moist atmospheres, in the
presence of subgrid-scale parameterizations, over complex topography, and in a
deep atmosphere [Yatunin2026](@cite).

Conservation is a design goal, and it constrains choices throughout the model.
This page collects those choices in one place and describes what is conserved,
what is not, and how to check.

## Why it matters

An energy leak of a fraction of a watt per square meter is small against the
roughly 240 W m⁻² of top-of-atmosphere radiation. Integrated over a century-long
simulation, it produces a temperature drift that cannot be distinguished from a
real climate signal [Thuburn2008](@cite).

Models that do not conserve by construction often add *energy fixers*
[Lauritzen2022, Jablonowski2011](@cite):
corrections that redistribute the imbalance to close the budget. A fixer hides
the error instead of removing it, and it obscures how energy moves through the
model. Here the discrete equations conserve to floating-point precision, so no
fixer is needed.

## Where conservation comes from

Conservation rests on four ingredients, and it fails if any one of them is
missing.

**Flux-conservative equations.** The governing equations for mass, energy, and
total water are written in flux form: their right-hand sides contain only
flux-divergence terms, with no non-conservative sources or sinks. The
domain-integrated quantities can then change only through fluxes across the
boundaries. This structure holds even for the subgrid-scale fluxes that
parameterizations contribute. See [Governing Equations](equations.md).

**Specific total energy as a prognostic variable.** The model prognoses
``e_{tot} = I + \Phi + \kappa + \kappa_{SGS}``, the specific total energy of
moist air, instead of a temperature-like variable. Total energy is an extensive
quantity satisfying a volume-integrated conservation law, so a flux-form
equation for it conserves it.

**A consistent thermodynamic formulation.** The internal energies, enthalpies,
and latent heats must be mutually consistent for the energy budget to close in
the presence of phase changes.
[Thermodynamics and the Working Fluid](thermodynamics.md) develops this,
including the reference-temperature invariance that any new energy flux must
respect.

**A mimetic discretization.** A discretization can break conservation that the
continuous equations have. The spectral element method's weak-form flux
divergences satisfy discrete analogues of the divergence and Stokes theorems;
direct stiffness summation preserves the discrete inner product; the vertical
averaging and difference operators satisfy discrete integration-by-parts and
averaging-by-parts identities; and the reconstructions between cell centers and
faces are chosen so that combinations of these operators satisfy discrete
conservation laws. See [Discretization and Operators](discretization.md).

## Consequences

Because momentum advection in vector-invariant form conserves kinetic energy and
vorticity globally, and because total energy is separately conserved, any
numerical conversion between kinetic and non-kinetic energy is performed
**solely** by the discretized pressure-gradient term and by the physical sources
and sinks. The advection scheme contributes none. Energy therefore moves between
the resolved scales and the thermodynamic reservoirs along one path, in moist
conditions as well as dry.

The subgrid-scale and numerical terms follow the same rule.
[Hyperdiffusion](hyperdiffusion.md) is written as flux divergences along
coordinate surfaces, so it redistributes without creating or destroying
anything; its enthalpy flux is decomposed to account for the energy carried by
the hyperdiffusive water flux, so total energy and water are conserved together.
The same decomposition governs the turbulent enthalpy flux and the
[sponge layer](sponge.md).

## What is not conserved

There is a trade-off. Potential temperature, which some models conserve
discretely along material trajectories, has no such guarantee here.

Potential temperature is conserved only for dry, adiabatic dynamics, and it is
not an extensive quantity satisfying a volume-integrated conservation law. Under
the moist, diabatic conditions of the real atmosphere, it is not conserved
physically either. Total energy is, so that is what the model conserves.

The effective moisture source in the mass continuity equation, which comes from
subgrid-scale water fluxes, is retained for generality. In Earth's atmosphere, it
is two or more orders of magnitude smaller than the other terms. It matters
where the condensable species is a major atmospheric constituent, such as carbon
dioxide on Mars.

## Checking conservation in a run

The model's test suite checks conservation, and you can measure it in any
simulation. Enable the conservation callback with the `check_conservation`
configuration key, then compare the first and last saved states:

```julia
import ClimaAtmos as CA

errors = CA.check_conservation(simulation)
```

This returns dimensionless relative errors:

  - `energy_conservation`: the change in atmospheric plus surface energy, less
    the net radiative input at the top, divided by the initial total energy.
  - `mass_conservation`: the change in total dry-plus-moist mass, divided by the
    initial mass.
  - `water_conservation`: the change in atmospheric plus surface water, divided
    by the final total atmospheric water. Zero for dry runs.

The surface terms matter: energy and water leave the atmosphere through surface
fluxes and precipitation, so a closed budget has to account for what the surface
receives. With a slab ocean the surface reservoir is explicit; otherwise the
accumulated surface fluxes stand in for it.

!!! note "Interpreting the numbers"

    These errors mean something only for setups whose boundary fluxes are all
    accounted for, and only when the run saved both endpoints. For the
    quantities the model conserves by construction, expect values near
    floating-point precision. A 32-bit run carries about seven significant
    decimal digits, so its floor is well above a 64-bit run's.

## Where this is implemented

| Concept              | Source                                                                                                 |
|:-------------------- |:------------------------------------------------------------------------------------------------------ |
| Conservation check   | [src/simulation/solve.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/simulation/solve.jl)    |
| Flux-form tendencies | [src/prognostic_equations/](https://github.com/CliMA/ClimaAtmos.jl/tree/main/src/prognostic_equations) |
| Conservation tests   | [test/](https://github.com/CliMA/ClimaAtmos.jl/tree/main/test)                                         |
