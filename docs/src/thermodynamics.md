# Thermodynamics and the Working Fluid

The thermodynamic formulation underpins the model's conservation properties.
ClimaAtmos prognoses the specific total energy of moist air, so the definitions
of internal energy, enthalpy, and latent heat determine whether energy budgets
close. This page sets out that formulation, following [Yatunin2026](@cite).

The formulation is implemented in
[Thermodynamics.jl](https://clima.github.io/Thermodynamics.jl/stable/), shared
with the other CliMA components, and the constants are defined in
[ClimaParams.jl](https://github.com/CliMA/ClimaParams.jl). This page explains
the assumptions and the reasoning; consult the Thermodynamics.jl documentation
for the function-level interface.

## The working fluid

The working fluid is moist air: a mixture of dry air and water vapor, both
treated as ideal gases, together with condensed water — suspended in clouds and
falling as precipitation.

All constituents are in **thermal** equilibrium, at the same temperature.
Condensates may sediment or fall, but they carry the air's temperature. The
model therefore neglects temperature differences between condensate and ambient
air, such as those from evaporative cooling of raindrops, when computing fluid
energetics. Microphysical process rates are unaffected.

Constituents need not be in full **thermodynamic** equilibrium.
Out-of-equilibrium phases such as supercooled liquid are therefore permitted,
and the condensed phases can carry their own prognostic equations with
microphysics schemes in which non-equilibrium phases coexist. Suspended
condensate and precipitation are part of the working fluid, which suits bulk
microphysics schemes that represent these components through moments of a
particle size distribution.

Composition is described by mass fractions: dry air ``q_d``, water vapor
``q_v``, liquid ``q_l``, and ice ``q_i``. The condensate fraction is
``q_c = q_l + q_i``, and the total water specific humidity is
``q_t = q_v + q_c``. Since these account for every component, ``q_t + q_d = 1``.

!!! note "No small-humidity assumption"

    The formulation nowhere assumes that specific humidities are small, so the
    same equations apply to planetary atmospheres in which a condensable species
    is a major constituent. On Mars, for example, condensation of carbon dioxide
    affects air mass and dynamics.

Bulk microphysics schemes often split the condensate further into suspended
cloud condensate and precipitation, ``q_l = q_l^{cl} + q_l^{pr}`` and ``q_i = q_i^{cl} + q_i^{pr}``. The equations of motion allow for that distinction; see
[Microphysics](microphysics.md).

## Equation of state

The pressure of the working fluid is the sum of the partial pressures of dry air
and water vapor. The volume — but not the mass — of the condensed phases is
neglected, because the specific volume of condensate is smaller than that of the
gas phases by a factor of about ``10^3``. The equation of state is then

```math
p = \rho R_m T,
```

with the specific mixture gas constant of moist air

```math
R_m(q) = R_d (1 - q_t) + R_v q_v
       = R_d \left[ 1 + (\epsilon_{dv} - 1) q_t - \epsilon_{dv} q_c \right],
```

which depends on the specific humidities through the ratio of the gas constants
of water vapor and dry air, ``\epsilon_{dv} = R_v / R_d \approx 1.61``.

## A calorically perfect fluid

The primary thermodynamic assumption is that the fluid is **calorically
perfect**: the isochoric specific heat capacities of the constituents are
constants. The isobaric capacities are then constant too, and the capacities of
moist air are the mass-weighted sums of those of its constituents,

```math
c_{vm}(q) = c_{vd} + (c_{vv} - c_{vd}) q_t + (c_{vl} - c_{vv}) q_l
          + (c_{vi} - c_{vv}) q_i,
```

with the standard relation ``c_{pm}(q) = c_{vm}(q) + R_m(q)`` carrying over from
the constituents to the mixture. Real specific heat capacities vary slightly
with temperature; treating them as constant introduces an error below 1% for dry
air and at most a few percent for the water phases.

The assumption has a consequence for latent heats [Ambaum2020](@cite).
Kirchhoff's relation, ``dL/dT = \Delta c_p``, integrates under constant specific heats to a latent heat that
is **linear** in temperature,

```math
L(T) = L_0 + \Delta c_p (T - T_0),
```

for a reference temperature ``T_0`` and the latent heat ``L_0`` at that
temperature. This applies to vaporization, fusion, and sublimation, and the
expected relation ``L_s(T) = L_v(T) + L_f(T)`` follows.

## Energies and enthalpies

Specific internal energies of the constituents are referenced to ``T_0``
[Romps2008](@cite). Two reference values may be chosen independently, because
dry air and water cannot be converted into one another: the reference specific
internal energy of liquid water is set to zero and that of dry air to ``-R_d T_0``, which simplifies the later enthalpy and enthalpy-flux expressions. The
internal energy of moist air is the mass-weighted sum,

```math
I(T, q) = c_{vm}(q) (T - T_0) + (q_t - q_c) I_{v,0} - q_i I_{i,0}
        - (1 - q_t) R_d T_0 ,
```

which can be inverted for temperature. That inversion is how the model recovers
temperature when specific internal energy and the specific humidities are the
primary thermodynamic state variables.

Specific enthalpies follow by adding ``p_\mu / \rho_\mu`` to each constituent's
internal energy, neglecting the specific volumes of the condensed phases:

```math
h(T, q) = c_{pm}(q) (T - T_0) + (q_t - q_c) L_{v,0} - q_i L_{f,0}
        = I(T, q) + R_m T .
```

Enthalpy is central to describing transport, appearing in both the grid-scale
and the subgrid-scale flux terms of the [governing equations](equations.md).

## Saturation vapor pressure

The Clausius–Clapeyron relation, combined with the linear latent heat above,
integrates to a closed-form expression for the saturation vapor pressure — the
Rankine–Kirchhoff approximation [Duarte2014, Romps2021](@cite). For a mixture of liquid and ice out of
thermodynamic equilibrium, as in mixed-phase clouds, a thermodynamically
consistent saturation vapor pressure uses the liquid-fraction-weighted average
[Pressel2015](@cite),
``L = \lambda_f L_v + (1 - \lambda_f) L_s`` with
``\lambda_f = q_l / q_c``.

The expression is invariant to the choice of reference temperature, provided the
reference latent heats are shifted with it. With the triple point of water as
the reference, the saturation vapor pressure over liquid between 248 and 325 K
lies within 0.4% of measured values, and the ratio of saturation vapor pressure
over ice to that over liquid between 233 and 273 K within 0.6%
[Yatunin2026](@cite).

## Optional local thermodynamic equilibrium

The model can optionally assume local thermodynamic equilibrium for cloudy air —
excluding falling precipitation — as many atmosphere models do. Gibbs' phase
rule then implies that three thermodynamic state variables suffice to determine
the partitioning into water phases, so the suspended condensate specific
humidities are obtained by saturation adjustment from the cloudy-air total
humidity, density, and internal energy, while the precipitation humidities
remain prognostic.

Strict equilibrium requires the liquid fraction to be a Heaviside function of
temperature. Supercooled liquid is out of equilibrium and exists between the
homogeneous ice nucleation temperature and the freezing temperature; a ramp
function between the two is often used to represent it [Kaul2015](@cite). That ramp is available
in the code, and it remains an approximation, since out-of-equilibrium phases
depend on the history of the air mass as well as on its thermodynamic state. By
default the model gives the condensed phases their own prognostic equations
instead. See [Microphysics](microphysics.md) for the available schemes.

## Reference temperature invariance

The reference temperature ``T_0`` is arbitrary, and the model's physics must not
depend on it. The formulation is constructed so that it does not: a shift ``T_0 \to T_0 + \delta T_0``, with the reference latent heats shifted consistently by
Kirchhoff's relation, offsets each constituent's specific internal energy and
enthalpy by a constant.

The offset for dry air differs from that for water, but the offsets for all
three water phases are *identical*. The extra terms in the total energy equation
therefore sum to zero through the total water conservation law, leaving the
dynamics unchanged. The same holds for the surface enthalpy flux boundary
condition. What remains of the dependence on ``T_0`` is the accuracy of the
linear approximation to the latent heats, which is why the triple point of
water, a value typical for the atmosphere, is the choice made here.

This invariance gives a check on any new energy flux added to the model: a flux
that breaks it is not thermodynamically consistent, and will not conserve energy
(see [Conservation Properties](conservation.md)). The subgrid-scale enthalpy
flux decomposition in [Hyperdiffusion](hyperdiffusion.md) and in the vertical
diffusion scheme follows from this requirement.

## Where this is implemented

| Concept                                   | Source                                                                                                                                   |
|:----------------------------------------- |:---------------------------------------------------------------------------------------------------------------------------------------- |
| Thermodynamic state and functions         | [Thermodynamics.jl](https://clima.github.io/Thermodynamics.jl/stable/)                                                                   |
| Constants and parameter values            | [ClimaParams.jl](https://github.com/CliMA/ClimaParams.jl)                                                                                |
| Parameter struct and accessors            | [src/parameters/Parameters.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameters/Parameters.jl)                            |
| Saturation adjustment and cloud diagnosis | [src/parameterized_tendencies/microphysics/](https://github.com/CliMA/ClimaAtmos.jl/tree/main/src/parameterized_tendencies/microphysics) |
| Thermodynamic state construction          | [src/cache/precomputed_quantities.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/cache/precomputed_quantities.jl)              |

The choice between equilibrium and non-equilibrium condensate is a configuration
option; see [Configuration Options](configuration_options.md) and
[Microphysics](microphysics.md).
