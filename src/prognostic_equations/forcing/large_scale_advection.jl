#####
##### Apply prescribed large-scale advection tendencies for total
##### specific humidity and total energy
#####

import Thermodynamics as TD
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields

"""
    large_scale_advection_tendency_ρq_tot(ᶜρ, thermo_params, ᶜT, ᶜp, q_tot, q_liq, q_ice, t, ls_adv)

Return the `ρq_tot` tendency from prescribed large-scale advection of total
specific humidity.

The profile function `ls_adv.prof_dqtdt(thermo_params, ᶜp, t, ᶜz)` supplies the
prescribed advective tendency of `q_tot` [1/s], which is multiplied by the density
to give the `ρq_tot` tendency.

# Arguments

  - `ᶜρ`: Cell-center air density [kg/m³].
  - `thermo_params`: Thermodynamic parameters, passed to the profile function.
  - `ᶜT`: Cell-center temperature; unused by this tendency [K].
  - `ᶜp`: Cell-center pressure [Pa].
  - `q_tot`, `q_liq`, `q_ice`: Specific humidities; unused by this tendency [kg/kg].
  - `t`: Current simulation time [s].
  - `ls_adv`: A `LargeScaleAdvection` object holding the profile functions, or any
    other value when large-scale advection is inactive.

# Returns

A lazy broadcast with `∂(ρq_tot)/∂t` [kg/m³/s], or a `NullBroadcasted()` when
`ls_adv` is not a `LargeScaleAdvection`. The caller (`additional_tendency!`) adds
it to `Yₜ.c.ρq_tot`.

The signature is shared with `large_scale_advection_tendency_ρe_tot` so both can be
called with one argument tuple.
"""
function large_scale_advection_tendency_ρq_tot(
    ᶜρ,
    thermo_params,
    ᶜT,
    ᶜp,
    q_tot,
    q_liq,
    q_ice,
    t,
    ls_adv,
)
    ls_adv isa LargeScaleAdvection || return NullBroadcasted()
    (; prof_dTdt, prof_dqtdt) = ls_adv
    ᶜz = Fields.coordinate_field(axes(ᶜρ)).z
    ᶜdqtdt_hadv = @. lazy(prof_dqtdt(thermo_params, ᶜp, t, ᶜz))
    return @. lazy(ᶜρ * ᶜdqtdt_hadv)
end

"""
    large_scale_advection_tendency_ρe_tot(ᶜρ, thermo_params, ᶜT, ᶜp, q_tot, q_liq, q_ice, t, ls_adv)

Return the `ρe_tot` tendency from prescribed large-scale advection of temperature
and total specific humidity.

The profile functions `ls_adv.prof_dTdt` and `ls_adv.prof_dqtdt`, both evaluated
as `prof(thermo_params, ᶜp, t, z)`, supply the prescribed advective tendencies of
`T` [K/s] and `q_tot` [1/s]. They are converted to an energy tendency with

```math
ρ \\left( c_{v,m} \\, \\partial_t T + e_{int,vap}(T) \\, \\partial_t q_{tot} \\right)
```

where `c_{v,m}` is the isochoric specific heat of the moist mixture and
`e_{int,vap}` the specific internal energy of water vapor. No potential-energy
term appears, because this represents horizontal advection at constant height.

# Arguments

  - `ᶜρ`: Cell-center air density [kg/m³].
  - `thermo_params`: Thermodynamic parameters.
  - `ᶜT`: Cell-center temperature [K].
  - `ᶜp`: Cell-center pressure [Pa].
  - `q_tot`, `q_liq`, `q_ice`: Specific humidities of total water, cloud liquid, and
    cloud ice [kg/kg].
  - `t`: Current simulation time [s].
  - `ls_adv`: A `LargeScaleAdvection` object holding the profile functions, or any
    other value when large-scale advection is inactive.

# Returns

A lazy broadcast with `∂(ρe_tot)/∂t` [W/m³], or a `NullBroadcasted()` when
`ls_adv` is not a `LargeScaleAdvection`. The caller (`additional_tendency!`) adds
it to `Yₜ.c.ρe_tot`.
"""
function large_scale_advection_tendency_ρe_tot(
    ᶜρ,
    thermo_params,
    ᶜT,
    ᶜp,
    q_tot,
    q_liq,
    q_ice,
    t,
    ls_adv,
)
    ls_adv isa LargeScaleAdvection || return NullBroadcasted()
    (; prof_dTdt, prof_dqtdt) = ls_adv
    z = Fields.coordinate_field(axes(ᶜρ)).z
    ᶜdTdt_hadv = @. lazy(prof_dTdt(thermo_params, ᶜp, t, z))
    ᶜdqtdt_hadv = @. lazy(prof_dqtdt(thermo_params, ᶜp, t, z))

    # Moisture advection term does not contain potential energy because
    # it's just horizontal advection of specific humidity
    return @. lazy(
        ᶜρ * (
            TD.cv_m(thermo_params, q_tot, q_liq, q_ice) * ᶜdTdt_hadv +
            TD.internal_energy_vapor(thermo_params, ᶜT) * ᶜdqtdt_hadv
        ),
    )
end
