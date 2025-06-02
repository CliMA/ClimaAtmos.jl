#####
##### Apply prescribed large-scale advection tendencies for total 
##### specific humidity and total energy
##### Apply prescribed large-scale advection tendencies for total 
##### specific humidity and total energy
#####

import Thermodynamics as TD
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields

"""
    large_scale_advection_tendency_ρq_tot(ᶜρ, thermo_params, ᶜts, t, ls_adv)

Computes the tendency of the total water content (`ρq_tot`) due to prescribed
large-scale advection of total specific humidity (`q_tot`).

If `ls_adv` is not a `LargeScaleAdvection` object (i.e., large-scale advection
is not active), it returns `NullBroadcasted()`, resulting in no tendency.

Otherwise, it retrieves a profile function `prof_dqtdt` from `ls_adv`. This
function provides the prescribed advective tendency of `q_tot` (i.e.,
``(\\partial q_{tot}/\\partial t)_{LS_adv}``) as a function of the thermodynamic
state (`ᶜts`), time (`t`), and height (`ᶜz`). The final tendency for `ρq_tot`
is then computed as ``ᶜρ * (\\partial q_{tot}/\\partial t)_{LS_adv}``.

Arguments:
- `ᶜρ`: Cell-center air density field.
- `thermo_params`: Thermodynamic parameters.
- `ᶜts`: Cell-center thermodynamic state field.
- `t`: Current simulation time.
- `ls_adv`: `LargeScaleAdvection` object containing profile functions for tendencies,
            or another type if large-scale advection is inactive.

Returns:
- A `ClimaCore.Fields.Field`, or a lazy broadcast over ClimaCore Fields,
  representing the tendency `∂(ρq_tot)/∂t` due to
  large-scale advection of `q_tot`, or `NullBroadcasted` if inactive.
"""
"""
    large_scale_advection_tendency_ρq_tot(ᶜρ, thermo_params, ᶜts, t, ls_adv)

Computes the tendency of the total water content (`ρq_tot`) due to prescribed
large-scale advection of total specific humidity (`q_tot`).

If `ls_adv` is not a `LargeScaleAdvection` object (i.e., large-scale advection
is not active), it returns `NullBroadcasted()`, resulting in no tendency.

Otherwise, it retrieves a profile function `prof_dqtdt` from `ls_adv`. This
function provides the prescribed advective tendency of `q_tot` (i.e.,
``(\\partial q_{tot}/\\partial t)_{LS_adv}``) as a function of the thermodynamic
state (`ᶜts`), time (`t`), and height (`ᶜz`). The final tendency for `ρq_tot`
is then computed as ``ᶜρ * (\\partial q_{tot}/\\partial t)_{LS_adv}``.

Arguments:
- `ᶜρ`: Cell-center air density field.
- `thermo_params`: Thermodynamic parameters.
- `ᶜts`: Cell-center thermodynamic state field.
- `t`: Current simulation time.
- `ls_adv`: `LargeScaleAdvection` object containing profile functions for tendencies,
            or another type if large-scale advection is inactive.

Returns:
- A `ClimaCore.Fields.Field`, or a lazy broadcast over ClimaCore Fields,
  representing the tendency `∂(ρq_tot)/∂t` due to
  large-scale advection of `q_tot`, or `NullBroadcasted` if inactive.
"""
function large_scale_advection_tendency_ρq_tot(
    ᶜρ,
    thermo_params,
    ᶜts,
    t,
    ls_adv,
)
    ls_adv isa LargeScaleAdvection || return NullBroadcasted()
    (; prof_dTdt, prof_dqtdt) = ls_adv
    ᶜz = Fields.coordinate_field(axes(ᶜρ)).z
    ᶜdqtdt_hadv = @. lazy(prof_dqtdt(thermo_params, ᶜts, t, ᶜz))
    return @. lazy(ᶜρ * ᶜdqtdt_hadv)
end

"""
    large_scale_advection_tendency_ρe_tot(ᶜρ, thermo_params, ᶜts, t, ls_adv)

Computes the tendency of total energy (`ρe_tot`) due to prescribed large-scale
advection of temperature (`T`) and total specific humidity (`q_tot`).

If `ls_adv` is not a `LargeScaleAdvection` object, it returns `NullBroadcasted()`.

Otherwise, it retrieves profile functions `prof_dTdt` and `prof_dqtdt` from `ls_adv`,
which provide the prescribed advective tendencies ``(\\partial T/\\partial t)_{LS_adv}``
and ``(\\partial q_{tot}/\\partial t)_{LS_adv}``, respectively.
The tendency for `ρe_tot` is then computed based on these, using the formula:
  `ρ * (cv_m * (∂T/∂t)_{LS_adv} + e_int_vapor(T) * (∂q_{tot}/∂t)_{LS_adv})`
where `cv_m` is the specific heat at constant volume for the moist air mixture,
and `e_int_vapor(T)` is the specific internal energy of water vapor at temperature `T`.
This conversion accounts for the change in internal energy due to changes in
temperature and the phase composition (assuming changes in `q_tot` primarily affect
vapor for this energy calculation).

Arguments:
- `ᶜρ`: Cell-center air density field.
- `thermo_params`: Thermodynamic parameters.
- `ᶜts`: Cell-center thermodynamic state field.
- `t`: Current simulation time.
- `ls_adv`: `LargeScaleAdvection` object containing profile functions for tendencies,
            or another type if large-scale advection is inactive.

Returns:
- A `ClimaCore.Fields.Field`, or a lazy broadcast over ClimaCore Fields,
  representing the tendency `∂(ρe_tot)/∂t` due to
  large-scale advection of `T` and `q_tot`, or `NullBroadcasted` if inactive.
"""
"""
    large_scale_advection_tendency_ρe_tot(ᶜρ, thermo_params, ᶜts, t, ls_adv)

Computes the tendency of total energy (`ρe_tot`) due to prescribed large-scale
advection of temperature (`T`) and total specific humidity (`q_tot`).

If `ls_adv` is not a `LargeScaleAdvection` object, it returns `NullBroadcasted()`.

Otherwise, it retrieves profile functions `prof_dTdt` and `prof_dqtdt` from `ls_adv`,
which provide the prescribed advective tendencies ``(\\partial T/\\partial t)_{LS_adv}``
and ``(\\partial q_{tot}/\\partial t)_{LS_adv}``, respectively.
The tendency for `ρe_tot` is then computed based on these, using the formula:
  `ρ * (cv_m * (∂T/∂t)_{LS_adv} + e_int_vapor(T) * (∂q_{tot}/∂t)_{LS_adv})`
where `cv_m` is the specific heat at constant volume for the moist air mixture,
and `e_int_vapor(T)` is the specific internal energy of water vapor at temperature `T`.
This conversion accounts for the change in internal energy due to changes in
temperature and the phase composition (assuming changes in `q_tot` primarily affect
vapor for this energy calculation).

Arguments:
- `ᶜρ`: Cell-center air density field.
- `thermo_params`: Thermodynamic parameters.
- `ᶜts`: Cell-center thermodynamic state field.
- `t`: Current simulation time.
- `ls_adv`: `LargeScaleAdvection` object containing profile functions for tendencies,
            or another type if large-scale advection is inactive.

Returns:
- A `ClimaCore.Fields.Field`, or a lazy broadcast over ClimaCore Fields,
  representing the tendency `∂(ρe_tot)/∂t` due to
  large-scale advection of `T` and `q_tot`, or `NullBroadcasted` if inactive.
"""
function large_scale_advection_tendency_ρe_tot(
    ᶜρ,
    thermo_params,
    ᶜts,
    t,
    ls_adv,
)
    ls_adv isa LargeScaleAdvection || return NullBroadcasted()
    (; prof_dTdt, prof_dqtdt) = ls_adv
    z = Fields.coordinate_field(axes(ᶜρ)).z
    ᶜdTdt_hadv = @. lazy(prof_dTdt(thermo_params, ᶜts, t, z))
    ᶜdqtdt_hadv = @. lazy(prof_dqtdt(thermo_params, ᶜts, t, z))

    # Moisture advection term does not contain potential energy because 
    # it's just horizontal advection of specific humidity

    # Moisture advection term does not contain potential energy because 
    # it's just horizontal advection of specific humidity
    return @. lazy(
        ᶜρ * (
            TD.cv_m(thermo_params, ᶜts) * ᶜdTdt_hadv +
            TD.internal_energy_vapor(thermo_params, ᶜts) * ᶜdqtdt_hadv
            TD.internal_energy_vapor(thermo_params, ᶜts) * ᶜdqtdt_hadv
        ),
    )
end
