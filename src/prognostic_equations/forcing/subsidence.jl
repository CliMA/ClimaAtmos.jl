#####
##### Tendencies due to prescribed subsidence
#####

import Thermodynamics as TD
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import ClimaCore.Operators as Operators
import ClimaCore.Geometry as Geometry

"""
    subsidence!(ᶜρχₜ, ᶜρ, ᶠu³, ᶜχ, scheme::Val)

Computes the tendency contribution to a density-weighted scalar `ρχ` due to
subsidence (vertical advection by a prescribed large-scale vertical velocity `ᶠu³`).

This function is dispatched based on the `scheme` argument to use different
numerical methods for reconstructing the advective flux `wχ` at cell faces:
- `::Val{:none}`: Uses a centered reconstruction (`ᶠu³ * ᶠinterp(ᶜχ)`).
- `::Val{:first_order}`: Uses a first-order upwind reconstruction (`ᶠupwind1(ᶠu³, ᶜχ)`).
- `::Val{:third_order}`: Uses a third-order upwind reconstruction (`ᶠupwind3(ᶠu³, ᶜχ)`).

The formulation `ᶜρ * (ᶜsubdivᵥ(Flux) - ᶜχ * ᶜsubdivᵥ(ᶠu³))` is equivalent to
`ᶜρ * (ᶠu³ ⋅ ∇ᶜχ)`, implementing the advective form. The result is subtracted
from `ᶜρχₜ`, effectively adding `ρ * (-ᶠu³ ⋅ ∇ᶜχ)` to it.

Arguments:
- `ᶜρχₜ`: Field for the tendency of the density-weighted scalar `ρχ`, modified in place.
- `ᶜρ`: Cell-center density field.
- `ᶠu³`: Face-valued field of prescribed vertical velocity (subsidence velocity `w`).
        Typically, `w < 0` for subsidence in an upward `z` coordinate.
- `ᶜχ`: Cell-center field of the specific scalar quantity `χ` being advected.
- `scheme`: A `Val` type specifying the advection scheme (e.g., `Val{:first_order}()`).
"""
subsidence!(ᶜρχₜ, ᶜρ, ᶠu³, ᶜχ, ::Val{:none}) =
    @. ᶜρχₜ -= ᶜρ * (ᶜsubdivᵥ(ᶠu³ * ᶠinterp(ᶜχ)) - ᶜχ * ᶜsubdivᵥ(ᶠu³)) # Centered difference ρ * (-w * ∂χ/∂z)
subsidence!(ᶜρχₜ, ᶜρ, ᶠu³, ᶜχ, ::Val{:first_order}) =
    @. ᶜρχₜ -= ᶜρ * (ᶜsubdivᵥ(ᶠupwind1(ᶠu³, ᶜχ)) - ᶜχ * ᶜsubdivᵥ(ᶠu³)) # 1st-order upwind ρ * (-w * ∂χ/∂z)
subsidence!(ᶜρχₜ, ᶜρ, ᶠu³, ᶜχ, ::Val{:third_order}) =
    @. ᶜρχₜ -= ᶜρ * (ᶜsubdivᵥ(ᶠupwind3(ᶠu³, ᶜχ)) - ᶜχ * ᶜsubdivᵥ(ᶠu³)) # 3rd-order upwind ρ * (-w * ∂χ/∂z)

function subsidence_velocity_field(ᶠz, ᶠlg, subsidence_profile)
    return @. lazy(
        subsidence_profile(ᶠz) * CT3(unit_basis_vector_data(CT3, ᶠlg)),
    )
end

"""
    subsidence_tendency_ρe_tot,
    subsidence_tendency_ρq_tot,
    subsidence_tendency_ρq_liq,
    subsidence_tendency_ρq_ice

Lazy subsidence tendency functions. Each computes the tendency for the respective variable using
first-order upwind advection. Returns `NullBroadcasted()` if `subsidence isa Nothing`.
All functions take `ᶜρ`, the scalar field (e.g., `ᶜh_tot`, `ᶜq_tot`), `ᶠsubsidence³`, and `subsidence`.
"""
function subsidence_tendency_ρe_tot(ᶜρ, ᶜh_tot, ᶠsubsidence³, subsidence)
    subsidence isa Nothing && return NullBroadcasted()
    return @. lazy(
        -ᶜρ * (ᶜsubdivᵥ(ᶠupwind1(ᶠsubsidence³, ᶜh_tot)) - ᶜh_tot * ᶜsubdivᵥ(ᶠsubsidence³)),
    )
end

function subsidence_tendency_ρq_tot(ᶜρ, ᶜq_tot, ᶠsubsidence³, subsidence)
    subsidence isa Nothing && return NullBroadcasted()
    return @. lazy(
        -ᶜρ * (ᶜsubdivᵥ(ᶠupwind1(ᶠsubsidence³, ᶜq_tot)) - ᶜq_tot * ᶜsubdivᵥ(ᶠsubsidence³)),
    )
end

function subsidence_tendency_ρq_liq(ᶜρ, ᶜq_liq, ᶠsubsidence³, subsidence)
    subsidence isa Nothing && return NullBroadcasted()
    return @. lazy(
        -ᶜρ * (ᶜsubdivᵥ(ᶠupwind1(ᶠsubsidence³, ᶜq_liq)) - ᶜq_liq * ᶜsubdivᵥ(ᶠsubsidence³)),
    )
end

function subsidence_tendency_ρq_ice(ᶜρ, ᶜq_ice, ᶠsubsidence³, subsidence)
    subsidence isa Nothing && return NullBroadcasted()
    return @. lazy(
        -ᶜρ * (ᶜsubdivᵥ(ᶠupwind1(ᶠsubsidence³, ᶜq_ice)) - ᶜq_ice * ᶜsubdivᵥ(ᶠsubsidence³)),
    )
end

"""
    subsidence_tendency!(Yₜ, Y, p, t, subsidence)

Applies subsidence tendencies to total energy (`ρe_tot`), total specific humidity
(`ρq_tot`), and other moisture species (`ρq_liq`, `ρq_ice`) if a `NonEquilMoistModel`
is used. Uses lazy tendency functions internally.

Arguments:
- `Yₜ`: The tendency state vector, modified in place.
- `Y`: The current state vector.
- `p`: Cache containing parameters, precomputed fields, and atmospheric model configurations.
- `t`: Current simulation time.
- `subsidence`: A `Subsidence` object or `Nothing`. If `Nothing`, no tendency is applied.
"""
subsidence_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing
function subsidence_tendency!(Yₜ, Y, p, t, ::Subsidence)
    (; moisture_model) = p.atmos
    subsidence_profile = p.atmos.subsidence.prof
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜts) = p.precomputed

    ᶠz = Fields.coordinate_field(axes(Y.f)).z
    ᶠlg = Fields.local_geometry_field(Y.f)
    ᶠsubsidence³ = p.scratch.ᶠtemp_CT3
    @. ᶠsubsidence³ = subsidence_velocity_field(ᶠz, ᶠlg, subsidence_profile)

    ᶜρ = Y.c.ρ
    ᶜh_tot = @. lazy(
        TD.total_specific_enthalpy(
            thermo_params,
            ᶜts,
            specific(Y.c.ρe_tot, ᶜρ),
        ),
    )
    @. Yₜ.c.ρe_tot += subsidence_tendency_ρe_tot(ᶜρ, ᶜh_tot, ᶠsubsidence³, p.atmos.subsidence)

    ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, ᶜρ))
    @. Yₜ.c.ρq_tot += subsidence_tendency_ρq_tot(ᶜρ, ᶜq_tot, ᶠsubsidence³, p.atmos.subsidence)

    if moisture_model isa NonEquilMoistModel
        ᶜq_liq = @. lazy(specific(Y.c.ρq_liq, ᶜρ))
        @. Yₜ.c.ρq_liq += subsidence_tendency_ρq_liq(ᶜρ, ᶜq_liq, ᶠsubsidence³, p.atmos.subsidence)

        ᶜq_ice = @. lazy(specific(Y.c.ρq_ice, ᶜρ))
        @. Yₜ.c.ρq_ice += subsidence_tendency_ρq_ice(ᶜρ, ᶜq_ice, ᶠsubsidence³, p.atmos.subsidence)
    end

    return nothing
end
