#####
##### Subsidence forcing
#####

import Thermodynamics as TD
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import ClimaCore.Operators as Operators

#####
##### No subsidence
#####

subsidence_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

#####
##### Subsidence
#####

subsidence!(ᶜρχₜ, ᶜρ, ᶠu³, ᶜχ, ::Val{:none}) =
    @. ᶜρχₜ -= ᶜρ * (ᶜsubdivᵥ(ᶠu³ * ᶠinterp(ᶜχ)) - ᶜχ * ᶜsubdivᵥ(ᶠu³))
subsidence!(ᶜρχₜ, ᶜρ, ᶠu³, ᶜχ, ::Val{:first_order}) =
    @. ᶜρχₜ -= ᶜρ * (ᶜsubdivᵥ(ᶠupwind1(ᶠu³, ᶜχ)) - ᶜχ * ᶜsubdivᵥ(ᶠu³))
subsidence!(ᶜρχₜ, ᶜρ, ᶠu³, ᶜχ, ::Val{:third_order}) =
    @. ᶜρχₜ -= ᶜρ * (ᶜsubdivᵥ(ᶠupwind3(ᶠu³, ᶜχ)) - ᶜχ * ᶜsubdivᵥ(ᶠu³)) # 3rd-order upwind ρ * (-w * ∂χ/∂z)


"""
    subsidence_tendency!(Yₜ, Y, p, t, subsidence_model::Subsidence)

Applies subsidence tendencies to total energy (`ρe_tot`), total specific humidity
(`ρq_tot`), and other moisture species (`ρq_liq`, `ρq_ice`) if a `NonEquilMoistModel`
is used.

The subsidence velocity profile `w_sub(z)` is obtained from `subsidence_model.prof`.
This profile is used to construct a face-valued vertical velocity field `ᶠsubsidence³`.
The `subsidence!` helper function is then called (currently with a first-order
upwind scheme) to compute and apply the vertical advective tendency for each relevant 
scalar quantity `χ`.

Arguments:
- `Yₜ`: The tendency state vector, modified in place.
- `Y`: The current state vector, used for density (`ρ`).
- `p`: Cache containing parameters, and the subsidence model object.
- `t`: Current simulation time.
- `subsidence`: The subsidence model object, containing the prescribed vertical
              velocity profile `Dᵥ`.
"""
function subsidence_tendency!(Yₜ, Y, p, t, subsidence::Subsidence)
    (; Dᵥ) = subsidence
    ᶜρ = Y.c.ρ
    (; moisture_model) = p.atmos
    subsidence_profile = p.atmos.subsidence.prof
    thermo_params = CAP.thermodynamics_params(p.params)

    ᶠz = Fields.coordinate_field(axes(Y.f)).z
    ᶠlg = Fields.local_geometry_field(Y.f)
    ᶠsubsidence³ = p.scratch.ᶠtemp_CT3
    @. ᶠsubsidence³ =
        subsidence_profile(ᶠz) * CT3(unit_basis_vector_data(CT3, ᶠlg))

    # LS Subsidence
    ᶜh_tot = @. lazy(TD.total_specific_enthalpy(thermo_params, ᶜts, specific(Y.c.ρe_tot, Y.c.ρ)))
    subsidence!(Yₜ.c.ρe_tot, Y.c.ρ, ᶠsubsidence³, ᶜh_tot, Val{:first_order}())
    subsidence!(
        Yₜ.c.ρq_tot,
        Y.c.ρ,
        ᶠsubsidence³,
        specific(Y.c.ρq_tot, Y.c.ρ),
        Val{:first_order}(),
    )
    if moisture_model isa NonEquilMoistModel
        subsidence!(
            Yₜ.c.ρq_liq,
            Y.c.ρ,
            ᶠsubsidence³,
            specific(Y.c.ρq_liq, Y.c.ρ),
            Val{:first_order}(),
        )
        subsidence!(
            Yₜ.c.ρq_ice,
            Y.c.ρ,
            ᶠsubsidence³,
            specific(Y.c.ρq_ice, Y.c.ρ),
            Val{:first_order}(),
        )
    end

    return nothing
end
