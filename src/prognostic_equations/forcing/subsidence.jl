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
    @. ᶜρχₜ -= ᶜρ * (ᶜsubdivᵥ(ᶠupwind3(ᶠu³, ᶜχ)) - ᶜχ * ᶜsubdivᵥ(ᶠu³))

function subsidence_tendency!(Yₜ, Y, p, t, ::Subsidence)
    (; moisture_model) = p.atmos
    subsidence_profile = p.atmos.subsidence.prof
    (; ᶜh_tot) = p.precomputed

    ᶠz = Fields.coordinate_field(axes(Y.f)).z
    ᶠlg = Fields.local_geometry_field(Y.f)
    ᶠsubsidence³ = p.scratch.ᶠtemp_CT3
    @. ᶠsubsidence³ =
        subsidence_profile(ᶠz) * CT3(unit_basis_vector_data(CT3, ᶠlg))

    # LS Subsidence
    subsidence!(Yₜ.c.ρe_tot, Y.c.ρ, ᶠsubsidence³, ᶜh_tot, Val{:first_order}())
    @. p.scratch.ᶜtemp_scalar_3 = Y.c.ρq_tot / Y.c.ρ
    q_tot = p.scratch.ᶜtemp_scalar_3
    subsidence!(Yₜ.c.ρq_tot, Y.c.ρ, ᶠsubsidence³, q_tot, Val{:first_order}())
    if moisture_model isa NonEquilMoistModel
        @. p.scratch.ᶜtemp_scalar_3 = Y.c.ρq_liq / Y.c.ρ
        q_liq = p.scratch.ᶜtemp_scalar_3
        subsidence!(
            Yₜ.c.ρq_liq,
            Y.c.ρ,
            ᶠsubsidence³,
            q_liq,
            Val{:first_order}(),
        )
        @. p.scratch.ᶜtemp_scalar_3 = Y.c.ρq_ice / Y.c.ρ
        q_ice = p.scratch.ᶜtemp_scalar_3
        subsidence!(
            Yₜ.c.ρq_ice,
            Y.c.ρ,
            ᶠsubsidence³,
            q_ice,
            Val{:first_order}(),
        )
    end

    return nothing
end
