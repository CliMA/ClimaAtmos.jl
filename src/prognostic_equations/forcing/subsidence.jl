#####
##### Subsidence forcing
#####

import Thermodynamics as TD
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import ClimaCore.Operators as Operators

subsidence_cache(Y, atmos::AtmosModel) = subsidence_cache(Y, atmos.subsidence)

#####
##### No subsidence
#####

subsidence_cache(Y, ::Nothing) = (;)
subsidence_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

#####
##### Subsidence
#####

function subsidence_cache(Y, ::Subsidence)
    FT = Spaces.undertype(axes(Y.f))
    return (;
        ᶠsubsidence = similar(Y.f, FT), # TODO: fix types
    )
end

subsidence!(ᶜρχₜ, ᶜρ, ᶠu³, ᶜχ, ::Val{:none}) =
    @. ᶜρχₜ -= ᶜρ * (ᶜsubdivᵥ(ᶠu³ * ᶠinterp(ᶜχ)) - ᶜχ * ᶜsubdivᵥ(ᶠu³))
subsidence!(ᶜρχₜ, ᶜρ, ᶠu³, ᶜχ, ::Val{:first_order}) =
    @. ᶜρχₜ -= ᶜρ * (ᶜsubdivᵥ(ᶠupwind1(ᶠu³, ᶜχ)) - ᶜχ * ᶜsubdivᵥ(ᶠu³))
subsidence!(ᶜρχₜ, ᶜρ, ᶠu³, ᶜχ, ::Val{:third_order}) =
    @. ᶜρχₜ -= ᶜρ * (ᶜsubdivᵥ(ᶠupwind3(ᶠu³, ᶜχ)) - ᶜχ * ᶜsubdivᵥ(ᶠu³))

function subsidence_tendency!(Yₜ, Y, p, t, ::Subsidence)
    (; moisture_model) = p.atmos
    subsidence_profile = p.atmos.subsidence.prof
    (; ᶠsubsidence) = p.subsidence
    (; ᶜh_tot, ᶜspecific) = p.precomputed

    ᶠz = Fields.coordinate_field(axes(ᶠsubsidence)).z
    ᶠlg = Fields.local_geometry_field(Y.f)
    @. ᶠsubsidence = subsidence_profile(ᶠz)
    ᶠsubsidence³ = p.scratch.ᶠtemp_CT3
    @. ᶠsubsidence³ = ᶠsubsidence * CT3(unit_basis_vector_data(CT3, ᶠlg))

    # LS Subsidence
    subsidence!(Yₜ.c.ρe_tot, Y.c.ρ, ᶠsubsidence³, ᶜh_tot, Val{:first_order}())
    subsidence!(
        Yₜ.c.ρq_tot,
        Y.c.ρ,
        ᶠsubsidence³,
        ᶜspecific.q_tot,
        Val{:first_order}(),
    )
    if moisture_model isa NonEquilMoistModel
        subsidence!(
            Yₜ.c.ρq_liq,
            Y.c.ρ,
            ᶠsubsidence³,
            ᶜspecific.q_liq,
            Val{:first_order}(),
        )
        subsidence!(
            Yₜ.c.ρq_ice,
            Y.c.ρ,
            ᶠsubsidence³,
            ᶜspecific.q_ice,
            Val{:first_order}(),
        )
    end

    return nothing
end
