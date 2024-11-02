#####
##### Viscous sponge
#####

import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore.Spaces as Spaces

viscous_sponge_cache(Y, atmos::AtmosModel) =
    viscous_sponge_cache(Y, atmos.viscous_sponge)

viscous_sponge_cache(Y, ::Nothing) = (;)
viscous_sponge_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

function viscous_sponge_cache(Y, viscous_sponge::ViscousSponge)
    (; κ₂, zd) = viscous_sponge
    FT = Spaces.undertype(axes(Y.c))
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠz = Fields.coordinate_field(Y.f).z
    ᶜαₘ = @. ifelse(ᶜz > zd, κ₂, FT(0))
    ᶠαₘ = @. ifelse(ᶠz > zd, κ₂, FT(0))
    zmax = maximum(ᶠz)
    ᶜβ_viscous = @. ᶜαₘ * sin(FT(π) / 2 * (ᶜz - zd) / (zmax - zd))^2
    ᶠβ_viscous = @. ᶠαₘ * sin(FT(π) / 2 * (ᶠz - zd) / (zmax - zd))^2
    return (; ᶜβ_viscous, ᶠβ_viscous)
end

function viscous_sponge_tendency!(Yₜ, Y, p, t, ::ViscousSponge)
    (; ᶜβ_viscous, ᶠβ_viscous) = p.viscous_sponge
    (; ᶜh_tot, ᶜspecific) = p.precomputed
    ᶜuₕ = Y.c.uₕ

    ᶜdivₕ_uₕ = @. p.scratch.ᶜtemp_scalar = wdivₕ(ᶜuₕ)
    Spaces.weighted_dss!(ᶜdivₕ_uₕ => p.scratch.ghost_buffer.ᶜtemp_scalar)
    ᶜcurlₕ_uₕ = @. p.scratch.ᶜtemp_C3 = C3(wcurlₕ(ᶜuₕ))
    Spaces.weighted_dss!(ᶜcurlₕ_uₕ => p.scratch.ghost_buffer.ᶜtemp_C3)
    @. Yₜ.c.uₕ += ᶜβ_viscous * (C12(wgradₕ(ᶜdivₕ_uₕ)) - C12(wcurlₕ(ᶜcurlₕ_uₕ)))

    ᶠgradₕ_u₃ = @. p.scratch.ᶠtemp_C12 = C12(wgradₕ(Y.f.u₃.components.data.:1))
    Spaces.weighted_dss!(ᶠgradₕ_u₃ => p.scratch.ghost_buffer.ᶠtemp_C12)
    @. Yₜ.f.u₃.components.data.:1 += ᶠβ_viscous * wdivₕ(ᶠgradₕ_u₃)

    ᶜgradₕ_h_tot = @. p.scratch.ᶜtemp_C12 = C12(wgradₕ(ᶜh_tot))
    Spaces.weighted_dss!(ᶜgradₕ_h_tot => p.scratch.ghost_buffer.ᶜtemp_C12)
    @. Yₜ.c.ρe_tot += ᶜβ_viscous * wdivₕ(Y.c.ρ * ᶜgradₕ_h_tot)
    for (ᶜρχₜ, ᶜχ, χ_name) in matching_subfields(Yₜ.c, ᶜspecific)
        χ_name == :e_tot && continue
        ᶜgradₕ_χ = @. p.scratch.ᶜtemp_C12 = C12(wgradₕ(ᶜχ))
        Spaces.weighted_dss!(ᶜgradₕ_χ => p.scratch.ghost_buffer.ᶜtemp_C12)
        @. ᶜρχₜ += ᶜβ_viscous * wdivₕ(Y.c.ρ * ᶜgradₕ_χ)
        @. Yₜ.c.ρ += ᶜβ_viscous * wdivₕ(Y.c.ρ * ᶜgradₕ_χ)
    end
end
