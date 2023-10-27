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

add_viscous_sponge_energy_tendency!(Yₜ, Y, p, t) =
    add_viscous_sponge_energy_tendency!(Yₜ, Y, p, t, p.atmos.energy_form)

function add_viscous_sponge_energy_tendency!(Yₜ, Y, p, t, ::TotalEnergy)
    (; ᶜβ_viscous) = p.viscous_sponge
    (; ᶜh_tot) = p
    ᶜρ = Y.c.ρ
    @. Yₜ.c.ρe_tot += ᶜβ_viscous * wdivₕ(ᶜρ * gradₕ(ᶜh_tot))
end

function add_viscous_sponge_energy_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::PotentialTemperature,
)
    (; ᶜβ_viscous) = p.viscous_sponge
    ᶜρ = Y.c.ρ
    @. Yₜ.c.ρθ += ᶜβ_viscous * wdivₕ(ᶜρ * gradₕ(Y.c.ρθ / ᶜρ))
end

function viscous_sponge_tendency!(Yₜ, Y, p, t, ::ViscousSponge)
    (; ᶜβ_viscous, ᶠβ_viscous) = p.viscous_sponge
    ᶜuₕ = Y.c.uₕ
    add_viscous_sponge_energy_tendency!(Yₜ, Y, p, t)
    @. Yₜ.c.uₕ +=
        ᶜβ_viscous * (
            wgradₕ(divₕ(ᶜuₕ)) - Geometry.project(
                Geometry.Covariant12Axis(),
                wcurlₕ(Geometry.project(Geometry.Covariant3Axis(), curlₕ(ᶜuₕ))),
            )
        )
    @. Yₜ.f.u₃.components.data.:1 +=
        ᶠβ_viscous * wdivₕ(gradₕ(Y.f.u₃.components.data.:1))
end
