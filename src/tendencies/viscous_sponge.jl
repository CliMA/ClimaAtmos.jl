#####
##### Viscous sponge
#####

import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore.Spaces as Spaces
import ClimaCore.Operators as Operators

viscous_sponge_cache(Y; kwargs...) =
    viscous_sponge_cache(Y, Spaces.undertype(axes(Y.c)); kwargs...)

function viscous_sponge_cache(
    Y,
    ::Type{FT};
    zd_viscous = FT(15e3),
    κ₂ = FT(1e5),
) where {FT}
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠz = Fields.coordinate_field(Y.f).z
    ᶜαₘ = @. ifelse(ᶜz > zd_viscous, κ₂, FT(0))
    ᶠαₘ = @. ifelse(ᶠz > zd_viscous, κ₂, FT(0))
    zmax = maximum(ᶠz)
    ᶜβ_viscous =
        @. ᶜαₘ * sin(FT(π) / 2 * (ᶜz - zd_viscous) / (zmax - zd_viscous))^2
    ᶠβ_viscous =
        @. ᶠαₘ * sin(FT(π) / 2 * (ᶠz - zd_viscous) / (zmax - zd_viscous))^2
    return (; ᶜβ_viscous, ᶠβ_viscous)
end

function viscous_sponge_tendency!(Yₜ, Y, p, t)
    (; ᶜβ_viscous, ᶠβ_viscous, ᶜp) = p
    divₕ = Operators.Divergence()
    wdivₕ = Operators.WeakDivergence()
    gradₕ = Operators.Gradient()
    wgradₕ = Operators.WeakGradient()
    curlₕ = Operators.Curl()
    wcurlₕ = Operators.WeakCurl()

    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    if :ρθ in propertynames(Y.c)
        @. Yₜ.c.ρθ += ᶜβ_viscous * wdivₕ(ᶜρ * gradₕ(Y.c.ρθ / ᶜρ))
    elseif :ρe_tot in propertynames(Y.c)
        @. Yₜ.c.ρe_tot += ᶜβ_viscous * wdivₕ(ᶜρ * gradₕ((Y.c.ρe_tot + ᶜp) / ᶜρ))
    elseif :ρe_int in propertynames(Y.c)
        @. Yₜ.c.ρe_int += ᶜβ_viscous * wdivₕ(ᶜρ * gradₕ((Y.c.ρe_int + ᶜp) / ᶜρ))
    end
    @. Yₜ.c.uₕ +=
        ᶜβ_viscous * (
            wgradₕ(divₕ(ᶜuₕ)) - Geometry.project(
                Geometry.Covariant12Axis(),
                wcurlₕ(Geometry.project(Geometry.Covariant3Axis(), curlₕ(ᶜuₕ))),
            )
        )
    @. Yₜ.f.w.components.data.:1 +=
        ᶠβ_viscous * wdivₕ(gradₕ(Y.f.w.components.data.:1))
end
