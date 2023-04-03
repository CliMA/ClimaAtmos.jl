#####
##### Viscous sponge
#####

import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore.Spaces as Spaces
import ClimaCore.Operators as Operators

viscous_sponge_cache(::Nothing, Y) = NamedTuple()
viscous_sponge_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

function viscous_sponge_cache(viscous_sponge::ViscousSponge, Y)
    (; ᶠκ₂, zd) = viscous_sponge
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


laplacian_viscosity_cache(::Nothing, Y) = NamedTuple()
laplacian_viscosity_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

function laplacian_viscosity_cache(laplacian_viscosity::LaplacianViscosity, Y)
    (; κ₂) = laplacian_viscosity
    FT = Spaces.undertype(axes(Y.c))
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠz = Fields.coordinate_field(Y.f).z
    ᶜαₘ = κ₂
    ᶠαₘ = κ₂
    zmax = maximum(ᶠz)
    ᶜκ₂ = @. ᶜαₘ  # No viscosity scaling in domain
    ᶠκ₂ = @. ᶠαₘ  # No viscosity scaling in domain
    return (; ᶜκ₂, ᶠκ₂)
end

function laplacian_viscosity_tendency!(Yₜ, Y, p, t, ::LaplacianViscosity)
    (; ᶜκ₂, ᶠκ₂, ᶜp) = p
    divₕ = Operators.Divergence()
    wdivₕ = Operators.WeakDivergence()
    gradₕ = Operators.Gradient()
    wgradₕ = Operators.WeakGradient()
    curlₕ = Operators.Curl()
    wcurlₕ = Operators.WeakCurl()

    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    if :ρθ in propertynames(Y.c)
        @. Yₜ.c.ρθ += ᶜκ₂ * wdivₕ(ᶜρ * gradₕ(Y.c.ρθ / ᶜρ))
    elseif :ρe_tot in propertynames(Y.c)
        @. Yₜ.c.ρe_tot += ᶜκ₂ * wdivₕ(ᶜρ * gradₕ((Y.c.ρe_tot + ᶜp) / ᶜρ))
    end

    point_type = eltype(Fields.local_geometry_field(axes(Y.c)).coordinates)
    if point_type <: Geometry.Abstract3DPoint
      @. Yₜ.c.uₕ +=
          ᶜκ₂ * (
              wgradₕ(divₕ(ᶜuₕ)) - Geometry.project(
                  Geometry.Covariant12Axis(),
                  wcurlₕ(Geometry.project(Geometry.Covariant3Axis(), curlₕ(ᶜuₕ))),
              )
          )
    elseif point_type <: Geometry.Abstract2DPoint
      @. Yₜ.c.uₕ += ᶜκ₂ * Geometry.project(Geometry.Covariant12Axis(), wgradₕ(divₕ(ᶜuₕ)))
      @. Yₜ.f.w.components.data.:1 += 
          ᶠκ₂ * wdivₕ(gradₕ(Y.f.w.components.data.:1))
    end
end
