#####
##### Advection
#####

using LinearAlgebra: ×, dot
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore.Operators as Operators

function horizontal_advection_tendency!(Yₜ, Y, p, t)
    divₕ = Operators.Divergence()
    gradₕ = Operators.Gradient()
    curlₕ = Operators.Curl()

    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜu, ᶜK, ᶜΦ, ᶜp, ᶜω³, ᶠω¹², ᶜp_ref) = p
    point_type = eltype(Fields.local_geometry_field(axes(Y.c)).coordinates)

    # Mass conservation
    @. Yₜ.c.ρ -= divₕ(ᶜρ * ᶜu)

    # Energy conservation
    if :ρθ in propertynames(Y.c)
        @. Yₜ.c.ρθ -= divₕ(Y.c.ρθ * ᶜu)
    elseif :ρe_tot in propertynames(Y.c)
        @. Yₜ.c.ρe_tot -= divₕ((Y.c.ρe_tot + ᶜp) * ᶜu)
    end

    # Momentum conservation
    if point_type <: Geometry.Abstract3DPoint
        @. ᶜω³ = curlₕ(ᶜuₕ)
        @. ᶠω¹² = curlₕ(ᶠw)
        @. Yₜ.c.uₕ -= gradₕ(ᶜp - ᶜp_ref) / ᶜρ + gradₕ(ᶜK + ᶜΦ)
    elseif point_type <: Geometry.Abstract2DPoint
        ᶜω³ .= tuple(zero(eltype(ᶜω³)))
        @. ᶠω¹² = Geometry.Contravariant12Vector(curlₕ(ᶠw))
        @. Yₜ.c.uₕ -=
            Geometry.Covariant12Vector(gradₕ(ᶜp - ᶜp_ref) / ᶜρ + gradₕ(ᶜK + ᶜΦ))
    end

    return nothing
end

function explicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    Fields.bycolumn(axes(Y.c)) do colidx
        explicit_vertical_advection_tendency!(Yₜ, Y, p, t, colidx)
    end
    return nothing
end

function explicit_vertical_advection_tendency!(Yₜ, Y, p, t, colidx)
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    (; ᶜu, ᶜK, ᶜω³, ᶠω¹², ᶠu³, ᶜf) = p
    (; ᶠinterp, ᶠcurlᵥ, ᶜinterp, ᶠgradᵥ) = p.operators
    ᶜJ = Fields.local_geometry_field(axes(ᶜρ)).J

    # Momentum conservation
    @. ᶠω¹²[colidx] += ᶠcurlᵥ(ᶜuₕ[colidx])
    @. Yₜ.c.uₕ[colidx] -=
        ᶜinterp(
            ᶠω¹²[colidx] × (ᶠinterp(ᶜρ[colidx] * ᶜJ[colidx]) * ᶠu³[colidx]),
        ) / (ᶜρ[colidx] * ᶜJ[colidx]) +
        (ᶜf[colidx] + ᶜω³[colidx]) ×
        Geometry.project(Geometry.Contravariant12Axis(), ᶜu[colidx])
    @. Yₜ.f.w[colidx] -=
        ᶠω¹²[colidx] ×
        ᶠinterp(Geometry.project(Geometry.Contravariant12Axis(), ᶜu[colidx])) +
        ᶠgradᵥ(ᶜK[colidx])

    return nothing
end
