#####
##### Advection
#####

using LinearAlgebra: ×, dot
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry

function horizontal_advection_tendency!(Yₜ, Y, p, t)
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜu, ᶜK, ᶜΦ, ᶜp, ᶜp_ref) = p
    ᶜω³ = p.ᶜtemp_CT3
    ᶠω¹² = p.ᶠtemp_CT12
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
        @. ᶠω¹² = CT12(curlₕ(ᶠw))
        @. Yₜ.c.uₕ -= C12(gradₕ(ᶜp - ᶜp_ref) / ᶜρ + gradₕ(ᶜK + ᶜΦ))
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
    (; ᶜu, ᶜK, ᶠu³, ᶜf) = p
    ᶜω³ = p.ᶜtemp_CT3
    ᶠω¹² = p.ᶠtemp_CT12
    ᶜJ = Fields.local_geometry_field(axes(ᶜρ)).J

    # Momentum conservation
    @. ᶠω¹²[colidx] += ᶠcurlᵥ(ᶜuₕ[colidx])
    @. Yₜ.c.uₕ[colidx] -=
        ᶜinterp(
            ᶠω¹²[colidx] × (ᶠinterp(ᶜρ[colidx] * ᶜJ[colidx]) * ᶠu³[colidx]),
        ) / (ᶜρ[colidx] * ᶜJ[colidx]) +
        (ᶜf[colidx] + ᶜω³[colidx]) × CT12(ᶜu[colidx])
    @. Yₜ.f.w[colidx] -=
        ᶠω¹²[colidx] × ᶠinterp(CT12(ᶜu[colidx])) + ᶠgradᵥ(ᶜK[colidx])

    return nothing
end
