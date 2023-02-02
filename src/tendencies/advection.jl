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
    (; ᶜuvw, ᶜK, ᶜΦ, ᶜts, ᶜp, ᶜω³, ᶠω¹², params) = p
    (; ᶜρ_ref, ᶜp_ref) = p
    point_type = eltype(Fields.local_geometry_field(axes(Y.c)).coordinates)

    # Mass conservation
    @. Yₜ.c.ρ -= divₕ(ᶜρ * ᶜuvw)

    # Energy conservation
    if :ρθ in propertynames(Y.c)
        @. Yₜ.c.ρθ -= divₕ(Y.c.ρθ * ᶜuvw)
    elseif :ρe_tot in propertynames(Y.c)
        @. Yₜ.c.ρe_tot -= divₕ((Y.c.ρe_tot + ᶜp) * ᶜuvw)
    elseif :ρe_int in propertynames(Y.c)
        if point_type <: Geometry.Abstract3DPoint
            @. Yₜ.c.ρe_int -=
                divₕ((Y.c.ρe_int + ᶜp) * ᶜuvw) -
                dot(gradₕ(ᶜp), Geometry.Contravariant12Vector(ᶜuₕ))
        else
            @. Yₜ.c.ρe_int -=
                divₕ((Y.c.ρe_int + ᶜp) * ᶜuvw) -
                dot(gradₕ(ᶜp), Geometry.Contravariant1Vector(ᶜuₕ))
        end
    end

    # Momentum conservation
    if point_type <: Geometry.Abstract3DPoint
        @. ᶜω³ = curlₕ(ᶜuₕ)
        @. ᶠω¹² = curlₕ(ᶠw)
        @. Yₜ.c.uₕ -= gradₕ(ᶜp - ᶜp_ref) / ᶜρ + gradₕ(ᶜK + ᶜΦ)
    elseif point_type <: Geometry.Abstract2DPoint
        ᶜω³ .= Ref(zero(eltype(ᶜω³)))
        @. ᶠω¹² = Geometry.Contravariant12Vector(curlₕ(ᶠw))
        @. Yₜ.c.uₕ -=
            Geometry.Covariant12Vector(gradₕ(ᶜp - ᶜp_ref) / ᶜρ + gradₕ(ᶜK + ᶜΦ))
    end

    # Tracer conservation
    for ᶜρc_name in filter(is_tracer_var, propertynames(Y.c))
        ᶜρc = getproperty(Y.c, ᶜρc_name)
        ᶜρcₜ = getproperty(Yₜ.c, ᶜρc_name)
        @. ᶜρcₜ -= divₕ(ᶜρc * ᶜuvw)
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
    ᶠw = Y.f.w
    C123 = Geometry.Covariant123Vector
    (; ᶜuvw, ᶜK, ᶜp, ᶜω³, ᶠω¹², ᶠu¹², ᶜu¹², ᶠu³, ᶜf) = p
    (; ᶜdivᵥ, ᶠinterp, ᶠwinterp, ᶠcurlᵥ, ᶜinterp, ᶠgradᵥ) = p.operators
    # Mass conservation
    J = Fields.local_geometry_field(axes(ᶜρ)).J
    @. Yₜ.c.ρ[colidx] -= ᶜdivᵥ(ᶠinterp(ᶜρ[colidx] * ᶜuₕ[colidx]))

    # Energy conservation
    if :ρθ in propertynames(Y.c)
        @. Yₜ.c.ρθ[colidx] -= ᶜdivᵥ(ᶠinterp(Y.c.ρθ[colidx] * ᶜuₕ[colidx]))
    elseif :ρe_tot in propertynames(Y.c)
        @. Yₜ.c.ρe_tot[colidx] -=
            ᶜdivᵥ(ᶠinterp((Y.c.ρe_tot[colidx] + ᶜp[colidx]) * ᶜuₕ[colidx]))
    elseif :ρe_int in propertynames(Y.c)
        @. Yₜ.c.ρe_int[colidx] -=
            ᶜdivᵥ(ᶠinterp((Y.c.ρe_int[colidx] + ᶜp[colidx]) * ᶜuₕ[colidx]))
    end

    # Momentum conservation
    @. ᶠω¹²[colidx] += ᶠcurlᵥ(ᶜuₕ[colidx])
    @. ᶜu¹²[colidx] =
        Geometry.project(Geometry.Contravariant12Axis(), ᶜuvw[colidx])
    @. ᶠu³[colidx] = Geometry.project(
        Geometry.Contravariant3Axis(),
        C123(ᶠinterp(ᶜuₕ[colidx])) + C123(ᶠw[colidx]),
    )
    @. Yₜ.c.uₕ[colidx] -=
        ᶜinterp(ᶠω¹²[colidx] × ᶠu³[colidx]) +
        (ᶜf[colidx] + ᶜω³[colidx]) ×
        (Geometry.project(Geometry.Contravariant12Axis(), ᶜuvw[colidx]))
    @. Yₜ.f.w[colidx] -=
        ᶠω¹²[colidx] × (ᶠinterp(J[colidx] * ᶜρ[colidx]) * ᶜu¹²[colidx]) /ᶠ
        interp(ᶜρ[colidx] * J[colidx]) + ᶠgradᵥ(ᶜK[colidx])

    # Tracer conservation
    for ᶜρc_name in filter(is_tracer_var, propertynames(Y.c))
        ᶜρc = getproperty(Y.c, ᶜρc_name)
        ᶜρcₜ = getproperty(Yₜ.c, ᶜρc_name)
        @. ᶜρcₜ[colidx] -= ᶜdivᵥ(ᶠinterp(ᶜρc[colidx] * ᶜuₕ[colidx]))
    end

    return nothing
end
