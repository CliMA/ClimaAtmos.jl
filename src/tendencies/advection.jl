#####
##### Advection
#####

using LinearAlgebra: ×, dot
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry

function horizontal_advection_tendency!(Yₜ, Y, p, t)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    (; ᶜu, ᶜK, ᶜp, ᶜΦ, ᶜp_ref) = p
    if n > 0
        (; ᶜu⁰, ᶜuʲs, ᶜρʲs) = p
    end

    @. Yₜ.c.ρ -= divₕ(Y.c.ρ * ᶜu)
    for j in 1:n
        @. Yₜ.c.sgsʲs.:($$j).ρa -= divₕ(Y.c.sgsʲs.:($$j).ρa * ᶜuʲs.:($$j))
    end

    if :ρθ in propertynames(Y.c)
        @. Yₜ.c.ρθ -= divₕ(Y.c.ρθ * ᶜu)
    elseif :ρe_tot in propertynames(Y.c)
        @. Yₜ.c.ρe_tot -= divₕ((Y.c.ρe_tot + ᶜp) * ᶜu)
    end
    for j in 1:n
        if :ρθ in propertynames(Y.c)
            @. Yₜ.c.sgsʲs.:($$j).ρaθ -= divₕ(Y.c.sgsʲs.:($$j).ρaθ * ᶜuʲs.:($$j))
        elseif :ρe_tot in propertynames(Y.c)
            @. Yₜ.c.sgsʲs.:($$j).ρae_tot -= divₕ(
                (
                    Y.c.sgsʲs.:($$j).ρae_tot +
                    Y.c.sgsʲs.:($$j).ρa * ᶜp / ᶜρʲs.:($$j)
                ) * ᶜuʲs.:($$j),
            )
        end
    end

    if n > 0
        @. Yₜ.c.sgs⁰.ρatke -= divₕ(Y.c.sgs⁰.ρatke * ᶜu⁰)
    end

    @. Yₜ.c.uₕ -= C12(gradₕ(ᶜp - ᶜp_ref) / Y.c.ρ + gradₕ(ᶜK + ᶜΦ))
    # Without the C12(), the right-hand side would be a C1 or C2 in 2D space.
end

function horizontal_tracer_advection_tendency!(Yₜ, Y, p, t)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    (; ᶜu) = p
    if n > 0
        (; ᶜuʲs) = p
    end

    for ρχ_name in filter(is_tracer_var, propertynames(Y.c))
        @. Yₜ.c.:($$ρχ_name) -= divₕ(Y.c.:($$ρχ_name) * ᶜu)
    end
    for j in 1:n
        for ρaχ_name in filter(is_tracer_var, propertynames(Y.c.sgsʲs.:($j)))
            @. Yₜ.c.sgsʲs.:($$j).:($$ρaχ_name) -=
                divₕ(Y.c.sgsʲs.:($$j).:($$ρaχ_name) * ᶜuʲs.:($$j))
        end
    end
end

function explicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    point_type = eltype(Fields.coordinate_field(Y.c))
    ᶜJ = Fields.local_geometry_field(Y.c).J
    (; ᶜu, ᶠu³, ᶜK, ᶜf) = p
    ᶜω³ = p.ᶜtemp_CT3
    ᶠω¹² = p.ᶠtemp_CT12
    if n > 0
        ᶠω¹²ʲs = p.ᶠtemp_CT12ʲs
    end

    if point_type <: Geometry.Abstract3DPoint
        @. ᶜω³ = curlₕ(Y.c.uₕ)
    elseif point_type <: Geometry.Abstract2DPoint
        @. ᶜω³ = zero(ᶜω³)
    end

    Fields.bycolumn(axes(Y.c)) do colidx
        @. ᶠω¹²[colidx] = ᶠcurlᵥ(Y.c.uₕ[colidx])
    end
    for j in 1:n
        @. ᶠω¹²ʲs.:($$j) = ᶠω¹²
    end
    @. ᶠω¹² += CT12(curlₕ(Y.f.w))
    for j in 1:n
        @. ᶠω¹²ʲs.:($$j) += CT12(curlₕ(Y.f.sgsʲs.:($$j).w))
    end
    # Without the CT12(), the right-hand side would be a CT1 or CT2 in 2D space.

    Fields.bycolumn(axes(Y.c)) do colidx
        @. Yₜ.c.uₕ[colidx] -=
            ᶜinterp(
                ᶠω¹²[colidx] ×
                (ᶠinterp(Y.c.ρ[colidx] * ᶜJ[colidx]) * ᶠu³[colidx]),
            ) / (Y.c.ρ[colidx] * ᶜJ[colidx]) +
            (ᶜf[colidx] + ᶜω³[colidx]) × CT12(ᶜu[colidx])
        @. Yₜ.f.w[colidx] -=
            ᶠω¹²[colidx] × ᶠinterp(CT12(ᶜu[colidx])) + ᶠgradᵥ(ᶜK[colidx])
        for j in 1:n
            # TODO: Figure out why putting these extractions outside of the
            # bycolumn triggers a lot of allocations.
            (; ᶜuʲs, ᶜKʲs) = p
            local ᶠω¹²ʲs = p.ᶠtemp_CT12ʲs # Adding `local` reduces allocations.

            @. Yₜ.f.sgsʲs.:($$j).w[colidx] -=
                ᶠω¹²ʲs.:($$j)[colidx] × ᶠinterp(CT12(ᶜuʲs.:($$j)[colidx])) +
                ᶠgradᵥ(ᶜKʲs.:($$j)[colidx])
        end
    end
end
