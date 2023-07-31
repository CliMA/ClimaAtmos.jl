#####
##### Advection
#####

using LinearAlgebra: ×, dot
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry

function horizontal_advection_tendency!(Yₜ, Y, p, t)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    (; ᶜu, ᶜK, ᶜp, ᶜΦ, ᶜp_ref, atmos) = p
    if p.atmos.turbconv_model isa AbstractEDMF
        (; ᶜu⁰) = p
    end
    if p.atmos.turbconv_model isa EDMFX
        (; ᶜuʲs) = p
    end

    @. Yₜ.c.ρ -= divₕ(Y.c.ρ * ᶜu)
    if p.atmos.turbconv_model isa EDMFX
        for j in 1:n
            @. Yₜ.c.sgsʲs.:($$j).ρa -= divₕ(Y.c.sgsʲs.:($$j).ρa * ᶜuʲs.:($$j))
        end
    end

    if :ρθ in propertynames(Y.c)
        @. Yₜ.c.ρθ -= divₕ(Y.c.ρθ * ᶜu)
    elseif :ρe_tot in propertynames(Y.c)
        (; ᶜh_tot) = p
        @. Yₜ.c.ρe_tot -= divₕ(Y.c.ρ * ᶜh_tot * ᶜu)
    end
    if p.atmos.turbconv_model isa EDMFX
        for j in 1:n
            if :ρθ in propertynames(Y.c)
                @. Yₜ.c.sgsʲs.:($$j).ρaθ -=
                    divₕ(Y.c.sgsʲs.:($$j).ρaθ * ᶜuʲs.:($$j))
            elseif :ρe_tot in propertynames(Y.c)
                (; ᶜh_totʲs) = p
                @. Yₜ.c.sgsʲs.:($$j).ρae_tot -=
                    divₕ(Y.c.sgsʲs.:($$j).ρa * ᶜh_totʲs.:($$j) * ᶜuʲs.:($$j))
            end
        end
    end

    if use_prognostic_tke(atmos)
        @. Yₜ.c.sgs⁰.ρatke -= divₕ(Y.c.sgs⁰.ρatke * ᶜu⁰)
    end

    @. Yₜ.c.uₕ -= C12(gradₕ(ᶜp - ᶜp_ref) / Y.c.ρ + gradₕ(ᶜK + ᶜΦ))
    # Without the C12(), the right-hand side would be a C1 or C2 in 2D space.
    return nothing
end

function horizontal_tracer_advection_tendency!(Yₜ, Y, p, t)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    (; ᶜu) = p
    if p.atmos.turbconv_model isa EDMFX
        (; ᶜuʲs) = p
    end

    for ρχ_name in filter(is_tracer_var, propertynames(Y.c))
        @. Yₜ.c.:($$ρχ_name) -= divₕ(Y.c.:($$ρχ_name) * ᶜu)
    end

    if p.atmos.turbconv_model isa EDMFX
        for j in 1:n
            for ρaχ_name in
                filter(is_tracer_var, propertynames(Y.c.sgsʲs.:($j)))
                @. Yₜ.c.sgsʲs.:($$j).:($$ρaχ_name) -=
                    divₕ(Y.c.sgsʲs.:($$j).:($$ρaχ_name) * ᶜuʲs.:($$j))
            end
        end
    end
    return nothing
end

function explicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    (; atmos) = p
    (; turbconv_model) = atmos
    n = n_prognostic_mass_flux_subdomains(turbconv_model)
    advect_tke = use_prognostic_tke(atmos)
    is_total_energy = p.atmos.energy_form isa TotalEnergy
    point_type = eltype(Fields.coordinate_field(Y.c))
    (; dt) = p.simulation
    ᶜJ = Fields.local_geometry_field(Y.c).J
    (; ᶜu, ᶠu³, ᶜK, ᶜf) = p
    (; edmfx_upwinding) = n > 0 || advect_tke ? p : all_nothing
    (; ᶜuʲs, ᶠu³ʲs, ᶜKʲs, ᶜρʲs, ᶜspecificʲs) = n > 0 ? p : all_nothing
    (; ᶜp, ᶜp_ref, ᶜρ_ref, ᶠgradᵥ_ᶜΦ) = n > 0 ? p : all_nothing
    (; ᶜh_totʲs) = n > 0 && is_total_energy ? p : all_nothing
    (; ᶠu³⁰) = advect_tke ? p : all_nothing
    ᶜρa⁰ = advect_tke ? (n > 0 ? p.ᶜρa⁰ : Y.c.ρ) : nothing
    ᶜtke⁰ = advect_tke ? (n > 0 ? p.ᶜspecific⁰.tke : p.ᶜtke⁰) : nothing
    ᶜ1 = p.ᶜtemp_scalar
    ᶜω³ = p.ᶜtemp_CT3
    ᶠω¹² = p.ᶠtemp_CT12
    ᶠω¹²ʲs = p.ᶠtemp_CT12ʲs

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
    @. ᶠω¹² += CT12(curlₕ(Y.f.u₃))
    for j in 1:n
        @. ᶠω¹²ʲs.:($$j) += CT12(curlₕ(Y.f.sgsʲs.:($$j).u₃))
    end
    # Without the CT12(), the right-hand side would be a CT1 or CT2 in 2D space.

    Fields.bycolumn(axes(Y.c)) do colidx
        @. Yₜ.c.uₕ[colidx] -=
            ᶜinterp(
                ᶠω¹²[colidx] ×
                (ᶠinterp(Y.c.ρ[colidx] * ᶜJ[colidx]) * ᶠu³[colidx]),
            ) / (Y.c.ρ[colidx] * ᶜJ[colidx]) +
            (ᶜf[colidx] + ᶜω³[colidx]) × CT12(ᶜu[colidx])
        @. Yₜ.f.u₃[colidx] -=
            ᶠω¹²[colidx] × ᶠinterp(CT12(ᶜu[colidx])) + ᶠgradᵥ(ᶜK[colidx])

        for j in 1:n
            @. Yₜ.f.sgsʲs.:($$j).u₃[colidx] -=
                ᶠω¹²ʲs.:($$j)[colidx] × ᶠinterp(CT12(ᶜuʲs.:($$j)[colidx])) +
                ᶠgradᵥ(ᶜKʲs.:($$j)[colidx])

            # TODO: Move this to implicit_vertical_advection_tendency!.
            @. Yₜ.f.sgsʲs.:($$j).u₃[colidx] -=
                (
                    ᶠgradᵥ(ᶜp[colidx] - ᶜp_ref[colidx]) +
                    ᶠinterp(ᶜρʲs.:($$j)[colidx] - ᶜρ_ref[colidx]) *
                    ᶠgradᵥ_ᶜΦ[colidx]
                ) / ᶠinterp(ᶜρʲs.:($$j)[colidx])
        end

        # TODO: Move this to implicit_vertical_advection_tendency!.
        for j in 1:n
            @. ᶜ1[colidx] = one(Y.c.ρ[colidx])
            vertical_transport!(
                Yₜ.c.sgsʲs.:($j).ρa[colidx],
                ᶜJ[colidx],
                Y.c.sgsʲs.:($j).ρa[colidx],
                ᶠu³ʲs.:($j)[colidx],
                ᶜ1[colidx],
                dt,
                edmfx_upwinding,
            )

            if :ρae_tot in propertynames(Yₜ.c.sgsʲs.:($j))
                vertical_transport!(
                    Yₜ.c.sgsʲs.:($j).ρae_tot[colidx],
                    ᶜJ[colidx],
                    Y.c.sgsʲs.:($j).ρa[colidx],
                    ᶠu³ʲs.:($j)[colidx],
                    ᶜh_totʲs.:($j)[colidx],
                    dt,
                    edmfx_upwinding,
                )
            end

            for (ᶜρaχʲₜ, ᶜχʲ, χ_name) in
                matching_subfields(Yₜ.c.sgsʲs.:($j), ᶜspecificʲs.:($j))
                χ_name == :e_tot && continue
                vertical_transport!(
                    ᶜρaχʲₜ[colidx],
                    ᶜJ[colidx],
                    Y.c.sgsʲs.:($j).ρa[colidx],
                    ᶠu³ʲs.:($j)[colidx],
                    ᶜχʲ[colidx],
                    dt,
                    edmfx_upwinding,
                )
            end
        end

        # TODO: Move this to implicit_vertical_advection_tendency!.
        # 12688
        if use_prognostic_tke(atmos) # advect_tke triggers allocations
            vertical_transport!(
                Yₜ.c.sgs⁰.ρatke[colidx],
                ᶜJ[colidx],
                ᶜρa⁰[colidx],
                ᶠu³⁰[colidx],
                ᶜtke⁰[colidx],
                dt,
                edmfx_upwinding,
            )
        end
    end
end
