#####
##### Advection
#####

using LinearAlgebra: ×, dot
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry

NVTX.@annotate function horizontal_advection_tendency!(Yₜ, Y, p, t)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    (; ᶜΦ, ᶜp_ref) = p.core
    (; ᶜu, ᶜK, ᶜp) = p.precomputed
    if p.atmos.turbconv_model isa AbstractEDMF
        (; ᶜu⁰) = p.precomputed
    end
    if p.atmos.turbconv_model isa PrognosticEDMFX
        (; ᶜuʲs) = p.precomputed
    end

    @. Yₜ.c.ρ -= wdivₕ(Y.c.ρ * ᶜu)
    if p.atmos.turbconv_model isa PrognosticEDMFX
        for j in 1:n
            @. Yₜ.c.sgsʲs.:($$j).ρa -= wdivₕ(Y.c.sgsʲs.:($$j).ρa * ᶜuʲs.:($$j))
        end
    end

    (; ᶜh_tot) = p.precomputed
    @. Yₜ.c.ρe_tot -= wdivₕ(Y.c.ρ * ᶜh_tot * ᶜu)

    if p.atmos.turbconv_model isa PrognosticEDMFX
        for j in 1:n
            @. Yₜ.c.sgsʲs.:($$j).mse -=
                wdivₕ(Y.c.sgsʲs.:($$j).mse * ᶜuʲs.:($$j)) -
                Y.c.sgsʲs.:($$j).mse * wdivₕ(ᶜuʲs.:($$j))
        end
    end

    if use_prognostic_tke(p.atmos.turbconv_model)
        @. Yₜ.c.sgs⁰.ρatke -= wdivₕ(Y.c.sgs⁰.ρatke * ᶜu⁰)
    end

    @. Yₜ.c.uₕ -= C12(gradₕ(ᶜp - ᶜp_ref) / Y.c.ρ + gradₕ(ᶜK + ᶜΦ))
    # Without the C12(), the right-hand side would be a C1 or C2 in 2D space.
    return nothing
end

NVTX.@annotate function horizontal_tracer_advection_tendency!(Yₜ, Y, p, t)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    (; ᶜu) = p.precomputed
    if p.atmos.turbconv_model isa PrognosticEDMFX
        (; ᶜuʲs) = p.precomputed
    end

    for ρχ_name in filter(is_tracer_var, propertynames(Y.c))
        @. Yₜ.c.:($$ρχ_name) -= wdivₕ(Y.c.:($$ρχ_name) * ᶜu)
    end

    if p.atmos.turbconv_model isa PrognosticEDMFX
        for j in 1:n
            @. Yₜ.c.sgsʲs.:($$j).q_tot -=
                wdivₕ(Y.c.sgsʲs.:($$j).q_tot * ᶜuʲs.:($$j)) -
                Y.c.sgsʲs.:($$j).q_tot * wdivₕ(ᶜuʲs.:($$j))
        end
    end

    return nothing
end

NVTX.@annotate function explicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    (; turbconv_model) = p.atmos
    n = n_prognostic_mass_flux_subdomains(turbconv_model)
    advect_tke = use_prognostic_tke(turbconv_model)
    point_type = eltype(Fields.coordinate_field(Y.c))
    (; params) = p
    (; dt) = p
    ᶜJ = Fields.local_geometry_field(Y.c).J
    (; ᶜf, ᶜΦ) = p.core
    (; ᶜu, ᶠu³, ᶜK) = p.precomputed
    (; edmfx_upwinding) = n > 0 || advect_tke ? p.atmos.numerics : all_nothing
    (; ᶜp, ᶜuʲs, ᶠu³ʲs, ᶜKʲs, ᶜρʲs) = n > 0 ? p.precomputed : all_nothing
    (; ᶜp_ref, ᶜρ_ref, ᶠgradᵥ_ᶜΦ) = n > 0 ? p.core : all_nothing
    (; ᶠu³⁰) = advect_tke ? p.precomputed : all_nothing
    (; energy_upwinding, tracer_upwinding, precip_upwinding) = p.atmos.numerics
    (; rayleigh_sponge, precip_model) = p.atmos
    (; ᶜspecific) = p.precomputed

    ᶜρa⁰ = advect_tke ? (n > 0 ? p.precomputed.ᶜρa⁰ : Y.c.ρ) : nothing
    ᶜρ⁰ = advect_tke ? (n > 0 ? p.precomputed.ᶜρ⁰ : Y.c.ρ) : nothing
    ᶜtke⁰ = advect_tke ? p.precomputed.ᶜtke⁰ : nothing
    ᶜa_scalar = p.scratch.ᶜtemp_scalar
    ᶜω³ = p.scratch.ᶜtemp_CT3
    ᶠω¹² = p.scratch.ᶠtemp_CT12
    ᶠω¹²ʲs = p.scratch.ᶠtemp_CT12ʲs
    FT = Spaces.undertype(axes(Y.c))

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

    ᶠz = Fields.coordinate_field(Y.f).z
    ᶠΦ = p.scratch.ᶠtemp_scalar
    @. ᶠΦ = CAP.grav(params) * ᶠz

    Fields.bycolumn(axes(Y.c)) do colidx
        if :ρe_tot in propertynames(Yₜ.c)
            (; ᶜh_tot) = p.precomputed
            for (coeff, upwinding) in ((1, energy_upwinding), (-1, Val(:none)))
                energy_upwinding isa Val{:none} && continue
                vertical_transport!(
                    coeff,
                    Yₜ.c.ρe_tot[colidx],
                    ᶜJ[colidx],
                    Y.c.ρ[colidx],
                    ᶠu³[colidx],
                    ᶜh_tot[colidx],
                    dt,
                    upwinding,
                )
            end
        end
        for (ᶜρχₜ, ᶜχ, χ_name) in matching_subfields(Yₜ.c, ᶜspecific)
            χ_name == :e_tot && continue
            for (coeff, upwinding) in ((1, tracer_upwinding), (-1, Val(:none)))
                tracer_upwinding isa Val{:none} && continue
                vertical_transport!(
                    coeff,
                    ᶜρχₜ[colidx],
                    ᶜJ[colidx],
                    Y.c.ρ[colidx],
                    ᶠu³[colidx],
                    ᶜχ[colidx],
                    dt,
                    upwinding,
                )
            end
        end

        if precip_model isa Microphysics1Moment
            # Advection of precipitation with the mean flow
            # is done with other tracers above.
            # Here we add the advection with precipitation terminal velocity
            # using first order upwind and free outflow bottom boundary condition

            ᶠu³ₚ = p.scratch.ᶠtemp_CT3
            ᶜqₚ = p.scratch.ᶜtemp_scalar
            lgf = Fields.local_geometry_field(Y.f)
            FT = Spaces.undertype(axes(Y.c))

            @. ᶠu³ₚ[colidx] =
                FT(-1) *
                ᶠinterp(p.precomputed.ᶜwᵣ[colidx]) *
                CT3(unit_basis_vector_data(CT3, lgf[colidx]))
            @. ᶜqₚ[colidx] = Y.c.ρq_rai[colidx] / Y.c.ρ[colidx]

            # TODO: Add support for SetDivergence to DivergenceF2C.
            ᶜdivᵥ_ρqₚ = Operators.DivergenceF2C(
                top = Operators.SetValue(C3(FT(0))),
                # bottom = Operators.SetDivergence(FT(0)),
            )

            for (coeff, upwinding) in ((1, precip_upwinding), (-1, Val(:none)))
                precip_upwinding isa Val{:none} && continue
                vertical_transport!(
                    coeff,
                    Yₜ.c.ρq_rai[colidx],
                    ᶜJ[colidx],
                    Y.c.ρ[colidx],
                    ᶠu³ₚ[colidx],
                    ᶜqₚ[colidx],
                    dt,
                    upwinding,
                    ᶜdivᵥ_ρqₚ,
                )
            end

            @. ᶠu³ₚ[colidx] =
                FT(-1) *
                ᶠinterp(p.precomputed.ᶜwₛ[colidx]) *
                CT3(unit_basis_vector_data(CT3, lgf[colidx]))
            @. ᶜqₚ[colidx] = Y.c.ρq_sno[colidx] / Y.c.ρ[colidx]
            for (coeff, upwinding) in ((1, precip_upwinding), (-1, Val(:none)))
                precip_upwinding isa Val{:none} && continue
                vertical_transport!(
                    coeff,
                    Yₜ.c.ρq_sno[colidx],
                    ᶜJ[colidx],
                    Y.c.ρ[colidx],
                    ᶠu³ₚ[colidx],
                    ᶜqₚ[colidx],
                    dt,
                    upwinding,
                    ᶜdivᵥ_ρqₚ,
                )
            end
        end
    end


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
            # For the updraft u_3 equation, we assume the grid-mean to be hydrostatic
            # and calcuate the buoyancy term relative to the grid-mean density.
            @. Yₜ.f.sgsʲs.:($$j).u₃[colidx] -=
                (
                    ᶠinterp(ᶜρʲs.:($$j)[colidx] - Y.c.ρ[colidx]) *
                    ᶠgradᵥ_ᶜΦ[colidx]
                ) / ᶠinterp(ᶜρʲs.:($$j)[colidx])

            # buoyancy term in mse equation
            @. Yₜ.c.sgsʲs.:($$j).mse[colidx] +=
                adjoint(CT3(ᶜinterp(Y.f.sgsʲs.:($$j).u₃[colidx]))) *
                (ᶜρʲs.:($$j)[colidx] - Y.c.ρ[colidx]) *
                ᶜgradᵥ(ᶠΦ[colidx]) / ᶜρʲs.:($$j)[colidx]
        end

        # TODO: Move this to implicit_vertical_advection_tendency!.
        for j in 1:n
            @. ᶜa_scalar[colidx] =
                draft_area(Y.c.sgsʲs.:($$j).ρa[colidx], ᶜρʲs.:($$j)[colidx])
            vertical_transport!(
                Yₜ.c.sgsʲs.:($j).ρa[colidx],
                ᶜJ[colidx],
                ᶜρʲs.:($j)[colidx],
                ᶠu³ʲs.:($j)[colidx],
                ᶜa_scalar[colidx],
                dt,
                edmfx_upwinding,
            )

            vertical_advection!(
                Yₜ.c.sgsʲs.:($j).mse[colidx],
                ᶠu³ʲs.:($j)[colidx],
                Y.c.sgsʲs.:($j).mse[colidx],
                edmfx_upwinding,
            )

            vertical_advection!(
                Yₜ.c.sgsʲs.:($j).q_tot[colidx],
                ᶠu³ʲs.:($j)[colidx],
                Y.c.sgsʲs.:($j).q_tot[colidx],
                edmfx_upwinding,
            )
        end

        # TODO: Move this to implicit_vertical_advection_tendency!.
        if use_prognostic_tke(turbconv_model) # advect_tke triggers allocations
            @. ᶜa_scalar[colidx] =
                ᶜtke⁰[colidx] * draft_area(ᶜρa⁰[colidx], ᶜρ⁰[colidx])
            vertical_transport!(
                Yₜ.c.sgs⁰.ρatke[colidx],
                ᶜJ[colidx],
                ᶜρ⁰[colidx],
                ᶠu³⁰[colidx],
                ᶜa_scalar[colidx],
                dt,
                edmfx_upwinding,
            )
        end
    end
end
