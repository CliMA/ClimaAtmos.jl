#####
##### Advection
#####

using LinearAlgebra: ×, dot
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry

NVTX.@annotate function horizontal_advection_tendency!(Yₜ, Y, p, t)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    (; ᶜΦ) = p.core
    (; ᶜu, ᶜK, ᶜp) = p.precomputed
    if p.atmos.turbconv_model isa AbstractEDMF
        if p.atmos.turbconv_model isa EDOnlyEDMFX
            ᶜu⁰ = ᶜu
        else
            (; ᶜu⁰) = p.precomputed
        end
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

    @. Yₜ.c.uₕ -= C12(gradₕ(ᶜp) / Y.c.ρ + gradₕ(ᶜK + ᶜΦ))
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
        # TODO - advection
        end
    end
    return nothing
end

NVTX.@annotate function explicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    (; turbconv_model) = p.atmos
    n = n_prognostic_mass_flux_subdomains(turbconv_model)
    advect_tke = use_prognostic_tke(turbconv_model)
    point_type = eltype(Fields.coordinate_field(Y.c))
    (; dt) = p
    ᶜJ = Fields.local_geometry_field(Y.c).J
    (; ᶜf³, ᶠf¹², ᶜΦ) = p.core
    (; ᶜu, ᶠu³, ᶜK) = p.precomputed
    (; edmfx_upwinding) = n > 0 || advect_tke ? p.atmos.numerics : all_nothing
    (; ᶜuʲs, ᶜKʲs, ᶠKᵥʲs) = n > 0 ? p.precomputed : all_nothing
    (; energy_upwinding, tracer_upwinding) = p.atmos.numerics
    (; ᶜspecific) = p.precomputed

    ᶠu³⁰ =
        advect_tke ?
        (
            turbconv_model isa EDOnlyEDMFX ? p.precomputed.ᶠu³ :
            p.precomputed.ᶠu³⁰
        ) : nothing
    ᶜρa⁰ = advect_tke ? (n > 0 ? p.precomputed.ᶜρa⁰ : Y.c.ρ) : nothing
    ᶜρ⁰ = advect_tke ? (n > 0 ? p.precomputed.ᶜρ⁰ : Y.c.ρ) : nothing
    ᶜtke⁰ = advect_tke ? p.precomputed.ᶜtke⁰ : nothing
    ᶜa_scalar = p.scratch.ᶜtemp_scalar
    ᶜω³ = p.scratch.ᶜtemp_CT3
    ᶠω¹² = p.scratch.ᶠtemp_CT12
    ᶠω¹²ʲs = p.scratch.ᶠtemp_CT12ʲs

    if point_type <: Geometry.Abstract3DPoint
        @. ᶜω³ = curlₕ(Y.c.uₕ)
    elseif point_type <: Geometry.Abstract2DPoint
        @. ᶜω³ = zero(ᶜω³)
    end

    @. ᶠω¹² = ᶠcurlᵥ(Y.c.uₕ)
    for j in 1:n
        @. ᶠω¹²ʲs.:($$j) = ᶠω¹²
    end
    @. ᶠω¹² += CT12(curlₕ(Y.f.u₃))
    for j in 1:n
        @. ᶠω¹²ʲs.:($$j) += CT12(curlₕ(Y.f.sgsʲs.:($$j).u₃))
    end
    # Without the CT12(), the right-hand side would be a CT1 or CT2 in 2D space.

    ᶜρ = Y.c.ρ

    # Full vertical advection of passive tracers (like liq, rai, etc) ...
    for (ᶜρχₜ, ᶜχ, χ_name) in matching_subfields(Yₜ.c, ᶜspecific)
        χ_name in (:e_tot, :q_tot) && continue
        vtt = vertical_transport(ᶜρ, ᶠu³, ᶜχ, float(dt), tracer_upwinding)
        @. ᶜρχₜ += vtt
    end
    # ... and upwinding correction of energy and total water.
    # (The central advection of energy and total water is done implicitly.)
    if energy_upwinding != Val(:none)
        (; ᶜh_tot) = p.precomputed
        vtt = vertical_transport(ᶜρ, ᶠu³, ᶜh_tot, float(dt), energy_upwinding)
        vtt_central = vertical_transport(ᶜρ, ᶠu³, ᶜh_tot, float(dt), Val(:none))
        @. Yₜ.c.ρe_tot += vtt - vtt_central
    end

    if !(p.atmos.moisture_model isa DryModel) && tracer_upwinding != Val(:none)
        ᶜq_tot = ᶜspecific.q_tot
        vtt = vertical_transport(ᶜρ, ᶠu³, ᶜq_tot, float(dt), tracer_upwinding)
        vtt_central = vertical_transport(ᶜρ, ᶠu³, ᶜq_tot, float(dt), Val(:none))
        @. Yₜ.c.ρq_tot += vtt - vtt_central
    end

    if isnothing(ᶠf¹²)
        # shallow atmosphere
        @. Yₜ.c.uₕ -=
            ᶜinterp(ᶠω¹² × (ᶠinterp(Y.c.ρ * ᶜJ) * ᶠu³)) / (Y.c.ρ * ᶜJ) +
            (ᶜf³ + ᶜω³) × CT12(ᶜu)
        @. Yₜ.f.u₃ -= ᶠω¹² × ᶠinterp(CT12(ᶜu)) + ᶠgradᵥ(ᶜK)
        for j in 1:n
            @. Yₜ.f.sgsʲs.:($$j).u₃ -=
                ᶠω¹²ʲs.:($$j) × ᶠinterp(CT12(ᶜuʲs.:($$j))) +
                ᶠgradᵥ(ᶜKʲs.:($$j) - ᶜinterp(ᶠKᵥʲs.:($$j)))
        end
    else
        # deep atmosphere
        @. Yₜ.c.uₕ -=
            ᶜinterp((ᶠf¹² + ᶠω¹²) × (ᶠinterp(Y.c.ρ * ᶜJ) * ᶠu³)) /
            (Y.c.ρ * ᶜJ) + (ᶜf³ + ᶜω³) × CT12(ᶜu)
        @. Yₜ.f.u₃ -= (ᶠf¹² + ᶠω¹²) × ᶠinterp(CT12(ᶜu)) + ᶠgradᵥ(ᶜK)
        for j in 1:n
            @. Yₜ.f.sgsʲs.:($$j).u₃ -=
                (ᶠf¹² + ᶠω¹²ʲs.:($$j)) × ᶠinterp(CT12(ᶜuʲs.:($$j))) +
                ᶠgradᵥ(ᶜKʲs.:($$j) - ᶜinterp(ᶠKᵥʲs.:($$j)))
        end
    end

    if use_prognostic_tke(turbconv_model) # advect_tke triggers allocations
        @. ᶜa_scalar = ᶜtke⁰ * draft_area(ᶜρa⁰, ᶜρ⁰)
        vtt = vertical_transport(ᶜρ⁰, ᶠu³⁰, ᶜa_scalar, dt, edmfx_upwinding)
        @. Yₜ.c.sgs⁰.ρatke += vtt
    end
end

edmfx_sgs_vertical_advection_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing

function edmfx_sgs_vertical_advection_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
)
    (; params) = p
    n = n_prognostic_mass_flux_subdomains(turbconv_model)
    (; dt) = p
    ᶜJ = Fields.local_geometry_field(Y.c).J
    (; edmfx_upwinding) = p.atmos.numerics
    (; ᶠu³ʲs, ᶠKᵥʲs, ᶜρʲs) = p.precomputed
    (; ᶠgradᵥ_ᶜΦ) = p.core

    ᶠz = Fields.coordinate_field(Y.f).z
    ᶜa_scalar = p.scratch.ᶜtemp_scalar
    ᶜu₃ʲ = p.scratch.ᶜtemp_C3
    ᶜKᵥʲ = p.scratch.ᶜtemp_scalar_2
    for j in 1:n
        # TODO: Add a biased GradientF2F operator in ClimaCore
        @. ᶜu₃ʲ = ᶜinterp(Y.f.sgsʲs.:($$j).u₃)
        @. ᶜKᵥʲ = ifelse(
            ᶜu₃ʲ.components.data.:1 > 0,
            ᶜleft_bias(ᶠKᵥʲs.:($$j)),
            ᶜright_bias(ᶠKᵥʲs.:($$j)),
        )
        # For the updraft u_3 equation, we assume the grid-mean to be hydrostatic
        # and calcuate the buoyancy term relative to the grid-mean density.
        @. Yₜ.f.sgsʲs.:($$j).u₃ -=
            (ᶠinterp(ᶜρʲs.:($$j) - Y.c.ρ) * ᶠgradᵥ_ᶜΦ) / ᶠinterp(ᶜρʲs.:($$j)) +
            ᶠgradᵥ(ᶜKᵥʲ)

        # buoyancy term in mse equation
        @. Yₜ.c.sgsʲs.:($$j).mse +=
            adjoint(CT3(ᶜinterp(Y.f.sgsʲs.:($$j).u₃))) *
            (ᶜρʲs.:($$j) - Y.c.ρ) *
            ᶜgradᵥ(CAP.grav(params) * ᶠz) / ᶜρʲs.:($$j)
    end

    for j in 1:n
        @. ᶜa_scalar = draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))
        vtt = vertical_transport(
            ᶜρʲs.:($j),
            ᶠu³ʲs.:($j),
            ᶜa_scalar,
            dt,
            edmfx_upwinding,
        )
        @. Yₜ.c.sgsʲs.:($$j).ρa += vtt

        va = vertical_advection(
            ᶠu³ʲs.:($j),
            Y.c.sgsʲs.:($j).mse,
            edmfx_upwinding,
        )
        @. Yₜ.c.sgsʲs.:($$j).mse += va

        va = vertical_advection(
            ᶠu³ʲs.:($j),
            Y.c.sgsʲs.:($j).q_tot,
            edmfx_upwinding,
        )
        @. Yₜ.c.sgsʲs.:($$j).q_tot += va
    end
end
