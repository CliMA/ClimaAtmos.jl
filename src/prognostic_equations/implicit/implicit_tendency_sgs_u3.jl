#####
##### Implicit tendencies
#####

import ClimaCore
import ClimaCore: Fields, Geometry

NVTX.@annotate function implicit_tendency_sgs_u₃!(Yₜ, Y, p, t)
    fill_with_nans!(p)
    Yₜ .= zero(eltype(Yₜ))

    (; params) = p
    (; turbconv_model, rayleigh_sponge) = p.atmos
    (; ᶠKᵥʲs, ᶠρ_diffʲs) = p.precomputed
    (; ᶠgradᵥ_ᶜΦ) = p.core
    (; ᶜturb_entrʲs, ᶜentrʲs) = p.precomputed
    (; ᶠu₃⁰) = p.precomputed

    turbconv_params = CAP.turbconv_params(params)
    α_b = CAP.pressure_normalmode_buoy_coeff1(turbconv_params)
    ᶠlg = Fields.local_geometry_field(Y.f)

    n = n_mass_flux_subdomains(turbconv_model)

    # implicit_vertical_advection_tendency!
    for j in 1:n
        rst_u₃ʲ = rayleigh_sponge_tendency_u₃(Y.f.sgsʲs.:($j).u₃, rayleigh_sponge)
        @. Yₜ.f.sgsʲs.:($$j).u₃ += rst_u₃ʲ
    end

    # edmfx_sgs_vertical_advection_tendency!
    if p.atmos.sgs_adv_mode == Implicit()
        for j in 1:n
            # For the updraft u_3 equation, we assume the grid-mean to be hydrostatic
            # and calcuate the buoyancy term relative to the grid-mean density.
            # We also include the buoyancy term in the nonhydrostatic pressure closure here.
            # TODO: Add a biased GradientF2F operator in ClimaCore
            @. Yₜ.f.sgsʲs.:($$j).u₃ -=
                (1 - α_b) * ᶠρ_diffʲs.:($$j) * ᶠgradᵥ_ᶜΦ + ᶠgradᵥ(ᶜleft_bias(ᶠKᵥʲs.:($$j)))
        end
    end

    # edmfx_entr_detr_tendency!
    if p.atmos.sgs_entr_detr_mode == Implicit()
        for j in 1:n
            ᶜentrʲ = ᶜentrʲs.:($j)
            ᶜturb_entrʲ = ᶜturb_entrʲs.:($j)
            @. Yₜ.f.sgsʲs.:($$j).u₃ +=
                (ᶠinterp(ᶜentrʲ) .+ ᶠinterp(ᶜturb_entrʲ)) *
                (ᶠu₃⁰ - Y.f.sgsʲs.:($$j).u₃)
        end
    end

    # edmfx_nh_pressure_drag_tendency!
    if p.atmos.edmfx_model.nh_pressure isa Val{true} &&
       p.atmos.sgs_nh_pressure_mode == Implicit()
        scale_height = CAP.R_d(params) * CAP.T_surf_ref(params) / CAP.grav(params)
        for j in 1:n
            @. Yₜ.f.sgsʲs.:($$j).u₃ -= ᶠupdraft_nh_pressure_drag(
                params,
                ᶠlg,
                Y.f.sgsʲs.:($$j).u₃,
                ᶠu₃⁰,
                scale_height,
            )
        end
    end

    # NOTE: This will zero out all momentum tendencies in the edmfx advection test
    # DO NOT add additional velocity tendencies after this function
    zero_velocity_tendency!(Yₜ, Y, p, t)

    return nothing
end

"""
    initialize_sgs_u₃!(Y, p, dtγ)

Initialize the stage value of SGS/updraft vertical velocity u₃ for each
EDMFX mass-flux subdomain.

This routine solves a reduced nonlinear stage equation for the updraft
vertical velocity based on

    ∂w/∂t + ∂k/∂z = b + ε (w₀ − w) − (α/H) (w − w₀)².

At an IMEX/ARK stage this becomes an algebraic equation for the new stage
value. For a single updraft we use the approximation

    w₀ − w = (ρ / ρa⁰)(w_env − w) ≈ −(ρ / ρa⁰) w,

so that entrainment acts as a linear sink in w and pressure drag becomes a
quadratic sink in w². After rearrangement the equation takes the form

    a w² + b w + c = 0.

The coefficients include contributions from the implicit stage term,
entrainment/detrainment, nonhydrostatic pressure drag, optional Rayleigh
damping, and optionally implicit vertical advection.

In the discretization used here the prognostic variable `u₃` is the
covariant vertical velocity component. The physical vertical velocity `w`
is related to it through division by Δz, and this scaling is applied when
forming the quadratic equation coefficients.

If SGS advection is implicit, the equations are vertically coupled through
the previously solved face value and are computed with
`Operators.column_accumulate!`. Otherwise each face-local quadratic is solved
independently using a numerically stable quadratic root formula.
"""
function initialize_sgs_u₃!(Y, p, dtγ)

    if p.atmos.turbconv_model isa PrognosticEDMFX

        (; params) = p
        (; turbconv_model, rayleigh_sponge) = p.atmos
        (; ᶠρ_diffʲs) = p.precomputed
        (; ᶠgradᵥ_ᶜΦ) = p.core
        (; ᶜturb_entrʲs, ᶜentrʲs) = p.precomputed
        FT = eltype(p.params)

        turbconv_params = CAP.turbconv_params(params)
        α_b = CAP.pressure_normalmode_buoy_coeff1(turbconv_params)
        α_d = CAP.pressure_normalmode_drag_coeff(turbconv_params)
        H_up_min = CAP.min_updraft_top(turbconv_params)
        scale_height = CAP.R_d(params) * CAP.T_surf_ref(params) / CAP.grav(params)
        drag_coeff = α_d / max(H_up_min, scale_height)

        # Approximation factor used in w₀ - w ≈ -(ρ / ρa⁰) w (single-updraft case).
        # For multiple updrafts we approximate ρ / ρa⁰ ≈ 1, which implies w₀ ≈ 0.
        ᶜρ_over_ρa⁰ = p.scratch.ᶜtemp_scalar
        @. ᶜρ_over_ρa⁰ = Y.c.ρ / ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model)

        ᶠdz = Fields.Δz_field(axes(Y.f))

        # Face-local coefficients of the rearranged quadratic equation
        #     a w² + b w + c = 0.
        ᶠa = p.scratch.ᶠtemp_scalar
        ᶠb = p.scratch.ᶠtemp_scalar_2
        ᶠc = p.scratch.ᶠtemp_scalar_3

        n = n_mass_flux_subdomains(turbconv_model)
        for j in 1:n

            # Start from the implicit stage term (w - wᵢ) / (γ Δt).
            @. ᶠa = 0
            @. ᶠb = 1 / dtγ
            @. ᶠc = -1 * (Y.f.sgsʲs.:($$j).u₃.components.data.:1 / dtγ)

            # Implicit entrainment/detrainment contributes a linear sink in w.
            if p.atmos.sgs_entr_detr_mode == Implicit()
                @. ᶠb += ᶠinterp((ᶜentrʲs.:($$j) + ᶜturb_entrʲs.:($$j)) * ᶜρ_over_ρa⁰)
            end

            # Implicit NH pressure drag contributes a quadratic sink in w².
            if p.atmos.edmfx_model.nh_pressure isa Val{true} &&
               p.atmos.sgs_nh_pressure_mode == Implicit()
                @. ᶠa += ᶠinterp(drag_coeff * ᶜρ_over_ρa⁰ * ᶜρ_over_ρa⁰) / ᶠdz
            end

            # Optional Rayleigh sponge adds extra linear damping near the top.
            if !isnothing(rayleigh_sponge)
                ᶠz = Fields.coordinate_field(Y.f.u₃).z
                zmax = z_max(axes(ᶠz))
                @. ᶠb += β_rayleigh_u₃(rayleigh_sponge, ᶠz, zmax)
            end

            if p.atmos.sgs_adv_mode == Implicit()
                # Implicit advection adds a local w² term and couples each face
                # to the previously solved face through w_prev².
                @. ᶠa += (1 / ᶠdz)^2 / 2
                @. ᶠc += (1 - α_b) * ᶠρ_diffʲs.:($$j) * ᶠgradᵥ_ᶜΦ.components.data.:1

                input = @. lazy(tuple(ᶠa, ᶠb, ᶠc, ᶠdz))
                Operators.column_accumulate!(
                    Y.f.sgsʲs.:($j).u₃,
                    input;
                    init = C3(FT(0)),
                ) do u₃_prev_face, (a_face, b_face, c_face, dz_face)

                    return C3(
                        (
                            -b_face + sqrt(
                                b_face * b_face -
                                4 * a_face *
                                min(0, c_face - (u₃_prev_face[1] / dz_face)^2 / 2),
                            )
                        ) / (2 * a_face),
                    )

                end
            else
                # Safe quadratic root to avoid cancellation and stay well behaved
                # when a is small.
                @. Y.f.sgsʲs.:($$j).u₃ =
                    C3(-2 * min(0, ᶠc) / (ᶠb + sqrt(ᶠb * ᶠb - 4 * ᶠa * min(0, ᶠc))))
            end
        end

    end

end
