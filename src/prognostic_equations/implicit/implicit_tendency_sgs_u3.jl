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
    ᶜu₃ʲ = p.scratch.ᶜtemp_C3
    ᶜKᵥʲ = p.scratch.ᶜtemp_scalar_2

    n = n_mass_flux_subdomains(turbconv_model)

    # implicit_vertical_advection_tendency!
    if turbconv_model isa PrognosticEDMFX
        for j in 1:n
            rst_u₃ʲ = rayleigh_sponge_tendency_u₃(Y.f.sgsʲs.:($j).u₃, rayleigh_sponge)
            @. Yₜ.f.sgsʲs.:($$j).u₃ += rst_u₃ʲ
        end
    end

    # edmfx_sgs_vertical_advection_tendency!
    if p.atmos.sgs_adv_mode == Implicit()
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
            # We also include the buoyancy term in the nonhydrostatic pressure closure here.
            @. Yₜ.f.sgsʲs.:($$j).u₃ -=
                (1 - α_b) * ᶠρ_diffʲs.:($$j) * ᶠgradᵥ_ᶜΦ + ᶠgradᵥ(ᶜKᵥʲ)
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

function initialize_sgs_u₃!(Y, p, γdt)
    ᶜdz = Fields.Δz_field(axes(Y.c))
    @. Y.f.sgsʲs.:(1).u₃ =
        C3(max(Y.f.sgsʲs.:(1).u₃.components.data.:1, 5 * ᶠinterp(ᶜdz)))
end
