#####
##### Mass flux closures for EDMFX
#####

import StaticArrays as SA
import Thermodynamics.Parameters as TDP
import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields

"""
    buoyancy(ρ_ref, ρ, gradᵥ_Φ)

    Compute the buoyancy acceleration vector.

    Arguments:
    - `ρ_ref`: Reference density [kg/m^3].
    - `ρ`: Density [kg/m^3].
    - `gradᵥ_Φ`: Covariant3Vector — gradient of geopotential (i.e., gravitational acceleration) [m/s²]

    Returns:
    - Buoyancy acceleration as a Covariant3Vector [m/s²]
"""
function buoyancy(ρ_ref, ρ, gradᵥ_Φ)
    result = (ρ_ref - ρ) / ρ * gradᵥ_Φ
    return result
end

"""
    vertical_buoyancy_acceleration(ρ_ref, ρ, gradᵥ_Φ, local_geometry)

    Compute the signed vertical component of the buoyancy acceleration vector in physical units.

    Calculates the buoyancy acceleration vector due to a density anomaly and then
    projects it onto the local vertical direction using the model's covariant geometry.

    Arguments:
    - `ρ_ref`: Reference density [kg/m³]
    - `ρ`: Density [kg/m³]
    - `gradᵥ_Φ`: Covariant3Vector — gradient of geopotential (i.e., gravitational acceleration) [m/s²]
    - `local_geometry`: Local geometry object for projecting onto vertical direction

    Returns:
    - Scalar acceleration in the vertical direction [m/s²], positive when buoyancy acts upward
"""
function vertical_buoyancy_acceleration(ρ_ref, ρ, gradᵥ_Φ, local_geometry)
    # Compute the full buoyancy acceleration vector (Covariant3Vector)
    buoy_vector = buoyancy(ρ_ref, ρ, gradᵥ_Φ)
    # Project onto vertical axis and return signed scalar value
    return projected_vector_data(C3, buoy_vector, local_geometry)
end


"""
    draft_area(ρa, ρ)

    Calculates draft area fraction given ρa and ρ.

    Arguments:
    - `ρa`: The product of air density `ρ` and area fraction `a`
    - `ρ`: The air density

    Returns:
    - The draft area fraction
"""
function draft_area(ρa, ρ)
    return ρa / ρ
end

"""
   Return the virtual mass term of the pressure closure for updrafts [m/s2 * m]

   Inputs (everything defined on cell faces):
   - params - set with model parameters
   - ᶠbuoyʲ - covariant3 or contravariant3 updraft buoyancy
"""
function ᶠupdraft_nh_pressure_buoyancy(params, ᶠbuoyʲ)
    turbconv_params = CAP.turbconv_params(params)
    # factor multiplier for pressure buoyancy terms (effective buoyancy is (1-α_b))
    α_b = CAP.pressure_normalmode_buoy_coeff1(turbconv_params)
    return α_b * ᶠbuoyʲ
end

"""
   Return the drag term of the pressure closure for updrafts [m/s2 * m]

   Inputs (everything defined on cell faces):
   - params - set with model parameters
   - ᶠlg - local geometry (needed to compute the norm inside a local function)
   - ᶠu3ʲ, ᶠu3⁰ - covariant3 or contravariant3 velocity for updraft and environment.
                  covariant3 velocity is used in prognostic edmf, and contravariant3
                  velocity is used in diagnostic edmf.
   - scale height - an approximation for updraft top height
"""
function ᶠupdraft_nh_pressure_drag(params, ᶠlg, ᶠu3ʲ, ᶠu3⁰, scale_height)
    turbconv_params = CAP.turbconv_params(params)
    # factor multiplier for pressure drag
    α_d = CAP.pressure_normalmode_drag_coeff(turbconv_params)
    H_up_min = CAP.min_updraft_top(turbconv_params)

    # Independence of aspect ratio hardcoded: α₂_asp_ratio² = FT(0)
    # We also used to have advection term here: α_a * w_up * div_w_up
    return α_d * (ᶠu3ʲ - ᶠu3⁰) * CC.Geometry._norm(ᶠu3ʲ - ᶠu3⁰, ᶠlg) /
           max(scale_height, H_up_min)
end

edmfx_nh_pressure_drag_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing
function edmfx_nh_pressure_drag_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
)
    if p.atmos.edmfx_model.nh_pressure isa Val{true} &&
       p.atmos.sgs_nh_pressure_mode == Explicit()
        (; params) = p
        n = n_mass_flux_subdomains(turbconv_model)
        ᶠlg = Fields.local_geometry_field(Y.f)
        scale_height = CAP.R_d(params) * CAP.T_surf_ref(params) / CAP.grav(params)
        # assume zero environmental velocity
        for j in 1:n
            @. Yₜ.f.sgsʲs.:($$j).u₃ -= ᶠupdraft_nh_pressure_drag(
                params,
                ᶠlg,
                Y.f.sgsʲs.:($$j).u₃,
                C3(0),
                scale_height,
            )
        end
    end
end

edmfx_vertical_diffusion_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing

function edmfx_vertical_diffusion_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
)
    if p.atmos.edmfx_model.vertical_diffusion isa Val{true}
        (; params) = p
        (; ᶜρʲs) = p.precomputed
        FT = eltype(p.params)
        turbconv_params = CAP.turbconv_params(params)
        n = n_mass_flux_subdomains(turbconv_model)
        ᶜdivᵥ_mse = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(0)),
            bottom = Operators.SetValue(C3(0)),
        )
        ᶜdivᵥ_q_tot = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(0)),
            bottom = Operators.SetValue(C3(0)),
        )

        (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = p.precomputed
        ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
        # scratch to prevent GPU Kernel parameter memory error
        ᶜmixing_length_field = p.scratch.ᶜtemp_scalar
        ᶜmixing_length_field .= ᶜmixing_length(Y, p)
        ᶜK_u = @. lazy(eddy_viscosity(turbconv_params, ᶜtke, ᶜmixing_length_field))
        ᶜprandtl_nvec = @. lazy(
            turbulent_prandtl_number(params, ᶜlinear_buoygrad, ᶜstrain_rate_norm),
        )
        ᶜK_h = @. lazy(eddy_diffusivity(ᶜK_u, ᶜprandtl_nvec))

        for j in 1:n
            ᶜρʲ = ᶜρʲs.:($j)
            ᶜmseʲ = Y.c.sgsʲs.:($j).mse
            ᶜq_totʲ = Y.c.sgsʲs.:($j).q_tot
            # Note: For this and other diffusive tendencies, we should use ρaʲ instead of ρʲ,
            # but it causes stability issues when ρaʲ is small
            @. Yₜ.c.sgsʲs.:($$j).mse -=
                ᶜdivᵥ_mse(-(ᶠinterp(ᶜρʲ) * ᶠinterp(ᶜK_h) * ᶠgradᵥ(ᶜmseʲ))) / ᶜρʲ
            @. Yₜ.c.sgsʲs.:($$j).q_tot -=
                ᶜdivᵥ_q_tot(-(ᶠinterp(ᶜρʲ) * ᶠinterp(ᶜK_h) * ᶠgradᵥ(ᶜq_totʲ))) / ᶜρʲ
            @. Yₜ.c.sgsʲs.:($$j).ρa -=
                Y.c.sgsʲs.:($$j).ρa / (1 - Y.c.sgsʲs.:($$j).q_tot) *
                ᶜdivᵥ_q_tot(-(ᶠinterp(ᶜρʲ) * ᶠinterp(ᶜK_h) * ᶠgradᵥ(ᶜq_totʲ))) / ᶜρʲ
        end

        if p.atmos.microphysics_model isa
           Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}
            α_precip = CAP.α_vert_diff_tracer(params)
            ᶜρʲ = ᶜρʲs.:(1)
            ᶜdivᵥ_q = Operators.DivergenceF2C(
                top = Operators.SetValue(C3(FT(0))),
                bottom = Operators.SetValue(C3(FT(0))),
            )

            microphysics_tracers = (
                (@name(c.sgsʲs.:(1).q_lcl), FT(1)),
                (@name(c.sgsʲs.:(1).q_icl), FT(1)),
                (@name(c.sgsʲs.:(1).q_rai), α_precip),
                (@name(c.sgsʲs.:(1).q_sno), α_precip),
                (@name(c.sgsʲs.:(1).n_lcl), FT(1)),
                (@name(c.sgsʲs.:(1).n_rai), α_precip),
            )

            # TODO: using unrolled_foreach here allocates! (breaks the flame tests
            # even though they use 0M microphysics)
            # MatrixFields.unrolled_foreach(cloud_tracers) do χʲ_name
            for (χʲ_name, α) in microphysics_tracers
                MatrixFields.has_field(Y, χʲ_name) || continue

                ᶜχʲ = MatrixFields.get_field(Y, χʲ_name)
                ᶜχʲₜ = MatrixFields.get_field(Yₜ, χʲ_name)

                @. ᶜχʲₜ -= ᶜdivᵥ_q(-(ᶠinterp(ᶜρʲ) * ᶠinterp(ᶜK_h) * α * ᶠgradᵥ(ᶜχʲ))) / ᶜρʲ
            end
        end
    end
end

"""
    enforce_physical_constraints!(Y, p, t, turbconv_model)

Enforce physical constraints on the model state `Y` in-place.

This function is used as a callback and is not a tendency evaluation. It applies
local corrective updates to keep prognostic variables in a physically admissible
range.

Currently, this includes:
- For prognostic EDMF, handling non-positive updraft area fractions by
  immediately mixing the affected updraft state with the environment.
- For one- and two-moment microphysics, enforcing non-negative condensate
  masses.
- When total moisture is positive, rescaling condensate masses so that their
  sum does not exceed total moisture.

These corrections are intended to prevent nonphysical states such as negative
area fractions, negative condensate masses, or condensate mass exceeding the
available total moisture. Ideally, the need for this correction is minimized 
by the numerical scheme.
"""
enforce_physical_constraints!(Y, p, t, turbconv_model) = nothing

function enforce_physical_constraints!(Y, p, t, turbconv_model::PrognosticEDMFX)

    if p.atmos.edmfx_model.filter isa Val{true}
        (; ᶜh_tot, ᶜK, ᶜρʲs) = p.precomputed
        FT = eltype(p.params)
        n = n_mass_flux_subdomains(turbconv_model)

        # Microphysics constraints
        if p.atmos.microphysics_model isa
           Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}

            ρq_cond = p.scratch.ᶜtemp_scalar
            ratio = p.scratch.ᶜtemp_scalar_2
            @. Y.c.ρq_lcl = max(0, Y.c.ρq_lcl)
            @. Y.c.ρq_icl = max(0, Y.c.ρq_icl)
            @. Y.c.ρq_rai = max(0, Y.c.ρq_rai)
            @. Y.c.ρq_sno = max(0, Y.c.ρq_sno)

            @. ρq_cond = Y.c.ρq_lcl + Y.c.ρq_icl + Y.c.ρq_rai + Y.c.ρq_sno
            @. ratio = ifelse(
                ρq_cond > eps(FT),
                min(1, max(0, Y.c.ρq_tot) / ρq_cond),
                1,
            )

            @. Y.c.ρq_lcl *= ratio
            @. Y.c.ρq_icl *= ratio
            @. Y.c.ρq_rai *= ratio
            @. Y.c.ρq_sno *= ratio
        end

        # Apply updraft constraints
        microphysics_tracers = (
            (@name(c.sgsʲs.:(1).q_lcl), @name(c.ρq_lcl)),
            (@name(c.sgsʲs.:(1).q_icl), @name(c.ρq_icl)),
            (@name(c.sgsʲs.:(1).q_rai), @name(c.ρq_rai)),
            (@name(c.sgsʲs.:(1).q_sno), @name(c.ρq_sno)),
            (@name(c.sgsʲs.:(1).n_lcl), @name(c.ρn_lcl)),
            (@name(c.sgsʲs.:(1).n_rai), @name(c.ρn_rai)),
        )
        for j in 1:n
            # clip updraft velocity and area fraction to zero if they are negative
            @. Y.c.sgsʲs.:($$j).ρa = max(0, min(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)))
            @. Y.f.sgsʲs.:($$j).u₃ =
                C3(max(Y.f.sgsʲs.:($$j).u₃.components.data.:1, 0))

            # clip updraft velocity to zero if the face-averaged area fraction is negligible.
            @. Y.f.sgsʲs.:($$j).u₃ =
                ifelse(
                    ᶠinterp(Y.c.sgsʲs.:($$j).ρa) < ϵ_numerics(FT),
                    C3(0),
                    Y.f.sgsʲs.:($$j).u₃,
                )

            # mix updraft mse and q_tot with the grid mean values if area fraction is negligible
            @. Y.c.sgsʲs.:($$j).mse =
                ifelse(
                    Y.c.sgsʲs.:($$j).ρa < ϵ_numerics(FT),
                    ᶜh_tot - ᶜK,
                    Y.c.sgsʲs.:($$j).mse,
                )
            @. Y.c.sgsʲs.:($$j).q_tot = ifelse(
                Y.c.sgsʲs.:($$j).ρa < ϵ_numerics(FT),
                specific(Y.c.ρq_tot, Y.c.ρ),
                # ensure mass conservation in subdomain decomposition ρaχʲ < ρχ
                min(
                    max(0, Y.c.sgsʲs.:($$j).q_tot),
                    max(0, Y.c.ρq_tot) / Y.c.sgsʲs.:($$j).ρa,
                ),
            )

            # mix the rest of the updraft microphysics tracers
            MatrixFields.unrolled_foreach(microphysics_tracers) do (χʲ_name, ρχ_name)
                MatrixFields.has_field(Y, χʲ_name) || return
                ᶜχʲ = MatrixFields.get_field(Y, χʲ_name)
                ᶜρχ = MatrixFields.get_field(Y, ρχ_name)
                @. ᶜχʲ = ifelse(
                    Y.c.sgsʲs.:($$j).ρa < ϵ_numerics(FT),
                    specific(ᶜρχ, Y.c.ρ),
                    # ensure mass conservation in subdomain decomposition ρaχʲ < ρχ
                    min(max(0, ᶜχʲ), max(0, ᶜρχ) / Y.c.sgsʲs.:($$j).ρa),
                )
            end
        end
        set_precomputed_quantities!(Y, p, t)
    end
end
