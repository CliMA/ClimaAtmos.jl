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
    (; ᶠnh_pressure₃_dragʲs) = p.precomputed
    n = n_mass_flux_subdomains(turbconv_model)
    for j in 1:n
        @. Yₜ.f.sgsʲs.:($$j).u₃ -= ᶠnh_pressure₃_dragʲs.:($$j)
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
        (; ᶜts, ᶜK, ᶜρʲs) = p.precomputed
        FT = eltype(p.params)
        thermo_params = CAP.thermodynamics_params(params)
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
        ᶜinv_ρ̂ = (@. lazy(
            specific(
                FT(1),
                Y.c.sgsʲs.:(1).ρa,
                FT(0),
                ᶜρʲs.:(1),
                turbconv_model,
            ),
        ))

        (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = p.precomputed
        ᶜtke⁰ = @. lazy(specific(Y.c.sgs⁰.ρatke, Y.c.ρ))
        # scratch to prevent GPU Kernel parameter memory error
        ᶜmixing_length_field = p.scratch.ᶜtemp_scalar
        ᶜmixing_length_field .= ᶜmixing_length(Y, p)
        ᶜK_u = @. lazy(eddy_viscosity(turbconv_params, ᶜtke⁰, ᶜmixing_length_field))
        ᶜprandtl_nvec = @. lazy(
            turbulent_prandtl_number(params, ᶜlinear_buoygrad, ᶜstrain_rate_norm),
        )
        ᶜK_h = @. lazy(eddy_diffusivity(ᶜK_u, ᶜprandtl_nvec))

        for j in 1:n
            ᶜρʲ = ᶜρʲs.:($j)
            ᶜρaʲ = Y.c.sgsʲs.:($j).ρa
            ᶜmseʲ = Y.c.sgsʲs.:($j).mse
            ᶜq_totʲ = Y.c.sgsʲs.:($j).q_tot
            # Note: For this and other diffusive tendencies, we should use ρaʲ instead of ρʲ,
            # but it causes stability issues when ρaʲ is small
            @. Yₜ.c.sgsʲs.:($$j).mse -=
                ᶜdivᵥ_mse(-(ᶠinterp(ᶜρʲ) * ᶠinterp(ᶜK_h) * ᶠgradᵥ(ᶜmseʲ))) / ᶜρʲ
            @. Yₜ.c.sgsʲs.:($$j).q_tot -=
                ᶜdivᵥ_q_tot(-(ᶠinterp(ᶜρʲ) * ᶠinterp(ᶜK_h) * ᶠgradᵥ(ᶜq_totʲ))) / ᶜρʲ
        end

        if p.atmos.moisture_model isa NonEquilMoistModel && (
            p.atmos.microphysics_model isa Microphysics1Moment ||
            p.atmos.microphysics_model isa Microphysics2Moment
        )
            α_precip = CAP.α_vert_diff_tracer(params)
            ᶜρaʲ = Y.c.sgsʲs.:(1).ρa
            ᶜdivᵥ_q = Operators.DivergenceF2C(
                top = Operators.SetValue(C3(FT(0))),
                bottom = Operators.SetValue(C3(FT(0))),
            )

            microphysics_tracers = (
                (@name(c.sgsʲs.:(1).q_liq), FT(1)),
                (@name(c.sgsʲs.:(1).q_ice), FT(1)),
                (@name(c.sgsʲs.:(1).q_rai), α_precip),
                (@name(c.sgsʲs.:(1).q_sno), α_precip),
                (@name(c.sgsʲs.:(1).n_liq), FT(1)),
                (@name(c.sgsʲs.:(1).n_rai), α_precip),
            )

            # TODO: using unrolled_foreach here allocates! (breaks the flame tests
            # even though they use 0M microphysics)
            # MatrixFields.unrolled_foreach(cloud_tracers) do χʲ_name
            for (χʲ_name, α) in microphysics_tracers
                MatrixFields.has_field(Y, χʲ_name) || continue

                ᶜχʲ = MatrixFields.get_field(Y, χʲ_name)
                ᶜχʲₜ = MatrixFields.get_field(Yₜ, χʲ_name)

                @. ᶜχʲₜ -=
                    ᶜinv_ρ̂ * ᶜdivᵥ_q(-(ᶠinterp(ᶜρaʲ) * ᶠinterp(ᶜK_h) * α * ᶠgradᵥ(ᶜχʲ)))
            end
        end
    end
end

"""
    edmfx_filter_tendency!(Yₜ, Y, p, t, turbconv_model)

Apply EDMF filters:
 - Relax u_3 to zero when it is negative
 - Relax ρa to zero when it is negative

This function modifies the tendency `Yₜ` in place based on the current state `Y`,
parameters `p`, time `t`, and the turbulence convection model `turbconv_model`.
It specifically targets the vertical velocity (`u₃`) and the product of density and area fraction (`ρa`)
for each sub-domain in the EDMFX model.
"""
edmfx_filter_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing

function edmfx_filter_tendency!(Yₜ, Y, p, t, turbconv_model::PrognosticEDMFX)

    n = n_mass_flux_subdomains(turbconv_model)
    (; dt) = p
    FT = eltype(p.params)

    if p.atmos.edmfx_model.filter isa Val{true}
        for j in 1:n
            @. Yₜ.f.sgsʲs.:($$j).u₃ -=
                C3(min(Y.f.sgsʲs.:($$j).u₃.components.data.:1, 0)) / FT(dt)
            @. Yₜ.c.sgsʲs.:($$j).ρa -= min(Y.c.sgsʲs.:($$j).ρa, 0) / FT(dt)
        end
    end
end
