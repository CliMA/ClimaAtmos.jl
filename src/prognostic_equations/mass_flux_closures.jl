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

    (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = p.precomputed
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
    ᶜtke⁰ = @. lazy(specific_tke(Y.c.ρ, Y.c.sgs⁰.ρatke, ᶜρa⁰, turbconv_model))
    # scratch to prevent GPU Kernel parameter memory error
    ᶜmixing_length_field = p.scratch.ᶜtemp_scalar
    ᶜmixing_length_field .= ᶜmixing_length(Y, p)
    ᶜK_u = @. lazy(eddy_viscosity(turbconv_params, ᶜtke⁰, ᶜmixing_length_field))
    ᶜprandtl_nvec = @. lazy(
        turbulent_prandtl_number(params, ᶜlinear_buoygrad, ᶜstrain_rate_norm),
    )
    ᶜK_h = @. lazy(eddy_diffusivity(ᶜK_u, ᶜprandtl_nvec))

    ᶜh_tot = @. lazy(
        TD.total_specific_enthalpy(
            thermo_params,
            ᶜts,
            specific(Y.c.ρe_tot, Y.c.ρ),
        ),
    )
    ᶜmse = @. lazy(ᶜh_tot - ᶜK)
    for j in 1:n
        ᶜρʲ = ᶜρʲs.:($j)
        ᶜρaʲ = Y.c.sgsʲs.:($j).ρa
        ᶜmseʲ = Y.c.sgsʲs.:($j).mse
        ᶜq_totʲ = Y.c.sgsʲs.:($j).q_tot
        ᶜinv_ρaʲ =
            (@. lazy(specific(FT(1), ᶜρaʲ, FT(0), Y.c.ρ, turbconv_model)))

        @. Yₜ.c.sgsʲs.:($$j).mse -=
            ᶜdivᵥ_mse(-(ᶠinterp(ᶜρaʲ) * ᶠinterp(ᶜK_h) * ᶠgradᵥ(ᶜmseʲ))) *
            ᶜinv_ρaʲ
        @. Yₜ.c.sgsʲs.:($$j).q_tot -=
            ᶜdivᵥ_q_tot(-(ᶠinterp(ᶜρaʲ) * ᶠinterp(ᶜK_h) * ᶠgradᵥ(ᶜq_totʲ))) *
            ᶜinv_ρaʲ
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

    if p.atmos.edmfx_model.filter isa Val{true}
        for j in 1:n
            @. Yₜ.f.sgsʲs.:($$j).u₃ -=
                C3(min(Y.f.sgsʲs.:($$j).u₃.components.data.:1, 0)) / float(dt)
            @. Yₜ.c.sgsʲs.:($$j).ρa -= min(Y.c.sgsʲs.:($$j).ρa, 0) / float(dt)
        end
    end
end
