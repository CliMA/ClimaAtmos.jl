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
    vertical_buoyancy_acceleration(ρ_diff, gradᵥ_Φ, local_geometry)

    Compute the signed vertical component of the buoyancy acceleration vector in physical units.

    Calculates the buoyancy acceleration vector due to a density anomaly and then
    projects it onto the local vertical direction using the model's covariant geometry.

    Arguments:
    - `ρ_ref`: Reference density [kg/m³]
    - `ρ`: Density [kg/m³]
    - `gradᵥ_Φ`: Covariant3Vector — gradient of geopotential (i.e., gravitational acceleration) [m/s²]
    - `local_geometry`: Local geometry object for projecting onto vertical direction
    - `ρ_diff`: Normalized density difference `(ρ - ρ_ref) / ρ` [-].

    Returns:
    - Scalar acceleration in the vertical direction [m/s²], positive when buoyancy acts upward
"""
function vertical_buoyancy_acceleration(ρ_ref, ρ, gradᵥ_Φ, local_geometry)
    # Compute the full buoyancy acceleration vector (Covariant3Vector)
    buoy_vector = buoyancy(ρ_ref, ρ, gradᵥ_Φ)
    # Project onto vertical axis and return signed scalar value
    return projected_vector_data(C3, buoy_vector, local_geometry)
end
function vertical_buoyancy_acceleration(ρ_diff, gradᵥ_Φ, local_geometry)
    # Compute the full buoyancy acceleration vector (Covariant3Vector)
    buoy_vector = -1 * ρ_diff * gradᵥ_Φ
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
Return the drag term of the pressure closure for updrafts [m/s2 * m].
This is a simplified version where the length scale is fixed at scale height.
This is only used in diagnostic EDMF.

Inputs (everything defined on cell faces):

  - params - set with model parameters
  - ᶠlg - local geometry (needed to compute the norm inside a local function)
  - ᶠu3ʲ, ᶠu3⁰ - covariant3 or contravariant3 velocity for updraft and environment.
    covariant3 velocity is used in prognostic edmf, and contravariant3
    velocity is used in diagnostic edmf.
"""
function ᶠupdraft_nh_pressure_drag(params, ᶠlg, ᶠu3ʲ, ᶠu3⁰)
    turbconv_params = CAP.turbconv_params(params)
    α_d = CAP.pressure_normalmode_drag_coeff(turbconv_params)
    H_up_min = CAP.min_updraft_top(turbconv_params)
    scale_height = CAP.R_d(params) * CAP.T_surf_ref(params) / CAP.grav(params)
    # We also used to have advection term here: α_a * w_up * div_w_up
    return α_d * (ᶠu3ʲ - ᶠu3⁰) * CC.Geometry._norm(ᶠu3ʲ - ᶠu3⁰, ᶠlg) /
           max(scale_height, H_up_min)
end

"""
    surface_mass_flux_coefficient(buoyancy_flux, z_i, ustar, a_s_max, c_u)

Effective surface area fraction of the convective mass flux entering the
EDMF updraft,

    a_s = a_s_max · w*³ / (w*³ + c_u · u*³),

where `w*³ = max(z_i · ⟨w'b'⟩_s, 0)` (`w*` is the Deardorff convective
velocity scale), `u*` is the friction velocity, and `c_u` is an
O(1) tuning coefficient setting the relative weight of shear vs.
buoyancy production in the blend (TOML key:
`EDMF_sfc_mass_flux_ustar_coeff`). The factor `w*³/(w*³+c_u·u*³)`
interpolates smoothly between free convection (`a_s → a_s_max`) and
shear-only conditions (`a_s → 0`). `a_s_max` is the asymptotic plume
area fraction in the free-convection limit. Used both to set the
surface mass flux magnitude (via [`surface_mass_flux`](@ref)) and to
specify the percentile range from which the high-tail buoyant scalar
values are sampled at the surface.
"""
@inline function surface_mass_flux_coefficient(
    buoyancy_flux,
    z_i,
    ustar,
    a_s_max,
    c_u,
)
    FT = typeof(ustar)
    w3 = max(z_i * buoyancy_flux, FT(0))
    return a_s_max * w3 / max(eps(FT), w3 + c_u * ustar^3)
end

"""
    surface_mass_flux(buoyancy_flux, ρ, z_i, ustar, a_s_max, c_u)

Surface EDMF updraft mass flux [kg/m²/s] entering the first cell:

    F_surf = a_s · ρ · w*,

with `a_s` given by [`surface_mass_flux_coefficient`](@ref) and
`w* = cbrt(max(z_i · ⟨w'b'⟩_s, 0))`. Returns zero in stable boundary
layers (`⟨w'b'⟩_s ≤ 0`).
"""
@inline function surface_mass_flux(buoyancy_flux, ρ, z_i, ustar, a_s_max, c_u)
    FT = typeof(ρ)
    w_star = cbrt(max(z_i * buoyancy_flux, FT(0)))
    a_s = surface_mass_flux_coefficient(buoyancy_flux, z_i, ustar, a_s_max, c_u)
    return a_s * ρ * w_star
end

# Private helper: clips grid-mean condensate tracers to non-negative values and
# rescales the condensate sum so it cannot exceed the available total moisture.
function enforce_grid_mean_microphysics_constraints!(Y, p, t)
    FT = eltype(p.params)
    ρq_cond = p.scratch.ᶜtemp_scalar
    ratio = p.scratch.ᶜtemp_scalar_2
    @. Y.c.ρq_lcl = max(FT(0), Y.c.ρq_lcl)
    @. Y.c.ρq_icl = max(FT(0), Y.c.ρq_icl)
    @. Y.c.ρq_rai = max(FT(0), Y.c.ρq_rai)
    @. Y.c.ρq_sno = max(FT(0), Y.c.ρq_sno)

    @. ρq_cond = Y.c.ρq_lcl + Y.c.ρq_icl + Y.c.ρq_rai + Y.c.ρq_sno
    @. ratio = ifelse(
        (ρq_cond > ϵ_numerics(FT)) & (Y.c.ρq_tot > ϵ_numerics(FT)),
        min(FT(1), Y.c.ρq_tot / ρq_cond),
        FT(0),
    )
    @. Y.c.ρq_lcl *= ratio
    @. Y.c.ρq_icl *= ratio
    @. Y.c.ρq_rai *= ratio
    @. Y.c.ρq_sno *= ratio
    return nothing
end

# Private helper: clips prognostic updraft area fraction and vertical velocity,
# relaxes updraft mse/q_tot toward the grid mean when ρa is negligible, and
# relaxes updraft microphysics tracers (q_lcl, q_icl, q_rai, q_sno, n_lcl, n_rai)
# toward the grid mean while enforcing the subdomain mass conservation bound ρaχʲ < ρχ.
# The microphysics tracer block is a no-op for 0M (has_field returns false).
# No-op when n_prognostic_mass_flux_subdomains == 0 (EDOnlyEDMFX, etc.).
function enforce_edmf_updraft_constraints!(Y, p, t, turbconv_model)
    FT = eltype(p.params)
    n = n_prognostic_mass_flux_subdomains(turbconv_model)
    n == 0 && return nothing
    (; ᶜh_tot, ᶜK, ᶜρʲs) = p.precomputed
    for j in 1:n
        # clip updraft area fraction and vertical velocity to non-negative values
        @. Y.c.sgsʲs.:($$j).ρa = max(0, min(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)))
        @. Y.f.sgsʲs.:($$j).u₃ =
            C3(max(Y.f.sgsʲs.:($$j).u₃.components.data.:1, 0))

        # clip updraft velocity to zero when face-averaged area fraction is negligible
        @. Y.f.sgsʲs.:($$j).u₃ = ifelse(
            ᶠinterp(Y.c.sgsʲs.:($$j).ρa) < ϵ_numerics(FT),
            C3(0),
            Y.f.sgsʲs.:($$j).u₃,
        )

        # relax updraft mse and q_tot toward the grid mean when ρa is negligible
        @. Y.c.sgsʲs.:($$j).mse = ifelse(
            Y.c.sgsʲs.:($$j).ρa < ϵ_numerics(FT),
            ᶜh_tot - ᶜK,
            Y.c.sgsʲs.:($$j).mse,
        )
        @. Y.c.sgsʲs.:($$j).q_tot = ifelse(
            Y.c.sgsʲs.:($$j).ρa < ϵ_numerics(FT),
            specific(Y.c.ρq_tot, Y.c.ρ),
            # ensure mass conservation: ρaχʲ < ρχ
            min(
                max(0, Y.c.sgsʲs.:($$j).q_tot),
                max(0, Y.c.ρq_tot) / Y.c.sgsʲs.:($$j).ρa,
            ),
        )

        # Auto-discovered SGS tracers: relax toward grid mean when ρa is
        # negligible; enforce mass conservation bound ρaχʲ < ρχ.
        for χ_name in sgs_tracer_names(Y)
            ρχ_name = get_ρχ_name(χ_name)
            MatrixFields.has_field(Y.c, ρχ_name) || continue
            ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:(1), χ_name)
            ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
            @. ᶜχʲ = ifelse(
                Y.c.sgsʲs.:($$j).ρa < ϵ_numerics(FT),
                specific(ᶜρχ, Y.c.ρ),
                # ensure mass conservation: ρaχʲ < ρχ
                min(max(0, ᶜχʲ), max(0, ᶜρχ) / Y.c.sgsʲs.:($$j).ρa),
            )
        end
    end
    return nothing
end

"""
    enforce_physical_constraints!(Y, p, t, atmos)

Enforce physical consistency of the model state by calling the appropriate
constraint helpers based on the active microphysics and turbulence-convection
models.
"""
function enforce_physical_constraints!(Y, p, t, atmos::AtmosModel)
    # Grid-mean microphysics: non-negativity + condensate ≤ total moisture.
    if atmos.microphysics_model isa
       Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}
        enforce_grid_mean_microphysics_constraints!(Y, p, t)
    end

    # EDMF updraft constraints: only active when the filter flag is enabled.
    # Each helper is a no-op for EDOnlyEDMFX (n_prognostic_mass_flux_subdomains == 0).
    if atmos.turbconv_model isa AbstractEDMF &&
       atmos.edmfx_model.filter isa Val{true}
        enforce_edmf_updraft_constraints!(Y, p, t, atmos.turbconv_model)
    end

    return nothing
end
