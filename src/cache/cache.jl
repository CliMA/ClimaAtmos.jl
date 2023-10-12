using LinearAlgebra: ×, norm, dot, Adjoint

import .Parameters as CAP
using ClimaCore: Operators, Fields, Limiters, Geometry, Spaces

import ClimaComms
using ClimaCore.Geometry: ⊗

import Thermodynamics as TD

using ClimaCore.Utilities: half

import ClimaCore.Fields: ColumnField

# Functions on which the model depends:
# CAP.R_d(params)         # dry specific gas constant
# CAP.kappa_d(params)     # dry adiabatic exponent
# CAP.T_triple(params)    # triple point temperature of water
# CAP.MSLP(params)        # reference pressure
# CAP.grav(params)        # gravitational acceleration
# CAP.Omega(params)       # rotation rate (only used if space is spherical)
# CAP.cv_d(params)        # dry isochoric specific heat capacity
# The value of cv_d is implied by the values of R_d and kappa_d

# The model also depends on f_plane_coriolis_frequency(params)
# This is a constant Coriolis frequency that is only used if space is flat
function default_cache(
    Y,
    params,
    atmos,
    spaces,
    numerics,
    simulation,
    surface_setup,
)

    FT = eltype(params)

    ᶜcoord = Fields.local_geometry_field(Y.c).coordinates
    grav = FT(CAP.grav(params))
    ᶜΦ = grav .* ᶜcoord.z

    if atmos.numerics.use_reference_state
        R_d = FT(CAP.R_d(params))
        MSLP = FT(CAP.MSLP(params))
        T_ref = FT(255)
        ᶜρ_ref = @. MSLP * exp(-grav * ᶜcoord.z / (R_d * T_ref)) / (R_d * T_ref)
        ᶜp_ref = @. ᶜρ_ref * R_d * T_ref
    else
        ᶜρ_ref = zero(ᶜΦ)
        ᶜp_ref = zero(ᶜΦ)
    end

    if eltype(ᶜcoord) <: Geometry.LatLongZPoint
        Ω = CAP.Omega(params)
        ᶜf = @. 2 * Ω * sind(ᶜcoord.lat)
    else
        f = CAP.f_plane_coriolis_frequency(params)
        ᶜf = map(_ -> f, ᶜcoord)
    end
    ᶜf = @. CT3(Geometry.WVector(ᶜf))

    quadrature_style = Spaces.horizontal_space(axes(Y.c)).quadrature_style
    do_dss = quadrature_style isa Spaces.Quadratures.GLL
    ghost_buffer =
        !do_dss ? (;) :
        (; c = Spaces.create_dss_buffer(Y.c), f = Spaces.create_dss_buffer(Y.f))

    limiter =
        isnothing(numerics.limiter) ? nothing :
        numerics.limiter(similar(Y.c, FT))

    net_energy_flux_toa = [Geometry.WVector(FT(0))]
    net_energy_flux_sfc = [Geometry.WVector(FT(0))]

    default_cache = (;
        is_init = Ref(true),
        simulation,
        atmos,
        sfc_setup = surface_setup(params),
        limiter,
        ᶜΦ,
        ᶠgradᵥ_ᶜΦ = ᶠgradᵥ.(ᶜΦ),
        ᶜρ_ref,
        ᶜp_ref,
        ᶜT = similar(Y.c, FT),
        ᶜf,
        ∂ᶜK_∂ᶠu₃ = similar(Y.c, BidiagonalMatrixRow{Adjoint{FT, CT3{FT}}}),
        params,
        do_dss,
        ghost_buffer,
        net_energy_flux_toa,
        net_energy_flux_sfc,
        env_thermo_quad = SGSQuadrature(FT),
        precomputed_quantities(Y, atmos)...,
        scratch = temporary_quantities(Y, atmos),
        hyperdiffusion_cache(Y, atmos, do_dss)...,
    )
    set_precomputed_quantities!(Y, default_cache, FT(0))
    default_cache.is_init[] = false
    return default_cache
end


# TODO: flip order so that NamedTuple() is fallback.
function additional_cache(Y, default_cache, params, atmos, dt)
    (; forcing_type, radiation_mode, turbconv_model) = atmos

    radiation_cache = if radiation_mode isa RRTMGPI.AbstractRRTMGPMode
        radiation_model_cache(
            Y,
            default_cache,
            params,
            radiation_mode;
            data_loader = rrtmgp_data_loader,
        )
    else
        radiation_model_cache(Y, params, radiation_mode)
    end

    return merge(
        (;
            rayleigh_sponge = rayleigh_sponge_cache(Y, atmos),
            viscous_sponge = viscous_sponge_cache(Y, atmos),
            precipitation = precipitation_cache(Y, atmos),
            large_scale_advection = large_scale_advection_cache(Y, atmos),
        ),
        subsidence_cache(Y, atmos.subsidence),
        edmf_coriolis_cache(Y, atmos.edmf_coriolis),
        forcing_cache(Y, forcing_type),
        radiation_cache,
        non_orographic_gravity_wave_cache(
            atmos.non_orographic_gravity_wave,
            atmos.model_config,
            Y,
        ),
        orographic_gravity_wave_cache(
            atmos.orographic_gravity_wave,
            Y,
            CAP.planet_radius(params),
        ),
        (; Δt = dt),
        (; turbconv_model),
    )
end
