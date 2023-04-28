using LinearAlgebra: ×, norm, dot

import ClimaAtmos.Parameters as CAP
using ClimaCore: Operators, Fields, Limiters, Geometry, Spaces

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

# Fields used to store variables that only need to be used in a single function
# but cannot be computed on the fly. Unlike the precomputed quantities, these
# can be modified at any point, so they should never be assumed to be unchanged
# between function calls.
function temporary_quantities(atmos, center_space, face_space)
    FT = Spaces.undertype(center_space)
    n = n_mass_flux_subdomains(atmos.turbconv_model)
    return (;
        ᶜtemp_scalar = Fields.Field(FT, center_space), # ᶜh_tot, ᶜh_totʲ
        ᶜtemp_CT3 = Fields.Field(CT3{FT}, center_space), # ᶜω³
        ᶠtemp_CT3 = Fields.Field(CT3{FT}, face_space), # ᶠuₕ³
        ᶠtemp_CT12 = Fields.Field(CT12{FT}, face_space), # ᶠω¹²
        ᶠtemp_CT12ʲs = Fields.Field(NTuple{n, CT12{FT}}, face_space), # ᶠω¹²ʲs
    )
end

function default_cache(
    Y,
    parsed_args,
    params,
    atmos,
    spaces,
    numerics,
    simulation,
    comms_ctx,
)
    FT = eltype(params)
    (; energy_upwinding, tracer_upwinding, density_upwinding, edmfx_upwinding) =
        numerics
    (; apply_limiter) = numerics
    ᶜcoord = Fields.local_geometry_field(Y.c).coordinates
    ᶠcoord = Fields.local_geometry_field(Y.f).coordinates
    R_d = FT(CAP.R_d(params))
    MSLP = FT(CAP.MSLP(params))
    grav = FT(CAP.grav(params))
    cv_d = FT(CAP.cv_d(params))
    T_0 = FT(CAP.T_0(params))
    T_ref = FT(255)
    ᶜΦ = CAP.grav(params) .* ᶜcoord.z
    ᶜρ_ref = @. MSLP * exp(-grav * ᶜcoord.z / (R_d * T_ref)) / (R_d * T_ref)
    ᶜp_ref = @. ᶜρ_ref * R_d * T_ref
    ᶜh_ref = @. TD.total_specific_enthalpy(cv_d*(T_ref-T_0) + grav * ᶜcoord.z, R_d, T_ref)
    if !parsed_args["use_hyperdiff_reference_state"]
        ᶜh_ref .*= 0
    end
    if !parsed_args["use_reference_state"]
        ᶜρ_ref .*= 0
        ᶜp_ref .*= 0
    end
    z_sfc = Fields.level(ᶠcoord.z, half)
    if eltype(ᶜcoord) <: Geometry.LatLongZPoint
        Ω = CAP.Omega(params)
        ᶜf = @. 2 * Ω * sind(ᶜcoord.lat)
        lat_sfc = Fields.level(ᶜcoord.lat, 1)
    else
        f = CAP.f_plane_coriolis_frequency(params)
        ᶜf = map(_ -> f, ᶜcoord)
        lat_sfc = map(_ -> eltype(params)(0), Fields.level(ᶜcoord, 1))
    end
    ᶜf = @. CT3(Geometry.WVector(ᶜf))
    T_sfc = @. 29 * exp(-lat_sfc^2 / (2 * 26^2)) + 271

    sfc_conditions =
        similar(Fields.level(Y.f, half), SF.SurfaceFluxConditions{FT})

    quadrature_style = Spaces.horizontal_space(axes(Y.c)).quadrature_style
    do_dss = quadrature_style isa Spaces.Quadratures.GLL
    ghost_buffer =
        !do_dss ? (;) :
        (; c = Spaces.create_dss_buffer(Y.c), f = Spaces.create_dss_buffer(Y.f))

    limiter =
        apply_limiter ? Limiters.QuasiMonotoneLimiter(similar(Y.c, FT)) :
        nothing

    net_energy_flux_toa = [sum(similar(Y.f, Geometry.WVector{FT})) * 0]
    net_energy_flux_toa[] = Geometry.WVector(FT(0))
    net_energy_flux_sfc = [sum(similar(Y.f, Geometry.WVector{FT})) * 0]
    net_energy_flux_sfc[] = Geometry.WVector(FT(0))

    default_cache = (;
        simulation,
        spaces,
        atmos,
        comms_ctx,
        test_dycore_consistency = parsed_args["test_dycore_consistency"],
        moisture_model = atmos.moisture_model,
        model_config = atmos.model_config,
        Yₜ = similar(Y), # only needed when using increment formulation
        limiter,
        ᶜΦ,
        ᶠgradᵥ_ᶜΦ = ᶠgradᵥ.(ᶜΦ),
        ᶜρ_ref,
        ᶜp_ref,
        ᶜh_ref,
        ᶜT = similar(Y.c, FT),
        ᶜf,
        sfc_conditions,
        z_sfc,
        T_sfc,
        ts_sfc = similar(
            Spaces.level(Y.f, half),
            thermo_state_type(atmos.moisture_model, FT),
        ),
        ∂ᶜK∂ᶠw_data = similar(
            Y.c,
            Operators.StencilCoefs{-half, half, NTuple{2, FT}},
        ),
        params,
        energy_upwinding,
        tracer_upwinding,
        density_upwinding,
        edmfx_upwinding,
        do_dss,
        ghost_buffer,
        net_energy_flux_toa,
        net_energy_flux_sfc,
        precomputed_quantities(Y, atmos)...,
        temporary_quantities(atmos, spaces.center_space, spaces.face_space)...,
        hyperdiffusion_cache(Y, atmos, do_dss)...,
    )
    set_precomputed_quantities!(Y, default_cache, FT(0))
    return default_cache
end

function dss!(Y, p, t)
    if p.do_dss
        Spaces.weighted_dss_start2!(Y.c, p.ghost_buffer.c)
        Spaces.weighted_dss_start2!(Y.f, p.ghost_buffer.f)
        Spaces.weighted_dss_internal2!(Y.c, p.ghost_buffer.c)
        Spaces.weighted_dss_internal2!(Y.f, p.ghost_buffer.f)
        Spaces.weighted_dss_ghost2!(Y.c, p.ghost_buffer.c)
        Spaces.weighted_dss_ghost2!(Y.f, p.ghost_buffer.f)
    end
end

function limited_tendency!(Yₜ, Y, p, t)
    Yₜ .= zero(eltype(Yₜ))
    set_precomputed_quantities!(Y, p, t)
    horizontal_tracer_advection_tendency!(Yₜ, Y, p, t)
    NVTX.@range "tracer hyperdiffusion tendency" color = colorant"yellow" begin
        tracer_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    end
end

function limiters_func!(Y, p, t, ref_Y)
    (; limiter) = p
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    if !isnothing(limiter)
        for ρχ_name in filter(is_tracer_var, propertynames(Y.c))
            Limiters.compute_bounds!(limiter, ref_Y.c.:($ρχ_name), ref_Y.c.ρ)
            Limiters.apply_limiter!(Y.c.:($ρχ_name), Y.c.ρ, limiter)
        end
        for j in 1:n
            for ρaχ_name in
                filter(is_tracer_var, propertynames(Y.c.sgsʲs.:($j)))
                ᶜρaχ_ref = ref_Y.c.sgsʲs.:($j).:($ρaχ_name)
                ᶜρa_ref = ref_Y.c.sgsʲs.:($j).ρa
                ᶜρaχ = Y.c.sgsʲs.:($j).:($ρaχ_name)
                ᶜρa = Y.c.sgsʲs.:($j).ρa
                Limiters.compute_bounds!(limiter, ᶜρaχ_ref, ᶜρa_ref)
                Limiters.apply_limiter!(ᶜρaχ, ᶜρa, limiter)
            end
        end
    end
end
