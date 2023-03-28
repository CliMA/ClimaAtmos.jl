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

# TODO: Rename ᶜK to ᶜκ, and rename ᶠw to ᶠuᵥ.
function precomputed_quantities(Y, atmos)
    FT = eltype(Y)
    TST = thermo_state_type(atmos.moisture_model, FT)
    return (;
        ᶜu = similar(Y.c, Geometry.Covariant123Vector{FT}),
        ᶠu³ = similar(Y.f, Geometry.Contravariant3Vector{FT}),
        ᶜK = similar(Y.c, FT),
        ᶜts = similar(Y.c, TST),
        ᶜp = similar(Y.c, FT),
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

    curlₕ = Operators.Curl()
    ᶜinterp = Operators.InterpolateF2C()
    ᶠinterp = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    ᶠwinterp = Operators.WeightedInterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    ᶜdivᵥ = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
    )
    ᶠgradᵥ = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
        top = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
    )
    ᶜgradᵥ = Operators.GradientF2C()
    ᶠcurlᵥ = Operators.CurlC2F(
        bottom = Operators.SetCurl(
            Geometry.Contravariant12Vector(FT(0), FT(0)),
        ),
        top = Operators.SetCurl(Geometry.Contravariant12Vector(FT(0), FT(0))),
    )
    ᶠupwind1 = Operators.UpwindBiasedProductC2F()
    ᶠupwind3 = Operators.Upwind3rdOrderBiasedProductC2F(
        bottom = Operators.ThirdOrderOneSided(),
        top = Operators.ThirdOrderOneSided(),
    )
    ᶠfct_boris_book = Operators.FCTBorisBook(
        bottom = Operators.FirstOrderOneSided(),
        top = Operators.FirstOrderOneSided(),
    )
    ᶠfct_zalesak = Operators.FCTZalesak(
        bottom = Operators.FirstOrderOneSided(),
        top = Operators.FirstOrderOneSided(),
    )

    (; energy_upwinding, tracer_upwinding, apply_limiter) = numerics
    ᶜcoord = Fields.local_geometry_field(Y.c).coordinates
    ᶠcoord = Fields.local_geometry_field(Y.f).coordinates
    R_d = FT(CAP.R_d(params))
    MSLP = FT(CAP.MSLP(params))
    grav = FT(CAP.grav(params))
    T_ref = FT(255)
    ᶜΦ = CAP.grav(params) .* ᶜcoord.z
    ᶜρ_ref = @. MSLP * exp(-grav * ᶜcoord.z / (R_d * T_ref)) / (R_d * T_ref)
    ᶜp_ref = @. ᶜρ_ref * R_d * T_ref
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
    ᶜf = @. Geometry.Contravariant3Vector(Geometry.WVector(ᶜf))
    T_sfc = @. 29 * exp(-lat_sfc^2 / (2 * 26^2)) + 271

    sfc_conditions =
        similar(Fields.level(Y.f, half), SF.SurfaceFluxConditions{FT})

    quadrature_style = Spaces.horizontal_space(axes(Y.c)).quadrature_style
    skip_dss = !(quadrature_style isa Spaces.Quadratures.GLL)
    if skip_dss
        ghost_buffer = (
            c = nothing,
            f = nothing,
            χ = nothing, # for hyperdiffusion
            χw = nothing, # for hyperdiffusion
            χuₕ = nothing, # for hyperdiffusion
            skip_dss = skip_dss, # skip DSS on non-GLL quadrature meshes
        )
        (:ρq_tot in propertynames(Y.c)) &&
            (ghost_buffer = (ghost_buffer..., ᶜχρq_tot = nothing))
    else
        ghost_buffer = (
            c = Spaces.create_dss_buffer(Y.c),
            f = Spaces.create_dss_buffer(Y.f),
            χ = Spaces.create_dss_buffer(Y.c.ρ), # for hyperdiffusion
            χw = Spaces.create_dss_buffer(Y.f.w.components.data.:1), # for hyperdiffusion
            χuₕ = Spaces.create_dss_buffer(Y.c.uₕ), # for hyperdiffusion
            skip_dss = skip_dss, # skip DSS on non-GLL quadrature meshes
        )
        (:ρq_tot in propertynames(Y.c)) && (
            ghost_buffer = (
                ghost_buffer...,
                ᶜχρq_tot = Spaces.create_dss_buffer(Y.c.ρ),
            )
        )
    end
    if apply_limiter
        tracers = filter(is_tracer_var, propertynames(Y.c))
        make_limiter =
            ᶜρc_name ->
                Limiters.QuasiMonotoneLimiter(getproperty(Y.c, ᶜρc_name))
        limiters = NamedTuple{tracers}(map(make_limiter, tracers))
    else
        limiters = nothing
    end
    pnc = propertynames(Y.c)
    ᶜρh_kwargs = :ρe_tot in pnc ? (; ᶜρh = similar(Y.c, FT)) : ()

    net_energy_flux_toa = [sum(similar(Y.f, Geometry.WVector{FT})) * 0]
    net_energy_flux_toa[] = Geometry.WVector(FT(0))
    net_energy_flux_sfc = [sum(similar(Y.f, Geometry.WVector{FT})) * 0]
    net_energy_flux_sfc[] = Geometry.WVector(FT(0))

    default_cache = (;
        simulation,
        operators = (;
            ᶜdivᵥ,
            ᶜgradᵥ,
            ᶜdivᵥ_stencil = Operators.Operator2Stencil(ᶜdivᵥ),
            ᶠgradᵥ_stencil = Operators.Operator2Stencil(ᶠgradᵥ),
            ᶜinterp_stencil = Operators.Operator2Stencil(ᶜinterp),
            ᶠinterp_stencil = Operators.Operator2Stencil(ᶠinterp),
            ᶠwinterp_stencil = Operators.Operator2Stencil(ᶠwinterp),
            ᶠinterp,
            ᶠwinterp,
            ᶠcurlᵥ,
            ᶜinterp,
            ᶠgradᵥ,
            curlₕ,
            ᶠupwind1,
            ᶠupwind3,
            ᶠfct_boris_book,
            ᶠfct_zalesak,
        ),
        spaces,
        atmos,
        comms_ctx,
        test_dycore_consistency = parsed_args["test_dycore_consistency"],
        moisture_model = atmos.moisture_model,
        model_config = atmos.model_config,
        Yₜ = similar(Y), # only needed when using increment formulation
        limiters,
        ᶜρh_kwargs...,
        ᶜΦ,
        ᶠgradᵥ_ᶜΦ = ᶠgradᵥ.(ᶜΦ),
        ᶜρ_ref,
        ᶜp_ref,
        ᶜT = similar(Y.c, FT),
        ᶜω³ = similar(Y.c, Geometry.Contravariant3Vector{FT}),
        ᶠω¹² = similar(Y.f, Geometry.Contravariant12Vector{FT}),
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
        ghost_buffer = ghost_buffer,
        net_energy_flux_toa,
        net_energy_flux_sfc,
        precomputed_quantities(Y, atmos)...,
    )
    set_precomputed_quantities!(Y, default_cache, FT(0))
    return default_cache
end

function dss!(Y, p, t)
    if !p.ghost_buffer.skip_dss
        Spaces.weighted_dss_start2!(Y.c, p.ghost_buffer.c)
        Spaces.weighted_dss_start2!(Y.f, p.ghost_buffer.f)
        Spaces.weighted_dss_internal2!(Y.c, p.ghost_buffer.c)
        Spaces.weighted_dss_internal2!(Y.f, p.ghost_buffer.f)
        Spaces.weighted_dss_ghost2!(Y.c, p.ghost_buffer.c)
        Spaces.weighted_dss_ghost2!(Y.f, p.ghost_buffer.f)
    end
end

function horizontal_limiter_tendency!(Yₜ, Y, p, t)
    Yₜ .= zero(eltype(Yₜ))
    set_precomputed_quantities!(Y, p, t)

    (; ᶜu) = p
    divₕ = Operators.Divergence()

    # Tracer conservation, horizontal advection
    for ᶜρc_name in filter(is_tracer_var, propertynames(Y.c))
        ᶜρc = getproperty(Y.c, ᶜρc_name)
        ᶜρcₜ = getproperty(Yₜ.c, ᶜρc_name)
        @. ᶜρcₜ -= divₕ(ᶜρc * ᶜu)
    end

    # Call hyperdiffusion
    hyperdiffusion_tracers_tendency!(Yₜ, Y, p, t)

    return nothing
end

function limiters_func!(Y, p, t, ref_Y)
    (; limiters) = p
    if !isnothing(limiters)
        for ᶜρc_name in filter(is_tracer_var, propertynames(Y.c))
            ρc_limiter = getproperty(limiters, ᶜρc_name)
            ᶜρc_ref = getproperty(ref_Y.c, ᶜρc_name)
            ᶜρc = getproperty(Y.c, ᶜρc_name)
            Limiters.compute_bounds!(ρc_limiter, ᶜρc_ref, ref_Y.c.ρ)
            Limiters.apply_limiter!(ᶜρc, Y.c.ρ, ρc_limiter)
        end
    end
end
