using LinearAlgebra: ×, norm, dot

import ClimaAtmos.Parameters as CAP
using ClimaCore: Operators, Fields, Limiters

using ClimaCore.Geometry: ⊗

import Thermodynamics as TD

using ClimaCore.Utilities: half

import ClimaCore.Fields: ColumnField

# Note: FT must be defined before `include("staggered_nonhydrostatic_model.jl")`

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

# To add additional terms to the explicit part of the tendency, define new
# methods for `additional_cache` and `additional_tendency!`.

const divₕ = Operators.Divergence()
const wdivₕ = Operators.WeakDivergence()
const gradₕ = Operators.Gradient()
const wgradₕ = Operators.WeakGradient()
const curlₕ = Operators.Curl()
const wcurlₕ = Operators.WeakCurl()

const ᶜinterp = Operators.InterpolateF2C()
const ᶠinterp = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const ᶠwinterp = Operators.WeightedInterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const ᶜdivᵥ = Operators.DivergenceF2C(
    top = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
    bottom = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
)
const ᶠgradᵥ = Operators.GradientC2F(
    bottom = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
    top = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
)
const ᶜgradᵥ = Operators.GradientF2C()
const ᶠcurlᵥ = Operators.CurlC2F(
    bottom = Operators.SetCurl(Geometry.Contravariant12Vector(FT(0), FT(0))),
    top = Operators.SetCurl(Geometry.Contravariant12Vector(FT(0), FT(0))),
)
const ᶠupwind1 = Operators.UpwindBiasedProductC2F()
const ᶠupwind3 = Operators.Upwind3rdOrderBiasedProductC2F(
    bottom = Operators.ThirdOrderOneSided(),
    top = Operators.ThirdOrderOneSided(),
)
const ᶠfct_boris_book = Operators.FCTBorisBook(
    bottom = Operators.FirstOrderOneSided(),
    top = Operators.FirstOrderOneSided(),
)
const ᶠfct_zalesak = Operators.FCTZalesak(
    bottom = Operators.FirstOrderOneSided(),
    top = Operators.FirstOrderOneSided(),
)

const C123 = Geometry.Covariant123Vector

get_cache(Y, parsed_args, params, spaces, atmos, numerics, simulation) = merge(
    default_cache(Y, parsed_args, params, atmos, spaces, numerics, simulation),
    additional_cache(Y, parsed_args, params, atmos, simulation.dt),
)

function default_cache(
    Y,
    parsed_args,
    params,
    atmos,
    spaces,
    numerics,
    simulation,
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
        lat_sfc = map(_ -> FT(0), Fields.level(ᶜcoord, 1))
    end
    ᶜf = @. Geometry.Contravariant3Vector(Geometry.WVector(ᶜf))
    T_sfc = @. 29 * exp(-lat_sfc^2 / (2 * 26^2)) + 271

    sfc_conditions =
        similar(Fields.level(Y.f, half), SF.SurfaceFluxConditions{FT})

    ts_type = CA.thermo_state_type(atmos.moisture_model, FT)
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
        tracers = filter(CA.is_tracer_var, propertynames(Y.c))
        make_limiter =
            ᶜρc_name ->
                Limiters.QuasiMonotoneLimiter(getproperty(Y.c, ᶜρc_name))
        limiters = NamedTuple{tracers}(map(make_limiter, tracers))
    else
        limiters = nothing
    end
    pnc = propertynames(Y.c)
    ᶜρh_kwargs =
        :ρe_tot in pnc || :ρe_int in pnc ? (; ᶜρh = similar(Y.c, FT)) : ()

    net_energy_flux_toa = [sum(similar(Y.f, Geometry.WVector{FT})) * 0]
    net_energy_flux_toa[] = Geometry.WVector(FT(0))
    net_energy_flux_sfc = [sum(similar(Y.f, Geometry.WVector{FT})) * 0]
    net_energy_flux_sfc[] = Geometry.WVector(FT(0))

    return (;
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
            ᶠupwind1,
            ᶠupwind3,
            ᶠfct_boris_book,
            ᶠfct_zalesak,
        ),
        spaces,
        atmos,
        test_dycore_consistency = parsed_args["test_dycore_consistency"],
        moisture_model = atmos.moisture_model,
        model_config = atmos.model_config,
        Yₜ = similar(Y), # only needed when using increment formulation
        limiters,
        ᶜρh_kwargs...,
        ᶜuvw = similar(Y.c, Geometry.Covariant123Vector{FT}),
        ᶜK = similar(Y.c, FT),
        ᶜΦ,
        ᶠgradᵥ_ᶜΦ = ᶠgradᵥ.(ᶜΦ),
        ᶜρ_ref,
        ᶜp_ref,
        ᶜts = similar(Y.c, ts_type),
        ᶜp = similar(Y.c, FT),
        ᶜT = similar(Y.c, FT),
        ᶜω³ = similar(Y.c, Geometry.Contravariant3Vector{FT}),
        ᶠω¹² = similar(Y.f, Geometry.Contravariant12Vector{FT}),
        ᶠu¹² = similar(Y.f, Geometry.Contravariant12Vector{FT}),
        ᶠu³ = similar(Y.f, Geometry.Contravariant3Vector{FT}),
        ᶜf,
        sfc_conditions,
        z_sfc,
        T_sfc,
        ts_sfc = similar(Spaces.level(Y.f, half), ts_type),
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
    )
end

function implicit_tendency!(Yₜ, Y, p, t)
    p.test_dycore_consistency && CA.fill_with_nans!(p)
    @nvtx "implicit tendency" color = colorant"yellow" begin
        Fields.bycolumn(axes(Y.c)) do colidx
            CA.implicit_vertical_advection_tendency!(Yₜ, Y, p, t, colidx)

            if p.turbconv_model isa CA.TurbulenceConvection.EDMFModel
                parent(Yₜ.c.turbconv[colidx]) .= zero(eltype(Yₜ))
                parent(Yₜ.f.turbconv[colidx]) .= zero(eltype(Yₜ))
                TCU.implicit_sgs_flux_tendency!(
                    Yₜ,
                    Y,
                    p,
                    t,
                    colidx,
                    p.turbconv_model,
                )
            end
        end
    end
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

function remaining_tendency!(Yₜ, Y, p, t)
    p.test_dycore_consistency && CA.fill_with_nans!(p)
    (; compressibility_model) = p
    @nvtx "remaining tendency" color = colorant"yellow" begin
        Yₜ .= zero(eltype(Yₜ))
        @nvtx "precomputed quantities" color = colorant"orange" begin
            CA.precomputed_quantities!(Y, p, t)
        end
        if compressibility_model isa CA.CompressibleFluid
            @nvtx "horizontal" color = colorant"orange" begin
                CA.horizontal_advection_tendency!(Yₜ, Y, p, t)
            end
            @nvtx "vertical" color = colorant"orange" begin
                CA.explicit_vertical_advection_tendency!(Yₜ, Y, p, t)
            end
        end
        @nvtx "additional_tendency!" color = colorant"orange" begin
            additional_tendency!(Yₜ, Y, p, t)
        end
        @nvtx "dss_remaining_tendency" color = colorant"blue" begin
            dss!(Yₜ, p, t)
        end
    end
    return Yₜ
end
