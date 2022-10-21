using Statistics: mean
using SurfaceFluxes
using CloudMicrophysics
const SF = SurfaceFluxes
const CCG = ClimaCore.Geometry
import ClimaAtmos.TurbulenceConvection as TC
import ClimaCore.Operators as CCO
const CM = CloudMicrophysics
import ClimaAtmos.Parameters as CAP

include("../staggered_nonhydrostatic_model.jl")
include("./topography.jl")
include("../initial_conditions.jl")

# 0-Moment Microphysics

microphysics_cache(Y, ::Nothing) = NamedTuple()
microphysics_cache(Y, ::Microphysics0Moment) = (
    ᶜS_ρq_tot = similar(Y.c, FT),
    ᶜλ = similar(Y.c, FT),
    ᶜ3d_rain = similar(Y.c, FT),
    ᶜ3d_snow = similar(Y.c, FT),
    col_integrated_rain = similar(ClimaCore.Fields.level(Y.c.ρ, 1), FT),
    col_integrated_snow = similar(ClimaCore.Fields.level(Y.c.ρ, 1), FT),
)

function zero_moment_microphysics_tendency!(Yₜ, Y, p, t, colidx)
    (;
        ᶜts,
        ᶜΦ,
        ᶜT,
        ᶜ3d_rain,
        ᶜ3d_snow,
        ᶜS_ρq_tot,
        ᶜλ,
        col_integrated_rain,
        col_integrated_snow,
        params,
    ) = p # assume ᶜts has been updated
    thermo_params = CAP.thermodynamics_params(params)
    cm_params = CAP.microphysics_params(params)
    @. ᶜS_ρq_tot[colidx] =
        Y.c.ρ[colidx] * CM.Microphysics0M.remove_precipitation(
            cm_params,
            TD.PhasePartition(thermo_params, ᶜts[colidx]),
        )
    @. Yₜ.c.ρq_tot[colidx] += ᶜS_ρq_tot[colidx]
    @. Yₜ.c.ρ[colidx] += ᶜS_ρq_tot[colidx]

    # update precip in cache for coupler's use
    # 3d rain and snow
    @. ᶜT[colidx] = TD.air_temperature(thermo_params, ᶜts[colidx])
    @. ᶜ3d_rain[colidx] =
        ifelse(ᶜT[colidx] >= FT(273.15), ᶜS_ρq_tot[colidx], FT(0))
    @. ᶜ3d_snow[colidx] =
        ifelse(ᶜT[colidx] < FT(273.15), ᶜS_ρq_tot[colidx], FT(0))
    CCO.column_integral_definite!(col_integrated_rain[colidx], ᶜ3d_rain[colidx])
    CCO.column_integral_definite!(col_integrated_snow[colidx], ᶜ3d_snow[colidx])

    @. col_integrated_rain[colidx] =
        col_integrated_rain[colidx] / CAP.ρ_cloud_liq(params)
    @. col_integrated_snow[colidx] =
        col_integrated_snow[colidx] / CAP.ρ_cloud_liq(params)

    # liquid fraction
    @. ᶜλ[colidx] = TD.liquid_fraction(thermo_params, ᶜts[colidx])

    if :ρe_tot in propertynames(Y.c)
        @. Yₜ.c.ρe_tot[colidx] +=
            ᶜS_ρq_tot[colidx] * (
                ᶜλ[colidx] *
                TD.internal_energy_liquid(thermo_params, ᶜts[colidx]) +
                (1 - ᶜλ[colidx]) *
                TD.internal_energy_ice(thermo_params, ᶜts[colidx]) +
                ᶜΦ[colidx]
            )
    elseif :ρe_int in propertynames(Y.c)
        @. Yₜ.c.ρe_int[colidx] +=
            ᶜS_ρq_tot[colidx] * (
                ᶜλ[colidx] *
                TD.internal_energy_liquid(thermo_params, ᶜts[colidx]) +
                (1 - ᶜλ[colidx]) *
                TD.internal_energy_ice(thermo_params, ᶜts[colidx])
            )
    end
    return nothing
end

# Vertical diffusion boundary layer parameterization

# Apply on potential temperature and moisture
# 1) turn the liquid_theta into theta version
# 2) have a total energy version (primary goal)

function vertical_diffusion_boundary_layer_cache(
    Y,
    ::Type{FT};
    surface_scheme = nothing,
    C_E::FT = FT(0),
    diffuse_momentum = true,
    coupled = false,
) where {FT}
    z_bottom = Spaces.level(Fields.coordinate_field(Y.c).z, 1)

    dif_flux_uₕ =
        Geometry.Contravariant3Vector.(zeros(axes(z_bottom))) .⊗
        Geometry.Covariant12Vector.(
            zeros(axes(z_bottom)),
            zeros(axes(z_bottom)),
        )
    dif_flux_energy = similar(z_bottom, Geometry.WVector{FT})
    dif_flux_ρq_tot = if :ρq_tot in propertynames(Y.c)
        similar(z_bottom, Geometry.WVector{FT})
    else
        Ref(Geometry.WVector(FT(0)))
    end

    cond_type = NamedTuple{(:shf, :lhf, :E, :ρτxz, :ρτyz), NTuple{5, FT}}
    surface_normal = Geometry.WVector.(ones(axes(Fields.level(Y.c, 1))))

    surface_scheme_params = if surface_scheme isa BulkSurfaceScheme
        (;
            Cd = ones(axes(Fields.level(Y.f, half))) .* FT(0.0044),
            Ch = ones(axes(Fields.level(Y.f, half))) .* FT(0.0044),
        )
    elseif surface_scheme isa MoninObukhovSurface
        (;
            z0m = ones(axes(Fields.level(Y.f, half))) .* FT(1e-5),
            z0b = ones(axes(Fields.level(Y.f, half))) .* FT(1e-5),
        )
    elseif isnothing(surface_scheme)
        NamedTuple()
    end

    return (;
        surface_scheme,
        C_E,
        ᶠp = similar(Y.f, FT),
        ᶠK_E = similar(Y.f, FT),
        surface_conditions = similar(z_bottom, cond_type),
        uₕ_int_local = similar(Spaces.level(Y.c.uₕ, 1), Geometry.UVVector{FT}),
        dif_flux_uₕ,
        dif_flux_uₕ_bc = similar(dif_flux_uₕ),
        dif_flux_energy,
        dif_flux_ρq_tot,
        dif_flux_energy_bc = similar(dif_flux_energy),
        dif_flux_ρq_tot_bc = similar(dif_flux_ρq_tot),
        diffuse_momentum,
        surface_scheme_params...,
        coupled,
        surface_normal,
        z_bottom,
    )
end

function eddy_diffusivity_coefficient(C_E, norm_v_a, z_a, p)
    p_pbl = FT(85000)
    p_strato = FT(10000)
    K_E = C_E * norm_v_a * z_a
    return p > p_pbl ? K_E : K_E * exp(-((p_pbl - p) / p_strato)^2)
end

function get_surface_density_and_humidity(T_sfc, ts_int, params)
    thermo_params = CAP.thermodynamics_params(params)
    T_int = TD.air_temperature(thermo_params, ts_int)
    Rm_int = TD.gas_constant_air(thermo_params, ts_int)
    ρ_sfc =
        TD.air_density(thermo_params, ts_int) *
        (T_sfc / T_int)^(TD.cv_m(thermo_params, ts_int) / Rm_int)
    q_sfc =
        TD.q_vap_saturation_generic(thermo_params, T_sfc, ρ_sfc, TD.Liquid())
    return ρ_sfc, q_sfc
end

surface_args(::BulkSurfaceScheme, p, colidx) = (p.Cd[colidx], p.Ch[colidx])

function saturated_surface_conditions(
    surface_scheme::BulkSurfaceScheme,
    Cd,
    Ch,
    T_sfc,
    ρ_sfc,
    q_sfc,
    ts_int,
    uₕ_int,
    z_int,
    z_sfc,
    params,
    coupled,
)

    # parametercs
    thermo_params = CAP.thermodynamics_params(params)
    sf_params = CAP.surface_fluxes_params(params)

    # get the near-surface thermal state
    if !coupled
        ρ_sfc, q_sfc = get_surface_density_and_humidity(T_sfc, ts_int, params)
    end
    ts_sfc = TD.PhaseEquil_ρTq(thermo_params, ρ_sfc, T_sfc, q_sfc)

    # wrap state values
    sc = SF.Coefficients{FT}(;
        state_in = SF.InteriorValues(z_int, (uₕ_int.u, uₕ_int.v), ts_int),
        state_sfc = SF.SurfaceValues(z_sfc, (FT(0), FT(0)), ts_sfc),
        z0m = FT(0),
        z0b = FT(0),
        Cd = Cd, #FT(0.001),
        Ch = Ch, #FT(0.0001),
    )

    # calculate all fluxes
    tsf = SF.surface_conditions(sf_params, sc, SF.FVScheme())

    E = SF.evaporation(sf_params, sc, tsf.Ch)

    return (;
        shf = tsf.shf,
        lhf = tsf.lhf,
        E = E,
        ρτxz = tsf.ρτxz,
        ρτyz = tsf.ρτyz,
    )
end

saturated_surface_conditions(::Nothing, args...) = nothing

surface_args(::MoninObukhovSurface, p, colidx) = (p.z0m[colidx], p.z0b[colidx])

function saturated_surface_conditions(
    surface_scheme::MoninObukhovSurface,
    z0m,
    z0b,
    T_sfc,
    ρ_sfc,
    q_sfc,
    ts_int,
    uₕ_int,
    z_int,
    z_sfc,
    params,
    coupled,
)

    # parameters
    thermo_params = CAP.thermodynamics_params(params)
    sf_params = CAP.surface_fluxes_params(params)

    # get the near-surface thermal state
    if !coupled
        ρ_sfc, q_sfc = get_surface_density_and_humidity(T_sfc, ts_int, params)
    end
    ts_sfc = TD.PhaseEquil_ρTq(thermo_params, ρ_sfc, T_sfc, q_sfc)

    # wrap state values
    sc = SF.ValuesOnly{FT}(;
        state_in = SF.InteriorValues(z_int, (uₕ_int.u, uₕ_int.v), ts_int),
        state_sfc = SF.SurfaceValues(z_sfc, (FT(0), FT(0)), ts_sfc),
        z0m = z0m,
        z0b = z0b,
    )

    # calculate all fluxes
    tsf = SF.surface_conditions(sf_params, sc)

    E = SF.evaporation(sf_params, sc, tsf.Ch)

    return (;
        shf = tsf.shf,
        lhf = tsf.lhf,
        E = E,
        ρτxz = tsf.ρτxz,
        ρτyz = tsf.ρτyz,
    )
end

function vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t)
    Fields.bycolumn(axes(Y.c.uₕ)) do colidx
        (; coupled) = p
        !coupled && get_surface_fluxes!(Y, p, colidx)
        vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t, colidx)
    end
end

function get_surface_fluxes!(Y, p, colidx)
    (; z_sfc, ᶜts, T_sfc, ρ_sfc, q_sfc) = p
    (;
        surface_conditions,
        dif_flux_uₕ,
        dif_flux_uₕ_bc,
        dif_flux_energy,
        dif_flux_energy_bc,
        dif_flux_ρq_tot,
        dif_flux_ρq_tot_bc,
        diffuse_momentum,
        z_bottom,
        params,
        coupled,
    ) = p
    (; uₕ_int_local, surface_normal) = p

    uₕ_int = Spaces.level(Y.c.uₕ[colidx], 1)
    @. uₕ_int_local[colidx] = Geometry.UVVector(uₕ_int)

    surf_args = surface_args(p.surface_scheme, p, colidx)

    surface_conditions[colidx] .=
        saturated_surface_conditions.(
            p.surface_scheme,
            surf_args...,
            T_sfc[colidx],
            ρ_sfc[colidx],
            q_sfc[colidx],
            Spaces.level(ᶜts[colidx], 1),
            uₕ_int_local[colidx],
            z_bottom[colidx],
            z_sfc[colidx],
            params,
            coupled,
        )
    if diffuse_momentum
        ρτxz = surface_conditions[colidx].ρτxz
        ρτyz = surface_conditions[colidx].ρτyz
        ρ_1 = Fields.level(Y.c.ρ[colidx], 1)
        @. dif_flux_uₕ[colidx] =
            Geometry.Contravariant3Vector(surface_normal[colidx]) ⊗
            Geometry.Covariant12Vector(
                Geometry.UVVector(ρτxz / ρ_1, ρτyz / ρ_1),
            )
        @. dif_flux_uₕ_bc[colidx] = -dif_flux_uₕ[colidx]
    end
    if :ρe_tot in propertynames(Y.c)
        if isnothing(p.surface_scheme)
            @. dif_flux_energy[colidx] *= 0
        else
            @. dif_flux_energy[colidx] = Geometry.WVector(
                surface_conditions[colidx].shf + surface_conditions[colidx].lhf,
            )
        end
        @. dif_flux_energy_bc[colidx] = -dif_flux_energy[colidx]
    end

    if :ρq_tot in propertynames(Y.c)
        if isnothing(p.surface_scheme)
            @. dif_flux_ρq_tot[colidx] *= 0
        else
            @. dif_flux_ρq_tot[colidx] =
                Geometry.WVector(surface_conditions[colidx].E)
        end
        @. dif_flux_ρq_tot_bc[colidx] = -dif_flux_ρq_tot[colidx]
    end
end

function vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t, colidx)
    ᶜρ = Y.c.ρ
    FT = Spaces.undertype(axes(ᶜρ))
    (; ᶜp, ᶠK_E, C_E) = p # assume ᶜts and ᶜp have been updated
    (;
        dif_flux_uₕ_bc,
        dif_flux_energy_bc,
        dif_flux_ρq_tot_bc,
        diffuse_momentum,
        ᶠp,
        z_bottom,
        uₕ_int_local,
    ) = p

    ᶠgradᵥ = Operators.GradientC2F() # apply BCs to ᶜdivᵥ, which wraps ᶠgradᵥ

    uₕ_int = Spaces.level(Y.c.uₕ[colidx], 1)
    @. uₕ_int_local[colidx] = Geometry.UVVector(uₕ_int)
    @. ᶠp[colidx] = ᶠinterp(ᶜp[colidx])
    @. ᶠK_E[colidx] = eddy_diffusivity_coefficient(
        C_E,
        norm(uₕ_int),
        z_bottom[colidx],
        ᶠp[colidx],
    )

    if diffuse_momentum
        ᶜdivᵥ = Operators.DivergenceF2C(
            top = Operators.SetValue(
                Geometry.Contravariant3Vector(FT(0)) ⊗
                Geometry.Covariant12Vector(FT(0), FT(0)),
            ),
            bottom = Operators.SetValue(dif_flux_uₕ_bc[colidx]),
        )
        @. Yₜ.c.uₕ[colidx] += ᶜdivᵥ(ᶠK_E[colidx] * ᶠgradᵥ(Y.c.uₕ[colidx]))
    end

    if :ρe_tot in propertynames(Y.c)
        ᶜdivᵥ = Operators.DivergenceF2C(
            top = Operators.SetValue(Geometry.WVector(FT(0))),
            bottom = Operators.SetValue(dif_flux_energy_bc[colidx]),
        )
        @. Yₜ.c.ρe_tot[colidx] += ᶜdivᵥ(
            ᶠK_E[colidx] *
            ᶠinterp(ᶜρ[colidx]) *
            ᶠgradᵥ((Y.c.ρe_tot[colidx] + ᶜp[colidx]) / ᶜρ[colidx]),
        )
    end

    if :ρq_tot in propertynames(Y.c)
        ᶜdivᵥ = Operators.DivergenceF2C(
            top = Operators.SetValue(Geometry.WVector(FT(0))),
            bottom = Operators.SetValue(dif_flux_ρq_tot_bc[colidx]),
        )
        @. Yₜ.c.ρq_tot[colidx] += ᶜdivᵥ(
            ᶠK_E[colidx] *
            ᶠinterp(ᶜρ[colidx]) *
            ᶠgradᵥ(Y.c.ρq_tot[colidx] / ᶜρ[colidx]),
        )
        @. Yₜ.c.ρ[colidx] += ᶜdivᵥ(
            ᶠK_E[colidx] *
            ᶠinterp(ᶜρ[colidx]) *
            ᶠgradᵥ(Y.c.ρq_tot[colidx] / ᶜρ[colidx]),
        )
    end
end
