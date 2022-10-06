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

##
## Additional tendencies
##

# Rayleigh sponge

function rayleigh_sponge_cache(
    Y,
    dt;
    zd_rayleigh = FT(15e3),
    α_rayleigh_uₕ = FT(1e-4),
    α_rayleigh_w = FT(1),
)
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠz = Fields.coordinate_field(Y.f).z
    ᶜαₘ_uₕ = @. ifelse(ᶜz > zd_rayleigh, α_rayleigh_uₕ, FT(0))
    ᶠαₘ_w = @. ifelse(ᶠz > zd_rayleigh, α_rayleigh_w, FT(0))
    zmax = maximum(ᶠz)
    ᶜβ_rayleigh_uₕ =
        @. ᶜαₘ_uₕ * sin(FT(π) / 2 * (ᶜz - zd_rayleigh) / (zmax - zd_rayleigh))^2
    ᶠβ_rayleigh_w =
        @. ᶠαₘ_w * sin(FT(π) / 2 * (ᶠz - zd_rayleigh) / (zmax - zd_rayleigh))^2
    return (; ᶜβ_rayleigh_uₕ, ᶠβ_rayleigh_w)
end

function rayleigh_sponge_tendency!(Yₜ, Y, p, t)
    (; ᶜβ_rayleigh_uₕ) = p
    @. Yₜ.c.uₕ -= ᶜβ_rayleigh_uₕ * Y.c.uₕ
end

# Viscous sponge

function viscous_sponge_cache(Y; zd_viscous = FT(15e3), κ₂ = FT(1e5))
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠz = Fields.coordinate_field(Y.f).z
    ᶜαₘ = @. ifelse(ᶜz > zd_viscous, κ₂, FT(0))
    ᶠαₘ = @. ifelse(ᶠz > zd_viscous, κ₂, FT(0))
    zmax = maximum(ᶠz)
    ᶜβ_viscous =
        @. ᶜαₘ * sin(FT(π) / 2 * (ᶜz - zd_viscous) / (zmax - zd_viscous))^2
    ᶠβ_viscous =
        @. ᶠαₘ * sin(FT(π) / 2 * (ᶠz - zd_viscous) / (zmax - zd_viscous))^2
    return (; ᶜβ_viscous, ᶠβ_viscous)
end

function viscous_sponge_tendency!(Yₜ, Y, p, t)
    (; ᶜβ_viscous, ᶠβ_viscous, ᶜp) = p
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    if :ρθ in propertynames(Y.c)
        @. Yₜ.c.ρθ += ᶜβ_viscous * wdivₕ(ᶜρ * gradₕ(Y.c.ρθ / ᶜρ))
    elseif :ρe_tot in propertynames(Y.c)
        @. Yₜ.c.ρe_tot += ᶜβ_viscous * wdivₕ(ᶜρ * gradₕ((Y.c.ρe_tot + ᶜp) / ᶜρ))
    elseif :ρe_int in propertynames(Y.c)
        @. Yₜ.c.ρe_int += ᶜβ_viscous * wdivₕ(ᶜρ * gradₕ((Y.c.ρe_int + ᶜp) / ᶜρ))
    end
    @. Yₜ.c.uₕ +=
        ᶜβ_viscous * (
            wgradₕ(divₕ(ᶜuₕ)) - Geometry.project(
                Geometry.Covariant12Axis(),
                wcurlₕ(Geometry.project(Geometry.Covariant3Axis(), curlₕ(ᶜuₕ))),
            )
        )
    @. Yₜ.f.w.components.data.:1 +=
        ᶠβ_viscous * wdivₕ(gradₕ(Y.f.w.components.data.:1))
end

forcing_cache(Y, ::Nothing) = NamedTuple()

# Held-Suarez forcing

forcing_cache(Y, ::HeldSuarezForcing) = (;
    ᶜσ = similar(Y.c, FT),
    ᶜheight_factor = similar(Y.c, FT),
    ᶜΔρT = similar(Y.c, FT),
    ᶜφ = deg2rad.(Fields.coordinate_field(Y.c).lat),
)

function held_suarez_tendency!(Yₜ, Y, p, t)
    Fields.bycolumn(axes(Y.c)) do colidx
        (; T_sfc, z_sfc, ᶜp, ᶜσ, ᶜheight_factor, ᶜΔρT, ᶜφ, params) = p # assume ᶜp has been updated

        R_d = FT(CAP.R_d(params))
        κ_d = FT(CAP.kappa_d(params))
        cv_d = FT(CAP.cv_d(params))
        day = FT(CAP.day(params))
        MSLP = FT(CAP.MSLP(params))
        grav = FT(CAP.grav(params))

        z_bottom = Spaces.level(Fields.coordinate_field(Y.c).z[colidx], 1)
        z_surface =
            Fields.Field(Fields.field_values(z_sfc[colidx]), axes(z_bottom))

        σ_b = FT(7 / 10)
        k_a = 1 / (40 * day)
        k_s = 1 / (4 * day)
        k_f = 1 / day
        if :ρq_tot in propertynames(Y.c)
            ΔT_y = FT(65)
            T_equator = FT(294)
        else
            ΔT_y = FT(60)
            T_equator = FT(315)
        end
        Δθ_z = FT(10)
        T_min = FT(200)

        @. ᶜσ[colidx] .=
            ᶜp[colidx] ./ (MSLP * exp(-grav * z_surface / R_d / T_sfc[colidx]))

        @. ᶜheight_factor[colidx] = max(0, (ᶜσ[colidx] - σ_b) / (1 - σ_b))
        @. ᶜΔρT[colidx] =
            (
                k_a +
                (k_s - k_a) * ᶜheight_factor[colidx] * (cos(ᶜφ[colidx])^2)^2
            ) *
            Y.c.ρ[colidx] *
            ( # ᶜT - ᶜT_equil
                ᶜp[colidx] / (Y.c.ρ[colidx] * R_d) - max(
                    T_min,
                    (
                        T_equator - ΔT_y * sin(ᶜφ[colidx])^2 -
                        Δθ_z * log(ᶜp[colidx] / MSLP) * cos(ᶜφ[colidx])^2
                    ) * fast_pow(ᶜσ[colidx], κ_d),
                )
            )

        @. Yₜ.c.uₕ[colidx] -= (k_f * ᶜheight_factor[colidx]) * Y.c.uₕ[colidx]
        if :ρθ in propertynames(Y.c)
            @. Yₜ.c.ρθ[colidx] -=
                ᶜΔρT[colidx] * fast_pow((MSLP / ᶜp[colidx]), κ_d)
        elseif :ρe_tot in propertynames(Y.c)
            @. Yₜ.c.ρe_tot[colidx] -= ᶜΔρT[colidx] * cv_d
        elseif :ρe_int in propertynames(Y.c)
            @. Yₜ.c.ρe_int[colidx] -= ᶜΔρT[colidx] * cv_d
        end
        nothing
    end
    return nothing
end

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

function zero_moment_microphysics_tendency!(Yₜ, Y, p, t)
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
    @. ᶜS_ρq_tot =
        Y.c.ρ * CM.Microphysics0M.remove_precipitation(
            cm_params,
            TD.PhasePartition(thermo_params, ᶜts),
        )
    @. Yₜ.c.ρq_tot += ᶜS_ρq_tot
    @. Yₜ.c.ρ += ᶜS_ρq_tot

    # update precip in cache for coupler's use
    # 3d rain and snow
    @. ᶜT = TD.air_temperature(thermo_params, ᶜts)
    @. ᶜ3d_rain = ifelse(ᶜT >= FT(273.15), ᶜS_ρq_tot, FT(0))
    @. ᶜ3d_snow = ifelse(ᶜT < FT(273.15), ᶜS_ρq_tot, FT(0))
    CCO.column_integral_definite!(col_integrated_rain, ᶜ3d_rain)
    CCO.column_integral_definite!(col_integrated_snow, ᶜ3d_snow)

    @. col_integrated_rain = col_integrated_rain / CAP.ρ_cloud_liq(params)
    @. col_integrated_snow = col_integrated_snow / CAP.ρ_cloud_liq(params)

    # liquid fraction
    @. ᶜλ = TD.liquid_fraction(thermo_params, ᶜts)

    if :ρe_tot in propertynames(Y.c)
        @. Yₜ.c.ρe_tot +=
            ᶜS_ρq_tot * (
                ᶜλ * TD.internal_energy_liquid(thermo_params, ᶜts) +
                (1 - ᶜλ) * TD.internal_energy_ice(thermo_params, ᶜts) +
                ᶜΦ
            )
    elseif :ρe_int in propertynames(Y.c)
        @. Yₜ.c.ρe_int +=
            ᶜS_ρq_tot * (
                ᶜλ * TD.internal_energy_liquid(thermo_params, ᶜts) +
                (1 - ᶜλ) * TD.internal_energy_ice(thermo_params, ᶜts)
            )
    end


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
    dif_flux_ρq_tot = if :ρq_tot in propertynames(Y.c)
        similar(z_bottom, Geometry.WVector{FT})
    else
        Ref(Geometry.WVector(FT(0)))
    end

    cond_type = NamedTuple{(:shf, :lhf, :E, :ρτxz, :ρτyz), NTuple{5, FT}}

    surface_normal = Geometry.WVector.(ones(axes(Fields.level(Y.c, 1))))

    dif_flux_energy = similar(z_bottom, Geometry.WVector{FT})
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

function saturated_surface_conditions(
    surface_scheme::BulkSurfaceScheme,
    T_sfc,
    ts_int,
    uₕ_int,
    z_int,
    z_sfc,
    params,
)
    Cd = surface_scheme.Cd
    Ch = surface_scheme.Ch
    # parameters
    thermo_params = CAP.thermodynamics_params(params)
    sf_params = CAP.surface_fluxes_params(params)

    # get the near-surface thermal state
    T_int = TD.air_temperature(thermo_params, ts_int)
    Rm_int = TD.gas_constant_air(thermo_params, ts_int)
    ρ_sfc =
        TD.air_density(thermo_params, ts_int) *
        (T_sfc / T_int)^(TD.cv_m(thermo_params, ts_int) / Rm_int)

    q_sfc =
        TD.q_vap_saturation_generic(thermo_params, T_sfc, ρ_sfc, TD.Liquid())
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

function saturated_surface_conditions(
    surface_scheme::MoninObukhovSurface,
    T_sfc,
    ts_int,
    uₕ_int,
    z_int,
    z_sfc,
    params,
)
    z0m = surface_scheme.z0m
    z0b = surface_scheme.z0b
    # parameters
    thermo_params = CAP.thermodynamics_params(params)
    sf_params = CAP.surface_fluxes_params(params)

    # get the near-surface thermal state
    T_int = TD.air_temperature(thermo_params, ts_int)
    Rm_int = TD.gas_constant_air(thermo_params, ts_int)

    ρ_sfc =
        TD.air_density(thermo_params, ts_int) *
        (T_sfc / T_int)^(TD.cv_m(thermo_params, ts_int) / Rm_int)
    q_sfc =
        TD.q_vap_saturation_generic(thermo_params, T_sfc, ρ_sfc, TD.Liquid())
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
    (; coupled) = p
    Fields.bycolumn(axes(Y.c.uₕ)) do colidx
        !coupled && get_surface_fluxes!(Y, p, colidx)
        vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t, colidx)
    end
end

function get_surface_fluxes!(Y, p, colidx)
    (; z_sfc, ᶜts, T_sfc) = p
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
    ) = p
    (; uₕ_int_local, surface_normal) = p

    uₕ_int = Spaces.level(Y.c.uₕ[colidx], 1)
    @. uₕ_int_local[colidx] = Geometry.UVVector(uₕ_int)

    surface_conditions[colidx] .=
        saturated_surface_conditions.(
            p.surface_scheme,
            T_sfc[colidx],
            Spaces.level(ᶜts[colidx], 1),
            uₕ_int_local[colidx],
            z_bottom[colidx],
            z_sfc[colidx],
            params,
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
