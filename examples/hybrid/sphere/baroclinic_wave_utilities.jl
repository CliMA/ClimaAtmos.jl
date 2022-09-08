using Statistics: mean
using SurfaceFluxes
using CloudMicrophysics
const SF = SurfaceFluxes
const CCG = ClimaCore.Geometry
import ClimaAtmos.TurbulenceConvection as TC
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
    (; T_sfc, z_sfc, ᶜp, ᶜσ, ᶜheight_factor, ᶜΔρT, ᶜφ, params) = p # assume ᶜp has been updated

    R_d = FT(CAP.R_d(params))
    κ_d = FT(CAP.kappa_d(params))
    cv_d = FT(CAP.cv_d(params))
    day = FT(CAP.day(params))
    MSLP = FT(CAP.MSLP(params))
    grav = FT(CAP.grav(params))

    z_bottom = Spaces.level(Fields.coordinate_field(Y.c).z, 1)
    z_surface = Fields.Field(Fields.field_values(z_sfc), axes(z_bottom))
    p_sfc = @. MSLP * exp(-grav * z_surface / R_d / T_sfc)

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

    p_int_size = size(parent(ᶜp))
    p_sfc = reshape(parent(p_sfc), (1, p_int_size[2:end]...))
    parent(ᶜσ) .= parent(ᶜp) ./ parent(p_sfc)

    @. ᶜheight_factor = max(0, (ᶜσ - σ_b) / (1 - σ_b))
    @. ᶜΔρT =
        (k_a + (k_s - k_a) * ᶜheight_factor * (cos(ᶜφ)^2)^2) *
        Y.c.ρ *
        ( # ᶜT - ᶜT_equil
            ᶜp / (Y.c.ρ * R_d) - max(
                T_min,
                (
                    T_equator - ΔT_y * sin(ᶜφ)^2 -
                    Δθ_z * log(ᶜp / MSLP) * cos(ᶜφ)^2
                ) * fast_pow(ᶜσ, κ_d),
            )
        )

    @. Yₜ.c.uₕ -= (k_f * ᶜheight_factor) * Y.c.uₕ
    if :ρθ in propertynames(Y.c)
        @. Yₜ.c.ρθ -= ᶜΔρT * fast_pow((MSLP / ᶜp), κ_d)
    elseif :ρe_tot in propertynames(Y.c)
        @. Yₜ.c.ρe_tot -= ᶜΔρT * cv_d
    elseif :ρe_int in propertynames(Y.c)
        @. Yₜ.c.ρe_int -= ᶜΔρT * cv_d
    end
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

vertical∫_col(field::ClimaCore.Fields.CenterExtrudedFiniteDifferenceField) =
    vertical∫_col(field, 1)
vertical∫_col(field::ClimaCore.Fields.FaceExtrudedFiniteDifferenceField) =
    vertical∫_col(field, ClimaCore.Operators.PlusHalf(1))

function vertical∫_col(
    field::ClimaCore.Fields.Field,
    one_index::Union{Int, ClimaCore.Operators.PlusHalf},
)
    Δz = Fields.local_geometry_field(field).∂x∂ξ.components.data.:9
    Ni, Nj, _, _, Nh = size(ClimaCore.Spaces.local_geometry_data(axes(field)))
    planar_field = similar(Fields.level(field, one_index))
    for inds in Iterators.product(1:Ni, 1:Nj, 1:Nh)
        Δz_col = column(Δz, inds...)
        field_col = column(field, inds...)
        planar_field_column = column(planar_field, inds...)
        parent(planar_field_column) .= sum(parent(field_col .* Δz_col))
    end
    return planar_field
end

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

    col_integrated_rain .=
        vertical∫_col(ᶜ3d_rain) ./ FT(CAP.ρ_cloud_liq(params))
    col_integrated_snow .=
        vertical∫_col(ᶜ3d_snow) ./ FT(CAP.ρ_cloud_liq(params))

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

# Note: ᶠv_a and ᶠz_a are 3D projections of 2D Fields (the values of uₕ and z at
#       the first cell center of every column, respectively).
# TODO: Allow ClimaCore to handle both 2D and 3D Fields in a single broadcast.
#       This currently results in a mismatched spaces error.
function vertical_diffusion_boundary_layer_cache(
    Y,
    ::Type{FT};
    surface_scheme = nothing,
    C_E::FT = FT(0),
    diffuse_momentum = true,
    coupled = false,
) where {FT}
    ᶠz_a = similar(Y.f, FT)
    z_bottom = Spaces.level(Fields.coordinate_field(Y.c).z, 1)
    Fields.field_values(ᶠz_a) .=
        Fields.field_values(z_bottom) .* one.(Fields.field_values(ᶠz_a))
    # TODO: fix VIJFH copyto! to remove the one.(...)

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

    surface_scheme_params = if surface_scheme == "bulk"
        (; Cd = FT(0.0044), Ch = FT(0.0044))
    elseif surface_scheme == "monin_obukhov"
        (; z0m = FT(1e-5), z0b = FT(1e-5))
    elseif isnothing(surface_scheme)
        NamedTuple()
    end
    return (;
        surface_scheme,
        ᶠv_a = similar(Y.f, eltype(Y.c.uₕ)),
        ᶠz_a,
        C_E,
        ᶠK_E = similar(Y.f, FT),
        surface_conditions = similar(z_bottom, cond_type),
        dif_flux_uₕ,
        dif_flux_energy = similar(z_bottom, Geometry.WVector{FT}),
        dif_flux_ρq_tot,
        surface_scheme_params...,
        diffuse_momentum,
        coupled,
        z_bottom,
    )
end

function eddy_diffusivity_coefficient(C_E, norm_v_a, z_a, p)
    p_pbl = FT(85000)
    p_strato = FT(10000)
    K_E = C_E * norm_v_a * z_a
    return p > p_pbl ? K_E : K_E * exp(-((p_pbl - p) / p_strato)^2)
end

function constant_T_saturated_surface_conditions(
    T_sfc,
    ts_int,
    uₕ_int,
    z_int,
    z_sfc,
    Cd,
    Ch,
    params,
)
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

function variable_T_saturated_surface_conditions(
    T_sfc,
    ts_int,
    uₕ_int,
    z_int,
    z_sfc,
    z0m,
    z0b,
    params,
)
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
    ᶜρ = Y.c.ρ
    (; z_sfc, ᶜts, ᶜp, T_sfc, ᶠv_a, ᶠz_a, ᶠK_E, C_E) = p # assume ᶜts and ᶜp have been updated
    (;
        surface_conditions,
        dif_flux_uₕ,
        dif_flux_energy,
        dif_flux_ρq_tot,
        diffuse_momentum,
        coupled,
        z_bottom,
        params,
    ) = p
    if p.surface_scheme == "bulk"
        (; Cd, Ch) = p
    elseif p.surface_scheme == "monin_obukhov"
        (; z0m, z0b) = p
    end
    surf_flux_params = CAP.surface_fluxes_params(params)

    ᶠgradᵥ = Operators.GradientC2F() # apply BCs to ᶜdivᵥ, which wraps ᶠgradᵥ

    Fields.field_values(ᶠv_a) .=
        Fields.field_values(Spaces.level(Y.c.uₕ, 1)) .*
        one.(Fields.field_values(ᶠz_a)) # TODO: fix VIJFH copyto! to remove this
    @. ᶠK_E = eddy_diffusivity_coefficient(C_E, norm(ᶠv_a), ᶠz_a, ᶠinterp(ᶜp))

    # TODO: Revisit z_surface construction when "space is not same instance" error is dealt with
    ᶜz_field = Fields.coordinate_field(Y.c).z
    ᶜz_interior = Fields.field_values(Fields.level(ᶜz_field, 1))
    z_surface = Fields.field_values(z_sfc)

    if !coupled
        if p.surface_scheme == "bulk"
            surface_conditions .=
                constant_T_saturated_surface_conditions.(
                    T_sfc,
                    Spaces.level(ᶜts, 1),
                    Geometry.UVVector.(Spaces.level(Y.c.uₕ, 1)),
                    Fields.Field(ᶜz_interior, axes(z_bottom)),
                    Fields.Field(z_surface, axes(z_bottom)),
                    Cd,
                    Ch,
                    params,
                )
        elseif p.surface_scheme == "monin_obukhov"
            surface_conditions .=
                variable_T_saturated_surface_conditions.(
                    T_sfc,
                    Spaces.level(ᶜts, 1),
                    Geometry.UVVector.(Spaces.level(Y.c.uₕ, 1)),
                    Fields.Field(ᶜz_interior, axes(z_bottom)),
                    Fields.Field(z_surface, axes(z_bottom)),
                    z0m,
                    z0b,
                    params,
                )
        end
    end

    if diffuse_momentum
        if !coupled
            ρτxz = surface_conditions.ρτxz
            ρτyz = surface_conditions.ρτyz
            u_space = axes(ρτxz) # TODO: delete when "space not the same instance" error is dealt with
            normal = Geometry.WVector.(ones(u_space)) # TODO: this will need to change for topography
            ρ_1 = Fields.Field(
                Fields.field_values(Fields.level(Y.c.ρ, 1)),
                u_space,
            ) # TODO: delete when "space not the same instance" error is dealt with
            parent(dif_flux_uₕ) .=  # TODO: remove parent when "space not the same instance" error is dealt with
                parent(
                    Geometry.Contravariant3Vector.(normal) .⊗
                    Geometry.Covariant12Vector.(
                        Geometry.UVVector.(ρτxz ./ ρ_1, ρτyz ./ ρ_1)
                    ),
                )
        end
        ᶜdivᵥ = Operators.DivergenceF2C(
            top = Operators.SetValue(
                Geometry.Contravariant3Vector(FT(0)) ⊗
                Geometry.Covariant12Vector(FT(0), FT(0)),
            ),
            bottom = Operators.SetValue(.-dif_flux_uₕ),
        )
        @. Yₜ.c.uₕ += ᶜdivᵥ(ᶠK_E * ᶠgradᵥ(Y.c.uₕ))
    end

    if :ρe_tot in propertynames(Y.c)
        if !coupled
            if isnothing(p.surface_scheme)
                @. dif_flux_energy *= 0
            else
                @. dif_flux_energy = Geometry.WVector(
                    surface_conditions.shf + surface_conditions.lhf,
                )
            end
        end
        ᶜdivᵥ = Operators.DivergenceF2C(
            top = Operators.SetValue(Geometry.WVector(FT(0))),
            bottom = Operators.SetValue(.-dif_flux_energy),
        )
        @. Yₜ.c.ρe_tot +=
            ᶜdivᵥ(ᶠK_E * ᶠinterp(ᶜρ) * ᶠgradᵥ((Y.c.ρe_tot + ᶜp) / ᶜρ))
    end

    if :ρq_tot in propertynames(Y.c)
        if !coupled
            if isnothing(p.surface_scheme)
                @. dif_flux_ρq_tot *= 0
            else
                @. dif_flux_ρq_tot = Geometry.WVector(surface_conditions.E)
            end
        end
        ᶜdivᵥ = Operators.DivergenceF2C(
            top = Operators.SetValue(Geometry.WVector(FT(0))),
            bottom = Operators.SetValue(.-dif_flux_ρq_tot),
        )
        @. Yₜ.c.ρq_tot += ᶜdivᵥ(ᶠK_E * ᶠinterp(ᶜρ) * ᶠgradᵥ(Y.c.ρq_tot / ᶜρ))
        @. Yₜ.c.ρ += ᶜdivᵥ(ᶠK_E * ᶠinterp(ᶜρ) * ᶠgradᵥ(Y.c.ρq_tot / ᶜρ))
    end
end
