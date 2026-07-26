import SurfaceFluxes as SF
import SurfaceFluxes.UniversalFunctions as UF
import ClimaCore.Fields: field_values

#####
##### MOST wind reconstruction
#####

"""
    most_point_wind(sfp, Δz, u★, z₀, L)
    most_layer_mean_wind(sfp, Δz, u★, z₀, L)
    most_cell_mean_wind(sfp, Δz_bot, Δz_top, u★, z₀, L)

MOST wind speed from `SF.compute_profile_value` at aerodynamic height `Δz`
(above the surface): the point value, the mean over `[z₀, Δz]`
(`LayerAverageScheme`, Nishizawa & Kitamura 2018 — what a finite-volume model
level value represents), and the mean over an elevated cell `[Δz_bot, Δz_top]`
(from differencing `Δz⋅⟨u⟩(Δz)`; requires `Δz_bot > z₀`).
"""
most_point_wind(sfp, Δz, u★, z₀, L) = SF.compute_profile_value(
    sfp,
    safe_obukhov_length(L),
    z₀,
    Δz,
    u★,
    zero(u★),
    UF.MomentumTransport(),
    SF.PointValueScheme(),
)
most_layer_mean_wind(sfp, Δz, u★, z₀, L) = SF.compute_profile_value(
    sfp,
    safe_obukhov_length(L),
    z₀,
    Δz,
    u★,
    zero(u★),
    UF.MomentumTransport(),
    SF.LayerAverageScheme(),
)
most_cell_mean_wind(sfp, Δz_bot, Δz_top, u★, z₀, L) =
    (
        Δz_top * most_layer_mean_wind(sfp, Δz_top, u★, z₀, L) -
        Δz_bot * most_layer_mean_wind(sfp, Δz_bot, u★, z₀, L)
    ) / (Δz_top - Δz_bot)

safe_obukhov_length(L) = ifelse(iszero(L), oftype(L, 1e-4), L)

level_wind_speed(u, local_geometry) = Geometry._norm(u, local_geometry)

"""
    most_10m_wind(sfp, roughness_spec, Δz₁, u₁, u★, L, anchored)

Pointwise 10 m point-value wind speed. With `anchored = true`, the MOST point
value at 10 m is anchored on `u₁`, the model's cell-1 layer-mean wind over
aerodynamic depth `Δz₁`: `u10 = u₁ ⋅ F_point(10) / F_mean(Δz₁)`, so the
gustiness contribution to `u★` cancels. With `anchored = false`, the wind is
recovered from `(u★, L, z₀)` alone.
"""
function most_10m_wind(sfp, roughness_spec, Δz₁, u₁, u★, L, anchored)
    z₀ = SF.momentum_roughness(roughness_spec, u★, sfp, nothing)
    u_10 = most_point_wind(sfp, oftype(u★, 10), u★, z₀, L)
    ratio = ifelse(
        anchored,
        u₁ / most_layer_mean_wind(sfp, Δz₁, u★, z₀, L),
        one(u★),
    )
    return max(u_10 * ratio, zero(u★))
end

"""
    most_layer1_wind(sfp, roughness_spec, Δz₁, Δz₂, u₂, u★, L, anchored)

Pointwise prediction of the cell-1 layer-mean wind, for validating the
anchoring approach against the actual level-1 wind. With `anchored = true`,
the MOST cell-1 mean is anchored on `u₂`, the model's cell-2 layer-mean wind
(cell tops at aerodynamic heights `Δz₁`, `Δz₂`); with `anchored = false` it is
recovered from `(u★, L, z₀)` alone.
"""
function most_layer1_wind(sfp, roughness_spec, Δz₁, Δz₂, u₂, u★, L, anchored)
    z₀ = SF.momentum_roughness(roughness_spec, u★, sfp, nothing)
    u₁ = most_layer_mean_wind(sfp, Δz₁, u★, z₀, L)
    ratio = ifelse(
        anchored,
        u₂ / most_cell_mean_wind(sfp, Δz₁, Δz₂, u★, z₀, L),
        one(u★),
    )
    return max(u₁ * ratio, zero(u★))
end

# Helper with surface field-interior fields mismatch
level_values(f, lev) = field_values(Fields.level(f, lev))

"""
    set_sea_salt_10m_wind!(u_10, Y, p; anchored = true)

Write the 10 m point wind speed into the surface field `u_10` (see
[`most_10m_wind`](@ref)). The emission tendency uses the default
`anchored = true`; `anchored = false` is kept for diagnostics.
"""
function set_sea_salt_10m_wind!(u_10, Y, p; anchored = true)
    FT = eltype(Y)
    sfp = CAP.surface_fluxes_params(p.params)
    roughness_spec = SF.COARE3RoughnessParams{FT}()
    (; sfc_conditions) = p.precomputed

    u_10_values = field_values(u_10)
    ustar_values = field_values(sfc_conditions.ustar)
    L_values = field_values(sfc_conditions.obukhov_length)
    ᶠz = Fields.coordinate_field(Y.f).z
    z_sfc_values = level_values(ᶠz, Fields.half)
    z_f1_values = level_values(ᶠz, Fields.half + 1)
    u1_values = level_values(Y.c.uₕ, 1)
    lg1_values = level_values(Fields.local_geometry_field(Y.c), 1)

    @. u_10_values = most_10m_wind(
        sfp,
        roughness_spec,
        z_f1_values - z_sfc_values,
        level_wind_speed(u1_values, lg1_values),
        ustar_values,
        L_values,
        anchored,
    )
    return u_10
end

"""
    set_most_layer1_wind!(out, Y, p; anchored = true, bias = false)

Write the predicted cell-1 layer-mean wind speed into the surface field `out`
(see [`most_layer1_wind`](@ref)); with `bias = true`, subtract the actual
level-1 wind so the field is the prediction error. Diagnostic validation of
the anchoring approach: the `anchored = true` prediction uses level 2 the way
the 10 m wind uses level 1.
"""
function set_most_layer1_wind!(out, Y, p; anchored = true, bias = false)
    FT = eltype(Y)
    sfp = CAP.surface_fluxes_params(p.params)
    roughness_spec = SF.COARE3RoughnessParams{FT}()
    (; sfc_conditions) = p.precomputed

    out_values = field_values(out)
    ustar_values = field_values(sfc_conditions.ustar)
    L_values = field_values(sfc_conditions.obukhov_length)
    ᶠz = Fields.coordinate_field(Y.f).z
    z_sfc_values = level_values(ᶠz, Fields.half)
    z_f1_values = level_values(ᶠz, Fields.half + 1)
    z_f2_values = level_values(ᶠz, Fields.half + 2)
    ᶜlg = Fields.local_geometry_field(Y.c)
    u2_values = level_values(Y.c.uₕ, 2)
    lg2_values = level_values(ᶜlg, 2)

    @. out_values = most_layer1_wind(
        sfp,
        roughness_spec,
        z_f1_values - z_sfc_values,
        z_f2_values - z_sfc_values,
        level_wind_speed(u2_values, lg2_values),
        ustar_values,
        L_values,
        anchored,
    )
    if bias
        u1_values = level_values(Y.c.uₕ, 1)
        lg1_values = level_values(ᶜlg, 1)
        @. out_values -= level_wind_speed(u1_values, lg1_values)
    end
    return out
end

#####
##### Gong (2003) source function with Jaegle (2011) SST correction
#####

"""
    _gong2003_r_integrand(r̂, ap)

Radius-dependent part of the [Gong2003](@cite) source function at
dimensionless dry radius `r̂ = r / ssa_r_ref`, with the shape parameter
`gong_theta` and the coefficient vectors `gong_A`, `gong_B`, `gong_F` read
from `ap = params.prognostic_aerosol_params`. The Gong fit is expressed
entirely in `r̂`, so the integrand is dimensionless; all dimensional
information enters through `ssa_r_ref` and `gong_dim_factor` (see
[`sea_salt_bin_flux_scales`](@ref)).
"""
function _gong2003_r_integrand(r̂, ap)
    θ = ap.gong_theta
    A = ap.gong_A
    B = ap.gong_B
    F = ap.gong_F
    a = A[1] * (1 + θ * r̂)^(A[2] * r̂^A[3])
    b = 1 - (log10(r̂) / B)
    return F[1] * r̂^(-a) * (1 + F[2] * r̂^F[3]) * 10^(F[4] * exp(-b^2))
end

function _gong_bin_integral(r̂_lo, r̂_hi, ap; N = 512)
    dr̂ = (r̂_hi - r̂_lo) / N
    s = (_gong2003_r_integrand(r̂_lo, ap) + _gong2003_r_integrand(r̂_hi, ap)) / 2
    for i in 1:(N - 1)
        s += _gong2003_r_integrand(r̂_lo + i * dr̂, ap)
    end
    return s * dr̂
end

"""
    sea_salt_bin_flux_scales(params, FT)

Per-bin dimensional number-flux scales `k` (m⁻² s⁻¹ per (m s⁻¹)^`p`, with
`p = gong_wind_exp`), one per emission bin, such that a bin's Gong number
flux is `k * u_10^p` (see [`sea_salt_emission_flux`](@ref)). Each scale is the
dimensionless Gong integral over the bin times `gong_dim_factor * ssa_r_ref`
— the source spectrum's dimensional prefactor (per meter of dry radius) times
the meters spanned by one unit of dimensionless radius — so the result is
independent of the units the fit was published in. The bin edges
(`ssa_bin_edges`, made dimensionless by `ssa_r_ref`) are the single source of
truth for the number of bins.
"""
function sea_salt_bin_flux_scales(params, ::Type{FT}) where {FT}
    ap = params.prognostic_aerosol_params
    r̂_edges = ap.ssa_bin_edges ./ ap.ssa_r_ref
    dim_prefactor = ap.gong_dim_factor * ap.ssa_r_ref
    return ntuple(Val(length(r̂_edges) - 1)) do i
        FT(dim_prefactor * _gong_bin_integral(r̂_edges[i], r̂_edges[i + 1], ap))
    end
end

"""
    sea_salt_particle_masses(params, FT)

Dry mass (kg) of a single sea salt particle for each bin, from the MERRA-2
bin radii and salt density in ClimaParams.
"""
function sea_salt_particle_masses(params, ::Type{FT}) where {FT}
    ap = params.prescribed_aerosol_params
    bin_radii = (
        ap.SSLT01_radius,
        ap.SSLT02_radius,
        ap.SSLT03_radius,
        ap.SSLT04_radius,
        ap.SSLT05_radius,
    )
    return map(radius -> FT(4π / 3 * radius^3 * ap.seasalt_density), bin_radii)
end

"""
    sea_salt_emission_flux(u_10, T_sfc_C, flux_scale, ap, sst_adjustment)

Upward sea salt number flux (m⁻² s⁻¹) for one bin: the bin's dimensional flux
scale (see [`sea_salt_bin_flux_scales`](@ref)) times `u_10^gong_wind_exp`
([Gong2003](@cite)). With `sst_adjustment = true`, the flux is scaled by the
[Jaegle2011](@cite) cubic SST correction, evaluated at the surface temperature
`T_sfc_C` in °C and clamped at zero. When broadcasting, pass
`ap = params.prognostic_aerosol_params` wrapped in a 1-tuple (`(ap,)`) so the
`NamedTuple` is treated as a scalar, as in the microphysics aerosol
activation.
"""
function sea_salt_emission_flux(u_10, T_sfc_C, flux_scale, ap, sst_adjustment)
    number_flux = flux_scale * abs(u_10)^ap.gong_wind_exp
    sst_adjustment || return number_flux
    c = ap.jaegle_C
    sst_factor = max(evalpoly(T_sfc_C, (c[1], c[2], -c[3], c[4])), zero(T_sfc_C))
    return number_flux * sst_factor
end

#####
##### Tendencies
#####

"""
    sea_salt_emission_tendency!(Yₜ, Y, p, t, seasalt_model; sst_adjustment = false)

Apply surface emission tendencies for prognostic sea salt bins (ρSSLT01 …).

The per-bin mass flux (kg m⁻² s⁻¹) is [`sea_salt_emission_flux`](@ref) scaled
by the particle mass and the ocean fraction, applied as a bottom boundary
condition using [`boundary_tendency_scalar`](@ref), in lowest model layer.
"""
sea_salt_emission_tendency!(Yₜ, Y, p, t, ::Union{Nothing, PrescribedSeaSalt}) =
    nothing
function sea_salt_emission_tendency!(
    Yₜ,
    Y,
    p,
    t,
    seasalt::PrognosticSeaSalt;
    sst_adjustment = false,
)
    FT = eltype(Y)
    ap = p.params.prognostic_aerosol_params
    T_freeze = TD.Parameters.T_freeze(CAP.thermodynamics_params(p.params))
    u_10 = set_sea_salt_10m_wind!(p.scratch.ᶠtemp_field_level, Y, p)
    flux_scales = sea_salt_bin_flux_scales(p.params, FT)
    particle_masses = sea_salt_particle_masses(p.params, FT)
    (; T_sfc) = p.precomputed.sfc_conditions
    ocean_fraction = p.ocean_fraction
    sfc_flux = p.scratch.sfc_temp_C3

    for (bin_index, name) in enumerate(bin_names(seasalt))
        ᶜρχ = getproperty(Y.c, Symbol(:ρ, name))
        ᶜρχₜ = getproperty(Yₜ.c, Symbol(:ρ, name))
        ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))

        @. sfc_flux = C3(
            sea_salt_emission_flux(
                u_10,
                T_sfc - T_freeze,
                flux_scales[bin_index],
                (ap,),
                sst_adjustment,
            ) *
            particle_masses[bin_index] *
            ocean_fraction,
        )

        btt = boundary_tendency_scalar(ᶜχ, sfc_flux)
        @. ᶜρχₜ -= btt

        if p.atmos.turbconv_model isa PrognosticEDMFX
            # assuming one updraft
            ᶜχʲₜ = getproperty(Yₜ.c.sgsʲs.:(1), name)
            @. ᶜχʲₜ -= specific(btt, p.precomputed.ᶜρʲs.:(1))
        end
    end
end

"""
    sea_salt_deposition_tendency!(Yₜ, Y, p, t, seasalt_model)

Apply deposition tendencies to all prognostic sea salt bins.

Currently implemented as a simple exponential decay. Separate wet and dry
deposition tendencies are forthcoming.
"""
sea_salt_deposition_tendency!(Yₜ, Y, p, t, ::Union{Nothing, PrescribedSeaSalt}) =
    nothing
function sea_salt_deposition_tendency!(Yₜ, Y, p, t, seasalt::PrognosticSeaSalt)
    FT = eltype(Y)
    ap = p.params.prognostic_aerosol_params

    # `τ_ssa` (days) is a mean lifetime, so the decay rate is 1/τ.
    λ = FT(1 / (ap.τ_ssa * ap.day))

    for name in bin_names(seasalt)
        ᶜρχ = getproperty(Y.c, Symbol(:ρ, name))
        ᶜρχₜ = getproperty(Yₜ.c, Symbol(:ρ, name))
        @. ᶜρχₜ -= λ * ᶜρχ
    end
end
