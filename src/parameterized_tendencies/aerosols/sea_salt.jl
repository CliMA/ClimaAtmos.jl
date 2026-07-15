import SurfaceFluxes as SF
import SurfaceFluxes.Parameters as SFP
import SurfaceFluxes.UniversalFunctions as UF
import ClimaCore.Fields: field_values

#####
##### MOST wind reconstruction
#####

"""
    most_point_wind(sfp, Δz, u★, z₀, L)
    most_layer_mean_wind(sfp, Δz, u★, z₀, L)

MOST wind speed from `SF.compute_profile_value` at aerodynamic height `Δz`
(above the surface): the point value, and the mean over `[z₀, Δz]`
(`LayerAverageScheme`, Nishizawa & Kitamura 2018 — what a finite-volume model
level value represents).
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

# Trapezoidal moment ∫ w(r̂80)·f(r̂80) dr̂80 of the Gong integrand over a
# dimensionless-r80 bin. Because the within-bin shape of the emitted spectrum
# is a fixed pdf (wind and SST factor out of dF/dr80), every sub-bin statistic
# the pipeline needs — number integral, mean particle volume, settling radius,
# lognormal fit — is a moment ratio computed here.
function _gong_bin_moment(w::W, r̂_lo, r̂_hi, ap; N = 512) where {W}
    f(r̂) = w(r̂) * _gong2003_r_integrand(r̂, ap)
    dr̂ = (r̂_hi - r̂_lo) / N
    s = (f(r̂_lo) + f(r̂_hi)) / 2
    for i in 1:(N - 1)
        s += f(r̂_lo + i * dr̂)
    end
    return s * dr̂
end

# RADIUS CONVENTIONS (each consumer documents its basis):
#   - r_dry: dry (solute-only) radius; `ssa_bin_edges` are DRY bounds and the
#     ρSSLTxx tracers carry dry mass.
#   - r80:   deliquesced radius at RH = 80%, the size variable of the Gong
#     (2003) spectrum dF/dr80; r80 = `r80_per_dry` · r_dry (GOCART).
#   - r_wet: deliquesced radius at ambient RH, r_wet = GF(RH) · r_dry (see
#     hygroscopic_growth.jl).
# Dimensionless r80 bin edges over which all Gong moments are taken.
_r̂80_edges(ap) = ap.r80_per_dry .* ap.ssa_bin_edges ./ ap.ssa_r_ref

"""
    sea_salt_bin_flux_scales(params, FT)

Per-bin dimensional number-flux scales `k` (m⁻² s⁻¹ per (m s⁻¹)^`p`, with
`p = gong_wind_exp`), one per emission bin, such that a bin's Gong number
flux is `k * u_10^p` (see [`sea_salt_emission_flux`](@ref)). Each scale is the
dimensionless Gong number integral over the bin's **r80** range (the dry
`ssa_bin_edges` times `r80_per_dry`, made dimensionless by `ssa_r_ref` — the
Gong spectrum lives in r80, not dry radius) times
`gong_dim_factor * ssa_r_ref` — the spectrum's dimensional prefactor per meter
of r80 times the meters spanned by one unit of dimensionless radius — so the
result is independent of the units the fit was published in. `ssa_bin_edges`
is the single source of truth for the number of bins.
"""
function sea_salt_bin_flux_scales(params, ::Type{FT}) where {FT}
    ap = params.prognostic_aerosol_params
    r̂_edges = _r̂80_edges(ap)
    dim_prefactor = ap.gong_dim_factor * ap.ssa_r_ref
    return ntuple(Val(length(r̂_edges) - 1)) do i
        FT(
            dim_prefactor *
            _gong_bin_moment(one, r̂_edges[i], r̂_edges[i + 1], ap),
        )
    end
end

"""
    sea_salt_particle_masses(params, FT)

Mean dry mass (kg) of a single emitted particle per bin,
`ρ_s · (4π/3) · ⟨r_dry³⟩/⟨1⟩` over the bin's emitted Gong sub-bin spectrum
(dry basis, `r_dry = r80 / r80_per_dry`). This is the exact number↔mass moment
of the emitted spectrum, and the same bridge the activation seam inverts
(see [`sea_salt_number_concentration`](@ref)), so emitted mass and diagnosed
number stay consistent.
"""
function sea_salt_particle_masses(params, ::Type{FT}) where {FT}
    ap = params.prognostic_aerosol_params
    ρ_s = params.prescribed_aerosol_params.seasalt_density
    r̂_edges = _r̂80_edges(ap)
    r_dry_per_r̂ = ap.ssa_r_ref / ap.r80_per_dry   # meters of r_dry per unit r̂80
    return ntuple(Val(length(r̂_edges) - 1)) do i
        M̂3 = _gong_bin_moment(r̂ -> r̂^3, r̂_edges[i], r̂_edges[i + 1], ap)
        M̂0 = _gong_bin_moment(one, r̂_edges[i], r̂_edges[i + 1], ap)
        FT(ρ_s * (4π / 3) * r_dry_per_r̂^3 * M̂3 / M̂0)
    end
end

"""
    sea_salt_bin_settling_radii(params, FT)

Mass-flux-weighted dry radius `√(⟨r_dry⁵⟩/⟨r_dry³⟩)` (m) per bin: the single
dry radius whose Stokes speed (∝ r²) carries the bin's settling **mass** flux
(v ∝ r², mass ∝ r³). Settling and dry deposition scale it by the growth
factor `GF(RH)` to get their wet working radius.
"""
function sea_salt_bin_settling_radii(params, ::Type{FT}) where {FT}
    ap = params.prognostic_aerosol_params
    r̂_edges = _r̂80_edges(ap)
    r_dry_per_r̂ = ap.ssa_r_ref / ap.r80_per_dry
    return ntuple(Val(length(r̂_edges) - 1)) do i
        M̂5 = _gong_bin_moment(r̂ -> r̂^5, r̂_edges[i], r̂_edges[i + 1], ap)
        M̂3 = _gong_bin_moment(r̂ -> r̂^3, r̂_edges[i], r̂_edges[i + 1], ap)
        FT(r_dry_per_r̂ * sqrt(M̂5 / M̂3))
    end
end

"""
    sea_salt_moment_cache(seasalt_model, params, FT)

Cache entry with the per-bin Gong-spectrum moments the sea salt tendencies
consume every stage ([`sea_salt_bin_flux_scales`](@ref),
[`sea_salt_particle_masses`](@ref), [`sea_salt_bin_settling_radii`](@ref)).
They are pure functions of the (run-constant) parameters, so `build_cache`
computes them once into `p.tracers.seasalt_moments` instead of re-running the
quadratures in every tendency call. Empty unless sea salt is prognostic.
"""
sea_salt_moment_cache(seasalt_model, params, ::Type{FT}) where {FT} = (;)
sea_salt_moment_cache(::PrognosticSeaSalt, params, ::Type{FT}) where {FT} = (;
    seasalt_moments = (;
        flux_scales = sea_salt_bin_flux_scales(params, FT),
        particle_masses = sea_salt_particle_masses(params, FT),
        settling_radii = sea_salt_bin_settling_radii(params, FT),
    )
)

"""
    sea_salt_bin_lognormal_fits(params, FT)

Per-bin `(r_g, σ_g)` lognormal fit (dry basis) to the truncated Gong sub-bin
spectrum, by number-weighted log-moment matching:
`r_g = exp⟨ln r_dry⟩` (m), `σ_g = exp(std(ln r_dry))`. Required by the
`Mode_κ` activation bridge (see sea_salt_activation.jl), which needs a
lognormal form.
"""
function sea_salt_bin_lognormal_fits(params, ::Type{FT}) where {FT}
    ap = params.prognostic_aerosol_params
    r̂_edges = _r̂80_edges(ap)
    r_dry_per_r̂ = ap.ssa_r_ref / ap.r80_per_dry
    return ntuple(Val(length(r̂_edges) - 1)) do i
        lnr(r̂) = log(r̂ * r_dry_per_r̂)
        M̂0 = _gong_bin_moment(one, r̂_edges[i], r̂_edges[i + 1], ap)
        μ = _gong_bin_moment(lnr, r̂_edges[i], r̂_edges[i + 1], ap) / M̂0
        m2 = _gong_bin_moment(r̂ -> lnr(r̂)^2, r̂_edges[i], r̂_edges[i + 1], ap) / M̂0
        (FT(exp(μ)), FT(exp(sqrt(max(m2 - μ^2, zero(μ))))))
    end
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
    ap = p.params.prognostic_aerosol_params
    T_freeze = TD.Parameters.T_freeze(CAP.thermodynamics_params(p.params))
    u_10 = set_sea_salt_10m_wind!(p.scratch.ᶠtemp_field_level, Y, p)
    (; flux_scales, particle_masses) = p.tracers.seasalt_moments
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
    sea_salt_settling_tendency!(Yₜ, Y, p, t, seasalt_model)

Explicit gravitational settling of the prognostic sea salt bins — a downward
vertical advection at the per-bin, slip-corrected Stokes terminal velocity:

    ∂(ρSSLTxx)/∂t -= ∇·(ρ · w_settle · χ)   (free outflow at the surface)

The velocity is evaluated at the bin's wet mass-flux-weighted radius
(`sea_salt_bin_settling_radii` times the cached growth factor `ᶜsslt_GF`) and
Courant-capped (`settling_courant_max`) for explicit stability; it is
materialized into scratch so the `ᶠright_bias`/`ᶜprecipdivᵥ` stencil kernel
stays small, as precipitation does. The free-outflow bottom boundary deposits
the gravitational flux `V_g · ρSSLTxx` at the surface — the gravitational part
of dry deposition — so [`sea_salt_dry_deposition_tendency!`](@ref) carries
only the turbulent part and nothing is double counted. Grid-mean only:
updraft (`sgsʲs`) bins are not settled (deferred with the
subdomain-sedimentation TODO).
"""
sea_salt_settling_tendency!(Yₜ, Y, p, t, ::Union{Nothing, PrescribedSeaSalt}) =
    nothing
function sea_salt_settling_tendency!(Yₜ, Y, p, t, seasalt::PrognosticSeaSalt)
    FT = eltype(Y)
    ap = p.params.prognostic_aerosol_params
    (; ᶜT, ᶜsslt_GF) = p.precomputed
    grav = FT(CAP.grav(p.params))
    R_d = FT(CAP.R_d(p.params))
    ρ_s = p.params.prescribed_aerosol_params.seasalt_density
    (; settling_radii) = p.tracers.seasalt_moments
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    ᶜΔz = Fields.Δz_field(Y.c)
    dt = float(p.dt)

    for (bin_index, name) in enumerate(bin_names(seasalt))
        ᶜρχ = getproperty(Y.c, Symbol(:ρ, name))
        ᶜρχₜ = getproperty(Yₜ.c, Symbol(:ρ, name))
        r_settle = settling_radii[bin_index]

        # ᶜtemp_scalar is written and consumed within this iteration.
        ᶜw = p.scratch.ᶜtemp_scalar
        @. ᶜw = min(
            sea_salt_settling_velocity(
                r_settle * ᶜsslt_GF,
                sea_salt_wet_density(ρ_s, ap.ρ_water, ᶜsslt_GF),
                Y.c.ρ,
                ᶜT,
                R_d,
                grav,
                (ap,),
            ),
            ap.settling_courant_max * ᶜΔz / dt,
        )
        @. ᶜρχₜ -= ᶜprecipdivᵥ(
            ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ *
            ᶠright_bias(Geometry.WVector(-(ᶜw)) * specific(ᶜρχ, Y.c.ρ)),
        )
    end
    return nothing
end

"""
    sea_salt_dry_deposition_tendency!(Yₜ, Y, p, t, seasalt_model)

Turbulent dry deposition of the prognostic sea salt bins as a surface-flux
sink, `ρ_flux|_sfc = -V_d,turb · ρSSLTxx|₁` with
`V_d,turb = 1/(R_a + R_s)` from [`sea_salt_dry_deposition_velocity`](@ref)
(MOST aerodynamic resistance + Zhang 2001 surface resistance, water/ocean
category everywhere for now). The gravitational part is deposited by the
settling term's free-outflow boundary, so the two sum to the full deposition
velocity without double counting. `V_d,turb` is Courant-capped so the explicit
sink cannot over-deplete the lowest cell in one step. Surface and level-1
fields live on different spaces, so the flux is assembled in one fused
broadcast over their data values, as in [`set_sea_salt_10m_wind!`](@ref).
"""
sea_salt_dry_deposition_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::Union{Nothing, PrescribedSeaSalt},
) = nothing
function sea_salt_dry_deposition_tendency!(
    Yₜ,
    Y,
    p,
    t,
    seasalt::PrognosticSeaSalt,
)
    FT = eltype(Y)
    ap = p.params.prognostic_aerosol_params
    (; sfc_conditions, ᶜT, ᶜsslt_GF) = p.precomputed
    sfp = CAP.surface_fluxes_params(p.params)
    uf_params = SFP.uf_params(sfp)
    κ_vk = SFP.von_karman_const(sfp)
    R_d = FT(CAP.R_d(p.params))
    grav = FT(CAP.grav(p.params))
    ρ_s = p.params.prescribed_aerosol_params.seasalt_density
    roughness_spec = SF.COARE3RoughnessParams{FT}()
    dt = float(p.dt)
    (; settling_radii) = p.tracers.seasalt_moments

    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠz = Fields.coordinate_field(Y.f).z
    z_sfc_values = level_values(ᶠz, Fields.half)
    z1_values = level_values(ᶜz, 1)
    ρ1_values = level_values(Y.c.ρ, 1)
    T1_values = level_values(ᶜT, 1)
    Δz1_values = level_values(Fields.Δz_field(Y.c), 1)
    GF1_values = level_values(ᶜsslt_GF, 1)
    ustar_values = field_values(sfc_conditions.ustar)
    L_values = field_values(sfc_conditions.obukhov_length)
    sfc_flux = p.scratch.sfc_temp_C3
    sfc_flux_values = field_values(sfc_flux)

    for (bin_index, name) in enumerate(bin_names(seasalt))
        ᶜρχ = getproperty(Y.c, Symbol(:ρ, name))
        ᶜρχₜ = getproperty(Yₜ.c, Symbol(:ρ, name))
        r_settle = settling_radii[bin_index]
        ρχ1_values = level_values(ᶜρχ, 1)

        # One fused kernel per bin: the surface settling speed feeds the Zhang
        # Stokes number (uncapped — the Courant cap is a numerical device for
        # the explicit sink, not part of the deposition physics).
        @. sfc_flux_values = C3(
            -min(
                sea_salt_dry_deposition_velocity(
                    sea_salt_settling_velocity(
                        r_settle * GF1_values,
                        sea_salt_wet_density(ρ_s, ap.ρ_water, GF1_values),
                        ρ1_values,
                        T1_values,
                        R_d,
                        grav,
                        (ap,),
                    ),
                    r_settle * GF1_values,
                    ρ1_values,
                    T1_values,
                    z1_values - z_sfc_values,
                    L_values,
                    SF.momentum_roughness(roughness_spec, ustar_values, sfp, nothing),
                    ustar_values,
                    uf_params,
                    κ_vk,
                    R_d,
                    (ap,),
                ),
                ap.settling_courant_max * Δz1_values / dt,
            ) * ρχ1_values,
        )

        ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
        btt = boundary_tendency_scalar(ᶜχ, sfc_flux)
        @. ᶜρχₜ -= btt
    end
    return nothing
end
