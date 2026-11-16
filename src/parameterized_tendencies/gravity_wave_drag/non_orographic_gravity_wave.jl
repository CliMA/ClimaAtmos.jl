#####
##### Non-orographic gravity wave parameterization
#####
using UnrolledUtilities
import ClimaCore.DataLayouts
import ClimaCore.Domains
import ClimaCore.Meshes
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore.Operators as Operator
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT

"""
    _beres_latent_heating(mp_bulk, mp_imp, thp, ρ, T, w, q_tot, q_lcl, q_icl, q_rai, q_sno)

Return Beres' transport-free latent heating rate for one updraft [K/s]:

    Q_lat = (1/cp⁽ʲ⁾) Σ_p L_p R_p⁽ʲ⁾

where `p` runs over the microphysical phase-change processes (e.g., condensation,
evaporation), `L_p` is the latent heat released or absorbed by process `p` [J/kg], and
`R_p⁽ʲ⁾` is the mass-mixing-ratio tendency [kg/kg/s], built from the CloudMicrophysics
1-moment aggregated phase-change tendencies `dq_{lcl,icl,rai,sno}_dt`. `cp⁽ʲ⁾` is the
subdomain moist heat capacity `TD.cp_m(thp, q_tot, q_lcl, q_icl)`.

Relative to a vapor reference, the liquid reservoir (`lcl + rai`) is weighted by `L_v` and
the ice reservoir (`icl + sno`) by `L_s`. Within the reservoir sums, same-phase mass
transfers cancel, and liquid↔ice transfers become `L_f = L_s − L_v`, so

    Q_lat = (1/cp⁽ʲ⁾) [ L_v·(dq_lcl_dt + dq_rai_dt) + L_s·(dq_icl_dt + dq_sno_dt) ]

with reference-`T₀` constants `L_v` and `L_s`. The 1M species tendencies are the sum of
two CM BMT calls — one on the bulk parameter set (every explicit process) and one on the
isolated implicit set (`cloud_liquid_formation` / `cloud_ice_formation` only) — so the
latent heating reflects every active 1M process regardless of which stage integrates each.
Condensate inputs are floored at zero and `q_tot` at the condensate sum before evaluating
`cp_m` and the tendencies. Called from `compute_beres_convective_heating!` when
`heating_latent` is enabled.
"""
@inline function _beres_latent_heating(
    mp_bulk,
    mp_imp,
    thp,
    ρ,
    T,
    w,
    q_tot,
    q_lcl,
    q_icl,
    q_rai,
    q_sno,
)
    FT = typeof(ρ)
    q_lcl = max(zero(FT), q_lcl)
    q_icl = max(zero(FT), q_icl)
    q_rai = max(zero(FT), q_rai)
    q_sno = max(zero(FT), q_sno)
    q_tot = max(zero(FT), q_tot, q_lcl + q_icl + q_rai + q_sno)
    src_bulk = BMT.bulk_microphysics_tendencies(
        BMT.InstantaneousVerbose(),
        BMT.Microphysics1Moment(),
        mp_bulk,
        thp,
        ρ,
        T,
        w,
        q_tot,
        q_lcl,
        q_icl,
        q_rai,
        q_sno,
    )
    src_imp = BMT.bulk_microphysics_tendencies(
        BMT.InstantaneousVerbose(),
        BMT.Microphysics1Moment(),
        mp_imp,
        thp,
        ρ,
        T,
        w,
        q_tot,
        q_lcl,
        q_icl,
        q_rai,
        q_sno,
    )
    Lv = TD.Parameters.LH_v0(thp)
    Ls = TD.Parameters.LH_s0(thp)
    cpʲ = TD.cp_m(thp, q_tot, q_lcl, q_icl)
    dq_lcl_dt = src_bulk.dq_lcl_dt + src_imp.dq_lcl_dt
    dq_icl_dt = src_bulk.dq_icl_dt + src_imp.dq_icl_dt
    dq_rai_dt = src_bulk.dq_rai_dt + src_imp.dq_rai_dt
    dq_sno_dt = src_bulk.dq_sno_dt + src_imp.dq_sno_dt
    return (Lv * (dq_lcl_dt + dq_rai_dt) + Ls * (dq_icl_dt + dq_sno_dt)) / cpʲ
end

# Per-column precomputed Beres launch spectrum, stored in the gravity-wave
# cache. The spectrum is deliberately wrapped in a struct rather than stored as
# a bare NTuple{nc, FT} field: with the bare tuple, the eltype of the 16-slot
# waveforcing input broadcast is a length-16 tuple type with a length-nc tuple
# nested inside, and Julia's type-complexity limiter truncates that nested
# tuple to a TypeVar while inferring the accumulate kernel. The abstractly
# typed slot then boxes, which the GPU cannot compile (InvalidIRError:
# gpu_gc_pool_alloc / checked getfield). A struct is opaque to the tuple
# complexity walk (its type parameters {NC, FT} are shallow), so the input
# eltype stays concrete. The stored bits are identical either way.
struct BeresB0{NC, FT}
    B0::NTuple{NC, FT}
end

"""
    non_orographic_gravity_wave_cache(Y, atmos::AtmosModel)
    non_orographic_gravity_wave_cache(Y, ::Nothing)
    non_orographic_gravity_wave_cache(Y, gw::NonOrographicGravityWave)

Allocate the non-orographic gravity wave (NOGW) cache for the state `Y`.

Returns an empty `NamedTuple` when the model is disabled. Otherwise the cache holds the
phase-speed grid `gw_c` (`nc = floor(2·cmax/dc + 1)` bins spanning `[-cmax, cmax]`), the
latitude-dependent AD99 source amplitude and band widths, per-column source/damp level
buffers, the working fields for the upward-marching kernel, and the accumulated forcings
`uforcing`/`vforcing` [m/s²]. Beres source-property fields are appended only when
`gw.beres_source` is a `BeresSourceParams` (see `_beres_cache_fields`).

The `AtmosModel` method errors when a Beres source is configured without a
`PrognosticEDMFX` turbulence-convection model with at least one updraft, whose draft state
the source reads.

The column and sphere branches differ in how the AD99 launch level is set: a column uses
the height `source_height` and damps from the model top down, a sphere uses
`source_pressure` and `damp_pressure`. Any other horizontal space is an error.

See also `non_orographic_gravity_wave_compute_tendency!`. The scheme is documented
on the *Non-orographic Gravity Waves* page.
"""
function non_orographic_gravity_wave_cache(Y, atmos::AtmosModel)
    nogw = atmos.non_orographic_gravity_wave
    # The Beres source reads the per-draft state of the prognostic EDMF scheme. The
    # config getter checks this for YAML runs; this check covers models built in Julia.
    if !isnothing(nogw) && !isnothing(nogw.beres_source)
        (; turbconv_model) = atmos
        turbconv_model isa PrognosticEDMFX &&
        n_mass_flux_subdomains(turbconv_model) >= 1 ||
            error(
                "The Beres convective gravity-wave source requires a PrognosticEDMFX " *
                "turbulence-convection model with at least one updraft; got " *
                "$(typeof(turbconv_model)).",
            )
    end
    return non_orographic_gravity_wave_cache(Y, nogw)
end

non_orographic_gravity_wave_cache(Y, ::Nothing) = (;)

"""
    _beres_cache_fields(Y, ::Nothing, ::Val)
    _beres_cache_fields(Y, ::BeresSourceParams, ::Val{nc})

Allocate the Beres convective-source fields of the NOGW cache. `Val(nc)` is the length of
the phase-speed grid and sizes the precomputed launch-spectrum fields.

Returns `(;)` for an AD99-only run, so those fields are absent from the cache and the Beres
branches are never entered. Otherwise returns the per-column source properties (`gw_Q0`,
`gw_h_heat`, `gw_u_heat`, `gw_v_heat`, `gw_N_source`, `gw_a_cover`, `gw_zbot`, `gw_ztop`,
`gw_beres_active`), the two heating profiles (`gw_Q_conv`, `gw_Q_conv_ic`), the shared
seven-slot column-reduction buffer `gw_reduce_result`, the Beres launch-level state
`beres_source_ρ_z_u_v_level`, the diagnostic fields `gw_halfsine`, `gw_launch_flux`,
`gw_c_centroid`, `gw_beres_mech_flux`, and the Beres-G / mechanical-source launch state:
the surface height under each column `gw_z_surface` [m], from which every Beres height is
measured, the launch-filter winds `gw_beres_wind_range` (`(u_min, u_max, v_min, v_max, u_L, v_L)` over the heating layer up to the launch level, with `u_L`, `v_L` at the launch
level) and `gw_beres_u_profile`, `gw_beres_v_profile` (u and v on the gather grid), the
precomputed per-column launch spectra `gw_beres_B0_u`, `gw_beres_B0_v` (each a
`BeresB0{nc, FT}`), the gathered in-cloud heating `gw_beres_gprofile` [K/s], an
`NTuple{N_BERES_PROFILE, FT}` (on `[max(z_bot, z_bot_floor), z_top]`), the mean model
level thickness over that window `gw_beres_dz_level` [m], and the obstacle-level updraft
velocity `gw_w_obstacle` [m/s].

Called from `non_orographic_gravity_wave_cache`.
"""
_beres_cache_fields(Y, ::Nothing, ::Val) = (;)

# The surface height [m] under each column, the height of the bottom face, as a level
# field on the cell centres (zero on a flat grid).
function _surface_height(Y)
    z_surface = similar(Fields.level(Y.c.ρ, 1))
    ᶠz = Fields.coordinate_field(Spaces.face_space(axes(Y.c))).z
    Fields.field_values(z_surface) .= Fields.field_values(Fields.level(ᶠz, half))
    return z_surface
end
function _beres_cache_fields(Y, ::BeresSourceParams, ::Val{nc}) where {nc}
    FT = Spaces.undertype(axes(Y.c))
    return (;
        gw_Q0 = similar(Fields.level(Y.c.ρ, 1)),
        gw_h_heat = similar(Fields.level(Y.c.ρ, 1)),
        gw_u_heat = similar(Fields.level(Y.c.ρ, 1)),
        gw_v_heat = similar(Fields.level(Y.c.ρ, 1)),
        gw_N_source = similar(Fields.level(Y.c.ρ, 1)),
        gw_beres_active = similar(Fields.level(Y.c.ρ, 1)),
        gw_zbot = similar(Fields.level(Y.c.ρ, 1)),
        gw_ztop = similar(Fields.level(Y.c.ρ, 1)),
        gw_Q_conv = similar(Y.c.ρ),
        gw_Q_conv_ic = similar(Y.c.ρ),
        # Containers for diagnostics (zero-initialized so a diagnostic read
        # before the first GW callback, e.g. right after a restart, is exact)
        gw_halfsine = zero(Y.c.ρ),
        gw_launch_flux = zero(Fields.level(Y.c.ρ, 1)),
        gw_c_centroid = zero(Fields.level(Y.c.ρ, 1)),
        gw_beres_mech_flux = zero(Fields.level(Y.c.ρ, 1)),
        gw_a_cover = similar(Fields.level(Y.c.ρ, 1)),
        gw_reduce_result = similar(
            Fields.level(Y.c.ρ, 1),
            Tuple{FT, FT, FT, FT, FT, FT, FT},
        ),
        # Beres launch-level state
        beres_source_ρ_z_u_v_level =
        similar(Fields.level(Y.c.ρ, 1), Tuple{FT, FT, FT, FT, FT}),
        # Beres-G (Ext 1) + mechanical (Ext 3) per-column state. The launch
        # spectrum B0 is precomputed per column (it is AD99-wavenumber
        # independent), so the propagation kernel reads it rather than
        # rebuilding it inside the `ink` loop. gprofile = gathered
        # in-cloud heating shape; w_obstacle = obstacle-level updraft velocity.
        gw_beres_B0_u = similar(Fields.level(Y.c.ρ, 1), BeresB0{nc, FT}),
        gw_beres_B0_v = similar(Fields.level(Y.c.ρ, 1), BeresB0{nc, FT}),
        gw_beres_gprofile =
        similar(Fields.level(Y.c.ρ, 1), NTuple{N_BERES_PROFILE, FT}),
        gw_beres_dz_level = similar(Fields.level(Y.c.ρ, 1)),
        gw_z_surface = _surface_height(Y),
        # Launch-filter winds: the range of u and v over the heating layer up to the
        # launch level and the winds there, and u, v on the gather grid.
        gw_beres_wind_range =
        similar(Fields.level(Y.c.ρ, 1), NTuple{6, FT}),
        gw_beres_u_profile =
        similar(Fields.level(Y.c.ρ, 1), NTuple{N_BERES_PROFILE, FT}),
        gw_beres_v_profile =
        similar(Fields.level(Y.c.ρ, 1), NTuple{N_BERES_PROFILE, FT}),
        gw_w_obstacle = similar(Fields.level(Y.c.ρ, 1)),
    )
end

function non_orographic_gravity_wave_cache(Y, gw::NonOrographicGravityWave)
    if iscolumn(axes(Y.c))
        FT = Spaces.undertype(axes(Y.c))
        (; source_height, Bw, Bn, Bt_0, dc, cmax, c0, nk, cw, cn) = gw

        nc = Int(floor(FT(2 * cmax / dc + 1)))
        c = ntuple(n -> FT((n - 1) * dc - cmax), Val(nc))
        source_ρ_z_u_v_level =
            similar(Fields.level(Y.c.ρ, 1), Tuple{FT, FT, FT, FT, FT})
        ᶜlevel = similar(Y.c.ρ, FT)
        for i in 1:Spaces.nlevels(axes(Y.c.ρ))
            fill!(Fields.level(ᶜlevel, i), i)
        end

        return (;
            gw_source_height = source_height,
            gw_source_ampl = Bt_0 .* ones(FT, axes(Fields.level(Y.c.ρ, 1))),
            gw_Bw = Bw .* ones(FT, axes(Fields.level(Y.c.ρ, 1))),
            gw_Bn = Bn .* ones(FT, axes(Fields.level(Y.c.ρ, 1))),
            gw_c = c,
            gw_dc = dc,
            gw_cmax = cmax,
            gw_cw = cw .* ones(FT, axes(Fields.level(Y.c.ρ, 1))),
            gw_cn = cn .* ones(FT, axes(Fields.level(Y.c.ρ, 1))),
            gw_c0 = c0,
            gw_flag = ones(FT, axes(Fields.level(Y.c.ρ, 1))),
            gw_nk = Int(nk),
            ᶜbuoyancy_frequency = similar(Y.c.ρ),
            ᶜuv_temp = similar(Y.c, Geometry.UVVector{FT}),
            gw_temp_surface = similar(Fields.level(Y.c.ρ, 1)),
            source_ρ_z_u_v_level,
            damp_level = similar(Fields.level(Y.c.ρ, 1)),
            ᶜlevel,
            u_waveforcing = similar(Y.c.ρ),
            v_waveforcing = similar(Y.c.ρ),
            uforcing = zero(Y.c.ρ),
            vforcing = zero(Y.c.ρ),
            gw_ncval = Val(nc),
            # The source-property fields are allocated only when the source is configured (see `_beres_cache_fields`).
            gw_beres_source = gw.beres_source,
            _beres_cache_fields(Y, gw.beres_source, Val(nc))...,
        )
    elseif issphere(axes(Y.c))

        FT = Spaces.undertype(axes(Y.c))
        (; source_pressure, damp_pressure, Bw, Bn, Bt_0, Bt_n, Bt_s, Bt_eq) = gw
        (; ϕ0_s, ϕ0_n, dϕ_n, dϕ_s, dc, cmax, c0, nk, cw, cw_tropics, cn) = gw

        nc = Int(floor(FT(2 * cmax / dc + 1)))
        c = ntuple(n -> FT((n - 1) * dc - cmax), Val(nc))

        ᶜlocal_geometry = Fields.local_geometry_field(Fields.level(Y.c, 1))
        lat = ᶜlocal_geometry.coordinates.lat

        gw_Bn = @. ifelse(dϕ_s <= lat <= dϕ_n, FT(0), Bn)
        gw_cw = @. ifelse(dϕ_s <= lat <= dϕ_n, cw_tropics, cw)
        gw_flag = @. ifelse(dϕ_s <= lat <= dϕ_n, FT(0), FT(1))
        gw_Bw = ones(FT, axes(lat)) .* Bw
        gw_cn = ones(FT, axes(lat)) .* cn

        source_p_ρ_z_u_v_level =
            similar(Fields.level(Y.c.ρ, 1), Tuple{FT, FT, FT, FT, FT, FT})
        ᶜlevel = similar(Y.c.ρ, FT)
        for i in 1:Spaces.nlevels(axes(Y.c.ρ))
            fill!(Fields.level(ᶜlevel, i), i)
        end

        # This is GFDL source specs -> a smooth function
        # source_ampl = @. Bt_0 +
        #     Bt_n * FT(0.5) * (FT(1) + tanh((lat - ϕ0_n) / dϕ_n)) +
        #     Bt_s * FT(0.5) * (FT(1) + tanh((lat - ϕ0_s) / dϕ_s))

        # This latitude depend source follows MiMA specs
        source_ampl = @. ifelse(
            (lat > ϕ0_n) | (lat < ϕ0_s),
            Bt_0 +
            Bt_n * FT(0.5) * (FT(1) + tanh((lat - ϕ0_n) / dϕ_n)) +
            Bt_s * FT(0.5) * (FT(1) + tanh((lat - ϕ0_s) / dϕ_s)),
            ifelse(
                dϕ_s <= lat <= dϕ_n,
                Bt_eq,
                ifelse(
                    dϕ_n <= lat <= ϕ0_n,
                    Bt_0 + (Bt_eq - Bt_0) / (ϕ0_n - dϕ_n) * (ϕ0_n - lat),
                    Bt_0 + (Bt_eq - Bt_0) / (ϕ0_s - dϕ_s) * (ϕ0_s - lat),
                ),
            ),
        )

        return (;
            gw_source_pressure = source_pressure,
            gw_damp_pressure = damp_pressure,
            gw_source_ampl = source_ampl,
            gw_Bw = gw_Bw,
            gw_Bn = gw_Bn,
            gw_c = c,
            gw_cw = gw_cw,
            gw_cn = gw_cn,
            gw_dc = dc,
            gw_cmax = cmax,
            gw_c0 = c0,
            gw_flag = gw_flag,
            gw_nk = Int(nk),
            ᶜbuoyancy_frequency = similar(Y.c.ρ),
            ᶜuv_temp = similar(Y.c, Geometry.UVVector{FT}),
            gw_temp_surface = similar(Fields.level(Y.c.ρ, 1)),
            source_p_ρ_z_u_v_level,
            damp_level = similar(Fields.level(Y.c.ρ, 1)),
            ᶜlevel,
            u_waveforcing = similar(Y.c.ρ),
            v_waveforcing = similar(Y.c.ρ),
            uforcing = zero(Y.c.ρ),
            vforcing = zero(Y.c.ρ),
            gw_ncval = Val(nc),
            # The source-property fields are allocated only when the source is configured (see `_beres_cache_fields`).
            gw_beres_source = gw.beres_source,
            _beres_cache_fields(Y, gw.beres_source, Val(nc))...,
        )
    else
        error("Only sphere and columns are supported")
    end
end

"""
    non_orographic_gravity_wave_compute_tendency!(Y, p, ::Nothing)
    non_orographic_gravity_wave_compute_tendency!(Y, p, ::NonOrographicGravityWave)

Compute the non-orographic gravity wave forcing and store it in the NOGW cache.

Mutates `p.non_orographic_gravity_wave` (in particular `uforcing` and `vforcing`
[m/s²], which are zeroed and then re-accumulated) and returns `nothing`; `Y` is read
only. This is the expensive half of the parameterization and runs on the `dt_nogw`
callback `nogw_model_callback!`; the cheap half,
`non_orographic_gravity_wave_apply_tendency!`, applies the cached forcing every
integrator step. The `::Nothing` method is a no-op.

Steps, in order:

 1. Compute the buoyancy frequency `N = sqrt(g/T · (dT/dz + g/c_pm))` at cell centers from
    `p.precomputed.ᶜT` and the moist heat capacity, floored at `sqrt(2.5e-5)` s⁻¹. `dT/dz`
    is the difference of the interpolated face temperatures across each cell, one-sided
    in the bottom and top cells. The
    cache field `ᶜbuoyancy_frequency` holds `N`, not `N²`, once this step completes.
 2. Locate the AD99 launch level per column and record `(ρ, u, v)` there: on a column, the
    level closest to `source_height`; on a sphere, the highest level with
    `p > source_pressure`. On a sphere the damp level is the lowest level with
    `p < damp_pressure`, lowered if needed so the damping layer spans at least
    `NOGW_MIN_DAMP_CELLS` top cells; on a column it is the top `NOGW_MIN_DAMP_CELLS` cells.
 3. When a Beres source is configured, call `compute_beres_convective_heating!` and
    then locate the Beres launch level, the first level (going up) with `z ≥ gw_ztop`.
 4. Loop over the `gw_nk` horizontal wavenumber bands and, for each, run the shared
    upward-marching kernel `waveforcing_column_accumulate!` for the AD99 source and
    (when configured) again for the Beres source, accumulating both into `uforcing` and
    `vforcing` via `postprocess_and_accumulate!`.
 5. Sanitize the forcing: any `NaN`/`Inf` is replaced by zero and the rest is clamped to
    ±3e-3 m/s², the same limit the orographic scheme uses. The replaced values are counted
    by `warn_nonfinite_gravity_wave_forcing`, which warns the first time there are any.

Reads `p.precomputed.ᶜT`, `ᶜq_tot_nonneg`, `ᶜq_liq`, `ᶜq_ice`, and (on a sphere) `ᶜp`, and
clobbers `p.scratch.ᶜtemp_scalar` through `ᶜtemp_scalar_6` and `temp_field_level`.
Described in [alexander1999](@cite) and [beres2004](@cite); see the *Non-orographic
Gravity Waves* page for the derivations.
"""
non_orographic_gravity_wave_compute_tendency!(Y, p, ::Nothing) = nothing

function non_orographic_gravity_wave_compute_tendency!(
    Y,
    p,
    ::NonOrographicGravityWave,
)
    #unpack
    (; ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    (; params) = p
    (;
        ᶜbuoyancy_frequency,
        ᶜuv_temp,
        damp_level,
        u_waveforcing,
        v_waveforcing,
        uforcing,
        vforcing,
        ᶜlevel,
        gw_ncval,
    ) = p.non_orographic_gravity_wave

    ᶜρ = Y.c.ρ
    ᶜz = Fields.coordinate_field(Y.c).z
    FT = Spaces.undertype(axes(Y.c))
    # parameters
    thermo_params = CAP.thermodynamics_params(params)
    grav = CAP.grav(params)

    # compute the buoyancy frequency N = sqrt(|N²|) [1/s], or sqrt(2.5e-5) s⁻¹ where
    # N² < 2.5e-5 s⁻², to avoid small numbers
    # Inside the column, dT/dz is computed as the difference of the interpolated face
    # temperatures across each cell. However, because ᶠinterp copies the end-cell value
    # to the boundary face, the bottom and top cells would get only about half the
    # gradient there. Therefore, a one-sided difference is taken across the face to
    # their only neighbouring cell instead.
    n_levels = Spaces.nlevels(axes(Y.c))
    ᶜface_above = Operators.TopBiasedF2C()
    ᶜface_below = Operators.BottomBiasedF2C()
    ᶠdTdz = @. lazy(w_component(Geometry.WVector(ᶠgradᵥ(ᶜT))))
    ᶜdTdz = @. lazy(
        ifelse(
            ᶜlevel == 1,
            ᶜface_above(ᶠdTdz),
            ifelse(
                ᶜlevel == n_levels,
                ᶜface_below(ᶠdTdz),
                w_component(Geometry.WVector(ᶜgradᵥ(ᶠinterp(ᶜT)))),
            ),
        ),
    )
    ᶜcp_m = @. lazy(TD.cp_m(thermo_params, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice))
    ᶜN² = @. lazy((grav / ᶜT) * (ᶜdTdz + grav / ᶜcp_m))
    @. ᶜbuoyancy_frequency = ifelse(ᶜN² < FT(2.5e-5), FT(sqrt(2.5e-5)), sqrt(abs(ᶜN²)))

    # prepare physical uv input variables for gravity_wave_forcing()
    @. ᶜuv_temp = Geometry.UVVector(Y.c.uₕ)
    ᶜu = ᶜuv_temp.components.data.:1
    ᶜv = ᶜuv_temp.components.data.:2

    if iscolumn(axes(Y.c))
        # source level: the index of the level that is closest to the source height,
        # the upper of the two when two levels are equally close
        (; gw_source_height, source_ρ_z_u_v_level) =
            p.non_orographic_gravity_wave

        input = Base.Broadcast.broadcasted(tuple, ᶜρ, ᶜz, ᶜu, ᶜv, ᶜlevel)
        Operators.column_reduce!(
            source_ρ_z_u_v_level,
            input;
        ) do (ρ_prev, z_prev, u_prev, v_prev, level_prev), (ρ, z, u, v, level)
            if abs(z_prev - gw_source_height) >= abs(z - gw_source_height)
                return (ρ, z, u, v, level)
            else
                return (ρ_prev, z_prev, u_prev, v_prev, level_prev)
            end
        end

        ᶜρ_source = source_ρ_z_u_v_level.:1
        ᶜu_source = source_ρ_z_u_v_level.:3
        ᶜv_source = source_ρ_z_u_v_level.:4
        source_level = source_ρ_z_u_v_level.:5
        # get the ρ,u,v value on the source level

        # A column has no damp pressure; the damping layer is the top cells.
        fill!(damp_level, highest_damp_level(Spaces.nlevels(axes(ᶜz))))

    elseif issphere(axes(Y.c))
        (; ᶜp) = p.precomputed
        (; gw_source_pressure, gw_damp_pressure, source_p_ρ_z_u_v_level) =
            p.non_orographic_gravity_wave
        # source level: the index of the highest level whose pressure is higher than source pressure

        input = Base.Broadcast.broadcasted(tuple, ᶜp, ᶜρ, ᶜz, ᶜu, ᶜv, ᶜlevel)
        Operators.column_reduce!(
            source_p_ρ_z_u_v_level,
            input,
        ) do (p_prev, ρ_prev, z_prev, u_prev, v_prev, level_prev),
        (p, ρ, z, u, v, level)
            if (p - gw_source_pressure) <= 0
                return (p_prev, ρ_prev, z_prev, u_prev, v_prev, level_prev)
            else
                return (p, ρ, z, u, v, level)
            end
        end

        ᶜρ_source = source_p_ρ_z_u_v_level.:2
        ᶜu_source = source_p_ρ_z_u_v_level.:4
        ᶜv_source = source_p_ρ_z_u_v_level.:5
        source_level = source_p_ρ_z_u_v_level.:6
        # get the ρ,u,v value on the source level

        # damp level: the index of the lowest level whose pressure is lower than the damp pressure
        input = Base.Broadcast.broadcasted(tuple, ᶜlevel, ᶜp)
        Operators.column_reduce!(
            damp_level,
            input;
            transform = first,
        ) do (level_prev, p_prev), (level, p)
            if (p_prev - gw_damp_pressure) >= 0
                return (level, p)
            else
                return (level_prev, p_prev)
            end
        end
        # The damping layer spans at least `NOGW_MIN_DAMP_CELLS` top cells. The warning
        # check reduces over all columns and MPI ranks, so it runs at the first call only.
        damp_level_max = highest_damp_level(Spaces.nlevels(axes(ᶜz)))
        if !NOGW_DAMP_LAYER_CHECKED[]
            NOGW_DAMP_LAYER_CHECKED[] = true
            maximum(damp_level) > damp_level_max &&
                @warn "Fewer than $NOGW_MIN_DAMP_CELLS levels lie above nogw_damp_pressure ($gw_damp_pressure Pa) in some columns; the non-orographic gravity-wave momentum that escapes the model top is spread over the top $NOGW_MIN_DAMP_CELLS cells there instead."
        end
        @. damp_level = min(damp_level, damp_level_max)
    else
        error("Only sphere and columns are supported")
    end

    # Compute Beres convective heating if enabled.
    if !isnothing(p.non_orographic_gravity_wave.gw_beres_source)
        compute_beres_convective_heating!(Y, p, ᶜbuoyancy_frequency)

        (; gw_ztop, gw_z_surface, beres_source_ρ_z_u_v_level) =
            p.non_orographic_gravity_wave
        # gw_ztop is a height above the surface; the launch level search runs in
        # absolute height.
        ᶜz_top_absolute = @. lazy(gw_ztop + gw_z_surface)
        set_beres_launch_state!(
            beres_source_ρ_z_u_v_level, ᶜρ, ᶜz, ᶜu, ᶜv, ᶜlevel, ᶜz_top_absolute,
        )
    end

    uforcing .= 0
    vforcing .= 0

    non_orographic_gravity_wave_forcing(
        ᶜu,
        ᶜv,
        ᶜbuoyancy_frequency,
        ᶜρ,
        ᶜz,
        ᶜlevel,
        source_level,
        damp_level,
        ᶜρ_source,
        ᶜu_source,
        ᶜv_source,
        uforcing,
        vforcing,
        gw_ncval,
        u_waveforcing,
        v_waveforcing,
        p,
    )

    # Sanitize the forcing here, once per callback, rather than at every stage that applies
    # it (same limit as OGW: 3e-3 m/s²). The cells it zeroes are counted first, so that a
    # NaN or Inf is reported instead of silently removed.
    warn_nonfinite_gravity_wave_forcing(uforcing, vforcing)
    DataLayouts.foreach_point(uforcing, vforcing) do ᶜuforcing, ᶜvforcing
        @inline
        @. ᶜuforcing = limited_gravity_wave_forcing(ᶜuforcing)
        @. ᶜvforcing = limited_gravity_wave_forcing(ᶜvforcing)
    end
end

"""
    limited_gravity_wave_forcing(forcing)

Return the drag `forcing` [m/s²] clamped to ±3e-3 m/s², the same limit the orographic
scheme uses, or zero if it is `NaN` or infinite.
"""
limited_gravity_wave_forcing(forcing) = ifelse(
    isnan(forcing) | isinf(forcing),
    zero(forcing),
    max(-oftype(forcing, 3e-3), min(oftype(forcing, 3e-3), forcing)),
)

# Whether a non-finite non-orographic gravity-wave forcing has been reported, which
# happens once per session.
const NOGW_NONFINITE_WARNED = Ref(false)

"""
    warn_nonfinite_gravity_wave_forcing(uforcing, vforcing)

Return the number of `NaN` or infinite values in the drag fields `uforcing` and
`vforcing`, and log a warning the first time in a session that it is not zero.

Called once per callback by `non_orographic_gravity_wave_compute_tendency!` just before
`limited_gravity_wave_forcing` replaces those values by zero, which would otherwise hide
them.
"""
function warn_nonfinite_gravity_wave_forcing(uforcing, vforcing)
    n_nonfinite =
        count(!isfinite, parent(uforcing)) + count(!isfinite, parent(vforcing))
    if n_nonfinite > 0 && !NOGW_NONFINITE_WARNED[]
        NOGW_NONFINITE_WARNED[] = true
        @warn "The non-orographic gravity-wave drag has $n_nonfinite NaN or infinite values in this callback; they are set to zero. Later occurrences are not reported."
    end
    return n_nonfinite
end

"""
    set_beres_launch_state!(beres_source_ρ_z_u_v_level, ᶜρ, ᶜz, ᶜu, ᶜv, ᶜlevel, gw_ztop)

Set the per-column Beres launch state: the density, height, winds, and level index of the
first level (going up) whose height meets or exceeds the convective envelope top `gw_ztop`,
or of the top level when no level reaches `gw_ztop`.

Mutates `beres_source_ρ_z_u_v_level`, a level field of `(ρ, z, u, v, level)` tuples that
`non_orographic_gravity_wave_forcing` reads as the Beres launch level. Called from
`non_orographic_gravity_wave_compute_tendency!` after `compute_beres_convective_heating!`
has set `gw_ztop`; `ᶜz` and `gw_ztop` are both absolute heights there (`gw_ztop` plus the
surface height).
"""
function set_beres_launch_state!(
    beres_source_ρ_z_u_v_level, ᶜρ, ᶜz, ᶜu, ᶜv, ᶜlevel, gw_ztop,
)
    FT = eltype(ᶜρ)
    top_level = FT(Spaces.nlevels(axes(ᶜρ)))
    beres_input = Base.Broadcast.broadcasted(
        tuple, ᶜρ, ᶜz, ᶜu, ᶜv, ᶜlevel, gw_ztop,
    )
    Operators.column_reduce!(
        beres_source_ρ_z_u_v_level,
        beres_input;
        init = (FT(0), FT(-Inf), FT(0), FT(0), FT(0)),
    ) do (ρ_prev, z_prev, u_prev, v_prev, level_prev),
    (ρ, z, u, v, level, ztop)
        # Take the first level (going up) whose z meets/exceeds ztop. When no level
        # does (a ztop above the top cell centre, or a NaN ztop, for which every
        # comparison is false), take the top level, so that the launch density is
        # never the zero of `init`.
        if !(z_prev >= ztop) && ((z >= ztop) || (level == top_level))
            return (ρ, z, u, v, level)
        else
            return (ρ_prev, z_prev, u_prev, v_prev, level_prev)
        end
    end
end

"""
    _beres_heating_centroid(Q⁺_integral, zQ⁺_integral)

Return the centroid `z_c = ∫z·Q⁺dz / ∫Q⁺dz` [m] of the positive in-cloud heating `Q⁺`
from its first two moments, or zero when `∫Q⁺dz` is not positive or the quotient is not
finite (an infinite heating gives `Inf/Inf = NaN`).
"""
@inline function _beres_heating_centroid(Q⁺_integral, zQ⁺_integral)
    z_c = ifelse(
        Q⁺_integral > eps(typeof(Q⁺_integral)),
        zQ⁺_integral / Q⁺_integral,
        zero(Q⁺_integral),
    )
    return ifelse(isfinite(z_c), z_c, zero(z_c))
end

"""
    _beres_heating_depth(Q⁺_integral, z²Q⁺_integral, z_c)

Return the depth `h = σ/√((π²−8)/(4π²))` [m] of the half-sine with the same spread `σ` as
the positive in-cloud heating `Q⁺`, with `σ² = ∫z²Q⁺dz/∫Q⁺dz − z_c²` and `z_c` from
`_beres_heating_centroid`. Returns zero when `∫Q⁺dz` is not positive or the result is not
finite.
"""
@inline function _beres_heating_depth(Q⁺_integral, z²Q⁺_integral, z_c)
    FT = typeof(Q⁺_integral)
    h_coef = FT(1) / sqrt((FT(π)^2 - FT(8)) / (FT(4) * FT(π)^2))  # ≈ 4.595
    h = ifelse(
        Q⁺_integral > eps(FT),
        h_coef * sqrt(max(z²Q⁺_integral / Q⁺_integral - z_c^2, zero(FT))),
        zero(FT),
    )
    return ifelse(isfinite(h), h, zero(h))
end

"""
    compute_beres_convective_heating!(Y, p, ᶜN)

Extract the per-column convective source properties that the Beres spectrum needs.

Mutates the Beres fields of `p.non_orographic_gravity_wave`; the return value is unused.
The outputs are the half-sine amplitude `gw_Q0` [K/s], the heating depth `gw_h_heat` [m], the
envelope bounds `gw_zbot`/`gw_ztop` [m above the local surface; every Beres height here is
measured from the surface, the bottom face of the column], the mass-weighted envelope means `gw_u_heat`,
`gw_v_heat` [m/s] and `gw_N_source` [1/s], the deposition factor `gw_a_cover` [-], and the
activation flag `gw_beres_active` (1 or 0). It also fills the Beres-G / mechanical-source
launch state: the gathered in-cloud heating `gw_beres_gprofile` [K/s], the
obstacle-level updraft velocity `gw_w_obstacle` [m/s], and the precomputed per-column
launch spectra `gw_beres_B0_u`/`gw_beres_B0_v` that `waveforcing_column_accumulate!` reads.
Called from `non_orographic_gravity_wave_compute_tendency!`; requires a mass-flux EDMF with
at least one updraft.

Two heating profiles are built from the mass-flux divergence of dry-static-energy
anomalies, `s = c_pd·T + g·z` (the `g·z` terms cancel in the anomaly, leaving
`c_pd·(Tʲ − T̄)`). Levels whose total updraft area `Σⱼ max(ρaʲ, 0)/ρʲ` is below `a_min`
are left out of the heating fit: their in-cloud heating is zero, so they enter neither the
moment fit nor the gather, and they are left out of the activation integral. The envelope
means (`u_heat`, `v_heat`, `N_source`, `a_cover`) still average over the whole envelope.

  - `gw_Q_conv` — the grid-mean Yanai heat source Q₁, i.e. the draft anomalies weighted by
    area fraction `aʲ`. Used for the activation gate and nowhere else, because
    `Q0_threshold` is calibrated to grid-mean magnitudes.
  - `gw_Q_conv_ic` — the ρaʲ-weighted in-cloud (conditional-mean) heating, with no area
    factor. This is the local amplitude convention Beres' linear theory expects: it sets
    the envelope and `gw_Q0`, and the deposited flux is later diluted by `gw_a_cover` in
    `waveforcing_column_accumulate!`, giving a grid-mean flux ∝ ā·Q_ic². When
    `heating_latent` is set, `gw_Q_conv_ic` is instead the ρaʲ-weighted in-cloud mean of
    the transport-free latent heating `_beres_latent_heating`; `gw_Q_conv` is untouched.

The envelope `[z_bot, z_top]` is obtained by moment matching over `z ≥ z_bot_floor` (2 km
by default, which excludes the EDMF boundary-layer and dry-thermal signal): a half-sine of
depth `h` has variance `h²(π²−8)/(4π²)`, so `h = σ/√((π²−8)/(4π²))` with `σ` the spread of
`max(gw_Q_conv_ic, 0)` about its centroid `z_c`, and `z_bot, z_top = z_c ∓ h/2` (`z_top`
clamped to the column's top cell centre, `z_bot` to `[0, z_top]`). The amplitude follows
from inverting the half-sine mean, `Q0 = (π/2)·∫ max(Q_ic, 0) dz / h`, with the same
integral. Activation requires the grid-mean amplitude over the part of the envelope above
the floor, `(π/2)·∫ gw_Q_conv dz / (z_top − max(z_bot, z_bot_floor))`, to exceed
`Q0_threshold`, `h > h_heat_min`, and the coverage `gw_a_cover ≥ a_min`, which keeps a
column with near-empty updrafts inactive.

The launch spectrum is built here once per column, not inside the propagation kernel,
because it does not depend on the AD99 horizontal wavenumber. With
`shape_mode = half_sine` it uses the classical half-sine shape factor; with
`sine_transform` it uses the sine transform of `gw_beres_gprofile` sampled on the part of the
clamped envelope above the floor, `[max(z_bot, z_bot_floor), z_top]` (heating below
`z_bot_floor` is left out, as in the moment fit). That profile is the positive part
`max(Q_ic, 0)` of the in-cloud heating (cooling is left out, as in the moment fit and `Q0`),
interpolated linearly between model levels, optionally smoothed in the vertical
(`beres_smoothing_width`), tapered to zero at both window edges over `gw_beres_dz_level`,
the mean level thickness in the window, and rescaled last so that it integrates to the
column heating `∫Q⁺dz` over `z ≥ z_bot_floor` (the integral behind `Q0`). Beres-G
transforms this heating itself, with no `Q0` factor; the spectrum drops vertical
wavenumbers `m > π/gw_beres_dz_level`, which the profile does not resolve.
The steady c≈0 bin adds the thermal and, if enabled, the mechanical (obstacle) source,
the latter driven by `gw_w_obstacle` sampled at `z_bot + beres_obstacle_frac·h`.

# Arguments

  - `Y`: Prognostic state; read only.
  - `p`: Cache. Reads `p.precomputed` draft fields and the horizontal wind `ᶜuv_temp`,
    which `non_orographic_gravity_wave_compute_tendency!` sets before calling this
    function.
  - `ᶜN`: Buoyancy frequency at cell centers [1/s], passed explicitly rather than read from
    the cache so the units are unambiguous at the call site.

# Notes

When `detailed_diagnostics` is set, the function additionally fills `gw_halfsine`,
`gw_launch_flux`, `gw_c_centroid` (the last two reduced from the precomputed
`gw_beres_B0_u`), and `gw_beres_mech_flux`; otherwise it returns before that block.

Described in [beres2004](@cite); see the *Non-orographic Gravity Waves* page.
"""
function compute_beres_convective_heating!(Y, p, ᶜN)
    (; turbconv_model) = p.atmos
    # `n_updrafts ≥ 1` is guaranteed here
    n_updrafts = n_mass_flux_subdomains(turbconv_model)
    draft_indices = StaticOneTo(n_updrafts)

    (; ᶠu³ʲs, ᶜρʲs, ᶜuʲs, ᶜTʲs, ᶜT) = p.precomputed
    (; ᶠu³) = p.precomputed
    (;
        gw_Q0,
        gw_h_heat,
        gw_u_heat,
        gw_v_heat,
        gw_N_source,
        gw_beres_active,
        gw_beres_source,
        gw_zbot,
        gw_ztop,
        gw_Q_conv,
        gw_Q_conv_ic,
        gw_a_cover,
        gw_reduce_result,
        gw_c,
        gw_ncval,
        gw_halfsine,
        gw_launch_flux,
        gw_c_centroid,
        gw_beres_gprofile,
        gw_beres_dz_level,
        gw_z_surface,
        gw_beres_wind_range,
        gw_beres_u_profile,
        gw_beres_v_profile,
        gw_w_obstacle,
        gw_beres_B0_u,
        gw_beres_B0_v,
        gw_beres_mech_flux,
        ᶜuv_temp,
    ) = p.non_orographic_gravity_wave

    FT = Spaces.undertype(axes(Y.c))
    ϵ = eps(FT)
    _zero = FT(0)
    ᶜρ = Y.c.ρ
    # Every Beres height (the envelope, the floor, the gather window and the Beres-G
    # transform origin) is measured from the local surface, not from sea level.
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶜz_above_surface = @. lazy(ᶜz - gw_z_surface)
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J

    # Compute DSE-based mass-flux Q₁ (Yanai apparent heat source):
    #   ρ·Q₁ ≈ -∂/∂z [Mᶜ·(s_c − s̄)]  where s = cp_d·T + g·z
    # Note 1: Beres activation is always determined by Q₁, regardless of whether we use 0M or 1M microphysics. Only latent heating computation differs between microphysics schemes.
    # Note 2: MSE (h = cp·T + gz + Lv·q) is conserved under condensation, so its mass-flux divergence gives Q₁−Q₂, masking the latent-heating signal. DSE works because Tʲ is saturation-adjusted: the warming from condensation along the parcel trajectory is already encoded in Tʲ, so cp_d·(Tʲ − T̄) weighs the cumulative latent heat release.
    #
    # The g·z terms cancel in (sʲ − s̄), leaving cp_d·(Tʲ − T̄).
    cp_d = FT(CAP.cp_d(p.params))

    # Levels whose total updraft area Σⱼ max(ρaʲ, 0)/ρʲ is below a_min (the same minimum
    # as the whole-column coverage gate) carry no in-cloud heating: there the draft is
    # numerically present but physically empty, and its conditional mean is noise. This
    # physical threshold, not eps(FT) on ρa, also makes Float32 and Float64 agree. The
    # area is ρaʲ/ρʲ, the draft's own area fraction, as in the EDMF scheme and in the
    # coverage a_cover below, so the level test and the coverage gate measure the same
    # thing. The level must also hold some draft mass (ρa > 0), so that a_min = 0 does
    # not divide by zero. It compares with 0, not eps(FT): in Float32, ρa at a_min falls
    # below eps(FT) at 30-35 km, so an eps(FT) test would depend on height.
    a_min = FT(gw_beres_source.a_min)

    # Compute Q_conv (grid-mean) and Q_conv_ic (in-cloud) in one fused column loop.
    DataLayouts.foreach_column(
        gw_Q_conv, gw_Q_conv_ic, Y.c, ᶜJ, ᶠJ, ᶠu³, ᶠu³ʲs, ᶜρʲs, ᶜTʲs, ᶜT,
    ) do ᶜQ_conv, ᶜQ_conv_ic, Yc, ᶜJ, ᶠJ, ᶠu³, ᶠu³ʲs, ᶜρʲs, ᶜTʲs, ᶜT
        @inline
        ᶜQ_conv_sum = @. zero(Yc.ρ)
        ᶜQ_conv_ic_sum = @. zero(Yc.ρ)
        ᶜρa_sum = @. zero(Yc.ρ)
        ᶜarea_sum = @. zero(Yc.ρ)
        for j in draft_indices
            ᶜρaʲ = Yc.sgsʲs.:($j).ρa
            ᶜρʲ = ᶜρʲs.:($j)
            ᶜTʲ = ᶜTʲs.:($j)
            ᶠu³ʲ = ᶠu³ʲs.:($j)

            # In-cloud DSE anomaly: cp_d·(Tʲ − T̄), no area factor
            # ᶜρʲs is a physical air density (~1 kg/m³), so this guard almost never
            # fires -- except at the domain top, where in Float32 ρʲ = p/(R·T) can
            # approach eps(FT). Kept for consistency with the `/ ᶜρʲ` divisions below
            # (which genuinely need it there) so all draft terms zero out together.
            ᶜa_scalar_ic = @. ifelse(ᶜρʲ > ϵ, cp_d * (ᶜTʲ - ᶜT), _zero)

            # DSE anomaly × area fraction: cp_d·(Tʲ − T̄) · (ρaʲ/ρʲ)
            ᶜa_scalar = @. ifelse(ᶜρʲ > ϵ, ᶜa_scalar_ic * (ᶜρaʲ / ᶜρʲ), _zero)

            # Face mass flux ρʲ·(u³ʲ − u³) of the velocity anomaly, shared by the centered
            # divergences -∂(ρʲ·u³_diff·χ)/∂z of both heatings (`vertical_transport` with
            # `Val(:none)`)
            ᶠρu³_diff = @. ᶠinterp(ᶜρʲ * ᶜJ) / ᶠJ * (ᶠu³ʲ - ᶠu³)

            # Convert from W/m³ to heating rate K/s (grid-mean Q₁)
            @. ᶜQ_conv_sum +=
                -(ᶜadvdivᵥ(ᶠρu³_diff * ᶠinterp(ᶜa_scalar))) / (Yc.ρ * cp_d)

            # In-cloud heating of draft j normalized by the draft density:
            #   Q_icʲ = −(1/(ρʲ)) ∂z[ρʲ·(wʲ−w̄)·(Tʲ−T̄)]
            # Accumulated ρaʲ-weighted so that for M>1 drafts the result is the
            # conditional mean over the total draft area,
            #   gw_Q_conv_ic = Σⱼ (ρaʲ · Q_icʲ) / Σⱼ ρaʲ
            @. ᶜQ_conv_ic_sum += ifelse(
                ᶜρʲ > ϵ,
                max(ᶜρaʲ, _zero) *
                -(ᶜadvdivᵥ(ᶠρu³_diff * ᶠinterp(ᶜa_scalar_ic))) / (ᶜρʲ * cp_d),
                _zero,
            )
            @. ᶜρa_sum += max(ᶜρaʲ, _zero)
            @. ᶜarea_sum += ifelse(ᶜρʲ > ϵ, max(ᶜρaʲ, _zero) / ᶜρʲ, _zero)
        end
        @. ᶜQ_conv = ᶜQ_conv_sum
        # Finalize the ρa-weighted in-cloud mean over drafts, on levels whose total
        # updraft area is at least a_min.
        @. ᶜQ_conv_ic = ifelse(
            (ᶜarea_sum >= a_min) & (ᶜρa_sum > _zero),
            ᶜQ_conv_ic_sum / ᶜρa_sum,
            _zero,
        )
    end

    # When enabled, replace the in-cloud DSE flux-divergence Q₁
    # just computed in gw_Q_conv_ic with Q_lat = (1/cp⁽ʲ⁾) Σ_p L_p R_p⁽ʲ⁾, the
    # ρaʲ-weighted in-cloud mean of the per-draft latent heating.
    # The grid-mean gw_Q_conv (Q₁) above is left untouched, as it still drives activation criteria and the z_bot threshold, which are calibrated to grid-mean Q₁ magnitudes.
    # The heating_latent flag requires 1M + PrognosticEDMFX at construction.
    if gw_beres_source.heating_latent
        # Run CM BMT on both 1M parameter sets and sum: the bulk set carries
        # every explicit process; the implicit set carries only
        # `cloud_liquid_formation` and `cloud_ice_formation` (nulled when
        # `implicit_microphysics: false`). The sum is the full latent heating
        # regardless of which stage integrates cond/subl, without double-
        # counting (the two sets are process-disjoint by construction).
        mp_bulk = CAP.microphysics_1m_params(p.params)
        mp_imp = CAP.microphysics_1m_implicit_params(p.params)
        thp = CAP.thermodynamics_params(p.params)
        ᶜq_tot_nonnegʲs = p.precomputed.ᶜq_tot_nonnegʲs
        gw_Q_conv_ic .= FT(0)
        for j in 1:n_updrafts
            ᶜρaʲ = Y.c.sgsʲs.:($j).ρa
            ᶜρʲ = ᶜρʲs.:($j)
            ᶜTʲ = ᶜTʲs.:($j)
            ᶜuʲ = ᶜuʲs.:($j)
            ᶜq_totʲ = ᶜq_tot_nonnegʲs.:($j)
            ᶜq_lclʲ = Y.c.sgsʲs.:($j).q_lcl
            ᶜq_iclʲ = Y.c.sgsʲs.:($j).q_icl
            ᶜq_raiʲ = Y.c.sgsʲs.:($j).q_rai
            ᶜq_snoʲ = Y.c.sgsʲs.:($j).q_sno
            # physical draft vertical velocity [m/s] for the velocity-dependent autoconversion
            ᶜwʲ_air = @. lazy(w_component(WVec(ᶜuʲ)))
            @. gw_Q_conv_ic += ifelse(
                ᶜρʲ > eps(FT),
                max(ᶜρaʲ, FT(0)) * _beres_latent_heating(
                    mp_bulk,
                    mp_imp,
                    thp,
                    ᶜρʲ,
                    ᶜTʲ,
                    ᶜwʲ_air,
                    ᶜq_totʲ,
                    ᶜq_lclʲ,
                    ᶜq_iclʲ,
                    ᶜq_raiʲ,
                    ᶜq_snoʲ,
                ),
                FT(0),
            )
        end
        # The draft weights of the in-cloud heating are the non-negative draft
        # densities max(ρaʲ, 0) [kg/m³].
        positive_draft_mass = sgsʲ -> max(sgsʲ.ρa, zero(sgsʲ.ρa))
        ᶜdraft_mass = @. lazy(draft_sum(positive_draft_mass, Y.c.sgsʲs))
        # Total draft area Σⱼ max(ρaʲ, 0)/ρʲ, as in the DSE path above.
        positive_draft_area =
            (sgsʲ, ρʲ) -> ifelse(ρʲ > ϵ, max(sgsʲ.ρa, zero(ρʲ)) / ρʲ, zero(ρʲ))
        ᶜdraft_area =
            @. lazy(mapreduce_with_init(positive_draft_area, +, Y.c.sgsʲs, ᶜρʲs))
        @. gw_Q_conv_ic = ifelse(
            (ᶜdraft_area >= a_min) & (ᶜdraft_mass > FT(0)),
            gw_Q_conv_ic / ᶜdraft_mass,
            FT(0),
        )
    end

    # `ᶜuv_temp` holds UVVector(uₕ), which `non_orographic_gravity_wave_compute_tendency!`
    # sets before this call.
    ᶜu = ᶜuv_temp.components.data.:1
    ᶜv = ᶜuv_temp.components.data.:2

    # Pass 1: set the convective envelope [z_bot, z_top] and depth h.
    # We construct the envelope by moment matching: fit a half-sine to the in-cloud
    # heating Q_conv_ic by its centroid z_c and spread σ. A half-sine of depth h has
    # variance h²·(π²−8)/(4π²), so h = σ/√(variance), z_bot = z_c − h/2,
    # z_top = z_c + h/2, and the amplitude Q0 = (π/2)·∫max(Q_ic,0)dz / h. This sits the
    # envelope on the actual heating.
    result_field = gw_reduce_result
    ᶜΔz_m = Fields.Δz_field(axes(Y.c))
    input1 =
        Base.Broadcast.broadcasted(tuple, ᶜz_above_surface, gw_Q_conv_ic, ᶜΔz_m)
    # `result_field` (gw_reduce_result) is a single 7-slot tuple field reused for
    # both passes. Pass 1 fills slots 1-3 with the heating moments
    # (∫Q⁺dz, ∫z·Q⁺dz, ∫z²·Q⁺dz) and slot 4 with the height of the column's top cell
    # centre, and carries slots 5-7 through untouched, i.e., the `_5..._7`
    # placeholders in the reducer below. Pass 2 then reuses all 7 slots as
    # (Q_integral, u_sum, v_sum, N_sum, mass_sum, Qic_integral, a_sum); see the
    # `reduce_init2` accumulator there.
    reduce_init = (FT(0), FT(0), FT(0), FT(0), FT(0), FT(0), FT(0))
    let _z_floor = gw_beres_source.z_bot_floor, _zero = FT(0)
        # Accumulate (∫Q⁺dz, ∫z·Q⁺dz, ∫z²·Q⁺dz) over z ≥ z_bot_floor into slots 1-3,
        # and the highest cell-centre height into slot 4.
        Operators.column_reduce!(
            result_field,
            input1;
            init = reduce_init,
        ) do (I0p, I1p, I2p, z_top_prev, _5, _6, _7), (z, Q_ic, dz)
            qpdz = ifelse(z >= _z_floor, max(Q_ic, _zero) * dz, _zero)
            return (
                I0p + qpdz,
                I1p + z * qpdz,
                I2p + z * z * qpdz,
                max(z_top_prev, z),
                _5,
                _6,
                _7,
            )
        end
    end
    half_π = FT(π) / FT(2)
    # The envelope and amplitude of every column in one loop.
    DataLayouts.foreach_column(
        gw_zbot, gw_ztop, gw_h_heat, gw_Q0, result_field,
    ) do gw_zbot, gw_ztop, gw_h_heat, gw_Q0, moments
        @inline
        (Q⁺_integral, zQ⁺_integral, z²Q⁺_integral, z_column_top) =
            (moments.:1, moments.:2, moments.:3, moments.:4)
        ᶜz_c = @. _beres_heating_centroid(Q⁺_integral, zQ⁺_integral)
        ᶜh = @. _beres_heating_depth(Q⁺_integral, z²Q⁺_integral, ᶜz_c)
        # z_top = z_c + h/2 and z_bot = z_c − h/2, with z_top clamped to the column's
        # own top cell centre (which differs between columns on a terrain-following
        # grid), so the launch level search finds a level, and z_bot to [0, z_top].
        @. gw_ztop = min(ᶜz_c + ᶜh / 2, z_column_top)
        @. gw_zbot = min(max(ᶜz_c - ᶜh / 2, _zero), gw_ztop)
        # Matched-half-sine amplitude Q0 = (π/2)·∫max(Q_ic,0)dz / h
        ᶜQ0 = @. ifelse(
            (ᶜh > ϵ) & (Q⁺_integral > _zero),
            max(half_π * Q⁺_integral / ᶜh, _zero),
            _zero,
        )
        @. gw_Q0 = ifelse(isnan(ᶜQ0) | isinf(ᶜQ0), _zero, ᶜQ0)
        @. gw_h_heat = ifelse(isnan(ᶜh) | isinf(ᶜh), _zero, ᶜh)
    end

    # Pass 2: within [z_bot, z_top], compute:
    #   - Q₀ integrals: Σ(Q · Δz) for the grid-mean (gating) and in-cloud
    #     (spectrum amplitude) heatings
    #   - Mass-weighted mean wind (u, v), buoyancy frequency (N), and
    #     updraft area fraction (a_cover, the Beres deposition factor)
    # All quantities drawn from the same convective envelope.

    ᶜΔz = Fields.Δz_field(axes(Y.c))
    # 3D envelope mask (2D z_bot/z_top broadcast over column) and total updraft area
    # fraction, evaluated in the column reduction
    ᶜin_env = @. lazy(
        ifelse(
            (ᶜz_above_surface >= gw_zbot) & (ᶜz_above_surface <= gw_ztop),
            FT(1),
            FT(0),
        ),
    )
    # Total updraft area of each level, Σⱼ max(ρaʲ, 0)/ρʲ without the drafts whose
    # density is at most ϵ, as for the in-cloud heating above. A draft with a negative
    # ρaʲ counts as empty, so it cannot lower the coverage of the others. Levels below
    # a_min are left out of the heating fit only (the moments, the gather, and the
    # activation integral); the envelope means u_heat, v_heat, N_source and the
    # coverage a_cover average over the whole envelope, so a column whose updraft is
    # nearly empty over most of its envelope has a small a_cover and fails the
    # coverage gate.
    nonnegative_draft_area =
        (sgsʲ, ρʲ) -> ifelse(ρʲ > ϵ, max(sgsʲ.ρa, zero(ρʲ)) / ρʲ, zero(ρʲ))
    ᶜa_level =
        @. lazy(mapreduce_with_init(nonnegative_draft_area, +, Y.c.sgsʲs, ᶜρʲs))
    # The activation integral counts only the heating at or above the floor, as the
    # moment fit and the gather do.
    z_floor = FT(gw_beres_source.z_bot_floor)
    ᶜabove_floor = @. lazy(ifelse(ᶜz_above_surface >= z_floor, FT(1), FT(0)))
    input2 = Base.Broadcast.broadcasted(
        tuple,
        gw_Q_conv,
        ᶜu,
        ᶜv,
        ᶜN,
        ᶜρ,
        ᶜΔz,
        ᶜin_env,
        gw_Q_conv_ic,
        ᶜabove_floor,
        ᶜa_level,
    )
    # Accumulator: (Q_integral, u_sum, v_sum, N_sum, mass_sum, Qic_integral, a_sum).
    # Same 7-slot `result_field` as Pass 1, which used only slots 1-3 for the
    # heating moments. Slot 6 (Qic_integral) is accumulated but currently unread:
    # the amplitude Q0 comes from the Pass 1 moments instead.
    _half = FT(0.5)
    reduce_init2 = (_zero, _zero, _zero, _zero, _zero, _zero, _zero)
    Operators.column_reduce!(
        result_field,
        input2;
        init = reduce_init2,
    ) do (
        Q_int_prev,
        u_sum_prev,
        v_sum_prev,
        N_sum_prev,
        m_sum_prev,
        Qic_int_prev,
        a_sum_prev,
    ),
    (Q, u, v, N, ρ, dz, env, Q_ic, above_floor, a_level)
        active = env > _half
        ρdz = ρ * dz
        Q_int = ifelse(
            active & (above_floor > _half) & (a_level >= a_min),
            Q_int_prev + Q * dz,
            Q_int_prev,
        )
        u_sum = ifelse(active, u_sum_prev + u * ρdz, u_sum_prev)
        v_sum = ifelse(active, v_sum_prev + v * ρdz, v_sum_prev)
        N_sum = ifelse(active, N_sum_prev + N * ρdz, N_sum_prev)
        m_sum = ifelse(active, m_sum_prev + ρdz, m_sum_prev)
        Qic_int = ifelse(active, Qic_int_prev + Q_ic * dz, Qic_int_prev)
        a_sum = ifelse(active, a_sum_prev + a_level * ρdz, a_sum_prev)
        return (Q_int, u_sum, v_sum, N_sum, m_sum, Qic_int, a_sum)
    end

    # Unpack Pass 2 results, and set the beres_active flag: GRID-MEAN amplitude above
    # threshold AND heating depth above minimum. The activation trigger always uses
    # grid-mean Q₁ (slot 1) to ensure the same activation criteria between 0M and 1M
    # microphysics. The grid-mean amplitude is the half-sine peak of the heating over the
    # part of the envelope above the floor, (π/2)·∫Q dz / (z_top − max(z_bot, floor)).
    Q0_threshold = gw_beres_source.Q0_threshold
    h_heat_min = gw_beres_source.h_heat_min
    _one = FT(1)
    N_source_default = FT(0.01)
    DataLayouts.foreach_column(
        gw_u_heat, gw_v_heat, gw_N_source, gw_a_cover, gw_beres_active, gw_h_heat,
        gw_zbot, gw_ztop, result_field,
    ) do gw_u_heat, gw_v_heat, gw_N_source, gw_a_cover, gw_beres_active, gw_h_heat,
    gw_zbot, gw_ztop, sums
        @inline
        (Q_integral, u_sum, v_sum, N_sum, mass_sum, a_sum) =
            (sums.:1, sums.:2, sums.:3, sums.:4, sums.:5, sums.:7)
        @. gw_u_heat = ifelse(mass_sum > ϵ, u_sum / mass_sum, _zero)
        @. gw_v_heat = ifelse(mass_sum > ϵ, v_sum / mass_sum, _zero)
        @. gw_N_source =
            ifelse(mass_sum > ϵ, N_sum / mass_sum, N_source_default)
        # Coverage: mass-weighted envelope mean of the updraft area fraction,
        # clamped to [0, 1]. Used as the Beres deposition (intermittency) factor.
        @. gw_a_cover = ifelse(
            mass_sum > ϵ,
            min(max(a_sum / mass_sum, _zero), _one),
            _zero,
        )
        ᶜfloor_width =
            @. gw_ztop - _beres_gather_bottom(gw_zbot, gw_ztop, z_floor)
        @. gw_beres_active = ifelse(
            (max(half_π * Q_integral / max(ᶜfloor_width, ϵ), _zero) > Q0_threshold) &
            (gw_h_heat > h_heat_min) &
            (gw_a_cover >= a_min),
            _one,
            _zero,
        )
    end

    # --- Beres-G (Ext 1) + mechanical (Ext 3) per-column launch state ---
    # Computed every call. The launch spectrum B0 is independent of the AD99
    # horizontal wavenumber, so it is precomputed here (once per column, with full
    # column access for the sine transform and obstacle velocity) and the
    # propagation kernel reads it rather than rebuilding it inside the `ink` loop.
    # half-sine + no-mechanical reproduces the classical kernel spectrum exactly.

    # (1) Gather the in-cloud heating shape onto a fixed uniform grid spanning the part
    # of the convective envelope above the floor, [max(z_bot, z_bot_floor), z_top],
    # linearly interpolated between the two model levels that bracket each point and
    # positive-part masked. z is the height above the local surface (rigid-ground
    # origin for sin(m z)). Heating below z_bot_floor is
    # excluded, mirroring the moment-fit mask in Pass 1: z_bot = z_c − h/2 may legally
    # dip below the floor, and without the mask the sine transform would ingest the
    # PBL/surface-flux heating the fit deliberately excluded. The window starts at the
    # floor rather than at z_bot, so the masked heating does not drop to zero inside
    # it, a step whose transform falls off only like 1/m. Baseline (half_sine) is
    # unaffected — only the V_G path reads the gathered profile.
    ng_val = Val(N_BERES_PROFILE)
    ᶜgather_bottom =
        @. lazy(_beres_gather_bottom(gw_zbot, gw_ztop, z_floor))
    gather_input = Base.Broadcast.broadcasted(
        tuple, ᶜz_above_surface, gw_Q_conv_ic, ᶜgather_bottom, gw_ztop,
    )
    # The do-closure is a GPU kernel argument, so it may capture only isbits
    # values — never the type FT itself (same rule the Pass-1 moment reduce
    # follows). All FT-derived constants are let-bound as plain numbers.
    let _z_floor = z_floor, _ng_val = ng_val
        Operators.column_reduce!(
            gw_beres_gprofile,
            gather_input;
            init = _beres_gather_init(FT, ng_val),
            transform = first,
        ) do state, level_input
            _beres_gather_step(state, level_input, _z_floor, _ng_val)
        end
    end
    # The mean model level thickness over the window, from the cells that overlap it.
    # The gathered profile interpolates the heating between levels, so it holds no
    # detail finer than the level spacing; this mean sets the edge taper below and the
    # shortest vertical wavelength the Beres-G spectrum uses.
    window_input =
        Base.Broadcast.broadcasted(
            tuple, ᶜz_above_surface, ᶜΔz, ᶜgather_bottom, gw_ztop,
        )
    Operators.column_reduce!(
        gw_beres_dz_level,
        window_input;
        init = (FT(0), FT(0)),
        transform = sum_count -> sum_count[1] / max(sum_count[2], one(sum_count[2])),
    ) do (dz_sum, n_cells), (z, dz, z_start, z_top)
        overlaps = (z + dz / 2 > z_start) & (z - dz / 2 < z_top)
        return (
            ifelse(overlaps, dz_sum + dz, dz_sum),
            ifelse(overlaps, n_cells + one(n_cells), n_cells),
        )
    end
    # Then, in this order: the optional vertical smoothing, the edge taper, and the
    # rescaling that gives the profile the column heating ∫Q⁺dz over z ≥ z_bot_floor,
    # the integral that also sets Q0 = (π/2)·∫Q⁺dz/h, so that Beres-C and Beres-G carry
    # the same column heating. The scaling comes last, so no heating is lost.
    smoothing_width = FT(gw_beres_source.beres_smoothing_width)
    two_over_π = FT(2) / FT(π)
    @. gw_beres_gprofile = _beres_scale_profile(
        _beres_taper_edges(
            _beres_smooth_profile(
                gw_beres_gprofile,
                gw_ztop - ᶜgather_bottom,
                smoothing_width,
            ),
            gw_ztop - ᶜgather_bottom,
            gw_beres_dz_level,
        ),
        gw_ztop - ᶜgather_bottom,
        two_over_π * gw_Q0 * gw_h_heat,
    )

    # (1b) Winds for the launch filter (`beres.launch_filter`): the range of u and v
    # over the heating layer, up to and including the launch level (the first level at
    # or above z_top), and u, v at the launch level; with the height-resolved filter
    # also u and v on the gather grid. The layer starts where the heating each shape
    # uses starts: at z_bot for the half-sine, at the gather window bottom
    # max(z_bot, z_bot_floor) for Beres-G, which has no heating below the floor.
    shape_general = gw_beres_source.beres_shape_general
    ᶜlayer_bottom = @. lazy(
        ifelse(
            shape_general,
            _beres_gather_bottom(gw_zbot, gw_ztop, z_floor),
            gw_zbot,
        ),
    )
    wind_input = Base.Broadcast.broadcasted(
        tuple, ᶜz_above_surface, ᶜu, ᶜv, ᶜlayer_bottom, gw_ztop,
    )
    Operators.column_reduce!(
        gw_beres_wind_range,
        wind_input;
        init = (FT(Inf), FT(-Inf), FT(Inf), FT(-Inf), FT(0), FT(0), FT(0)),
        transform = _beres_wind_range_result,
    ) do state, level_input
        _beres_wind_range_step(state, level_input)
    end
    if gw_beres_source.launch_filter == BERES_LAUNCH_FILTER_HEIGHT_RESOLVED
        let _ng_val = ng_val
            for (wind_profile, ᶜwind) in
                ((gw_beres_u_profile, ᶜu), (gw_beres_v_profile, ᶜv))
                wind_gather_input = Base.Broadcast.broadcasted(
                    tuple, ᶜz_above_surface, ᶜwind, ᶜgather_bottom, gw_ztop,
                )
                Operators.column_reduce!(
                    wind_profile,
                    wind_gather_input;
                    init = _beres_gather_init(FT, ng_val),
                    transform = first,
                ) do state, level_input
                    _beres_gather_wind_step(state, level_input, _ng_val)
                end
            end
        end
    end

    # (2) Obstacle-level updraft velocity w_b for the mechanical source (Ext 3):
    # the per-draft physical vertical velocity where the draft penetrates the
    # stable shear layer above the convective top, sampled at obstacle height =
    # z_bot + obstacle_frac·h (default frac = 1 ⇒ z_top). Single updraft (draft 1);
    # M>1 aggregation deferred (the manuscript configurations use M=1).
    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶜw_up = p.scratch.ᶜtemp_scalar_3
    @. ᶜw_up = get_physical_w(ᶜuʲs.:(1), ᶜlg)
    obstacle_frac = FT(gw_beres_source.beres_obstacle_frac)
    wobs_input = Base.Broadcast.broadcasted(
        tuple, ᶜz_above_surface, ᶜw_up, gw_zbot, gw_h_heat,
    )
    Operators.column_reduce!(
        gw_w_obstacle,
        wobs_input;
        init = (FT(0), FT(Inf)),
        transform = first,
    ) do (w_best, d_best), (z, w_up, z_bot, h)
        z_obs = z_bot + obstacle_frac * h
        d = abs(z - z_obs)
        return d < d_best ? (w_up, d) : (w_best, d_best)
    end

    # (3) Precompute the per-column launch spectra (u and v azimuths); see
    # `_beres_precompute_launch_spectra!`.
    _beres_precompute_launch_spectra!(p.non_orographic_gravity_wave)

    # --- Diagnostics for source-shape & launched-spectrum ---
    gw_beres_source.detailed_diagnostics || return

    # Build the launched half-sine Q0·sin(π(z−z_bot)/h) natively, IN the column
    # where (Q0, z_bot, h) are mutually consistent, so the offline comparison to
    # Q_conv_ic is a single linear remap rather than a nonlinear reconstruction
    # from three independently-remapped 2D fields.
    @. gw_halfsine = ifelse(
        (ᶜz_above_surface >= gw_zbot) & (ᶜz_above_surface <= gw_zbot + gw_h_heat),
        gw_beres_active *
        gw_Q0 *
        sin(FT(π) * (ᶜz_above_surface - gw_zbot) / max(gw_h_heat, eps(FT))),
        FT(0),
    )

    # Launched-spectrum summaries reduced from the SAME precomputed launch spectrum
    # gw_beres_B0_u the forcing kernel deposits (so they are Beres-G / mechanical
    # aware, not a recomputed half-sine):
    # - total launched flux magnitude Σ|B0| (× a_cover deposition factor),
    # - the flux-weighted phase-speed centroid ⟨c⟩ = Σ c|B0| / Σ|B0|,
    # both in the zonal direction.
    let cgrid = gw_c
        cen(b0) = _beres_b0_centroid(b0, cgrid)
        @. gw_launch_flux = ifelse(
            gw_beres_active > FT(0.5),
            gw_a_cover * _beres_b0_fluxmag(gw_beres_B0_u),
            FT(0),
        )
        @. gw_c_centroid = ifelse(
            gw_beres_active > FT(0.5),
            cen(gw_beres_B0_u),
            FT(0),
        )
    end

    # Mechanical (Ext 3) c≈0 launched flux β_mech, zonal direction — the
    # band-resolved Ext-3 export. β_mech is deposited only in the c≈0 (low-`c`)
    # bin and its sign carries the E/W deceleration direction (opposes U).
    let beres = gw_beres_source
        mech(active, u, N, w) = _beres_mech_flux_value(active, u, N, w, beres)
        @. gw_beres_mech_flux =
            mech(gw_beres_active, gw_u_heat, gw_N_source, gw_w_obstacle)
    end
end

"""
    _beres_gather_bottom(z_bot, z_top, z_floor)

Return the bottom of the Beres-G gather window [m], `max(z_bot, z_floor)` capped at `z_top`.

The gather masks the heating below `z_floor` (`z_bot_floor`), as the moment fit does, so a
window starting at a `z_bot` below the floor would hold a step from zero to the heating
at the floor. The window, and the sine transform of the gathered profile, start at the
floor instead.
"""
@inline _beres_gather_bottom(z_bot, z_top, z_floor) =
    min(max(z_bot, z_floor), z_top)

"""
    _beres_taper_edges(profile, span, width)

Return the gathered heating `profile` (an `NTuple{ng}` on a window of width `span` [m])
multiplied by a raised-cosine taper that falls from 1 to 0 over the distance `width` [m],
at most half the window, from each window edge, so the profile is zero at both edges.

The window edges cut the heating where the moment fit puts them, generally where it is
not zero. A jump at an edge makes the sine transform fall off only like `1/m`, which puts
flux into the slow bins next to `u_heat` that grows like `1/|ĉ|`. The taper removes the
jump. `width` is the mean model level thickness over the window, `gw_beres_dz_level`,
the same scale below which the Beres-G spectrum drops vertical wavenumbers, so the taper
smooths the edges at the shortest wavelength the spectrum keeps. The heating the taper
removes is not lost: the rescaling that follows (`_beres_scale_profile`) gives the profile
the column heating again.
"""
@inline function _beres_taper_edges(profile, span, width)
    t = _beres_unwrap_gprofile(profile)
    FT = typeof(span)
    ng = length(t)
    dz_g = span / FT(ng - 1)
    w = max(min(width, span / 2), eps(FT))
    return ntuple(Val(ng)) do i
        d = FT(min(i - 1, ng - i)) * dz_g   # distance to the nearer window edge
        factor = ifelse(d < w, (1 - cos(FT(π) * d / w)) / 2, FT(1))
        t[i] * factor
    end
end

"""
    _beres_smooth_profile(profile, span, width)

Return the gathered heating `profile` (an `NTuple{ng}` on a window of width `span` [m])
smoothed in the vertical by a Gaussian of standard deviation `width` [m], normalized at
each point over the window, or the profile unchanged when `width ≤ 0` (the default,
`BeresSourceParams.beres_smoothing_width = 0`).
"""
@inline function _beres_smooth_profile(profile, span, width)
    t = _beres_unwrap_gprofile(profile)
    FT = typeof(span)
    width > zero(width) || return t
    ng = length(t)
    dz_g = span / FT(ng - 1)
    # Literal tuple indices only: a runtime index into an NTuple does not compile on
    # the GPU.
    return ntuple(Val(ng)) do i
        weights = ntuple(j -> exp(-((FT(i - j) * dz_g)^2) / (2 * width^2)), Val(ng))
        sum(ntuple(j -> weights[j] * t[j], Val(ng))) / sum(weights)
    end
end

"""
    _beres_scale_profile(profile, span, column_heating)

Return the gathered heating `profile` (an `NTuple{ng}` on a window of width `span` [m])
rescaled so that its integral over the window (trapezoid rule on the gather points) is
`column_heating` [K·m/s], the column integral `∫Q⁺dz` over `z ≥ z_bot_floor` that also
sets `Q0 = (π/2)·∫Q⁺dz/h`. Then Beres-C and Beres-G carry the same column heating, and
none of it is lost to the window, the taper or the smoothing. A window integral that is
not positive (an inactive or degenerate column) gives a zero profile.
"""
@inline function _beres_scale_profile(profile, span, column_heating)
    t = _beres_unwrap_gprofile(profile)
    FT = typeof(span)
    ng = length(t)
    window_integral = span / FT(ng - 1) * (sum(t) - (t[1] + t[ng]) / 2)
    ratio = ifelse(
        window_integral > zero(FT),
        column_heating / window_integral,
        zero(FT),
    )
    # A separate name, not a reassignment: a variable reassigned and captured by the
    # closure below would be boxed, which the GPU cannot compile.
    scale = ifelse(isfinite(ratio), ratio, zero(FT))
    return ntuple(i -> t[i] * scale, Val(ng))
end

"""
    _beres_gather_init(FT, Val(ng))

Return the starting state of the Beres-G gather, `(values, z_prev, Q_prev)`: `ng` zero
samples and a zero heating at `z = 0`, so the segment below the lowest level runs from the
ground with zero heating.
"""
_beres_gather_init(::Type{FT}, ng_val::Val) where {FT} =
    (ntuple(_ -> FT(0), ng_val), FT(0), FT(0))

"""
    _beres_gather_step(state, (z, Q_ic, z_bot, z_top), z_floor, Val(ng))

One level of the Beres-G gather, run upward through a column by `column_reduce!`.

`state = (values, z_prev, Q_prev)` holds the `ng` samples gathered so far and the height
[m] and masked heating [K/s] of the level below. The heating at this level is masked to its
positive part `max(Q_ic, 0)`, and to zero below `z_floor` (the `z_bot_floor` the moment fit
also uses). `z_bot` here is the window bottom from `_beres_gather_bottom`. Each sample point
`z_i = z_bot + (i-1)·(z_top - z_bot)/(ng-1)` in the segment
`z_prev < z_i ≤ z` takes the linear interpolation of the masked heating between the two
levels, so the gathered profile is the piecewise-linear interpolant of the masked heating
on the model levels. Returns the updated state; `first` of the final state is the profile.
"""
@inline function _beres_gather_step(
    (values, z_prev, Q_prev),
    (z, Q_ic, z_bot, z_top),
    z_floor,
    ng_val::Val{ng},
) where {ng}
    FT = typeof(z)
    dz_g = max(z_top - z_bot, eps(FT)) / FT(ng - 1)
    Q_pos = ifelse(z >= z_floor, max(Q_ic, zero(FT)), zero(FT))
    new_values = ntuple(ng_val) do i
        z_i = z_bot + (i - 1) * dz_g
        in_segment = (z_prev < z_i) & (z_i <= z)
        # Successive levels are strictly increasing, so z > z_prev.
        weight = (z_i - z_prev) / (z - z_prev)
        ifelse(in_segment, Q_prev + weight * (Q_pos - Q_prev), values[i])
    end
    return (new_values, z, Q_pos)
end

"""
    _beres_gather_wind_step(state, (z, wind, z_bot, z_top), Val(ng))

One level of gathering a wind component onto the Beres-G gather grid, as
`_beres_gather_step` gathers the heating but with no masking: each point
`z_i = z_bot + (i-1)·(z_top - z_bot)/(ng-1)` in the segment `z_prev < z_i ≤ z` takes the
linear interpolation of the wind between the two levels. The height-resolved launch
filter reads the result.
"""
@inline function _beres_gather_wind_step(
    (values, z_prev, wind_prev),
    (z, wind, z_bot, z_top),
    ng_val::Val{ng},
) where {ng}
    FT = typeof(z)
    dz_g = max(z_top - z_bot, eps(FT)) / FT(ng - 1)
    new_values = ntuple(ng_val) do i
        z_i = z_bot + (i - 1) * dz_g
        in_segment = (z_prev < z_i) & (z_i <= z)
        weight = (z_i - z_prev) / (z - z_prev)
        ifelse(in_segment, wind_prev + weight * (wind - wind_prev), values[i])
    end
    return (new_values, z, wind)
end

"""
    _beres_wind_range_step(state, (z, u, v, z_bot, z_top))

One level of the reduction that sets the launch-filter winds of a column, run upward.

`state = (u_min, u_max, v_min, v_max, u_L, v_L, done)`. Levels from `z_bot` (the bottom of
the heating the shape uses: the envelope bottom for the half-sine, the gather window bottom
for Beres-G) up to and including the first level at or above `z_top` (the launch level)
update the ranges; the launch level also sets `u_L`, `v_L` and `done`. Heights are above
the surface.
"""
@inline function _beres_wind_range_step(
    (u_min, u_max, v_min, v_max, u_L, v_L, done),
    (z, u, v, z_bot, z_top),
)
    FT = typeof(z)
    in_layer = (z >= z_bot) & (done < FT(0.5))
    at_launch = in_layer & (z >= z_top)
    return (
        ifelse(in_layer, min(u_min, u), u_min),
        ifelse(in_layer, max(u_max, u), u_max),
        ifelse(in_layer, min(v_min, v), v_min),
        ifelse(in_layer, max(v_max, v), v_max),
        ifelse(at_launch, u, u_L),
        ifelse(at_launch, v, v_L),
        ifelse(at_launch, one(done), done),
    )
end
_beres_wind_range_result(state) =
    (state[1], state[2], state[3], state[4], state[5], state[6])

"""
    _beres_precompute_launch_spectra!(nogw)

Fill the per-column Beres launch spectra `gw_beres_B0_u` and `gw_beres_B0_v` of the NOGW
cache `nogw` from the source fields already in it (`gw_beres_active`, `gw_u_heat`,
`gw_v_heat`, `gw_Q0`, `gw_h_heat`, `gw_N_source`, `gw_zbot`, `gw_ztop`,
`gw_beres_gprofile`, `gw_beres_dz_level`, `gw_w_obstacle`) and the source parameters
`gw_beres_source`.

Called at the end of `compute_beres_convective_heating!`. It is a separate function so that
tests and offline drivers that set the source fields by hand, or swap `gw_beres_source`,
can rebuild the spectrum the propagation kernel reads.
"""
function _beres_precompute_launch_spectra!(nogw)
    (;
        gw_beres_source,
        gw_c,
        gw_ncval,
        gw_beres_active,
        gw_u_heat,
        gw_v_heat,
        gw_Q0,
        gw_h_heat,
        gw_N_source,
        gw_zbot,
        gw_ztop,
        gw_beres_gprofile,
        gw_beres_dz_level,
        gw_beres_wind_range,
        gw_beres_u_profile,
        gw_beres_v_profile,
        gw_w_obstacle,
        gw_beres_B0_u,
        gw_beres_B0_v,
    ) = nogw
    # Zeros for non-convecting columns; half-sine or Beres-G transient bins per the shape
    # mode, plus the steady c≈0 bin (β_therm + β_mech). The phase-speed grid
    # `cgrid`, `ncv`, and `beres` are CAPTURED in the closure (plain tuples/Val
    # would be iterated if passed as broadcast args); only fields are broadcast.
    # The Beres-G transform runs over the window the gather sampled,
    # [_beres_gather_bottom(z_bot, z_top, z_bot_floor), z_top], with
    # `span = z_top - bottom` (NOT gw_h_heat, the unclamped moment depth, which
    # differs from the gather grid when the envelope reaches below the floor or is
    # capped at the column top — that mismatch would evaluate V_G at the wrong
    # altitudes). gw_h_heat stays the half-sine depth (V_hs / Q0 / activation).
    # GPU note: the shape mode is decided HERE, on the CPU, not inside the
    # kernel, and is threaded into the launch-spectrum builder as a COMPILE-TIME
    # `Val` (`Val(true)` for the transform branch, `Val(false)` for half-sine),
    # NOT read from the runtime `beres.beres_shape_general` field inside the
    # kernel. That keeps `use_VG` a compile-time constant, so each broadcast
    # compiles exactly ONE shape path: half-sine passes `nothing` and strips the
    # sine-transform code (`_beres_gs`), while the transform branch keeps a
    # concrete `NTuple` profile so `_beres_gs` stays statically dispatched. A
    # runtime-Bool switch instead forces BOTH branches into the kernel and widens
    # the profile to `Union{Nothing,NTuple}` -> dynamic dispatch / GC allocation
    # (the InvalidIRError the transform path used to fail with).
    let beres = gw_beres_source, cgrid = gw_c, ncv = gw_ncval,
        z_floor = gw_beres_source.z_bot_floor

        # One builder per azimuth (`Val(1)` for u, `Val(2)` for v), which selects
        # that azimuth's winds for the launch filter.
        if beres.beres_shape_general
            b0g(azimuth) = function (
                active, u, Q0, h, N, zbot, ztop, gprof, dz_level, w, wind_range,
                wind_profile,
            )
                z_gather = _beres_gather_bottom(zbot, ztop, z_floor)
                taper_points =
                    dz_level / max((ztop - z_gather) / (N_BERES_PROFILE - 1), eps(dz_level))
                launch_filter = _beres_launch_filter(
                    beres, wind_range, wind_profile, azimuth, taper_points,
                )
                return BeresB0(
                    _beres_launch_spectrum_or_zero(
                        active, cgrid, u, Q0, h, N, z_gather, ztop - z_gather,
                        gprof, w, beres, ncv, Val(true); dz_level, launch_filter,
                    ),
                )
            end
            b0g_u = b0g(Val(1))
            b0g_v = b0g(Val(2))
            @. gw_beres_B0_u = b0g_u(
                gw_beres_active, gw_u_heat, gw_Q0, gw_h_heat, gw_N_source,
                gw_zbot, gw_ztop, gw_beres_gprofile, gw_beres_dz_level,
                gw_w_obstacle, gw_beres_wind_range, gw_beres_u_profile,
            )
            @. gw_beres_B0_v = b0g_v(
                gw_beres_active, gw_v_heat, gw_Q0, gw_h_heat, gw_N_source,
                gw_zbot, gw_ztop, gw_beres_gprofile, gw_beres_dz_level,
                gw_w_obstacle, gw_beres_wind_range, gw_beres_v_profile,
            )
        else
            b0hs(azimuth) =
                (active, u, Q0, h, N, zbot, ztop, w, wind_range) -> BeresB0(
                    _beres_launch_spectrum_or_zero(
                        active, cgrid, u, Q0, h, N, zbot, ztop - zbot, nothing, w,
                        beres, ncv, Val(false);
                        launch_filter = _beres_launch_filter(
                            beres, wind_range, nothing, azimuth,
                        ),
                    ),
                )
            b0hs_u = b0hs(Val(1))
            b0hs_v = b0hs(Val(2))
            @. gw_beres_B0_u = b0hs_u(
                gw_beres_active, gw_u_heat, gw_Q0, gw_h_heat, gw_N_source,
                gw_zbot, gw_ztop, gw_w_obstacle, gw_beres_wind_range,
            )
            @. gw_beres_B0_v = b0hs_v(
                gw_beres_active, gw_v_heat, gw_Q0, gw_h_heat, gw_N_source,
                gw_zbot, gw_ztop, gw_w_obstacle, gw_beres_wind_range,
            )
        end
    end
    return nothing
end

"""
    non_orographic_gravity_wave_apply_tendency!(Yₜ, Y, p, t, ::Nothing)
    non_orographic_gravity_wave_apply_tendency!(Yₜ, Y, p, t, ::NonOrographicGravityWave)

Add the cached non-orographic gravity wave drag to the horizontal momentum tendency.

Mutates `Yₜ.c.uₕ`. The `::Nothing` method is a no-op.

This runs every integrator step, while the forcing itself is refreshed only on the
`dt_nogw` callback by `non_orographic_gravity_wave_compute_tendency!`, which also clamps
it to ±3e-3 m/s² and replaces any `NaN`/`Inf` by zero; between callbacks the same forcing
is reapplied.
"""
non_orographic_gravity_wave_apply_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

function non_orographic_gravity_wave_apply_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::NonOrographicGravityWave,
)
    (; uforcing, vforcing) = p.non_orographic_gravity_wave
    @. Yₜ.c.uₕ += Geometry.Covariant12Vector(Geometry.UVVector(uforcing, vforcing))
end

"""
    non_orographic_gravity_wave_forcing(ᶜu, ᶜv, ᶜbf, ᶜρ, ᶜz, ᶜlevel, source_level,
                                        damp_level, ᶜρ_source, ᶜu_source, ᶜv_source,
                                        uforcing, vforcing, gw_ncval, u_waveforcing,
                                        v_waveforcing, p)

Accumulate the gravity wave drag from every source and wavenumber band into the forcings.

Mutates `uforcing` and `vforcing`, in m/s², adding to whatever they already hold, so callers
zero them first) along with the working fields `u_waveforcing`/`v_waveforcing`, and returns
`nothing`. Called from `non_orographic_gravity_wave_compute_tendency!`.

The kernel needs level `k+1` values alongside level `k`, which a ClimaCore column operator
cannot index directly, so `ᶜρ`, `ᶜu`, `ᶜv`, `ᶜbf`, and `ᶜz` are first shifted down one
level by `field_shiftlevel_up!` into scratch fields. The extrapolated values used at the
top are geometric for ρ (`ρ_end²/ρ_end₋₁`), linear for `u`, `v`, and `z`, and a copy of the
top level for the buoyancy frequency.

Per horizontal wavenumber band `ink` (`gw_nk` bands, one by default) the AD99 background
source is marched upward for the zonal and meridional directions and deposited by
`postprocess_and_accumulate!`; when a Beres source is configured, the same pair of calls is
repeated for it. Both sources use the same kernel
`waveforcing_column_accumulate!` and differ only in the launch spectrum, the launch level,
and the intermittency factor.

# Arguments

  - `ᶜu`, `ᶜv`: Physical horizontal wind components at cell centers [m/s].
  - `ᶜbf`: Buoyancy frequency `N` at cell centers [1/s].
  - `ᶜρ`, `ᶜz`, `ᶜlevel`: Density [kg/m³], height [m], and 1-based level index [-].
  - `source_level`, `damp_level`: AD99 launch and damping level indices [-].
  - `ᶜρ_source`, `ᶜu_source`, `ᶜv_source`: AD99 launch-level density and winds.
  - `gw_ncval`: `Val(nc)` with `nc` the number of phase-speed bins.
  - `p`: Cache; the NOGW cache supplies the source parameters and
    `p.scratch.ᶜtemp_scalar` through `ᶜtemp_scalar_6` and `temp_field_level` are clobbered.
"""
function non_orographic_gravity_wave_forcing(
    ᶜu,
    ᶜv,
    ᶜbf,
    ᶜρ,
    ᶜz,
    ᶜlevel,
    source_level,
    damp_level,
    ᶜρ_source,
    ᶜu_source,
    ᶜv_source,
    uforcing,
    vforcing,
    gw_ncval::Val{nc},
    u_waveforcing,
    v_waveforcing,
    p,
) where {nc}
    # unpack parameters
    (;
        gw_source_ampl,
        gw_Bw,
        gw_Bn,
        gw_c,
        gw_cw,
        gw_cn,
        gw_flag,
        gw_c0,
        gw_nk,
        gw_temp_surface,
        gw_beres_source,
    ) = p.non_orographic_gravity_wave

    # Temporary scratch fields for shifting levels up
    ᶜρ_p1 = p.scratch.ᶜtemp_scalar
    ᶜz_p1 = p.scratch.ᶜtemp_scalar_2
    ᶜu_p1 = p.scratch.ᶜtemp_scalar_3
    ᶜv_p1 = p.scratch.ᶜtemp_scalar_4
    ᶜbf_p1 = p.scratch.ᶜtemp_scalar_5

    FT = eltype(ᶜρ) # Define the floating point type

    # Using interpolate operator, generate the field of ρ,u,v,z with on level shifted up.
    # Fields.level with differing level indices do not share the same space and cannot be
    # combined in broadcast operations.
    ρ_endlevel = Fields.level(ᶜρ, Spaces.nlevels(axes(ᶜρ)))
    ρ_endlevel_m1 = Fields.level(ᶜρ, Spaces.nlevels(axes(ᶜρ)) - 1)
    Fields.field_values(gw_temp_surface) .=
        Fields.field_values(ρ_endlevel) .* Fields.field_values(ρ_endlevel) ./
        Fields.field_values(ρ_endlevel_m1)
    field_shiftlevel_up!(ᶜρ, ᶜρ_p1, gw_temp_surface)

    u_endlevel = Fields.level(ᶜu, Spaces.nlevels(axes(ᶜu)))
    u_endlevel_m1 = Fields.level(ᶜu, Spaces.nlevels(axes(ᶜu)) - 1)
    Fields.field_values(gw_temp_surface) .=
        FT(2) .* Fields.field_values(u_endlevel) .-
        Fields.field_values(u_endlevel_m1)
    field_shiftlevel_up!(ᶜu, ᶜu_p1, gw_temp_surface)

    v_endlevel = Fields.level(ᶜv, Spaces.nlevels(axes(ᶜv)))
    v_endlevel_m1 = Fields.level(ᶜv, Spaces.nlevels(axes(ᶜv)) - 1)
    Fields.field_values(gw_temp_surface) .=
        FT(2) .* Fields.field_values(v_endlevel) .-
        Fields.field_values(v_endlevel_m1)
    field_shiftlevel_up!(ᶜv, ᶜv_p1, gw_temp_surface)

    field_shiftlevel_up!(
        ᶜbf,
        ᶜbf_p1,
        Fields.level(ᶜbf, Spaces.nlevels(axes(ᶜbf))),
    )

    z_endlevel = Fields.level(ᶜz, Spaces.nlevels(axes(ᶜz)))
    z_endlevel_m1 = Fields.level(ᶜz, Spaces.nlevels(axes(ᶜz)) - 1)
    Fields.field_values(gw_temp_surface) .=
        FT(2) .* Fields.field_values(z_endlevel) .-
        Fields.field_values(z_endlevel_m1)
    field_shiftlevel_up!(ᶜz, ᶜz_p1, gw_temp_surface)

    # The kernel turns the flux each layer loses into a drag by dividing by an estimate of
    # the layer's mass, `√(ρ_k·ρ_{k+1})·Δz`. Here that estimate is replaced by the mean mass
    # of the two cells on either side of the layer, `m = ρ·J/ΔA` (`ΔA`: the area at the
    # bottom of the column). Half of each cell lies inside the layer, so the drag given to
    # the two cells then adds up exactly to the flux the layer lost, on any grid.
    # `u_waveforcing` is not written until the kernel runs, so it serves as scratch here.
    ᶜlayer_ratio = p.scratch.ᶜtemp_scalar_6
    ᶜρJ_p1 = u_waveforcing
    ᶜJ = Fields.local_geometry_field(axes(ᶜρ)).J
    @. ᶜlayer_ratio = ᶜρ * ᶜJ
    field_shiftlevel_up!(ᶜlayer_ratio, ᶜρJ_p1, FT(0))
    @. ᶜlayer_ratio =
        sqrt(ᶜρ * ᶜρ_p1) * (ᶜz_p1 - ᶜz) / ((ᶜlayer_ratio + ᶜρJ_p1) / 2)
    ᶠspace = Spaces.face_space(axes(ᶜρ))
    Fields.field_values(gw_temp_surface) .=
        Fields.field_values(
            Fields.level(Fields.local_geometry_field(ᶠspace).J, half),
        ) ./ Fields.field_values(Fields.level(Fields.Δz_field(ᶠspace), half))
    DataLayouts.foreach_column(ᶜlayer_ratio, gw_temp_surface) do ᶜratio, ΔA_bottom
        @inline
        @. ᶜratio *= ΔA_bottom
    end

    mask_u = StaticBitVector{nc}(_ -> true)
    mask_v = StaticBitVector{nc}(_ -> true)
    # We use StaticBitVector here because the unrolled_reduce function in Julia can
    # cause memory allocation issues when the mask has more than 32 elements.
    # StaticBitVector stores 8 boolean values in a UInt8, allowing efficient storage
    # for up to 256 gravity wave break data.
    level_end = Spaces.nlevels(axes(ᶜρ))

    # Per-column rescaling for the escaped-flux redistribution.
    gw_deposit_scale = p.scratch.temp_field_level
    ᶜmasked_ρ = @. lazy(ifelse(ᶜlevel >= damp_level, ᶜρ, FT(0)))
    Operators.column_integral_definite!(gw_deposit_scale, ᶜmasked_ρ)
    Fields.field_values(gw_deposit_scale) .=
        Fields.field_values(ρ_endlevel) .*
        sqrt.(
            Fields.field_values(ρ_endlevel) ./
            Fields.field_values(ρ_endlevel_m1),
        ) .*
        (
            Fields.field_values(z_endlevel) .-
            Fields.field_values(z_endlevel_m1)
        ) ./ Fields.field_values(gw_deposit_scale)

    # Collect the per-column fields each source mode needs into a broadcasted
    # tuple. Both modes share the same 7 fields in slots 1–7; slots 8–16 hold the
    # 9 fields specific to that mode (AD99 source vs. Beres convective source).
    #
    # Shared slots 1–7: u/v_kp1, bf_kp1, ρ_k, ρ_kp1, z_kp1, z_k, level.
    # AD99 slots 8–16: u/v_source, ρ_source, source_level, Bw, Bn, cw, cn, flag,
    #   source_ampl.
    # Beres slots 8–16: beres_active, Q0, h, u/v_heat, N_source, beres_source_level,
    #   beres_ρ_source, beres_u/v_source, a_cover.
    input_u_ad99 = @. lazy(
        tuple(
            ᶜu_p1,
            ᶜbf_p1,
            ᶜρ,
            ᶜρ_p1,
            ᶜz_p1,
            ᶜz,
            ᶜlevel,
            ᶜu_source,
            ᶜρ_source,
            source_level,
            gw_Bw,
            gw_Bn,
            gw_cw,
            gw_cn,
            gw_flag,
            gw_source_ampl,
        ),
    )
    input_v_ad99 = @. lazy(
        tuple(
            ᶜv_p1,
            ᶜbf_p1,
            ᶜρ,
            ᶜρ_p1,
            ᶜz_p1,
            ᶜz,
            ᶜlevel,
            ᶜv_source,
            ᶜρ_source,
            source_level,
            gw_Bw,
            gw_Bn,
            gw_cw,
            gw_cn,
            gw_flag,
            gw_source_ampl,
        ),
    )
    # Slot 8 carries the precomputed per-column launch spectrum B0 (NTuple{nc});
    # the kernel reads it instead of rebuilding the Beres spectrum at level 1.
    # Slots 9–12 (Q0, h, u/v_heat, N) are no longer read by the Beres kernel but
    # are kept so the Beres tuple stays 16-wide and the AD99/Beres shared index
    # logic typechecks for both specializations.
    #
    # Beres per-column inputs are built only when the convective source is
    # configured (the fields do not exist in the cache otherwise).
    input_u_beres = nothing
    input_v_beres = nothing
    if !isnothing(gw_beres_source)
        (;
            gw_beres_active,
            gw_Q0,
            gw_h_heat,
            gw_u_heat,
            gw_v_heat,
            gw_N_source,
            gw_a_cover,
            beres_source_ρ_z_u_v_level,
            gw_beres_B0_u,
            gw_beres_B0_v,
        ) = p.non_orographic_gravity_wave
        # Beres launch-level state (per-column).
        beres_ρ_source = beres_source_ρ_z_u_v_level.:1
        beres_u_source = beres_source_ρ_z_u_v_level.:3
        beres_v_source = beres_source_ρ_z_u_v_level.:4
        beres_source_level = beres_source_ρ_z_u_v_level.:5
        input_u_beres = @. lazy(
            tuple(
                ᶜu_p1,
                ᶜbf_p1,
                ᶜρ,
                ᶜρ_p1,
                ᶜz_p1,
                ᶜz,
                ᶜlevel,
                gw_beres_B0_u,
                gw_Q0,
                gw_h_heat,
                gw_u_heat,
                gw_N_source,
                beres_source_level,
                beres_ρ_source,
                beres_u_source,
                gw_a_cover,
            ),
        )
        input_v_beres = @. lazy(
            tuple(
                ᶜv_p1,
                ᶜbf_p1,
                ᶜρ,
                ᶜρ_p1,
                ᶜz_p1,
                ᶜz,
                ᶜlevel,
                gw_beres_B0_v,
                gw_Q0,
                gw_h_heat,
                gw_v_heat,
                gw_N_source,
                beres_source_level,
                beres_ρ_source,
                beres_v_source,
                gw_a_cover,
            ),
        )
    end

    # loop over all wave lengths
    for ink in 1:gw_nk
        # --- AD99 background source (always active) ---
        waveforcing_column_accumulate!(
            u_waveforcing, mask_u, input_u_ad99,
            gw_c, gw_c0, gw_nk, ink, level_end,
            gw_ncval, nothing, Val(:ad99),
        )
        waveforcing_column_accumulate!(
            v_waveforcing, mask_v, input_v_ad99,
            gw_c, gw_c0, gw_nk, ink, level_end,
            gw_ncval, nothing, Val(:ad99),
        )
        postprocess_and_accumulate!(
            uforcing, vforcing, u_waveforcing, v_waveforcing,
            damp_level, ᶜlevel, level_end, gw_deposit_scale,
            ᶜlayer_ratio,
        )

        # --- Beres convective source (when configured) ---
        if !isnothing(gw_beres_source)
            waveforcing_column_accumulate!(
                u_waveforcing, mask_u, input_u_beres,
                gw_c, gw_c0, gw_nk, ink, level_end,
                gw_ncval, gw_beres_source, Val(:beres),
            )
            waveforcing_column_accumulate!(
                v_waveforcing, mask_v, input_v_beres,
                gw_c, gw_c0, gw_nk, ink, level_end,
                gw_ncval, gw_beres_source, Val(:beres),
            )
            postprocess_and_accumulate!(
                uforcing, vforcing, u_waveforcing, v_waveforcing,
                damp_level, ᶜlevel, level_end, gw_deposit_scale,
                ᶜlayer_ratio,
            )
        end
    end
    return nothing
end

"""
    postprocess_and_accumulate!(uforcing, vforcing, u_waveforcing, v_waveforcing,
                                damp_level, ᶜlevel, level_end, deposit_scale,
                                ᶜlayer_ratio)

Redistribute the escaped flux of one source and add its drag to the forcing fields.

Mutates `uforcing`/`vforcing`, which are incremented, in one column loop. Called once per
wavenumber band and source from `non_orographic_gravity_wave_forcing`, after the pair of
`waveforcing_column_accumulate!` calls that set `u_waveforcing`/`v_waveforcing`.

The top level of the half-level drag left by the upward march holds the momentum flux that
escaped through the model top. The rest of the drag `X` is multiplied by `ᶜlayer_ratio`
(set in `non_orographic_gravity_wave_forcing`) and moved to full levels by averaging
adjacent half-levels, `½(X[k-1/2] + X[k+1/2])`, with zero below the bottom level and at
the top level. The escaped flux is spread over the levels at and above `damp_level` by
`gw_deposit`, scaled by the per-column `deposit_scale`. The drag from waves breaking
inside the column and the redistributed escaped flux each conserve momentum exactly; the
*Non-orographic Gravity Waves* docs page explains why. On a sphere
`damp_level` is a field of level indices; single-column tests pass one level index.
"""
function postprocess_and_accumulate!(
    uforcing, vforcing, u_waveforcing, v_waveforcing,
    damp_level, ᶜlevel, level_end, deposit_scale, ᶜlayer_ratio,
)
    FT = eltype(uforcing)
    # The drag at the half level below each full level, zero below the bottom level
    ᶠbelow = Operators.BottomBiasedC2F(; bottom = Operators.SetValue(FT(0)))
    ᶜbelow = Operators.BottomBiasedF2C()
    (_zero, _half, top_level) = (FT(0), FT(0.5), FT(level_end))
    # A level index is captured by the loop, while a field of them is a loop argument.
    (damp_level_fields, damp_level_index) =
        damp_level isa Fields.Field ? ((damp_level,), nothing) : ((), damp_level)
    DataLayouts.foreach_column(
        uforcing, vforcing, u_waveforcing, v_waveforcing, ᶜlevel, deposit_scale,
        ᶜlayer_ratio, damp_level_fields,
    ) do ᶜuforcing, ᶜvforcing, ᶜu_waveforcing, ᶜv_waveforcing, ᶜlevel,
    ᶜdeposit_scale, ᶜlayer_ratio, damp_level_fields
        @inline
        ᶜdamp_level =
            isnothing(damp_level_index) ? damp_level_fields[1] : damp_level_index
        unrolled_foreach((
            (ᶜuforcing, ᶜu_waveforcing),
            (ᶜvforcing, ᶜv_waveforcing),
        )) do (ᶜforcing, ᶜwaveforcing)
            @inline
            # The momentum flux that escaped through the model top
            ᶜescaped_flux = Fields.level(ᶜwaveforcing, level_end)
            # The half-level drag, divided by the mean mass of the two cells it spans
            ᶜlayer_drag = @. lazy(ᶜwaveforcing * ᶜlayer_ratio)
            @. ᶜforcing += gw_deposit(
                ᶜescaped_flux,
                _half * (
                    ifelse(ᶜlevel == top_level, _zero, ᶜlayer_drag) +
                    ᶜbelow(ᶠbelow(ᶜlayer_drag))
                ),
                ᶜdamp_level,
                ᶜlevel,
                ᶜdeposit_scale,
            )
        end
    end
end

"""
    compute_ad99_spectrum(c, u_source, Bw, Bn, cw, cn, c0, flag, gw_ncval)

Return the AD99 launch spectrum for one column, as an `NTuple{nc}` of `B₀(c)` [m²/s²].

A thin named wrapper around the AD99 method of `wave_source`, so that
`waveforcing_column_accumulate!` selects between the two source spectra by name rather than
by argument-count dispatch.
"""
compute_ad99_spectrum(c, u_source, Bw, Bn, cw, cn, c0, flag, gw_ncval) =
    wave_source(c, u_source, Bw, Bn, cw, cn, c0, flag, gw_ncval)

"""
    compute_beres_spectrum(beres::BeresSourceParams, beres_active_val, c, u_heat_val,
                           Q0_val, h_val, N_val, gw_ncval)
    compute_beres_spectrum(::Nothing, beres_active_val, c, u_heat_val, Q0_val, h_val,
                           N_val, gw_ncval)

Return the Beres launch spectrum for one column, as an `NTuple{nc}` of the momentum flux
`B₀(c)` carried by each phase-speed bin [Pa].

Delegates to the `BeresSourceParams` method of `wave_source` when the column is convecting
(`beres_active_val > 0.5`), and returns all zeros otherwise.

The `::Nothing` method covers the AD99-only configuration.

The propagation kernel no longer calls this function: it reads the per-column spectrum
precomputed in `compute_beres_convective_heating!`. This half-sine-only path is kept for
tests and diagnostics.
"""
function compute_beres_spectrum(
    beres::BeresSourceParams,
    beres_active_val,
    c,
    u_heat_val,
    Q0_val,
    h_val,
    N_val,
    gw_ncval::Val{nc},
) where {nc}
    FT1 = typeof(Q0_val)
    if beres_active_val > FT1(0.5)
        wave_source(c, u_heat_val, Q0_val, h_val, N_val, beres, gw_ncval)
    else
        # No Beres contribution for non-convecting columns
        ntuple(_ -> FT1(0), Val(nc))
    end
end

function compute_beres_spectrum(
    ::Nothing, beres_active_val, c, u_heat_val, Q0_val, h_val, N_val,
    gw_ncval::Val{nc},
) where {nc}
    ntuple(_ -> typeof(Q0_val)(0), Val(nc))
end

"""
    _beres_launch_summary(beres, active, c, u_heat, Q0, h, N, gw_ncval) -> (s0, s1)

Summarize the launched Beres spectrum of one column into two scalars.

Returns the total launched flux magnitude `s0 = Σₙ |B₀(cₙ)|` and the phase-speed first
moment `s1 = Σₙ cₙ·|B₀(cₙ)|`, whose ratio `s1/s0` is the flux-weighted spectral centroid
`⟨c⟩` [m/s]. The launched Beres spectrum does not depend on the horizontal wavenumber band
(the `ink` loop of the kernel), only on phase speed, so this single evaluation captures the
whole launched spectrum.

Called from `compute_beres_convective_heating!` to fill the `gw_launch_flux` and
`gw_c_centroid` diagnostics.
"""
@inline function _beres_launch_summary(
    beres,
    active,
    c,
    u_heat,
    Q0,
    h,
    N,
    gw_ncval::Val{nc},
) where {nc}
    FT1 = typeof(Q0)
    B0 = compute_beres_spectrum(beres, active, c, u_heat, Q0, h, N, gw_ncval)
    s0 = FT1(0)
    s1 = FT1(0)
    for n in 1:nc
        b = abs(B0[n])
        s0 += b
        s1 += c[n] * b
    end
    return (s0, s1)
end

"""
    waveforcing_column_accumulate!(waveforcing, mask, input, c, c0, nk, ink, level_end,
                                   gw_ncval, beres_source, source_mode::Val{MODE})

March one source spectrum up each column and write the half-level drag it deposits.

Mutates `waveforcing` [m/s²]; the return value is unused. This is the kernel shared by both
sources; `MODE` is `:ad99` or `:beres` and selects the launch spectrum, the launch level,
and the intermittency factor at compile time.

At the bottom level the launch spectrum `B₀(c)` is set once (AD99: evaluated by
`compute_ad99_spectrum`; Beres: read from the per-column spectrum precomputed in
`compute_beres_convective_heating!`) and then carried upward by the column
accumulator together with a `StaticBitVector` mask of the phase-speed bins still
propagating. At each level at or above `source_level`, every unmasked bin is tested in
order and removed from the mask when it fails:

 1. Critical level, `ĉ = c[n] - u = 0`.
 2. Total internal reflection, `|ĉ|·k ≥ ω_r` with `ω_r² = N²k²/(k² + 1/(4Hb²))` and `Hb`
    the local density scale height.
 3. Breaking, `B₀[n]/ĉ³ ≥ ½(ρ_{k+1}/ρ_source)·k/N` for AD99, whose `B₀` is a flux per
    unit mass [m²/s²], and `B₀[n]/ĉ³ ≥ ½ρ_{k+1}·k/N` for Beres, whose `B₀` is a stress
    [Pa]; or a sign change of `ĉ` relative to its launch-level value (critical-level
    absorption).

A bin removed by test 1 or 3 adds its `B₀[n]` to the layer's flux sum `fm`; a reflected
bin (test 2) is removed without depositing anything, which is the only way the kernel
discards launched flux. At the top level every surviving bin is removed and deposited
there, so the flux escaping the model top is available to `gw_deposit`.

No test is applied at the launch level itself: the first one uses the level above it, so a
bin already unstable at launch deposits into the layer between the launch level and the
next level. This departs from [alexander1999](@cite), whose Eq. (22) instead removes such
waves from the launch spectrum; the *Non-orographic Gravity Waves* page gives the reasons.

The drag is then
`(ρ_source/√(ρ_k·ρ_{k+1}))·fm·ε/(z_{k+1} - z_k)`, zero below the launch level, where the
intermittency `ε` is `calc_intermitency` for AD99 and the convective coverage
`a_cover/(ρ_source·nk)` for Beres.

# Arguments

  - `waveforcing`: Output drag at cell centers [m/s²]; overwritten.
  - `mask`: Initial `StaticBitVector{nc}` of active phase-speed bins. A bit vector is used
    because `unrolled_reduce` over a plain tuple mask allocates past 32 elements.
  - `input`: Broadcasted 16-slot tuple field. Slots 1-7 are shared
    (`u_{k+1}`, `N_{k+1}`, `ρ_k`, `ρ_{k+1}`, `z_{k+1}`, `z_k`, level index); slots 8-16 hold
    the AD99 source fields or the Beres source fields, read by index under the `MODE` guard
    so each specialization stays at 16 fields. For Beres, slot 8 is the precomputed launch
    spectrum (`gw_beres_B0_u` or `gw_beres_B0_v`, a `BeresB0`); slots 9-12 are carried but
    not read.
  - `c`, `c0`: Phase-speed grid and its AD99 center [m/s].
  - `nk`, `ink`: Number of horizontal wavenumber bands and the current band [-]. The
    wavelength of band `ink` is `30·10^ink` km.
  - `level_end`: Index of the topmost level [-].
  - `beres_source`: `BeresSourceParams` or `nothing`. Not read by the kernel any more (the
    Beres spectrum is precomputed); kept so the call signature is unchanged.

The per-level body lives in `waveforcing_accumulate_step`, a named function rather than a
closure, so that inference of the Beres specialization stays concrete on the GPU.

Described in [alexander1999](@cite) and [beres2004](@cite); see the *Non-orographic Gravity
Waves* page.
"""
function waveforcing_column_accumulate!(
    waveforcing,
    mask,
    input,
    c,
    c0,
    nk,
    ink,
    level_end,
    gw_ncval::Val{nc},
    beres_source,
    source_mode::Val{MODE},
) where {nc, MODE}
    FT = eltype(waveforcing)
    kwv = 2 * FT(π) / (30 * FT(10)^ink * 1000) # wave number of gravity waves
    # Here we use column_accumulate function to pass the variable B0 and mask through different levels, and calculate waveforcing at each level.
    # The accumulate body lives in the named top-level function
    # `waveforcing_accumulate_step` rather than in the do-block. With the Beres
    # input tuple carrying the NTuple{nc} launch spectrum in slot 8, inference
    # of the same body as an anonymous closure nested inside the accumulate
    # kernel exhausts its budget and widens the state to an abstract type; on
    # the GPU that surfaces as InvalidIRError (gpu_gc_pool_alloc / checked
    # getfield) in the :beres specialization. A named function gets its own
    # precisely-typed specialization, which inference resolves exactly.
    Operators.column_accumulate!(
        waveforcing,
        input;
        init = (FT(0.0), mask, FT(NaN), ntuple(i -> FT(NaN), Val(nc))),
        transform = first,
    ) do state, inp
        waveforcing_accumulate_step(
            state, inp, c, c0, nk, kwv, level_end, gw_ncval, source_mode,
        )
    end
end

# One level step of the waveforcing accumulation; see the note above the
# column_accumulate! call for why this must be a named top-level function.
# `kwv` is the horizontal wavenumber of band `ink`, computed once per call of
# `waveforcing_column_accumulate!`.
@inline function waveforcing_accumulate_step(
    state,
    inp,
    c,
    c0,
    nk,
    kwv,
    level_end,
    gw_ncval::Val{nc},
    ::Val{MODE},
) where {nc, MODE}
    (wave_forcing, mask, Bsum_or_NaN, B0_or_NaNs) = state
    # Inputs are read by index, not destructured: the two source modes share
    # only slots 1–7; slots 8–16 hold different fields per mode. Selecting the
    # mode-specific fields under the compile-time `MODE` guard caps each
    # specialization's parameter tuple at 16 fields.
    #
    # The source launch level differs by mode: AD99 uses a fixed level
    # (pressure-based on the sphere, height-based in a column); Beres launches
    # at the top of the convective heating envelope (gw_ztop), set per column
    # in compute_tendency!.
    u_kp1 = inp[1]
    bf_kp1 = inp[2]
    ρ_k = inp[3]
    ρ_kp1 = inp[4]
    z_kp1 = inp[5]
    z_k = inp[6]
    level = inp[7]
    FT1 = typeof(u_kp1)
    # Slots 8–16 hold AD99 source fields or Beres convective fields. These locals are captured by the inner `unrolled_reduce` closure, so we select them with a single-assignment ternary.
    # Unused fields get a `zero(FT1)` dummy so the dead branch still typechecks.
    u_source_eff = MODE == :ad99 ? inp[8] : inp[15]
    ρ_source_eff = MODE == :ad99 ? inp[9] : inp[14]
    source_level_eff = MODE == :ad99 ? inp[10] : inp[13]
    Bw = MODE == :ad99 ? inp[11] : zero(FT1)
    Bn = MODE == :ad99 ? inp[12] : zero(FT1)
    cw = MODE == :ad99 ? inp[13] : zero(FT1)
    cn = MODE == :ad99 ? inp[14] : zero(FT1)
    flag = MODE == :ad99 ? inp[15] : zero(FT1)
    source_ampl = MODE == :ad99 ? inp[16] : zero(FT1)
    # Beres slot 8 is the precomputed launch spectrum (a BeresB0-wrapped
    # NTuple{nc}; see the struct's docstring for why it must be wrapped); for
    # the AD99 specialization it is a dead zero tuple. The old per-column
    # scalars (Q0/h/u_heat/N at slots 9–12) are no longer consumed — B0 already
    # folds them in (shape mode, steady β_therm + β_mech) from
    # compute_tendency!.
    beres_B0 =
        MODE == :beres ? inp[8].B0 : ntuple(_ -> zero(FT1), gw_ncval)
    beres_a_cover = MODE == :ad99 ? zero(FT1) : inp[16]

    k2 = kwv * kwv

    # Breaking threshold on B0/ĉ³. The AD99 B0 is a flux per unit mass at the launch
    # level [m²/s²], so the stress-based limit ½ρ_{k+1}·k/N is divided by ρ_source; the
    # Beres B0 is already a stress [Pa] and takes the limit as it is.
    fac =
        FT1(0.5) * (MODE == :ad99 ? ρ_kp1 / ρ_source_eff : ρ_kp1) * kwv / bf_kp1
    Hb = (z_kp1 - z_k) / log(ρ_k / ρ_kp1) # density scale height
    alp2 = FT1(0.25) / (Hb * Hb)
    ω_r = sqrt((bf_kp1 * bf_kp1 * k2) / (k2 + alp2)) # omc: (critical frequency that marks total internal reflection)

    # calculate momentum flux carried by gravity waves with different phase speeds.
    B0, Bsum = if level == 1
        mask = StaticBitVector{nc}(_ -> true)
        B1 = if MODE == :ad99
            compute_ad99_spectrum(
                c, u_source_eff, Bw, Bn, cw, cn, c0, flag, gw_ncval,
            )
        else # MODE == :beres — read the precomputed per-column launch spectrum
            beres_B0
        end
        Bsum1 = sum(abs, B1)
        B1, Bsum1
    else
        B0_or_NaNs, Bsum_or_NaN
    end

    if level >= source_level_eff
        # check break condition for each gravity waves and calculate momentum flux of breaking gravity waves at each level
        # We use the unrolled_reduce function here because it performs better for parallel execution on the GPU, avoiding type instabilities.
        # However, we need to prevent it from being inlined on the CPU to avoid large compilation times for several test cases in CI.
        # Note that @noinline has no effect on the GPU, which requires all kernel code to be inlined.
        (mask, fm) = @noinline unrolled_reduce(
            StaticOneTo(nc),
            (mask, FT1(0.0)),
        ) do (mask, fm), (n)
            if (mask[n]) == true
                c_hat = c[n] - u_kp1 # c0mu
                # The phase speed matches the wind at this level: this is the wave's
                # critical level, so it is absorbed in the layer below and deposits
                # its flux there, like the sign-change test further down (which is
                # skipped here because the reflection test needs `c_hat ≠ 0`).
                if c_hat == zero(c_hat)
                    mask = Base.setindex(mask, false, n)
                    fm = fm + B0[n]
                else
                    c_hat0 = c[n] - u_source_eff
                    # define the criterion which determines if wave is reflected at this level (test).
                    test = abs(c_hat) * kwv - ω_r
                    if test >= zero(test)
                        # wave has undergone total internal reflection. remove it from the propagating set.
                        mask = Base.setindex(mask, false, n)
                    else
                        if level == level_end
                            # this is added in MiMA implementation:
                            # all momentum flux that escapes across the model top
                            # is deposited to the extra level being added so that
                            # momentum flux is conserved
                            mask = Base.setindex(mask, false, n)
                            fm = fm + B0[n]
                        else
                            # if wave is not reflected at this level, determine if it is
                            # breaking at this level (Foc >= 0), or if wave speed relative to
                            # windspeed has changed sign from its value at the source level
                            # (c_hat0[n] * c_hat <= 0). if it is above the source level and is
                            # breaking, then add its momentum flux to the accumulated sum at
                            # this level.
                            # set mask=0 to remove phase speed band c[n] from the set of active
                            # waves moving upwards to the next level.
                            Foc = B0[n] / (c_hat)^3 - fac
                            if Foc >= zero(Foc) || (c_hat0 * c_hat <= zero(c_hat))
                                mask = Base.setindex(mask, false, n)
                                fm = fm + B0[n]
                            end
                        end
                    end # (test >= 0)

                end # (c_hat == 0)
            end # mask = 0
            return (mask, fm)
        end

        # compute the gravity wave momentum flux forcing
        # obtained across the entire wave spectrum at this level.
        eps = if MODE == :ad99
            calc_intermitency(ρ_source_eff, source_ampl, nk, FT1(Bsum))
        else # MODE == :beres
            # A zero launch density (no launch level found) gives no deposit rather
            # than an infinite one.
            ifelse(
                ρ_source_eff > zero(FT1),
                beres_a_cover / (ρ_source_eff * FT1(nk)),
                zero(FT1),
            )
        end
        rbh = sqrt(ρ_k * ρ_kp1)
        wave_forcing = (ρ_source_eff / rbh) * FT1(fm) * eps / (z_kp1 - z_k)
    end
    # Below the launch level the block is skipped and `wave_forcing` passes
    # through unchanged from the level below. It starts at the `FT(0)` from
    # `init` at level 1, so every level below the launch level returns zero.
    return (wave_forcing, mask, Bsum, B0)
end

"""
    calc_intermitency(ρ_source_level, source_ampl, nk, Bsum)

Return the AD99 wave intermittency factor `ε = F_S0/(ρ₀·nk·Σ|B₀|)` [-].

The launched spectrum is the flux of a wave train that is present only a fraction of the
time; `ε` rescales it so the long-term mean flux matches the prescribed, latitude-dependent
source amplitude `source_ampl` [kg/m/s²]. Assumes a constant phase-speed spacing `Δc`, so
the spacing cancels between `source_ampl` and `Bsum` and does not appear here. Returns
zero when the spectrum is empty (`Bsum = 0`, e.g. with `nogw_Bw = nogw_Bn = 0`): no flux
is launched then, and the division would give an infinite `ε` and a `NaN` drag.

Called from `waveforcing_column_accumulate!` in `:ad99` mode; the Beres mode uses the
convective coverage instead.
"""
function calc_intermitency(ρ_source_level, source_ampl, nk, Bsum)
    return ifelse(
        Bsum > zero(Bsum),
        (source_ampl / ρ_source_level / nk) / Bsum,
        zero(Bsum),
    )
end

"""
    NOGW_MIN_DAMP_CELLS

Number of top cells the non-orographic gravity-wave damping layer always spans [-].

The damping layer receives the momentum flux that escapes through the model top. On a
sphere it starts at the lowest level above `nogw_damp_pressure`; when fewer than
`NOGW_MIN_DAMP_CELLS` levels lie above that pressure (a model top near or below it), it
would otherwise shrink to one or two top cells, which then take all the escaped flux.
"""
const NOGW_MIN_DAMP_CELLS = 3

# Whether the damping layer has been checked for fewer than `NOGW_MIN_DAMP_CELLS` levels
# above `nogw_damp_pressure`, which happens once per session.
const NOGW_DAMP_LAYER_CHECKED = Ref(false)

"""
    highest_damp_level(nlevels)

Return the highest level index at which the damping layer may start, so that it spans at
least `NOGW_MIN_DAMP_CELLS` top cells of a column with `nlevels` levels.
"""
highest_damp_level(nlevels) = nlevels - NOGW_MIN_DAMP_CELLS + 1

"""
    gw_deposit(wave_forcing_top, wave_forcing, damp_level, level, deposit_scale)

Add the momentum flux that escaped through the model top back into the sponge layer.

Returns `wave_forcing` unchanged below `damp_level`, and
`wave_forcing + wave_forcing_top * deposit_scale` at and above it, so the escaped flux
is deposited as a uniform acceleration over the damping layer and column momentum is
conserved. Applied pointwise by `postprocess_and_accumulate!`.

# Arguments

  - `wave_forcing_top`: Drag carried by the waves that reached the model top [m/s²].
  - `wave_forcing`: Drag already accumulated at this level [m/s²].
  - `damp_level`: Index of the lowest damping level [-].
  - `level`: Index of the current level [-].
  - `deposit_scale`: Per-column ratio of the mass of the top half-level (where
    `wave_forcing_top` was diagnosed) to the integrated mass of the damping layer [-].
"""
function gw_deposit(
    wave_forcing_top,
    wave_forcing,
    damp_level,
    level,
    deposit_scale,
)
    # A heavier level needs more flux for the same acceleration, so `deposit_scale`
    # rescales `wave_forcing_top` to the acceleration that the receiving levels'
    # combined mass can support.
    if level >= damp_level
        wave_forcing = wave_forcing + wave_forcing_top * deposit_scale
    end
    return wave_forcing
end

"""
    field_shiftlevel_up!(ᶜexample_field, ᶜshifted_field, Boundary_value)

Shift a center-valued field down one level so that level `k` holds the level `k+1` value.

Mutates `ᶜshifted_field`, setting `ᶜshifted_field[k] = ᶜexample_field[k+1]` for interior
levels and `Boundary_value` at the topmost level. This is how
`non_orographic_gravity_wave_forcing` gives the kernel access to level `k+1`, which
ClimaCore column operators cannot index directly; the shift is a round trip to the faces
using top-biased interpolation in both directions.
"""
function field_shiftlevel_up!(ᶜexample_field, ᶜshifted_field, Boundary_value)
    T1 = Operators.TopBiasedC2F(; top = Operators.SetValue(Boundary_value))
    T2 = Operators.TopBiasedF2C(;)
    ᶜshifted_field .= T2.(T1.(ᶜexample_field))
end

"""
    wave_source(c, u_source, Bw, Bn, cw, cn, c0, flag, gw_ncval)

Return the AD99 launch spectrum `B₀(c)` as an `NTuple{nc}` [m²/s²].

The spectrum is the sum of a wide and a narrow Gaussian in phase speed, each of half-width
at half-maximum `cw` and `cn` and peak amplitude `Bw` and `Bn` at `c0`, multiplied by
`sgn(c - u_source)` so that waves faster and slower than the launch-level wind carry
momentum of opposite sign.

`flag` selects the reference frame of the Gaussians: `1` measures phase speed relative to
the ground (extratropics), `0` relative to the launch-level wind `u_source` (tropics). The
sign factor always uses the wind-relative speed.

# Notes

The launch spectrum is symmetric about its peak, so in the tropical frame it carries no net
momentum at launch; the atmosphere filters one side out aloft.

Described in [alexander1999](@cite) Eq. 17; see the *Non-orographic Gravity Waves* page.
"""
function wave_source(
    c,
    u_source,
    Bw,
    Bn,
    cw,
    cn,
    c0,
    flag,
    gw_ncval::Val{nc},
) where {nc}
    ntuple(
        n ->
            sign((c[n] - u_source)) * (
                Bw * exp(
                    -log(2.0f0) *
                    (
                        (c[n] * flag + (c[n] - u_source) * (1 - flag) - c0) /
                        cw
                    )^2,
                ) +
                Bn * exp(
                    -log(2.0f0) *
                    (
                        (c[n] * flag + (c[n] - u_source) * (1 - flag) - c0) /
                        cn
                    )^2,
                )
            ),
        Val(nc),
    )
end

# Number of uniform samples in the gathered in-cloud heating profile used by the
# Beres-G (sine_transform) shape factor. Compile-time constant for GPU type
# stability. The gather interpolates the heating linearly between model levels, so
# it carries no detail finer than the model levels; 64 samples resolve that
# interpolant for a heating layer up to about 64 model levels deep. For a smooth
# shape known everywhere (gate 2), 64 samples give ~4e-4 relative error over a
# 10-km layer. Bump (and re-alloc-check) if finer profiles are needed.
const N_BERES_PROFILE = 64

"""
    _beres_gs(gprofile, z_bot, span, m)

Finite sine transform `gs(m) = ∫_0^{z_top} g(z) sin(m z) dz` (Beres-G,
manuscript Eq. `eq:BeresG_Bkv_eq`) of the in-cloud heating shape `g`, sampled
uniformly on the convective envelope `[z_bot, z_top]` with `span = z_top - z_bot`
(the CLAMPED envelope width — `span` must equal the width the gather used, NOT the
unclamped moment depth `h`, which can differ when the envelope is floored at 0 or
capped at the domain top). `gprofile` is the `NTuple{ng}` of the heating `g` [K/s]
at `z_i = z_bot + (i-1)·span/(ng-1)`; `g ≡ 0` below `z_bot` (masked), so
`∫_0^{z_top} = ∫_{z_bot}^{z_top}` with `z` measured from the rigid ground `z=0` (the
local surface)
(the `W(0)=0` Dirichlet origin — Beres-G is sensitive to launch altitude through
`sin(m z)`, unlike the half-sine `V_hs`).

The squared transform `gs(m)²` is the Beres-G shape factor `V_G_sq`, used at the
**same** call site and with the **same** `·m²/(N²−ν̂²)²` placement as
`V_hs_sq`, by the algebraic parity `V_hs_sq(m,h) = |gs_halfsine(m)|²` (gate 1).

The quadrature is the Filon-type exact per-segment rule for the piecewise-linear
interpolant of `g` (no oscillatory-quadrature error at any `m`). For a smooth shape
sampled exactly at the `ng` points, the only error is the `O(dz²)` interpolation error,
asserted by the convergence gate in `test_beres_unit.jl`. The gathered model profile is
itself a linear interpolant of the heating on the model levels, so its transform also
carries the interpolation error at the model level spacing, which is coarser. The evaluation uses a rotation recurrence: `z_a`
advances by the constant `dz`, so `(sin(m z_a), cos(m z_a))` advances by a fixed
rotation by `x = m·dz` — 2 `sincos` calls TOTAL instead of 4 transcendentals per
segment, which removed a 19x GPU cost on the B0-build broadcast (95% of the
sine_transform GW call, profiled 2026-08-02). Three cancellations of the naive
per-segment form are also removed, all as exact rewrites (this evaluator is
strictly MORE accurate than the one it replaced, max Float32 error vs the
BigFloat-exact rule 4.2e-4 vs 1.6e-3 on production `m` sweeps):

  - `I0 = (cos(m z_a) − cos(m z_b))/m` → `2 sin(x/2) sin(m z_a + x/2)/m`
    (sum-to-product; difference of near-equal cosines eliminated);
  - `coefA = (sin x − x cos x)/m²` has leading term `x³/3`; evaluated by Taylor
    series for `x < 1` (10 terms, truncation ≪ Float32 eps; production
    median `x ≈ 0.02`), closed form above;
  - `coefB = (x sin x + cos x − 1)/m²` via the exact half-angle identity
    `2·sin(x/2)·(x cos(x/2) − sin(x/2))` (cancellation-free);
  - the rotation is stepped in Numerical-Recipes correction form (the update is
    a small correction to the previous pair, not a full remultiply).

`@generated` full unroll: the recurrence carries sequential state, which
`ntuple`'s stateless closure cannot express, so the unroll is generated with
LITERAL tuple indices `gprofile[i]` — runtime `NTuple` indexing spills the tuple
to memory to support dynamic indexing, which on the GPU is a heap allocation
(gpu_gc_pool_alloc) and fails to compile (InvalidIRError). Summation is strictly
left-to-right, matching the accumulation order of the rule it replaced.
"""
@generated function _beres_gs(
    gprofile::NTuple{ng, FT},
    z_bot,
    span,
    m,
) where {ng, FT}
    nseg = ng - 1
    body = Expr(:block)
    push!(
        body.args,
        quote
            m < $FT(1e-12) && return $FT(0)
            dz = span / $FT($ng - 1)
            inv_m = $FT(1) / m
            inv_m2 = inv_m * inv_m
            inv_dz = $FT(1) / dz
            x = m * dz
            sh, ch = sincos(x / $FT(2))
            x2 = x * x
            # coefB numerator: x sin x + (cos x − 1) = 2 sh (x ch − sh), exact
            # and cancellation-free (x ch − sh → x/2 as x → 0).
            sB = $FT(2) * sh * (x * ch - sh)
            # coefA numerator: sin x − x cos x, leading term x³/3 and no
            # cancellation-free half-angle form → Taylor series for x < 1 with
            # a_k = (−1)^(k+1)·2k/(2k+1)! on x^(2k+1) (10 terms: truncation at
            # x = 1 is ~1e-21, below Float64 eps); closed form for x ≥ 1. Runs
            # ONCE per (column, m), so the Horner chain is effectively free.
            if x < $FT(1)
                sA =
                    x * x2 *
                    (
                        $FT(1 / 3) +
                        x2 * (
                            $FT(-1 / 30) +
                            x2 * (
                                $FT(1 / 840) +
                                x2 * (
                                    $FT(-1 / 45360) +
                                    x2 * (
                                        $FT(2.505210838544172e-7) +
                                        x2 * (
                                            $FT(-1.927085761688332e-9) +
                                            x2 * (
                                                $FT(1.0705890838020212e-11) +
                                                x2 * (
                                                    $FT(-4.498147311143654e-14) +
                                                    x2 * (
                                                        $FT(1.4796686193355087e-16) +
                                                        x2 *
                                                        $FT(-3.914785847691228e-19)
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
            else
                c_x = $FT(1) - $FT(2) * sh * sh
                s_x = $FT(2) * sh * ch
                sA = s_x - x * c_x
            end
            coefA = sA * inv_m2
            coefB = sB * inv_m2
            two_sh_inv_m = $FT(2) * sh * inv_m
            # rotation by x in correction form: rot_a = 1 − cos x, rot_b = sin x
            rot_a = $FT(2) * sh * sh
            rot_b = $FT(2) * sh * ch
            $(Symbol(:sma_, 1)), $(Symbol(:cma_, 1)) = sincos(m * z_bot)
        end,
    )
    for i in 1:nseg
        cma_i, sma_i = Symbol(:cma_, i), Symbol(:sma_, i)
        cma_n, sma_n = Symbol(:cma_, i + 1), Symbol(:sma_, i + 1)
        ga_i, seg_i = Symbol(:ga_, i), Symbol(:seg_, i)
        push!(
            body.args,
            quote
                # sin(m z_a + x/2), for the cancellation-free I0
                $(Symbol(:smid_, i)) = $sma_i * ch + $cma_i * sh
                $cma_n = $cma_i - (rot_a * $cma_i + rot_b * $sma_i)
                $sma_n = $sma_i - (rot_a * $sma_i - rot_b * $cma_i)
                $ga_i = gprofile[$i]
                $seg_i =
                    $ga_i * (two_sh_inv_m * $(Symbol(:smid_, i))) +
                    (gprofile[$(i + 1)] - $ga_i) *
                    inv_dz *
                    ($cma_i * coefA + $sma_i * coefB)
            end,
        )
    end
    acc = Symbol(:seg_, 1)
    for i in 2:nseg
        acc = :($acc + $(Symbol(:seg_, i)))
    end
    push!(body.args, :(return $acc))
    return body
end

"""
    _beres_shape_factor_sq(shape_general, gprofile, z_bot, h, span, m, m_max)

The squared vertical shape factor entering `R² = (·)·m²/(N²−ν̂²)²`, dispatching
between the classical half-sine `V_hs_sq(m,h)` (`shape_general=false`, or
`gprofile===nothing`) and the Beres-G generalized `|gs(m)|²`
(`shape_general=true`). `h` is the (unclamped) half-sine depth used by `V_hs_sq`;
`span = z_top - z_bot` is the gather/transform grid width used by `_beres_gs` —
they differ only for a clamped envelope. Holding the `·m²/(N²−ν̂²)²` downstream
factor fixed is the `1/m²`-parity pinned by gate 1. The Beres-G factor is zero for
`m > m_max`, the shortest vertical wavelength the gathered profile resolves (`m_max = π/dz_level`, or `Inf` for no cut); the half-sine factor ignores `m_max`.
"""
@inline _beres_shape_factor_sq(::Bool, ::Nothing, z_bot, h, span, m, m_max) =
    V_hs_sq(m, h)
@inline function _beres_shape_factor_sq(
    shape_general::Bool,
    gprofile::NTuple,
    z_bot,
    h,
    span,
    m,
    m_max,
)
    shape_general || return V_hs_sq(m, h)
    return m > m_max ? zero(m) : _beres_gs(gprofile, z_bot, span, m)^2
end

"""
    V_hs_sq(m, h)

Return the squared vertical shape factor of a half-sine heating profile.

For a heating layer of depth `h` [m] and vertical wavenumber `m` [1/m],

    V_hs_sq = (π/h)² · sin²(m·h) / (m² − (π/h)²)²    [m²]

evaluated in `sinc` form with `δ = m·h − π`, which stays finite at the resonance
`m = π/h` (vertical wavelength `2h`), where it takes the value `h²/4`.

Shared by both Beres source paths: the transient integrand in
`_beres_spectrum_single_h` and the steady wave in `_beres_steady_flux`. Both build the
squared response `R² = V_hs_sq·m²/(N² − ν̂²)²` from it.

# Notes

`sinc` here is the unnormalized `sin(x)/x`, inlined deliberately; `Base.sinc` is the
normalized `sin(πx)/(πx)` and would be wrong.
"""
@inline function V_hs_sq(m, h)
    FT = typeof(m)
    δ = m * h - FT(π)
    sinc_δ = abs(δ) < FT(1e-10) ? FT(1) : sin(δ) / δ
    return FT(π)^2 * h^2 * sinc_δ^2 / (m * h + FT(π))^2
end

"""
    _beres_spectrum_single_h(c_n, c_hat, h, u_heat, N2, N_source, Q0_sq, σ_x_sq,
                             ν_min, dν, n_groups, boole_w, shape_general = false,
                             gprofile = nothing, z_bot = 0, span = 0, m_max = Inf)

Return the unsigned launched momentum flux of one phase-speed bin and one heating depth.

The Beres spectral density is integrated over ground-relative frequency
`ν ∈ [ν_min, ν_max]` at fixed phase speed `c_n`, using composite Boole's rule over
`n_groups` panels of five nodes with weights `boole_w` and node spacing `dν`. The caller
applies the sign and the scale factor.

Each node contributes the product of

  - the horizontal heating spectrum `Q0²·G_k² = Q0²·(σ_x²/2)·exp(−k²σ_x²/2)`, with
    `k = ν/c_n`;
  - the squared vertical response `R² = S²(m)·m²/(N² − ν̂²)²`, with `m` from
    `m² = k²(N²/ν̂² − 1)` and the squared shape factor `S²(m)` from
    `_beres_shape_factor_sq`: the half-sine `V_hs_sq(m, h)` by default, or, when
    `shape_general` is true and `gprofile` is given, the squared sine transform of the
    gathered heating profile on `[z_bot, z_bot + span]`, zero for `m > m_max`;
  - the propagation factor `√(N² − ν̂²)/|ν̂|` and the normalization `1/√(2π)`; and
  - the `ν → c` Jacobian `ν/c_n²`,

where `ν̂ = ν − k·u_heat` is the intrinsic frequency. Nodes that are evanescent or singular
(`|ν̂| < 1e-4·N_source`, `|ν̂| ≥ N_source`, or `m² ≤ 0`) contribute zero.

# Notes

`c_hat` is accepted for symmetry with the caller's bookkeeping but is not read; the
integrand depends on the intrinsic frequency `ν̂`, not on the intrinsic phase speed.

The function is `@noinline` as a GPU compile-time guard: the launch-spectrum assembler
unrolls this body once per phase-speed bin (nc = 251). Inlined, that yields a huge basic
block that LLVM's SLP store-chain vectorizer works on for hours (observed: more than
75 min stuck in vectorizeStoreChain while compiling the B0 broadcast kernel). Not inlined,
the body compiles once and is called 251 times, with bit-identical results, a ~100x
smaller block, and negligible runtime cost at the `dt_nogw` cadence.

Described in [beres2004](@cite) Eqs. 23 and 29-30; see the *Non-orographic Gravity Waves*
page.
"""
@noinline function _beres_spectrum_single_h(
    c_n,
    c_hat,
    h,
    u_heat,
    N2,
    N_source,
    Q0_sq,
    σ_x_sq,
    ν_min,
    dν,
    n_groups,
    boole_w,
    shape_general = false,
    gprofile = nothing,
    z_bot = zero(c_n),
    span = zero(c_n),
    m_max = oftype(c_n, Inf),
)
    FT = typeof(c_n)
    π_val = FT(π)
    integral = FT(0)

    for g in 1:n_groups
        for j in 1:5
            idx = (g - 1) * 4 + j
            ν_j = ν_min + FT(idx - 1) * dν

            k = ν_j / c_n
            ν_hat = ν_j - k * u_heat

            ν_hat_min = FT(1e-4) * N_source
            if abs(ν_hat) < ν_hat_min
                continue
            end
            if abs(ν_hat) >= N_source
                continue
            end

            m_sq = k^2 * (N2 / ν_hat^2 - FT(1))
            if m_sq <= FT(0)
                continue
            end

            m = sqrt(m_sq)

            N2_minus_νhat2 = N2 - ν_hat^2

            # Squared vertical shape factor, shared with the steady path.
            # Classical: V_hs_sq (half-sine). Beres-G: |gs(m)|² (sine transform of
            # the EDMF profile) — same call site, same downstream m²/(N²−ν̂²)²
            # (gate-1 1/m² parity). R² = shape_sq·m²/(N²−ν̂²)²; only R² is needed
            # (the sign of R is squared away in B_sq).
            shape_sq = _beres_shape_factor_sq(
                shape_general,
                gprofile,
                z_bot,
                h,
                span,
                m,
                m_max,
            )
            R_sq = shape_sq * m^2 / N2_minus_νhat2^2

            Q0Gk_sq =
                Q0_sq * σ_x_sq / FT(2) * exp(-k^2 * σ_x_sq / FT(2))
            B_sq = Q0Gk_sq * R_sq

            propagation_factor = sqrt(N2_minus_νhat2) / abs(ν_hat)
            F_kν =
                FT(1) / sqrt(FT(2) * π_val) * propagation_factor * B_sq

            jacobian = ν_j / c_n^2
            f_val = F_kν * jacobian

            w = boole_w[j] * FT(2) * dν / FT(45)
            integral = integral + w * f_val
        end
    end
    return integral
end

"""
    _beres_steady_horizontal_const(σ_x, L_system)

Return the horizontal-scale factor `H` of the steady source amplitude [m²].

`H` measures how strongly a convective cell of half-width `σ_x` [m] projects onto gravity
waves, summed over all horizontal scales down to the largest system size `L_system` [m]
and over both signs of the horizontal wavenumber `k`:

    H = ∫_{|k| > k_min} G_k²/|k| dk = (σ_x²/2) · E₁(x),   x = k_min²·σ_x²/2,   k_min = 2π/L_system

[beres2004](@cite) Eq. 34 prints `H_k = G_k²/k`, which is odd in `k` and would integrate
to zero over both signs; with the flux signed by `sgn(ĉ) = sgn(ν̂/k)`, both signs of `k`
give steady flux of the same sign, so the integrand is `G_k²/|k|`. Both halves are
counted here, which equals the `ν → 0` limit of the transient integrand summed over both
signs of `k` (up to the hydrostatic `m₀ = N/|U|`).

The exponential integral is evaluated from its small-argument series
`E₁(x) ≈ −γ_E − ln(x) + x`, which is accurate for the intended `x ≪ 1` and needs only a
`log`, so it is GPU-safe. Outside that regime (`x ≥ 1`) `H` is set to zero rather than
extrapolated.

Called from `_beres_steady_flux`; this is the only place `L_system` enters. Described in
[beres2004](@cite) Eq. 32.
"""
@inline function _beres_steady_horizontal_const(σ_x, L_system)
    FT = typeof(σ_x)
    γ = FT(0.5772156649015329)
    a = σ_x^2 / FT(2)
    k_min = FT(2) * FT(π) / L_system
    x = a * k_min^2
    E1 = x < FT(1) ? (-γ - log(x) + x) : FT(0)
    return a * E1
end

"""
    _beres_steady_flux(U, N_source, h, Q0, scale_factor, n_h_avg, Δh_frac, σ_x,
                       L_system, dc_frac, ν_min, shape_general = false,
                       gprofile = nothing, z_bot = 0, span = 0, m_max = Inf)

Return the launch flux of the steady (ν = 0) convective wave for one azimuth.

The time-mean convective heating acts like a fixed hill in the mean wind `U` [m/s] and
radiates a single ground-stationary wave of vertical wavenumber `m₀ = N/|U|`:

    F_steady = −sign(U) · scale_factor · (1/√(2π)) · Q0² · Q_t(0)² · V_hs_sq(m₀,h) · H
               / (N·|U|³)

The sign opposes `U`, so the wave decelerates the flow like mountain-wave drag. The caller
deposits this into the `c ≈ 0` phase-speed bin, which the transient spectrum leaves empty,
so nothing is double-counted; the two components carry orthogonal frequencies.

Everything except two pieces is shared with the transient path: `H` is the horizontal
constant from `_beres_steady_horizontal_const`, and `Q_t(0)² = dc_frac·ν_min` is the
zero-frequency temporal weight. Neither [beres2004](@cite) nor its WACCM use
(Beres et al. 2005, which has no steady term) fixes these two: `dc_frac` is the fraction
of the white transient spectrum carried down to `ν = 0` over the width `ν_min` (default 1),
and `L_system` sets the smallest wavenumber in `H`, on which `H` depends logarithmically.
Both are free knobs. When
`n_h_avg > 1` the half-sine factor is averaged over `n_h_avg` depths in
`h ± Δh_frac·h`, the same resonance smoothing the transient path applies.

Returns exactly zero for `|U| < 1e-6` m/s, where `m₀ → ∞`, the stationary wave vanishes,
and the `1/|U|³` factor would otherwise blow up. With the gathered profile (Beres-G), it also
returns zero when `m₀` exceeds what the profile resolves, `m₀·span/(ng-1) > π`, or the
wavenumber `m_max = π/dz_level` of the model levels it was gathered from.

Described in [beres2004](@cite) Eqs. 31-34; see the *Non-orographic Gravity Waves* page.
"""
@inline function _beres_steady_flux(
    U,
    N_source,
    h,
    Q0,
    scale_factor,
    n_h_avg,
    Δh_frac,
    σ_x,
    L_system,
    dc_frac,
    ν_min,
    shape_general = false,
    gprofile = nothing,
    z_bot = zero(U),
    span = zero(U),
    m_max = oftype(U, Inf),
)
    FT = typeof(U)
    # U → 0 guard: m₀ = N/|U| → ∞, the stationary wave vanishes; also protects
    # the 1/U³ below from blowing up.
    if abs(U) < FT(1e-6)
        return FT(0)
    end
    Uabs = abs(U)
    m0 = N_source / Uabs

    # Vertical shape factor at m₀. Classical: half-sine V_hs_sq, optionally
    # h-averaged to smooth the m₀ ≈ π/h resonance (the transient `n_h_avg`
    # mechanism). Beres-G: |gs(m₀)|² of the gathered profile (single eval — the
    # real profile self-smooths; Ext 1 applies to the thermal-steady term too).
    # `span = z_top - z_bot` (clamped envelope width) must match the gather grid.
    Vbar = if shape_general && gprofile !== nothing
        # The gathered profile resolves vertical wavelengths down to two gather
        # spacings. Beyond that, i.e. as U → 0, its transform is set by the profile's
        # edges and kinks rather than by the heating, falls off only like 1/m₀, and
        # Vbar/|U|³ grows like 1/|U|, so the steady flux is zero there.
        # It is also zero beyond the wavenumber `m_max` of the model levels.
        dz_gather = span / FT(length(gprofile) - 1)
        (m0 * dz_gather > FT(π)) | (m0 > m_max) ? FT(0) :
        _beres_gs(gprofile, z_bot, span, m0)^2
    elseif n_h_avg <= 1
        V_hs_sq(m0, h)
    else
        Δh = Δh_frac * h
        h_min = h - Δh
        dh = FT(2) * Δh / FT(n_h_avg - 1)
        acc = FT(0)
        for ih in 1:n_h_avg
            acc += V_hs_sq(m0, h_min + FT(ih - 1) * dh)
        end
        acc / FT(n_h_avg)
    end

    H = _beres_steady_horizontal_const(σ_x, L_system)
    Qt0_sq = dc_frac * ν_min

    amp =
        scale_factor * (FT(1) / sqrt(FT(2) * FT(π))) * Q0^2 * Qt0_sq * Vbar * H /
        (N_source * Uabs^3)

    # Sign: oppose U (β < 0 ⇒ decelerating mountain-wave-like drag).
    return -sign(U) * amp
end

"""
    _beres_mech_flux(U, N_source, w_obstacle, scale_factor, mech_weight, σ_x, L_system)

Mechanical / obstacle steady (ν=0) launch flux (Extension 3). A convective
updraft is a moving obstacle the mean wind `U` flows over, radiating a stationary
wave at `m₀ = N/|U|` — the same `ν=0` slot as the thermal steady term — with the
Garner base-flux scaling (manuscript Eqs. `eq:garner_baseflux`–`eq:tau_mech`):

    τ_mech ~ ρ₀ U w_b² / N,   h_eff = w_b/N

quadratic in the obstacle-level draft velocity `w_b`, **linear in U** (vanishes as
`U→0`). We implement the `eq:tau_mech` `U^{+1}` scaling, NOT `eq:beta_total`'s
divergent `∝ 1/U` form (a manuscript inconsistency flagged for author fix).

Carried through the **same** change of variables as the thermal steady term: same
horizontal constant `H`, same `1/√(2π)`, same `scale_factor`, same `sign(U)`
deceleration, into the same `c≈0` bin. Carries **no heating shape factor** (bare
obstacle scaling — Ext 1's `V_G` generalizes `β_therm` only). Its size relative
to `β_therm` is set by `mech_weight` (the `ν=0` weight, mechanical analogue of the
thermal `Q_t²(0) = dc_frac·ν_min`), not by the transient calibration. The thermal
term's `/(N·|U|³)` becomes the obstacle `·|U|/N` here, reproducing the
manuscript's `β_therm ∝ U^{-3}` vs. `β_mech ∝ U^{+1}` split.

Frame note: placed at `c≈0` (cell-stationary, manuscript convention). At `c≈0`
the mechanical wave meets its critical level near the convective outflow and
deposits low (like the thermal steady term); reaching QBO altitudes would require
the deferred cell-speed / low-`c` framing — an open modeling decision (see report).
"""
@inline function _beres_mech_flux(
    U,
    N_source,
    w_obstacle,
    scale_factor,
    mech_weight,
    σ_x,
    L_system,
)
    FT = typeof(U)
    if abs(U) < FT(1e-6)
        return FT(0)   # no obstacle problem without environmental wind
    end
    Uabs = abs(U)
    H = _beres_steady_horizontal_const(σ_x, L_system)
    # τ_mech ~ ρ₀ U w_b²/N (ρ₀ folded into scale_factor·mech_weight, as the
    # thermal term folds ρ₀/(Lτ)); ∝ |U| ⇒ vanishes as U→0.
    amp =
        scale_factor * (FT(1) / sqrt(FT(2) * FT(π))) * mech_weight *
        w_obstacle^2 * Uabs * H / N_source
    # Sign: oppose U (decelerating, like the thermal steady term).
    return -sign(U) * amp
end

"""
    wave_source(c, u_heat, Q0, h, N_source, beres::BeresSourceParams, gw_ncval)

Return the Beres convective launch spectrum `B₀(c)` as an `NTuple{nc}` of the momentum flux
carried by each phase-speed bin [Pa], with the dimensional prefactor absorbed into
`beres_scale_factor`.

Dispatch on `BeresSourceParams` distinguishes this from the AD99 Gaussian method above,
whose `B₀` is a flux per unit mass [m²/s²] that the AD99 intermittency rescales.

This is the classical half-sine spectrum only: it calls `_beres_launch_spectrum` with no
gathered heating profile and zero obstacle velocity, so `beres_shape_general` and
`beres_mechanical_source` have no effect here. The production model does not call it; it
precomputes the full spectrum (including the Beres-G shape and the mechanical source) in
`compute_beres_convective_heating!`. It is kept for tests and diagnostics.

For each phase-speed bin, `_beres_spectrum_single_h` integrates the spectral density over
frequency, a flux per unit phase speed, and the result is signed by `sgn(ĉ)` with
`ĉ = c - u_heat` the intrinsic phase speed, then multiplied by `beres_scale_factor` and by
the bin width `dc` to give the flux in the bin. Bins with `|ĉ| < 1e-6` or `|c| < 1e-6` m/s
are set to zero, the latter because the `ν → c` change of variables is singular there.
When `n_h_avg > 1`, the integral is averaged over `n_h_avg` heating depths spanning
`h ± Δh_frac·h`, which smooths the `mh = π` resonance.

The steady (ν = 0) component from `_beres_steady_flux` is added to the `c ≈ 0` bin, which
the transient spectrum leaves empty. It is included only when `beres_steady_source` is set
and the phase-speed grid has an exact `c ≈ 0` bin; on a grid without one, adding it would
corrupt a nonzero bin, so it is dropped instead.

# Arguments

  - `c`: Phase-speed grid [m/s].
  - `u_heat`: Mass-weighted mean wind in the heating layer, for this azimuth [m/s].
  - `Q0`: Half-sine heating amplitude [K/s].
  - `h`: Heating depth [m].
  - `N_source`: Mean buoyancy frequency in the heating layer [1/s].
  - `beres`: Source parameters.
  - `gw_ncval`: `Val(nc)` with `nc` the number of phase-speed bins.

Described in [beres2004](@cite) Eqs. 23 and 29-30; see the *Non-orographic Gravity Waves*
page.
"""
function wave_source(
    c,
    u_heat,
    Q0,
    h,
    N_source,
    beres::BeresSourceParams,
    gw_ncval::Val{nc},
) where {nc}
    # Classical half-sine launch spectrum (no gathered profile, no mechanical
    # term). Production Beres-G goes through `_beres_launch_spectrum` with the
    # gathered EDMF profile and obstacle velocity; this wrapper preserves the
    # test/diagnostic interface and the exact half-sine arithmetic (gprofile =
    # nothing ⇒ V_hs everywhere regardless of `beres_shape_general`; w_obstacle =
    # 0 ⇒ no mechanical contribution regardless of `beres_mechanical_source`).
    FT = typeof(u_heat)
    return _beres_launch_spectrum(
        c, u_heat, Q0, h, N_source, FT(0), FT(0), nothing, FT(0), beres,
        gw_ncval,
    )
end

"""
    BeresLaunchFilter(mode, wind_min, wind_max, margin, wind_profile, wind_launch,
                      taper_points)

The launch filter of one Beres azimuth in one column; see `BeresSourceParams`
(`launch_filter`). `mode` is `BERES_LAUNCH_FILTER_NONE`, `BERES_LAUNCH_FILTER_BERES2004`
or `BERES_LAUNCH_FILTER_HEIGHT_RESOLVED`; `wind_min`, `wind_max` [m/s] are the range of
this azimuth's wind over the heating layer up to the launch level, `margin` [m/s] widens
that range on both sides, `wind_profile` is the wind on the gather grid (Beres-G) or
`nothing` (half-sine), `wind_launch` [m/s] is the wind at the launch level, and
`taper_points` is the width, in gather spacings, of the raised-cosine taper the
height-resolved filter puts on the edge of the heating it keeps (0 for a hard edge).
"""
struct BeresLaunchFilter{FT, P}
    mode::Int
    wind_min::FT
    wind_max::FT
    margin::FT
    wind_profile::P
    wind_launch::FT
    taper_points::FT
end

# The launch filter of azimuth 1 (u) or 2 (v) from the column's launch-filter winds
# `wind_range = (u_min, u_max, v_min, v_max, u_L, v_L)` and that azimuth's wind on the
# gather grid (`nothing` for the half-sine path).
@inline function _beres_launch_filter(
    beres,
    wind_range_raw,
    wind_profile_raw,
    ::Val{azimuth},
    taper_points = 0,
) where {azimuth}
    wind_range = _beres_unwrap_gprofile(wind_range_raw)
    FT = typeof(wind_range[1])
    (i_min, i_max, i_launch) = azimuth == 1 ? (1, 2, 5) : (3, 4, 6)
    return BeresLaunchFilter(
        beres.launch_filter,
        wind_range[i_min],
        wind_range[i_max],
        FT(beres.launch_filter_margin),
        _beres_unwrap_gprofile(wind_profile_raw),
        wind_range[i_launch],
        FT(taper_points),
    )
end

# Whether the filter removes the phase speed `c` outright: the Beres (2004) filter removes
# every `c` in the wind range over the layer (± margin); the height-resolved filter does
# so only without a gathered profile to mask (the half-sine path).
@inline _beres_filter_removes(::Nothing, c) = false
@inline function _beres_filter_removes(f::BeresLaunchFilter, c)
    in_band = (c >= f.wind_min - f.margin) & (c <= f.wind_max + f.margin)
    removes_band =
        (f.mode == BERES_LAUNCH_FILTER_BERES2004) | (
            (f.mode == BERES_LAUNCH_FILTER_HEIGHT_RESOLVED) &
            (f.wind_profile === nothing)
        )
    return removes_band & in_band
end
# Whether `c` lies in the filter band at all, for the source terms with no profile to
# mask (the mechanical steady term).
@inline _beres_filter_band(::Nothing, c) = false
@inline _beres_filter_band(f::BeresLaunchFilter, c) =
    (f.mode != BERES_LAUNCH_FILTER_NONE) &
    (c >= f.wind_min - f.margin) &
    (c <= f.wind_max + f.margin)

"""
    _beres_filter_profile(launch_filter, gprofile, c)

Return the gathered heating profile the phase speed `c` sees through the launch filter.
With the height-resolved filter, the heating at a gather point contributes only if the
wind between that point and the launch level never comes within `margin` of `c`: a wave
of phase speed `c` excited lower meets a critical level on its way up through the
heating layer. Every point at and below the highest such crossing is set to zero, and the
points above it rise to the full profile over `taper_points` gather spacings with a
raised cosine, as the window edges are tapered, so the cut adds no jump to the profile.
The taper moves the effective cut up by about one level spacing (`taper_points` gather
spacings is one `dz_level`), and the heating under it is lost: unlike the window-edge
taper, this masking is per phase speed and is not rescaled to the column heating.
Otherwise the profile is returned unchanged.
"""
@inline _beres_filter_profile(::Nothing, gprofile, c) = gprofile
@inline _beres_filter_profile(::BeresLaunchFilter{FT, Nothing}, gprofile, c) where {FT} =
    gprofile
@inline function _beres_filter_profile(
    f::BeresLaunchFilter{FT, <:NTuple{ng}},
    gprofile::NTuple{ng},
    c,
) where {FT, ng}
    f.mode == BERES_LAUNCH_FILTER_HEIGHT_RESOLVED || return gprofile
    # The highest segment whose winds bracket c ± margin: between gather points s and
    # s + 1, or, for s = ng, between the top gather point and the launch level. Every
    # point at and below its lower end s is cut. (Literal tuple indices only: a
    # runtime index into an NTuple does not compile on the GPU.)
    w = f.wind_profile
    i_cut = maximum(
        ntuple(Val(ng)) do s
            w_upper = s < ng ? w[min(s + 1, ng)] : f.wind_launch
            crosses =
                (min(w[s], w_upper) <= c + f.margin) &
                (max(w[s], w_upper) >= c - f.margin)
            ifelse(crosses, s, 0)
        end,
    )
    width = max(f.taper_points, eps(FT))
    return ntuple(Val(ng)) do j
        d = FT(j - i_cut)   # distance above the cut, in gather spacings
        factor = ifelse(
            (i_cut > 0) & (d < f.taper_points),
            (1 - cos(FT(π) * d / width)) / 2,
            one(FT),
        )
        ifelse(j > i_cut, gprofile[j] * factor, zero(gprofile[j]))
    end
end

"""
    N_BERES_BIN_SAMPLES

Number of equally spaced phase speeds across each bin `[c - dc/2, c + dc/2]` over which a
transient Beres-G bin averages the spectral density [-]. Chosen from a convergence check
on a frozen 1M state (400 active columns, 63-level grid, 0.8 m/s bins): going from 8 to 16
samples changes the bin values by 0.3% (summed absolute change over all bins and
columns), the median column-total flux by 0.1%, and the area-weighted total by 0.01%;
going from 4 to 8 changes the bin values by 1.9%.
"""
const N_BERES_BIN_SAMPLES = 8

"""
    _beres_transient_bin(c_n, dc, u_heat, h, N2, N_source, Q0_sq, σ_x_sq, ν_min, dν,
                         n_groups, boole_w, use_VG, gprofile, z_bot, span, m_max,
                         n_h_avg, Δh_frac, scale_factor, launch_filter,
                         Val(n_bin_samples))

Return the signed transient momentum flux carried by the phase-speed bin centred at
`c_n` of width `dc` [Pa]: `scale_factor` times the spectral density, signed by the
intrinsic phase speed `ĉ = c - u_heat`, times `dc`. The `c = 0` bin carries no transient
flux (it holds the steady term).

With the gathered profile (`use_VG`), the signed density is averaged over
`n_bin_samples` equally spaced phase speeds across the bin, `c_n + dc·((k - 1/2)/n - 1/2)`,
instead of taken at its centre. The Beres-G density varies on scales finer than a bin near
`u_heat` (the sine transform measured from the ground oscillates in the vertical
wavenumber `m ≈ N/|ĉ|`), so a value at the bin centre swings from bin to bin as `u_heat`
moves; the bin average does not. Sub-samples with `|ĉ|` or `|c|` below `1e-6` m/s
contribute zero. The half-sine path takes the centre value, averaged over `n_h_avg`
heating depths in `h ± Δh_frac·h` when `n_h_avg > 1`.

`launch_filter` (a `BeresLaunchFilter`, or `nothing` for none) removes the phase speeds in
the wind range over the heating layer or, height-resolved, masks the profile each phase
speed sees (`_beres_filter_profile`).

Shared by `_beres_launch_spectrum` and the five-way diagnostic `_beres_banded_spectrum`.

The function is `@noinline` as a GPU compile-time guard, as `_beres_spectrum_single_h`:
the launch-spectrum builder unrolls one call per phase-speed bin (251), and inlined, the
sample loop and the launch filter's profile masking in each make a block that LLVM takes
hours to compile.
"""
@noinline function _beres_transient_bin(
    c_n,
    dc,
    u_heat,
    h,
    N2,
    N_source,
    Q0_sq,
    σ_x_sq,
    ν_min,
    dν,
    n_groups,
    boole_w,
    use_VG,
    gprofile,
    z_bot,
    span,
    m_max,
    n_h_avg,
    Δh_frac,
    scale_factor,
    launch_filter,
    ::Val{n_bin_samples},
) where {n_bin_samples}
    FT = typeof(c_n)
    abs(c_n) < FT(1e-6) && return FT(0)
    if use_VG
        acc = FT(0)
        for k in 1:n_bin_samples
            c_k = c_n + dc * ((FT(k) - FT(0.5)) / FT(n_bin_samples) - FT(0.5))
            ĉ_k = c_k - u_heat
            usable =
                (abs(ĉ_k) >= FT(1e-6)) & (abs(c_k) >= FT(1e-6)) &
                !_beres_filter_removes(launch_filter, c_k)
            acc +=
                usable ?
                sign(ĉ_k) * _beres_spectrum_single_h(
                    c_k, ĉ_k, h, u_heat, N2, N_source, Q0_sq, σ_x_sq, ν_min, dν,
                    n_groups, boole_w, use_VG,
                    _beres_filter_profile(launch_filter, gprofile, c_k), z_bot,
                    span, m_max,
                ) : FT(0)
        end
        # The integral is a flux per unit phase speed; times the bin width it is the
        # flux the bin carries.
        return scale_factor * (acc / FT(n_bin_samples)) * dc
    end
    c_hat = c_n - u_heat
    ((abs(c_hat) < FT(1e-6)) | _beres_filter_removes(launch_filter, c_n)) &&
        return FT(0)
    result = if n_h_avg <= 1
        _beres_spectrum_single_h(
            c_n, c_hat, h, u_heat, N2, N_source,
            Q0_sq, σ_x_sq, ν_min, dν, n_groups, boole_w,
            use_VG, gprofile, z_bot, span, m_max,
        )
    else
        # Half-sine averaged over n_h_avg values of h in [h-Δh, h+Δh] per
        # Beres (2004) Section 2, Figure 4.
        Δh = Δh_frac * h
        h_min = h - Δh
        h_max = h + Δh
        dh = (h_max - h_min) / FT(n_h_avg - 1)
        acc = FT(0)
        for ih in 1:n_h_avg
            h_i = h_min + FT(ih - 1) * dh
            acc += _beres_spectrum_single_h(
                c_n, c_hat, h_i, u_heat, N2, N_source,
                Q0_sq, σ_x_sq, ν_min, dν, n_groups, boole_w,
                false, nothing, z_bot, span,
            )
        end
        acc / FT(n_h_avg)
    end
    # The integral is a flux per unit phase speed; times the bin width it is the flux
    # the bin carries.
    return sign(c_hat) * scale_factor * result * dc
end

"""
    _beres_launch_spectrum(c, u_heat, Q0, h, N_source, z_bot, span, gprofile, w_obstacle, beres, gw_ncval)

The full per-column Beres launch momentum-flux spectrum `B0` (NTuple over phase
speed), assembled once per column (it is independent of the AD99 horizontal
wavenumber, so the propagation kernel reads it precomputed rather than rebuilding
it inside the `ink` loop). Each bin holds the momentum flux it carries [Pa]: a
transient bin is the Beres (2004) spectral density, integrated over intrinsic
frequency, times the bin width `dc`; the `c≈0` bin carries the steady (`ν=0`)
terms, which are a total flux already. `h` is the half-sine depth (V_hs/Q0);
`z_bot` and `span` are the bottom and width of the window the gather sampled on,
used by the Beres-G sine transform, and the keyword `dz_level` is the mean model level
thickness over that window: Beres-G drops vertical wavenumbers `m > π/dz_level` (no cut
for the default `0`).

`gprofile` (the gathered heating [K/s], `NTuple{ng}` on `[z_bot, z_top]`) and the flag
`beres.beres_shape_general` select the vertical shape factor:

  - `gprofile === nothing` or `shape_general=false` → classical half-sine `V_hs`
    (with `n_h_avg` resonance averaging), Beres (2004);
  - `shape_general=true` with a profile → Beres-G `|gs(m)|²` (Ext 1), single
    evaluation (the real profile self-smooths; `n_h_avg` bypassed).

The `c≈0` bin sums `β_therm` (thermal steady, shape-aware) and, when
`beres_mechanical_source`, `β_mech` (Ext 3 obstacle term from `w_obstacle`).
"""
function _beres_launch_spectrum(
    c,
    u_heat,
    Q0,
    h,
    N_source,
    z_bot,
    span,
    gprofile,
    w_obstacle,
    beres::BeresSourceParams,
    gw_ncval::Val{nc},
    shape_mode::Val{SG} = Val(beres.beres_shape_general);
    dz_level = zero(u_heat),
    n_bin_samples::Val = Val(N_BERES_BIN_SAMPLES),
    launch_filter = nothing,
) where {nc, SG}
    (;
        σ_x,
        ν_min,
        ν_max,
        n_ν,
        beres_scale_factor,
        n_h_avg,
        Δh_frac,
        beres_steady_source,
        beres_steady_dc_frac,
        beres_L_system,
        beres_mechanical_source,
        beres_mech_weight,
    ) = beres
    FT = typeof(u_heat)
    scale_factor = FT(beres_scale_factor)
    L_system = FT(beres_L_system)
    # Beres-G uses the gathered profile only when the shape flag is on AND a
    # profile is supplied (the wave_source wrapper passes `nothing` ⇒ always
    # half-sine). `SG` is the flag threaded as a COMPILE-TIME `Val` (the
    # production b0g/b0hs broadcasts pass `Val(true)`/`Val(false)`), so `use_VG`
    # const-folds and the GPU kernel compiles exactly one shape path. A runtime
    # `beres.beres_shape_general` read here would instead force BOTH the
    # half-sine and `_beres_gs` transform branches into the kernel (dynamic
    # dispatch / GC allocation). Behaviour is unchanged: the flag still selects
    # the shape, so a profile present with the flag off stays exact half-sine.
    use_VG = SG && (gprofile !== nothing)
    # Beres-G drops vertical wavenumbers above π/dz_level: the gathered profile
    # interpolates the heating between model levels, so it holds no information at
    # shorter vertical wavelengths, only its kinks and edges. No cut without a spacing.
    m_max = dz_level > zero(dz_level) ? FT(π) / dz_level : FT(Inf)

    boole_w = (FT(7), FT(32), FT(12), FT(32), FT(7))
    dν = (ν_max - ν_min) / FT(n_ν - 1)
    n_groups = (n_ν - 1) ÷ 4

    N2 = N_source^2
    σ_x_sq = σ_x^2
    # Beres-G transforms the gathered heating itself [K/s], so its amplitude factor is
    # 1; the half-sine path multiplies its unit shape by Q0.
    Q0_eff = use_VG ? one(Q0) : Q0
    Q0_sq = Q0_eff^2

    # Steady (ν=0): a ground-stationary wave landing in the c≈0 bin (kept empty by
    # the transient spectrum, so no double-counting). Deposits only with an exact
    # c=0 bin (else `clamp` would corrupt a nonzero bin → zero it). β_tot =
    # β_therm + β_mech (Ext 3); each is independently flag-gated.
    dc = c[2] - c[1]
    cmax = -c[1]
    n_zero = clamp(round(Int, cmax / dc) + 1, 1, nc)
    has_c0_bin = abs(c[n_zero]) < FT(1e-6)
    # The launch filter treats the c = 0 bin as any other: removed when c = 0 lies in
    # the filter band, or, height-resolved, built from the heating above the highest
    # point where the wind between it and the launch level changes sign.
    steady_therm =
        (
            beres_steady_source && has_c0_bin &&
            !_beres_filter_removes(launch_filter, zero(FT))
        ) ?
        _beres_steady_flux(
            u_heat,
            N_source,
            h,
            Q0_eff,
            scale_factor,
            n_h_avg,
            Δh_frac,
            σ_x,
            L_system,
            FT(beres_steady_dc_frac),
            ν_min,
            use_VG,
            _beres_filter_profile(launch_filter, gprofile, zero(FT)),
            z_bot,
            span,
            m_max,
        ) : FT(0)
    steady_mech =
        (
            beres_mechanical_source && has_c0_bin &&
            !_beres_filter_band(launch_filter, zero(FT))
        ) ?
        _beres_mech_flux(
            u_heat,
            N_source,
            w_obstacle,
            scale_factor,
            FT(beres_mech_weight),
            σ_x,
            L_system,
        ) : FT(0)
    steady_flux = steady_therm + steady_mech

    ntuple(
        n -> begin
            c_n = c[n]
            val = _beres_transient_bin(
                c_n, dc, u_heat, h, N2, N_source, Q0_sq, σ_x_sq, ν_min, dν,
                n_groups, boole_w, use_VG, gprofile, z_bot, span, m_max,
                n_h_avg, Δh_frac, scale_factor, launch_filter, n_bin_samples,
            )

            # Deposit the steady (ν=0) flux into the c≈0 bin only.
            val + steady_flux * FT(n == n_zero)
        end,
        Val(nc),
    )
end

# When an `NTuple`-valued field (here `gw_beres_gprofile`) is read inside a
# ClimaCore `@.` broadcast, ClimaCore wraps its per-node value in a
# `ClimaCore.Utilities.AutoBroadcaster{NTuple{ng,FT}}` (so arithmetic broadcasts
# elementwise over the tuple's subfields). The Beres-G shape functions
# (`_beres_gs`, `_beres_shape_factor_sq`) dispatch on the raw `NTuple`, so the
# wrapper does not match them and the call goes dynamic on the GPU -> heap alloc
# (`gpu_gc_pool_alloc`) -> InvalidIRError. Unwrap it back to the `NTuple` here.
# No-op for `nothing` (half-sine path) and for a plain `NTuple` (CPU / unit-test
# path, where no AutoBroadcaster is ever introduced).
@inline _beres_unwrap_gprofile(x::ClimaCore.Utilities.AutoBroadcaster) =
    getfield(x, :itr)
@inline _beres_unwrap_gprofile(x) = x

# Per-column launch spectrum gated by the activation flag: zeros for non-convecting
# columns (mirrors the old in-kernel `compute_beres_spectrum` gate), else the full
# Beres / Beres-G spectrum. Called from a `@.` broadcast over surface fields in
# `compute_beres_convective_heating!` (B0 is AD99-wavenumber independent, so it is
# precomputed once per column rather than rebuilt inside the kernel `ink` loop).
@inline function _beres_launch_spectrum_or_zero(
    active,
    c,
    u_heat,
    Q0,
    h,
    N_source,
    z_bot,
    span,
    gprofile_raw,
    w_obstacle,
    beres::BeresSourceParams,
    gw_ncval::Val{nc},
    shape_mode::Val = Val(beres.beres_shape_general);
    dz_level = zero(u_heat),
    launch_filter = nothing,
) where {nc}
    FT = typeof(u_heat)
    # Unwrap ClimaCore's AutoBroadcaster (if present) to the concrete NTuple.
    gprofile = _beres_unwrap_gprofile(gprofile_raw)
    if active > FT(0.5)
        return _beres_launch_spectrum(
            c, u_heat, Q0, h, N_source, z_bot, span, gprofile, w_obstacle,
            beres, gw_ncval, shape_mode; dz_level, launch_filter,
        )
    else
        return ntuple(_ -> FT(0), gw_ncval)
    end
end

# Total launched-flux magnitude Σ|B0[n]| of a precomputed launch spectrum B0
# (the `gw_launch_flux` diagnostic, now Beres-G / mechanical aware via B0).
@inline function _beres_b0_fluxmag(B0::NTuple{nc, FT}) where {nc, FT}
    s = FT(0)
    for n in 1:nc
        s += abs(B0[n])
    end
    return s
end
@inline _beres_b0_fluxmag(b::BeresB0) = _beres_b0_fluxmag(b.B0)

# Flux-weighted phase-speed centroid ⟨c⟩ = Σ c|B0| / Σ|B0| of a launch spectrum.
@inline function _beres_b0_centroid(B0::NTuple{nc, FT}, c) where {nc, FT}
    s0 = FT(0)
    s1 = FT(0)
    for n in 1:nc
        b = abs(B0[n])
        s0 += b
        s1 += c[n] * b
    end
    return s0 > eps(FT) ? s1 / s0 : FT(0)
end
@inline _beres_b0_centroid(b::BeresB0, c) = _beres_b0_centroid(b.B0, c)

# Band-resolved signed launched flux from B0 over a phase-speed band, for the
# standing band-resolved reporting requirement (low-`c` |c|≤c_band vs. rest, with
# eastward c>0 and westward c<0 kept separate by the sign of c). `want_low`
# selects |c|≤c_band; `want_east` selects c>0. Signed sum (keeps E/W direction).
@inline function _beres_b0_band(
    B0::NTuple{nc, FT},
    c,
    c_band,
    want_low::Bool,
    want_east::Bool,
) where {nc, FT}
    s = FT(0)
    for n in 1:nc
        cn = c[n]
        is_low = abs(cn) <= c_band
        is_east = cn > FT(0)
        if (is_low == want_low) && (is_east == want_east)
            s += B0[n]
        end
    end
    return s
end

# Signed mechanical (Ext 3) steady c≈0 flux β_mech for the diagnostic export.
# Zero unless the column is active AND the mechanical source is enabled.
@inline function _beres_mech_flux_value(
    active,
    u_heat,
    N_source,
    w_obstacle,
    beres::BeresSourceParams,
)
    FT = typeof(u_heat)
    (active > FT(0.5) && beres.beres_mechanical_source) || return FT(0)
    return _beres_mech_flux(
        u_heat,
        N_source,
        w_obstacle,
        FT(beres.beres_scale_factor),
        FT(beres.beres_mech_weight),
        FT(beres.σ_x),
        FT(beres.beres_L_system),
    )
end

#####
##### Five-way native band-flux diagnostics (CPU-side)
#####
# Computes, per native column, the launched band fluxes for five variants of the
# Beres source that share ALL inputs and differ only in flags/profile:
#   base  — half-sine shape (production baseline)
#   ext1  — sine transform of the gathered in-cloud profile
#   ext3  — half-sine + mechanical steady source
#   ext13 — both extensions
#   hs0   — half-sine profile pushed THROUGH the transform at the actual z_bot
#           (separates elevation interference from profile shape)
# Runs on the CPU at diagnostic time (results are per-column SCALARS, so the
# lat-lon remap afterwards is a harmless linear average — unlike remapping the
# nonlinear operator's INPUTS, which produced inconsistent profile/envelope
# pairs offline). The bin loop mirrors _beres_launch_spectrum exactly but
# accumulates band sums directly (no NTuple{nc} construction: that unrolled
# tuple is an hours-long CPU compile and a failed GPU compile).
# Limitation: all five variants use the model's one launch-filter wind range,
# gw_beres_wind_range, taken over the layer of the shape the run uses (from z_bot
# for a half-sine run, from max(z_bot, z_bot_floor) for a Beres-G run). The
# variants of the other shape therefore see that run's band, and the
# height-resolved filter of the transform variants uses the run's u profile on the
# Beres-G window. Rebuilding the band per variant would need the column winds,
# which this CPU-side pass does not read.

# Low-|c| band edge (m/s) separating the QBO-relevant slow waves from the rest.
const BERES_FIVEWAY_C_BAND = 15

# Copy of `b` with the two extension flags replaced (struct is immutable).
function _beres_with_flags(
    b::BeresSourceParams{FT},
    shape_general::Bool,
    mechanical::Bool,
) where {FT}
    return BeresSourceParams{FT}(;
        (n => getfield(b, n) for n in fieldnames(BeresSourceParams{FT}))...,
        beres_shape_general = shape_general,
        beres_mechanical_source = mechanical,
    )
end

# Analytic half-sine of unit peak (depth h, based at z_bot) sampled on the gather grid
# over [z_bot, z_top] (the hs0 variant; the caller multiplies it by Q0).
function _beres_hs_gprofile(z_bot::FT, z_top::FT, h::FT) where {FT}
    span = max(z_top - z_bot, eps(FT))
    dz_g = span / FT(N_BERES_PROFILE - 1)
    hh = max(h, eps(FT))
    raw = ntuple(Val(N_BERES_PROFILE)) do i
        zi = z_bot + (i - 1) * dz_g
        zi <= z_bot + hh ? max(sin(FT(π) * (zi - z_bot) / hh), FT(0)) : FT(0)
    end
    return raw
end

# Band-accumulating mirror of _beres_launch_spectrum: identical steady block and
# per-bin body, but sums (signed, |.|) per band instead of building NTuple{nc}.
# Bands: |c| <= c_band is "low"; c > 0 is "E" (the c = 0 steady bin counts as
# lowW). Returns (lowE, lowW, highE, highW, lowE_abs, lowW_abs, highE_abs,
# highW_abs, c0, c0_abs) as Float64. The trailing c0 pair isolates the
# stationary c = 0 bin (the steady/standing-wave deposit, also included in
# lowW — subtract for the transient slow-westward flux); it is zero when the
# phase-speed grid has no exact c = 0 bin.
function _beres_banded_spectrum(
    c,
    u_heat::FT,
    Q0::FT,
    h::FT,
    N_source::FT,
    z_bot::FT,
    span::FT,
    gprofile,
    w_obstacle::FT,
    beres::BeresSourceParams{FT},
    nc::Int,
    c_band::FT,
    dz_level::FT = zero(FT),
    n_bin_samples::Val = Val(N_BERES_BIN_SAMPLES),
    launch_filter = nothing,
) where {FT}
    (;
        σ_x, ν_min, ν_max, n_ν, beres_scale_factor, n_h_avg, Δh_frac,
        beres_steady_source, beres_steady_dc_frac, beres_L_system,
        beres_shape_general, beres_mechanical_source, beres_mech_weight,
    ) = beres
    scale_factor = FT(beres_scale_factor)
    L_system = FT(beres_L_system)
    use_VG = beres_shape_general && (gprofile !== nothing)
    m_max = dz_level > zero(dz_level) ? FT(π) / dz_level : FT(Inf)

    boole_w = (FT(7), FT(32), FT(12), FT(32), FT(7))
    dν = (ν_max - ν_min) / FT(n_ν - 1)
    n_groups = (n_ν - 1) ÷ 4
    N2 = N_source^2
    σ_x_sq = σ_x^2
    Q0_eff = use_VG ? one(Q0) : Q0   # as in `_beres_launch_spectrum`
    Q0_sq = Q0_eff^2

    dc = c[2] - c[1]
    cmax = -c[1]
    n_zero = clamp(round(Int, cmax / dc) + 1, 1, nc)
    has_c0_bin = abs(c[n_zero]) < FT(1e-6)
    steady_therm =
        (
            beres_steady_source && has_c0_bin &&
            !_beres_filter_removes(launch_filter, zero(FT))
        ) ?
        _beres_steady_flux(
            u_heat, N_source, h, Q0_eff, scale_factor, n_h_avg, Δh_frac, σ_x,
            L_system, FT(beres_steady_dc_frac), ν_min, use_VG,
            _beres_filter_profile(launch_filter, gprofile, zero(FT)),
            z_bot, span, m_max,
        ) : FT(0)
    steady_mech =
        (
            beres_mechanical_source && has_c0_bin &&
            !_beres_filter_band(launch_filter, zero(FT))
        ) ?
        _beres_mech_flux(
            u_heat, N_source, w_obstacle, scale_factor,
            FT(beres_mech_weight), σ_x, L_system,
        ) : FT(0)
    steady_flux = steady_therm + steady_mech

    sums = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for n in 1:nc
        c_n = c[n]
        val = _beres_transient_bin(
            c_n, dc, u_heat, h, N2, N_source, Q0_sq, σ_x_sq, ν_min, dν,
            n_groups, boole_w, use_VG, gprofile, z_bot, span, m_max,
            n_h_avg, Δh_frac, scale_factor, launch_filter, n_bin_samples,
        )
        v = Float64(val + steady_flux * FT(n == n_zero))
        band = (abs(c_n) <= c_band) ? (c_n > 0 ? 1 : 2) : (c_n > 0 ? 3 : 4)
        is_c0 = has_c0_bin && n == n_zero
        sums = ntuple(
            k ->
                k == band ? sums[k] + v :
                k == band + 4 ? sums[k] + abs(v) :
                (k == 9 && is_c0) ? sums[k] + v :
                ((k == 10 && is_c0) ? sums[k] + abs(v) : sums[k]),
            Val(10),
        )
    end
    return sums
end

# Memoized per-time computation of all five-way maps (CPU). Fills
# gw_fiveway_maps[] with a Dict short_name => Array shaped like parent(gw_Q0).
# Memo via module globals, NOT cache Refs: Refs in the cache change its type
# (forcing kernel recompiles), and per-simulation state is fine as globals for
# one-simulation-per-process drivers.
const _B5_MEMO_T = Ref{Any}(nothing)
const _B5_MEMO_MAPS = Ref{Any}(nothing)

function _compute_beres_fiveway!(p, time)
    nogw = p.non_orographic_gravity_wave
    _B5_MEMO_T[] == time && return _B5_MEMO_MAPS[]

    (; gw_beres_source, gw_c) = nogw
    beres = gw_beres_source
    FT = typeof(beres.σ_x)
    nc = length(gw_c)
    c_band = FT(BERES_FIVEWAY_C_BAND)

    active = Array(parent(nogw.gw_beres_active))
    Q0a = Array(parent(nogw.gw_Q0))
    ha = Array(parent(nogw.gw_h_heat))
    zba = Array(parent(nogw.gw_zbot))
    zta = Array(parent(nogw.gw_ztop))
    Na = Array(parent(nogw.gw_N_source))
    ua = Array(parent(nogw.gw_u_heat))   # envelope-mean zonal wind (Pass 2)
    wa = Array(parent(nogw.gw_w_obstacle))
    gpa = Array(parent(nogw.gw_beres_gprofile))  # (..., N_BERES_PROFILE, ...)
    dza = Array(parent(nogw.gw_beres_dz_level))
    wra = Array(parent(nogw.gw_beres_wind_range))  # (..., 6, ...)
    upa = Array(parent(nogw.gw_beres_u_profile))   # (..., N_BERES_PROFILE, ...)

    variants = (
        ("base", _beres_with_flags(beres, false, false), false),
        ("ext1", _beres_with_flags(beres, true, false), false),
        ("ext3", _beres_with_flags(beres, false, true), false),
        ("ext13", _beres_with_flags(beres, true, true), false),
        ("hs0", _beres_with_flags(beres, true, false), true),
    )
    bands = ("lowE", "lowW", "highE", "highW")
    maps = Dict{String, Array{FT}}()
    for (vname, _, _) in variants, (kb, bname) in enumerate(bands)
        maps["nogw_b5_$(vname)_$(bname)"] = zeros(FT, size(active))
        maps["nogw_b5_$(vname)_$(bname)_abs"] = zeros(FT, size(active))
    end
    for (vname, _, _) in variants
        maps["nogw_b5_$(vname)_c0"] = zeros(FT, size(active))
        maps["nogw_b5_$(vname)_c0_abs"] = zeros(FT, size(active))
    end

    # parent layout: 2D fields are (..., 1, ...) with one flat "F" axis; the
    # gprofile field has N_BERES_PROFILE entries on that axis. Identify it by
    # matching sizes against the 2D parent.
    sz2 = size(active)
    szg = size(gpa)
    fdim = findfirst(d -> szg[d] == N_BERES_PROFILE && sz2[d] == 1, 1:ndims(gpa))
    isnothing(fdim) && error("cannot locate gprofile component axis")

    for I in CartesianIndices(active)
        active[I] > FT(0.5) || continue
        z_bot = zba[I]
        z_top = zta[I]
        # The gathered profile lies on the window above the floor, as in the model.
        z_gather = _beres_gather_bottom(z_bot, z_top, beres.z_bot_floor)
        gprof = ntuple(
            k -> gpa[CartesianIndex(ntuple(
                d -> d == fdim ? k : I[d],
                Val(ndims(gpa)),
            ))],
            Val(N_BERES_PROFILE),
        )
        # The half-sine heating of the column, Q0·sin, for the hs0 variant (Beres-G
        # transforms the heating itself).
        gprof_hs = Q0a[I] .* _beres_hs_gprofile(z_bot, z_top, ha[I])
        # The zonal launch filter, as the model builds it for the u azimuth.
        fdim6 = findfirst(d -> size(wra, d) == 6 && sz2[d] == 1, 1:ndims(wra))
        component(A, fd, k) =
            A[CartesianIndex(ntuple(d -> d == fd ? k : I[d], Val(ndims(A))))]
        wind_range = ntuple(k -> component(wra, fdim6, k), Val(6))
        u_profile = ntuple(k -> component(upa, fdim, k), Val(N_BERES_PROFILE))
        for (vname, vparams, use_hs) in variants
            z_window = use_hs ? z_bot : z_gather
            uses_profile = vparams.beres_shape_general
            taper_points =
                dza[I] / max((z_top - z_window) / (N_BERES_PROFILE - 1), eps(FT))
            launch_filter = _beres_launch_filter(
                vparams, wind_range, uses_profile ? u_profile : nothing, Val(1),
                taper_points,
            )
            sums = _beres_banded_spectrum(
                gw_c, ua[I], Q0a[I], ha[I], Na[I], z_window, z_top - z_window,
                use_hs ? gprof_hs : gprof, wa[I], vparams, nc, c_band, dza[I],
                Val(N_BERES_BIN_SAMPLES), launch_filter,
            )
            for (kb, bname) in enumerate(bands)
                maps["nogw_b5_$(vname)_$(bname)"][I] = FT(sums[kb])
                maps["nogw_b5_$(vname)_$(bname)_abs"][I] = FT(sums[kb + 4])
            end
            maps["nogw_b5_$(vname)_c0"][I] = FT(sums[9])
            maps["nogw_b5_$(vname)_c0_abs"][I] = FT(sums[10])
        end
    end

    _B5_MEMO_MAPS[] = maps
    _B5_MEMO_T[] = time
    return maps
end
