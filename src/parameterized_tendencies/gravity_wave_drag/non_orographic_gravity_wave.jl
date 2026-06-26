#####
##### Non-orographic gravity wave parameterization
#####
using UnrolledUtilities
import ClimaCore.Domains
import ClimaCore.Meshes
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore.Operators as Operator
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT

"""
    _beres_latent_heating(mp, thp, ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno)

Canonical transport-free latent heating for one updraft (manuscript
`eq:edmf_heating`):

    Q_lat = (1/cp⁽ʲ⁾) Σ_p L_p R_p⁽ʲ⁾   [K/s]

Built from the CloudMicrophysics 1-moment **aggregated** phase-change tendencies
`dq_{lcl,icl,rai,sno}_dt` (the tested `_aggregate_tendencies`, returned by
`InstantaneousVerbose`). Relative to a vapor reference the liquid reservoir
(`lcl + rai`) carries `L_v` and the ice reservoir (`icl + sno`) carries `L_s`;
within the reservoir sums the same-phase mass transfers (autoconversion,
like-phase accretion) cancel, and liquid↔ice transfers net to `L_f = L_s − L_v`
automatically. So

    Q_lat = (1/cp⁽ʲ⁾) [ L_v·(dq_lcl_dt + dq_rai_dt) + L_s·(dq_icl_dt + dq_sno_dt) ]

is algebraically identical to the explicit per-process `Σ_p L_p R_p` form (the
explicit form is the independent oracle in `test_beres_latent_heating.jl`), but
cannot suffer a process-selection error: the aggregation is the single tested
source of truth for which terms enter and with what sign.

`cp⁽ʲ⁾` is the subdomain moist heat capacity `TD.cp_m(thp, q_tot, q_lcl, q_icl)`
(`main.tex:288`); `L_v = LH_v0`, `L_s = LH_s0` are reference-`T₀` constants
(matching the manuscript's constant `L_p`). Requires 1-moment microphysics +
PrognosticEDMFX (gated at construction). `@inline` + scalar args so it composes
inside a GPU broadcast with the `InstantaneousVerbose` source dead-code-eliminated
down to the four aggregated tendencies.
"""
@inline function _beres_latent_heating(
    mp,
    thp,
    ρ,
    T,
    q_tot,
    q_lcl,
    q_icl,
    q_rai,
    q_sno,
)
    src = BMT.bulk_microphysics_tendencies(
        BMT.InstantaneousVerbose(),
        BMT.Microphysics1Moment(),
        mp,
        thp,
        ρ,
        T,
        q_tot,
        q_lcl,
        q_icl,
        q_rai,
        q_sno,
    )
    Lv = TD.Parameters.LH_v0(thp)
    Ls = TD.Parameters.LH_s0(thp)
    cpʲ = TD.cp_m(thp, q_tot, q_lcl, q_icl)
    return (
        Lv * (src.dq_lcl_dt + src.dq_rai_dt) +
        Ls * (src.dq_icl_dt + src.dq_sno_dt)
    ) / cpʲ
end

non_orographic_gravity_wave_cache(Y, atmos::AtmosModel) =
    non_orographic_gravity_wave_cache(Y, atmos.non_orographic_gravity_wave)

non_orographic_gravity_wave_cache(Y, ::Nothing) = (;)

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
            ᶜdTdz = similar(Y.c.ρ),
            source_ρ_z_u_v_level,
            source_level = similar(Fields.level(Y.c.ρ, 1)),
            damp_level = similar(Fields.level(Y.c.ρ, 1)),
            ᶜlevel,
            u_waveforcing = similar(Y.c.ρ),
            v_waveforcing = similar(Y.c.ρ),
            u_waveforcing_top = similar(Fields.level(Y.c.ρ, 1)),
            v_waveforcing_top = similar(Fields.level(Y.c.ρ, 1)),
            uforcing = zero(Y.c.ρ),
            vforcing = zero(Y.c.ρ),
            gw_ncval = Val(nc),
            # Beres fields (always allocated)
            gw_Q0 = similar(Fields.level(Y.c.ρ, 1)),
            gw_h_heat = similar(Fields.level(Y.c.ρ, 1)),
            gw_u_heat = similar(Fields.level(Y.c.ρ, 1)),
            gw_v_heat = similar(Fields.level(Y.c.ρ, 1)),
            gw_N_source = similar(Fields.level(Y.c.ρ, 1)),
            gw_beres_active = similar(Fields.level(Y.c.ρ, 1)),
            gw_beres_source = gw.beres_source,
            gw_zbot = similar(Fields.level(Y.c.ρ, 1)),
            gw_ztop = similar(Fields.level(Y.c.ρ, 1)),
            gw_Q_conv = similar(Y.c.ρ),
            gw_Q_conv_ic = similar(Y.c.ρ),
            # Native half-sine source profile Q0·sin(π(z−z_bot)/h) (3D), and the
            # launched-spectrum summaries (2D): total launched flux magnitude and
            # the flux-weighted phase-speed centroid. Emitted as diagnostics so
            # the half-sine vs Q_conv_ic comparison is a single-field remap
            # (apples-to-apples), not an offline reconstruction from three
            # independently-remapped 2D fields (Q0, z_bot, h).
            gw_halfsine = similar(Y.c.ρ),
            gw_launch_flux = similar(Fields.level(Y.c.ρ, 1)),
            gw_c_centroid = similar(Fields.level(Y.c.ρ, 1)),
            gw_a_cover = similar(Fields.level(Y.c.ρ, 1)),
            gw_reduce_result = similar(
                Fields.level(Y.c.ρ, 1),
                Tuple{FT, FT, FT, FT, FT, FT, FT},
            ),
            # Beres launch-level state (ρ, z, u, v, level) — per-column,
            # populated from gw_ztop in compute_tendency!. Kernel uses this
            # in MODE == :beres; AD99 path ignores it.
            beres_source_ρ_z_u_v_level =
            similar(Fields.level(Y.c.ρ, 1), Tuple{FT, FT, FT, FT, FT}),
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
            ᶜdTdz = similar(Y.c.ρ),
            source_p_ρ_z_u_v_level,
            source_level = similar(Fields.level(Y.c.ρ, 1)),
            damp_level = similar(Fields.level(Y.c.ρ, 1)),
            ᶜlevel,
            u_waveforcing = similar(Y.c.ρ),
            v_waveforcing = similar(Y.c.ρ),
            u_waveforcing_top = similar(Fields.level(Y.c.ρ, 1)),
            v_waveforcing_top = similar(Fields.level(Y.c.ρ, 1)),
            uforcing = zero(Y.c.ρ),
            vforcing = zero(Y.c.ρ),
            gw_ncval = Val(nc),
            # Beres fields (always allocated; gw_beres_source is nothing or BeresSourceParams)
            gw_Q0 = similar(Fields.level(Y.c.ρ, 1)),
            gw_h_heat = similar(Fields.level(Y.c.ρ, 1)),
            gw_u_heat = similar(Fields.level(Y.c.ρ, 1)),
            gw_v_heat = similar(Fields.level(Y.c.ρ, 1)),
            gw_N_source = similar(Fields.level(Y.c.ρ, 1)),
            gw_beres_active = similar(Fields.level(Y.c.ρ, 1)),
            gw_beres_source = gw.beres_source,
            gw_zbot = similar(Fields.level(Y.c.ρ, 1)),
            gw_ztop = similar(Fields.level(Y.c.ρ, 1)),
            gw_Q_conv = similar(Y.c.ρ),
            gw_Q_conv_ic = similar(Y.c.ρ),
            # Native half-sine source profile Q0·sin(π(z−z_bot)/h) (3D), and the
            # launched-spectrum summaries (2D): total launched flux magnitude and
            # the flux-weighted phase-speed centroid. Emitted as diagnostics so
            # the half-sine vs Q_conv_ic comparison is a single-field remap
            # (apples-to-apples), not an offline reconstruction from three
            # independently-remapped 2D fields (Q0, z_bot, h).
            gw_halfsine = similar(Y.c.ρ),
            gw_launch_flux = similar(Fields.level(Y.c.ρ, 1)),
            gw_c_centroid = similar(Fields.level(Y.c.ρ, 1)),
            gw_a_cover = similar(Fields.level(Y.c.ρ, 1)),
            gw_reduce_result = similar(
                Fields.level(Y.c.ρ, 1),
                Tuple{FT, FT, FT, FT, FT, FT, FT},
            ),
            # Beres launch-level state (ρ, z, u, v, level) — per-column,
            # populated from gw_ztop in compute_tendency!. Kernel uses this
            # in MODE == :beres; AD99 path ignores it.
            beres_source_ρ_z_u_v_level =
            similar(Fields.level(Y.c.ρ, 1), Tuple{FT, FT, FT, FT, FT}),
        )
    else
        error("Only sphere and columns are supported")
    end
end

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
        ᶜdTdz,
        ᶜbuoyancy_frequency,
        source_level,
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

    # compute buoyancy frequency
    ᶜdTdz .= Geometry.WVector.(ᶜgradᵥ.(ᶠinterp.(ᶜT))).components.data.:1
    ᶜcp_m = @. lazy(TD.cp_m(thermo_params, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice))

    @. ᶜbuoyancy_frequency =
        (grav / ᶜT) * (ᶜdTdz + grav / ᶜcp_m)
    ᶜbuoyancy_frequency = @. ifelse(
        ᶜbuoyancy_frequency < FT(2.5e-5),
        FT(sqrt(2.5e-5)),
        sqrt(abs(ᶜbuoyancy_frequency)),
    ) # to avoid small numbers

    # prepare physical uv input variables for gravity_wave_forcing()
    ᶜu = Geometry.UVVector.(Y.c.uₕ).components.data.:1
    ᶜv = Geometry.UVVector.(Y.c.uₕ).components.data.:2

    if iscolumn(axes(Y.c))
        # source level: the index of the level that is closest to the source height
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

        fill!(damp_level, Spaces.nlevels(axes(ᶜz)))

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
    else
        error("Only sphere and columns are supported")
    end

    ᶜu = Geometry.UVVector.(Y.c.uₕ).components.data.:1
    ᶜv = Geometry.UVVector.(Y.c.uₕ).components.data.:2

    # Compute Beres convective heating if enabled. Pass the local
    # ᶜbuoyancy_frequency (already √ N²) so the routine doesn't have to
    # re-derive N from the cache (which still holds N² in-place from line 206).
    if !isnothing(p.non_orographic_gravity_wave.gw_beres_source)
        compute_beres_convective_heating!(Y, p, ᶜbuoyancy_frequency)

        (; gw_ztop, beres_source_ρ_z_u_v_level) = p.non_orographic_gravity_wave
        beres_input = Base.Broadcast.broadcasted(
            tuple, ᶜρ, ᶜz, ᶜu, ᶜv, ᶜlevel, gw_ztop,
        )
        Operators.column_reduce!(
            beres_source_ρ_z_u_v_level,
            beres_input;
            init = (FT(0), FT(-Inf), FT(0), FT(0), FT(0)),
        ) do (ρ_prev, z_prev, u_prev, v_prev, level_prev),
        (ρ, z, u, v, level, ztop)
            # Take the first level (going up) whose z meets/exceeds ztop.
            if (z_prev < ztop) && (z >= ztop)
                return (ρ, z, u, v, level)
            else
                return (ρ_prev, z_prev, u_prev, v_prev, level_prev)
            end
        end
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

end

"""
    compute_beres_convective_heating!(Y, p, ᶜN)

Extract per-column convective source properties from EDMF for the Beres (2004) spectrum: Q0 (half-sine amplitude), h (depth), u_heat/v_heat (mean wind), N_source, a_cover (envelope-mean updraft area, the deposition factor), and the beres_active flag.

Heating fields:

  - `gw_Q_conv` — grid-mean heating (carries area fraction aʲ); used for envelope detection and activation gating, calibrated to grid-mean magnitudes.
  - `gw_Q_conv_ic` — in-cloud heating (no area factor); the amplitude convention Beres' linear theory expects. Q0 is built from this, then deposited flux is diluted by `a_cover` (see `waveforcing_column_accumulate!`), giving flux ∝ ā·Q_ic².

Envelope `[z_bot, z_top]`: `z_top` = highest level with updraft area > `1e-3`; `z_bot` = lowest level with `z ≥ z_bot_floor` AND `Q_conv > z_bot_Q_threshold` (floor needed to skip the EDMF PBL/dry-thermal signal below ~1 km).

`ᶜN` is N (s⁻¹), passed in because the cached `ᶜbuoyancy_frequency` still holds N².
"""
function compute_beres_convective_heating!(Y, p, ᶜN)
    (; turbconv_model) = p.atmos
    n_updrafts = n_mass_flux_subdomains(turbconv_model)

    if n_updrafts == 0
        # No EDMF — Beres inactive everywhere
        p.non_orographic_gravity_wave.gw_beres_active .= 0
        p.non_orographic_gravity_wave.gw_a_cover .= 0
        p.non_orographic_gravity_wave.gw_Q_conv_ic .= 0
        p.non_orographic_gravity_wave.gw_halfsine .= 0
        p.non_orographic_gravity_wave.gw_launch_flux .= 0
        p.non_orographic_gravity_wave.gw_c_centroid .= 0
        return
    end

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
    ) = p.non_orographic_gravity_wave

    FT = Spaces.undertype(axes(Y.c))
    ᶜρ = Y.c.ρ
    ᶜz = Fields.coordinate_field(Y.c).z

    # Compute DSE-based mass-flux Q₁ (Yanai apparent heat source):
    #   ρ·Q₁ ≈ -∂/∂z [Mᶜ·(s_c − s̄)]  where s = cp_d·T + g·z
    # MSE (h = cp·T + gz + Lv·q) is conserved under condensation, so its
    # mass-flux divergence gives Q₁−Q₂, masking the latent-heating signal.
    # DSE works because Tʲ is saturation-adjusted: the warming from
    # condensation along the parcel trajectory is already encoded in Tʲ,
    # so cp_d·(Tʲ − T̄) carries the cumulative latent heat release.
    # The g·z terms cancel in (sʲ − s̄), leaving cp_d·(Tʲ − T̄).
    ᶜQ_conv = p.scratch.ᶜtemp_scalar_2
    ᶜQ_conv .= FT(0)
    cp_d = FT(CAP.cp_d(p.params))

    # Scratch fields for face velocity anomaly and cell-center scalars
    ᶠu³_diff = p.scratch.ᶠtemp_CT3
    ᶜa_scalar = p.scratch.ᶜtemp_scalar
    # In-cloud DSE anomaly (no area factor) and ρa-weighting denominator
    ᶜa_scalar_ic = p.scratch.ᶜtemp_scalar_5
    ᶜρa_sum = p.scratch.ᶜtemp_scalar_6

    # For DiagnosticEDMFX, ρa is in p.precomputed
    # For PrognosticEDMFX, ρa is in Y.c.sgsʲs.:($j)
    has_prognostic_sgs =
        hasproperty(Y.c, :sgsʲs) && n_updrafts > 0
    if !has_prognostic_sgs && haskey(p.precomputed, :ᶜρaʲs)
        ᶜρaʲs_all = p.precomputed.ᶜρaʲs
    end

    # Compute Q_conv (grid-mean), Q_conv_ic (in-cloud) and total area fraction
    # in one pass.
    ᶜa_up = p.scratch.ᶜtemp_scalar_4
    ᶜa_up .= FT(0)
    gw_Q_conv_ic .= FT(0)
    ᶜρa_sum .= FT(0)
    for j in 1:n_updrafts
        # Velocity anomaly at faces (contravariant)
        @. ᶠu³_diff = ᶠu³ʲs.:($$j) - ᶠu³

        ᶜρaʲ = if has_prognostic_sgs
            Y.c.sgsʲs.:($j).ρa
        else
            ᶜρaʲs_all.:($j)
        end

        # In-cloud DSE anomaly: cp_d·(Tʲ − T̄), no area factor
        @. ᶜa_scalar_ic =
            ifelse(ᶜρʲs.:($$j) > eps(FT), cp_d * (ᶜTʲs.:($$j) - ᶜT), FT(0))

        # DSE anomaly × area fraction: cp_d·(Tʲ − T̄) · (ρaʲ/ρʲ)
        @. ᶜa_scalar =
            ifelse(
                ᶜρʲs.:($$j) > eps(FT),
                ᶜa_scalar_ic * (ᶜρaʲ / ᶜρʲs.:($$j)),
                FT(0),
            )

        # Divergence: -∂(ρʲ·u³_diff·a_scalar)/∂z on cell centers
        # Val(:none) = centered differencing (dt unused)
        vtt = vertical_transport(
            ᶜρʲs.:($j),
            ᶠu³_diff,
            ᶜa_scalar,
            FT(1),
            Val(:none),
        )
        # Convert from W/m³ to heating rate K/s (grid-mean Q₁)
        @. ᶜQ_conv += vtt / (ᶜρ * cp_d)

        # In-cloud heating of draft j: same mass-flux divergence WITHOUT the
        # area factor, normalized by the draft (not grid-mean) density:
        #   Q_icʲ = −(1/(ρʲ·c_p)) ∂z[ρʲ·(wʲ−w̄)·c_p·(Tʲ−T̄)]
        # Accumulated ρaʲ-weighted so that for M>1 drafts the result is the
        # conditional mean over the total draft area (÷ Σρaʲ below).
        vtt_ic = vertical_transport(
            ᶜρʲs.:($j),
            ᶠu³_diff,
            ᶜa_scalar_ic,
            FT(1),
            Val(:none),
        )
        @. gw_Q_conv_ic += ifelse(
            ᶜρʲs.:($$j) > eps(FT),
            max(ᶜρaʲ, FT(0)) * vtt_ic / (ᶜρʲs.:($$j) * cp_d),
            FT(0),
        )
        @. ᶜρa_sum += max(ᶜρaʲ, FT(0))

        # Total updraft area fraction (used to set z_top and a_cover below)
        @. ᶜa_up += ifelse(ᶜρʲs.:($$j) > eps(FT), ᶜρaʲ / ᶜρʲs.:($$j), FT(0))
    end

    # Finalize the ρa-weighted in-cloud mean over drafts
    @. gw_Q_conv_ic =
        ifelse(ᶜρa_sum > eps(FT), gw_Q_conv_ic / ᶜρa_sum, FT(0))

    # Canonical transport-free latent-heating override (manuscript
    # eq:edmf_heating). When enabled, replace the in-cloud DSE flux-divergence Q₁
    # just computed in gw_Q_conv_ic with Q_lat = (1/cp⁽ʲ⁾) Σ_p L_p R_p⁽ʲ⁾, the
    # ρaʲ-weighted in-cloud mean of the per-draft latent heating. The grid-mean
    # gw_Q_conv (Q₁) above is left untouched — it still drives activation gating
    # and the z_bot threshold, which are calibrated to grid-mean Q₁ magnitudes.
    # The heating_latent flag is gated to 1M + PrognosticEDMFX at construction, so
    # the prognostic condensate fields and ᶜq_tot_nonnegʲs accessed here exist.
    if gw_beres_source.heating_latent
        mp = CAP.microphysics_1m_params(p.params)
        thp = CAP.thermodynamics_params(p.params)
        ᶜq_tot_nonnegʲs = p.precomputed.ᶜq_tot_nonnegʲs
        gw_Q_conv_ic .= FT(0)
        ᶜρa_sum .= FT(0)
        for j in 1:n_updrafts
            ᶜρaʲ = Y.c.sgsʲs.:($j).ρa
            ᶜρʲ = ᶜρʲs.:($j)
            ᶜTʲ = ᶜTʲs.:($j)
            ᶜq_totʲ = ᶜq_tot_nonnegʲs.:($j)
            ᶜq_lclʲ = Y.c.sgsʲs.:($j).q_lcl
            ᶜq_iclʲ = Y.c.sgsʲs.:($j).q_icl
            ᶜq_raiʲ = Y.c.sgsʲs.:($j).q_rai
            ᶜq_snoʲ = Y.c.sgsʲs.:($j).q_sno
            @. gw_Q_conv_ic += ifelse(
                ᶜρʲ > eps(FT),
                max(ᶜρaʲ, FT(0)) * _beres_latent_heating(
                    mp,
                    thp,
                    ᶜρʲ,
                    ᶜTʲ,
                    ᶜq_totʲ,
                    ᶜq_lclʲ,
                    ᶜq_iclʲ,
                    ᶜq_raiʲ,
                    ᶜq_snoʲ,
                ),
                FT(0),
            )
            @. ᶜρa_sum += max(ᶜρaʲ, FT(0))
        end
        @. gw_Q_conv_ic =
            ifelse(ᶜρa_sum > eps(FT), gw_Q_conv_ic / ᶜρa_sum, FT(0))
    end

    # Persist Q_conv into cache before scratch field is reused
    @. gw_Q_conv = ᶜQ_conv

    ᶜu = Geometry.UVVector.(Y.c.uₕ).components.data.:1
    ᶜv = Geometry.UVVector.(Y.c.uₕ).components.data.:2

    # Pass 1: set the convective envelope [z_bot, z_top] and depth h.
    #   moment_matched (the only active mode; moment_envelope is hardwired true):
    #     fit a half-sine to the in-cloud heating Q_conv_ic by its 1st/2nd moments
    #     — centroid z_c and spread σ. A half-sine of depth h has variance
    #     h²·(π²−8)/(4π²), so h = σ/√(that); z_bot = z_c − h/2, z_top = z_c + h/2,
    #     and Q0 = (π/2)·∫max(Q_ic,0)dz / h. This sits the envelope on the actual
    #     heating.
    #   area_threshold (DEPRECATED, retained gated `false`): z_top = highest level
    #     with updraft area > a_thresh (plume top); z_bot = lowest level above
    #     z_bot_floor where grid-mean Q_conv > z_bot_Q_threshold. It was often
    #     narrow and misplaced relative to the in-cloud heating peak.
    # Both leave gw_u_heat = z_bot, gw_v_heat = z_top, gw_h_heat = h for Pass 2.
    result_field = gw_reduce_result
    if gw_beres_source.moment_envelope
        ᶜΔz_m = Fields.Δz_field(axes(Y.c))
        input1 = Base.Broadcast.broadcasted(tuple, ᶜz, gw_Q_conv_ic, ᶜΔz_m)
        reduce_init = (FT(0), FT(0), FT(0), FT(0), FT(0), FT(0), FT(0))
        let _z_floor = gw_beres_source.z_bot_floor, _zero = FT(0)
            # Accumulate (∫Q⁺dz, ∫z·Q⁺dz, ∫z²·Q⁺dz) over z ≥ z_bot_floor.
            # NB: capture only Float scalars (_z_floor, _zero) here — referencing
            # the type `FT` inside the reducer closure makes it non-isbits
            # (carries `Type{FT}`) and fails GPU compilation. The area-mode
            # closure below follows the same rule.
            Operators.column_reduce!(
                result_field,
                input1;
                init = reduce_init,
            ) do (I0p, I1p, I2p, _4, _5, _6, _7), (z, Q_ic, dz)
                qpdz = ifelse(z >= _z_floor, max(Q_ic, _zero) * dz, _zero)
                return (
                    I0p + qpdz,
                    I1p + z * qpdz,
                    I2p + z * z * qpdz,
                    _4,
                    _5,
                    _6,
                    _7,
                )
            end
        end
        h_coef = FT(1) / sqrt((FT(π)^2 - FT(8)) / (FT(4) * FT(π)^2))  # ≈ 4.595
        # z_c (held temporarily in gw_zbot)
        @. gw_zbot = ifelse(
            result_field.:1 > eps(FT),
            result_field.:2 / result_field.:1,
            FT(0),
        )
        # h = h_coef · σ,   σ² = ∫z²Q⁺/∫Q⁺ − z_c²
        @. gw_h_heat = ifelse(
            result_field.:1 > eps(FT),
            h_coef *
            sqrt(max(result_field.:3 / result_field.:1 - gw_zbot^2, FT(0))),
            FT(0),
        )
        @. gw_ztop = gw_zbot + gw_h_heat / 2              # z_top = z_c + h/2
        @. gw_zbot = max(gw_zbot - gw_h_heat / 2, FT(0))  # z_bot = z_c − h/2
        # Clamp z_top to the domain top. The area envelope's z_top is always a
        # grid level, so the launch-level search (a column reduce that returns
        # the first level with z ≥ z_top) always finds one. The moment z_top is
        # unbounded above (a tiny, high-noise column can push it past the top
        # cell), which would leave source_level = 0 (invalid) and crash. Keep it
        # grid-bounded and z_bot ≤ z_top.
        z_dom_top = maximum(ᶜz)
        @. gw_ztop = min(gw_ztop, z_dom_top)
        @. gw_zbot = min(gw_zbot, gw_ztop)
        # Matched-half-sine amplitude Q0 = (π/2)·∫max(Q_ic,0)dz / h
        @. gw_Q0 = ifelse(
            (gw_h_heat > eps(FT)) & (result_field.:1 > FT(0)),
            max(FT(π) / FT(2) * result_field.:1 / gw_h_heat, FT(0)),
            FT(0),
        )
        @. gw_u_heat = gw_zbot
        @. gw_v_heat = gw_ztop
    else
        # DEPRECATED: area_threshold envelope path. moment_matched (above) is the
        # only mode reachable from config (moment_envelope is hardwired true). This
        # branch is retained for reference / direct struct construction in tests and
        # is no longer exercised by any production run.
        input1 = Base.Broadcast.broadcasted(tuple, ᶜz, ᶜa_up, ᶜQ_conv)
        reduce_init = (FT(-Inf), FT(Inf), FT(0), FT(0), FT(0), FT(0), FT(0))
        let _a_thresh = FT(1e-3), _Q_thresh = gw_beres_source.z_bot_Q_threshold,
            _z_floor = gw_beres_source.z_bot_floor

            Operators.column_reduce!(
                result_field,
                input1;
                init = reduce_init,
            ) do (z_top_prev, z_bot_prev, _3, _4, _5, _6, _7), (z, a, Q)
                z_top = ifelse(a > _a_thresh, max(z_top_prev, z), z_top_prev)
                z_bot = ifelse(
                    (z >= _z_floor) & (Q > _Q_thresh),
                    min(z_bot_prev, z),
                    z_bot_prev,
                )
                return (z_top, z_bot, _3, _4, _5, _6, _7)
            end
        end

        @. gw_v_heat = ifelse(
            isfinite(result_field.:1) & isfinite(result_field.:2),
            result_field.:1,
            FT(0),
        )
        @. gw_u_heat = ifelse(
            isfinite(result_field.:1) & isfinite(result_field.:2),
            result_field.:2,
            FT(0),
        )
        @. gw_h_heat = max(gw_v_heat - gw_u_heat, FT(0))
    end
    @. gw_h_heat = ifelse(isnan(gw_h_heat) | isinf(gw_h_heat), FT(0), gw_h_heat)

    # Persist zbot/ztop before gw_u_heat/gw_v_heat get overwritten by mean winds
    @. gw_zbot = gw_u_heat
    @. gw_ztop = gw_v_heat

    # Pass 2: within [z_bot, z_top], compute:
    #   - Q₀ integrals: Σ(Q · Δz) for the grid-mean (gating) and in-cloud
    #     (spectrum amplitude) heatings
    #   - Mass-weighted mean wind (u, v), buoyancy frequency (N), and
    #     updraft area fraction (a_cover, the Beres deposition factor)
    # All quantities drawn from the same physical envelope — the continuous
    # convective column from cloud base to plume top.
    # `ᶜN` is supplied by the caller (already √ N², s⁻¹).
    ᶜΔz = Fields.Δz_field(axes(Y.c))
    # Precompute 3D envelope mask (2D z_bot/z_top broadcast over column)
    ᶜz_bot = gw_u_heat  # 2D field, broadcasts over column via @.
    ᶜz_top = gw_v_heat
    ᶜin_env = p.scratch.ᶜtemp_scalar_3  # free scratch (no longer holds ᶜw_up)
    @. ᶜin_env = ifelse((ᶜz >= ᶜz_bot) & (ᶜz <= ᶜz_top), FT(1), FT(0))
    input2 = Base.Broadcast.broadcasted(
        tuple,
        ᶜQ_conv,
        ᶜu,
        ᶜv,
        ᶜN,
        ᶜρ,
        ᶜΔz,
        ᶜin_env,
        gw_Q_conv_ic,
        ᶜa_up,
    )
    # Accumulator: (Q_integral, u_sum, v_sum, N_sum, mass_sum, Qic_integral, a_sum)
    _zero = FT(0)
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
    (Q, u, v, N, ρ, dz, env, Q_ic, a_up)
        active = env > _half
        ρdz = ρ * dz
        Q_int = ifelse(active, Q_int_prev + Q * dz, Q_int_prev)
        u_sum = ifelse(active, u_sum_prev + u * ρdz, u_sum_prev)
        v_sum = ifelse(active, v_sum_prev + v * ρdz, v_sum_prev)
        N_sum = ifelse(active, N_sum_prev + N * ρdz, N_sum_prev)
        m_sum = ifelse(active, m_sum_prev + ρdz, m_sum_prev)
        Qic_int = ifelse(active, Qic_int_prev + Q_ic * dz, Qic_int_prev)
        a_sum = ifelse(active, a_sum_prev + a_up * ρdz, a_sum_prev)
        return (Q_int, u_sum, v_sum, N_sum, m_sum, Qic_int, a_sum)
    end

    # Unpack Pass 2 results
    mass_sum = result_field.:5
    @. gw_u_heat = ifelse(mass_sum > eps(FT), result_field.:2 / mass_sum, FT(0))
    @. gw_v_heat = ifelse(mass_sum > eps(FT), result_field.:3 / mass_sum, FT(0))
    @. gw_N_source =
        ifelse(mass_sum > eps(FT), result_field.:4 / mass_sum, FT(0.01))

    # Coverage: mass-weighted envelope mean of the updraft area fraction,
    # clamped to [0, 1]. Used as the Beres deposition (intermittency) factor.
    @. gw_a_cover = ifelse(
        mass_sum > eps(FT),
        min(max(result_field.:7 / mass_sum, FT(0)), FT(1)),
        FT(0),
    )

    # Spectrum amplitude. In moment_matched mode gw_Q0 was already set from the
    # column moments in Pass 1; in area_threshold mode set it here from the
    # IN-CLOUD heating integral over [z_bot, z_top]: Q₀ = (π/2)·Σ(Q_ic·Δz)/h.
    # DEPRECATED: this block belongs to the area_threshold path (see above) and is
    # never taken in production (moment_envelope is hardwired true).
    if !gw_beres_source.moment_envelope
        @. gw_Q0 = result_field.:6
        @. gw_Q0 = ifelse(
            gw_h_heat > FT(0),
            max(FT(π) / FT(2) * gw_Q0 / gw_h_heat, FT(0)),
            FT(0),
        )
    end
    @. gw_Q0 = ifelse(isnan(gw_Q0) | isinf(gw_Q0), FT(0), gw_Q0)

    # Set beres_active flag: GRID-MEAN amplitude above threshold AND heating
    # depth above minimum. Gating stays on the grid-mean Q₁ (slot 1) because
    # `beres_Q0_threshold` is calibrated to grid-mean magnitudes; only the
    # launch amplitude uses the in-cloud convention.
    Q0_threshold = gw_beres_source.Q0_threshold
    h_heat_min = gw_beres_source.h_heat_min
    @. gw_beres_active = ifelse(
        (
            max(
                FT(π) / FT(2) * result_field.:1 / max(gw_h_heat, eps(FT)),
                FT(0),
            ) > Q0_threshold
        ) & (gw_h_heat > h_heat_min),
        FT(1),
        FT(0),
    )

    # --- Native source-shape & launched-spectrum diagnostics (Q2 resolution) ---
    # These are verification-only outputs (the half-sine source profile and the
    # launched-spectrum summaries); they do NOT feed the forcing kernel. Skip them
    # entirely unless detailed (debug/manuscript) diagnostics are requested — a
    # production GCM run only needs the net drag. The `nogw_halfsine` /
    # `nogw_launch_flux` / `nogw_c_centroid` diagnostics are gated on the same flag.
    gw_beres_source.detailed_diagnostics || return
    # Build the launched half-sine Q0·sin(π(z−z_bot)/h) over the heating depth h
    # natively, IN the column where (Q0, z_bot, h) are mutually consistent, so
    # the offline comparison to Q_conv_ic is a single linear remap of one field
    # rather than a nonlinear reconstruction from three independently-remapped
    # 2D fields. Gated by beres_active; zero elsewhere.
    # Condition must stay in the 3D (column) broadcast space: `&` cannot combine
    # a pure-2D level-space bool (gw_beres_active>0.5) with a 3D one (ᶜz≥gw_zbot)
    # — ClimaCore rejects the mixed DataF/VF spaces. So the envelope range (both
    # operands 3D) is the mask, and the 2D gates fold into the multiplied value:
    # gw_beres_active (0/1) zeroes inactive columns, and max(h,eps) guards the
    # divide (h=0 ⇒ inactive ⇒ value already 0).
    @. gw_halfsine = ifelse(
        (ᶜz >= gw_zbot) & (ᶜz <= gw_zbot + gw_h_heat),
        gw_beres_active *
        gw_Q0 *
        sin(FT(π) * (ᶜz - gw_zbot) / max(gw_h_heat, eps(FT))),
        FT(0),
    )

    # Launched-spectrum summaries from the SAME per-column source state the
    # forcing kernel uses (ink-independent for Beres): total launched flux
    # magnitude Σ|B0| (× a_cover deposition factor) and the flux-weighted
    # phase-speed centroid ⟨c⟩ = Σ c|B0| / Σ|B0|, both in the zonal (gw_u_heat)
    # direction. Two scalar broadcasts (the spectrum is cheap for inactive
    # columns and only ~tens of columns are active). `gw_c` (an NTuple) and
    # `gw_ncval` are captured in the closure body — NOT passed as broadcast
    # args — because broadcasting a Tuple would zip it element-wise instead of
    # treating it as a scalar grid.
    let beres = gw_beres_source, cgrid = gw_c, ncv = gw_ncval
        flux_mag(active, u_src, Q0, h, N) =
            _beres_launch_summary(beres, active, cgrid, u_src, Q0, h, N, ncv)[1]
        centroid(active, u_src, Q0, h, N) = begin
            (s0, s1) =
                _beres_launch_summary(beres, active, cgrid, u_src, Q0, h, N, ncv)
            ifelse(s0 > eps(typeof(s0)), s1 / s0, zero(s0))
        end
        @. gw_launch_flux = ifelse(
            gw_beres_active > FT(0.5),
            gw_a_cover *
            flux_mag(gw_beres_active, gw_u_heat, gw_Q0, gw_h_heat, gw_N_source),
            FT(0),
        )
        @. gw_c_centroid = ifelse(
            gw_beres_active > FT(0.5),
            centroid(gw_beres_active, gw_u_heat, gw_Q0, gw_h_heat, gw_N_source),
            FT(0),
        )
    end
end

non_orographic_gravity_wave_apply_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

function non_orographic_gravity_wave_apply_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::NonOrographicGravityWave,
)

    (; uforcing, vforcing) = p.non_orographic_gravity_wave
    FT = Spaces.undertype(axes(Y.c))

    # Constrain forcing (same limit as OGW: 3e-3 m/s²)
    # Use ifelse to also catch NaN (IEEE max/min propagate NaN)
    @. uforcing = ifelse(
        isnan(uforcing) | isinf(uforcing),
        FT(0),
        max(FT(-3e-3), min(FT(3e-3), uforcing)),
    )
    @. vforcing = ifelse(
        isnan(vforcing) | isinf(vforcing),
        FT(0),
        max(FT(-3e-3), min(FT(3e-3), vforcing)),
    )

    @. Yₜ.c.uₕ +=
        Geometry.Covariant12Vector.(Geometry.UVVector.(uforcing, vforcing))

end

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
        gw_beres_active,
        gw_Q0,
        gw_h_heat,
        gw_u_heat,
        gw_v_heat,
        gw_N_source,
        gw_a_cover,
        gw_beres_source,
        beres_source_ρ_z_u_v_level,
    ) = p.non_orographic_gravity_wave

    # Beres launch-level state (per-column). Always extracted — even in
    # AD99-only mode the values flow through the kernel input tuple, but the
    # MODE == :ad99 branch ignores them.
    beres_ρ_source = beres_source_ρ_z_u_v_level.:1
    beres_u_source = beres_source_ρ_z_u_v_level.:3
    beres_v_source = beres_source_ρ_z_u_v_level.:4
    beres_source_level = beres_source_ρ_z_u_v_level.:5

    # Temporary scratch fields for shifting levels up
    ᶜρ_p1 = p.scratch.ᶜtemp_scalar
    ᶜz_p1 = p.scratch.ᶜtemp_scalar_2
    ᶜu_p1 = p.scratch.ᶜtemp_scalar_3
    ᶜv_p1 = p.scratch.ᶜtemp_scalar_4
    ᶜbf_p1 = p.scratch.ᶜtemp_scalar_5

    FT = eltype(ᶜρ) # Define the floating point type

    # Using interpolate operator, generate the field of ρ,u,v,z with on level shifted up
    ρ_endlevel = Fields.level(ᶜρ, Spaces.nlevels(axes(ᶜρ)))
    ρ_endlevel_m1 = Fields.level(ᶜρ, Spaces.nlevels(axes(ᶜρ)) - 1)
    Boundary_value = Fields.Field(
        Fields.field_values(ρ_endlevel) .* Fields.field_values(ρ_endlevel) ./
        Fields.field_values(ρ_endlevel_m1),
        axes(ρ_endlevel),
    )
    field_shiftlevel_up!(ᶜρ, ᶜρ_p1, Boundary_value)

    u_endlevel = Fields.level(ᶜu, Spaces.nlevels(axes(ᶜu)))
    u_endlevel_m1 = Fields.level(ᶜu, Spaces.nlevels(axes(ᶜu)) - 1)
    Boundary_value = Fields.Field(
        FT(2) .* Fields.field_values(u_endlevel) .-
        Fields.field_values(u_endlevel_m1),
        axes(u_endlevel),
    )
    field_shiftlevel_up!(ᶜu, ᶜu_p1, Boundary_value)

    v_endlevel = Fields.level(ᶜv, Spaces.nlevels(axes(ᶜv)))
    v_endlevel_m1 = Fields.level(ᶜv, Spaces.nlevels(axes(ᶜv)) - 1)
    Boundary_value = Fields.Field(
        FT(2) .* Fields.field_values(v_endlevel) .-
        Fields.field_values(v_endlevel_m1),
        axes(v_endlevel),
    )
    field_shiftlevel_up!(ᶜv, ᶜv_p1, Boundary_value)

    Boundary_value = Fields.level(ᶜbf, Spaces.nlevels(axes(ᶜbf)))
    field_shiftlevel_up!(ᶜbf, ᶜbf_p1, Boundary_value)

    z_endlevel = Fields.level(ᶜz, Spaces.nlevels(axes(ᶜz)))
    z_endlevel_m1 = Fields.level(ᶜz, Spaces.nlevels(axes(ᶜz)) - 1)
    Boundary_value = Fields.Field(
        FT(2) .* Fields.field_values(z_endlevel) .-
        Fields.field_values(z_endlevel_m1),
        axes(z_endlevel),
    )
    field_shiftlevel_up!(ᶜz, ᶜz_p1, Boundary_value)

    mask_u = StaticBitVector{nc}(_ -> true)
    mask_v = StaticBitVector{nc}(_ -> true)
    # We use StaticBitVector here because the unrolled_reduce function in Julia can
    # cause memory allocation issues when the mask has more than 32 elements.
    # StaticBitVector stores 8 boolean values in a UInt8, allowing efficient storage
    # for up to 256 gravity wave break data.
    level_end = Spaces.nlevels(axes(ᶜρ))

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
    input_u_beres = @. lazy(
        tuple(
            ᶜu_p1,
            ᶜbf_p1,
            ᶜρ,
            ᶜρ_p1,
            ᶜz_p1,
            ᶜz,
            ᶜlevel,
            gw_beres_active,
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
            gw_beres_active,
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

    u_waveforcing_top = p.non_orographic_gravity_wave.u_waveforcing_top
    v_waveforcing_top = p.non_orographic_gravity_wave.v_waveforcing_top
    gw_avg_scratch = p.scratch.ᶜtemp_scalar_6

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
            u_waveforcing, v_waveforcing,
            u_waveforcing_top, v_waveforcing_top,
            uforcing, vforcing,
            damp_level, ᶜlevel, level_end, gw_avg_scratch,
        )

        # --- Beres convective source (when configured) ---
        # Compile-time eliminated when gw_beres_source === nothing (BS=Nothing)
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
                u_waveforcing, v_waveforcing,
                u_waveforcing_top, v_waveforcing_top,
                uforcing, vforcing,
                damp_level, ᶜlevel, level_end, gw_avg_scratch,
            )
        end
    end
    return nothing
end

# Post-process u/v waveforcing pair and accumulate into forcing fields.
# gw_average! overwrites the scratch field that the input tuples alias as ᶜρ_p1,
# so both column_accumulate! calls must complete BEFORE this is called.
function postprocess_and_accumulate!(
    u_waveforcing, v_waveforcing,
    u_waveforcing_top, v_waveforcing_top,
    uforcing, vforcing,
    damp_level, ᶜlevel, level_end, scratch,
)
    # Extract momentum flux at model top
    copyto!(
        Fields.field_values(u_waveforcing_top),
        Fields.field_values(
            Fields.level(u_waveforcing, Spaces.nlevels(axes(u_waveforcing))),
        ),
    )
    fill!(Fields.level(u_waveforcing, Spaces.nlevels(axes(u_waveforcing))), 0)

    copyto!(
        Fields.field_values(v_waveforcing_top),
        Fields.field_values(
            Fields.level(v_waveforcing, Spaces.nlevels(axes(v_waveforcing))),
        ),
    )
    fill!(Fields.level(v_waveforcing, Spaces.nlevels(axes(v_waveforcing))), 0)

    # Interpolate from center to face (clobbers scratch)
    gw_average!(u_waveforcing, scratch)
    gw_average!(v_waveforcing, scratch)

    # Deposit escaped momentum flux above damp level
    @. u_waveforcing = gw_deposit(
        u_waveforcing_top, u_waveforcing, damp_level, ᶜlevel, level_end,
    )
    @. v_waveforcing = gw_deposit(
        v_waveforcing_top, v_waveforcing, damp_level, ᶜlevel, level_end,
    )

    # Accumulate into forcing
    @. uforcing = uforcing + u_waveforcing
    @. vforcing = vforcing + v_waveforcing
end

# Explicit source spectrum helpers for the two-pass accumulation.
compute_ad99_spectrum(c, u_source, Bw, Bn, cw, cn, c0, flag, gw_ncval) =
    wave_source(c, u_source, Bw, Bn, cw, cn, c0, flag, gw_ncval)

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

# Fallback for when beres_source is nothing (AD99-only mode).
# This is never called at runtime but must exist for GPU compilation
# since both branches of the MODE dispatch must be compilable.
function compute_beres_spectrum(
    ::Nothing, beres_active_val, c, u_heat_val, Q0_val, h_val, N_val,
    gw_ncval::Val{nc},
) where {nc}
    ntuple(_ -> typeof(Q0_val)(0), Val(nc))
end

# Summarize the launched Beres source spectrum for a single column into two
# scalars: the total launched flux magnitude Σ_n |B0(c_n)| and the phase-speed
# first moment Σ_n c_n·|B0(c_n)|. Their ratio is the flux-weighted spectral
# centroid ⟨c⟩. The Beres spectrum is wavenumber-(ink-)independent, so this one
# evaluation captures the full launched spectrum. GPU-safe: B0 is a stack
# NTuple (no heap alloc) and `c` is an isbits NTuple passed as a scalar grid.
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

# Using column_accumulate function, calculate the gravity wave forcing at each point.
# source_mode::Val{:ad99} or Val{:beres} selects which source spectrum and intermittency to use.
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
    # Here we use column_accumulate function to pass the variable B0 and mask through different levels, and calculate waveforcing at each level.
    Operators.column_accumulate!(
        waveforcing,
        input;
        init = (FT(0.0), mask, FT(NaN), ntuple(i -> FT(NaN), Val(nc))),
        transform = first,
    ) do (wave_forcing, mask, Bsum_or_NaN, B0_or_NaNs), inp
        # The input tuple is indexed (not destructured in the signature) because
        # the two source modes share only slots 1–7; slots 8–16 are mode-specific.
        # Reading the mode-specific fields under the compile-time `MODE` guard
        # keeps each specialization's parameter tuple at 16 fields, under the
        # 4 KiB GPU kernel-parameter limit (sm_60). The union of both modes' fields
        # (25) overflows it. See the `input_*_ad99` / `input_*_beres` tuples built
        # in `non_orographic_gravity_wave_forcing`.
        #
        # The effective source-level state differs by mode: AD99 keeps its fixed
        # source level (pressure-based on sphere, height-based in column); Beres
        # launches at the top of the convective heating envelope (gw_ztop),
        # populated per column in compute_tendency!.
        u_kp1 = inp[1]
        bf_kp1 = inp[2]
        ρ_k = inp[3]
        ρ_kp1 = inp[4]
        z_kp1 = inp[5]
        z_k = inp[6]
        level = inp[7]
        FT1 = typeof(u_kp1)
        # Slots 8–16 are mode-specific (AD99 source fields vs. Beres convective
        # fields). Each local is assigned exactly ONCE, via a ternary — NOT an
        # if/else that assigns in both branches. These locals are captured by the
        # inner `unrolled_reduce` closure, and a variable assigned in more than one
        # branch is boxed (`Core.Box`, a heap alloc), which is illegal in a GPU
        # kernel (`gpu_gc_pool_alloc` → InvalidIRError). MODE is a compile-time
        # constant, so each ternary folds to its live slot; the dead-side index is
        # never evaluated (and is in-bounds for the other mode's 16-tuple anyway).
        # Fields the active mode does not use get a type-matched `zero(FT1)` dummy
        # so the dead spectrum branch below still compiles.
        u_source_eff = MODE == :ad99 ? inp[8] : inp[15]
        ρ_source_eff = MODE == :ad99 ? inp[9] : inp[14]
        source_level_eff = MODE == :ad99 ? inp[10] : inp[13]
        Bw = MODE == :ad99 ? inp[11] : zero(FT1)
        Bn = MODE == :ad99 ? inp[12] : zero(FT1)
        cw = MODE == :ad99 ? inp[13] : zero(FT1)
        cn = MODE == :ad99 ? inp[14] : zero(FT1)
        flag = MODE == :ad99 ? inp[15] : zero(FT1)
        source_ampl = MODE == :ad99 ? inp[16] : zero(FT1)
        beres_active_val = MODE == :ad99 ? zero(FT1) : inp[8]
        Q0_val = MODE == :ad99 ? zero(FT1) : inp[9]
        h_val = MODE == :ad99 ? zero(FT1) : inp[10]
        u_heat_val = MODE == :ad99 ? zero(FT1) : inp[11]
        N_val = MODE == :ad99 ? zero(FT1) : inp[12]
        beres_a_cover = MODE == :ad99 ? zero(FT1) : inp[16]

        kwv = 2.0 * π / ((30.0 * (10.0^ink)) * 1.e3) # wave number of gravity waves
        k2 = kwv * kwv

        fac = FT1(0.5) * (ρ_kp1 / ρ_source_eff) * kwv / bf_kp1
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
            else # MODE == :beres
                compute_beres_spectrum(
                    beres_source, beres_active_val,
                    c, u_heat_val, Q0_val, h_val, N_val, gw_ncval,
                )
            end
            Bsum1 = sum(abs, B1)
            B1, Bsum1
        else
            B0_or_NaNs, Bsum_or_NaN
        end

        if level >= source_level_eff - 1
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
                    # f phase speed matches the wind speed, remove c(n) from the set of propagating waves.
                    if c_hat == 0.0
                        mask = Base.setindex(mask, false, n)
                    else
                        c_hat0 = c[n] - u_source_eff
                        # define the criterion which determines if wave is reflected at this level (test).
                        test = abs(c_hat) * kwv - ω_r
                        if test >= 0.0
                            # wave has undergone total internal reflection. remove it from the propagating set.
                            mask = Base.setindex(mask, false, n)
                        else
                            if level == level_end
                                # this is added in MiMA implementation:
                                # all momentum flux that escapes across the model top
                                # is deposited to the extra level being added so that
                                # momentum flux is conserved
                                mask = Base.setindex(mask, false, n)
                                if level >= source_level_eff
                                    fm = fm + B0[n]
                                end
                            else
                                # if wave is not reflected at this level, determine if it is
                                # breaking at this level (Foc >= 0), or if wave speed relative to
                                # windspeed has changed sign from its value at the source level
                                # (c_hat0[n] * c_hat <= 0). if it is above the source level and is
                                # breaking, then add its momentum flux to the accumulated sum at
                                # this level.
                                # set mask=0.0 to remove phase speed band c[n] from the set of active
                                # waves moving upwards to the next level.
                                Foc = B0[n] / (c_hat)^3 - fac
                                if Foc >= 0.0 || (c_hat0 * c_hat <= 0.0)
                                    mask = Base.setindex(mask, false, n)
                                    if level >= source_level_eff
                                        fm = fm + B0[n]
                                    end
                                end
                            end
                        end # (test >= 0.0)

                    end #(c_hat == 0.0)
                end # mask = 0
                return (mask, fm)
            end

            # compute the gravity wave momentum flux forcing
            # obtained across the entire wave spectrum at this level.
            eps = if MODE == :ad99
                calc_intermitency(ρ_source_eff, source_ampl, nk, FT1(Bsum))
            else # MODE == :beres
                beres_a_cover / (ρ_source_eff * FT1(nk))
            end
            if level >= source_level_eff
                rbh = sqrt(ρ_k * ρ_kp1)
                wave_forcing = (ρ_source_eff / rbh) * FT1(fm) * eps / (z_kp1 - z_k)
            else
                wave_forcing = FT1(0.0)
            end
        end
        return (wave_forcing, mask, Bsum, B0)

    end
end

# calculate the intermittency factor eps -> assuming constant Δc.
function calc_intermitency(ρ_source_level, source_ampl, nk, Bsum)
    return (source_ampl / ρ_source_level / nk) / Bsum
end

function gw_average!(wave_forcing, wave_forcing_m1)
    FT = eltype(wave_forcing)
    L1 = Operators.LeftBiasedC2F(; bottom = Operators.SetValue(FT(0.0)))
    L2 = Operators.LeftBiasedF2C(;)
    wave_forcing_m1 .= L2.(L1.(wave_forcing))
    @. wave_forcing = FT(0.5) * (wave_forcing + wave_forcing_m1)
end

function gw_deposit(wave_forcing_top, wave_forcing, damp_level, level, height)
    if level >= damp_level
        wave_forcing =
            wave_forcing + wave_forcing_top / (height + 2 - damp_level)
    end
    return wave_forcing
end

function field_shiftlevel_up!(ᶜexample_field, ᶜshifted_field, Boundary_value)
    R1 = Operators.RightBiasedC2F(; top = Operators.SetValue(Boundary_value))
    R2 = Operators.RightBiasedF2C(;)
    ᶜshifted_field .= R2.(R1.(ᶜexample_field))
end

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

"""
    V_hs_sq(m, h)

Squared vertical half-sine shape factor for heating depth `h` at wavenumber `m`:

    V_hs_sq = (π/h)² · sin²(m·h) / (m² − (π/h)²)²

written in `sinc` form (δ = m·h − π) so it's finite at resonance m = π/h (→ h²/4).

Used by the steady (ν=0) path. Same shape lives inside `R` of `_beres_spectrum_single_h` (R² = V_hs_sq·m²/(N²−ν̂²)²)
but is deliberately NOT factored out, both path (steady and transient) stays bit-identical.

`sinc` here is unnormalized `sin(x)/x` by choice (consistent). Avoid Base.sinc which is normalized.
"""
@inline function V_hs_sq(m, h)
    FT = typeof(m)
    δ = m * h - FT(π)
    sinc_δ = abs(δ) < FT(1e-10) ? FT(1) : sin(δ) / δ
    return FT(π)^2 * h^2 * sinc_δ^2 / (m * h + FT(π))^2
end

"""
    _beres_spectrum_single_h(c_n, c_hat, h, ...)

Compute the Beres (2004) momentum flux integrand for a single phase speed bin
and a single heating depth h.
"""
@inline function _beres_spectrum_single_h(
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
            m_h = m * h

            N2_minus_νhat2 = N2 - ν_hat^2

            δ = m_h - π_val
            sinc_δ = abs(δ) < FT(1e-10) ? FT(1) : sin(δ) / δ
            sin_over_denom = -sinc_δ / (m_h + π_val)
            R = π_val * m * h * sin_over_denom / N2_minus_νhat2

            Q0Gk_sq =
                Q0_sq * σ_x_sq / FT(2) * exp(-k^2 * σ_x_sq / FT(2))
            B_sq = Q0Gk_sq * R^2

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

Horizontal-scale factor `H` of the steady (ν=0) source amplitude (Beres Eq. 32):
how strongly a convective cell of width `σ_x` projects onto gravity waves, summed
over horizontal scales down to the system size `L_system`. Closed form:

    H = (σ_x²/4) · E1(x),   x = k_min²·σ_x²/2,   k_min = 2π/L_system

using `E1(x) ≈ −γ − ln(x) + x` (valid since `x ≪ 1`; GPU-safe, just a `log`).
"""
@inline function _beres_steady_horizontal_const(σ_x, L_system)
    FT = typeof(σ_x)
    γ = FT(0.5772156649015329)
    a = σ_x^2 / FT(2)
    k_min = FT(2) * FT(π) / L_system
    x = a * k_min^2
    E1 = x < FT(1) ? (-γ - log(x) + x) : FT(0)
    return a / FT(2) * E1
end

"""
    _beres_steady_flux(U, N_source, h, Q0, scale_factor, n_h_avg, Δh_frac, σ_x, L_system, dc_frac, ν_min)

Launch flux of the steady (ν=0) wave for one azimuth: time-mean convective heating
acts like a hill in the wind `U`, radiating one stationary wave (`m₀ = N/|U|`).
Signed to decelerate `U`; summed alongside the transient bins (Beres 2004 Eqs. 31–34):

    F_steady = −sign(U) · scale_factor · (1/√(2π)) · Q0² · Q_t(0)² · V_hs_sq(m₀,h) · H / (N·|U|³)

Reuses the transient primitives; the only new pieces are:
• `H` — horizontal constant (`_beres_steady_horizontal_const`), the sole new `L`-dependence.
• `Q_t(0)² = dc_frac · ν_min` — the DC weight. `dc_frac` is a user knob (default 1)
setting the steady's frequency band-width relative to one transient bin.

Steady and transient carry orthogonal frequencies, so no double-counting; their ratio
is `scale_factor`-independent and ≈ O(1) for defaults (pinned in `test_beres_steady_source.jl`).
`U → 0` returns 0 smoothly (guards the `1/U³`).
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
)
    FT = typeof(U)
    # U → 0 guard: m₀ = N/|U| → ∞, the stationary wave vanishes; also protects
    # the 1/U³ below from blowing up.
    if abs(U) < FT(1e-6)
        return FT(0)
    end
    Uabs = abs(U)
    m0 = N_source / Uabs

    # Half-sine shape, optionally h-averaged to smooth the m₀ ≈ π/h resonance
    # (identical mechanism to the transient `n_h_avg` loop).
    Vbar = if n_h_avg <= 1
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
    wave_source(c, u_heat, Q0, h, N_source, beres::BeresSourceParams, gw_ncval)

Compute the Beres (2004) convective gravity wave momentum flux spectrum.
Dispatches on `BeresSourceParams` to distinguish from the AD Gaussian method.

Implements Eqs. (23), (29)-(30) from Beres, Alexander & Holton (2004, JAS).
When `n_h_avg > 1`, averages the spectrum over multiple h values in the range
`h ± Δh_frac * h` to smooth the resonance peaks, following the paper's
recommendation (Section 2, Figure 4).

Returns `NTuple{nc, FT}` in units consistent with the AD `wave_source`.
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
    ) = beres
    FT = typeof(u_heat)
    scale_factor = FT(beres_scale_factor)

    boole_w = (FT(7), FT(32), FT(12), FT(32), FT(7))
    dν = (ν_max - ν_min) / FT(n_ν - 1)
    n_groups = (n_ν - 1) ÷ 4

    N2 = N_source^2
    σ_x_sq = σ_x^2
    Q0_sq = Q0^2

    # Steady (ν=0) contribution: a ground-stationary wave, so it lands in the c≈0 bin
    # (kept empty by the transient spectrum, so no double-counting). It deposits only
    # when (a) the steady source is enabled — true by default, with no YAML switch; the
    # field exists so internal tools can toggle it at a fixed grid — and (b) an exact
    # c=0 bin exists. Without a c=0 bin, `clamp` would otherwise point n_zero at a
    # nonzero edge bin and silently corrupt it, so we zero the steady flux instead
    # (the graceful no-op that makes the grid the user-facing on/off control).
    dc = c[2] - c[1]
    cmax = -c[1]
    n_zero = clamp(round(Int, cmax / dc) + 1, 1, nc)
    has_c0_bin = abs(c[n_zero]) < FT(1e-6)
    steady_flux =
        (beres_steady_source && has_c0_bin) ?
        _beres_steady_flux(
            u_heat,
            N_source,
            h,
            Q0,
            scale_factor,
            n_h_avg,
            Δh_frac,
            σ_x,
            FT(beres_L_system),
            FT(beres_steady_dc_frac),
            ν_min,
        ) : FT(0)

    ntuple(
        n -> begin
            c_n = c[n]
            c_hat = c_n - u_heat

            val = if abs(c_hat) < FT(1e-6) || abs(c_n) < FT(1e-6)
                FT(0)
            else
                if n_h_avg <= 1
                    # Single h value (no averaging)
                    result = _beres_spectrum_single_h(
                        c_n, c_hat, h, u_heat, N2, N_source,
                        Q0_sq, σ_x_sq, ν_min, dν, n_groups, boole_w,
                    )
                else
                    # Average over n_h_avg values of h in [h - Δh, h + Δh]
                    # per Beres (2004) Section 2, Figure 4
                    Δh = Δh_frac * h
                    h_min = h - Δh
                    h_max = h + Δh
                    dh = (h_max - h_min) / FT(n_h_avg - 1)
                    result = FT(0)
                    for ih in 1:n_h_avg
                        h_i = h_min + FT(ih - 1) * dh
                        result += _beres_spectrum_single_h(
                            c_n, c_hat, h_i, u_heat, N2, N_source,
                            Q0_sq, σ_x_sq, ν_min, dν, n_groups, boole_w,
                        )
                    end
                    result = result / FT(n_h_avg)
                end

                sign(c_hat) * scale_factor * result
            end

            # Deposit the steady (ν=0) flux into the c≈0 bin only.
            val + steady_flux * FT(n == n_zero)
        end,
        Val(nc),
    )
end
