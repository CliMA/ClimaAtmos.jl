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
            gw_a_cover = similar(Fields.level(Y.c.ρ, 1)),
            gw_reduce_result = similar(
                Fields.level(Y.c.ρ, 1),
                Tuple{FT, FT, FT, FT, FT, FT, FT},
            ),
            gw_deep_count = Fields.zeros(FT, axes(Fields.level(Y.c.ρ, 1))),
            gw_cb_count = Fields.zeros(FT, axes(Fields.level(Y.c.ρ, 1))),
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
            gw_a_cover = similar(Fields.level(Y.c.ρ, 1)),
            gw_reduce_result = similar(
                Fields.level(Y.c.ρ, 1),
                Tuple{FT, FT, FT, FT, FT, FT, FT},
            ),
            gw_deep_count = Fields.zeros(FT, axes(Fields.level(Y.c.ρ, 1))),
            gw_cb_count = Fields.zeros(FT, axes(Fields.level(Y.c.ρ, 1))),
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

        # Per-column Beres launch level: lowest model level with z ≥ gw_ztop.
        # Beres' source spectrum B₀(c) is the far-field flux radiated above the
        # heating layer, so the correct launch level is the top of the
        # convective envelope (gw_ztop) — NOT the AD99 fixed source_pressure
        # level. The AD99 source_level remains as-is for the AD99 spectrum.
        # When Beres is inactive in a column, gw_ztop = 0 and the bottom level
        # is picked; the spectrum is zero anyway so the choice is irrelevant.
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

Extract convective heating properties from EDMF for the Beres (2004) source spectrum.
Computes per-column: Q0 (half-sine heating amplitude = (π/2)·depth-mean of the
IN-CLOUD heating Q_conv_ic), h (heating depth), u_heat/v_heat (mean wind),
N_source (buoyancy freq), a_cover (envelope-mean updraft area fraction, used as
the Beres deposition/intermittency factor), and beres_active flag.

Two heating fields are maintained:
  * `gw_Q_conv` — GRID-MEAN apparent heating Q₁ (the DSE anomaly carries the
    updraft area fraction aʲ). Used for envelope detection and activation
    gating, whose thresholds are calibrated to grid-mean magnitudes.
  * `gw_Q_conv_ic` — IN-CLOUD (per-draft conditional-mean) heating: same DSE
    mass-flux construction WITHOUT the area factor, normalized by draft density.
    This is the amplitude convention Beres' linear theory is forced with (her
    squall-line reference Q₀ ≈ 0.004 K/s); CAM applies the analogous grid-mean →
    local conversion (`CF = 20`, assumed 5% convective fraction) in
    `gw_convect.F90`. The spectrum amplitude Q0 is built from this field; the
    flux deposited on the grid mean is then diluted by `a_cover` (see
    `waveforcing_column_accumulate!`), giving the physically correct
    flux ∝ ā·Q_ic² (linear in coverage, quadratic in local amplitude).

The convective envelope is `[z_bot, z_top]` where
  * `z_top` = highest level with updraft area fraction above `1e-3`,
  * `z_bot` = lowest level with `z ≥ z_bot_floor` AND `Q_conv > z_bot_Q_threshold`.
The altitude floor is required because EDMF Q_conv has a strong PBL/dry-thermal
signal below ~1 km that would otherwise drag z_bot to the surface (see audit Q4).

`ᶜN` is the cell-centre buoyancy frequency N (s⁻¹) — passed in directly because
the cached `ᶜbuoyancy_frequency` field still holds N² from the in-place write
in `non_orographic_gravity_wave_compute_tendency!` and shouldn't be re-sqrt'd
inside this routine.
"""
function compute_beres_convective_heating!(Y, p, ᶜN)
    (; turbconv_model) = p.atmos
    n_updrafts = n_mass_flux_subdomains(turbconv_model)

    if n_updrafts == 0
        # No EDMF — Beres inactive everywhere
        p.non_orographic_gravity_wave.gw_beres_active .= 0
        p.non_orographic_gravity_wave.gw_a_cover .= 0
        p.non_orographic_gravity_wave.gw_Q_conv_ic .= 0
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
        gw_deep_count,
        gw_cb_count,
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

    # Persist Q_conv into cache before scratch field is reused
    @. gw_Q_conv = ᶜQ_conv

    ᶜu = Geometry.UVVector.(Y.c.uₕ).components.data.:1
    ᶜv = Geometry.UVVector.(Y.c.uₕ).components.data.:2

    # Pass 1: find convective envelope [z_bot, z_top].
    #   z_top: highest level where updraft area fraction > a_thresh (plume top).
    #   z_bot: lowest level above z_bot_floor where Q_conv > Q_threshold
    #          (deep-convective heating, not PBL/dry-thermal contamination — the
    #          floor is essential because EDMF Q_conv has a strong PBL signal
    #          below ~1 km in the tropics; see audit Q4).
    result_field = gw_reduce_result
    input1 = Base.Broadcast.broadcasted(tuple, ᶜz, ᶜa_up, ᶜQ_conv)
    # Accumulator: (z_top, z_bot_raw, _3, _4, _5, _6, _7). Sentinels:
    #   z_top  = -Inf → no level qualified
    #   z_bot  = +Inf → no level qualified
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

    # Extract z_top, z_bot_raw. Sanitize: if either sentinel survives (no level
    # qualified), force the envelope to zero so beres_active flips off downstream.
    # Read the unaltered result_field slots inside ifelse to avoid in-place
    # ordering hazards (the validity test depends on BOTH slots).
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
    @. gw_h_heat = ifelse(isnan(gw_h_heat) | isinf(gw_h_heat), FT(0), gw_h_heat)

    # Persist zbot/ztop before gw_u_heat/gw_v_heat get overwritten by mean winds
    @. gw_zbot = gw_u_heat
    @. gw_ztop = gw_v_heat

    # Count callback invocations and deep convection events (z_top > 10km)
    @. gw_cb_count += FT(1)
    @. gw_deep_count += ifelse(gw_v_heat > FT(10000), FT(1), FT(0))

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

    # Spectrum amplitude from the IN-CLOUD heating:
    # Q₀ = (π/2) · Σ(Q_ic·Δz) / h, clamped ≥ 0
    @. gw_Q0 = result_field.:6
    @. gw_Q0 = ifelse(
        gw_h_heat > FT(0),
        max(FT(π) / FT(2) * gw_Q0 / gw_h_heat, FT(0)),
        FT(0),
    )
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

    # Collect all required fields in a broadcasted object
    # Beres fields are always included (zero when disabled; kernel dispatches on gw_beres_source type)
    input_u = @. lazy(
        tuple(
            ᶜu_p1,
            ᶜu_source,
            ᶜbf_p1,
            ᶜρ,
            ᶜρ_p1,
            ᶜρ_source,
            ᶜz_p1,
            ᶜz,
            source_level,
            gw_Bw,
            gw_Bn,
            gw_cw,
            gw_cn,
            gw_flag,
            ᶜlevel,
            gw_source_ampl,
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
    input_v = @. lazy(
        tuple(
            ᶜv_p1,
            ᶜv_source,
            ᶜbf_p1,
            ᶜρ,
            ᶜρ_p1,
            ᶜρ_source,
            ᶜz_p1,
            ᶜz,
            source_level,
            gw_Bw,
            gw_Bn,
            gw_cw,
            gw_cn,
            gw_flag,
            ᶜlevel,
            gw_source_ampl,
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
            u_waveforcing, mask_u, input_u,
            gw_c, gw_c0, gw_nk, ink, level_end,
            gw_ncval, nothing, Val(:ad99),
        )
        waveforcing_column_accumulate!(
            v_waveforcing, mask_v, input_v,
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
                u_waveforcing, mask_u, input_u,
                gw_c, gw_c0, gw_nk, ink, level_end,
                gw_ncval, gw_beres_source, Val(:beres),
            )
            waveforcing_column_accumulate!(
                v_waveforcing, mask_v, input_v,
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
# gw_average! overwrites the scratch field that input_u/v alias as ᶜρ_p1,
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
    ) do (wave_forcing, mask, Bsum_or_NaN, B0_or_NaNs),
    (
        u_kp1,
        u_source,
        bf_kp1,
        ρ_k,
        ρ_kp1,
        ρ_source,
        z_kp1,
        z_k,
        source_level,
        Bw,
        Bn,
        cw,
        cn,
        flag,
        level,
        source_ampl,
        beres_active_val,
        Q0_val,
        h_val,
        u_heat_val,
        N_val,
        beres_source_level,
        beres_ρ_source,
        beres_u_source,
        beres_a_cover,
    )

        # MODE-dispatched launch-level state. AD99 keeps its fixed source level
        # (pressure-based on sphere, height-based in column). Beres launches at
        # the top of the convective heating envelope (gw_ztop), populated per
        # column in compute_tendency!. The ternary on MODE is a compile-time
        # branch via Val{MODE}, so the dead path is elided.
        source_level_eff = MODE == :ad99 ? source_level : beres_source_level
        ρ_source_eff = MODE == :ad99 ? ρ_source : beres_ρ_source
        u_source_eff = MODE == :ad99 ? u_source : beres_u_source

        FT1 = typeof(u_kp1)
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
                # Beres B₀(c) is in physical momentum-flux units for the LOCAL
                # (in-cloud) heating amplitude Q₀ (Q₀², σ_x², α all bundled in
                # compute_beres_spectrum). The deposition is therefore diluted by
                # the convective coverage ā (envelope-mean updraft area fraction):
                # only the fraction ā of the grid cell radiates, so the grid-mean
                # flux is ā·(local flux) — the exact analog of AD99's
                # intermittency ε. Breaking levels are still computed at the
                # LOCAL amplitude (B₀ itself is not rescaled).
                # Remaining factors are bookkeeping for the shared forcing code:
                #   1/ρ_source — cancels the ρ_source multiplier in wave_forcing
                #   1/nk        — distributes total flux across nk azimuths
                # Unlike AD99, no rescaling by Bsum is needed: the Beres B₀(c)
                # already encodes the physical Q₀ amplitude.
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
    _beres_spectrum_single_h(c_n, c_hat, h, ...)

Compute the Beres (2004) momentum flux integrand for a single phase speed bin
and a single heating depth h. This is the inner frequency integral.
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
    (; σ_x, ν_min, ν_max, n_ν, beres_scale_factor, n_h_avg, Δh_frac) = beres
    FT = typeof(u_heat)
    scale_factor = FT(beres_scale_factor)

    boole_w = (FT(7), FT(32), FT(12), FT(32), FT(7))
    dν = (ν_max - ν_min) / FT(n_ν - 1)
    n_groups = (n_ν - 1) ÷ 4

    N2 = N_source^2
    σ_x_sq = σ_x^2
    Q0_sq = Q0^2

    ntuple(
        n -> begin
            c_n = c[n]
            c_hat = c_n - u_heat

            if abs(c_hat) < FT(1e-6) || abs(c_n) < FT(1e-6)
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
        end,
        Val(nc),
    )
end
