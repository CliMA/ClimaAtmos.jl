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
    if iscolumn_or_box(axes(Y.c))
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
            # Beres fields (always allocated, inactive for column/box)
            gw_Q0 = similar(Fields.level(Y.c.ρ, 1)),
            gw_h_heat = similar(Fields.level(Y.c.ρ, 1)),
            gw_u_heat = similar(Fields.level(Y.c.ρ, 1)),
            gw_v_heat = similar(Fields.level(Y.c.ρ, 1)),
            gw_N_source = similar(Fields.level(Y.c.ρ, 1)),
            gw_beres_active = similar(Fields.level(Y.c.ρ, 1)),
            gw_beres_source = nothing,
            gw_zbot = similar(Fields.level(Y.c.ρ, 1)),
            gw_ztop = similar(Fields.level(Y.c.ρ, 1)),
            gw_Q_conv = similar(Y.c.ρ),
            gw_reduce_result = similar(
                Fields.level(Y.c.ρ, 1),
                Tuple{FT, FT, FT, FT, FT, FT},
            ),
            gw_deep_count = Fields.zeros(FT, axes(Fields.level(Y.c.ρ, 1))),
            gw_cb_count = Fields.zeros(FT, axes(Fields.level(Y.c.ρ, 1))),
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
            gw_reduce_result = similar(
                Fields.level(Y.c.ρ, 1),
                Tuple{FT, FT, FT, FT, FT, FT},
            ),
            gw_deep_count = Fields.zeros(FT, axes(Fields.level(Y.c.ρ, 1))),
            gw_cb_count = Fields.zeros(FT, axes(Fields.level(Y.c.ρ, 1))),
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

    if iscolumn_or_box(axes(Y.c))
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

    # Compute Beres convective heating if enabled
    if !isnothing(p.non_orographic_gravity_wave.gw_beres_source)
        compute_beres_convective_heating!(Y, p)
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

    # _uf_data = Array(parent(uforcing))
    # _vf_data = Array(parent(vforcing))
    # _sf = p.non_orographic_gravity_wave.gw_beres_source.beres_scale_factor
    # println("[Beres debug] cb=$(_beres_dbg_count[]) FORCING: u absmax=$(maximum(abs.(_uf_data))) v absmax=$(maximum(abs.(_vf_data))) scale=$(_sf)")

end

"""
    compute_beres_convective_heating!(Y, p)

Extract convective heating properties from EDMF for the Beres (2004) source spectrum.
Computes per-column: Q0 (max heating rate), h (heating depth), u_heat/v_heat (mean wind),
N_source (buoyancy freq), and beres_active flag.
"""
# const _beres_dbg_count = Ref(0)
function compute_beres_convective_heating!(Y, p)
    (; turbconv_model) = p.atmos
    n_updrafts = n_mass_flux_subdomains(turbconv_model)

    if n_updrafts == 0
        # No EDMF — Beres inactive everywhere
        p.non_orographic_gravity_wave.gw_beres_active .= 0
        return
    end

    # _beres_dbg_count[] += 1
    # _do_debug = true  # always print — debugging active

    (; ᶠu³ʲs, ᶜKʲs, ᶜρʲs, ᶜh_tot, ᶜuʲs) = p.precomputed
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
        gw_reduce_result,
        gw_deep_count,
        gw_cb_count,
        ᶜbuoyancy_frequency,
    ) = p.non_orographic_gravity_wave

    FT = Spaces.undertype(axes(Y.c))
    ᶜρ = Y.c.ρ
    ᶜz = Fields.coordinate_field(Y.c).z

    # Compute convective heating via mass-flux divergence (Yanai Q₁):
    #   Q₁ = -1/(ρ·cp) · ∂/∂z [Σⱼ ρʲ·(u³ʲ-u³)·aʲ·(mseʲ + Kʲ - h_tot)]
    # Following the same pattern as edmfx_sgs_flux.jl:54-88.
    ᶜQ_conv = p.scratch.ᶜtemp_scalar_2
    ᶜQ_conv .= FT(0)
    cp_d = FT(CAP.cp_d(p.params))

    # Scratch fields for face velocity anomaly and cell-center scalar
    ᶠu³_diff = p.scratch.ᶠtemp_CT3
    ᶜa_scalar = p.scratch.ᶜtemp_scalar

    # For DiagnosticEDMFX, ρa and mse are in p.precomputed
    # For PrognosticEDMFX, ρa and mse are in Y.c.sgsʲs.:($j)
    has_prognostic_sgs =
        hasproperty(Y.c, :sgsʲs) && n_updrafts > 0
    if !has_prognostic_sgs && haskey(p.precomputed, :ᶜρaʲs)
        ᶜρaʲs_all = p.precomputed.ᶜρaʲs
        ᶜmseʲs_all = p.precomputed.ᶜmseʲs
    end

    for j in 1:n_updrafts
        # Velocity anomaly at faces (contravariant)
        @. ᶠu³_diff = ᶠu³ʲs.:($$j) - ᶠu³

        # Enthalpy anomaly × area fraction (same as edmfx_sgs_flux.jl:65-67)
        ᶜmseʲ = if has_prognostic_sgs
            Y.c.sgsʲs.:($j).mse
        else
            ᶜmseʲs_all.:($j)
        end
        ᶜρaʲ = if has_prognostic_sgs
            Y.c.sgsʲs.:($j).ρa
        else
            ᶜρaʲs_all.:($j)
        end

        # Area fraction = ρaʲ/ρʲ; protect against 0/0 when no updraft
        @. ᶜa_scalar =
            ifelse(
                ᶜρʲs.:($$j) > eps(FT),
                (ᶜmseʲ + ᶜKʲs.:($$j) - ᶜh_tot) * (ᶜρaʲ / ᶜρʲs.:($$j)),
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
        # Convert from W/m³ to heating rate K/s
        @. ᶜQ_conv += vtt / (ᶜρ * cp_d)

        # if _do_debug
        #     _Q_data = Array(parent(ᶜQ_conv))
        #     println("[Beres debug] cb=$(_beres_dbg_count[]) j=$j Q_conv: min=$(minimum(_Q_data)) max=$(maximum(_Q_data)) NaN=$(count(isnan, _Q_data))")
        # end
    end

    # Clean NaN/Inf from boundary stencil artifacts
    # @. ᶜQ_conv = ifelse(isnan(ᶜQ_conv) | isinf(ᶜQ_conv), FT(0), ᶜQ_conv)

    # Persist Q_conv into cache before scratch field is reused
    @. gw_Q_conv = ᶜQ_conv

    # Compute max updraft vertical velocity and total area fraction (cell centers).
    ᶜw_up = p.scratch.ᶜtemp_scalar_3
    ᶜa_up = p.scratch.ᶜtemp_scalar_4
    ᶜw_up .= FT(0)
    ᶜa_up .= FT(0)
    for j in 1:n_updrafts
        @. ᶜw_up = max(ᶜw_up, w_component(Geometry.WVector(ᶜuʲs.:($$j))))
        ᶜρaʲ = if has_prognostic_sgs
            Y.c.sgsʲs.:($j).ρa
        else
            p.precomputed.ᶜρaʲs.:($j)
        end
        @. ᶜa_up += ifelse(ᶜρʲs.:($$j) > eps(FT), ᶜρaʲ / ᶜρʲs.:($$j), FT(0))
    end

    ᶜu = Geometry.UVVector.(Y.c.uₕ).components.data.:1
    ᶜv = Geometry.UVVector.(Y.c.uₕ).components.data.:2

    # Pass 1: find convective envelope [z_bot, z_top].
    # z_peak: height of max updraft velocity (convective core)
    # z_top: highest level where area fraction > threshold (plume top)
    # z_bot = z_top - 2*z_peak, clamped to ≥ 3 km (above PBL)
    result_field = gw_reduce_result
    input1 = Base.Broadcast.broadcasted(tuple, ᶜz, ᶜw_up, ᶜa_up)
    # Accumulator: (w_max, z_peak, z_top, _unused, _unused, _unused)
    reduce_init =
        (FT(0), FT(0), FT(-Inf), FT(0), FT(0), FT(0))
    let _a_thresh = FT(1e-3)
        Operators.column_reduce!(
            result_field,
            input1;
            init = reduce_init,
        ) do (w_max, z_peak_prev, z_top_prev, _4, _5, _6), (z, w, a)
            # Track height of maximum updraft velocity
            new_peak = w > w_max
            w_best = ifelse(new_peak, w, w_max)
            z_peak = ifelse(new_peak, z, z_peak_prev)
            # Track highest level where area fraction exceeds threshold
            z_top = ifelse(a > _a_thresh, max(z_top_prev, z), z_top_prev)
            return (w_best, z_peak, z_top, _4, _5, _6)
        end
    end

    # Extract results and compute z_bot
    @. gw_N_source = result_field.:2   # temporarily holds z_peak
    @. gw_v_heat = result_field.:3     # z_top
    # z_bot = 2*z_peak - z_top, clamped above 3 km to exclude BL thermals
    # (mirrors the upper half of the plume below the peak)
    @. gw_u_heat = max(
        FT(2) * gw_N_source - gw_v_heat,
        FT(3000),
    )
    # Sanitize: if z_top was never set (no active updraft), zero everything
    @. gw_u_heat = ifelse(gw_v_heat < FT(0), FT(0), gw_u_heat)
    @. gw_v_heat = ifelse(gw_v_heat < FT(0), FT(0), gw_v_heat)
    @. gw_h_heat = max(gw_v_heat - gw_u_heat, FT(0))
    @. gw_h_heat = ifelse(isnan(gw_h_heat) | isinf(gw_h_heat), FT(0), gw_h_heat)

    # Persist zbot/ztop before gw_u_heat/gw_v_heat get overwritten by mean winds
    @. gw_zbot = gw_u_heat
    @. gw_ztop = gw_v_heat

    # Count callback invocations and deep convection events (z_top > 10km)
    @. gw_cb_count += FT(1)
    @. gw_deep_count += ifelse(gw_v_heat > FT(10000), FT(1), FT(0))

    # Pass 2: within [z_bot, z_top], compute:
    #   - Q₀ integral: Σ(Q_net · Δz) for Beres half-sine conversion
    #   - Mass-weighted mean wind (u, v) and buoyancy frequency (N)
    # All quantities drawn from the same physical envelope — the continuous
    # convective column from cloud base to plume top.
    ᶜN = p.scratch.ᶜtemp_scalar
    @. ᶜN = sqrt(abs(ᶜbuoyancy_frequency))
    ᶜΔz = Fields.Δz_field(axes(Y.c))
    # Precompute 3D envelope mask (2D z_bot/z_top broadcast over column)
    ᶜz_bot = gw_u_heat  # 2D field, broadcasts over column via @.
    ᶜz_top = gw_v_heat
    ᶜin_env = p.scratch.ᶜtemp_scalar_3  # reuse (ᶜw_up no longer needed)
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
    )
    # Accumulator: (Q_integral, u_sum, v_sum, N_sum, mass_sum, _unused)
    _zero = FT(0)
    _half = FT(0.5)
    reduce_init2 = (_zero, _zero, _zero, _zero, _zero, _zero)
    Operators.column_reduce!(
        result_field,
        input2;
        init = reduce_init2,
    ) do (Q_int_prev, u_sum_prev, v_sum_prev, N_sum_prev, m_sum_prev, _6),
    (Q, u, v, N, ρ, dz, env)
        active = env > _half
        ρdz = ρ * dz
        Q_int = ifelse(active, Q_int_prev + Q * dz, Q_int_prev)
        u_sum = ifelse(active, u_sum_prev + u * ρdz, u_sum_prev)
        v_sum = ifelse(active, v_sum_prev + v * ρdz, v_sum_prev)
        N_sum = ifelse(active, N_sum_prev + N * ρdz, N_sum_prev)
        m_sum = ifelse(active, m_sum_prev + ρdz, m_sum_prev)
        return (Q_int, u_sum, v_sum, N_sum, m_sum, _6)
    end

    # Unpack Pass 2 results
    mass_sum = result_field.:5
    @. gw_u_heat = ifelse(mass_sum > eps(FT), result_field.:2 / mass_sum, FT(0))
    @. gw_v_heat = ifelse(mass_sum > eps(FT), result_field.:3 / mass_sum, FT(0))
    @. gw_N_source =
        ifelse(mass_sum > eps(FT), result_field.:4 / mass_sum, FT(0.01))
    @. gw_Q0 = result_field.:1

    # Finalize Q₀ = (π/2) · Σ(Q_net·Δz) / h, clamped ≥ 0
    @. gw_Q0 = ifelse(
        gw_h_heat > FT(0),
        max(FT(π) / FT(2) * gw_Q0 / gw_h_heat, FT(0)),
        FT(0),
    )
    @. gw_Q0 = ifelse(isnan(gw_Q0) | isinf(gw_Q0), FT(0), gw_Q0)

    # Set beres_active flag: Q0 above threshold AND heating depth above minimum
    Q0_threshold = gw_beres_source.Q0_threshold
    h_heat_min = gw_beres_source.h_heat_min
    @. gw_beres_active = ifelse(
        (gw_Q0 > Q0_threshold) & (gw_h_heat > h_heat_min),
        FT(1),
        FT(0),
    )

    # if _do_debug
    #     _Q0d = Array(parent(gw_Q0))
    #     _hd = Array(parent(gw_h_heat))
    #     _ud = Array(parent(gw_u_heat))
    #     _Nd = Array(parent(gw_N_source))
    #     _ad = Array(parent(gw_beres_active))
    #     _active_mask = _ad .> 0.5
    #     _n_active = sum(_active_mask)
    #     if _n_active > 0
    #         println("[Beres debug] cb=$(_beres_dbg_count[]) TROPICS ACTIVE ($(_n_active)/$(length(_ad))): Q0=[$(minimum(_Q0d[_active_mask])),$(maximum(_Q0d[_active_mask]))] h=[$(minimum(_hd[_active_mask])),$(maximum(_hd[_active_mask]))] u=[$(minimum(_ud[_active_mask])),$(maximum(_ud[_active_mask]))] N=[$(minimum(_Nd[_active_mask])),$(maximum(_Nd[_active_mask]))]")
    #     else
    #         println("[Beres debug] cb=$(_beres_dbg_count[]) NO ACTIVE COLUMNS (Q0 max=$(maximum(_Q0d)) h_heat max=$(maximum(_hd)))")
    #     end
    # end
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
    # _do_debug = false
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
        gw_beres_source,
    ) = p.non_orographic_gravity_wave

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
        ),
    )

    # # Host-side debug: compute Beres spectrum for ONE tropical column to verify
    # if _do_debug
    #     _Q0_h = Array(parent(gw_Q0))
    #     _h_h = Array(parent(gw_h_heat))
    #     _u_h = Array(parent(gw_u_heat))
    #     _N_h = Array(parent(gw_N_source))
    #     _ba_h = Array(parent(gw_beres_active))
    #     _sa_h = Array(parent(gw_source_ampl))
    #     _rs_h = Array(parent(ᶜρ_source))
    #     _sl_h = Array(parent(source_level))
    #     # Find the active column with max Q0 (most intense convection)
    #     _Q0_masked = ifelse.(_ba_h .> 0.5, _Q0_h, 0.0f0)
    #     _idx = argmax(vec(_Q0_masked))
    #     if _Q0_masked[_idx] > 0
    #         println("[Beres HOST DEBUG] cb=$(_beres_dbg_count[]) max-Q0 col idx=$(_idx)")
    #         println("  Q0=$(_Q0_h[_idx]) h=$(_h_h[_idx]) u_heat=$(_u_h[_idx]) N=$(_N_h[_idx])")
    #         println("  source_ampl=$(_sa_h[_idx]) ρ_source=$(_rs_h[_idx]) source_level=$(_sl_h[_idx])")
    #         # Compute spectrum on host
    #         _beres_p = gw_beres_source
    #         _c_tup = gw_c
    #         _nc = length(_c_tup)
    #         _B = wave_source(
    #             _c_tup,
    #             Float64(_u_h[_idx]),
    #             Float64(_Q0_h[_idx]),
    #             Float64(_h_h[_idx]),
    #             Float64(_N_h[_idx]),
    #             BeresSourceParams{Float64}(;
    #                 Q0_threshold = Float64(_beres_p.Q0_threshold),
    #                 beres_scale_factor = Float64(_beres_p.beres_scale_factor),
    #                 σ_x = Float64(_beres_p.σ_x),
    #                 ν_min = Float64(_beres_p.ν_min),
    #                 ν_max = Float64(_beres_p.ν_max),
    #                 n_ν = _beres_p.n_ν,
    #             ),
    #             Val(_nc),
    #         )
    #         _Bsum = sum(abs, _B)
    #         _Bmax = maximum(abs, _B)
    #         println("  HOST spectrum: Bsum=$(_Bsum) Bmax=$(_Bmax) nc=$(_nc)")
    #         # Compute eps for both paths
    #         _eps_ad = Float64(_sa_h[_idx]) / (Float64(_rs_h[_idx]) * Float64(gw_nk) * _Bsum)
    #         _eps_beres = 1.0 / (Float64(_rs_h[_idx]) * Float64(gw_nk))
    #         println("  eps_ad=$(_eps_ad) eps_beres=$(_eps_beres)")
    #         # Check breaking condition for dominant phase speed
    #         _c_peak_idx = argmax(abs.(collect(_B)))
    #         _c_peak = _c_tup[_c_peak_idx]
    #         _c_hat = _c_peak - Float64(_u_h[_idx])
    #         println("  peak c=$(_c_peak) c_hat=$(_c_hat) B0_peak=$(_B[_c_peak_idx])")
    #         println("  B0/c_hat^3=$(abs(_B[_c_peak_idx]) / abs(_c_hat)^3)")
    #     else
    #         println("[Beres HOST DEBUG] cb=$(_beres_dbg_count[]) NO active columns found in beres_active")
    #     end
    # end

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
# gw_average! clobbers scratch (aliased to ᶜρ_p1 in input_u/v), so both
# column_accumulate! calls must complete BEFORE this is called.
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
    )

        FT1 = typeof(u_kp1)
        kwv = 2.0 * π / ((30.0 * (10.0^ink)) * 1.e3) # wave number of gravity waves
        k2 = kwv * kwv

        fac = FT1(0.5) * (ρ_kp1 / ρ_source) * kwv / bf_kp1
        Hb = (z_kp1 - z_k) / log(ρ_k / ρ_kp1) # density scale height
        alp2 = FT1(0.25) / (Hb * Hb)
        ω_r = sqrt((bf_kp1 * bf_kp1 * k2) / (k2 + alp2)) # omc: (critical frequency that marks total internal reflection)

        # calculate momentum flux carried by gravity waves with different phase speeds.
        B0, Bsum = if level == 1
            mask = StaticBitVector{nc}(_ -> true)
            B1 = if MODE == :ad99
                compute_ad99_spectrum(
                    c, u_source, Bw, Bn, cw, cn, c0, flag, gw_ncval,
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

        if level >= source_level - 1
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
                        c_hat0 = c[n] - u_source
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
                                if level >= source_level
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
                                    if level >= source_level
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
                calc_intermitency(ρ_source, source_ampl, nk, FT1(Bsum))
            else # MODE == :beres
                FT1(1.0) / (ρ_source * FT1(nk))
            end
            if level >= source_level
                rbh = sqrt(ρ_k * ρ_kp1)
                wave_forcing = (ρ_source / rbh) * FT1(fm) * eps / (z_kp1 - z_k)
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

            Gk_sq =
                Q0_sq * σ_x_sq / FT(2) * exp(-k^2 * σ_x_sq / FT(2))
            B_sq = Gk_sq * R^2

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
