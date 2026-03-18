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
            ᶜQ_conv = similar(Y.c.ρ),
            gw_Q0 = similar(Fields.level(Y.c.ρ, 1)),
            gw_h_heat = similar(Fields.level(Y.c.ρ, 1)),
            gw_u_heat = similar(Fields.level(Y.c.ρ, 1)),
            gw_v_heat = similar(Fields.level(Y.c.ρ, 1)),
            gw_N_source = similar(Fields.level(Y.c.ρ, 1)),
            gw_beres_active = similar(Fields.level(Y.c.ρ, 1)),
            gw_beres_source = nothing,
            gw_reduce_result = similar(Fields.level(Y.c.ρ, 1), Tuple{FT, FT, FT, FT, FT, FT}),
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
            ᶜQ_conv = similar(Y.c.ρ),
            gw_Q0 = similar(Fields.level(Y.c.ρ, 1)),
            gw_h_heat = similar(Fields.level(Y.c.ρ, 1)),
            gw_u_heat = similar(Fields.level(Y.c.ρ, 1)),
            gw_v_heat = similar(Fields.level(Y.c.ρ, 1)),
            gw_N_source = similar(Fields.level(Y.c.ρ, 1)),
            gw_beres_active = similar(Fields.level(Y.c.ρ, 1)),
            gw_beres_source = gw.beres_source,
            gw_reduce_result = similar(Fields.level(Y.c.ρ, 1), Tuple{FT, FT, FT, FT, FT, FT}),
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

end

"""
    compute_beres_convective_heating!(Y, p)

Extract convective heating properties from EDMF for the Beres (2004) source spectrum.
Computes per-column: Q0 (max heating rate), h (heating depth), u_heat/v_heat (mean wind),
N_source (buoyancy freq), and beres_active flag.
"""
function compute_beres_convective_heating!(Y, p)
    (; turbconv_model) = p.atmos
    n_updrafts = n_mass_flux_subdomains(turbconv_model)

    if n_updrafts == 0
        # No EDMF — Beres inactive everywhere
        p.non_orographic_gravity_wave.gw_beres_active .= 0
        return
    end

    (; ᶜT) = p.precomputed
    (; ᶜTʲs, ᶠu³ʲs) = p.precomputed
    (;
        ᶜQ_conv,
        gw_Q0,
        gw_h_heat,
        gw_u_heat,
        gw_v_heat,
        gw_N_source,
        gw_beres_active,
        gw_beres_source,
        gw_reduce_result,
        ᶜbuoyancy_frequency,
    ) = p.non_orographic_gravity_wave

    FT = Spaces.undertype(axes(Y.c))
    ᶜρ = Y.c.ρ
    ᶜz = Fields.coordinate_field(Y.c).z

    # Compute convective heating proxy: sum over updrafts of (Tʲ - T_env) * ρa_j * w_j / (ρ * cp_d)
    ᶜQ_conv .= FT(0)
    cp_d = FT(CAP.cp_d(p.params))

    # For DiagnosticEDMFX, ρa is in p.precomputed.ᶜρaʲs
    # For PrognosticEDMFX, ρa is in Y.c.sgsʲs.:($j).ρa
    has_prognostic_sgs =
        hasproperty(Y.c, :sgsʲs) && n_updrafts > 0
    if !has_prognostic_sgs && haskey(p.precomputed, :ᶜρaʲs)
        ᶜρaʲs_all = p.precomputed.ᶜρaʲs
    end

    for j in 1:n_updrafts
        ᶜTʲ = ᶜTʲs.:($j)
        ᶜρaʲ = if has_prognostic_sgs
            Y.c.sgsʲs.:($j).ρa
        else
            ᶜρaʲs_all.:($j)
        end
        ᶠu³ʲ = ᶠu³ʲs.:($j)
        # Interpolate face vertical velocity to cell centers and extract w
        ᶜinterp = Operators.InterpolateF2C()
        ᶜwʲ = p.scratch.ᶜtemp_scalar
        @. ᶜwʲ = Geometry.WVector(ᶜinterp(Geometry.WVector(ᶠu³ʲ))).components.data.:1
        # Heating proxy: (Tʲ - T) * ρaʲ * wʲ / (ρ * cp)
        @. ᶜQ_conv += (ᶜTʲ - ᶜT) * ᶜρaʲ * ᶜwʲ / (ᶜρ * cp_d)
    end

    ᶜu = Geometry.UVVector.(Y.c.uₕ).components.data.:1
    ᶜv = Geometry.UVVector.(Y.c.uₕ).components.data.:2

    # First pass: find Q0 (column max of |Q_conv|)
    Operators.column_reduce!(gw_Q0, ᶜQ_conv) do Q_prev, Q
        return max(abs(Q_prev), abs(Q))
    end

    # Second pass: find heating depth and mean wind in heating region
    result_field = gw_reduce_result
    input2 = Base.Broadcast.broadcasted(
        tuple,
        ᶜQ_conv,
        ᶜz,
        ᶜu,
        ᶜv,
        ᶜbuoyancy_frequency,
        ᶜρ,
    )
    eps_weight = eps(FT)
    reduce_init =
        (FT(Inf), FT(-Inf), FT(0), FT(0), FT(0), FT(0))
    Operators.column_reduce!(
        result_field,
        input2;
        init = reduce_init,
    ) do (z_bot_prev, z_top_prev, u_sum_prev, v_sum_prev, bf_sum_prev, w_sum_prev),
    (Q, z, u, v, bf, ρ)
        z_bot = z_bot_prev
        z_top = z_top_prev
        u_sum = u_sum_prev
        v_sum = v_sum_prev
        bf_sum = bf_sum_prev
        w_sum = w_sum_prev
        weight = abs(Q) * ρ
        if weight > eps_weight
            z_bot = min(z_bot, z)
            z_top = max(z_top, z)
            u_sum = u_sum + u * weight
            v_sum = v_sum + v * weight
            bf_sum = bf_sum + bf * weight
            w_sum = w_sum + weight
        end
        return (z_bot, z_top, u_sum, v_sum, bf_sum, w_sum)
    end

    # Unpack results (with NaN/Inf protection)
    @. gw_Q0 = ifelse(isnan(gw_Q0) | isinf(gw_Q0), FT(0), gw_Q0)
    @. gw_h_heat = max(result_field.:2 - result_field.:1, FT(1000.0)) # min 1 km
    @. gw_h_heat = ifelse(isnan(gw_h_heat) | isinf(gw_h_heat), FT(1000.0), gw_h_heat)
    weight_sum = result_field.:6
    @. gw_u_heat = ifelse(weight_sum > eps(FT), result_field.:3 / weight_sum, FT(0))
    @. gw_v_heat = ifelse(weight_sum > eps(FT), result_field.:4 / weight_sum, FT(0))
    @. gw_N_source =
        ifelse(weight_sum > eps(FT), result_field.:5 / weight_sum, FT(0.01))

    # Set beres_active flag: Q0 above threshold AND in tropical band (gw_flag == 0)
    (; gw_flag) = p.non_orographic_gravity_wave
    Q0_threshold = gw_beres_source.Q0_threshold
    @. gw_beres_active =
        ifelse(gw_Q0 > Q0_threshold && gw_flag < FT(0.5), FT(1), FT(0))
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

    # loop over all wave lengths
    for ink in 1:gw_nk
        # Accumulate zonal wave forcing in every column
        waveforcing_column_accumulate!(
            u_waveforcing,
            mask_u,
            input_u,
            gw_c,
            gw_c0,
            gw_nk,
            ink,
            level_end,
            gw_ncval,
            gw_beres_source,
        )

        # Accumulate meridional wave forcing in every column
        waveforcing_column_accumulate!(
            v_waveforcing,
            mask_v,
            input_v,
            gw_c,
            gw_c0,
            gw_nk,
            ink,
            level_end,
            gw_ncval,
            gw_beres_source,
        )

        #extract the momentum flux outside the model top.
        u_waveforcing_top = p.non_orographic_gravity_wave.u_waveforcing_top
        copyto!(
            Fields.field_values(u_waveforcing_top),
            Fields.field_values(
                Fields.level(
                    u_waveforcing,
                    Spaces.nlevels(axes(u_waveforcing)),
                ),
            ),
        )
        fill!(
            Fields.level(u_waveforcing, Spaces.nlevels(axes(u_waveforcing))),
            0,
        )

        v_waveforcing_top = p.non_orographic_gravity_wave.v_waveforcing_top
        copyto!(
            Fields.field_values(v_waveforcing_top),
            Fields.field_values(
                Fields.level(
                    v_waveforcing,
                    Spaces.nlevels(axes(v_waveforcing)),
                ),
            ),
        )
        fill!(
            Fields.level(v_waveforcing, Spaces.nlevels(axes(v_waveforcing))),
            0,
        )

        # interpolate the waveforcing from center to face
        gw_average!(u_waveforcing, p.scratch.ᶜtemp_scalar)
        gw_average!(v_waveforcing, p.scratch.ᶜtemp_scalar)

        # The momentum flux outside the model top will be evenly deposited onto the levels between the damp level and the model top.
        @. u_waveforcing = gw_deposit(
            u_waveforcing_top,
            u_waveforcing,
            damp_level,
            ᶜlevel,
            level_end,
        )
        @. v_waveforcing = gw_deposit(
            v_waveforcing_top,
            v_waveforcing,
            damp_level,
            ᶜlevel,
            level_end,
        )

        # update gravity wave forcing
        @. uforcing = uforcing + u_waveforcing
        @. vforcing = vforcing + v_waveforcing

    end
    return nothing
end

# Compile-time dispatch for source spectrum selection.
# When beres_source::Nothing, the Beres branch is eliminated at compile time.
compute_source_spectrum(
    ::Nothing,
    _,
    c,
    u_source,
    _,
    _,
    _,
    _,
    Bw,
    Bn,
    cw,
    cn,
    c0,
    flag,
    gw_ncval,
) = wave_source(c, u_source, Bw, Bn, cw, cn, c0, flag, gw_ncval)

function compute_source_spectrum(
    beres::BeresSourceParams,
    beres_active_val,
    c,
    u_source,
    u_heat_val,
    Q0_val,
    h_val,
    N_val,
    Bw,
    Bn,
    cw,
    cn,
    c0,
    flag,
    gw_ncval,
)
    if beres_active_val > typeof(beres_active_val)(0.5)
        wave_source(c, u_heat_val, Q0_val, h_val, N_val, beres, gw_ncval)
    else
        wave_source(c, u_source, Bw, Bn, cw, cn, c0, flag, gw_ncval)
    end
end

# Using column_accumulate function, calculate the gravity wave forcing at each point.
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
    beres_source = nothing,
) where {nc}
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
            B1 = compute_source_spectrum(
                beres_source,
                beres_active_val,
                c,
                u_source,
                u_heat_val,
                Q0_val,
                h_val,
                N_val,
                Bw,
                Bn,
                cw,
                cn,
                c0,
                flag,
                gw_ncval,
            )
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
            eps = calc_intermitency(ρ_source, source_ampl, nk, FT1(Bsum))
            #calculate intermittency factor
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
