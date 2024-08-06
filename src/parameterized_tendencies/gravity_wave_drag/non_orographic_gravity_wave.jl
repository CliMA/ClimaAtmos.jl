#####
##### Non-orographic gravity wave parameterization
#####

import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore.Operators as Operators

non_orographic_gravity_wave_cache(Y, atmos::AtmosModel) =
    non_orographic_gravity_wave_cache(
        Y,
        atmos.non_orographic_gravity_wave,
        atmos.model_config,
    )

non_orographic_gravity_wave_cache(Y, ::Nothing, ::AbstractModelConfig) = (;)

non_orographic_gravity_wave_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

function non_orographic_gravity_wave_cache(
    Y,
    gw::NonOrographyGravityWave,
    ::SingleColumnModel,
)
    FT = Spaces.undertype(axes(Y.c))
    (; source_height, Bw, Bn, Bt_0, dc, cmax, c0, nk, cw, cn) = gw

    nc = Int(floor(FT(2 * cmax / dc + 1)))
    c = [FT((n - 1) * dc - cmax) for n in 1:nc]
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
        gw_B0 = similar(c),
        gw_c = c,
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
        uforcing = similar(Y.c.ρ),
        vforcing = similar(Y.c.ρ),
        gw_ncval = Val(nc),
    )
end

function non_orographic_gravity_wave_cache(
    Y,
    gw::NonOrographyGravityWave,
    ::SphericalModel,
)

    FT = Spaces.undertype(axes(Y.c))
    (; source_pressure, damp_pressure, Bw, Bn, Bt_0, Bt_n, Bt_s, Bt_eq) = gw
    (; ϕ0_s, ϕ0_n, dϕ_n, dϕ_s, dc, cmax, c0, nk, cw, cw_tropics, cn) = gw

    nc = Int(floor(FT(2 * cmax / dc + 1)))
    c = [FT((n - 1) * dc - cmax) for n in 1:nc]

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
        gw_B0 = similar(c),
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
        uforcing = similar(Y.c.ρ),
        vforcing = similar(Y.c.ρ),
        gw_ncval = Val(nc),
    )
end

function non_orographic_gravity_wave_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::NonOrographyGravityWave,
)
    #unpack
    (; ᶜT,) = p.core
    (; ᶜts) = p.precomputed
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
        gw_cmax,
        gw_dc,
        gw_ncval,
    ) = p.non_orographic_gravity_wave
    (; model_config) = p.atmos
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
    ) = p.non_orographic_gravity_wave

    if model_config isa SingleColumnModel
        (; gw_source_height, source_ρ_z_u_v_level) =
            p.non_orographic_gravity_wave
    elseif model_config isa SphericalModel
        (; gw_source_pressure, gw_damp_pressure, source_p_ρ_z_u_v_level) =
            p.non_orographic_gravity_wave
    end
    ᶜρ = Y.c.ρ
    ᶜz = Fields.coordinate_field(Y.c).z
    FT = Spaces.undertype(axes(Y.c))
    # parameters
    thermo_params = CAP.thermodynamics_params(params)
    grav = CAP.grav(params)

    # compute buoyancy frequency
    @. ᶜT = TD.air_temperature(thermo_params, ᶜts)

    ᶜdTdz .= Geometry.WVector.(ᶜgradᵥ.(ᶠinterp.(ᶜT))).components.data.:1

    @. ᶜbuoyancy_frequency =
        (grav / ᶜT) * (ᶜdTdz + grav / TD.cp_m(thermo_params, ᶜts))
    ᶜbuoyancy_frequency = @. ifelse(
        ᶜbuoyancy_frequency < FT(2.5e-5),
        FT(sqrt(2.5e-5)),
        sqrt(abs(ᶜbuoyancy_frequency)),
    ) # to avoid small numbers

    # prepare physical uv input variables for gravity_wave_forcing()
    ᶜu = Geometry.UVVector.(Y.c.uₕ).components.data.:1
    ᶜv = Geometry.UVVector.(Y.c.uₕ).components.data.:2

    if model_config isa SingleColumnModel
        # source level: the index of the level that is closest to the source height

        input = Base.Broadcast.broadcasted(tuple, ᶜρ, ᶜz, ᶜu, ᶜv, ᶜlevel)
        Operators.column_reduce!(
            source_level,
            input;
        ) do (ρ_prev, z_prev, u_prev, v_prev, level_prev), (ρ, z, u, v, level)
            if abs(z_prev - gw_source_height) >= abs(z - gw_source_height)
                return (ρ, z, u, v, level, gw_source_height)
            else
                return (ρ_prev, z_prev, u_prev, v_prev, level_prev)
            end
        end

        ᶜρ_source = source_ρ_z_u_v_level.:1
        ᶜu_source = source_ρ_z_u_v_level.:3
        ᶜv_source = source_ρ_z_u_v_level.:4
        source_level = source_ρ_z_u_v_level.:5


        fill!(damp_level, Spaces.nlevels(axes(ᶜz)))

    elseif model_config isa SphericalModel
        (; ᶜp) = p.precomputed
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


        # damp level: the index of the lowest level whose pressure is lower than the damp pressure

        input = Base.Broadcast.broadcasted(tuple, ᶜlevel, ᶜp, gw_damp_pressure)
        Operators.column_reduce!(
            damp_level,
            input;
            transform = first,
        ) do (level_prev, p_prev, gw_damp_pressure),
        (level, p, gw_damp_pressure)
            if (p_prev - gw_damp_pressure) >= 0
                return (level, p, gw_damp_pressure)
            else
                return (level_prev, p_prev, gw_damp_pressure)
            end
        end

    end

    ᶜu = Geometry.UVVector.(Y.c.uₕ).components.data.:1
    ᶜv = Geometry.UVVector.(Y.c.uₕ).components.data.:2
    ᶜρ_p1 = similar(ᶜρ)
    ᶜz_p1 = similar(ᶜz)
    ᶜu_p1 = similar(ᶜu)
    ᶜv_p1 = similar(ᶜv)
    ᶜbf_p1 = similar(ᶜbuoyancy_frequency)

    uforcing .= 0
    vforcing .= 0

    @time non_orographic_gravity_wave_forcing(
        ᶜu,
        ᶜv,
        ᶜbuoyancy_frequency,
        ᶜρ,
        ᶜz,
        ᶜlevel,
        source_level,
        damp_level,
        gw_source_ampl,
        ᶜρ_source,
        ᶜu_source,
        ᶜv_source,
        uforcing,
        vforcing,
        gw_Bw,
        gw_Bn,
        gw_cw,
        gw_cn,
        gw_flag,
        gw_ncval,
        gw_c,
        gw_c0,
        gw_nk,
        gw_cmax,
        gw_dc,
        u_waveforcing,
        v_waveforcing,
        ᶜρ_p1,
        ᶜu_p1,
        ᶜv_p1,
        ᶜbf_p1,
        ᶜz_p1,
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
    gw_source_ampl,
    ᶜρ_source,
    ᶜu_source,
    ᶜv_source,
    uforcing,
    vforcing,
    gw_Bw,
    gw_Bn,
    gw_cw,
    gw_cn,
    gw_flag,
    gw_ncval::Val{nc},
    gw_c,
    c0,
    nk,
    cmax,
    dc,
    u_waveforcing,
    v_waveforcing,
    ᶜρ_p1,
    ᶜu_p1,
    ᶜv_p1,
    ᶜbf_p1,
    ᶜz_p1,
) where {nc}
    nci = get_nc(gw_ncval)
    FT = eltype(ᶜρ)
    ρ_end = Fields.level(ᶜρ, Spaces.nlevels(axes(ᶜρ)))
    ρ_endm1 = Fields.level(ᶜρ, Spaces.nlevels(axes(ᶜρ)) - 1)
    Boundary_value = Fields.Field(
        Fields.field_values(ρ_end) .* Fields.field_values(ρ_end) ./
        Fields.field_values(ρ_endm1),
        axes(ρ_end),
    )
    R1 = Operators.RightBiasedC2F(; top = Operators.SetValue(Boundary_value))
    R2 = Operators.RightBiasedF2C(;)
    ᶜρ_p1 .= R2.(R1.(ᶜρ))

    u_end = Fields.level(ᶜu, Spaces.nlevels(axes(ᶜu)))
    u_endm1 = Fields.level(ᶜu, Spaces.nlevels(axes(ᶜu)) - 1)
    Boundary_value = Fields.Field(
        FT(2) .* Fields.field_values(u_end) .- Fields.field_values(u_endm1),
        axes(u_end),
    )
    R1 = Operators.RightBiasedC2F(; top = Operators.SetValue(Boundary_value))
    R2 = Operators.RightBiasedF2C(;)
    ᶜu_p1 .= R2.(R1.(ᶜu))

    v_end = Fields.level(ᶜv, Spaces.nlevels(axes(ᶜv)))
    v_endm1 = Fields.level(ᶜv, Spaces.nlevels(axes(ᶜv)) - 1)
    Boundary_value = Fields.Field(
        FT(2) .* Fields.field_values(v_end) .- Fields.field_values(v_endm1),
        axes(v_end),
    )
    R1 = Operators.RightBiasedC2F(; top = Operators.SetValue(Boundary_value))
    R2 = Operators.RightBiasedF2C(;)
    ᶜv_p1 .= R2.(R1.(ᶜv))

    Boundary_value = Fields.level(ᶜbf, Spaces.nlevels(axes(ᶜbf)))
    R1 = Operators.RightBiasedC2F(; top = Operators.SetValue(Boundary_value))
    R2 = Operators.RightBiasedF2C(;)
    ᶜbf_p1 .= R2.(R1.(ᶜbf))

    z_end = Fields.level(ᶜz, Spaces.nlevels(axes(ᶜz)))
    z_endm1 = Fields.level(ᶜz, Spaces.nlevels(axes(ᶜz)) - 1)
    Boundary_value = Fields.Field(
        FT(2) .* Fields.field_values(z_end) .- Fields.field_values(z_endm1),
        axes(z_end),
    )
    R1 = Operators.RightBiasedC2F(; top = Operators.SetValue(Boundary_value))
    R2 = Operators.RightBiasedF2C(;)
    ᶜz_p1 .= R2.(R1.(ᶜz))

    mask = BitVector(ones(nc))
    level_end= Spaces.nlevels(axes(ᶜρ))
    B1 = ntuple(i -> 0.0, Val(nc))

    for ink in 1:nk

        input_u = Base.Broadcast.broadcasted(
            tuple,
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
            gw_c,
            gw_flag,
            ᶜlevel,
            gw_source_ampl,
        )


        input_v = Base.Broadcast.broadcasted(
            tuple,
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
            gw_c,
            gw_flag,
            ᶜlevel,
            gw_source_ampl,
        )

        Operators.column_accumulate!(
            u_waveforcing,
            input_u;
            init = (Float32(0.0), mask, 0.0, B1),
            transform = first,
        ) do (wave_forcing, mask, Bsum, B0),
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
            c,
            flag,
            level,
            source_ampl,
        )

            FT1 = typeof(u_kp1)
            kwv = 2.0 * π / ((30.0 * (10.0^ink)) * 1.e3)
            k2 = kwv * kwv

            # loop over all wave lengths


            # here ᶜu has one additional level above model top
            fac = FT1(0.5) * (ρ_kp1 / ρ_source) * kwv / bf_kp1
            Hb = (z_kp1 - z_k) / log(ρ_k / ρ_kp1) # density scale height
            alp2 = FT1(0.25) / (Hb * Hb)
            ω_r = sqrt((bf_kp1 * bf_kp1 * k2) / (k2 + alp2)) # omc: (critical frequency that marks total internal reflection)

            fm = 0.0
            if level == 1
                #mask = ntuple(i -> true, Val(nc))
                mask .=1
                Bsum = 0.0
                B0 = ntuple(n -> sign(( (n - 1) * dc - cmax-u_kp1) ) * (
                        Bw * exp(
                            -log(2.0) *
                            ((( (n - 1) * dc - cmax) * flag + ( (n - 1) * dc - cmax-u_source)  * (1 - flag) - c0) / cw)^2,
                        ) +
                        Bn * exp(
                            -log(2.0) *
                            ((( (n - 1) * dc - cmax)  * flag +  ( (n - 1) * dc - cmax-u_source) * (1 - flag) - c0) / cn)^2,
                        )
                    ),Val(nc))
                    Bsum = sum(abs.(B0))   
            end
            for n in 1:nci
                # check only those waves which are still propagating, i.e., mask = 1.0
                if (mask[n]) == 1
                    c_hat = c[n] - u_kp1 # c0mu
                    # f phase speed matches the wind speed, remove c(n) from the set of propagating waves.
                    if c_hat == 0.0
                        mask[n] =0 
                    else
                        c_hat0 = c[n] - u_source
                        # define the criterion which determines if wave is reflected at this level (test).
                        test = abs(c_hat) * kwv - ω_r
                        if test >= 0.0
                            # wave has undergone total internal reflection. remove it from the propagating set.
                            mask[n] =0 
                        else
                            if level ==level_end
                                # this is added in MiMA implementation:
                                # all momentum flux that escapes across the model top
                                # is deposited to the extra level being added so that
                                # momentum flux is conserved
                                mask[n] =0 
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
                                Foc = B0[n] / FT1((c_hat)^3) - fac
                                if Foc >= 0.0 || (c_hat0 * c_hat <= 0.0)
                                    mask[n] =0 
                                    if level >= source_level
                                        fm = fm + B0[n]
                                    end
                                end
                            end
                        end # (test >= 0.0)
                    end #(c_hat == 0.0)
                end # mask = 0

            end # nc: phase speed loop

            # compute the gravity wave momentum flux forcing
            # obtained across the entire wave spectrum at this level.
            eps = calc_intermitency(ρ_source, source_ampl, nk, FT1(Bsum))
            if level >= source_level
                rbh = sqrt(ρ_k * ρ_kp1)
                wave_forcing =
                    (ρ_source / rbh) * FT1(fm) * eps / (z_kp1 - z_k)
            else
                wave_forcing = FT1(0.0)
            end
            return (wave_forcing, mask, Bsum, B0)

        end


    
        Operators.column_accumulate!(
            v_waveforcing,
            input_v;
            init = (Float32(0.0), mask, 0.0, B1),
            transform = first,
        ) do (wave_forcing, mask, Bsum, B0),
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
            c,
            flag,
            level,
            source_ampl,
        )

            FT2 = typeof(u_kp1)
            kwv = 2.0 * π / ((30.0 * (10.0^ink)) * 1.e3)
            k2 = kwv * kwv

            # loop over all wave lengths


            # here ᶜu has one additional level above model top
            fac = FT2(0.5) * (ρ_kp1 / ρ_source) * kwv / bf_kp1
            Hb = (z_kp1 - z_k) / log(ρ_k / ρ_kp1) # density scale height
            alp2 = FT2(0.25) / (Hb * Hb)
            ω_r = sqrt((bf_kp1 * bf_kp1 * k2) / (k2 + alp2)) # omc: (critical frequency that marks total internal reflection)

            fm = 0.0
            if level == 1
                #mask = ntuple(i -> true, Val(nc))
                mask .=1
                Bsum = 0.0
                B0 = ntuple(n -> sign(( (n - 1) * dc - cmax-u_kp1) ) * (
                        Bw * exp(
                            -log(2.0) *
                            ((( (n - 1) * dc - cmax) * flag + ( (n - 1) * dc - cmax-u_source)  * (1 - flag) - c0) / cw)^2,
                        ) +
                        Bn * exp(
                            -log(2.0) *
                            ((( (n - 1) * dc - cmax)  * flag +  ( (n - 1) * dc - cmax-u_source) * (1 - flag) - c0) / cn)^2,
                        )
                    ),Val(nc))
                    Bsum = sum(abs.(B0))   
            end
            for n in 1:nci
                # check only those waves which are still propagating, i.e., mask = 1.0
                if (mask[n]) == 1
                    c_hat = c[n] - u_kp1 # c0mu
                    # f phase speed matches the wind speed, remove c(n) from the set of propagating waves.
                    if c_hat == 0.0
                        mask[n] =0 
                    else
                        c_hat0 = c[n] - u_source
                        # define the criterion which determines if wave is reflected at this level (test).
                        test = abs(c_hat) * kwv - ω_r
                        if test >= 0.0
                            # wave has undergone total internal reflection. remove it from the propagating set.
                            mask[n] =0 
                        else
                            if level == level_end
                                # this is added in MiMA implementation:
                                # all momentum flux that escapes across the model top
                                # is deposited to the extra level being added so that
                                # momentum flux is conserved
                                mask[n] =0 
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
                                Foc = B0[n] / FT2((c_hat)^3) - fac
                                if Foc >= 0.0 || (c_hat0 * c_hat <= 0.0)
                                    mask[n] =0 
                                    if level >= source_level
                                        fm = fm + B0[n]
                                    end
                                end
                            end
                        end # (test >= 0.0)
                    end #(c_hat == 0.0)
                end # mask = 0

            end # nc: phase speed loop

            # compute the gravity wave momentum flux forcing
            # obtained across the entire wave spectrum at this level.
            eps = calc_intermitency(ρ_source, source_ampl, nk, FT2(Bsum))
            if level >= source_level
                rbh = sqrt(ρ_k * ρ_kp1)
                wave_forcing =
                    (ρ_source / rbh) * FT2(fm) * eps / (z_kp1 - z_k)
            else
                wave_forcing = FT2(0.0)
            end
            return (wave_forcing, mask, Bsum, B0)

        end

        u_waveforcing_top = similar(
            Fields.level(u_waveforcing, Spaces.nlevels(axes(u_waveforcing))),
        )
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

        v_waveforcing_top = similar(
            Fields.level(v_waveforcing, Spaces.nlevels(axes(v_waveforcing))),
        )
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

        gw_average!(u_waveforcing)
        gw_average!(v_waveforcing)

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

        @. uforcing = uforcing + u_waveforcing
        @. vforcing = vforcing + v_waveforcing

    end
    return nothing
end

# calculate the intermittency factor eps -> assuming constant Δc.

function calc_intermitency(ρ_source_level, source_ampl, nk, Bsum)
    return (source_ampl / ρ_source_level / nk) / Bsum
end

function gw_average!(wave_forcing)
    L1 = Operators.LeftBiasedC2F(; bottom = Operators.SetValue(0))
    L2 = Operators.LeftBiasedF2C(;)
    wave_forcing_m1 = L2.(L1.(wave_forcing))
    @. wave_forcing = 0.5 * (wave_forcing + wave_forcing_m1)
end


function gw_deposit(wave_forcing_top, wave_forcing, damp_level, level, height)
    if level >= damp_level
        wave_forcing =
            wave_forcing + wave_forcing_top / (height + 2 - damp_level)
    end
    return wave_forcing
end

function get_nc(::Val{nc}) where {nc}
    nc
end