#####
##### Non-orographic gravity wave parameterization
#####

import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry

non_orographic_gravity_wave_cache(atmos, Y) = non_orographic_gravity_wave_cache(
    atmos.non_orographic_gravity_wave,
    atmos.model_config,
    Y,
)

non_orographic_gravity_wave_cache(::Nothing, ::AbstractModelConfig, Y) =
    NamedTuple()

non_orographic_gravity_wave_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

function non_orographic_gravity_wave_cache(
    gw::NonOrographyGravityWave,
    ::SingleColumnModel,
    Y,
)
    FT = Spaces.undertype(axes(Y.c))
    (; source_height, Bw, Bn, Bt_0, dc, cmax, c0, nk, cw, cn) = gw

    nc = Int(floor(FT(2 * cmax / dc + 1)))
    c = [FT((n - 1) * dc - cmax) for n in 1:nc]

    return (;
        gw_source_height = source_height,
        gw_source_ampl = Bt_0 .* ones(FT, axes(Fields.level(Y.c.ρ, 1))),
        gw_Bw = Bw .* ones(FT, axes(Fields.level(Y.c.ρ, 1))),
        gw_Bn = Bn .* ones(FT, axes(Fields.level(Y.c.ρ, 1))),
        gw_c = c,
        gw_cw = cw .* ones(FT, axes(Fields.level(Y.c.ρ, 1))),
        gw_cn = cn .* ones(FT, axes(Fields.level(Y.c.ρ, 1))),
        gw_c0 = c0,
        gw_flag = ones(FT, axes(Fields.level(Y.c.ρ, 1))),
        gw_nk = Int(nk),
        ᶜbuoyancy_frequency = similar(Y.c.ρ),
        ᶜdTdz = similar(Y.c.ρ),
    )
end

function non_orographic_gravity_wave_cache(
    gw::NonOrographyGravityWave,
    ::SphericalModel,
    Y,
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
        gw_c0 = c0,
        gw_flag = gw_flag,
        gw_nk = Int(nk),
        ᶜbuoyancy_frequency = similar(Y.c.ρ),
        ᶜdTdz = similar(Y.c.ρ),
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
    (; ᶜts, ᶜT, ᶜdTdz, ᶜbuoyancy_frequency, params) = p
    model_config = p.atmos.model_config
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
    ) = p

    if model_config isa SingleColumnModel
        (; gw_source_height) = p
    elseif model_config isa SphericalModel
        (; gw_source_pressure, gw_damp_pressure) = p
    end
    ᶜρ = Y.c.ρ
    ᶜz = Fields.coordinate_field(Y.c).z
    FT = Spaces.undertype(axes(Y.c))
    # parameters
    thermo_params = CAP.thermodynamics_params(params)
    grav = CAP.grav(params)

    # compute buoyancy frequency
    @. ᶜT = TD.air_temperature(thermo_params, ᶜts)

    parent(ᶜdTdz) .= parent(Geometry.WVector.(ᶜgradᵥ.(ᶠinterp.(ᶜT))))

    ᶜbuoyancy_frequency =
        @. (grav / ᶜT) * (ᶜdTdz + grav / TD.cp_m(thermo_params, ᶜts))
    ᶜbuoyancy_frequency = @. ifelse(
        ᶜbuoyancy_frequency < FT(2.5e-5),
        FT(sqrt(2.5e-5)),
        sqrt(abs(ᶜbuoyancy_frequency)),
    ) # to avoid small numbers

    if model_config isa SingleColumnModel
        # source level: the index of the level that is closest to the source height
        source_level = similar(Fields.level(Y.c.ρ, 1))
        Fields.bycolumn(axes(ᶜρ)) do colidx
            parent(source_level[colidx]) .=
                argmin(abs.(parent(ᶜz[colidx]) .- gw_source_height))[1]
        end
        # damp level: for now we only deposit to top level for column setup
        damp_level = similar(Fields.level(Y.c.ρ, 1))
        Fields.bycolumn(axes(ᶜρ)) do colidx
            parent(damp_level[colidx]) .= length(parent(ᶜz[colidx]))
        end
    elseif model_config isa SphericalModel
        (; ᶜp) = p
        # source level: the index of the highest level whose pressure is higher than source pressure
        source_level = similar(Fields.level(Y.c.ρ, 1))
        Fields.bycolumn(axes(ᶜρ)) do colidx
            parent(source_level[colidx]) .=
                findlast(parent(ᶜp[colidx]) .> gw_source_pressure)[1]
        end
        # damp level: the index of the lowest level whose pressure is lower than the damp pressure
        damp_level = similar(Fields.level(Y.c.ρ, 1))
        Fields.bycolumn(axes(ᶜρ)) do colidx
            if sum(parent(ᶜp[colidx]) .< gw_damp_pressure) == 0
                parent(damp_level[colidx]) .= length(parent(ᶜz[colidx]))
            else
                parent(damp_level[colidx]) .=
                    findfirst(parent(ᶜp[colidx]) .< gw_damp_pressure)[1]
            end
        end
    end

    # prepare physical uv input variables for gravity_wave_forcing()
    u_phy = Geometry.UVVector.(Y.c.uₕ).components.data.:1
    v_phy = Geometry.UVVector.(Y.c.uₕ).components.data.:2

    # a place holder to store physical forcing on uv
    uforcing = ones(axes(u_phy))
    vforcing = ones(axes(u_phy))

    # GW parameterization applied bycolume
    Fields.bycolumn(axes(ᶜρ)) do colidx
        parent(uforcing[colidx]) .= non_orographic_gravity_wave_forcing(
            copy(vec(parent(u_phy[colidx]))),
            copy(vec(parent(ᶜbuoyancy_frequency[colidx]))),
            copy(vec(parent(ᶜρ[colidx]))),
            copy(vec(parent(ᶜz[colidx]))),
            Int(parent(source_level[colidx])[1]),
            Int(parent(damp_level[colidx])[1]),
            parent(gw_source_ampl[colidx])[1],
            parent(gw_Bw[colidx])[1],
            parent(gw_Bn[colidx])[1],
            parent(gw_cw[colidx])[1],
            parent(gw_cn[colidx])[1],
            parent(gw_flag[colidx])[1],
            gw_c,
            gw_c0,
            gw_nk,
        )

        parent(vforcing[colidx]) .= non_orographic_gravity_wave_forcing(
            copy(vec(parent(v_phy[colidx]))),
            copy(vec(parent(ᶜbuoyancy_frequency[colidx]))),
            copy(vec(parent(ᶜρ[colidx]))),
            copy(vec(parent(ᶜz[colidx]))),
            Int(parent(source_level[colidx])[1]),
            Int(parent(damp_level[colidx])[1]),
            parent(gw_source_ampl[colidx])[1],
            parent(gw_Bw)[1],
            parent(gw_Bn)[1],
            parent(gw_cw)[1],
            parent(gw_cn)[1],
            parent(gw_flag)[1],
            gw_c,
            gw_c0,
            gw_nk,
        )

    end

    # physical uv forcing converted to Covariant12Vector and added up to uₕ tendencies
    @. Yₜ.c.uₕ +=
        Geometry.Covariant12Vector.(Geometry.UVVector.(uforcing, vforcing))
    return nothing
end

function non_orographic_gravity_wave_forcing(
    ᶜu,
    ᶜbf,
    ᶜρ,
    ᶜz,
    source_level,
    damp_level,
    source_ampl,
    Bw,
    Bn,
    cw,
    cn,
    flag,
    c,
    c0,
    nk,
)
    FT = eltype(ᶜz)
    # add an extra layer above model top so that forcing between the very top 
    # model layer and the upper boundary can be calculated 
    append!(ᶜu, FT(2) * ᶜu[end] - ᶜu[end - 1])
    append!(ᶜρ, ᶜρ[end] * ᶜρ[end] / ᶜρ[end - 1])
    append!(ᶜbf, ᶜbf[end])
    append!(ᶜz, FT(2) * ᶜz[end] - ᶜz[end - 1])

    # wave spectra and the source amplitude
    nc = length(c)
    c_hat0 = c .- ᶜu[source_level] # c0mu0
    Bw_exp = @. exp(-log(2.0) * ((c * flag + c_hat0 * (1 - flag) - c0) / cw)^2)
    Bn_exp = @. exp(-log(2.0) * ((c * flag + c_hat0 * (1 - flag) - c0) / cn)^2)
    B0 = @. sign(c_hat0) * (Bw * Bw_exp + Bn * Bn_exp)

    Bsum = sum(abs.(B0))
    if (Bsum == 0.0)
        error("zero flux input at source level")
    end
    # intermittency
    eps = calc_intermitency(ᶜρ[source_level], source_ampl, nk, Bsum)

    # horizontal wave length
    kwv = [2.0 * π / ((30.0 * (10.0^n)) * 1.e3) for n in 1:nk]
    k2 = kwv .* kwv

    # forcing
    wave_forcing = zeros(length(ᶜu))
    gwf = zeros(length(ᶜu) - 1)
    for ink in 1:nk # loop over all wave lengths

        mask = ones(nc)  # mask to determine which waves propagate upward
        for k in source_level:length(ᶜu) # here ᶜu has one additional level above model top
            fac = FT(0.5) * (ᶜρ[k] / ᶜρ[source_level]) * kwv[ink] / ᶜbf[k]

            ᶜHb = -(ᶜz[k] - ᶜz[k - 1]) / log(ᶜρ[k] / ᶜρ[k - 1])  # density scale height
            alp2 = 0.25 / (ᶜHb * ᶜHb)
            ω_r = sqrt((ᶜbf[k] * ᶜbf[k] * k2[ink]) / (k2[ink] + alp2)) # omc: (critical frequency that marks total internal reflection)

            fm = FT(0)
            for n in 1:nc
                # check only those waves which are still propagating, i.e., mask = 1.0
                if (mask[n]) == 1.0
                    c_hat = c[n] - ᶜu[k] # c0mu
                    # f phase speed matches the wind speed, remove c(n) from the set of propagating waves.
                    if c_hat == 0.0
                        mask[n] = 0.0
                    else
                        # define the criterion which determines if wave is reflected at this level (test).
                        test = abs(c_hat) * kwv[ink] - ω_r
                        if test >= 0.0
                            # wave has undergone total internal reflection. remove it from the propagating set.
                            mask[n] = 0.0
                        else
                            if k == length(ᶜu)
                                # this is added in MiMA implementation:
                                # all momentum flux that escapes across the model top
                                # is deposited to the extra level being added so that
                                # momentum flux is conserved
                                mask[n] = 0.0
                                if k > source_level
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
                                if Foc >= 0.0 || (c_hat0[n] * c_hat <= 0.0)
                                    mask[n] = 0.0
                                    if k > source_level
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
            if k > source_level
                rbh = sqrt(ᶜρ[k] * ᶜρ[k - 1])
                wave_forcing[k] =
                    (ᶜρ[source_level] / rbh) * fm * eps / (ᶜz[k] - ᶜz[k - 1])
                if k == length(ᶜu)
                    wave_forcing[k - 1] = 0.5 * wave_forcing[k - 1]
                else
                    wave_forcing[k - 1] =
                        0.5 * (wave_forcing[k - 1] + wave_forcing[k])
                end
            else
                wave_forcing[k] = 0.0
            end

        end # k

        # model top: deposit remaining momentum flux that goes across the model top 
        # to the levels above the damp level
        # This is not included in Joan Alexander's original code nor the GFDL implementation;
        # but is added in MiMA based on Tiffany Shaw's paper: 
        # https://journals.ametsoc.org/view/journals/clim/22/10/2009jcli2688.1.xml?tab_body=pdf
        for k in damp_level:(length(ᶜu) - 1)
            wave_forcing[k] =
                wave_forcing[k] +
                wave_forcing[end] / (length(ᶜu) + 1 - damp_level)
        end

        # forcing
        for k in source_level:(length(ᶜu) - 1)
            gwf[k] = gwf[k] + wave_forcing[k]
        end

    end # ink

    return gwf
end

# calculate the intermittency factor eps -> assuming constant Δc.

function calc_intermitency(ρ_source_level, source_ampl, nk, Bsum)
    return (source_ampl / ρ_source_level / nk) / Bsum
end
