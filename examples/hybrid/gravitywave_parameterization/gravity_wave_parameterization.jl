function gravity_wave_cache(
    Y,
    ::Type{FT};
    source_height = FT(15000),
    Bm = FT(1.2),
    F_S0 = FT(4e-3),
    dc = FT(0.6),
    cmax = FT(99.6),
    c0 = FT(0),
    kwv = FT(2π / 100e5),
    cw = FT(40.0),
) where {FT}

    nc = Int(floor(FT(2 * cmax / dc + 1)))
    c = [FT((n - 1) * dc - cmax) for n in 1:nc]

    return (;
        gw_source_height = source_height,
        gw_F_S0 = F_S0,
        gw_Bm = Bm,
        gw_c = c,
        gw_cw = cw,
        gw_c0 = c0,
        gw_nk = length(kwv),
        gw_k = kwv,
        gw_k2 = kwv .^ 2,
        ᶜbuoyancy_frequency = similar(Y.c.ρ),
        ᶜdTdz = similar(Y.c.ρ),
    )
end

function gravity_wave_tendency!(Yₜ, Y, p, t)
    #unpack
    (;
        gw_source_height,
        gw_Bm,
        gw_F_S0,
        gw_c,
        gw_cw,
        gw_c0,
        gw_nk,
        gw_k,
        gw_k2,
    ) = p
    (; ᶜts, ᶜT, ᶜdTdz, ᶜbuoyancy_frequency, params) = p
    ᶜρ = Y.c.ρ
    # parameters
    thermo_params = CAP.thermodynamics_params(params)
    grav = FT(CAP.grav(params))

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
    # alternative
    # ᶜbuoyancy_frequency = [i > 2.5e-5 ? sqrt(i) : sqrt(2.5e-5)  for i in ᶜbuoyancy_frequency]
    # TODO: create an extra layer at model top so that the gravity wave forcing
    # .     occurring between the topmost model level and the upper boundary
    # .     may be calculated
    #

    # source level: get the index of the level that is closest to the source height (GFDL uses the fist level below instead)
    source_level = argmin(
        abs.(
            unique(parent(Fields.coordinate_field(Y.c).z) .- gw_source_height)
        ),
    )
    ᶠz = Fields.coordinate_field(Y.f).z

    # TODO: prepare input variables for gravity_wave_forcing()
    #       1. grab column wise data
    #       2. convert coviant uv to physical uv
    # need to make everything an array befere
    u_phy = Geometry.UVVector.(Y.c.uₕ).components.data.:1
    v_phy = Geometry.UVVector.(Y.c.uₕ).components.data.:2

    uforcing = ones(axes(u_phy))
    vforcing = similar(v_phy)
    # bycolume
    Fields.bycolumn(axes(ᶜρ)) do colidx
        parent(uforcing[colidx]) .= gravity_wave_forcing(
            parent(u_phy[colidx]),
            source_level,
            gw_F_S0,
            gw_Bm,
            gw_c,
            gw_cw,
            gw_c0,
            gw_nk,
            gw_k,
            gw_k2,
            parent(ᶜbuoyancy_frequency[colidx]),
            parent(ᶜρ[colidx]),
            parent(ᶠz[colidx]),
        )
        parent(vforcing[colidx]) .= gravity_wave_forcing(
            parent(v_phy[colidx]),
            source_level,
            gw_F_S0,
            gw_Bm,
            gw_c,
            gw_cw,
            gw_c0,
            gw_nk,
            gw_k,
            gw_k2,
            parent(ᶜbuoyancy_frequency[colidx]),
            parent(ᶜρ[colidx]),
            parent(ᶠz[colidx]),
        )

    end

    @. Yₜ.c.uₕ +=
        Geometry.Covariant12Vector.(Geometry.UVVector.(uforcing, vforcing))

end

function gravity_wave_forcing(
    ᶜu,
    source_level,
    F_S0,
    Bm,
    c,
    cw,
    c0,
    nk,
    kwv,
    k2,
    ᶜbf,
    ᶜρ,
    ᶠz,
)

    nc = length(c)

    # define wave momentum flux (B0) at source level for each phase
    # speed n, and the sum over all phase speeds (Bsum), which is needed
    # to calculate the intermittency.

    c_hat0 = c .- ᶜu[source_level]
    # In GFDL code, flag is always 1: c = c0 * flag + c_hat0 * (1 - flag)
    Bexp = @. exp(-log(2.0) * ((c - c0) / cw)^2)
    B0 = @. sign(c_hat0) * Bm * Bexp
    # In GFDL code, it is: B0 = @. sign(c_hat0) * (Bw * Bexp + Bn * Bexp)
    # where Bw = Bm is the wide band and Bn is the narrow band
    Bsum = sum(abs.(B0))
    if (Bsum == 0.0)
        error("zero flux input at source level")
    end
    # define the intermittency factor eps. Assumes constant Δc.
    eps = (F_S0 / ᶜρ[source_level] / nk) / Bsum # in GFDL code: eps = (ampl * 1.5 / nk) / Bsum, ampl is equivalent to F_S)/ρ_source 

    ᶜdz = ᶠz[2:end] - ᶠz[1:(end - 1)]
    wave_forcing = zeros(nc)
    gwf = zeros(length(ᶜu))
    for ink in 1:nk # loop over all wave lengths

        mask = ones(nc)  # mask to determine which waves propagate upward
        for k in source_level:length(ᶜu)
            fac = FT(0.5) * (ᶜρ[k] / ᶜρ[source_level]) * kwv[ink] / ᶜbf[k]

            ᶜHb = ᶜdz[k] / log(ᶜρ[k - 1] / ᶜρ[k])  # density scale height
            alp2 = 0.25 / (ᶜHb * ᶜHb)
            ω_r = sqrt((ᶜbf[k] * ᶜbf[k] * k2[ink]) / (k2[ink] + alp2)) # ω_r (critical frequency that marks total internal reflection)

            fm = FT(0)
            for n in 1:nc
                # check only those waves which are still propagating, i.e., mask = 1.0
                if (mask[n]) == 1.0
                    c_hat = c[n] - ᶜu[k]
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
                            # if wave is not reflected at this level, determine if it is
                            # breaking at this level (Foc >= 0), or if wave speed relative to
                            # windspeed has changed sign from its value at the source level
                            # (c_hat0[n] * c_hat <= 0). if it is above the source level and is
                            # breaking, then add its momentum flux to the accumulated sum at
                            # this level.
                            # set mask=0.0 to remove phase speed band c[n] from the set of active 
                            # waves moving upwards to the next level.
                            if c_hat0[n] * c_hat <= 0.0
                                mask[n] = 0.0
                                if k > source_level
                                    fm = fm + B0[n]
                                end
                            else
                                Foc = B0[n] / (c_hat)^3 - fac
                                if Foc >= 0.0
                                    mask[n] = 0.0
                                    if k > source_level
                                        fm = fm + B0[n]
                                    end
                                end
                            end
                        end # (test >= 0.0)
                    end #(c_hat == 0.0)
                end # mask = 0

            end # phase speed loop

            # TODO: GFDL option to dump remaining flux at the top of the model

            # compute the gravity wave momentum flux forcing 
            # obtained across the entire wave spectrum at this level.
            if k > source_level
                rbh = sqrt(ᶜρ[k] * ᶜρ[k - 1])
                wave_forcing[k] = (ᶜρ[source_level] / rbh) * fm * eps / ᶜdz[k]
                wave_forcing[k - 1] =
                    0.5 * (wave_forcing[k - 1] + wave_forcing[k])
            else
                wave_forcing[k] = 0.0
            end

        end # k

        for k in source_level:length(ᶜu)
            gwf[k] = gwf[k] + wave_forcing[k]
        end

    end # ink 

    return gwf
end
