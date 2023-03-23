function update_aux!(
    edmf::EDMFModel,
    grid::Grid,
    state::State,
    surf,
    param_set::APS,
    t::Real,
    Δt::Real,
)
    #####
    ##### Unpack common variables
    #####

    a_bulk_bcs = a_bulk_boundary_conditions(surf, edmf)
    Ifb = CCO.InterpolateC2F(; a_bulk_bcs...)
    thermo_params = TCP.thermodynamics_params(param_set)
    microphys_params = TCP.microphysics_params(param_set)
    N_up = n_updrafts(edmf)
    kc_surf = kc_surface(grid)
    kf_surf = kf_surface(grid)
    c_m = mixing_length_params(edmf).c_m
    KM = center_aux_turbconv(state).KM
    KH = center_aux_turbconv(state).KH
    oblength = obukhov_length(surf)
    FT = float_type(state)
    prog_gm = center_prog_grid_mean(state)
    prog_gm_f = face_prog_grid_mean(state)
    aux_up = center_aux_updrafts(state)
    aux_up_f = face_aux_updrafts(state)
    aux_en = center_aux_environment(state)
    aux_en_f = face_aux_environment(state)
    aux_gm = center_aux_grid_mean(state)
    aux_gm_f = face_aux_grid_mean(state)
    aux_tc_f = face_aux_turbconv(state)
    aux_tc = center_aux_turbconv(state)
    aux_bulk = center_aux_bulk(state)
    prog_en = center_prog_environment(state)
    aux_en_2m = center_aux_environment_2m(state)
    prog_up = center_prog_updrafts(state)
    prog_up_f = face_prog_updrafts(state)
    ρ_f = aux_gm_f.ρ
    p_c = center_aux_grid_mean_p(state)
    ρ_c = prog_gm.ρ
    aux_en_unsat = aux_en.unsat
    aux_en_sat = aux_en.sat
    m_entr_detr = aux_tc.ϕ_temporary
    ∇m_entr_detr = aux_tc.ψ_temporary
    wvec = CC.Geometry.WVector
    max_area = edmf.max_area
    ts_gm = center_aux_grid_mean_ts(state)
    ts_env = center_aux_environment(state).ts
    e_kin = center_aux_grid_mean_e_kin(state)

    prog_gm_uₕ = grid_mean_uₕ(state)
    Ic = CCO.InterpolateF2C()
    #####
    ##### center variables
    #####
    C123 = CCG.Covariant123Vector

    @inbounds for i in 1:N_up
        @. aux_up[i].e_kin =
            LA.norm_sqr(
                C123(prog_gm_uₕ) + C123(Ic(CCG.WVector(prog_up_f[i].w))),
            ) / 2
    end

    @inbounds for k in real_center_indices(grid)
        #####
        ##### Set primitive variables
        #####
        e_pot = geopotential(thermo_params, grid.zc.z[k])
        @inbounds for i in 1:N_up
            if prog_up[i].ρarea[k] / ρ_c[k] >= edmf.minimum_area
                aux_up[i].e_tot[k] =
                    prog_up[i].ρae_tot[k] / prog_up[i].ρarea[k]
                aux_up[i].q_tot[k] = prog_up[i].ρaq_tot[k] / prog_up[i].ρarea[k]
                aux_up[i].area[k] = prog_up[i].ρarea[k] / ρ_c[k]
            else
                aux_up[i].e_tot[k] = aux_gm.e_tot[k]
                aux_up[i].q_tot[k] = aux_gm.q_tot[k]
                aux_up[i].area[k] = 0
                aux_up[i].e_kin[k] = e_kin[k]
            end
            e_int = aux_up[i].e_tot[k] - aux_up[i].e_kin[k] - e_pot
            ts_up_i = thermo_state_peq(
            #ts_up_i = if edmf.moisture_model isa DryModel
            #    TD.PhaseDry_pθ(thermo_params, p_c[k], aux_up[i].θ_liq_ice[k])
            #elseif edmf.moisture_model isa EquilMoistModel
            #    TD.PhaseEquil_pθq(
                    thermo_params,
                    p_c[k],
                    e_int,
                    aux_up[i].q_tot[k],
                )
            aux_up[i].θ_liq_ice[k] = TD.liquid_ice_pottemp(thermo_params, ts_up_i)
            #elseif edmf.moisture_model isa NonEquilMoistModel
            #    error("Unsupported moisture model")
            #end
            #aux_up[i].e_tot[k] = TD.total_energy(
            #    thermo_params,
            #    ts_up_i,
            #    aux_up[i].e_kin[k],
            #    e_pot,
            #)
            aux_up[i].h_tot[k] = TD.total_specific_enthalpy(
                thermo_params,
                ts_up_i,
                aux_up[i].e_tot[k],
            )
        end

        #####
        ##### compute bulk
        #####
        aux_bulk.q_tot[k] = 0
        aux_bulk.h_tot[k] = 0
        aux_bulk.area[k] = sum(i -> aux_up[i].area[k], 1:N_up)
        if aux_bulk.area[k] > 0
            @inbounds for i in 1:N_up
                a_k = aux_up[i].area[k]
                a_bulk_k = aux_bulk.area[k]
                aux_bulk.q_tot[k] += a_k * aux_up[i].q_tot[k] / a_bulk_k
                aux_bulk.h_tot[k] += a_k * aux_up[i].h_tot[k] / a_bulk_k
            end
        else
            aux_bulk.q_tot[k] = aux_gm.q_tot[k]
            aux_bulk.h_tot[k] = aux_gm.h_tot[k]
        end
        aux_en.area[k] = 1 - aux_bulk.area[k]
        aux_en.tke[k] = prog_en.ρatke[k] / (ρ_c[k] * aux_en.area[k])
    end

    @. aux_en_f.w = prog_gm_f.w / (1 - Ifb(aux_bulk.area))
    @inbounds for i in 1:N_up
        @. aux_en_f.w -=
            Ifb(aux_up[i].area) * prog_up_f[i].w / (1 - Ifb(aux_bulk.area))
    end

    @. aux_en.e_kin =
        LA.norm_sqr(C123(prog_gm_uₕ) + C123(Ic(wvec(aux_en_f.w)))) / 2

    @inbounds for k in real_center_indices(grid)
        e_pot = geopotential(thermo_params, grid.zc.z[k])

        #####
        ##### decompose_environment
        #####
        a_bulk_c = aux_bulk.area[k]
        val1 = 1 / (1 - a_bulk_c)
        val2 = a_bulk_c * val1
        aux_en.q_tot[k] =
            max(val1 * aux_gm.q_tot[k] - val2 * aux_bulk.q_tot[k], 0) #Yair - this is here to prevent negative QT
        aux_en.h_tot[k] = val1 * aux_gm.h_tot[k] - val2 * aux_bulk.h_tot[k]

        h_en = enthalpy(aux_en.h_tot[k], e_pot, aux_en.e_kin[k])
        ts_env[k] = if edmf.moisture_model isa DryModel
            TD.PhaseDry_ph(thermo_params, p_c[k], h_en)
        elseif edmf.moisture_model isa EquilMoistModel
            TD.PhaseEquil_phq(thermo_params, p_c[k], h_en, aux_en.q_tot[k])
        elseif edmf.moisture_model isa NonEquilMoistModel
            error("Add support got non-equilibrium thermo states")
        end
        ts_en = ts_env[k]
        aux_en.θ_liq_ice[k] = TD.liquid_ice_pottemp(thermo_params, ts_en)
        aux_en.e_tot[k] =
            TD.total_energy(thermo_params, ts_en, aux_en.e_kin[k], e_pot)
        aux_en.T[k] = TD.air_temperature(thermo_params, ts_en)
        aux_en.θ_virt[k] = TD.virtual_pottemp(thermo_params, ts_en)
        aux_en.θ_dry[k] = TD.dry_pottemp(thermo_params, ts_en)
        aux_en.q_liq[k] = TD.liquid_specific_humidity(thermo_params, ts_en)
        aux_en.q_ice[k] = TD.ice_specific_humidity(thermo_params, ts_en)
        rho = TD.air_density(thermo_params, ts_en)
        aux_en.buoy[k] = buoyancy_c(thermo_params, ρ_c[k], rho)
        aux_en.RH[k] = TD.relative_humidity(thermo_params, ts_en)
    end

    microphysics(grid, state, edmf, edmf.precip_model, Δt, param_set)

    @inbounds for k in real_center_indices(grid)
        a_bulk_c = aux_bulk.area[k]
        @inbounds for i in 1:N_up
            if aux_up[i].area[k] < edmf.minimum_area &&
               k > kc_surf &&
               aux_up[i].area[k - 1] > 0.0
                qt = aux_up[i].q_tot[k - 1]
                h = aux_up[i].θ_liq_ice[k - 1]
                ts_up = if edmf.moisture_model isa DryModel
                    TD.PhaseDry_pθ(thermo_params, p_c[k], h)
                elseif edmf.moisture_model isa EquilMoistModel
                    TD.PhaseEquil_pθq(thermo_params, p_c[k], h, qt)
                elseif edmf.moisture_model isa NonEquilMoistModel
                    error("Unsupported moisture model")
                end

            else
                ts_up = if edmf.moisture_model isa DryModel
                    TD.PhaseDry_pθ(thermo_params, p_c[k], aux_up[i].θ_liq_ice[k])
                elseif edmf.moisture_model isa EquilMoistModel
                    TD.PhaseEquil_pθq(
                        thermo_params,
                        p_c[k],
                        aux_up[i].θ_liq_ice[k],
                        aux_up[i].q_tot[k],
                    )
                elseif edmf.moisture_model isa NonEquilMoistModel
                    error("Unsupported moisture model")
                end
            end
            aux_up[i].q_liq[k] =
                TD.liquid_specific_humidity(thermo_params, ts_up)
            aux_up[i].q_ice[k] = TD.ice_specific_humidity(thermo_params, ts_up)
            aux_up[i].T[k] = TD.air_temperature(thermo_params, ts_up)
            ρ = TD.air_density(thermo_params, ts_up)
            aux_up[i].buoy[k] = buoyancy_c(thermo_params, ρ_c[k], ρ)
            aux_up[i].RH[k] = TD.relative_humidity(thermo_params, ts_up)
        end
        aux_gm.buoy[k] = (1.0 - aux_bulk.area[k]) * aux_en.buoy[k]
        @inbounds for i in 1:N_up
            aux_gm.buoy[k] += aux_up[i].area[k] * aux_up[i].buoy[k]
        end
        @inbounds for i in 1:N_up
            aux_up[i].buoy[k] -= aux_gm.buoy[k]
        end
        aux_en.buoy[k] -= aux_gm.buoy[k]


        #####
        ##### compute bulk thermodynamics
        #####
        aux_bulk.q_liq[k] = 0
        aux_bulk.q_ice[k] = 0
        aux_bulk.T[k] = 0
        aux_bulk.buoy[k] = 0
        if a_bulk_c > 0
            @inbounds for i in 1:N_up
                aux_bulk.q_liq[k] +=
                    aux_up[i].area[k] * aux_up[i].q_liq[k] / a_bulk_c
                aux_bulk.q_ice[k] +=
                    aux_up[i].area[k] * aux_up[i].q_ice[k] / a_bulk_c
                aux_bulk.T[k] += aux_up[i].area[k] * aux_up[i].T[k] / a_bulk_c
                aux_bulk.buoy[k] +=
                    aux_up[i].area[k] * aux_up[i].buoy[k] / a_bulk_c
            end
        else
            aux_bulk.T[k] = aux_en.T[k]
        end

        #####
        ##### update_GMV_diagnostics
        #####
        aux_gm.q_liq[k] = (
            aux_bulk.area[k] * aux_bulk.q_liq[k] +
            (1 - aux_bulk.area[k]) * aux_en.q_liq[k]
        )
        aux_gm.q_ice[k] = (
            aux_bulk.area[k] * aux_bulk.q_ice[k] +
            (1 - aux_bulk.area[k]) * aux_en.q_ice[k]
        )
        aux_gm.T[k] = (
            aux_bulk.area[k] * aux_bulk.T[k] +
            (1 - aux_bulk.area[k]) * aux_en.T[k]
        )
        aux_gm.buoy[k] = (
            aux_bulk.area[k] * aux_bulk.buoy[k] +
            (1 - aux_bulk.area[k]) * aux_en.buoy[k]
        )

        has_condensate =
            TD.has_condensate(aux_bulk.q_liq[k] + aux_bulk.q_ice[k])
        aux_bulk.cloud_fraction[k] = if has_condensate && a_bulk_c > 1e-3
            1
        else
            0
        end

        @. aux_gm.tke = aux_en.area * aux_en.tke
        @. aux_gm.tke +=
            0.5 *
            aux_en.area *
            LA.norm_sqr(C123(Ic(wvec(aux_en_f.w - prog_gm_f.w))))
        @inbounds for i in 1:N_up
            @. aux_gm.tke +=
                0.5 *
                aux_up[i].area *
                LA.norm_sqr(C123(Ic(wvec(prog_up_f[i].w - prog_gm_f.w))))
        end
    end
    #####
    ##### face variables: diagnose primitive, diagnose env and compute bulk
    #####
    parent(aux_tc_f.bulk.w) .= 0
    @inbounds for i in 1:N_up
        a_up = aux_up[i].area
        a_up_bcs = a_up_boundary_conditions(surf, edmf, i)
        Ifu = CCO.InterpolateC2F(; a_up_bcs...)
        @. aux_tc_f.bulk.w += ifelse(
            Ifb(aux_bulk.area) > 0,
            Ifu(a_up) * prog_up_f[i].w / Ifb(aux_bulk.area),
            CCG.Covariant3Vector(FT(0)),
        )
    end

    #####
    ##### compute_updraft_closures
    #####
    # entrainment and detrainment
    @inbounds for i in 1:N_up
        @. aux_up[i].entr = FT(5e-4)
        @. aux_up[i].detr = pi_groups_detrainment!(
            aux_gm.tke,
            aux_up[i].area,
            aux_up[i].RH,
            aux_en.area,
            aux_en.tke,
            aux_en.RH,
        )
    end
    # updraft pressure
    # TODO @. aux_up_f[i].nh_pressure = compute_nh_pressure(...)
    compute_nh_pressure!(state, grid, edmf, surf)

    #####
    ##### compute_eddy_diffusivities_tke
    #####

    # Subdomain exchange term
    Ic = CCO.InterpolateF2C()
    b_exch = center_aux_turbconv(state).b_exch
    parent(b_exch) .= 0
    w_en = aux_en_f.w
    @inbounds for i in 1:N_up
        w_up = prog_up_f[i].w
        @. b_exch +=
            aux_up[i].area *
            Ic(wcomponent(CCG.WVector(w_up))) *
            aux_up[i].detr / aux_en.area *
            (1 / 2 * (Ic(wcomponent(CCG.WVector(w_up - w_en))))^2 - aux_en.tke)
    end

    Shear² = center_aux_turbconv(state).Shear²
    ∂qt∂z = center_aux_turbconv(state).∂qt∂z
    ∂θl∂z = center_aux_turbconv(state).∂θl∂z
    ∂θv∂z = center_aux_turbconv(state).∂θv∂z
    ∂qt∂z_sat = center_aux_turbconv(state).∂qt∂z_sat
    ∂θl∂z_sat = center_aux_turbconv(state).∂θl∂z_sat
    ∂θv∂z_unsat = center_aux_turbconv(state).∂θv∂z_unsat

    ∇0_bcs = (; bottom = CCO.Extrapolate(), top = CCO.Extrapolate())
    If0 = CCO.InterpolateC2F(; ∇0_bcs...)
    uvw = face_aux_turbconv(state).uvw

    uₕ_gm = grid_mean_uₕ(state)
    w_en = aux_en_f.w
    # compute shear
    k̂ = center_aux_turbconv(state).k̂

    # TODO: Will need to be changed with topography
    local_geometry = CC.Fields.local_geometry_field(axes(ρ_c))
    @. k̂ = CCG.Contravariant3Vector(CCG.WVector(FT(1)), local_geometry)
    Ifuₕ = uₕ_bcs()
    ∇uvw = CCO.GradientF2C()
    @. uvw = C123(Ifuₕ(uₕ_gm)) + C123(CCG.WVector(w_en))
    @. Shear² = LA.norm_sqr(adjoint(∇uvw(uvw)) * k̂)

    ∇c = CCO.DivergenceF2C()
    q_tot_en = aux_en.q_tot
    θ_liq_ice_en = aux_en.θ_liq_ice
    θ_virt_en = aux_en.θ_virt
    @. ∂qt∂z = ∇c(wvec(If0(q_tot_en)))
    @. ∂θl∂z = ∇c(wvec(If0(θ_liq_ice_en)))
    @. ∂θv∂z = ∇c(wvec(If0(θ_virt_en)))

    # Second order approximation: Use dry and cloudy environmental fields.
    cf = aux_en.cloud_fraction
    shm = copy(cf)
    pshm = parent(shm)
    shrink_mask!(pshm, vec(cf))
    mix_len_params = mixing_length_params(edmf)

    # Since NaN*0 ≠ 0, we need to conditionally replace
    # our gradients by their default values.
    @. ∂qt∂z_sat = ∇c(wvec(If0(aux_en_sat.q_tot)))
    @. ∂θl∂z_sat = ∇c(wvec(If0(aux_en_sat.θ_liq_ice)))
    @. ∂θv∂z_unsat = ∇c(wvec(If0(aux_en_unsat.θ_virt)))
    @. ∂qt∂z_sat = ifelse(shm == 0, ∂qt∂z, ∂qt∂z_sat)
    @. ∂θl∂z_sat = ifelse(shm == 0, ∂θl∂z, ∂θl∂z_sat)
    @. ∂θv∂z_unsat = ifelse(shm == 0, ∂θv∂z, ∂θv∂z_unsat)

    @inbounds for k in real_center_indices(grid)

        # buoyancy_gradients
        if edmf.bg_closure == BuoyGradMean()
            # First order approximation: Use environmental mean fields.
            ts_en = ts_env[k]
            bg_kwargs = (;
                t_sat = aux_en.T[k],
                qv_sat = TD.vapor_specific_humidity(thermo_params, ts_en),
                qt_sat = aux_en.q_tot[k],
                θ_sat = aux_en.θ_dry[k],
                θ_liq_ice_sat = aux_en.θ_liq_ice[k],
                ∂θv∂z_unsat = ∂θv∂z[k],
                ∂qt∂z_sat = ∂qt∂z[k],
                ∂θl∂z_sat = ∂θl∂z[k],
                p = p_c[k],
                en_cld_frac = aux_en.cloud_fraction[k],
                ρ = ρ_c[k],
            )
            bg_model = EnvBuoyGrad(edmf.bg_closure; bg_kwargs...)

        elseif edmf.bg_closure == BuoyGradQuadratures()

            bg_kwargs = (;
                t_sat = aux_en_sat.T[k],
                qv_sat = aux_en_sat.q_vap[k],
                qt_sat = aux_en_sat.q_tot[k],
                θ_sat = aux_en_sat.θ_dry[k],
                θ_liq_ice_sat = aux_en_sat.θ_liq_ice[k],
                ∂θv∂z_unsat = ∂θv∂z_unsat[k],
                ∂qt∂z_sat = ∂qt∂z_sat[k],
                ∂θl∂z_sat = ∂θl∂z_sat[k],
                p = p_c[k],
                en_cld_frac = aux_en.cloud_fraction[k],
                ρ = ρ_c[k],
            )
            bg_model = EnvBuoyGrad(edmf.bg_closure; bg_kwargs...)
        else
            error(
                "Something went wrong. The buoyancy gradient model is not specified",
            )
        end
        bg = buoyancy_gradients(param_set, edmf.moisture_model, bg_model)

        # Limiting stratification scale (Deardorff, 1976)
        # compute ∇Ri and Pr
        ∇_Ri = gradient_Richardson_number(
            mix_len_params,
            bg.∂b∂z,
            Shear²[k],
            FT(eps(FT)),
        )
        aux_tc.prandtl_nvec[k] =
            turbulent_Prandtl_number(mix_len_params, oblength, ∇_Ri)

        ml_model = MinDisspLen{FT}(;
            z = FT(grid.zc[k].z),
            obukhov_length = oblength,
            tke_surf = aux_en.tke[kc_surf],
            ustar = get_ustar(surf),
            Pr = aux_tc.prandtl_nvec[k],
            p = p_c[k],
            ∇b = bg,
            Shear² = Shear²[k],
            tke = aux_en.tke[k],
            b_exch = b_exch[k],
        )

        aux_tc.mixing_length[k] =
            mixing_length(mix_len_params, param_set, ml_model)

        KM[k] = c_m * aux_tc.mixing_length[k] * sqrt(max(aux_en.tke[k], 0))
        KH[k] = KM[k] / aux_tc.prandtl_nvec[k]

        aux_en_2m.tke.buoy[k] = -aux_en.area[k] * ρ_c[k] * KH[k] * bg.∂b∂z
    end

    #####
    ##### compute covarinaces tendendies
    #####
    tke_press = aux_en_2m.tke.press
    w_en = aux_en_f.w
    parent(tke_press) .= 0
    @inbounds for i in 1:N_up
        w_up = prog_up_f[i].w
        nh_press = aux_up_f[i].nh_pressure
        @. tke_press +=
            (Ic(wcomponent(CCG.WVector(w_en - w_up)))) *
            prog_up[i].ρarea *
            Ic(nh_press)
    end

    compute_covariance_shear(edmf, grid, state, Val(:tke), Val(:w), Val(:w))

    # TODO defined again in compute_covariance_shear and compute_covaraince
    @. aux_en_2m.tke.rain_src = 0

    compute_diffusive_fluxes(edmf, grid, state, surf, param_set)

    compute_precipitation_formation_tendencies(
        grid,
        state,
        edmf,
        edmf.precip_model,
        Δt,
        param_set,
    )
    return nothing
end
