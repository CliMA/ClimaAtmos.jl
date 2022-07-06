function update_aux!(edmf::EDMFModel, grid::Grid, state::State, surf::SurfaceBase, param_set::APS, t::Real, Δt::Real)
    #####
    ##### Unpack common variables
    #####
    thermo_params = TCP.thermodynamics_params(param_set)
    microphys_params = TCP.microphysics_params(param_set)
    N_up = n_updrafts(edmf)
    kc_surf = kc_surface(grid)
    kf_surf = kf_surface(grid)
    kc_toa = kc_top_of_atmos(grid)
    c_m = mixing_length_params(edmf).c_m
    KM = center_aux_turbconv(state).KM
    KH = center_aux_turbconv(state).KH
    obukhov_length = surf.obukhov_length
    FT = float_type(state)
    prog_gm = center_prog_grid_mean(state)
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
    p_c = aux_gm.p
    ρ_c = prog_gm.ρ
    aux_en_unsat = aux_en.unsat
    aux_en_sat = aux_en.sat
    m_entr_detr = aux_tc.ϕ_temporary
    ∇m_entr_detr = aux_tc.ψ_temporary
    wvec = CC.Geometry.WVector
    max_area = edmf.max_area
    ts_gm = center_aux_grid_mean(state).ts
    ts_env = center_aux_environment(state).ts

    prog_gm_uₕ = grid_mean_uₕ(state)
    Ic = CCO.InterpolateF2C()
    #####
    ##### center variables
    #####
    C123 = CCG.Covariant123Vector
    @. aux_en.e_kin = LA.norm_sqr(C123(prog_gm_uₕ) + C123(Ic(wvec(aux_en_f.w)))) / 2

    @inbounds for i in 1:N_up
        @. aux_up[i].e_kin = LA.norm_sqr(C123(prog_gm_uₕ) + C123(Ic(wvec(aux_up_f[i].w)))) / 2
    end

    @inbounds for k in real_center_indices(grid)
        #####
        ##### Set primitive variables
        #####
        e_pot = geopotential(param_set, grid.zc.z[k])
        @inbounds for i in 1:N_up
            if prog_up[i].ρarea[k] / ρ_c[k] >= edmf.minimum_area
                aux_up[i].θ_liq_ice[k] = prog_up[i].ρaθ_liq_ice[k] / prog_up[i].ρarea[k]
                aux_up[i].q_tot[k] = prog_up[i].ρaq_tot[k] / prog_up[i].ρarea[k]
                aux_up[i].area[k] = prog_up[i].ρarea[k] / ρ_c[k]
            else
                aux_up[i].θ_liq_ice[k] = aux_gm.θ_liq_ice[k]
                aux_up[i].q_tot[k] = aux_gm.q_tot[k]
                aux_up[i].area[k] = 0
                aux_up[i].e_kin[k] = aux_gm.e_kin[k]
            end
            thermo_args = ()
            if edmf.moisture_model isa NonEquilibriumMoisture
                if prog_up[i].ρarea[k] / ρ_c[k] >= edmf.minimum_area
                    aux_up[i].q_liq[k] = prog_up[i].ρaq_liq[k] / prog_up[i].ρarea[k]
                    aux_up[i].q_ice[k] = prog_up[i].ρaq_ice[k] / prog_up[i].ρarea[k]
                else
                    aux_up[i].q_liq[k] = prog_gm.q_liq[k]
                    aux_up[i].q_ice[k] = prog_gm.q_ice[k]
                end
                thermo_args = (aux_up[i].q_liq[k], aux_up[i].q_ice[k])
            end
            ts_up_i = thermo_state_pθq(param_set, p_c[k], aux_up[i].θ_liq_ice[k], aux_up[i].q_tot[k], thermo_args...)
            aux_up[i].e_tot[k] = TD.total_energy(thermo_params, ts_up_i, aux_up[i].e_kin[k], e_pot)
            aux_up[i].h_tot[k] = total_enthalpy(param_set, aux_up[i].e_tot[k], ts_up_i)
        end

        #####
        ##### compute bulk
        #####
        aux_bulk.q_tot[k] = 0
        aux_bulk.h_tot[k] = 0
        aux_bulk.θ_liq_ice[k] = 0
        aux_bulk.area[k] = sum(i -> aux_up[i].area[k], 1:N_up)
        if aux_bulk.area[k] > 0
            @inbounds for i in 1:N_up
                a_k = aux_up[i].area[k]
                a_bulk_k = aux_bulk.area[k]
                aux_bulk.q_tot[k] += a_k * aux_up[i].q_tot[k] / a_bulk_k
                aux_bulk.θ_liq_ice[k] += a_k * aux_up[i].θ_liq_ice[k] / a_bulk_k
                aux_bulk.h_tot[k] += a_k * aux_up[i].h_tot[k] / a_bulk_k
            end
        else
            aux_bulk.q_tot[k] = aux_gm.q_tot[k]
            aux_bulk.θ_liq_ice[k] = aux_gm.θ_liq_ice[k]
            aux_bulk.h_tot[k] = aux_gm.h_tot[k]
        end
        if edmf.moisture_model isa NonEquilibriumMoisture
            aux_bulk.q_liq[k] = 0
            aux_bulk.q_ice[k] = 0
            if aux_bulk.area[k] > 0
                @inbounds for i in 1:N_up
                    a_k = aux_up[i].area[k]
                    a_bulk_k = aux_bulk.area[k]
                    aux_bulk.q_liq[k] += a_k * aux_up[i].q_liq[k] / a_bulk_k
                    aux_bulk.q_ice[k] += a_k * aux_up[i].q_ice[k] / a_bulk_k
                end
            else
                aux_bulk.q_liq[k] = prog_gm.q_liq[k]
                aux_bulk.q_ice[k] = prog_gm.q_ice[k]
            end
        end
        aux_en.area[k] = 1 - aux_bulk.area[k]
        aux_en.tke[k] = prog_en.ρatke[k] / (ρ_c[k] * aux_en.area[k])
        if edmf.thermo_covariance_model isa PrognosticThermoCovariances
            aux_en.Hvar[k] = prog_en.ρaHvar[k] / (ρ_c[k] * aux_en.area[k])
            aux_en.QTvar[k] = prog_en.ρaQTvar[k] / (ρ_c[k] * aux_en.area[k])
            aux_en.HQTcov[k] = prog_en.ρaHQTcov[k] / (ρ_c[k] * aux_en.area[k])
        end

        #####
        ##### decompose_environment
        #####
        a_bulk_c = aux_bulk.area[k]
        val1 = 1 / (1 - a_bulk_c)
        val2 = a_bulk_c * val1
        aux_en.q_tot[k] = max(val1 * aux_gm.q_tot[k] - val2 * aux_bulk.q_tot[k], 0) #Yair - this is here to prevent negative QT
        aux_en.h_tot[k] = val1 * aux_gm.h_tot[k] - val2 * aux_bulk.h_tot[k]
        if edmf.moisture_model isa NonEquilibriumMoisture
            aux_en.q_liq[k] = max(val1 * prog_gm.q_liq[k] - val2 * aux_bulk.q_liq[k], 0)
            aux_en.q_ice[k] = max(val1 * prog_gm.q_ice[k] - val2 * aux_bulk.q_ice[k], 0)
        end

        #####
        ##### condensation, etc (done via saturation_adjustment or non-equilibrium) and buoyancy
        #####
        thermo_args = if edmf.moisture_model isa EquilibriumMoisture
            ()
        elseif edmf.moisture_model isa NonEquilibriumMoisture
            (aux_en.q_liq[k], aux_en.q_ice[k])
        else
            error("Something went wrong. The moisture_model options are equilibrium or nonequilibrium")
        end

        h_en = enthalpy(aux_en.h_tot[k], e_pot, aux_en.e_kin[k])
        ts_env[k] = thermo_state_phq(param_set, p_c[k], h_en, aux_en.q_tot[k], thermo_args...)
        ts_en = ts_env[k]
        aux_en.θ_liq_ice[k] = TD.liquid_ice_pottemp(thermo_params, ts_en)
        aux_en.e_tot[k] = TD.total_energy(thermo_params, ts_en, aux_en.e_kin[k], e_pot)
        aux_en.T[k] = TD.air_temperature(thermo_params, ts_en)
        aux_en.θ_virt[k] = TD.virtual_pottemp(thermo_params, ts_en)
        aux_en.θ_dry[k] = TD.dry_pottemp(thermo_params, ts_en)
        aux_en.q_liq[k] = TD.liquid_specific_humidity(thermo_params, ts_en)
        aux_en.q_ice[k] = TD.ice_specific_humidity(thermo_params, ts_en)
        rho = TD.air_density(thermo_params, ts_en)
        aux_en.buoy[k] = buoyancy_c(param_set, ρ_c[k], rho)
        aux_en.RH[k] = TD.relative_humidity(thermo_params, ts_en)
    end

    microphysics(edmf.en_thermo, grid, state, edmf, edmf.precip_model, Δt, param_set)

    @inbounds for k in real_center_indices(grid)
        a_bulk_c = aux_bulk.area[k]
        @inbounds for i in 1:N_up
            if aux_up[i].area[k] < edmf.minimum_area && k > kc_surf && aux_up[i].area[k - 1] > 0.0
                qt = aux_up[i].q_tot[k - 1]
                h = aux_up[i].θ_liq_ice[k - 1]
                if edmf.moisture_model isa EquilibriumMoisture
                    ts_up = thermo_state_pθq(param_set, p_c[k], h, qt)
                elseif edmf.moisture_model isa NonEquilibriumMoisture
                    ql = aux_up[i].q_liq[k - 1]
                    qi = aux_up[i].q_ice[k - 1]
                    ts_up = thermo_state_pθq(param_set, p_c[k], h, qt, ql, qi)
                else
                    error("Something went wrong. emdf.moisture_model options are equilibrium or nonequilibrium")
                end
            else
                if edmf.moisture_model isa EquilibriumMoisture
                    ts_up = thermo_state_pθq(param_set, p_c[k], aux_up[i].θ_liq_ice[k], aux_up[i].q_tot[k])
                elseif edmf.moisture_model isa NonEquilibriumMoisture
                    ts_up = thermo_state_pθq(
                        param_set,
                        p_c[k],
                        aux_up[i].θ_liq_ice[k],
                        aux_up[i].q_tot[k],
                        aux_up[i].q_liq[k],
                        aux_up[i].q_ice[k],
                    )
                else
                    error("Something went wrong. emdf.moisture_model options are equilibrium or nonequilibrium")
                end
            end
            aux_up[i].q_liq[k] = TD.liquid_specific_humidity(thermo_params, ts_up)
            aux_up[i].q_ice[k] = TD.ice_specific_humidity(thermo_params, ts_up)
            aux_up[i].T[k] = TD.air_temperature(thermo_params, ts_up)
            ρ = TD.air_density(thermo_params, ts_up)
            aux_up[i].buoy[k] = buoyancy_c(param_set, ρ_c[k], ρ)
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
        aux_bulk.RH[k] = 0
        aux_bulk.buoy[k] = 0
        if a_bulk_c > 0
            @inbounds for i in 1:N_up
                aux_bulk.q_liq[k] += aux_up[i].area[k] * aux_up[i].q_liq[k] / a_bulk_c
                aux_bulk.q_ice[k] += aux_up[i].area[k] * aux_up[i].q_ice[k] / a_bulk_c
                aux_bulk.T[k] += aux_up[i].area[k] * aux_up[i].T[k] / a_bulk_c
                aux_bulk.RH[k] += aux_up[i].area[k] * aux_up[i].RH[k] / a_bulk_c
                aux_bulk.buoy[k] += aux_up[i].area[k] * aux_up[i].buoy[k] / a_bulk_c
            end
        else
            aux_bulk.RH[k] = aux_en.RH[k]
            aux_bulk.T[k] = aux_en.T[k]
        end

        #####
        ##### update_GMV_diagnostics
        #####
        aux_gm.q_liq[k] = (aux_bulk.area[k] * aux_bulk.q_liq[k] + (1 - aux_bulk.area[k]) * aux_en.q_liq[k])
        aux_gm.q_ice[k] = (aux_bulk.area[k] * aux_bulk.q_ice[k] + (1 - aux_bulk.area[k]) * aux_en.q_ice[k])
        aux_gm.T[k] = (aux_bulk.area[k] * aux_bulk.T[k] + (1 - aux_bulk.area[k]) * aux_en.T[k])
        aux_gm.buoy[k] = (aux_bulk.area[k] * aux_bulk.buoy[k] + (1 - aux_bulk.area[k]) * aux_en.buoy[k])

        has_condensate = TD.has_condensate(aux_bulk.q_liq[k] + aux_bulk.q_ice[k])
        aux_bulk.cloud_fraction[k] = if has_condensate && a_bulk_c > 1e-3
            1
        else
            0
        end
    end
    #####
    ##### face variables: diagnose primitive, diagnose env and compute bulk
    #####
    # TODO: figure out why `ifelse` is allocating
    @inbounds for i in 1:N_up
        a_surf = area_surface_bc(surf, edmf, i)
        a_up_bcs = a_up_boundary_conditions(surf, edmf, i)
        If = CCO.InterpolateC2F(; a_up_bcs...)
        a_min = edmf.minimum_area
        a_up = aux_up[i].area
        @. aux_up_f[i].w = ifelse(If(a_up) >= a_min, max(prog_up_f[i].ρaw / (ρ_f * If(a_up)), 0), FT(0))
    end
    @inbounds for i in 1:N_up
        aux_up_f[i].w[kf_surf] = w_surface_bc(surf)
    end

    parent(aux_tc_f.bulk.w) .= 0
    a_bulk_bcs = a_bulk_boundary_conditions(surf, edmf)
    Ifb = CCO.InterpolateC2F(; a_bulk_bcs...)
    @inbounds for i in 1:N_up
        a_up = aux_up[i].area
        a_up_bcs = a_up_boundary_conditions(surf, edmf, i)
        Ifu = CCO.InterpolateC2F(; a_up_bcs...)
        @. aux_tc_f.bulk.w += ifelse(Ifb(aux_bulk.area) > 0, Ifu(a_up) * aux_up_f[i].w / Ifb(aux_bulk.area), FT(0))
    end
    # Assuming w_gm = 0!
    @. aux_en_f.w = -Ifb(aux_bulk.area) / (1 - Ifb(aux_bulk.area)) * aux_tc_f.bulk.w

    #####
    #####  diagnose_GMV_moments
    #####
    get_GMV_CoVar(edmf, grid, state, Val(:Hvar), Val(:θ_liq_ice), Val(:θ_liq_ice))
    get_GMV_CoVar(edmf, grid, state, Val(:QTvar), Val(:q_tot), Val(:q_tot))
    get_GMV_CoVar(edmf, grid, state, Val(:HQTcov), Val(:θ_liq_ice), Val(:q_tot))

    #####
    ##### compute_updraft_closures
    #####
    #TODO - AJ add the non-equilibrium tendency computation here
    if edmf.moisture_model isa NonEquilibriumMoisture
        compute_nonequilibrium_moisture_tendencies!(grid, state, edmf, Δt, param_set)
    end
    compute_entr_detr!(state, grid, edmf, param_set, surf, Δt, edmf.entr_closure)
    compute_nh_pressure!(state, grid, edmf, surf)

    #####
    ##### compute_eddy_diffusivities_tke
    #####

    # Subdomain exchange term
    ∇c = CCO.DivergenceF2C()
    Ic = CCO.InterpolateF2C()
    b_exch = center_aux_turbconv(state).b_exch
    parent(b_exch) .= 0
    a_en = aux_en.area
    w_en = aux_en_f.w
    tke_en = aux_en.tke
    @inbounds for i in 1:N_up
        a_up = aux_up[i].area
        w_up = aux_up_f[i].w
        δ_dyn = aux_up[i].detr_sc
        ε_turb = aux_up[i].frac_turb_entr
        @. b_exch +=
            a_up * Ic(w_up) * δ_dyn / a_en * (1 / 2 * (Ic(w_up) - Ic(w_en))^2 - tke_en) -
            a_up * Ic(w_up) * (Ic(w_up) - Ic(w_en)) * ε_turb * Ic(w_en) / a_en
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

    uₕ_gm = grid_mean_uₕ(state)
    w_en = aux_en_f.w
    # compute shear

    # TODO: Will need to be changed with topography
    local_geometry = CC.Fields.local_geometry_field(axes(ρ_c))
    k̂ = @. CCG.Contravariant3Vector(CCG.WVector(FT(1)), local_geometry)
    Ifuₕ = uₕ_bcs()
    ∇uvw = CCO.GradientF2C()
    uvw = @. C123(Ifuₕ(uₕ_gm)) + C123(wvec(w_en))
    @. Shear² = LA.norm_sqr(adjoint(∇uvw(uvw)) * k̂)

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
            error("Something went wrong. The buoyancy gradient model is not specified")
        end
        bg = buoyancy_gradients(param_set, bg_model)

        # Limiting stratification scale (Deardorff, 1976)
        # compute ∇Ri and Pr
        ∇_Ri = gradient_Richardson_number(mix_len_params, bg.∂b∂z, Shear²[k], FT(eps(FT)))
        aux_tc.prandtl_nvec[k] = turbulent_Prandtl_number(mix_len_params, obukhov_length, ∇_Ri)

        ml_model = MinDisspLen{FT}(;
            z = FT(grid.zc[k].z),
            obukhov_length = obukhov_length,
            tke_surf = aux_en.tke[kc_surf],
            ustar = surf.ustar,
            Pr = aux_tc.prandtl_nvec[k],
            p = p_c[k],
            ∇b = bg,
            Shear² = Shear²[k],
            tke = aux_en.tke[k],
            b_exch = b_exch[k],
        )

        ml = mixing_length(mix_len_params, param_set, ml_model)
        aux_tc.mls[k] = ml.min_len_ind
        aux_tc.mixing_length[k] = ml.mixing_length
        aux_tc.ml_ratio[k] = ml.ml_ratio

        KM[k] = c_m * ml.mixing_length * sqrt(max(aux_en.tke[k], 0))
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
        w_up = aux_up_f[i].w
        nh_press = aux_up_f[i].nh_pressure
        @. tke_press += (Ic(w_en) - Ic(w_up)) * Ic(nh_press)
    end

    compute_covariance_entr(edmf, grid, state, Val(:tke), Val(:w), Val(:w))
    compute_covariance_entr(edmf, grid, state, Val(:Hvar), Val(:θ_liq_ice), Val(:θ_liq_ice))
    compute_covariance_entr(edmf, grid, state, Val(:QTvar), Val(:q_tot), Val(:q_tot))
    compute_covariance_entr(edmf, grid, state, Val(:HQTcov), Val(:θ_liq_ice), Val(:q_tot))
    compute_covariance_shear(edmf, grid, state, Val(:tke), Val(:w), Val(:w))
    compute_covariance_shear(edmf, grid, state, Val(:Hvar), Val(:θ_liq_ice), Val(:θ_liq_ice))
    compute_covariance_shear(edmf, grid, state, Val(:QTvar), Val(:q_tot), Val(:q_tot))
    compute_covariance_shear(edmf, grid, state, Val(:HQTcov), Val(:θ_liq_ice), Val(:q_tot))
    compute_covariance_dissipation(edmf, grid, state, Val(:tke), param_set)
    compute_covariance_dissipation(edmf, grid, state, Val(:Hvar), param_set)
    compute_covariance_dissipation(edmf, grid, state, Val(:QTvar), param_set)
    compute_covariance_dissipation(edmf, grid, state, Val(:HQTcov), param_set)

    # TODO defined again in compute_covariance_shear and compute_covaraince
    @inbounds for k in real_center_indices(grid)
        aux_en_2m.tke.rain_src[k] = 0
        aux_en_2m.Hvar.rain_src[k] = ρ_c[k] * aux_en.area[k] * 2 * aux_en.Hvar_rain_dt[k]
        aux_en_2m.QTvar.rain_src[k] = ρ_c[k] * aux_en.area[k] * 2 * aux_en.QTvar_rain_dt[k]
        aux_en_2m.HQTcov.rain_src[k] = ρ_c[k] * aux_en.area[k] * aux_en.HQTcov_rain_dt[k]
    end

    get_GMV_CoVar(edmf, grid, state, Val(:tke), Val(:w), Val(:w))

    compute_diffusive_fluxes(edmf, grid, state, surf, param_set)

    # TODO: use dispatch
    if edmf.precip_model isa Clima1M
        # helper to calculate the rain velocity
        # TODO: assuming w_gm = 0
        # TODO: verify translation
        term_vel_rain = aux_tc.term_vel_rain
        term_vel_snow = aux_tc.term_vel_snow
        prog_pr = center_prog_precipitation(state)

        #precip_fraction = compute_precip_fraction(edmf, state)

        @inbounds for k in real_center_indices(grid)
            term_vel_rain[k] = CM1.terminal_velocity(microphys_params, rain_type, ρ_c[k], prog_pr.q_rai[k])# / precip_fraction)
            term_vel_snow[k] = CM1.terminal_velocity(microphys_params, snow_type, ρ_c[k], prog_pr.q_sno[k])# / precip_fraction)
        end
    end

    ### Diagnostic thermodynamiccovariances
    if edmf.thermo_covariance_model isa DiagnosticThermoCovariances
        flux1 = surf.shf / TD.cp_m(thermo_params, ts_gm[kc_surf])
        flux2 = surf.ρq_tot_flux
        zLL::FT = grid.zc[kc_surf].z
        ustar = surf.ustar
        oblength = surf.obukhov_length
        prog_gm = center_prog_grid_mean(state)
        ρLL = prog_gm.ρ[kc_surf]
        update_diagnostic_covariances!(edmf, grid, state, param_set, Val(:Hvar))
        update_diagnostic_covariances!(edmf, grid, state, param_set, Val(:QTvar))
        update_diagnostic_covariances!(edmf, grid, state, param_set, Val(:HQTcov))
        @inbounds for k in real_center_indices(grid)
            aux_en.Hvar[k] = max(aux_en.Hvar[k], 0)
            aux_en.QTvar[k] = max(aux_en.QTvar[k], 0)
            aux_en.HQTcov[k] = max(aux_en.HQTcov[k], -sqrt(aux_en.Hvar[k] * aux_en.QTvar[k]))
            aux_en.HQTcov[k] = min(aux_en.HQTcov[k], sqrt(aux_en.Hvar[k] * aux_en.QTvar[k]))
        end
        ae_surf = 1 - aux_bulk.area[kc_surf]
        aux_en.Hvar[kc_surf] = ae_surf * get_surface_variance(flux1 / ρLL, flux1 / ρLL, ustar, zLL, oblength)
        aux_en.QTvar[kc_surf] = ae_surf * get_surface_variance(flux2 / ρLL, flux2 / ρLL, ustar, zLL, oblength)
        aux_en.HQTcov[kc_surf] = ae_surf * get_surface_variance(flux1 / ρLL, flux2 / ρLL, ustar, zLL, oblength)
    end
    compute_precipitation_formation_tendencies(grid, state, edmf, edmf.precip_model, Δt, param_set)
    return nothing
end
