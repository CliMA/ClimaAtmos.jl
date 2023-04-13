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

    N_up = n_updrafts(edmf)
    aux_up = center_aux_updrafts(state)

    a_bulk_bcs = a_bulk_boundary_conditions(surf, edmf)
    Ifb = CCO.InterpolateC2F(; a_bulk_bcs...)
    thermo_params = TCP.thermodynamics_params(param_set)
    microphys_params = TCP.microphysics_params(param_set)
    
    kc_surf = kc_surface(grid)
    kf_surf = kf_surface(grid)
    c_m = mixing_length_params(edmf).c_m
    KM = center_aux_turbconv(state).KM
    KH = center_aux_turbconv(state).KH
    oblength = obukhov_length(surf)
    FT = float_type(state)
    prog_gm = center_prog_grid_mean(state)
    prog_gm_f = face_prog_grid_mean(state)
    aux_up_f = face_aux_updrafts(state)
    aux_en = center_aux_environment(state)
    aux_en_f = face_aux_environment(state)
    aux_gm = center_aux_grid_mean(state)
    aux_gm_f = face_aux_grid_mean(state)
    aux_tc_f = face_aux_turbconv(state)
    aux_tc = center_aux_turbconv(state)
    aux_bulk = center_aux_bulk(state)
    prog_en = center_prog_environment(state)
    prog_up = center_prog_updrafts(state)
    prog_up_f = face_prog_updrafts(state)
    ρ_f = aux_gm_f.ρ
    p_c = center_aux_grid_mean_p(state)
    ρ_c = prog_gm.ρ
    zc = Fields.coordinate_field(axes(ρ_c)).z
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

    e_pot(z) = geopotential(thermo_params, z)
    thresh_area(prog_up, ρ_c) = prog_up[i].ρarea / ρ_c[k] >= edmf.minimum_area
    @inbounds for i in 1:N_up
        @. aux_up[i].h_tot = ifelse(
            prog_up[i].ρarea / ρ_c >= edmf.minimum_area,
            prog_up[i].ρah_tot / prog_up[i].ρarea,
            aux_gm.h_tot,
        )
        @. aux_up[i].q_tot = ifelse(
            prog_up[i].ρarea / ρ_c >= edmf.minimum_area,
            prog_up[i].ρaq_tot / prog_up[i].ρarea,
            aux_gm.q_tot,
        )
        @. aux_up[i].area = ifelse(
            prog_up[i].ρarea / ρ_c >= edmf.minimum_area,
            prog_up[i].ρarea / ρ_c,
            0,
        )
        @. aux_up[i].e_kin = ifelse(
            prog_up[i].ρarea / ρ_c >= edmf.minimum_area,
            aux_up[i].e_kin,
            e_kin,
        )
        #####
        ##### Set primitive variables
        #####
        e_int = @. aux_up[i].e_tot - aux_up[i].e_kin - e_pot(zc)
        if edmf.moisture_model isa DryModel
            @. aux_up[i].ts =
                TD.PhaseDry_pe(thermo_params, p_c, e_int)
        elseif edmf.moisture_model isa EquilMoistModel
            @. aux_up[i].ts = TD.PhaseEquil_peq(
                thermo_params,
                p_c,
                e_int,
                aux_up[i].q_tot,
            )
        elseif edmf.moisture_model isa NonEquilMoistModel
            error("Unsupported moisture model")
        end

        ts_up = aux_up[i].ts
        @. aux_up[i].θ_liq_ice = TD.liquid_ice_pottemp(thermo_params, ts_up)
        @. aux_up[i].h_tot =
            TD.total_specific_enthalpy(thermo_params, ts_up, aux_up[i].e_tot)
        @. aux_up[i].q_liq = TD.liquid_specific_humidity(thermo_params, ts_up)
        @. aux_up[i].q_ice = TD.ice_specific_humidity(thermo_params, ts_up)
        @. aux_up[i].T = TD.air_temperature(thermo_params, ts_up)
        @. aux_up[i].RH = TD.relative_humidity(thermo_params, ts_up)

    end

    #####
    ##### compute bulk
    #####
    @. aux_bulk.area = 0
    @inbounds for i in 1:N_up
        @. aux_bulk.area += aux_up[i].area
    end

    @. aux_bulk.q_tot = 0
    @. aux_bulk.h_tot = 0
    @inbounds for i in 1:N_up
        @. aux_bulk.q_tot += aux_up[i].area * aux_up[i].q_tot / aux_bulk.area
        @. aux_bulk.h_tot += aux_up[i].area * aux_up[i].h_tot / aux_bulk.area
    end
    @inbounds for i in 1:N_up
        @. aux_bulk.q_tot =
            ifelse(aux_bulk.area > 0, aux_bulk.q_tot, aux_gm.q_tot)
        @. aux_bulk.h_tot =
            ifelse(aux_bulk.area > 0, aux_bulk.h_tot, aux_gm.h_tot)
    end

    @. aux_en.area = 1 - aux_bulk.area
    @. aux_en.tke = prog_en.ρatke / (ρ_c * aux_en.area)

    @. aux_en_f.w = prog_gm_f.w / (1 - Ifb(aux_bulk.area))
    @inbounds for i in 1:N_up
        @. aux_en_f.w -=
            Ifb(aux_up[i].area) * prog_up_f[i].w / (1 - Ifb(aux_bulk.area))
    end

    @. aux_en.e_kin =
        LA.norm_sqr(C123(prog_gm_uₕ) + C123(Ic(wvec(aux_en_f.w)))) / 2

    #####
    ##### decompose_environment
    #####
    val1(aux_bulk) = 1 / (1 - aux_bulk.area)
    val2(aux_bulk) = aux_bulk.area * val1(aux_bulk)
    #Yair - this is here to prevent negative QT
    @. aux_en.q_tot =
        max(val1(aux_bulk) * aux_gm.q_tot - val2(aux_bulk) * aux_bulk.q_tot, 0)
    @. aux_en.h_tot =
        val1(aux_bulk) * aux_gm.h_tot - val2(aux_bulk) * aux_bulk.h_tot

    if edmf.moisture_model isa DryModel
        @. aux_en.ts = TD.PhaseDry_ph(
            thermo_params,
            p_c,
            enthalpy(aux_en.h_tot, e_pot(zc), aux_en.e_kin),
        )
    elseif edmf.moisture_model isa EquilMoistModel
        @. aux_en.ts = TD.PhaseEquil_phq(
            thermo_params,
            p_c,
            enthalpy(aux_en.h_tot, e_pot(zc), aux_en.e_kin),
            aux_en.q_tot,
        )
    elseif edmf.moisture_model isa NonEquilMoistModel
        error("Add support got non-equilibrium thermo states")
    end
    ts_en = aux_en.ts
    @. aux_en.θ_liq_ice = TD.liquid_ice_pottemp(thermo_params, ts_en)
    @. aux_en.e_tot =
        TD.total_energy(thermo_params, ts_en, aux_en.e_kin, e_pot(zc))
    @. aux_en.T = TD.air_temperature(thermo_params, ts_en)
    @. aux_en.θ_virt = TD.virtual_pottemp(thermo_params, ts_en)
    @. aux_en.θ_dry = TD.dry_pottemp(thermo_params, ts_en)
    @. aux_en.q_liq = TD.liquid_specific_humidity(thermo_params, ts_en)
    @. aux_en.q_ice = TD.ice_specific_humidity(thermo_params, ts_en)
    @. aux_en.RH = TD.relative_humidity(thermo_params, ts_en)

    microphysics(state, edmf, edmf.precip_model, Δt, param_set)

    # compute the buoyancy
    LBF_ρ = CCO.LeftBiasedC2F(; bottom = CCO.SetValue(ρ_f[kf_surf]))
    #LBF_ρ = CCO.LeftBiasedC2F(; bottom = CCO.SetValue(ρ_c[kc_surf]))
    LBF_a = CCO.LeftBiasedC2F(; bottom = CCO.SetValue(edmf.surface_area))
    @. aux_en_f.buoy = buoyancy_c(
        thermo_params,
        ρ_f,
        LBF_ρ(TD.air_density(thermo_params, aux_en.ts)),
    )
    @inbounds for i in 1:N_up
        @. aux_up_f[i].buoy = buoyancy_c(
            thermo_params,
            ρ_f,
            LBF_ρ(TD.air_density(thermo_params, aux_up[i].ts)),
        )
    end
    @. aux_gm_f.buoy = (1.0 - LBF_a(aux_bulk.area)) * aux_en_f.buoy
    @inbounds for i in 1:N_up
        @. aux_gm_f.buoy += LBF_a(aux_up[i].area) * aux_up_f[i].buoy
    end
    @inbounds for i in 1:N_up
        @. aux_up_f[i].buoy -= aux_gm_f.buoy
    end
    # Only needed for diagnostics/plotting
    @. aux_en_f.buoy -= aux_gm_f.buoy
    @. aux_tc_f.bulk.buoy_up1 = aux_up_f[1].buoy


    #####
    ##### compute bulk thermodynamics
    #####
    @. aux_bulk.q_liq = 0
    @. aux_bulk.q_ice = 0
    @. aux_bulk.T = 0
    @inbounds for i in 1:N_up
        @. aux_bulk.q_liq += aux_up[i].area * aux_up[i].q_liq / aux_bulk.area
        @. aux_bulk.q_ice += aux_up[i].area * aux_up[i].q_ice / aux_bulk.area
        @. aux_bulk.T += aux_up[i].area * aux_up[i].T / aux_bulk.area
    end
    @. aux_bulk.q_liq = ifelse(aux_bulk.area > 0, aux_bulk.q_liq, 0)
    @. aux_bulk.q_ice = ifelse(aux_bulk.area > 0, aux_bulk.q_ice, 0)
    @. aux_bulk.T = ifelse(aux_bulk.area > 0, aux_bulk.T, aux_en.T)

    #####
    ##### update_GMV_diagnostics
    #####
    @. aux_gm.q_liq =
        aux_bulk.area * aux_bulk.q_liq + (1 - aux_bulk.area) * aux_en.q_liq
    @. aux_gm.q_ice =
        aux_bulk.area * aux_bulk.q_ice + (1 - aux_bulk.area) * aux_en.q_ice
    @. aux_gm.T = aux_bulk.area * aux_bulk.T + (1 - aux_bulk.area) * aux_en.T

    @. aux_bulk.cloud_fraction = ifelse(
        TD.has_condensate(aux_bulk.q_liq + aux_bulk.q_ice) &&
        aux_bulk.area > 1e-3,
        1,
        0,
    )

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
    compute_nh_pressure!(state, edmf, surf)

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

    bg = center_aux_turbconv(state).buoy_grad

    # buoyancy_gradients
    if edmf.bg_closure isa BuoyGradMean
        # First order approximation: Use environmental mean fields.
        @. bg = buoyancy_gradients(
            param_set,
            edmf.moisture_model,
            EnvBuoyGrad(
                edmf.bg_closure,
                aux_en.T,                                          # t_sat
                TD.vapor_specific_humidity(thermo_params, ts_env), # qv_sat
                aux_en.q_tot,                                      # qt_sat
                aux_en.θ_dry,                                      # θ_sat
                aux_en.θ_liq_ice,                                  # θ_liq_ice_sat
                ∂θv∂z,                                             # ∂θv∂z_unsat
                ∂qt∂z,                                             # ∂qt∂z_sat
                ∂θl∂z,                                             # ∂θl∂z_sat
                p_c,                                               # p
                aux_en.cloud_fraction,                             # en_cld_frac
                ρ_c,                                               # ρ
            ),
        )
    elseif edmf.bg_closure isa BuoyGradQuadratures
        @. bg = buoyancy_gradients(
            param_set,
            edmf.moisture_model,
            EnvBuoyGrad(
                edmf.bg_closure,
                aux_en_sat.T,          # t_sat
                aux_en_sat.q_vap,      # qv_sat
                aux_en_sat.q_tot,      # qt_sat
                aux_en_sat.θ_dry,      # θ_sat
                aux_en_sat.θ_liq_ice,  # θ_liq_ice_sat
                ∂θv∂z_unsat,           # ∂θv∂z_unsat
                ∂qt∂z_sat,             # ∂qt∂z_sat
                ∂θl∂z_sat,             # ∂θl∂z_sat
                p_c,                   # p
                aux_en.cloud_fraction, # en_cld_frac
                ρ_c,                   # ρ
            ),
        )
    else
        error(
            "Something went wrong. The buoyancy gradient model is not specified",
        )
    end

    # Limiting stratification scale (Deardorff, 1976)
    # compute ∇Ri and Pr
    @. aux_tc.prandtl_nvec = turbulent_Prandtl_number(
        mix_len_params,
        oblength,
        gradient_Richardson_number(
            mix_len_params,
            bg.∂b∂z,
            Shear²,
            FT(eps(FT)),
        ),
    )

    tke_surf = aux_en.tke[kc_surf]
    ustar_surf = get_ustar(surf)

    @. aux_tc.mixing_length = mixing_length(
        mix_len_params,
        param_set,
        MinDisspLen{FT}(
            zc,                  # z
            oblength,            # obukhov_length
            tke_surf,            # tke_surf
            ustar_surf,          # ustar
            aux_tc.prandtl_nvec, # Pr
            p_c,                 # p
            bg,                  # ∇b
            Shear²,              # Shear²
            aux_en.tke,          # tke
            b_exch,              # b_exch
        ),
    )

    @. KM = c_m * aux_tc.mixing_length * sqrt(max(aux_en.tke, 0))
    @. KH = KM / aux_tc.prandtl_nvec

    compute_diffusive_fluxes(edmf, grid, state, surf, param_set)

    compute_precipitation_formation_tendencies(
        state,
        edmf,
        edmf.precip_model,
        Δt,
        param_set,
    )
    return nothing
end
