##### Compute moisture deficit entr/detr
function compute_entr_detr!(
    state::State,
    grid::Grid,
    edmf::EDMFModel,
    εδ_closure::MDEntr,
)
    FT = float_type(state)
    N_up = n_updrafts(edmf)
    max_area = edmf.max_area

    aux_tc = center_aux_turbconv(state)
    aux_up = center_aux_updrafts(state)
    aux_en = center_aux_environment(state)
    aux_en_f = face_aux_environment(state)

    prog_up_f = face_prog_updrafts(state)
    prog_gm_f = face_prog_grid_mean(state)

    w_up_c = aux_tc.w_up_c
    w_en_c = aux_tc.w_en_c
    m_entr_detr = aux_tc.ϕ_temporary
    ∇m_entr_detr = aux_tc.ψ_temporary

    plume_scale_height = ntuple(N_up) do i
        compute_plume_scale_height(grid, state, edmf.H_up_min, i)
    end

    wvec = CC.Geometry.WVector
    Ic = CCO.InterpolateF2C()
    ∇c = CCO.DivergenceF2C()
    LB = CCO.LeftBiasedC2F(; bottom = CCO.SetValue(FT(0)))

    @inbounds for i in 1:N_up
        # compute ∇m at cell centers
        a_up = aux_up[i].area
        w_up = prog_up_f[i].w
        w_en = aux_en_f.w
        w_gm = prog_gm_f.w
        # TODO: should we interpolate in local or covariant basis?
        @. m_entr_detr =
            a_up * (
                Ic(wcomponent(CCG.WVector(w_up))) -
                wcomponent(CCG.WVector(Ic(w_gm)))
            )
        @. ∇m_entr_detr = ∇c(wvec(LB(m_entr_detr)))
        @. w_up_c = Ic(wcomponent(CCG.WVector(w_up)))
        @. w_en_c = Ic(wcomponent(CCG.WVector(w_en)))

        @inbounds for k in real_center_indices(grid)
            if aux_up[i].area[k] > 0.0

                # Get parameter values
                # TODO - refactor this after we get rid of namelist stuff
                γ_lim = εδ_params(εδ_closure).γ_lim
                β_lim = εδ_params(εδ_closure).β_lim
                μ_0 = εδ_params(εδ_closure).μ_0
                β = εδ_params(εδ_closure).β
                χ = εδ_params(εδ_closure).χ
                c_ε = εδ_params(εδ_closure).c_ε
                c_γ = εδ_params(εδ_closure).c_γ
                c_λ = εδ_params(εδ_closure).c_λ
                q_cond_up = TD.condensate(
                    TD.PhasePartition(
                        aux_up[i].q_tot[k],
                        aux_up[i].q_liq[k],
                        aux_up[i].q_ice[k],
                    ),
                )
                q_cond_en = TD.condensate(
                    TD.PhasePartition(
                        aux_en.q_tot[k],
                        aux_en.q_liq[k],
                        aux_en.q_ice[k],
                    ),
                )
                c_δ = if !TD.has_condensate(q_cond_up + q_cond_en)
                    FT(0)
                else
                    εδ_closure.params.c_δ
                end

                # Define helper variables
                w_up = w_up_c[k] # updraft vertical velocity
                w_en = w_en_c[k] # environment vertical velocity
                b_up = aux_up[i].buoy[k] # updraft buoyancy
                b_en = aux_en.buoy[k] # environment buoyancy
                a_up = aux_up[i].area[k] # updraft area fraction
                a_en = aux_en.area[k] # environment area fraction
                RH_up = aux_up[i].RH[k] # updraft relative humidity
                RH_en = aux_en.RH[k] # environment relative humidity
                H_up = plume_scale_height[i] # plume scale height
                tke_en = aux_en.tke[k] # environment tke

                # Compute buoyancy and velocity difference
                Δb = b_up - b_en
                Δw = w_up - w_en
                Δw += copysign(FT(εδ_params(εδ_closure).w_min), Δw)

                # Compute nondimensional entrainment and detrainment
                # functions following Cohen et al. (JAMES, 2020)
                μ_ij = (χ - a_up / (a_up + a_en)) * Δb / Δw
                exp_arg = μ_ij / μ_0
                D_ε = 1 / (1 + exp(-exp_arg))
                D_δ = 1 / (1 + exp(exp_arg))
                M_δ = (max((RH_up)^β - (RH_en)^β, 0))^(1 / β)
                M_ε = (max((RH_en)^β - (RH_up)^β, 0))^(1 / β)
                ε_nondim = (c_ε * D_ε + c_δ * M_ε)
                δ_nondim = (c_ε * D_δ + c_δ * M_δ)

                # compute inverse timescale
                l_1 = c_λ * abs(Δb / sqrt(tke_en + 1e-8))
                l_2 = abs(Δb / Δw)
                l = SA.SVector(l_1, l_2)
                λ = lamb_smooth_minimum(l, FT(0.1), FT(0.0005))

                # compute entrainment inverse length scale
                εδ_dim_scale = λ / Δw

                # compute area limiter for additional detrainment at updraft tops
                logistic_term = (2 - 1 / (1 + exp(-γ_lim * (max_area - a_up))))
                area_limiter = (logistic_term)^β_lim - 1

                # fractional dynamical entrainment / detrainment rates [1 / m]
                # given non-dimensional rates
                aux_up[i].entr_sc[k] = εδ_dim_scale * ε_nondim
                aux_up[i].detr_sc[k] = εδ_dim_scale * (δ_nondim + area_limiter)

                # turbulent entrainment
                aux_up[i].frac_turb_entr[k] =
                    w_up * a_up > 0 ?
                    2 * c_γ * sqrt(max(tke_en, 0)) / (w_up * H_up) : FT(0)
            else
                aux_up[i].entr_sc[k] = FT(0)
                aux_up[i].detr_sc[k] = FT(0)
                aux_up[i].frac_turb_entr[k] = FT(0)
            end

            aux_up[i].entr_turb_dyn[k] =
                aux_up[i].entr_sc[k] + aux_up[i].frac_turb_entr[k]
            aux_up[i].detr_turb_dyn[k] =
                aux_up[i].detr_sc[k] + aux_up[i].frac_turb_entr[k]
        end
    end
end

##### Compute constant entr/detr
function compute_entr_detr!(
    state::State,
    grid::Grid,
    edmf::EDMFModel,
    εδ_closure::ConstantEntrDetrModel,
)
    FT = float_type(state)
    N_up = n_updrafts(edmf)
    aux_up = center_aux_updrafts(state)

    @inbounds for i in 1:N_up
        @inbounds for k in real_center_indices(grid)
            if aux_up[i].area[k] > 0.0
                # Compute fractional and turbulent entrainment/detrainment
                aux_up[i].entr_sc[k] = FT(0.001)
                aux_up[i].detr_sc[k] = FT(0.001)
                aux_up[i].frac_turb_entr[k] = FT(0)
            else
                aux_up[i].entr_sc[k] = FT(0)
                aux_up[i].detr_sc[k] = FT(0)
                aux_up[i].frac_turb_entr[k] = FT(0)
            end

            aux_up[i].entr_turb_dyn[k] =
                aux_up[i].entr_sc[k] + aux_up[i].frac_turb_entr[k]
            aux_up[i].detr_turb_dyn[k] =
                aux_up[i].detr_sc[k] + aux_up[i].frac_turb_entr[k]
        end
    end
end
