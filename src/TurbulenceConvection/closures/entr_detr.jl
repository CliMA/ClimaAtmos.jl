#### Entrainment-Detrainment kernels

function compute_turbulent_entrainment(
    c_γ::FT,
    a_up::FT,
    w_up::FT,
    tke::FT,
    H_up::FT,
) where {FT}

    ε_turb = if w_up * a_up > 0
        2 * c_γ * sqrt(max(tke, 0)) / (w_up * H_up)
    else
        FT(0)
    end

    return ε_turb
end

function compute_inverse_timescale(
    εδ_model,
    b_up::FT,
    b_en::FT,
    w_up::FT,
    w_en::FT,
    tke::FT,
) where {FT}
    Δb = b_up - b_en
    Δw = get_Δw(εδ_model, w_up, w_en)
    c_λ = εδ_params(εδ_model).c_λ

    l_1 = c_λ * abs(Δb / sqrt(tke + 1e-8))
    l_2 = abs(Δb / Δw)
    l = SA.SVector(l_1, l_2)
    return lamb_smooth_minimum(l, FT(0.1), FT(0.0005))
end

function get_Δw(εδ_model, w_up::FT, w_en::FT) where {FT}
    Δw = w_up - w_en
    Δw += copysign(FT(εδ_params(εδ_model).w_min), Δw)
    return Δw
end

function get_MdMdz(M::FT, dMdz::FT) where {FT}
    MdMdz_ε = max(dMdz / max(M, eps(FT)), 0)
    MdMdz_δ = max(-dMdz / max(M, eps(FT)), 0)
    return MdMdz_ε, MdMdz_δ
end

function entrainment_inv_length_scale(
    εδ_model,
    b_up::FT,
    b_en::FT,
    w_up::FT,
    w_en::FT,
    tke::FT,
) where {FT}
    Δw = get_Δw(εδ_model, w_up, w_en)
    λ = compute_inverse_timescale(εδ_model, b_up, b_en, w_up, w_en, tke)
    return (λ / Δw)
end

"""
    εδ_dyn(εδ_model, εδ_vars, ε_nondim, δ_nondim)

Returns the fractional dynamical entrainment and detrainment rates [1/m] given non-dimensional rates

Parameters:
 - `εδ_model`       :: entrainment-detrainment model
 - `εδ_vars`        :: structure containing variables
 - `ε_nondim`       :: nondimensional fractional entrainment
 - `δ_nondim`       :: nondimensional fractional detrainment
"""
function εδ_dyn(εδ_model, εδ_vars, ε_nondim, δ_nondim)
    εδ_dim_scale = entrainment_inv_length_scale(
        εδ_model,
        εδ_vars.b_up,
        εδ_vars.b_en,
        εδ_vars.w_up,
        εδ_vars.w_en,
        εδ_vars.tke_en,
    )

    area_limiter = max_area_limiter(εδ_model, εδ_vars.max_area, εδ_vars.a_up)

    c_div = εδ_params(εδ_model).c_div
    MdMdz_ε, MdMdz_δ = get_MdMdz(εδ_vars.M, εδ_vars.dMdz) .* c_div

    # fractional dynamical entrainment / detrainment [1 / m]
    ε_dyn = εδ_dim_scale * ε_nondim + MdMdz_ε
    δ_dyn = εδ_dim_scale * (δ_nondim + area_limiter) + MdMdz_δ

    return ε_dyn, δ_dyn
end

"""
    entr_detr(εδ_model, εδ_vars)

Returns the fractional dynamical entrainment and detrainment rates [1/m],
as well as the turbulent entrainment rate

Parameters:
 - `εδ_model`       :: type of non-dimensional model for entrainment/detrainment
 - `εδ_vars`        :: structure containing variables
"""
function entr_detr(εδ_model, εδ_vars)
    FT = eltype(εδ_vars.q_cond_up)

    # fractional entrainment / detrainment
    ε_nondim, δ_nondim = non_dimensional_function(εδ_model, εδ_vars)
    ε_dyn, δ_dyn = εδ_dyn(εδ_model, εδ_vars, ε_nondim, δ_nondim)

    # turbulent entrainment
    ε_turb = compute_turbulent_entrainment(
        εδ_params(εδ_model).c_γ,
        εδ_vars.a_up,
        εδ_vars.w_up,
        εδ_vars.tke_en,
        εδ_vars.H_up,
    )

    return EntrDetr{FT}(ε_dyn, δ_dyn, ε_turb)
end

##### Compute entr detr
function compute_entr_detr!(
    state::State,
    grid::Grid,
    edmf::EDMFModel,
    εδ_closure::AbstractEntrDetrModel,
)
    FT = float_type(state)
    N_up = n_updrafts(edmf)
    aux_up = center_aux_updrafts(state)
    prog_up_f = face_prog_updrafts(state)
    aux_en = center_aux_environment(state)
    aux_en_f = face_aux_environment(state)
    prog_gm_f = face_prog_grid_mean(state)
    aux_tc = center_aux_turbconv(state)
    w_up_c = aux_tc.w_up_c
    w_en_c = aux_tc.w_en_c
    m_entr_detr = aux_tc.ϕ_temporary
    ∇m_entr_detr = aux_tc.ψ_temporary
    wvec = CC.Geometry.WVector
    max_area = edmf.max_area
    plume_scale_height = map(1:N_up) do i
        compute_plume_scale_height(grid, state, edmf.H_up_min, i)
    end
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
            # entrainment

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

            if aux_up[i].area[k] > 0.0
                εδ_model_vars = (;
                    q_cond_up = q_cond_up, # updraft condensate (liquid water + ice)
                    q_cond_en = q_cond_en, # environment condensate (liquid water + ice)
                    w_up = w_up_c[k], # updraft vertical velocity
                    w_en = w_en_c[k], # environment vertical velocity
                    b_up = aux_up[i].buoy[k], # updraft buoyancy
                    b_en = aux_en.buoy[k], # environment buoyancy
                    dMdz = ∇m_entr_detr[k], # updraft momentum divergence
                    M = m_entr_detr[k], # updraft momentum
                    a_up = aux_up[i].area[k], # updraft area fraction
                    a_en = aux_en.area[k], # environment area fraction
                    RH_up = aux_up[i].RH[k], # updraft relative humidity
                    RH_en = aux_en.RH[k], # environment relative humidity
                    max_area = max_area, # maximum updraft area
                    H_up = plume_scale_height[i], # plume scale height
                    tke_en = aux_en.tke[k], # environment tke
                )

                # Compute fractional and turbulent entrainment/detrainment
                er = entr_detr(εδ_closure, εδ_model_vars)
                aux_up[i].entr_sc[k] = er.ε_dyn
                aux_up[i].detr_sc[k] = er.δ_dyn
                aux_up[i].frac_turb_entr[k] = er.ε_turb
            else
                aux_up[i].entr_sc[k] = FT(0)
                aux_up[i].detr_sc[k] = FT(0)
                aux_up[i].frac_turb_entr[k] = FT(0)
            end
        end
    end
end
