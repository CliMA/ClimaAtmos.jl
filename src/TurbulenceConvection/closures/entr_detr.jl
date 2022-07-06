#### Entrainment-Detrainment kernels

function compute_turbulent_entrainment(c_γ::FT, a_up::FT, w_up::FT, tke::FT, H_up::FT) where {FT}

    ε_turb = if w_up * a_up > 0
        2 * c_γ * sqrt(max(tke, 0)) / (w_up * H_up)
    else
        FT(0)
    end

    return ε_turb
end

function compute_inverse_timescale(εδ_model, b_up::FT, b_en::FT, w_up::FT, w_en::FT, tke::FT) where {FT}
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
    zc_i::FT,
    ::BuoyVelEntrDimScale,
) where {FT}
    Δw = get_Δw(εδ_model, w_up, w_en)
    λ = compute_inverse_timescale(εδ_model, b_up, b_en, w_up, w_en, tke)
    return (λ / Δw)
end

function entrainment_inv_length_scale(
    εδ_model,
    b_up::FT,
    b_en::FT,
    w_up::FT,
    w_en::FT,
    tke::FT,
    zc_i::FT,
    ::InvZEntrDimScale,
) where {FT}
    return (1 / zc_i)
end

function entrainment_inv_length_scale(
    εδ_model,
    b_up::FT,
    b_en::FT,
    w_up::FT,
    w_en::FT,
    tke::FT,
    zc_i::FT,
    ::InvMeterEntrDimScale,
) where {FT}
    return FT(1)
end

"""A convenience wrapper for entrainment_inv_length_scale"""
function entrainment_inv_length_scale(εδ_model, εδ_vars, dim_scale)
    return entrainment_inv_length_scale(
        εδ_model,
        εδ_vars.b_up,
        εδ_vars.b_en,
        εδ_vars.w_up,
        εδ_vars.w_en,
        εδ_vars.tke_en,
        εδ_vars.zc_i,
        dim_scale,
    )
end

"""
    εδ_dyn(εδ_model, εδ_vars, entr_dim_scale, detr_dim_scale, ε_nondim, δ_nondim)

Returns the fractional dynamical entrainment and detrainment rates [1/m] given non-dimensional rates

Parameters:
 - `εδ_model`       :: entrainment-detrainment model
 - `εδ_vars`        :: structure containing variables
 - `entr_dim_scale` :: type of dimensional fractional entrainment scale
 - `detr_dim_scale` :: type of dimensional fractional detrainment scale
 - `ε_nondim`       :: nondimensional fractional entrainment
 - `δ_nondim`       :: nondimensional fractional detrainment
"""
function εδ_dyn(εδ_model, εδ_vars, entr_dim_scale, detr_dim_scale, ε_nondim, δ_nondim)
    FT = eltype(εδ_vars.q_cond_up)
    ε_dim_scale = entrainment_inv_length_scale(εδ_model, εδ_vars, entr_dim_scale)
    δ_dim_scale = entrainment_inv_length_scale(εδ_model, εδ_vars, detr_dim_scale)

    area_limiter = max_area_limiter(εδ_model, εδ_vars.max_area, εδ_vars.a_up)

    c_div = εδ_params(εδ_model).c_div
    MdMdz_ε, MdMdz_δ = get_MdMdz(εδ_vars.M, εδ_vars.dMdz) .* c_div

    # fractional dynamical entrainment / detrainment [1 / m]
    ε_dyn = ε_dim_scale * ε_nondim + MdMdz_ε
    δ_dyn = δ_dim_scale * (δ_nondim + area_limiter) + MdMdz_δ

    return ε_dyn, δ_dyn
end

"""
    entr_detr(εδ_model, εδ_vars, entr_dim_scale, detr_dim_scale)

Returns the fractional dynamical entrainment and detrainment rates [1/m],
as well as the turbulent entrainment rate

Parameters:
 - `εδ_model`       :: type of non-dimensional model for entrainment/detrainment
 - `εδ_vars`        :: structure containing variables
 - `entr_dim_scale` :: type of dimensional fractional entrainment scale
 - `detr_dim_scale` :: type of dimensional fractional detrainment scale
"""
function entr_detr(εδ_model, εδ_vars, entr_dim_scale, detr_dim_scale)
    FT = eltype(εδ_vars.q_cond_up)

    # fractional entrainment / detrainment
    ε_nondim, δ_nondim = non_dimensional_function(εδ_model, εδ_vars)
    ε_dyn, δ_dyn = εδ_dyn(εδ_model, εδ_vars, entr_dim_scale, detr_dim_scale, ε_nondim, δ_nondim)

    # turbulent entrainment
    ε_turb =
        compute_turbulent_entrainment(εδ_params(εδ_model).c_γ, εδ_vars.a_up, εδ_vars.w_up, εδ_vars.tke_en, εδ_vars.H_up)

    return EntrDetr{FT}(ε_dyn, δ_dyn, ε_turb, ε_nondim, δ_nondim)
end

##### Compute entr detr
function compute_entr_detr!(
    state::State,
    grid::Grid,
    edmf::EDMFModel,
    param_set::APS,
    surf::SurfaceBase,
    Δt::Real,
    εδ_closure::AbstractEntrDetrModel,
)
    FT = float_type(state)
    N_up = n_updrafts(edmf)
    aux_up = center_aux_updrafts(state)
    prog_up = center_prog_updrafts(state)
    aux_up_f = face_aux_updrafts(state)
    aux_en = center_aux_environment(state)
    aux_en_f = face_aux_environment(state)
    prog_gm_f = face_prog_grid_mean(state)
    prog_gm = center_prog_grid_mean(state)
    aux_gm = center_aux_grid_mean(state)
    aux_tc = center_aux_turbconv(state)
    p_c = aux_gm.p
    ρ_c = prog_gm.ρ
    g::FT = TCP.grav(param_set)
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
        w_up = aux_up_f[i].w
        w_en = aux_en_f.w
        w_gm = prog_gm_f.w
        @. m_entr_detr = a_up * (Ic(w_up) - toscalar(Ic(w_gm)))
        @. ∇m_entr_detr = ∇c(wvec(LB(m_entr_detr)))
        @. w_up_c = Ic(w_up)
        @. w_en_c = Ic(w_en)
        @inbounds for k in real_center_indices(grid)
            # entrainment

            q_cond_up = TD.condensate(TD.PhasePartition(aux_up[i].q_tot[k], aux_up[i].q_liq[k], aux_up[i].q_ice[k]))
            q_cond_en = TD.condensate(TD.PhasePartition(aux_en.q_tot[k], aux_en.q_liq[k], aux_en.q_ice[k]))

            if aux_up[i].area[k] > 0.0
                εδ_model_vars = (;
                    q_cond_up = q_cond_up, # updraft condensate (liquid water + ice)
                    q_cond_en = q_cond_en, # environment condensate (liquid water + ice)
                    w_up = w_up_c[k], # updraft vertical velocity
                    w_en = w_en_c[k], # environment vertical velocity
                    b_up = aux_up[i].buoy[k], # updraft buoyancy
                    b_en = aux_en.buoy[k], # environment buoyancy
                    tke_gm = aux_gm.tke[k], # grid mean tke
                    tke_en = aux_en.tke[k], # environment tke
                    dMdz = ∇m_entr_detr[k], # updraft momentum divergence
                    M = m_entr_detr[k], # updraft momentum
                    a_up = aux_up[i].area[k], # updraft area fraction
                    a_en = aux_en.area[k], # environment area fraction
                    H_up = plume_scale_height[i], # plume scale height
                    ref_H = p_c[k] / (ρ_c[k] * g), # reference state scale height
                    RH_up = aux_up[i].RH[k], # updraft relative humidity
                    RH_en = aux_en.RH[k], # environment relative humidity
                    max_area = max_area, # maximum updraft area
                    zc_i = FT(grid.zc[k].z), # vertical coordinate
                    Δt = Δt, # Model time step
                    ε_nondim = aux_up[i].ε_nondim[k], # nondimensional fractional dynamical entrainment
                    δ_nondim = aux_up[i].δ_nondim[k], # nondimensional fractional dynamical detrainment
                    wstar = surf.wstar, # convective velocity
                    entr_Π_subset = entrainment_Π_subset(edmf), # indices of Pi groups to include
                )

                # update fractional and turbulent entr/detr
                if εδ_closure isa PrognosticNoisyRelaxationProcess
                    # fractional dynamical entrainment from prognostic state
                    ε_nondim, δ_nondim = prog_up[i].ε_nondim[k], prog_up[i].δ_nondim[k]
                    mean_model = εδ_closure.mean_model
                    ε_dyn, δ_dyn =
                        εδ_dyn(mean_model, εδ_model_vars, edmf.entr_dim_scale, edmf.detr_dim_scale, ε_nondim, δ_nondim)
                    # turbulent & mean nondimensional entrainment
                    er = entr_detr(mean_model, εδ_model_vars, edmf.entr_dim_scale, edmf.detr_dim_scale)
                else
                    # fractional, turbulent & nondimensional entrainment
                    er = entr_detr(εδ_closure, εδ_model_vars, edmf.entr_dim_scale, edmf.detr_dim_scale)
                    ε_dyn, δ_dyn = er.ε_dyn, er.δ_dyn
                end
                aux_up[i].entr_sc[k] = ε_dyn
                aux_up[i].detr_sc[k] = δ_dyn
                aux_up[i].frac_turb_entr[k] = er.ε_turb
                # update nondimensional entr/detr
                aux_up[i].ε_nondim[k] = er.ε_nondim
                aux_up[i].δ_nondim[k] = er.δ_nondim
            else
                aux_up[i].entr_sc[k] = 0.0
                aux_up[i].detr_sc[k] = 0.0
                aux_up[i].frac_turb_entr[k] = 0.0
                aux_up[i].ε_nondim[k] = 0.0
                aux_up[i].δ_nondim[k] = 0.0
            end
        end
    end
end

function compute_entr_detr!(
    state::State,
    grid::Grid,
    edmf::EDMFModel,
    param_set::APS,
    surf::SurfaceBase,
    Δt::Real,
    εδ_model::AbstractNonLocalEntrDetrModel,
)
    FT = float_type(state)
    N_up = n_updrafts(edmf)
    aux_up = center_aux_updrafts(state)
    aux_up_f = face_aux_updrafts(state)
    aux_en = center_aux_environment(state)
    aux_en_f = face_aux_environment(state)
    prog_gm_f = face_prog_grid_mean(state)
    aux_gm = center_aux_grid_mean(state)
    aux_tc = center_aux_turbconv(state)
    prog_gm = center_prog_grid_mean(state)
    p_c = aux_gm.p
    ρ_c = prog_gm.ρ
    g::FT = TCP.grav(param_set)
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
    c_div = εδ_params(εδ_model).c_div
    @inbounds for i in 1:N_up
        # compute ∇m at cell centers
        a_up = aux_up[i].area
        w_up = aux_up_f[i].w
        w_en = aux_en_f.w
        w_gm = prog_gm_f.w
        @. m_entr_detr = a_up * (Ic(w_up) - Ic(toscalar(w_gm)))
        @. ∇m_entr_detr = ∇c(wvec(LB(m_entr_detr)))
        @. w_up_c = Ic(w_up)
        @. w_en_c = Ic(w_en)

        @inbounds for k in real_center_indices(grid)
            # entrainment

            q_cond_up = TD.condensate(TD.PhasePartition(aux_up[i].q_tot[k], aux_up[i].q_liq[k], aux_up[i].q_ice[k]))
            q_cond_en = TD.condensate(TD.PhasePartition(aux_en.q_tot[k], aux_en.q_liq[k], aux_en.q_ice[k]))

            if aux_up[i].area[k] > 0.0
                εδ_model_vars = (;
                    q_cond_up = q_cond_up, # updraft condensate (liquid water + ice)
                    q_cond_en = q_cond_en, # environment condensate (liquid water + ice)
                    w_up = w_up_c[k], # updraft vertical velocity
                    w_en = w_en_c[k], # environment vertical velocity
                    b_up = aux_up[i].buoy[k], # updraft buoyancy
                    b_en = aux_en.buoy[k], # environment buoyancy
                    tke_gm = aux_gm.tke[k], # grid mean tke
                    tke_en = aux_en.tke[k], # environment tke
                    dMdz = ∇m_entr_detr[k], # updraft momentum divergence
                    M = m_entr_detr[k], # updraft momentum
                    a_up = aux_up[i].area[k], # updraft area fraction
                    a_en = aux_en.area[k], # environment area fraction
                    H_up = plume_scale_height[i], # plume scale height
                    ref_H = p_c[k] / (ρ_c[k] * g), # reference state scale height
                    RH_up = aux_up[i].RH[k], # updraft relative humidity
                    RH_en = aux_en.RH[k], # environment relative humidity
                    max_area = max_area, # maximum updraft area
                    zc_i = FT(grid.zc[k].z), # vertical coordinate
                    Δt = Δt, # Model time step
                    ε_nondim = aux_up[i].ε_nondim[k], # nondimensional fractional dynamical entrainment
                    δ_nondim = aux_up[i].δ_nondim[k], # nondimensional fractional dynamical detrainment
                    wstar = surf.wstar, # convective velocity
                    entr_Π_subset = entrainment_Π_subset(edmf), # indices of Pi groups to include
                )
                Π = non_dimensional_groups(εδ_model, εδ_model_vars)
                @assert length(Π) == n_Π_groups(edmf)
                for Π_i in 1:length(entrainment_Π_subset(edmf))
                    aux_up[i].Π_groups[Π_i][k] = Π[Π_i]
                end

            else
                for Π_i in 1:length(entrainment_Π_subset(edmf))
                    aux_up[i].Π_groups[Π_i][k] = 0.0
                end
            end
        end

        Π_groups = parent(aux_up[i].Π_groups)
        ε_nondim = parent(aux_up[i].ε_nondim)
        δ_nondim = parent(aux_up[i].δ_nondim)
        non_dimensional_function!(ε_nondim, δ_nondim, Π_groups, εδ_model)

        @inbounds for k in real_center_indices(grid)
            ε_turb = compute_turbulent_entrainment(
                FT(εδ_params(εδ_model).c_γ),
                aux_up[i].area[k],
                w_up_c[k],
                aux_en.tke[k],
                FT(plume_scale_height[i]),
            )
            ε_dim_scale = entrainment_inv_length_scale(
                εδ_model,
                aux_up[i].buoy[k],
                aux_en.buoy[k],
                w_up_c[k],
                w_en_c[k],
                aux_en.tke[k],
                FT(grid.zc[k].z),
                edmf.entr_dim_scale,
            )
            δ_dim_scale = entrainment_inv_length_scale(
                εδ_model,
                aux_up[i].buoy[k],
                aux_en.buoy[k],
                w_up_c[k],
                w_en_c[k],
                aux_en.tke[k],
                FT(grid.zc[k].z),
                edmf.detr_dim_scale,
            )
            area_limiter = max_area_limiter(εδ_model, max_area, aux_up[i].area[k])
            MdMdz_ε, MdMdz_δ = get_MdMdz(m_entr_detr[k], ∇m_entr_detr[k]) .* c_div

            aux_up[i].entr_sc[k] = ε_dim_scale * aux_up[i].ε_nondim[k] + MdMdz_ε
            aux_up[i].detr_sc[k] = δ_dim_scale * (aux_up[i].δ_nondim[k] + area_limiter) + MdMdz_δ
            aux_up[i].frac_turb_entr[k] = ε_turb
        end

        @. aux_up[i].ε_nondim = ifelse(aux_up[i].area > 0, aux_up[i].ε_nondim, 0)
        @. aux_up[i].δ_nondim = ifelse(aux_up[i].area > 0, aux_up[i].δ_nondim, 0)
        @. aux_up[i].entr_sc = ifelse(aux_up[i].area > 0, aux_up[i].entr_sc, 0)
        @. aux_up[i].detr_sc = ifelse(aux_up[i].area > 0, aux_up[i].detr_sc, 0)
        @. aux_up[i].frac_turb_entr = ifelse(aux_up[i].area > 0, aux_up[i].frac_turb_entr, 0)

    end
end
