"""
    compute_nh_pressure!(
        state::State,
        grid::Grid,
        edmf::EDMFModel,
        surf::SurfaceBase,
    )

Computes the
    - perturbation pressure gradient
    - (perturbation?) pressure advection
    - (perturbation?) pressure drag

for all updrafts, following [He2020](@cite), given:

 - `state`: state
 - `grid`: grid
 - `edmf`: EDMF model
 - `surf`: surface model
"""
function compute_nh_pressure!(state::State, grid::Grid, edmf::EDMFModel, surf)

    FT = float_type(state)
    N_up = n_updrafts(edmf)
    kc_surf = kc_surface(grid)
    kc_toa = kc_top_of_atmos(grid)

    Ifc = CCO.InterpolateF2C()
    wvec = CC.Geometry.WVector
    w_bcs = (; bottom = CCO.SetValue(wvec(FT(0))), top = CCO.SetValue(wvec(FT(0))))
    ∇ = CCO.DivergenceC2F(; w_bcs...)
    aux_up = center_aux_updrafts(state)
    aux_up_f = face_aux_updrafts(state)
    aux_gm_f = face_aux_grid_mean(state)
    aux_en_f = face_aux_environment(state)
    ρ_f = aux_gm_f.ρ
    plume_scale_height = map(1:N_up) do i
        compute_plume_scale_height(grid, state, edmf.H_up_min, i)
    end

    # Note: Independence of aspect ratio hardcoded in implementation.
    α₂_asp_ratio² = FT(0)
    press_model_params = pressure_model_params(edmf)
    α_b = press_model_params.α_b
    α_a = press_model_params.α_a
    α_d = press_model_params.α_d

    @inbounds for i in 1:N_up
        # pressure
        b_up = aux_up[i].buoy
        a_up = aux_up[i].area
        w_up = aux_up_f[i].w
        H_up = plume_scale_height[i]
        w_en = aux_en_f.w

        b_bcs = (; bottom = CCO.SetValue(b_up[kc_surf]), top = CCO.SetValue(b_up[kc_toa]))
        a_bcs = a_up_boundary_conditions(surf, edmf, i)
        Ifb = CCO.InterpolateC2F(; b_bcs...)
        Ifa = CCO.InterpolateC2F(; a_bcs...)

        nh_press_buoy = aux_up_f[i].nh_pressure_b
        nh_press_adv = aux_up_f[i].nh_pressure_adv
        nh_press_drag = aux_up_f[i].nh_pressure_drag
        nh_pressure = aux_up_f[i].nh_pressure

        @. nh_press_buoy = Int(Ifa(a_up) > 0) * -α_b / (1 + α₂_asp_ratio²) * ρ_f * Ifa(a_up) * Ifb(b_up)
        @. nh_press_adv = Int(Ifa(a_up) > 0) * ρ_f * Ifa(a_up) * α_a * w_up * ∇(wvec(Ifc(w_up)))
        # drag as w_dif and account for downdrafts
        @. nh_press_drag = Int(Ifa(a_up) > 0) * -1 * ρ_f * Ifa(a_up) * α_d * (w_up - w_en) * abs(w_up - w_en) / H_up
        @. nh_pressure = nh_press_buoy + nh_press_adv + nh_press_drag
    end
    return nothing
end
