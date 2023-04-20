function compute_updraft_top(updraft_area)
    (; value, z) = column_findlastvalue(a -> a[] > 1e-3, updraft_area)
    return z[]
end

"""
    compute_nh_pressure!(
        param_set,
        buoy_up,
        w_up,
        div_w_up,
        w_up_en_diff,
        updraft_top,
    )
where
 - `param_set`: contains model parameters
 - `buoy_up`: updraft buoyancy
 - `w_up`: updraft vertical velocity
 - `div_w_up`: updraft vertical velocity divergence
 - `w_up_en_diff`: difference between updraft and environment vertical velocity
 - `updraft_top`: height at which updraft terminates

Returns the updraft perturbation pressure gradient advection and drag
following [He2020](@cite)
"""
function compute_nh_pressure!(
    param_set,
    buoy_up,
    w_up,
    div_w_up,
    w_up_en_diff,
    updraft_top,
)
    # factor multiplier for pressure buoyancy terms (effective buoyancy is (1-α_b))
    α_b = TCP.pressure_normalmode_buoy_coeff1(param_set)
    # factor multiplier for pressure advection
    α_a = TCP.pressure_normalmode_adv_coeff(param_set)
    # factor multiplier for pressure drag
    α_d = TCP.pressure_normalmode_drag_coeff(param_set)

    # Independence of aspect ratio hardcoded: α₂_asp_ratio² = FT(0)

    H_up_min = TCP.min_updraft_top(param_set)
    plume_scale_height = max(updraft_top, H_up_min)

    return -α_b * buoy_up + α_a * w_up * div_w_up -
           α_d * w_up_en_diff * abs(w_up_en_diff) / plume_scale_height
end
