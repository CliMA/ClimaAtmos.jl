"""
    function pi_groups_detrainment!(gm_tke, up_area, up_RH, en_area, en_tke, en_RH)

    - gm_w - grid mean vertical velocity
    - up_area - updraft area
    - up_RH - updraft relative humidity
    - en_area - environment area
    - en_w - environment vertical velocity
    - en_RH - environment relative humidity

  Computes detrainment based on Π-groups
"""
function pi_groups_detrainment!(gm_w::FT, up_area::FT, up_RH::FT, en_area::FT, en_w::FT, en_RH::FT) where {FT}

    Π2 = (gm_w^2 - en_area * en_w^2) / (gm_w^2 + eps(FT))
    Π4 = up_RH - en_RH

    return up_area > FT(0) ?
        max(-0.0102 + 0.0612 * Π2 + 0.0827 * Π4, FT(0)) : FT(0)
end
