"""
    function pi_groups_detrainment!(gm_tke, up_area, up_RH, en_area, en_tke, en_RH)

    - gm_tke - grid mean turbulent kinetic energy
    - up_area - updraft area
    - up_RH - updraft relative humidity
    - en_area - environment area
    - en_tke - environment turbulent kinetic energy
    - en_RH - environment relative humidity

  Computes detrainment based on Π-groups
"""
function pi_groups_detrainment!(gm_tke::FT, up_area::FT, up_RH::FT, en_area::FT, en_tke::FT, en_RH::FT) where {FT}

    Π2 = (gm_tke - en_area * en_tke) / (gm_tke + eps(FT))
    Π4 = up_RH - en_RH

    return up_area > FT(0) ?
        max(-0.0102 + 0.0612 * Π2 + 0.0827 * Π4, FT(0)) : FT(0)
end
