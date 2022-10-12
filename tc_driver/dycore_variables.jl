#####
##### Fields
#####

import ClimaAtmos.TurbulenceConvection as TC
import ClimaCore as CC
import ClimaCore.Geometry as CCG
import ClimaCore.Geometry: ⊗

##### Auxiliary fields

# Center only
cent_aux_vars_gm_moisture(FT, ::CA.NonEquilMoistModel) = (;
    ∇q_liq_gm = FT(0),
    ∇q_ice_gm = FT(0),
    dqldt_rad = FT(0),
    dqidt_rad = FT(0),
    dqldt = FT(0),
    dqidt = FT(0),
    dqldt_hadv = FT(0),
    dqidt_hadv = FT(0),
    ql_nudge = FT(0),
    qi_nudge = FT(0),
    dqldt_fluc = FT(0),
    dqidt_fluc = FT(0),
)
cent_aux_vars_gm_moisture(FT, ::CA.EquilMoistModel) = NamedTuple()
cent_aux_vars_gm(FT, local_geometry, edmf) = (;
    tke = FT(0),
    Hvar = FT(0),
    QTvar = FT(0),
    HQTcov = FT(0),
    q_liq = FT(0),
    q_ice = FT(0),
    RH = FT(0),
    s = FT(0),
    T = FT(0),
    buoy = FT(0),
    cloud_fraction = FT(0),
    H_third_m = FT(0),
    W_third_m = FT(0),
    QT_third_m = FT(0),
    # From RadiationBase
    dTdt_rad = FT(0), # horizontal advection temperature tendency
    dqtdt_rad = FT(0), # horizontal advection moisture tendency
    # From ForcingBase
    subsidence = FT(0), #Large-scale subsidence
    dTdt_hadv = FT(0), #Horizontal advection of temperature
    dqtdt_hadv = FT(0), #Horizontal advection of moisture
    T_nudge = FT(0), #Reference T profile for relaxation tendency
    qt_nudge = FT(0), #Reference qt profile for relaxation tendency
    dTdt_fluc = FT(0), #Vertical turbulent advection of temperature
    dqtdt_fluc = FT(0), #Vertical turbulent advection of moisture
    u_nudge = FT(0), #Reference u profile for relaxation tendency
    v_nudge = FT(0), #Reference v profile for relaxation tendency
    uₕ_g = CCG.Covariant12Vector(CCG.UVVector(FT(0), FT(0)), local_geometry), #Geostrophic u velocity
    ∇MSE_gm = FT(0),
    ∇q_tot_gm = FT(0),
    cent_aux_vars_gm_moisture(FT, edmf.moisture_model)...,
    θ_virt = FT(0),
    Ri = FT(0),
    θ_liq_ice = FT(0),
    q_tot = FT(0),
    h_tot = FT(0),
)
cent_aux_vars(FT, local_geometry, edmf) = (;
    cent_aux_vars_gm(FT, local_geometry, edmf)...,
    TC.cent_aux_vars_edmf(FT, local_geometry, edmf)...,
)

# Face only
face_aux_vars_gm_moisture(FT, ::CA.NonEquilMoistModel) =
    (; sgs_flux_q_liq = FT(0), sgs_flux_q_ice = FT(0))
face_aux_vars_gm_moisture(FT, ::CA.EquilMoistModel) = NamedTuple()
face_aux_vars_gm(FT, local_geometry, edmf) = (;
    massflux_s = FT(0),
    diffusive_flux_s = FT(0),
    total_flux_s = FT(0),
    f_rad = FT(0),
    sgs_flux_h_tot = FT(0),
    sgs_flux_q_tot = FT(0),
    face_aux_vars_gm_moisture(FT, edmf.moisture_model)...,
    sgs_flux_uₕ = CCG.Covariant3Vector(FT(0)) ⊗
                  CCG.Covariant12Vector(FT(0), FT(0)),
    p = FT(0),
    ρ = FT(0),
)
face_aux_vars(FT, local_geometry, edmf) = (;
    face_aux_vars_gm(FT, local_geometry, edmf)...,
    TC.face_aux_vars_edmf(FT, local_geometry, edmf)...,
)
