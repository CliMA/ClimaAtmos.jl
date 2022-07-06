#####
##### Fields
#####

import TurbulenceConvection as TC
import ClimaCore as CC
import ClimaCore.Geometry as CCG
import ClimaCore.Geometry: ⊗

##### Auxiliary fields

# Center only
cent_aux_vars_gm_moisture(FT, ::TC.NonEquilibriumMoisture) = (;
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
cent_aux_vars_gm_moisture(FT, ::TC.EquilibriumMoisture) = NamedTuple()
cent_aux_vars_gm(FT, local_geometry, edmf) = (;
    ts = TC.thermo_state(FT, edmf.moisture_model),
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
    p = FT(0),
    e_kin = FT(0),
    h_tot = FT(0),
)
cent_aux_vars(FT, local_geometry, edmf) =
    (; cent_aux_vars_gm(FT, local_geometry, edmf)..., TC.cent_aux_vars_edmf(FT, local_geometry, edmf)...)

# Face only
face_aux_vars_gm_moisture(FT, ::TC.NonEquilibriumMoisture) = (; sgs_flux_q_liq = FT(0), sgs_flux_q_ice = FT(0))
face_aux_vars_gm_moisture(FT, ::TC.EquilibriumMoisture) = NamedTuple()
face_aux_vars_gm(FT, local_geometry, edmf) = (;
    massflux_s = FT(0),
    diffusive_flux_s = FT(0),
    total_flux_s = FT(0),
    f_rad = FT(0),
    sgs_flux_h_tot = FT(0),
    sgs_flux_q_tot = FT(0),
    face_aux_vars_gm_moisture(FT, edmf.moisture_model)...,
    sgs_flux_uₕ = CCG.Covariant3Vector(FT(0)) ⊗ CCG.Covariant12Vector(FT(0), FT(0)),
    p = FT(0),
    ρ = FT(0),
)
face_aux_vars(FT, local_geometry, edmf) =
    (; face_aux_vars_gm(FT, local_geometry, edmf)..., TC.face_aux_vars_edmf(FT, local_geometry, edmf)...)

##### Diagnostic fields

# Center only
cent_diagnostic_vars_gm(FT, local_geometry) = NamedTuple()
cent_diagnostic_vars(FT, local_geometry, edmf) =
    (; cent_diagnostic_vars_gm(FT, local_geometry)..., TC.cent_diagnostic_vars_edmf(FT, local_geometry, edmf)...)

# Face only
face_diagnostic_vars_gm(FT, local_geometry) = NamedTuple()
face_diagnostic_vars(FT, local_geometry, edmf) =
    (; face_diagnostic_vars_gm(FT, local_geometry)..., TC.face_diagnostic_vars_edmf(FT, local_geometry, edmf)...)

# Single value per column diagnostic variables
single_value_per_col_diagnostic_vars_gm(FT) = (;
    Tsurface = FT(0),
    shf = FT(0),
    lhf = FT(0),
    ustar = FT(0),
    wstar = FT(0),
    lwp_mean = FT(0),
    iwp_mean = FT(0),
    rwp_mean = FT(0),
    swp_mean = FT(0),
    cutoff_precipitation_rate = FT(0),
    cloud_base_mean = FT(0),
    cloud_top_mean = FT(0),
    cloud_cover_mean = FT(0),
)
single_value_per_col_diagnostic_vars(FT, edmf) =
    (; single_value_per_col_diagnostic_vars_gm(FT)..., TC.single_value_per_col_diagnostic_vars_edmf(FT, edmf)...)

##### Prognostic fields

# Center only
cent_prognostic_vars(::Type{FT}, local_geometry, edmf) where {FT} =
    (; cent_prognostic_vars_gm(FT, local_geometry, edmf)..., TC.cent_prognostic_vars_edmf(FT, edmf)...)
cent_prognostic_vars_gm_moisture(::Type{FT}, ::TC.NonEquilibriumMoisture) where {FT} = (; q_liq = FT(0), q_ice = FT(0))
cent_prognostic_vars_gm_moisture(::Type{FT}, ::TC.EquilibriumMoisture) where {FT} = NamedTuple()
cent_prognostic_vars_gm(::Type{FT}, local_geometry, edmf) where {FT} = (;
    ρ = FT(0),
    uₕ = CCG.Covariant12Vector(CCG.UVVector(FT(0), FT(0)), local_geometry),
    ρe_tot = FT(0),
    ρq_tot = FT(0),
    cent_prognostic_vars_gm_moisture(FT, edmf.moisture_model)...,
)

# Face only
face_prognostic_vars(::Type{FT}, local_geometry, edmf) where {FT} =
    (; w = CCG.Covariant3Vector(FT(0)), TC.face_prognostic_vars_edmf(FT, local_geometry, edmf)...)

# TC.face_prognostic_vars_edmf(FT, edmf) = (;) # could also use this for empty model
