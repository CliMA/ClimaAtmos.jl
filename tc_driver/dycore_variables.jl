#####
##### Fields
#####

import ClimaAtmos.TurbulenceConvection as TC
import ClimaCore as CC
import ClimaCore.Geometry as CCG
import ClimaCore.Geometry: ⊗

##### Auxiliary fields

# Center only
cent_aux_vars_gm(FT, local_geometry, edmf) = (;
    tke = FT(0),
    Hvar = FT(0),
    QTvar = FT(0),
    HQTcov = FT(0),
    q_liq = FT(0),
    q_ice = FT(0),
    RH = FT(0),
    T = FT(0),
    buoy = FT(0),
    cloud_fraction = FT(0),
    θ_virt = FT(0),
    Ri = FT(0),
    θ_liq_ice = FT(0),
    q_tot = FT(0),
    h_tot = FT(0),
)
cent_aux_vars(FT, local_geometry, atmos, edmf) = (;
    cent_aux_vars_gm(FT, local_geometry, edmf)...,
    TC.cent_aux_vars_edmf(FT, local_geometry, atmos)...,
)

# Face only
face_aux_vars_gm(FT, local_geometry, atmos, edmf) = (;
    sgs_flux_h_tot = CCG.Covariant3Vector(FT(0)),
    sgs_flux_q_tot = CCG.Covariant3Vector(FT(0)),
    sgs_flux_uₕ = CCG.Covariant3Vector(FT(0)) ⊗
                  CCG.Covariant12Vector(FT(0), FT(0)),
    ρ = FT(0),
)
face_aux_vars(FT, local_geometry, atmos, edmf) = (;
    face_aux_vars_gm(FT, local_geometry, atmos, edmf)...,
    TC.face_aux_vars_edmf(FT, local_geometry, edmf)...,
)
