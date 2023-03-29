#####
##### Fields
#####
import ClimaCore.Geometry: ⊗

# Helpers for adding empty thermodynamic state fields:
thermo_state_zeros(::DryModel, FT) = zero(TD.PhaseDry{FT})
thermo_state_zeros(::EquilMoistModel, FT) = zero(TD.PhaseEquil{FT})
thermo_state_zeros(::NonEquilMoistModel, FT) = zero(TD.PhaseNonEquil{FT})

##### Auxiliary fields

# Center only
cent_aux_vars_up(FT, local_geometry, edmf) = (;
    q_liq = FT(0),
    q_ice = FT(0),
    T = FT(0),
    RH = FT(0),
    buoy = FT(0),
    area = FT(0),
    q_tot = FT(0),
    θ_liq_ice = FT(0),
    e_tot = FT(0),
    e_kin = FT(0),
    h_tot = FT(0),
    θ_liq_ice_tendency_precip_formation = FT(0),
    e_tot_tendency_precip_formation = FT(0),
    qt_tendency_precip_formation = FT(0),
    qr_tendency_precip_formation = FT(0),
    qs_tendency_precip_formation = FT(0),
    detr = FT(0),
    entr = FT(0),
)
cent_aux_vars_edmf(::Type{FT}, local_geometry, atmos) where {FT} = (;
    turbconv = (;
        ϕ_temporary = FT(0),
        ψ_temporary = FT(0),
        k̂ = CCG.Contravariant3Vector(CCG.WVector(FT(1)), local_geometry),
        bulk = (;
            area = FT(0),
            h_tot = FT(0),
            buoy = FT(0),
            q_tot = FT(0),
            q_liq = FT(0),
            q_ice = FT(0),
            T = FT(0),
            cloud_fraction = FT(0),
            e_tot_tendency_precip_formation = FT(0),
            qt_tendency_precip_formation = FT(0),
            qr_tendency_precip_formation = FT(0),
            qs_tendency_precip_formation = FT(0),
            filter_flag_1 = FT(0), # tmp flag for testing filters
            filter_flag_2 = FT(0), # tmp flag for testing filters
            filter_flag_3 = FT(0), # tmp flag for testing filters
            filter_flag_4 = FT(0), # tmp flag for testing filters
        ),
        up = ntuple(
            i -> cent_aux_vars_up(FT, local_geometry, atmos.turbconv_model),
            Val(n_updrafts(atmos.turbconv_model)),
        ),
        en = (;
            ts = thermo_state_zeros(atmos.moisture_model, FT),
            area = FT(0),
            q_tot = FT(0),
            q_liq = FT(0),
            q_ice = FT(0),
            θ_liq_ice = FT(0),
            e_tot = FT(0),
            e_kin = FT(0),
            h_tot = FT(0),
            θ_virt = FT(0),
            θ_dry = FT(0),
            RH = FT(0),
            T = FT(0),
            buoy = FT(0),
            cloud_fraction = FT(0),
            tke = FT(0),
            #θ_liq_ice_tendency_precip_formation = FT(0),
            e_tot_tendency_precip_formation = FT(0),
            qt_tendency_precip_formation = FT(0),
            qr_tendency_precip_formation = FT(0),
            qs_tendency_precip_formation = FT(0),
            unsat = (; q_tot = FT(0), θ_dry = FT(0), θ_virt = FT(0)),
            sat = (;
                T = FT(0),
                q_vap = FT(0),
                q_tot = FT(0),
                θ_dry = FT(0),
                θ_liq_ice = FT(0),
            ),
        ),
        KM = FT(0),
        KH = FT(0),
        mixing_length = FT(0),
        prandtl_nvec = FT(0),
        # Variable Prandtl number initialized as neutral value.
        b_exch = FT(0),
        w_up_c = FT(0),
        w_en_c = FT(0),
        Shear² = FT(0),
        ∂θv∂z = FT(0),
        ∂qt∂z = FT(0),
        ∂θl∂z = FT(0),
        ∂θv∂z_unsat = FT(0),
        ∂qt∂z_sat = FT(0),
        ∂θl∂z_sat = FT(0),
    )
)

# Face only
face_aux_vars_up(FT, local_geometry) =
    (; nh_pressure = FT(0), massflux = CCG.Covariant3Vector(FT(0)))
face_aux_vars_edmf(::Type{FT}, local_geometry, edmf) where {FT} = (;
    turbconv = (;
        bulk = (; w = CCG.Covariant3Vector(FT(0))),
        ρ_ae_KM = FT(0),
        ρ_ae_KH = FT(0),
        ρ_ae_K = FT(0),
        en = (; w = CCG.Covariant3Vector(FT(0))),
        up = ntuple(
            i -> face_aux_vars_up(FT, local_geometry),
            Val(n_updrafts(edmf)),
        ),
        massflux = CCG.Covariant3Vector(FT(0)),
        massflux_h = CCG.Covariant3Vector(FT(0)),
        massflux_qt = CCG.Covariant3Vector(FT(0)),
        ϕ_temporary = CCG.Covariant3Vector(FT(0)),
        diffusive_flux_h = CCG.Covariant3Vector(FT(0)),
        diffusive_flux_qt = CCG.Covariant3Vector(FT(0)),
        diffusive_flux_uₕ = CCG.Covariant3Vector(FT(0)) ⊗
                            CCG.Covariant12Vector(FT(0), FT(0)),
        uvw = CCG.Covariant123Vector(CCG.WVector(FT(0)), local_geometry),
    )
)
