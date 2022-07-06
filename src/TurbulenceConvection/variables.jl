#####
##### Fields
#####
import ClimaCore.Geometry: ⊗

# Helpers for adding empty thermodynamic state fields:
thermo_state(FT, ::EquilibriumMoisture) = TD.PhaseEquil{FT}(0, 0, 0, 0, 0)
thermo_state(FT, ::NonEquilibriumMoisture) = TD.PhaseNonEquil{FT}(0, 0, TD.PhasePartition(FT(0), FT(0), FT(0)))

##### Auxiliary fields

# Center only
cent_aux_vars_en_2m(FT) = (;
    dissipation = FT(0),
    shear = FT(0),
    entr_gain = FT(0),
    detr_loss = FT(0),
    press = FT(0),
    buoy = FT(0),
    interdomain = FT(0),
    rain_src = FT(0),
)
cent_aux_vars_up_moisture(FT, ::NonEquilibriumMoisture) = (;
    ql_tendency_precip_formation = FT(0),
    qi_tendency_precip_formation = FT(0),
    ql_tendency_noneq = FT(0),
    qi_tendency_noneq = FT(0),
)
cent_aux_vars_up_moisture(FT, ::EquilibriumMoisture) = NamedTuple()
cent_aux_vars_up(FT, local_geometry, edmf) = (;
    ts = thermo_state(FT, edmf.moisture_model),
    q_liq = FT(0),
    q_ice = FT(0),
    T = FT(0),
    RH = FT(0),
    s = FT(0),
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
    cent_aux_vars_up_moisture(FT, edmf.moisture_model)...,
    entr_sc = FT(0),
    detr_sc = FT(0),
    ε_nondim = FT(0),  # nondimensional entrainment
    δ_nondim = FT(0),  # nondimensional detrainment
    frac_turb_entr = FT(0),
    entr_turb_dyn = FT(0),
    detr_turb_dyn = FT(0),
    asp_ratio = FT(0),
    Π_groups = ntuple(i -> FT(0), n_Π_groups(edmf)),
)
cent_aux_vars_edmf_bulk_moisture(FT, ::NonEquilibriumMoisture) = (;
    ql_tendency_precip_formation = FT(0),
    qi_tendency_precip_formation = FT(0),
    ql_tendency_noneq = FT(0),
    qi_tendency_noneq = FT(0),
)
cent_aux_vars_edmf_bulk_moisture(FT, ::EquilibriumMoisture) = NamedTuple()
cent_aux_vars_edmf_en_moisture(FT, ::NonEquilibriumMoisture) = (;
    ql_tendency_precip_formation = FT(0),
    qi_tendency_precip_formation = FT(0),
    ql_tendency_noneq = FT(0),
    qi_tendency_noneq = FT(0),
)
cent_aux_vars_edmf_en_moisture(FT, ::EquilibriumMoisture) = NamedTuple()
cent_aux_vars_edmf_moisture(FT, ::NonEquilibriumMoisture) = (;
    massflux_tendency_ql = FT(0),
    massflux_tendency_qi = FT(0),
    diffusive_tendency_ql = FT(0),
    diffusive_tendency_qi = FT(0),
)
cent_aux_vars_edmf_moisture(FT, ::EquilibriumMoisture) = NamedTuple()
cent_aux_vars_edmf(::Type{FT}, local_geometry, edmf) where {FT} = (;
    turbconv = (;
        ϕ_temporary = FT(0),
        ψ_temporary = FT(0),
        bulk = (;
            area = FT(0),
            θ_liq_ice = FT(0),
            e_tot = FT(0),
            h_tot = FT(0),
            RH = FT(0),
            buoy = FT(0),
            q_tot = FT(0),
            q_liq = FT(0),
            q_ice = FT(0),
            T = FT(0),
            cloud_fraction = FT(0),
            e_tot_tendency_precip_formation = FT(0),
            qt_tendency_precip_formation = FT(0),
            cent_aux_vars_edmf_bulk_moisture(FT, edmf.moisture_model)...,
        ),
        up = ntuple(i -> cent_aux_vars_up(FT, local_geometry, edmf), Val(n_updrafts(edmf))),
        en = (;
            ts = thermo_state(FT, edmf.moisture_model),
            w = FT(0),
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
            s = FT(0),
            T = FT(0),
            buoy = FT(0),
            cloud_fraction = FT(0),
            tke = FT(0),
            Hvar = FT(0),
            QTvar = FT(0),
            HQTcov = FT(0),
            #θ_liq_ice_tendency_precip_formation = FT(0),
            e_tot_tendency_precip_formation = FT(0),
            qt_tendency_precip_formation = FT(0),
            cent_aux_vars_edmf_en_moisture(FT, edmf.moisture_model)...,
            unsat = (; q_tot = FT(0), θ_dry = FT(0), θ_virt = FT(0)),
            sat = (; T = FT(0), q_vap = FT(0), q_tot = FT(0), θ_dry = FT(0), θ_liq_ice = FT(0)),
            Hvar_rain_dt = FT(0),
            QTvar_rain_dt = FT(0),
            HQTcov_rain_dt = FT(0),
        ),
        θ_liq_ice_tendency_precip_sinks = FT(0),
        e_tot_tendency_precip_sinks = FT(0),
        qt_tendency_precip_sinks = FT(0),
        qr_tendency_evap = FT(0),
        qs_tendency_melt = FT(0),
        qs_tendency_dep_sub = FT(0),
        qr_tendency_advection = FT(0),
        qs_tendency_advection = FT(0),
        en_2m = (;
            tke = cent_aux_vars_en_2m(FT),
            Hvar = cent_aux_vars_en_2m(FT),
            QTvar = cent_aux_vars_en_2m(FT),
            HQTcov = cent_aux_vars_en_2m(FT),
        ),
        term_vel_rain = FT(0),
        term_vel_snow = FT(0),
        KM = FT(0),
        KH = FT(0),
        mixing_length = FT(0),
        massflux_tendency_h = FT(0),
        massflux_tendency_qt = FT(0),
        diffusive_tendency_h = FT(0),
        diffusive_tendency_qt = FT(0),
        cent_aux_vars_edmf_moisture(FT, edmf.moisture_model)...,
        prandtl_nvec = FT(0),
        # Added by Ignacio : Length scheme in use (mls), and smooth min effect (ml_ratio)
        # Variable Prandtl number initialized as neutral value.
        mls = FT(0),
        b_exch = FT(0),
        ml_ratio = FT(0),
        w_up_c = FT(0),
        w_en_c = FT(0),
        Shear² = FT(0),
        ∂θv∂z = FT(0),
        ∂qt∂z = FT(0),
        ∂θl∂z = FT(0),
        ∂θv∂z_unsat = FT(0),
        ∂qt∂z_sat = FT(0),
        ∂θl∂z_sat = FT(0),
        l_entdet = FT(0),
        ϕ_gm = FT(0), # temporary for grid-mean variables
        ϕ_gm_cov = FT(0), # temporary for grid-mean covariance variables
        ϕ_en_cov = FT(0), # temporary for environmental covariance variables
        ϕ_up_cubed = FT(0), # temporary for cubed updraft variables in grid mean 3rd moment functions
    )
)

# Face only
face_aux_vars_up(FT, local_geometry) = (;
    w = FT(0),
    nh_pressure = FT(0),
    nh_pressure_b = FT(0),
    nh_pressure_adv = FT(0),
    nh_pressure_drag = FT(0),
    massflux = FT(0),
)
face_aux_vars_edmf_moisture(FT, ::NonEquilibriumMoisture) = (;
    massflux_en = FT(0), # TODO: is this the right place for this?
    massflux_ql = FT(0),
    massflux_qi = FT(0),
    diffusive_flux_ql = FT(0),
    diffusive_flux_qi = FT(0),
)
face_aux_vars_edmf_moisture(FT, ::EquilibriumMoisture) = NamedTuple()
face_aux_vars_edmf(::Type{FT}, local_geometry, edmf) where {FT} = (;
    turbconv = (;
        bulk = (; w = FT(0)),
        ρ_ae_KM = FT(0),
        ρ_ae_KH = FT(0),
        ρ_ae_K = FT(0),
        en = (; w = FT(0)),
        up = ntuple(i -> face_aux_vars_up(FT, local_geometry), Val(n_updrafts(edmf))),
        massflux = FT(0),
        massflux_h = FT(0),
        massflux_qt = FT(0),
        ϕ_temporary = FT(0),
        diffusive_flux_h = FT(0),
        diffusive_flux_qt = FT(0),
        face_aux_vars_edmf_moisture(FT, edmf.moisture_model)...,
        diffusive_flux_uₕ = CCG.Covariant3Vector(FT(0)) ⊗ CCG.Covariant12Vector(FT(0), FT(0)),
    )
)

##### Diagnostic fields

# Center only
cent_diagnostic_vars_edmf(FT, local_geometry, edmf) = (;
    turbconv = (;
        asp_ratio = FT(0),
        entr_sc = FT(0),
        ε_nondim = FT(0),
        detr_sc = FT(0),
        δ_nondim = FT(0),
        massflux = FT(0),
        frac_turb_entr = FT(0),
    )
)

# Face only
face_diagnostic_vars_edmf(FT, local_geometry, edmf) = (;
    turbconv = (; nh_pressure = FT(0), nh_pressure_adv = FT(0), nh_pressure_drag = FT(0), nh_pressure_b = FT(0)),
    precip = (; rain_flux = FT(0), snow_flux = FT(0)),
)

# Single value per column diagnostic variables
single_value_per_col_diagnostic_vars_edmf(FT, edmf) = (;
    turbconv = (;
        env_cloud_base = FT(0),
        env_cloud_top = FT(0),
        env_cloud_cover = FT(0),
        env_lwp = FT(0),
        env_iwp = FT(0),
        updraft_cloud_cover = FT(0),
        updraft_cloud_base = FT(0),
        updraft_cloud_top = FT(0),
        updraft_lwp = FT(0),
        updraft_iwp = FT(0),
        Hd = FT(0),
    )
)

##### Prognostic fields

# Center only
cent_prognostic_vars_up_noisy_relaxation(::Type{FT}, ::PrognosticNoisyRelaxationProcess) where {FT} =
    (; ε_nondim = FT(0), δ_nondim = FT(0))
cent_prognostic_vars_up_noisy_relaxation(::Type{FT}, _) where {FT} = NamedTuple()
cent_prognostic_vars_up_moisture(::Type{FT}, ::EquilibriumMoisture) where {FT} = NamedTuple()
cent_prognostic_vars_up_moisture(::Type{FT}, ::NonEquilibriumMoisture) where {FT} = (; ρaq_liq = FT(0), ρaq_ice = FT(0))
cent_prognostic_vars_up(::Type{FT}, edmf) where {FT} = (;
    ρarea = FT(0),
    ρaθ_liq_ice = FT(0),
    ρaq_tot = FT(0),
    cent_prognostic_vars_up_noisy_relaxation(FT, edmf.entr_closure)...,
    cent_prognostic_vars_up_moisture(FT, edmf.moisture_model)...,
)

cent_prognostic_vars_en(::Type{FT}, edmf) where {FT} =
    (; ρatke = FT(0), cent_prognostic_vars_en_thermo(FT, edmf.thermo_covariance_model)...)
cent_prognostic_vars_en_thermo(::Type{FT}, ::DiagnosticThermoCovariances) where {FT} = NamedTuple()
cent_prognostic_vars_en_thermo(::Type{FT}, ::PrognosticThermoCovariances) where {FT} =
    (; ρaHvar = FT(0), ρaQTvar = FT(0), ρaHQTcov = FT(0))
cent_prognostic_vars_edmf(::Type{FT}, edmf) where {FT} = (;
    turbconv = (;
        en = cent_prognostic_vars_en(FT, edmf),
        up = ntuple(i -> cent_prognostic_vars_up(FT, edmf), Val(n_updrafts(edmf))),
        pr = (; q_rai = FT(0), q_sno = FT(0)),
    )
)
# cent_prognostic_vars_edmf(FT, edmf) = (;) # could also use this for empty model

# Face only
face_prognostic_vars_up(::Type{FT}, local_geometry) where {FT} = (; ρaw = FT(0))
face_prognostic_vars_edmf(::Type{FT}, local_geometry, edmf) where {FT} =
    (; turbconv = (; up = ntuple(i -> face_prognostic_vars_up(FT, local_geometry), Val(n_updrafts(edmf)))))
