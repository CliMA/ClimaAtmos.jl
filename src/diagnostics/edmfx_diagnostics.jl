# This file is included in Diagnostics.jl

# EDMFX diagnostics

###
# Updraft area fraction (3d)
###
compute_arup(state, cache, time) =
    compute_arup(state, cache, time, cache.atmos.turbconv_model)
compute_arup(_, _, _, turbconv_model) = error_diagnostic_variable("arup", turbconv_model)

function compute_arup(state, cache, _, ::PrognosticEDMFX)
    ᶜρaʲ = (state.c.sgsʲs.:1).ρa
    ᶜρʲ = cache.precomputed.ᶜρʲs.:1
    return @. lazy(draft_area(ᶜρaʲ, ᶜρʲ))
end

function compute_arup(_, cache, _, ::DiagnosticEDMFX)
    ᶜρaʲ = cache.precomputed.ᶜρaʲs.:1
    ᶜρʲ = cache.precomputed.ᶜρʲs.:1
    return @. lazy(draft_area(ᶜρaʲ, ᶜρʲ))
end

add_diagnostic_variable!(short_name = "arup", units = "",
    long_name = "Updraft Area Fraction",
    comments = "Area fraction of the first updraft",
    compute = compute_arup,
)

###
# Updraft density (3d)
###
compute_rhoaup(state, cache, time) =
    compute_rhoaup(state, cache, time, cache.atmos.turbconv_model)
compute_rhoaup(_, _, _, turbconv_model) =
    error_diagnostic_variable("rhoaup", turbconv_model)

compute_rhoaup(_, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX}) =
    cache.precomputed.ᶜρʲs.:1

add_diagnostic_variable!(short_name = "rhoaup", units = "kg m^-3",
    long_name = "Updraft Air Density",
    comments = "Density of the first updraft",
    compute = compute_rhoaup,
)

###
# Updraft w velocity (3d)
###
compute_waup(state, cache, time) =
    compute_waup(state, cache, time, cache.atmos.turbconv_model)
compute_waup(_, _, _, turbconv_model) = error_diagnostic_variable("waup", turbconv_model)

compute_waup(_, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX}) =
    @. lazy(w_component(WVec(cache.precomputed.ᶜuʲs.:1)))

add_diagnostic_variable!(short_name = "waup", units = "m s^-1",
    long_name = "Updraft Upward Air Velocity",
    comments = "Vertical wind component of the first updraft",
    compute = compute_waup,
)

###
# Updraft temperature (3d)
###
compute_taup(state, cache, time) =
    compute_taup(state, cache, time, cache.atmos.turbconv_model)
compute_taup(_, _, _, turbconv_model) = error_diagnostic_variable("taup", turbconv_model)

compute_taup(_, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX}) =
    cache.precomputed.ᶜTʲs.:1

add_diagnostic_variable!(short_name = "taup", units = "K",
    long_name = "Updraft Air Temperature",
    comments = "Temperature of the first updraft",
    compute = compute_taup,
)

###
# Updraft potential temperature (3d)
###
compute_thetaaup(state, cache, time) =
    compute_thetaaup(state, cache, time, cache.atmos.turbconv_model)
compute_thetaaup(_, _, _, turbconv_model) =
    error_diagnostic_variable("thetaaup", turbconv_model)

function compute_thetaaup(_, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX})
    thermo_params = CAP.thermodynamics_params(cache.params)
    (; ᶜTʲs, ᶜρʲs, ᶜq_tot_nonnegʲs, ᶜq_liqʲs, ᶜq_iceʲs) = cache.precomputed
    return @. lazy(
        TD.potential_temperature(
            thermo_params,
            ᶜTʲs.:1, ᶜρʲs.:1, ᶜq_tot_nonnegʲs.:1, ᶜq_liqʲs.:1, ᶜq_iceʲs.:1,
        ),
    )
end

add_diagnostic_variable!(short_name = "thetaaup", units = "K",
    long_name = "Updraft Air Potential Temperature",
    comments = "Potential Temperature of the first updraft",
    compute = compute_thetaaup,
)

###
# Updraft specific enthalpy (3d)
###
compute_haup(state, cache, time) =
    compute_haup(state, cache, time, cache.atmos.turbconv_model)
compute_haup(_, _, _, turbconv_model) = error_diagnostic_variable("haup", turbconv_model)

function compute_haup(_, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX})
    thermo_params = CAP.thermodynamics_params(cache.params)
    (; ᶜTʲs, ᶜq_tot_nonnegʲs, ᶜq_liqʲs, ᶜq_iceʲs) = cache.precomputed
    return @. lazy(
        TD.enthalpy(
            thermo_params, ᶜTʲs.:1, ᶜq_tot_nonnegʲs.:1, ᶜq_liqʲs.:1,
            ᶜq_iceʲs.:1,
        ),
    )
end

add_diagnostic_variable!(short_name = "haup", units = "m^2 s^-2",
    long_name = "Updraft Air Specific Enthalpy",
    comments = "Specific enthalpy of the first updraft",
    compute = compute_haup,
)

###
# Updraft total specific humidity (3d)
###
compute_husup(state, cache, time) = compute_husup(
    state, cache, time, cache.atmos.microphysics_model, cache.atmos.turbconv_model,
)
compute_husup(_, _, _, _, _) =
    error_diagnostic_variable("Can only compute updraft specific humidity \
                               with a moist model and with EDMFX")

# TODO: use the actual q_tot
compute_husup(_, cache, _, ::MoistMicrophysics, ::Union{PrognosticEDMFX, DiagnosticEDMFX}) =
    cache.precomputed.ᶜq_tot_nonnegʲs.:1

add_diagnostic_variable!(short_name = "husup", units = "kg kg^-1",
    long_name = "Updraft Specific Humidity",
    comments = "Specific humidity of the first updraft",
    compute = compute_husup,
)

###
# Updraft relative humidity (3d)
###
compute_hurup(state, cache, time) = compute_hurup(
    state, cache, time, cache.atmos.microphysics_model, cache.atmos.turbconv_model,
)
compute_hurup(_, _, _, _, _) =
    error_diagnostic_variable("Can only compute updraft relative humidity \
                               and with a moist model and with EDMFX")

function compute_hurup(_, cache, _,
    ::MoistMicrophysics, ::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    (; ᶜTʲs, ᶜp, ᶜq_tot_nonnegʲs, ᶜq_liqʲs, ᶜq_iceʲs) = cache.precomputed
    return @. lazy(
        TD.relative_humidity(
            thermo_params, ᶜTʲs.:1, ᶜp, ᶜq_tot_nonnegʲs.:1, ᶜq_liqʲs.:1,
            ᶜq_iceʲs.:1,
        ),
    )
end

add_diagnostic_variable!(short_name = "hurup", units = "",
    long_name = "Updraft Relative Humidity",
    comments = "Relative humidity of the first updraft",
    compute = compute_hurup,
)

###
# Updraft liquid water specific humidity and number mixing ratio (3d)
###
compute_clwup(state, cache, time) = compute_clwup(
    state, cache, time, cache.atmos.microphysics_model, cache.atmos.turbconv_model,
)
compute_clwup(_, _, _, _, _) =
    error_diagnostic_variable("Can only compute updraft liquid water specific humidity \
                               and with a moist model and with EDMFX")

compute_clwup(_, cache, _,
    ::EquilibriumMicrophysics0M, ::Union{PrognosticEDMFX, DiagnosticEDMFX},
) = cache.precomputed.ᶜq_liqʲs.:1

compute_clwup(state, _, _, ::NonEquilibriumMicrophysics, ::PrognosticEDMFX) =
    (state.c.sgsʲs.:1).q_lcl

compute_clwup(_, cache, _, ::NonEquilibriumMicrophysics, ::DiagnosticEDMFX) =
    cache.precomputed.ᶜq_lclʲs.:1

add_diagnostic_variable!(
    short_name = "clwup", units = "kg kg^-1",
    long_name = "Updraft Mass Fraction of Cloud Liquid Water",
    comments = """
    This is calculated as the mass of cloud liquid water in the first updraft divided by
    the mass of air (including the water in all phases) in the first updraft.
    """,
    compute = compute_clwup,
)

compute_cdncup(state, cache, time) = compute_cdncup(state, cache, time,
    cache.atmos.microphysics_model, cache.atmos.turbconv_model,
)
compute_cdncup(_, _, _, _, _) =
    error_diagnostic_variable("Can only compute updraft rain number mixing ratio \
                               with a 2M precip model and with EDMFX")

compute_cdncup(state, _, _, ::NonEquilibriumMicrophysics2M, ::PrognosticEDMFX) =
    (state.c.sgsʲs.:1).n_lcl

add_diagnostic_variable!(short_name = "cdncup", units = "kg^-1",
    long_name = "Updraft Number Mixing Ratio of Cloud Liquid Water",
    comments = """
    This is calculated as the number of cloud water droplets in the updraft divided by
    the mass of air (including the water in all phases) in the updraft.
    """,
    compute = compute_cdncup,
)

###
# Updraft ice water specific humidity (3d)
###
compute_cliup(state, cache, time) = compute_cliup(state, cache, time,
    cache.atmos.microphysics_model, cache.atmos.turbconv_model,
)
compute_cliup(_, _, _, _, _) =
    error_diagnostic_variable("Can only compute updraft ice water specific humidity \
                               with a moist model and with EDMFX")

compute_cliup(_, cache, _,
    ::EquilibriumMicrophysics0M, ::Union{PrognosticEDMFX, DiagnosticEDMFX},
) = cache.precomputed.ᶜq_iceʲs.:1

compute_cliup(state, _, _, ::NonEquilibriumMicrophysics, ::PrognosticEDMFX) =
    (state.c.sgsʲs.:1).q_icl
compute_cliup(_, cache, _, ::NonEquilibriumMicrophysics, ::DiagnosticEDMFX) =
    cache.precomputed.ᶜq_iclʲs.:1

add_diagnostic_variable!(short_name = "cliup", units = "kg kg^-1",
    long_name = "Updraft Mass Fraction of Cloud Ice",
    comments = """
    This is calculated as the mass of cloud ice in the first updraft divided by
    the mass of air (including the water in all phases) in the first updraft.
    """,
    compute = compute_cliup,
)

###
# Updraft rain water specific humidity and number mixing ratio (3d)
###
compute_husraup(state, cache, time) = compute_husraup(state, cache, time,
    cache.atmos.microphysics_model, cache.atmos.turbconv_model,
)
compute_husraup(_, _, _, _, _) =
    error_diagnostic_variable("Can only compute updraft rain water specific humidity
                               with a 1M or 2M precip model and with EDMFX")

compute_husraup(state, _, _,
    ::Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}, ::PrognosticEDMFX,
) = (state.c.sgsʲs.:1).q_rai
compute_husraup(_, cache, _, ::NonEquilibriumMicrophysics1M, ::DiagnosticEDMFX) =
    cache.precomputed.ᶜq_raiʲs.:1

add_diagnostic_variable!(short_name = "husraup", units = "kg kg^-1",
    long_name = "Updraft Mass Fraction of Rain",
    comments = """
    This is calculated as the mass of rain in the first updraft divided by
    the mass of air (including the water in all phases) in the first updraft.
    """,
    compute = compute_husraup,
)

compute_ncraup(state, cache, time) = compute_ncraup(state, cache, time,
    cache.atmos.microphysics_model, cache.atmos.turbconv_model,
)
compute_ncraup(_, _, _, _, _) =
    error_diagnostic_variable("Can only compute updraft rain number mixing ratio
                               with a 2M precip model and with EDMFX")

compute_ncraup(state, _, _, ::NonEquilibriumMicrophysics2M, ::PrognosticEDMFX) =
    (state.c.sgsʲs.:1).n_rai

add_diagnostic_variable!(short_name = "ncraup", units = "kg^-1",
    long_name = "Updraft Number Mixing Ratio of Rain",
    comments = """
    This is calculated as the number of raindrops in the updraft divided by
    the mass of air (including the water in all phases) in the updraft.
    """,
    compute = compute_ncraup,
)

###
# Updraft snow specific humidity (3d)
###
compute_hussnup(state, cache, time) = compute_hussnup(state, cache, time,
    cache.atmos.microphysics_model, cache.atmos.turbconv_model,
)
compute_hussnup(_, _, _, _, _) =
    error_diagnostic_variable("Can only compute updraft snow specific humidity
                               with a 1M or 2M precip model and with EDMFX")

compute_hussnup(state, _, _,
    ::Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}, ::PrognosticEDMFX,
) = (state.c.sgsʲs.:1).q_sno
compute_hussnup(_, cache, _, ::NonEquilibriumMicrophysics1M, ::DiagnosticEDMFX) =
    cache.precomputed.ᶜq_snoʲs.:1

add_diagnostic_variable!(short_name = "hussnup", units = "kg kg^-1",
    long_name = "Updraft Mass Fraction of Snow",
    comments = """
    This is calculated as the mass of snow in the first updraft divided by
    the mass of air (including the water in all phases) in the first updraft.
    """,
    compute = compute_hussnup,
)

###
# Entrainment (3d)
###
compute_entr(state, cache, time) =
    compute_entr(state, cache, time, cache.atmos.turbconv_model)
compute_entr(_, _, _, turbconv_model) = error_diagnostic_variable("entr", turbconv_model)

function compute_entr(state, cache, _, ::PrognosticEDMFX)
    (; ᶜentr_vel_scaleʲs, ᶜarea_bounding_entr_detrʲs, ᶜuʲs) = cache.precomputed
    ᶜlg = Fields.local_geometry_field(state.c)
    return @. lazy(
        compute_entrainment(
            ᶜentr_vel_scaleʲs.:1,
            ᶜarea_bounding_entr_detrʲs.:1,
            get_physical_w(ᶜuʲs.:1, ᶜlg),
        ),
    )
end

compute_entr(_, cache, _, ::DiagnosticEDMFX) = cache.precomputed.ᶜentrʲs.:1

add_diagnostic_variable!(short_name = "entr", units = "s^-1",
    long_name = "Entrainment rate",
    comments = "Entrainment rate of the first updraft",
    compute = compute_entr,
)

compute_turbentr(state, cache, time) =
    compute_turbentr(state, cache, time, cache.atmos.turbconv_model)
compute_turbentr(_, _, _, turbconv_model) =
    error_diagnostic_variable("turbentr", turbconv_model)

compute_turbentr(_, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX}) =
    cache.precomputed.ᶜturb_entrʲs.:1

add_diagnostic_variable!(short_name = "turbentr", units = "s^-1",
    long_name = "Turbulent entrainment rate",
    comments = "Turbulent entrainment rate of the first updraft",
    compute = compute_turbentr,
)

###
# Detrainment (3d)
###
compute_detr(state, cache, time) =
    compute_detr(state, cache, time, cache.atmos.turbconv_model)
compute_detr(_, _, _, turbconv_model) = error_diagnostic_variable("detr", turbconv_model)

function compute_detr(state, cache, _, ::PrognosticEDMFX)
    (; ᶜρ_diffʲs, ᶜρʲs, ᶜarea_bounding_entr_detrʲs) = cache.precomputed
    (; ᶠgradᵥ_ᶜΦ) = cache.core
    turbconv_params = CAP.turbconv_params(cache.params)
    detr_buoy_inv_tau_max = CAP.detr_buoy_inv_tau_max(turbconv_params)
    detr_model = cache.atmos.edmfx_model.detr_model
    ᶠlg = Fields.local_geometry_field(state.f)
    ᶠdz = Fields.Δz_field(axes(state.f))
    ρaʲ = state.c.sgsʲs.:(1).ρa
    u₃ʲ = state.f.sgsʲs.:(1).u₃
    # Evaluate the buoyancy inverse time scale at faces (where w and grad_Φ are
    # naturally defined) and interpolate to centers for smoother behaviour.
    ᶜbuoy_inv_time_scale = @. lazy(
        ᶜinterp(
            detr_buoy_inv_time_scale(
                u₃ʲ.components.data.:1 / ᶠdz,
                vertical_buoyancy_acceleration(
                    ᶠinterp(ᶜρ_diffʲs.:1),
                    ᶠgradᵥ_ᶜΦ,
                    ᶠlg,
                ),
                detr_buoy_inv_tau_max,
            ),
        ),
    )
    return @. lazy(
        compute_detrainment(
            turbconv_params,
            draft_area(ρaʲ, ᶜρʲs.:1),
            ρaʲ,
            ᶜbuoy_inv_time_scale,
            ᶜdivᵥ(ᶠleft_bias(ρaʲ) * u₃ʲ),
            ᶜarea_bounding_entr_detrʲs.:1,
            detr_model,
        ),
    )
end

compute_detr(_, cache, _, ::DiagnosticEDMFX) = cache.precomputed.ᶜdetrʲs.:1

add_diagnostic_variable!(short_name = "detr", units = "s^-1",
    long_name = "Detrainment rate",
    comments = "Detrainment rate of the first updraft",
    compute = compute_detr,
)

###
# Environment area fraction (3d)
###
compute_aren(state, cache, time) =
    compute_aren(state, cache, time, cache.atmos.turbconv_model)
compute_aren(_, _, _, turbconv_model) = error_diagnostic_variable("aren", turbconv_model)

function compute_aren(state, cache, _, turbconv_model::PrognosticEDMFX)
    thermo_params = CAP.thermodynamics_params(cache.params)
    ᶜρa⁰ = @. lazy(ρa⁰(state.c.ρ, state.c.sgsʲs, turbconv_model))
    (; ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = cache.precomputed
    ᶜρ⁰ = @. lazy(
        TD.air_density(
            thermo_params,
            ᶜT⁰,
            ᶜp,
            ᶜq_tot_nonneg⁰,
            ᶜq_liq⁰,
            ᶜq_ice⁰,
        ),
    )
    return @. lazy(draft_area(ᶜρa⁰, ᶜρ⁰))
end

compute_aren(_, cache, _, ::DiagnosticEDMFX) =
    @. lazy(1 - draft_area(cache.precomputed.ᶜρaʲs.:1, cache.precomputed.ᶜρʲs.:1))

add_diagnostic_variable!(short_name = "aren", units = "",
    long_name = "Environment Area Fraction",
    compute = compute_aren,
)

###
# Environment density (3d)
###
compute_rhoaen(state, cache, time) =
    compute_rhoaen(state, cache, time, cache.atmos.turbconv_model)
compute_rhoaen(_, _, _, turbconv_model) =
    error_diagnostic_variable("rhoaen", turbconv_model)

function compute_rhoaen(_, cache, _, ::PrognosticEDMFX)
    thermo_params = CAP.thermodynamics_params(cache.params)
    (; ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = cache.precomputed
    return @. lazy(
        TD.air_density(
            thermo_params,
            ᶜT⁰,
            ᶜp,
            ᶜq_tot_nonneg⁰,
            ᶜq_liq⁰,
            ᶜq_ice⁰,
        ),
    )
end

add_diagnostic_variable!(short_name = "rhoaen", units = "kg m^-3",
    long_name = "Environment Air Density",
    compute = compute_rhoaen,
)

###
# Environment w velocity (3d)
###
compute_waen(state, cache, time) =
    compute_waen(state, cache, time, cache.atmos.turbconv_model)
compute_waen(_, _, _, turbconv_model) = error_diagnostic_variable("waen", turbconv_model)

compute_waen(_, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX}) =
    @. lazy(w_component(WVec(cache.precomputed.ᶜu⁰)))

add_diagnostic_variable!(short_name = "waen", units = "m s^-1",
    long_name = "Environment Upward Air Velocity",
    comments = "Vertical wind component of the environment",
    compute = compute_waen,
)

###
# Environment temperature (3d)
###
compute_taen(state, cache, time) =
    compute_taen(state, cache, time, cache.atmos.turbconv_model)
compute_taen(_, _, _, turbconv_model) = error_diagnostic_variable("taen", turbconv_model)

compute_taen(_, cache, _, ::PrognosticEDMFX) = cache.precomputed.ᶜT⁰

add_diagnostic_variable!(short_name = "taen", units = "K",
    long_name = "Environment Air Temperature",
    compute = compute_taen,
)

###
# Environment potential temperature (3d)
###
compute_thetaaen(state, cache, time) =
    compute_thetaaen(state, cache, time, cache.atmos.turbconv_model)
compute_thetaaen(_, _, _, turbconv_model) =
    error_diagnostic_variable("thetaaen", turbconv_model)

function compute_thetaaen(_, cache, _, ::PrognosticEDMFX)
    thermo_params = CAP.thermodynamics_params(cache.params)
    (; ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = cache.precomputed
    ᶜρ⁰ = @. lazy(
        TD.air_density(
            thermo_params,
            ᶜT⁰,
            ᶜp,
            ᶜq_tot_nonneg⁰,
            ᶜq_liq⁰,
            ᶜq_ice⁰,
        ),
    )
    return @. lazy(
        TD.potential_temperature(
            thermo_params, ᶜT⁰, ᶜρ⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰,
        ),
    )
end

add_diagnostic_variable!(short_name = "thetaaen", units = "K",
    long_name = "Environment Air Potential Temperature",
    compute = compute_thetaaen,
)

###
# Environment specific enthalpy (3d)
###
compute_haen(state, cache, time) =
    compute_haen(state, cache, time, cache.atmos.turbconv_model)
compute_haen(_, _, _, turbconv_model) = error_diagnostic_variable("haen", turbconv_model)

function compute_haen(_, cache, _, ::PrognosticEDMFX)
    thermo_params = CAP.thermodynamics_params(cache.params)
    (; ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = cache.precomputed
    return @. lazy(
        TD.enthalpy(thermo_params, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰),
    )
end

add_diagnostic_variable!(short_name = "haen", units = "K",
    long_name = "Environment Air Specific Enthalpy",
    compute = compute_haen,
)

###
# Environment total specific humidity (3d)
###
compute_husen(state, cache, time) = compute_husen(state, cache, time,
    cache.atmos.microphysics_model, cache.atmos.turbconv_model,
)
compute_husen(_, _, _, _, _) =
    error_diagnostic_variable("Can only compute environment specific humidity \
                               with a moist model and with PrognosticEDMFX")

compute_husen(_, cache, _, ::MoistMicrophysics, ::PrognosticEDMFX) =
    cache.precomputed.ᶜq_tot_nonneg⁰

add_diagnostic_variable!(short_name = "husen", units = "kg kg^-1",
    long_name = "Environment Specific Humidity",
    compute = compute_husen,
)

###
# Environment relative humidity (3d)
###
compute_huren(state, cache, time) = compute_huren(state, cache, time,
    cache.atmos.microphysics_model, cache.atmos.turbconv_model,
)
compute_huren(_, _, _, _, _) =
    error_diagnostic_variable("Can only compute environment relative humidity \
                               with a moist model and with PrognosticEDMFX")

function compute_huren(_, cache, _, ::MoistMicrophysics, ::PrognosticEDMFX)
    thermo_params = CAP.thermodynamics_params(cache.params)
    (; ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = cache.precomputed
    return @. lazy(
        TD.relative_humidity(
            thermo_params, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰,
        ),
    )
end

add_diagnostic_variable!(short_name = "huren", units = "",
    long_name = "Environment Relative Humidity",
    compute = compute_huren,
)

###
# Environment liquid water specific humidity and number mixing ratio (3d)
###
compute_clwen(state, cache, time) = compute_clwen(state, cache, time,
    cache.atmos.microphysics_model, cache.atmos.turbconv_model,
)
compute_clwen(_, _, _, _, _) =
    error_diagnostic_variable(
        "Can only compute environment liquid water specific humidity \
         with a moist model and with PrognosticEDMFX",
    )

compute_clwen(_, cache, _, ::EquilibriumMicrophysics0M, ::PrognosticEDMFX) =
    cache.precomputed.ᶜq_liq⁰

compute_clwen(state, cache, _, ::NonEquilibriumMicrophysics, ::PrognosticEDMFX) =
    ᶜspecific_env_value(@name(q_lcl), state, cache)

add_diagnostic_variable!(short_name = "clwen", units = "kg kg^-1",
    long_name = "Envrionment Mass Fraction of Cloud Liquid Water",
    comments = """
    This is calculated as the mass of cloud liquid water in the environment divided by
    the mass of air (including the water in all phases) in the environment.
    """,
    compute = compute_clwen,
)

compute_cdncen(state, cache, time) = compute_cdncen(state, cache, time,
    cache.atmos.microphysics_model, cache.atmos.turbconv_model,
)
compute_cdncen(_, _, _, _, _) =
    error_diagnostic_variable(
        "Can only compute environment cloud liquid water \
         number mixing ratio with a 2M model and with PrognosticEDMFX",
    )

compute_cdncen(state, cache, _, ::NonEquilibriumMicrophysics2M, ::PrognosticEDMFX) =
    ᶜspecific_env_value(@name(n_lcl), state, cache)

add_diagnostic_variable!(short_name = "cdncen", units = "kg^-1",
    long_name = "Environment Number Mixing Ratio of Cloud Liquid Water",
    comments = """
    This is calculated as the number of cloud liquid droplets in the environment divided by
    the mass of air (including the water in all phases) in the environment.
    """,
    compute = compute_cdncen,
)

###
# Environment ice water specific humidity (3d)
###
compute_clien(state, cache, time) = compute_clien(
    state, cache, time, cache.atmos.microphysics_model, cache.atmos.turbconv_model,
)
compute_clien(_, _, _, _, _) =
    error_diagnostic_variable("Can only compute environment ice water specific humidity \
                               with a moist model and with PrognosticEDMFX")

compute_clien(_, cache, _, ::EquilibriumMicrophysics0M, ::PrognosticEDMFX) =
    cache.precomputed.ᶜq_ice⁰

compute_clien(state, cache, _, ::NonEquilibriumMicrophysics, ::PrognosticEDMFX) =
    ᶜspecific_env_value(@name(q_icl), state, cache)

add_diagnostic_variable!(short_name = "clien", units = "kg kg^-1",
    long_name = "Environment Mass Fraction of Cloud Ice",
    comments = """
    This is calculated as the mass of cloud ice in the environment divided by
    the mass of air (including the water in all phases) in the environment.
    """,
    compute = compute_clien,
)

###
# Environment rain water specific humidity and number mixing ratio (3d)
###
compute_husraen(state, cache, time) = compute_husraen(
    state, cache, time, cache.atmos.microphysics_model, cache.atmos.turbconv_model,
)
compute_husraen(_, _, _, _, _) =
    error_diagnostic_variable("Can only compute environment rain specific humidity \
                               with a 1M or 2M model and with PrognosticEDMFX")

compute_husraen(state, cache, _,
    ::Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}, ::PrognosticEDMFX,
) = ᶜspecific_env_value(@name(q_rai), state, cache)

add_diagnostic_variable!(short_name = "husraen", units = "kg kg^-1",
    long_name = "Environment Mass Fraction of Rain",
    comments = """
    This is calculated as the mass of rain in the environment divided by
    the mass of air (including the water in all phases) in the environment.
    """,
    compute = compute_husraen,
)

compute_ncraen(state, cache, time) = compute_ncraen(
    state, cache, time, cache.atmos.microphysics_model, cache.atmos.turbconv_model,
)
compute_ncraen(_, _, _, _, _) =
    error_diagnostic_variable("Can only compute environment rain number mixing ratio \
                               with a 2M model and with PrognosticEDMFX")

compute_ncraen(state, cache, _, ::NonEquilibriumMicrophysics2M, ::PrognosticEDMFX) =
    ᶜspecific_env_value(@name(n_rai), state, cache)

add_diagnostic_variable!(short_name = "ncraen", units = "kg^-1",
    long_name = "Environment Number Mixing Ratio of Rain",
    comments = """
    This is calculated as the number of raindrops in the environment divided by
    the mass of air (including the water in all phases) in the environment.
    """,
    compute = compute_ncraen,
)

###
# Environment snow water specific humidity (3d)
###
compute_hussnen(state, cache, time) = compute_hussnen(
    state, cache, time, cache.atmos.microphysics_model, cache.atmos.turbconv_model,
)
compute_hussnen(_, _, _, _, _) =
    error_diagnostic_variable("Can only compute environment snow specific humidity \
                               with a 1M or 2M model and with PrognosticEDMFX")

compute_hussnen(state, cache, _,
    ::Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}, ::PrognosticEDMFX,
) = ᶜspecific_env_value(@name(q_sno), state, cache)

add_diagnostic_variable!(short_name = "hussnen", units = "kg kg^-1",
    long_name = "Environment Mass Fraction of Snow",
    comments = """
    This is calculated as the mass of snow in the environment divided by
    the mass of air (including the water in all phases) in the environment.
    """,
    compute = compute_hussnen,
)

###
# Turbulent kinetic energy (3d)
###
compute_tke(state, cache, time) =
    compute_tke(state, cache, time, cache.atmos.turbconv_model)
compute_tke(_, _, _, turbconv_model) = error_diagnostic_variable("tke", turbconv_model)

compute_tke(state, _, _, ::Union{EDOnlyEDMFX, PrognosticEDMFX, DiagnosticEDMFX}) =
    @. lazy(specific(state.c.ρtke, state.c.ρ))

add_diagnostic_variable!(short_name = "tke", units = "m^2 s^-2",
    long_name = "Environment Turbulent Kinetic Energy",
    compute = compute_tke,
)

###
# Environment Wall Constrained Mixing Length (3d)
###
compute_lmixw(state, cache, time) =
    compute_lmixw(state, cache, time, cache.atmos.turbconv_model)
compute_lmixw(_, _, _, turbconv_model) = error_diagnostic_variable("lmixw", turbconv_model)
compute_lmixw(state, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX}) =
    ᶜmixing_length(state, cache, Val(:wall))

add_diagnostic_variable!(short_name = "lmixw", units = "m",
    long_name = "Environment Wall Constrained Mixing Length",
    compute = compute_lmixw,
)

###
# Environment TKE Balanced Mixing Length (3d)
###
compute_lmixtke(state, cache, time) =
    compute_lmixtke(state, cache, time, cache.atmos.turbconv_model)
compute_lmixtke(_, _, _, turbconv_model) =
    error_diagnostic_variable("lmixtke", turbconv_model)

compute_lmixtke(state, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX}) =
    ᶜmixing_length(state, cache, Val(:tke))

add_diagnostic_variable!(short_name = "lmixtke", units = "m",
    long_name = "Environment TKE Balanced Mixing Length",
    compute = compute_lmixtke,
)

###
# Environment Stability Mixing Length (3d)
###
compute_lmixb(state, cache, time) =
    compute_lmixb(state, cache, time, cache.atmos.turbconv_model)
compute_lmixb(_, _, _, turbconv_model) = error_diagnostic_variable("lmixb", turbconv_model)

compute_lmixb(state, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX}) =
    ᶜmixing_length(state, cache, Val(:buoy))

add_diagnostic_variable!(short_name = "lmixb", units = "m",
    long_name = "Environment Static Stability Mixing Length",
    compute = compute_lmixb,
)

###
# Diffusivity of heat (3d)
###
compute_edt(state, cache, time) = compute_edt(
    state, cache, time, cache.atmos.vertical_diffusion, cache.atmos.turbconv_model,
)
compute_edt(_, _, _, _, _) =
    error_diagnostic_variable("Can only compute heat diffusivity with \
                               vertical diffusion or EDMFX")

compute_edt(state, cache, _, model::VerticalDiffusion, ::Nothing) =
    ᶜcompute_eddy_diffusivity_coefficient(state.c.uₕ, cache.precomputed.ᶜp, model)
compute_edt(state, _, _, model::DecayWithHeightDiffusion, ::Nothing) =
    ᶜcompute_eddy_diffusivity_coefficient(state.c.ρ, model)

function compute_edt(state, cache, _,
    ::Nothing, ::Union{PrognosticEDMFX, DiagnosticEDMFX, EDOnlyEDMFX},
)
    turbconv_params = CAP.turbconv_params(cache.params)
    (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = cache.precomputed
    (; params) = cache

    ᶜtke = @. lazy(specific(state.c.ρtke, state.c.ρ))
    ᶜmixing_length_field = ᶜmixing_length(state, cache)
    ᶜK_u = @. lazy(eddy_viscosity(turbconv_params, ᶜtke, ᶜmixing_length_field))
    ᶜprandtl_nvec =
        @. lazy(turbulent_prandtl_number(params, ᶜlinear_buoygrad, ᶜstrain_rate_norm))
    ᶜK_h = @. lazy(eddy_diffusivity(ᶜK_u, ᶜprandtl_nvec))
    return ᶜK_h
end

add_diagnostic_variable!(short_name = "edt", units = "m^2 s^-1",
    long_name = "Eddy Diffusivity Coefficient for Temperature",
    standard_name = "atmosphere_heat_diffusivity",
    comments = "Vertical diffusion coefficient for temperature due to parameterized eddies",
    compute = compute_edt,
)

###
# Diffusivity of momentum (3d)
###
compute_evu(state, cache, time) = compute_evu(
    state, cache, time, cache.atmos.vertical_diffusion, cache.atmos.turbconv_model,
)
compute_evu(_, _, _, _, _) =
    error_diagnostic_variable("Can only compute momentum diffusivity with \
                               vertical diffusion or EDMFX")

# this setup assumes ᶜK_u = ᶜK_h
compute_evu(state, cache, _, model::VerticalDiffusion, ::Nothing) =
    ᶜcompute_eddy_diffusivity_coefficient(state.c.uₕ, cache.precomputed.ᶜp, model)
compute_evu(state, _, _, model::DecayWithHeightDiffusion, ::Nothing) =
    ᶜcompute_eddy_diffusivity_coefficient(state.c.ρ, model)

function compute_evu(
    state,
    cache,
    _,
    ::Nothing,
    ::Union{PrognosticEDMFX, DiagnosticEDMFX, EDOnlyEDMFX},
)
    turbconv_params = CAP.turbconv_params(cache.params)
    ᶜtke = @. lazy(specific(state.c.ρtke, state.c.ρ))
    ᶜmixing_length_field = ᶜmixing_length(state, cache)
    ᶜK_u = @. lazy(eddy_viscosity(turbconv_params, ᶜtke, ᶜmixing_length_field))
    return ᶜK_u
end

add_diagnostic_variable!(short_name = "evu", units = "m^2 s^-1",
    long_name = "Eddy Viscosity Coefficient for Momentum",
    standard_name = "atmosphere_momentum_diffusivity",
    comments = "Vertical diffusion coefficient for momentum due to parameterized eddies",
    compute = compute_evu,
)

###############################################################################
# MIXING LENGTH PI GROUPS (features for data-driven mixing length models)
###############################################################################

###
# Mixing Length Pi1: strain_rate / buoyancy_gradient
###
compute_mlpi1(state, cache, time) =
    compute_mlpi1(state, cache, time, cache.atmos.turbconv_model)
compute_mlpi1(_, _, _, turbconv_model) = error_diagnostic_variable("mlpi1", turbconv_model)

function compute_mlpi1(_, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX, EDOnlyEDMFX})
    (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = cache.precomputed
    FT = eltype(ᶜlinear_buoygrad)
    return @. lazy(ᶜstrain_rate_norm / (ᶜlinear_buoygrad + eps(FT)))
end

add_diagnostic_variable!(short_name = "mlpi1", units = "",
    long_name = "Mixing Length Pi1: strain/buoygrad",
    comments = "Ratio of strain rate norm to buoyancy gradient",
    compute = compute_mlpi1,
)

###
# Mixing Length Pi2: TKE / (buoyancy_gradient * z^2)
###
compute_mlpi2(state, cache, time) =
    compute_mlpi2(state, cache, time, cache.atmos.turbconv_model)
compute_mlpi2(_, _, _, turbconv_model) = error_diagnostic_variable("mlpi2", turbconv_model)

function compute_mlpi2(state, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX, EDOnlyEDMFX})
    (; ᶜlinear_buoygrad) = cache.precomputed
    ᶜz = Fields.coordinate_field(state.c).z
    z_sfc = Fields.level(Fields.coordinate_field(state.f).z, Fields.half)
    ᶜtke = @. lazy(specific(state.c.ρtke, state.c.ρ))
    FT = eltype(ᶜlinear_buoygrad)
    ᶜz_above_sfc = @. lazy(ᶜz - z_sfc)
    return @. lazy(ᶜtke / (ᶜlinear_buoygrad * ᶜz_above_sfc * ᶜz_above_sfc + eps(FT)))
end

add_diagnostic_variable!(short_name = "mlpi2", units = "",
    long_name = "Mixing Length Pi2: TKE/(buoygrad*z^2)",
    comments = "TKE normalized by stratification and height squared",
    compute = compute_mlpi2,
)

###
# Mixing Length Pi3: TKE / (delta_w)^2
###
compute_mlpi3(state, cache, time) =
    compute_mlpi3(state, cache, time, cache.atmos.turbconv_model)
compute_mlpi3(_, _, _, turbconv_model) = error_diagnostic_variable("mlpi3", turbconv_model)

function compute_mlpi3(state, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX})
    (; ᶜu⁰, ᶜuʲs) = cache.precomputed
    ᶜlg = Fields.local_geometry_field(state.c)
    ᶜtke = @. lazy(specific(state.c.ρtke, state.c.ρ))
    FT = eltype(ᶜtke)
    ᶜw_env = @. lazy(get_physical_w(ᶜu⁰, ᶜlg))
    ᶜw_up = @. lazy(get_physical_w(ᶜuʲs.:1, ᶜlg))
    ᶜdelta_w_sq = @. lazy(max((ᶜw_up - ᶜw_env)^2, eps(FT)))
    return @. lazy(ᶜtke / ᶜdelta_w_sq)
end

add_diagnostic_variable!(short_name = "mlpi3", units = "",
    long_name = "Mixing Length Pi3: TKE/delta_w^2",
    comments = "TKE normalized by updraft-environment velocity difference squared",
    compute = compute_mlpi3,
)

###
# z / Obukhov length (stability parameter)
###
compute_zobu(state, cache, time) =
    compute_zobu(state, cache, time, cache.atmos.turbconv_model)
compute_zobu(_, _, _, turbconv_model) = error_diagnostic_variable("zobu", turbconv_model)

function compute_zobu(state, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX, EDOnlyEDMFX})
    (; obukhov_length) = cache.precomputed.sfc_conditions
    ᶜz = Fields.coordinate_field(state.c).z
    z_sfc = Fields.level(Fields.coordinate_field(state.f).z, Fields.half)
    FT = eltype(ᶜz)
    ᶜz_above_sfc = @. lazy(ᶜz - z_sfc)
    # Use sign-preserving safe division for Obukhov length
    obu_safe = @. lazy(ifelse(
        obukhov_length < FT(0),
        min(obukhov_length, -eps(FT)),
        max(obukhov_length, eps(FT)),
    ))
    return @. lazy(ᶜz_above_sfc / obu_safe)
end

add_diagnostic_variable!(short_name = "zobu", units = "",
    long_name = "z/Obukhov Length",
    comments = "Monin-Obukhov stability parameter z/L",
    compute = compute_zobu,
)

###
# dz / Obukhov length (resolution parameter)
###
compute_resobu(state, cache, time) =
    compute_resobu(state, cache, time, cache.atmos.turbconv_model)
compute_resobu(_, _, _, turbconv_model) = error_diagnostic_variable("resobu", turbconv_model)

function compute_resobu(state, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX, EDOnlyEDMFX})
    (; obukhov_length) = cache.precomputed.sfc_conditions
    ᶜdz = Fields.Δz_field(axes(state.c))
    FT = eltype(ᶜdz)
    obu_safe = @. lazy(ifelse(
        obukhov_length < FT(0),
        min(obukhov_length, -eps(FT)),
        max(obukhov_length, eps(FT)),
    ))
    return @. lazy(ᶜdz / obu_safe)
end

add_diagnostic_variable!(short_name = "resobu", units = "",
    long_name = "dz/Obukhov Length",
    comments = "Grid resolution normalized by Obukhov length",
    compute = compute_resobu,
)

###
# Obukhov length (broadcast to 3D for convenience)
###
compute_obukhov(state, cache, time) =
    compute_obukhov(state, cache, time, cache.atmos.turbconv_model)
compute_obukhov(_, _, _, turbconv_model) = error_diagnostic_variable("obukhov", turbconv_model)

function compute_obukhov(state, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX, EDOnlyEDMFX})
    (; obukhov_length) = cache.precomputed.sfc_conditions
    FT = eltype(state.c.ρ)
    return @. lazy(state.c.ρ * FT(0) + obukhov_length)
end

add_diagnostic_variable!(short_name = "obukhov", units = "m",
    long_name = "Obukhov Length",
    comments = "Monin-Obukhov length from surface layer (broadcast to 3D)",
    compute = compute_obukhov,
)

###############################################################################
# ENTRAINMENT/DETRAINMENT PI GROUPS (features for entr/detr models)
###############################################################################

###
# Entrainment Pi1: z * (buoy_up - buoy_env) / (w_up - w_env)^2
###
compute_entrpi1(state, cache, time) =
    compute_entrpi1(state, cache, time, cache.atmos.turbconv_model)
compute_entrpi1(_, _, _, turbconv_model) = error_diagnostic_variable("entrpi1", turbconv_model)

function compute_entrpi1(state, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX})
    (; ᶜuʲs, ᶜu⁰, ᶜρʲs) = cache.precomputed
    (; ᶜgradᵥ_ᶠΦ) = cache.core
    ᶜz = Fields.coordinate_field(state.c).z
    z_sfc = Fields.level(Fields.coordinate_field(state.f).z, Fields.half)
    ᶜlg = Fields.local_geometry_field(state.c)
    FT = eltype(ᶜz)

    ᶜwʲ = @. lazy(get_physical_w(ᶜuʲs.:1, ᶜlg))
    ᶜw⁰ = @. lazy(get_physical_w(ᶜu⁰, ᶜlg))
    ᶜbuoyʲ = @. lazy(vertical_buoyancy_acceleration(state.c.ρ, ᶜρʲs.:1, ᶜgradᵥ_ᶠΦ, ᶜlg))
    ᶜbuoy⁰ = FT(0)  # Environment buoyancy is reference (zero)

    ᶜz_above_sfc = @. lazy(ᶜz - z_sfc)
    vel_diff_sq = @. lazy((ᶜwʲ - ᶜw⁰)^2 + eps(FT))
    return @. lazy(ᶜz_above_sfc * (ᶜbuoyʲ - ᶜbuoy⁰) / vel_diff_sq)
end

add_diagnostic_variable!(short_name = "entrpi1", units = "",
    long_name = "Entrainment Pi1: z*delta_buoy/delta_w^2",
    comments = "Buoyancy-velocity ratio for entrainment",
    compute = compute_entrpi1,
)

###
# Entrainment Pi2: TKE / (w_up - w_env)^2
###
compute_entrpi2(state, cache, time) =
    compute_entrpi2(state, cache, time, cache.atmos.turbconv_model)
compute_entrpi2(_, _, _, turbconv_model) = error_diagnostic_variable("entrpi2", turbconv_model)

function compute_entrpi2(state, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX})
    (; ᶜuʲs, ᶜu⁰) = cache.precomputed
    ᶜlg = Fields.local_geometry_field(state.c)
    ᶜtke = @. lazy(specific(state.c.ρtke, state.c.ρ))
    FT = eltype(ᶜtke)

    ᶜwʲ = @. lazy(get_physical_w(ᶜuʲs.:1, ᶜlg))
    ᶜw⁰ = @. lazy(get_physical_w(ᶜu⁰, ᶜlg))
    vel_diff_sq = @. lazy((ᶜwʲ - ᶜw⁰)^2 + eps(FT))
    return @. lazy(max(ᶜtke, FT(0)) / vel_diff_sq)
end

add_diagnostic_variable!(short_name = "entrpi2", units = "",
    long_name = "Entrainment Pi2: TKE/delta_w^2",
    comments = "TKE-velocity ratio for entrainment",
    compute = compute_entrpi2,
)

###
# Entrainment Pi3: sqrt(updraft area fraction)
###
compute_entrpi3(state, cache, time) =
    compute_entrpi3(state, cache, time, cache.atmos.turbconv_model)
compute_entrpi3(_, _, _, turbconv_model) = error_diagnostic_variable("entrpi3", turbconv_model)

function compute_entrpi3(state, cache, _, ::PrognosticEDMFX)
    (; ᶜρʲs) = cache.precomputed
    ᶜρaʲ = (state.c.sgsʲs.:1).ρa
    ᶜaʲ = @. lazy(draft_area(ᶜρaʲ, ᶜρʲs.:1))
    FT = eltype(ᶜaʲ)
    return @. lazy(sqrt(max(ᶜaʲ, FT(0))))
end

function compute_entrpi3(_, cache, _, ::DiagnosticEDMFX)
    (; ᶜρaʲs, ᶜρʲs) = cache.precomputed
    ᶜaʲ = @. lazy(draft_area(ᶜρaʲs.:1, ᶜρʲs.:1))
    FT = eltype(ᶜaʲ)
    return @. lazy(sqrt(max(ᶜaʲ, FT(0))))
end

add_diagnostic_variable!(short_name = "entrpi3", units = "",
    long_name = "Entrainment Pi3: sqrt(area)",
    comments = "Square root of updraft area fraction",
    compute = compute_entrpi3,
)

###
# Entrainment Pi4: RH_up - RH_env
###
compute_entrpi4(state, cache, time) = compute_entrpi4(
    state, cache, time, cache.atmos.microphysics_model, cache.atmos.turbconv_model,
)
compute_entrpi4(_, _, _, _, _) =
    error_diagnostic_variable("Can only compute entrpi4 with a moist model and EDMFX")

function compute_entrpi4(_, cache, _, ::MoistMicrophysics, ::PrognosticEDMFX)
    thermo_params = CAP.thermodynamics_params(cache.params)
    (; ᶜTʲs, ᶜT⁰, ᶜp, ᶜq_tot_nonnegʲs, ᶜq_liqʲs, ᶜq_iceʲs) = cache.precomputed
    (; ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = cache.precomputed
    ᶜRHʲ = @. lazy(TD.relative_humidity(
        thermo_params, ᶜTʲs.:1, ᶜp, ᶜq_tot_nonnegʲs.:1, ᶜq_liqʲs.:1, ᶜq_iceʲs.:1,
    ))
    ᶜRH⁰ = @. lazy(TD.relative_humidity(
        thermo_params, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰,
    ))
    return @. lazy(ᶜRHʲ - ᶜRH⁰)
end

function compute_entrpi4(_, cache, _, ::MoistMicrophysics, ::DiagnosticEDMFX)
    thermo_params = CAP.thermodynamics_params(cache.params)
    (; ᶜTʲs, ᶜT, ᶜp, ᶜq_tot_nonnegʲs, ᶜq_liqʲs, ᶜq_iceʲs) = cache.precomputed
    (; ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = cache.precomputed
    ᶜRHʲ = @. lazy(TD.relative_humidity(
        thermo_params, ᶜTʲs.:1, ᶜp, ᶜq_tot_nonnegʲs.:1, ᶜq_liqʲs.:1, ᶜq_iceʲs.:1,
    ))
    ᶜRH⁰ = @. lazy(TD.relative_humidity(
        thermo_params, ᶜT, ᶜp, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice,
    ))
    return @. lazy(ᶜRHʲ - ᶜRH⁰)
end

add_diagnostic_variable!(short_name = "entrpi4", units = "",
    long_name = "Entrainment Pi4: RH_up - RH_env",
    comments = "Relative humidity difference between updraft and environment",
    compute = compute_entrpi4,
)

###
# Entrainment Pi5: z / pressure_scale_height
###
compute_entrpi5(state, cache, time) =
    compute_entrpi5(state, cache, time, cache.atmos.turbconv_model)
compute_entrpi5(_, _, _, turbconv_model) = error_diagnostic_variable("entrpi5", turbconv_model)

function compute_entrpi5(state, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX, EDOnlyEDMFX})
    thermo_params = CAP.thermodynamics_params(cache.params)
    (; ᶜp) = cache.precomputed
    ᶜz = Fields.coordinate_field(state.c).z
    z_sfc = Fields.level(Fields.coordinate_field(state.f).z, Fields.half)
    FT = eltype(ᶜz)
    g = TDP.grav(thermo_params)
    # Pressure scale height H = p / (ρ * g)
    ref_H = @. lazy(ᶜp / (state.c.ρ * g))
    ᶜz_above_sfc = @. lazy(ᶜz - z_sfc)
    return @. lazy(ᶜz_above_sfc / max(ref_H, eps(FT)))
end

add_diagnostic_variable!(short_name = "entrpi5", units = "",
    long_name = "Entrainment Pi5: z/H_scale",
    comments = "Height normalized by pressure scale height",
    compute = compute_entrpi5,
)

###############################################################################
# ADDITIONAL USEFUL FIELDS FOR ML TRAINING
###############################################################################

###
# Updraft - environment vertical velocity difference
###
compute_deltaw(state, cache, time) =
    compute_deltaw(state, cache, time, cache.atmos.turbconv_model)
compute_deltaw(_, _, _, turbconv_model) = error_diagnostic_variable("deltaw", turbconv_model)

function compute_deltaw(state, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX})
    (; ᶜu⁰, ᶜuʲs) = cache.precomputed
    ᶜlg = Fields.local_geometry_field(state.c)
    ᶜw_env = @. lazy(get_physical_w(ᶜu⁰, ᶜlg))
    ᶜw_up = @. lazy(get_physical_w(ᶜuʲs.:1, ᶜlg))
    return @. lazy(ᶜw_up - ᶜw_env)
end

add_diagnostic_variable!(short_name = "deltaw", units = "m s^-1",
    long_name = "Updraft-Environment Vertical Velocity Difference",
    comments = "w_updraft - w_environment",
    compute = compute_deltaw,
)

###
# Updraft buoyancy
###
compute_buoyup(state, cache, time) =
    compute_buoyup(state, cache, time, cache.atmos.turbconv_model)
compute_buoyup(_, _, _, turbconv_model) = error_diagnostic_variable("buoyup", turbconv_model)

function compute_buoyup(state, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX})
    (; ᶜρʲs) = cache.precomputed
    (; ᶜgradᵥ_ᶠΦ) = cache.core
    ᶜlg = Fields.local_geometry_field(state.c)
    return @. lazy(vertical_buoyancy_acceleration(state.c.ρ, ᶜρʲs.:1, ᶜgradᵥ_ᶠΦ, ᶜlg))
end

add_diagnostic_variable!(short_name = "buoyup", units = "m s^-2",
    long_name = "Updraft Buoyancy Acceleration",
    comments = "Vertical buoyancy acceleration of the first updraft",
    compute = compute_buoyup,
)

###
# Gradient Richardson number
###
compute_rigrad(state, cache, time) =
    compute_rigrad(state, cache, time, cache.atmos.turbconv_model)
compute_rigrad(_, _, _, turbconv_model) = error_diagnostic_variable("rigrad", turbconv_model)

function compute_rigrad(_, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX, EDOnlyEDMFX})
    (; params) = cache
    (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = cache.precomputed
    return @. lazy(gradient_richardson_number(params, ᶜlinear_buoygrad, ᶜstrain_rate_norm))
end

add_diagnostic_variable!(short_name = "rigrad", units = "",
    long_name = "Gradient Richardson Number",
    comments = "Ratio of buoyancy gradient to shear squared",
    compute = compute_rigrad,
)

###
# Turbulent Prandtl number
###
compute_prandtl(state, cache, time) =
    compute_prandtl(state, cache, time, cache.atmos.turbconv_model)
compute_prandtl(_, _, _, turbconv_model) = error_diagnostic_variable("prandtl", turbconv_model)

function compute_prandtl(_, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX, EDOnlyEDMFX})
    (; params) = cache
    (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = cache.precomputed
    return @. lazy(turbulent_prandtl_number(params, ᶜlinear_buoygrad, ᶜstrain_rate_norm))
end

add_diagnostic_variable!(short_name = "prandtl", units = "",
    long_name = "Turbulent Prandtl Number",
    comments = "Ratio of momentum diffusivity to heat diffusivity",
    compute = compute_prandtl,
)
