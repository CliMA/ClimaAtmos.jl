# This file is included in Diagnostics.jl

# EDMFX diagnostics

###
# Updraft area fraction (3d)
###
compute_arup(state, cache, time) =
    compute_arup(state, cache, time, cache.atmos.turbconv_model)
compute_arup(_, _, _, turbconv_model) = error_diagnostic_variable("arup", turbconv_model)

function compute_arup(state, cache, _, turbconv_model::PrognosticEDMFX)
    ρaʲ = (state.c.sgsʲs.:1).ρa
    thermo_params = CAP.thermodynamics_params(cache.params)
    ρʲ = @. lazy(TD.air_density(thermo_params, cache.precomputed.ᶜtsʲs.:1))
    return @. lazy(draft_area(ρaʲ, ρʲ))
end
function compute_arup(_, cache, _, turbconv_model::DiagnosticEDMFX)
    ρaʲ = cache.precomputed.ᶜρaʲs.:1
    thermo_params = CAP.thermodynamics_params(cache.params)
    ρʲ = @. lazy(TD.air_density(thermo_params, cache.precomputed.ᶜtsʲs.:1))
    return @. lazy(draft_area(ρaʲ, ρʲ))
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

function compute_rhoaup(_, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX})
    thermo_params = CAP.thermodynamics_params(cache.params)
    return @. lazy(TD.air_density(thermo_params, cache.precomputed.ᶜtsʲs.:1))
end

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
    @. lazy(w_component(Geometry.WVector(cache.precomputed.ᶜuʲs.:1)))

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

function compute_taup(_, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX})
    thermo_params = CAP.thermodynamics_params(cache.params)
    return @. lazy(TD.air_temperature(thermo_params, cache.precomputed.ᶜtsʲs.:1))
end

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
    return @. lazy(TD.dry_pottemp(thermo_params, cache.precomputed.ᶜtsʲs.:1))
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
    return @. lazy(TD.specific_enthalpy(thermo_params, cache.precomputed.ᶜtsʲs.:1))
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
    state, cache, time, cache.atmos.moisture_model, cache.atmos.turbconv_model,
)
compute_husup(_, _, _, moisture_model, turbconv_model) =
    error_diagnostic_variable(
        "Can only compute updraft specific humidity with a moist model and with EDMFX",
    )

function compute_husup(_, cache, _,
    ::Union{EquilMoistModel, NonEquilMoistModel}, ::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    return @. lazy(TD.total_specific_humidity(thermo_params, cache.precomputed.ᶜtsʲs.:1))
end

add_diagnostic_variable!(short_name = "husup", units = "kg kg^-1",
    long_name = "Updraft Specific Humidity",
    comments = "Specific humidity of the first updraft",
    compute = compute_husup,
)

###
# Updraft relative humidity (3d)
###
compute_hurup(state, cache, time) = compute_hurup(
    state, cache, time, cache.atmos.moisture_model, cache.atmos.turbconv_model,
)
compute_hurup(_, _, _, moisture_model, turbconv_model) =
    error_diagnostic_variable(
        "Can only compute updraft relative humidity and with a moist model and with EDMFX",
    )

function compute_hurup(_, cache, _,
    ::Union{EquilMoistModel, NonEquilMoistModel}, ::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    return @. lazy(TD.relative_humidity(thermo_params, cache.precomputed.ᶜtsʲs.:1))
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
    state, cache, time, cache.atmos.moisture_model, cache.atmos.turbconv_model,
)
compute_clwup(_, _, _, moisture_model, turbconv_model) =
    error_diagnostic_variable(
        "Can only compute updraft liquid water specific humidity and with a moist model and with EDMFX",
    )

function compute_clwup(_, cache, _,
    ::EquilMoistModel, ::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    return @. lazy(TD.liquid_specific_humidity(thermo_params, cache.precomputed.ᶜtsʲs.:1))
end
compute_clwup(state, _, _, ::NonEquilMoistModel, ::PrognosticEDMFX) =
    (state.c.sgsʲs.:1).q_liq
compute_clwup(_, cache, _, ::NonEquilMoistModel, ::DiagnosticEDMFX) =
    cache.precomputed.ᶜq_liqʲs.:1

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
compute_cdncup(_, _, _, microphysics_model, turbconv_model) =
    error_diagnostic_variable(
        "Can only compute updraft rain number mixing ratio with a 2M precip model and with EDMFX",
    )

compute_cdncup(state, _, _, ::Microphysics2Moment, ::PrognosticEDMFX) =
    (state.c.sgsʲs.:1).n_liq

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
    cache.atmos.moisture_model, cache.atmos.turbconv_model,
)
compute_cliup(_, _, _, moisture_model, turbconv_model) =
    error_diagnostic_variable(
        "Can only compute updraft ice water specific humidity and with a moist model and with EDMFX",
    )

function compute_cliup(_, cache, _,
    ::EquilMoistModel, ::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    return @. lazy(TD.ice_specific_humidity(thermo_params, cache.precomputed.ᶜtsʲs.:1))
end
compute_cliup(state, _, _, ::NonEquilMoistModel, ::PrognosticEDMFX) =
    (state.c.sgsʲs.:1).q_ice
compute_cliup(_, cache, _, ::NonEquilMoistModel, ::DiagnosticEDMFX) =
    cache.precomputed.ᶜq_iceʲs.:1

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
compute_husraup(_, _, _, microphysics_model, turbconv_model) =
    error_diagnostic_variable(
        "Can only compute updraft rain water specific humidity with a 1M or 2M precip model and with EDMFX",
    )

compute_husraup(state, _, _,
    ::Union{Microphysics1Moment, Microphysics2Moment}, ::PrognosticEDMFX,
) = (state.c.sgsʲs.:1).q_rai
compute_husraup(_, cache, _, ::Microphysics1Moment, ::DiagnosticEDMFX) =
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
compute_ncraup(_, _, _, microphysics_model, turbconv_model) =
    error_diagnostic_variable(
        "Can only compute updraft rain number mixing ratio with a 2M precip model and with EDMFX",
    )

compute_ncraup(state, _, _, ::Microphysics2Moment, ::PrognosticEDMFX) =
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
compute_hussnup(_, _, _, microphysics_model, turbconv_model) =
    error_diagnostic_variable(
        "Can only compute updraft snow specific humidity with a 1M or 2M precip model and with EDMFX",
    )

compute_hussnup(state, _, _,
    ::Union{Microphysics1Moment, Microphysics2Moment}, ::PrognosticEDMFX,
) = (state.c.sgsʲs.:1).q_sno
compute_hussnup(_, cache, _, ::Microphysics1Moment, ::DiagnosticEDMFX) =
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

compute_entr(_, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX}) =
    cache.precomputed.ᶜentrʲs.:1

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

compute_detr(_, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX}) =
    cache.precomputed.ᶜdetrʲs.:1

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
    ᶜρ⁰ = @. lazy(TD.air_density(thermo_params, cache.precomputed.ᶜts⁰))
    return @. lazy(draft_area(ᶜρa⁰, ᶜρ⁰))
end

function compute_aren(state, cache, _, turbconv_model::DiagnosticEDMFX)
    thermo_params = CAP.thermodynamics_params(cache.params)
    ᶜρaʲ = cache.precomputed.ᶜρaʲs.:1
    ᶜρʲ = @. lazy(TD.air_density(thermo_params, cache.precomputed.ᶜts⁰))
    return @. lazy(1 - draft_area(ᶜρaʲ, ᶜρʲ))
end

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
    return @. lazy(TD.air_density(thermo_params, cache.precomputed.ᶜts⁰))  # ᶜρ⁰
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
    @. lazy(w_component(Geometry.WVector(cache.precomputed.ᶜu⁰)))

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

function compute_taen(_, cache, _, ::PrognosticEDMFX)
    thermo_params = CAP.thermodynamics_params(cache.params)
    return @. lazy(TD.air_temperature(thermo_params, cache.precomputed.ᶜts⁰))
end

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
    return @. lazy(TD.dry_pottemp(thermo_params, cache.precomputed.ᶜts⁰))
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
    return @. lazy(TD.dry_pottemp(thermo_params, cache.precomputed.ᶜts⁰))
end

add_diagnostic_variable!(short_name = "haen", units = "K",
    long_name = "Environment Air Specific Enthalpy",
    compute = compute_haen,
)

###
# Environment total specific humidity (3d)
###
compute_husen(state, cache, time) = compute_husen(state, cache, time,
    cache.atmos.moisture_model, cache.atmos.turbconv_model,
)
compute_husen(_, _, _, moisture_model, turbconv_model) =
    error_diagnostic_variable(
        "Can only compute updraft specific humidity and with a moist model and with EDMFX",
    )

function compute_husen(_, cache, _,
    ::Union{EquilMoistModel, NonEquilMoistModel}, ::PrognosticEDMFX,
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    return @. lazy(TD.total_specific_humidity(thermo_params, cache.precomputed.ᶜts⁰))
end

add_diagnostic_variable!(short_name = "husen", units = "kg kg^-1",
    long_name = "Environment Specific Humidity",
    compute = compute_husen,
)

###
# Environment relative humidity (3d)
###
compute_huren(state, cache, time) = compute_huren(state, cache, time,
    cache.atmos.moisture_model, cache.atmos.turbconv_model,
)
compute_huren(_, _, _, moisture_model, turbconv_model) =
    error_diagnostic_variable(
        "Can only compute updraft relative humidity and with a moist model and with EDMFX",
    )

function compute_huren(_, cache, _,
    ::Union{EquilMoistModel, NonEquilMoistModel}, ::PrognosticEDMFX,
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    return @. lazy(TD.relative_humidity(thermo_params, cache.precomputed.ᶜts⁰))
end

add_diagnostic_variable!(short_name = "huren", units = "",
    long_name = "Environment Relative Humidity",
    compute = compute_huren,
)

###
# Environment liquid water specific humidity and number mixing ratio (3d)
###
compute_clwen(state, cache, time) = compute_clwen(state, cache, time,
    cache.atmos.moisture_model, cache.atmos.turbconv_model,
)
compute_clwen(_, _, _, moisture_model, turbconv_model) =
    error_diagnostic_variable(
        "Can only compute updraft liquid water specific humidity and with a moist model and with EDMFX",
    )

function compute_clwen(_, cache, _, ::EquilMoistModel, ::PrognosticEDMFX)
    thermo_params = CAP.thermodynamics_params(cache.params)
    return @. lazy(TD.liquid_specific_humidity(thermo_params, cache.precomputed.ᶜts⁰))
end

function compute_clwen(state, cache, _, ::NonEquilMoistModel, ::PrognosticEDMFX)
    return ᶜspecific_env_value(@name(q_liq), state, cache)
end

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
compute_cdncen(_, _, _, microphysics_model, turbconv_model) =
    error_diagnostic_variable(
        "Can only compute updraft cloud liquid water number mixing ratio with a 2M model and with EDMFX",
    )

compute_cdncen(state, cache, _, ::Microphysics2Moment, ::PrognosticEDMFX) =
    ᶜspecific_env_value(@name(n_liq), state, cache)

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
    state, cache, time, cache.atmos.moisture_model, cache.atmos.turbconv_model,
)
compute_clien(_, _, _, moisture_model, turbconv_model) =
    error_diagnostic_variable(
        "Can only compute updraft ice water specific humidity and with a moist model and with EDMFX",
    )

function compute_clien(_, cache, _, ::EquilMoistModel, ::PrognosticEDMFX)
    thermo_params = CAP.thermodynamics_params(cache.params)
    return @. lazy(TD.ice_specific_humidity.(thermo_params, cache.precomputed.ᶜts⁰))
end

compute_clien(state, cache, _, ::NonEquilMoistModel, ::PrognosticEDMFX) =
    ᶜspecific_env_value(@name(q_ice), state, cache)

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
compute_husraen(_, _, _, microphysics_model, turbconv_model) =
    error_diagnostic_variable(
        "Can only compute updraft rain specific humidity with a 1M or 2M model and with EDMFX",
    )

compute_husraen(state, cache, _,
    ::Union{Microphysics1Moment, Microphysics2Moment}, ::PrognosticEDMFX,
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
compute_ncraen(_, _, _, microphysics_model, turbconv_model) =
    error_diagnostic_variable(
        "Can only compute updraft rain number mixing ratio with a 2M model and with EDMFX",
    )

compute_ncraen(state, cache, _, ::Microphysics2Moment, ::PrognosticEDMFX) =
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
compute_hussnen(_, _, _, microphysics_model, turbconv_model) =
    error_diagnostic_variable(
        "Can only compute updraft snow specific humidity with a 1M or 2M model and with EDMFX",
    )

compute_hussnen(state, cache, _,
    ::Union{Microphysics1Moment, Microphysics2Moment}, ::PrognosticEDMFX,
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
# Environment turbulent kinetic energy (3d)
###
compute_tke(state, cache, time) =
    compute_tke(state, cache, time, cache.atmos.turbconv_model)
compute_tke(_, _, _, turbconv_model) = error_diagnostic_variable("tke", turbconv_model)

function compute_tke(state, _, _,
    turbconv_model::Union{EDOnlyEDMFX, PrognosticEDMFX, DiagnosticEDMFX},
)
    ᶜρ = state.c.ρ
    ᶜsgs⁰ = state.c.sgs⁰
    # ᶜρa⁰_pedmfx = @. lazy(ρa⁰(ρ, state.c.sgsʲs, turbconv_model))
    # ᶜρa⁰ = turbconv_model isa PrognosticEDMFX ? ᶜρa⁰_pedmfx : ρ
    ᶜtke⁰ = @. lazy(specific(ᶜsgs⁰.ρatke, ᶜρ))
    return ᶜtke⁰
end

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
compute_edt(_, _, _, vertical_diffusion, turbconv_model) =
    error_diagnostic_variable(
        "Can only compute heat diffusivity with vertical diffusion or EDMFX",
    )

compute_edt(state, cache, _, model::VerticalDiffusion, ::Nothing) =
    ᶜcompute_eddy_diffusivity_coefficient(state.c.uₕ, cache.precomputed.ᶜp, model)
compute_edt(state, _, _, model::DecayWithHeightDiffusion, ::Nothing) =
    ᶜcompute_eddy_diffusivity_coefficient(state.c.ρ, model)

function compute_edt(state, cache, _,
    vertdiff::Nothing, turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    turbconv_params = CAP.turbconv_params(cache.params)
    (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = cache.precomputed
    ᶜtke⁰ = @. lazy(specific(state.c.sgs⁰.ρatke, state.c.ρ))
    ᶜmixing_length_field = ᶜmixing_length(state, cache)
    ᶜK_u = @. lazy(eddy_viscosity(turbconv_params, ᶜtke⁰, ᶜmixing_length_field))
    ᶜprandtl_nvec =
        @. lazy(turbulent_prandtl_number(cache.params, ᶜlinear_buoygrad, ᶜstrain_rate_norm))
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
compute_evu(_, _, _, vertical_diffusion, turbconv_model) =
    error_diagnostic_variable(
        "Can only compute momentum diffusivity with vertical diffusion or EDMFX",
    )

compute_evu(state, cache, _, model::VerticalDiffusion, ::Nothing) =
    ᶜcompute_eddy_diffusivity_coefficient(state.c.uₕ, cache.precomputed.ᶜp, model)
compute_evu(state, _, _, model::DecayWithHeightDiffusion, ::Nothing) =
    ᶜcompute_eddy_diffusivity_coefficient(state.c.ρ, model)

function compute_evu(state, cache, _,
    ::Nothing, turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    turbconv_params = CAP.turbconv_params(cache.params)
    ᶜtke⁰ = @. lazy(specific(state.c.sgs⁰.ρatke, state.c.ρ))
    ᶜmixing_length_field = ᶜmixing_length(state, cache)
    ᶜK_u = @. lazy(eddy_viscosity(turbconv_params, ᶜtke⁰, ᶜmixing_length_field))
    return ᶜK_u
end

add_diagnostic_variable!(short_name = "evu", units = "m^2 s^-1",
    long_name = "Eddy Viscosity Coefficient for Momentum",
    standard_name = "atmosphere_momentum_diffusivity",
    comments = "Vertical diffusion coefficient for momentum due to parameterized eddies",
    compute = compute_evu,
)
