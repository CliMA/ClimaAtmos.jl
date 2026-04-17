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
    error_diagnostic_variable("Can only compute updraft specific humidity \
                               with a moist model and with EDMFX")

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
    error_diagnostic_variable("Can only compute updraft relative humidity \
                               with a moist model and with EDMFX")

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
    error_diagnostic_variable("Can only compute updraft liquid water specific humidity \
                               with a moist model and with EDMFX")

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
    error_diagnostic_variable("Can only compute updraft cloud liquid water \
                               number mixing ratio with a 2M model and with EDMFX")

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
    error_diagnostic_variable("Can only compute updraft ice water specific humidity \
                               with a moist model and with EDMFX")

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
    error_diagnostic_variable("Can only compute updraft rain specific humidity \
                               with a 1M or 2M model and with EDMFX")

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
    error_diagnostic_variable("Can only compute updraft rain number mixing ratio \
                               with a 2M model and with EDMFX")

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
    error_diagnostic_variable("Can only compute updraft snow specific humidity \
                               with a 1M or 2M model and with EDMFX")

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

###
# Updraft mass flux (3d)
###
compute_mfup(state, cache, time) =
    compute_mfup(state, cache, time, cache.atmos.turbconv_model)
compute_mfup(_, _, _, turbconv_model) =
    error_diagnostic_variable("mfup", turbconv_model)

function compute_mfup(state, cache, _, ::PrognosticEDMFX)
    (; ᶠu³ʲs, ᶜρʲs) = cache.precomputed
    ᶜρaʲ = (state.c.sgsʲs.:1).ρa
    ᶜJ = Fields.local_geometry_field(state.c).J
    ᶠJ = Fields.local_geometry_field(cache.precomputed.ᶠu³).J
    ᶠflux = @. lazy(ᶠinterp(ᶜρaʲ * ᶜJ) / ᶠJ * ᶠu³ʲs.:1)
    return @. lazy(ᶜinterp(w_component(Geometry.WVector(ᶠflux))))
end

function compute_mfup(state, cache, _, ::DiagnosticEDMFX)
    (; ᶜρaʲs, ᶠu³ʲs) = cache.precomputed
    ᶜJ = Fields.local_geometry_field(state.c).J
    ᶠJ = Fields.local_geometry_field(cache.precomputed.ᶠu³).J
    ᶠflux = @. lazy(ᶠinterp(ᶜρaʲs.:1 * ᶜJ) / ᶠJ * ᶠu³ʲs.:1)
    return @. lazy(ᶜinterp(w_component(Geometry.WVector(ᶠflux))))
end

add_diagnostic_variable!(short_name = "mfup", units = "kg m^-2 s^-1",
    long_name = "Updraft Mass Flux",
    comments = """
    Vertical mass flux of the first updraft: ρa * w.
    Represents the mass transport by the updraft per unit area.
    Computed on faces with Jacobian terms for metric consistency, then interpolated to cell centers.
    """,
    compute = compute_mfup,
)

###
# Total SGS mass flux (3d)
###
compute_sgsmf(state, cache, time) =
    compute_sgsmf(state, cache, time, cache.atmos.turbconv_model)
compute_sgsmf(_, _, _, turbconv_model) =
    error_diagnostic_variable("sgsmf", turbconv_model)

function compute_sgsmf(state, cache, _, turbconv_model::PrognosticEDMFX)
    thermo_params = CAP.thermodynamics_params(cache.params)
    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶠu³) = cache.precomputed
    (; ᶠu³ʲs, ᶜρʲs) = cache.precomputed
    (; ᶠu³⁰, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜp) = cache.precomputed

    ᶜJ = Fields.local_geometry_field(state.c).J
    ᶠJ = Fields.local_geometry_field(ᶠu³).J

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
    ᶜρa⁰ = @. lazy(ρa⁰(state.c.ρ, state.c.sgsʲs, turbconv_model))

    # Updraft 1 contribution (assuming n=1 for simplicity, can be extended)
    ᶜρaʲ = (state.c.sgsʲs.:1).ρa
    ᶠu³_diffʲ = @. lazy(ᶠu³ʲs.:1 - ᶠu³)
    ᶠfluxʲ = @. lazy(ᶠinterp(ᶜρaʲ * ᶜJ) / ᶠJ * ᶠu³_diffʲ)

    # Environment contribution
    ᶠu³_diff⁰ = @. lazy(ᶠu³⁰ - ᶠu³)
    ᶠflux⁰ = @. lazy(ᶠinterp(ᶜρa⁰ * ᶜJ) / ᶠJ * ᶠu³_diff⁰)

    # Total flux
    ᶠflux_total = @. lazy(ᶠfluxʲ + ᶠflux⁰)
    return @. lazy(ᶜinterp(w_component(Geometry.WVector(ᶠflux_total))))
end

function compute_sgsmf(state, cache, _, turbconv_model::DiagnosticEDMFX)
    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶠu³) = cache.precomputed
    (; ᶜρaʲs, ᶠu³ʲs) = cache.precomputed

    ᶜJ = Fields.local_geometry_field(state.c).J
    ᶠJ = Fields.local_geometry_field(ᶠu³).J

    # Updraft 1 contribution (DiagnosticEDMFX doesn't include environment in mass flux)
    ᶠu³_diffʲ = @. lazy(ᶠu³ʲs.:1 - ᶠu³)
    ᶠflux = @. lazy(ᶠinterp(ᶜρaʲs.:1 * ᶜJ) / ᶠJ * ᶠu³_diffʲ)
    return @. lazy(ᶜinterp(w_component(Geometry.WVector(ᶠflux))))
end

add_diagnostic_variable!(short_name = "sgsmf", units = "kg m^-2 s^-1",
    long_name = "SGS Mass Flux",
    comments = """
    Total subgrid-scale vertical mass flux from EDMFX updrafts and environment.
    Computed as the sum of ρa * (w - w̄) over all subdomains, where w̄ is the grid-mean velocity.
    Uses Jacobian terms for metric consistency, matching the actual tendency computation.
    """,
    compute = compute_sgsmf,
)

###
# SGS mass flux of total enthalpy (3d)
###
compute_sgsmfht(state, cache, time) =
    compute_sgsmfht(state, cache, time, cache.atmos.turbconv_model)
compute_sgsmfht(_, _, _, turbconv_model) =
    error_diagnostic_variable("sgsmfht", turbconv_model)

function compute_sgsmfht(state, cache, _, turbconv_model::PrognosticEDMFX)
    thermo_params = CAP.thermodynamics_params(cache.params)
    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶠu³) = cache.precomputed
    (; ᶠu³ʲs, ᶜKʲs, ᶜρʲs) = cache.precomputed
    (; ᶠu³⁰, ᶜK⁰, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜp) = cache.precomputed
    (; ᶜh_tot) = cache.precomputed

    ᶜJ = Fields.local_geometry_field(state.c).J
    ᶠJ = Fields.local_geometry_field(ᶠu³).J

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
    ᶜρa⁰ = @. lazy(ρa⁰(state.c.ρ, state.c.sgsʲs, turbconv_model))

    # Updraft 1 contribution
    ᶜρaʲ = (state.c.sgsʲs.:1).ρa
    ᶜρʲ = ᶜρʲs.:1
    ᶜmseʲ = (state.c.sgsʲs.:1).mse
    ᶜh_devʲ = @. lazy((ᶜmseʲ + ᶜKʲs.:1 - ᶜh_tot) * draft_area(ᶜρaʲ, ᶜρʲ))
    ᶠu³_diffʲ = @. lazy(ᶠu³ʲs.:1 - ᶠu³)
    ᶠfluxʲ = @. lazy(ᶠinterp(ᶜρʲ * ᶜJ) / ᶠJ * ᶠu³_diffʲ * ᶠinterp(ᶜh_devʲ))

    # Environment contribution
    ᶜmse⁰ = ᶜspecific_env_mse(state, cache)
    ᶜh_dev⁰ = @. lazy((ᶜmse⁰ + ᶜK⁰ - ᶜh_tot) * draft_area(ᶜρa⁰, ᶜρ⁰))
    ᶠu³_diff⁰ = @. lazy(ᶠu³⁰ - ᶠu³)
    ᶠflux⁰ = @. lazy(ᶠinterp(ᶜρ⁰ * ᶜJ) / ᶠJ * ᶠu³_diff⁰ * ᶠinterp(ᶜh_dev⁰))

    # Total flux
    ᶠflux_total = @. lazy(ᶠfluxʲ + ᶠflux⁰)
    return @. lazy(ᶜinterp(w_component(Geometry.WVector(ᶠflux_total))))
end

function compute_sgsmfht(state, cache, _, turbconv_model::DiagnosticEDMFX)
    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶠu³) = cache.precomputed
    (; ᶜρaʲs, ᶜρʲs, ᶠu³ʲs, ᶜKʲs, ᶜmseʲs) = cache.precomputed
    (; ᶜh_tot) = cache.precomputed

    ᶜJ = Fields.local_geometry_field(state.c).J
    ᶠJ = Fields.local_geometry_field(ᶠu³).J

    # Updraft 1 contribution
    ᶜh_devʲ = @. lazy((ᶜmseʲs.:1 + ᶜKʲs.:1 - ᶜh_tot) * draft_area(ᶜρaʲs.:1, ᶜρʲs.:1))
    ᶠu³_diffʲ = @. lazy(ᶠu³ʲs.:1 - ᶠu³)
    ᶠflux = @. lazy(ᶠinterp(ᶜρʲs.:1 * ᶜJ) / ᶠJ * ᶠu³_diffʲ * ᶠinterp(ᶜh_devʲ))
    return @. lazy(ᶜinterp(w_component(Geometry.WVector(ᶠflux))))
end

add_diagnostic_variable!(short_name = "sgsmfht", units = "W m^-2",
    long_name = "SGS Mass Flux of Total Enthalpy",
    comments = """
    Subgrid-scale vertical mass flux of total enthalpy from EDMFX updrafts and environment.
    Represents transport of energy by resolved SGS circulations relative to the grid mean flow.
    Computed on faces with Jacobian terms for metric consistency, matching the actual tendency computation.
    """,
    compute = compute_sgsmfht,
)

###
# SGS mass flux of total moisture (3d)
###
compute_sgsmfqt(state, cache, time) = compute_sgsmfqt(
    state, cache, time, cache.atmos.microphysics_model, cache.atmos.turbconv_model,
)
compute_sgsmfqt(_, _, _, _, _) =
    error_diagnostic_variable("Can only compute SGS mass flux of moisture \
                               with a moist model and with EDMFX")

function compute_sgsmfqt(state, cache, _,
    ::MoistMicrophysics, turbconv_model::PrognosticEDMFX,
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶠu³) = cache.precomputed
    (; ᶠu³ʲs, ᶜρʲs) = cache.precomputed
    (; ᶠu³⁰, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜp) = cache.precomputed

    ᶜJ = Fields.local_geometry_field(state.c).J
    ᶠJ = Fields.local_geometry_field(ᶠu³).J

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
    ᶜρa⁰ = @. lazy(ρa⁰(state.c.ρ, state.c.sgsʲs, turbconv_model))
    ᶜq_tot = @. lazy(specific(state.c.ρq_tot, state.c.ρ))

    # Updraft 1 contribution
    ᶜρaʲ = (state.c.sgsʲs.:1).ρa
    ᶜρʲ = ᶜρʲs.:1
    ᶜq_totʲ = (state.c.sgsʲs.:1).q_tot
    ᶜq_devʲ = @. lazy((ᶜq_totʲ - ᶜq_tot) * draft_area(ᶜρaʲ, ᶜρʲ))
    ᶠu³_diffʲ = @. lazy(ᶠu³ʲs.:1 - ᶠu³)
    ᶠfluxʲ = @. lazy(ᶠinterp(ᶜρʲ * ᶜJ) / ᶠJ * ᶠu³_diffʲ * ᶠinterp(ᶜq_devʲ))

    # Environment contribution
    ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), state, cache)
    ᶜq_dev⁰ = @. lazy((ᶜq_tot⁰ - ᶜq_tot) * draft_area(ᶜρa⁰, ᶜρ⁰))
    ᶠu³_diff⁰ = @. lazy(ᶠu³⁰ - ᶠu³)
    ᶠflux⁰ = @. lazy(ᶠinterp(ᶜρ⁰ * ᶜJ) / ᶠJ * ᶠu³_diff⁰ * ᶠinterp(ᶜq_dev⁰))

    # Total flux
    ᶠflux_total = @. lazy(ᶠfluxʲ + ᶠflux⁰)
    return @. lazy(ᶜinterp(w_component(Geometry.WVector(ᶠflux_total))))
end

function compute_sgsmfqt(state, cache, _,
    ::MoistMicrophysics, turbconv_model::DiagnosticEDMFX,
)
    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶠu³) = cache.precomputed
    (; ᶜρaʲs, ᶜρʲs, ᶠu³ʲs, ᶜq_totʲs) = cache.precomputed

    ᶜJ = Fields.local_geometry_field(state.c).J
    ᶠJ = Fields.local_geometry_field(ᶠu³).J
    ᶜq_tot = @. lazy(specific(state.c.ρq_tot, state.c.ρ))

    # Updraft 1 contribution
    ᶜq_devʲ = @. lazy((ᶜq_totʲs.:1 - ᶜq_tot) * draft_area(ᶜρaʲs.:1, ᶜρʲs.:1))
    ᶠu³_diffʲ = @. lazy(ᶠu³ʲs.:1 - ᶠu³)
    ᶠflux = @. lazy(ᶠinterp(ᶜρʲs.:1 * ᶜJ) / ᶠJ * ᶠu³_diffʲ * ᶠinterp(ᶜq_devʲ))
    return @. lazy(ᶜinterp(w_component(Geometry.WVector(ᶠflux))))
end

add_diagnostic_variable!(short_name = "sgsmfqt", units = "kg m^-2 s^-1",
    long_name = "SGS Mass Flux of Total Moisture",
    comments = """
    Subgrid-scale vertical mass flux of total specific humidity from EDMFX updrafts and environment.
    Represents transport of moisture by resolved SGS circulations relative to the grid mean flow.
    Computed on faces with Jacobian terms for metric consistency, matching the actual tendency computation.
    """,
    compute = compute_sgsmfqt,
)

###
# SGS diffusive flux of total enthalpy (3d)
###
compute_sgsdfht(state, cache, time) =
    compute_sgsdfht(state, cache, time, cache.atmos.turbconv_model)
compute_sgsdfht(_, _, _, turbconv_model) =
    error_diagnostic_variable("sgsdfht", turbconv_model)

function compute_sgsdfht(state, cache, _, ::Union{PrognosticEDMFX, DiagnosticEDMFX, EDOnlyEDMFX})
    (; params) = cache
    turbconv_params = CAP.turbconv_params(params)
    (; ᶜlinear_buoygrad, ᶜstrain_rate_norm, ᶜh_tot) = cache.precomputed
    ᶜtke = @. lazy(specific(state.c.ρtke, state.c.ρ))
    ᶜmixing_length_field = ᶜmixing_length(state, cache)
    ᶜK_u = @. lazy(eddy_viscosity(turbconv_params, ᶜtke, ᶜmixing_length_field))
    ᶜprandtl_nvec = @. lazy(
        turbulent_prandtl_number(params, ᶜlinear_buoygrad, ᶜstrain_rate_norm),
    )
    ᶜK_h = @. lazy(eddy_diffusivity(ᶜK_u, ᶜprandtl_nvec))

    # Face-centered diffusive flux: -ρ K_h ∂h_tot/∂z
    # Matches actual tendency: ᶠρaK_h = ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜK_h)
    ᶠρK_h = @. lazy(ᶠinterp(state.c.ρ) * ᶠinterp(ᶜK_h))
    ᶠflux = @. lazy(-(ᶠρK_h * ᶠgradᵥ(ᶜh_tot)))
    return @. lazy(ᶜinterp(w_component(Geometry.WVector(ᶠflux))))
end

add_diagnostic_variable!(short_name = "sgsdfht", units = "W m^-2",
    long_name = "SGS Diffusive Flux of Total Enthalpy",
    comments = """
    Subgrid-scale vertical diffusive flux of total enthalpy from EDMFX environment turbulence.
    Parameterized using eddy diffusivity closure: -ρ K_h ∂h_tot/∂z
    Computed on faces then interpolated to cell centers, matching the actual tendency computation.
    """,
    compute = compute_sgsdfht,
)

###
# SGS diffusive flux of total moisture (3d)
###
compute_sgsdfqt(state, cache, time) = compute_sgsdfqt(
    state, cache, time, cache.atmos.microphysics_model, cache.atmos.turbconv_model,
)
compute_sgsdfqt(_, _, _, _, _) =
    error_diagnostic_variable("Can only compute SGS diffusive flux of moisture \
                               with a moist model and with EDMFX")

function compute_sgsdfqt(state, cache, _,
    ::MoistMicrophysics, ::Union{PrognosticEDMFX, DiagnosticEDMFX, EDOnlyEDMFX},
)
    (; params) = cache
    turbconv_params = CAP.turbconv_params(params)
    (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = cache.precomputed
    ᶜtke = @. lazy(specific(state.c.ρtke, state.c.ρ))
    ᶜmixing_length_field = ᶜmixing_length(state, cache)
    ᶜK_u = @. lazy(eddy_viscosity(turbconv_params, ᶜtke, ᶜmixing_length_field))
    ᶜprandtl_nvec = @. lazy(
        turbulent_prandtl_number(params, ᶜlinear_buoygrad, ᶜstrain_rate_norm),
    )
    ᶜK_h = @. lazy(eddy_diffusivity(ᶜK_u, ᶜprandtl_nvec))

    # Face-centered diffusive flux: -ρ K_h ∂q_tot/∂z
    ᶜq_tot = @. lazy(specific(state.c.ρq_tot, state.c.ρ))
    ᶠρK_h = @. lazy(ᶠinterp(state.c.ρ) * ᶠinterp(ᶜK_h))
    ᶠflux = @. lazy(-(ᶠρK_h * ᶠgradᵥ(ᶜq_tot)))
    return @. lazy(ᶜinterp(w_component(Geometry.WVector(ᶠflux))))
end

add_diagnostic_variable!(short_name = "sgsdfqt", units = "kg m^-2 s^-1",
    long_name = "SGS Diffusive Flux of Total Moisture",
    comments = """
    Subgrid-scale vertical diffusive flux of total specific humidity from EDMFX environment turbulence.
    Parameterized using eddy diffusivity closure: -ρ K_h ∂q_tot/∂z
    Computed on faces then interpolated to cell centers, matching the actual tendency computation.
    """,
    compute = compute_sgsdfqt,
)
