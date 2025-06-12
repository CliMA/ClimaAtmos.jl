# This file is included in Diagnostics.jl

# EDMFX diagnostics

###
# Updraft area fraction (3d)
###
compute_arup!(out, state, cache, time) =
    compute_arup!(out, state, cache, time, cache.atmos.turbconv_model)
compute_arup!(_, _, _, _, turbconv_model::T) where {T} =
    error_diagnostic_variable("arup", turbconv_model)

function compute_arup!(out, state, cache, time, turbconv_model::PrognosticEDMFX)
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return draft_area.(
            (state.c.sgsʲs.:1).ρa,
            TD.air_density.(thermo_params, cache.precomputed.ᶜtsʲs.:1),
        )
    else
        out .=
            draft_area.(
                (state.c.sgsʲs.:1).ρa,
                TD.air_density.(thermo_params, cache.precomputed.ᶜtsʲs.:1),
            )
    end
end

function compute_arup!(out, state, cache, time, turbconv_model::DiagnosticEDMFX)
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return draft_area.(
            cache.precomputed.ᶜρaʲs.:1,
            TD.air_density.(thermo_params, cache.precomputed.ᶜtsʲs.:1),
        )
    else
        out .=
            draft_area.(
                cache.precomputed.ᶜρaʲs.:1,
                TD.air_density.(thermo_params, cache.precomputed.ᶜtsʲs.:1),
            )
    end
end

add_diagnostic_variable!(
    short_name = "arup",
    long_name = "Updraft Area Fraction",
    units = "",
    comments = "Area fraction of the first updraft",
    compute! = compute_arup!,
)

###
# Updraft density (3d)
###
compute_rhoaup!(out, state, cache, time) =
    compute_rhoaup!(out, state, cache, time, cache.atmos.turbconv_model)
compute_rhoaup!(_, _, _, _, turbconv_model::T) where {T} =
    error_diagnostic_variable("rhoaup", turbconv_model)

function compute_rhoaup!(
    out,
    state,
    cache,
    time,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return TD.air_density.(thermo_params, cache.precomputed.ᶜtsʲs.:1)
    else
        out .= TD.air_density.(thermo_params, cache.precomputed.ᶜtsʲs.:1)
    end
end

add_diagnostic_variable!(
    short_name = "rhoaup",
    long_name = "Updraft Air Density",
    units = "kg m^-3",
    comments = "Density of the first updraft",
    compute! = compute_rhoaup!,
)

###
# Updraft w velocity (3d)
###
compute_waup!(out, state, cache, time) =
    compute_waup!(out, state, cache, time, cache.atmos.turbconv_model)
compute_waup!(_, _, _, _, turbconv_model::T) where {T} =
    error_diagnostic_variable("waup", turbconv_model)

function compute_waup!(
    out,
    state,
    cache,
    time,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    if isnothing(out)
        return copy(w_component.(Geometry.WVector.(cache.precomputed.ᶜuʲs.:1)))
    else
        out .= w_component.(Geometry.WVector.(cache.precomputed.ᶜuʲs.:1))
    end
end

add_diagnostic_variable!(
    short_name = "waup",
    long_name = "Updraft Upward Air Velocity",
    units = "m s^-1",
    comments = "Vertical wind component of the first updraft",
    compute! = compute_waup!,
)

###
# Updraft temperature (3d)
###
compute_taup!(out, state, cache, time) =
    compute_taup!(out, state, cache, time, cache.atmos.turbconv_model)
compute_taup!(_, _, _, _, turbconv_model::T) where {T} =
    error_diagnostic_variable("taup", turbconv_model)

function compute_taup!(
    out,
    state,
    cache,
    time,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return TD.air_temperature.(thermo_params, cache.precomputed.ᶜtsʲs.:1)
    else
        out .= TD.air_temperature.(thermo_params, cache.precomputed.ᶜtsʲs.:1)
    end
end

add_diagnostic_variable!(
    short_name = "taup",
    long_name = "Updraft Air Temperature",
    units = "K",
    comments = "Temperature of the first updraft",
    compute! = compute_taup!,
)

###
# Updraft potential temperature (3d)
###
compute_thetaaup!(out, state, cache, time) =
    compute_thetaaup!(out, state, cache, time, cache.atmos.turbconv_model)
compute_thetaaup!(_, _, _, _, turbconv_model::T) where {T} =
    error_diagnostic_variable("thetaaup", turbconv_model)

function compute_thetaaup!(
    out,
    state,
    cache,
    time,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return TD.dry_pottemp.(thermo_params, cache.precomputed.ᶜtsʲs.:1)
    else
        out .= TD.dry_pottemp.(thermo_params, cache.precomputed.ᶜtsʲs.:1)
    end
end

add_diagnostic_variable!(
    short_name = "thetaaup",
    long_name = "Updraft Air Potential Temperature",
    units = "K",
    comments = "Potential Temperature of the first updraft",
    compute! = compute_thetaaup!,
)

###
# Updraft specific enthalpy (3d)
###
compute_haup!(out, state, cache, time) =
    compute_haup!(out, state, cache, time, cache.atmos.turbconv_model)
compute_haup!(_, _, _, _, turbconv_model::T) where {T} =
    error_diagnostic_variable("haup", turbconv_model)

function compute_haup!(
    out,
    state,
    cache,
    time,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return TD.specific_enthalpy.(thermo_params, cache.precomputed.ᶜtsʲs.:1)
    else
        out .= TD.specific_enthalpy.(thermo_params, cache.precomputed.ᶜtsʲs.:1)
    end
end

add_diagnostic_variable!(
    short_name = "haup",
    long_name = "Updraft Air Specific Enthalpy",
    units = "m^2 s^-2",
    comments = "Specific enthalpy of the first updraft",
    compute! = compute_haup!,
)

###
# Updraft total specific humidity (3d)
###
compute_husup!(out, state, cache, time) = compute_husup!(
    out,
    state,
    cache,
    time,
    cache.atmos.moisture_model,
    cache.atmos.turbconv_model,
)
compute_husup!(
    _,
    _,
    _,
    _,
    moisture_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft specific humidity with a moist model and with EDMFX",
)

function compute_husup!(
    out,
    state,
    cache,
    time,
    moisture_model::Union{EquilMoistModel, NonEquilMoistModel},
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return TD.total_specific_humidity.(
            thermo_params,
            cache.precomputed.ᶜtsʲs.:1,
        )
    else
        out .=
            TD.total_specific_humidity.(
                thermo_params,
                cache.precomputed.ᶜtsʲs.:1,
            )
    end
end

add_diagnostic_variable!(
    short_name = "husup",
    long_name = "Updraft Specific Humidity",
    units = "kg kg^-1",
    comments = "Specific humidity of the first updraft",
    compute! = compute_husup!,
)

###
# Updraft relative humidity (3d)
###
compute_hurup!(out, state, cache, time) = compute_hurup!(
    out,
    state,
    cache,
    time,
    cache.atmos.moisture_model,
    cache.atmos.turbconv_model,
)
compute_hurup!(
    _,
    _,
    _,
    _,
    moisture_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft relative humidity and with a moist model and with EDMFX",
)

function compute_hurup!(
    out,
    state,
    cache,
    time,
    moisture_model::Union{EquilMoistModel, NonEquilMoistModel},
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return TD.relative_humidity.(thermo_params, cache.precomputed.ᶜtsʲs.:1)
    else
        out .= TD.relative_humidity.(thermo_params, cache.precomputed.ᶜtsʲs.:1)
    end
end

add_diagnostic_variable!(
    short_name = "hurup",
    long_name = "Updraft Relative Humidity",
    units = "",
    comments = "Relative humidity of the first updraft",
    compute! = compute_hurup!,
)

###
# Updraft liquid water specific humidity (3d)
###
compute_clwup!(out, state, cache, time) = compute_clwup!(
    out,
    state,
    cache,
    time,
    cache.atmos.moisture_model,
    cache.atmos.turbconv_model,
)
compute_clwup!(
    _,
    _,
    _,
    _,
    moisture_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft liquid water specific humidity and with a moist model and with EDMFX",
)

function compute_clwup!(
    out,
    state,
    cache,
    time,
    moisture_model::EquilMoistModel,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return TD.liquid_specific_humidity.(
            thermo_params,
            cache.precomputed.ᶜtsʲs.:1,
        )
    else
        out .=
            TD.liquid_specific_humidity.(
                thermo_params,
                cache.precomputed.ᶜtsʲs.:1,
            )
    end
end
function compute_clwup!(
    out,
    state,
    cache,
    time,
    moisture_model::NonEquilMoistModel,
    turbconv_model::PrognosticEDMFX,
)
    if isnothing(out)
        return (state.c.sgsʲs.:1).q_liq
    else
        out .= (state.c.sgsʲs.:1).q_liq
    end
end

add_diagnostic_variable!(
    short_name = "clwup",
    long_name = "Updraft Mass Fraction of Cloud Liquid Water",
    units = "kg kg^-1",
    comments = """
    This is calculated as the mass of cloud liquid water in the first updraft divided by
    the mass of air (including the water in all phases) in the first updraft.
    """,
    compute! = compute_clwup!,
)

###
# Updraft ice water specific humidity (3d)
###
compute_cliup!(out, state, cache, time) = compute_cliup!(
    out,
    state,
    cache,
    time,
    cache.atmos.moisture_model,
    cache.atmos.turbconv_model,
)
compute_cliup!(
    _,
    _,
    _,
    _,
    moisture_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft ice water specific humidity and with a moist model and with EDMFX",
)

function compute_cliup!(
    out,
    state,
    cache,
    time,
    moisture_model::EquilMoistModel,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return TD.ice_specific_humidity.(
            thermo_params,
            cache.precomputed.ᶜtsʲs.:1,
        )
    else
        out .=
            TD.ice_specific_humidity.(thermo_params, cache.precomputed.ᶜtsʲs.:1)
    end
end
function compute_cliup!(
    out,
    state,
    cache,
    time,
    moisture_model::NonEquilMoistModel,
    turbconv_model::PrognosticEDMFX,
)
    if isnothing(out)
        return (state.c.sgsʲs.:1).q_ice
    else
        out .= (state.c.sgsʲs.:1).q_ice
    end
end

add_diagnostic_variable!(
    short_name = "cliup",
    long_name = "Updraft Mass Fraction of Cloud Ice",
    units = "kg kg^-1",
    comments = """
    This is calculated as the mass of cloud ice in the first updraft divided by
    the mass of air (including the water in all phases) in the first updraft.
    """,
    compute! = compute_cliup!,
)

###
# Updraft rain water specific humidity (3d)
###
compute_husraup!(out, state, cache, time) = compute_husraup!(
    out,
    state,
    cache,
    time,
    cache.atmos.precip_model,
    cache.atmos.turbconv_model,
)
compute_husraup!(
    _,
    _,
    _,
    _,
    precip_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft rain water specific humidity with a 1M precip model and with EDMFX",
)

function compute_husraup!(
    out,
    state,
    cache,
    time,
    precip_model::Microphysics1Moment,
    turbconv_model::PrognosticEDMFX,
)
    if isnothing(out)
        return (state.c.sgsʲs.:1).q_rai
    else
        out .= (state.c.sgsʲs.:1).q_rai
    end
end

add_diagnostic_variable!(
    short_name = "husraup",
    long_name = "Updraft Mass Fraction of Rain",
    units = "kg kg^-1",
    comments = """
    This is calculated as the mass of rain in the first updraft divided by
    the mass of air (including the water in all phases) in the first updraft.
    """,
    compute! = compute_husraup!,
)

###
# Updraft snow specific humidity (3d)
###
compute_hussnup!(out, state, cache, time) = compute_hussnup!(
    out,
    state,
    cache,
    time,
    cache.atmos.precip_model,
    cache.atmos.turbconv_model,
)
compute_hussnup!(
    _,
    _,
    _,
    _,
    precip_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft snow specific humidity with a 1M precip model and with EDMFX",
)

function compute_hussnup!(
    out,
    state,
    cache,
    time,
    precip_model::Microphysics1Moment,
    turbconv_model::PrognosticEDMFX,
)
    if isnothing(out)
        return (state.c.sgsʲs.:1).q_sno
    else
        out .= (state.c.sgsʲs.:1).q_sno
    end
end

add_diagnostic_variable!(
    short_name = "hussnup",
    long_name = "Updraft Mass Fraction of Snow",
    units = "kg kg^-1",
    comments = """
    This is calculated as the mass of snow in the first updraft divided by
    the mass of air (including the water in all phases) in the first updraft.
    """,
    compute! = compute_hussnup!,
)

###
# Entrainment (3d)
###
compute_entr!(out, state, cache, time) =
    compute_entr!(out, state, cache, time, cache.atmos.turbconv_model)
compute_entr!(_, _, _, _, turbconv_model::T) where {T} =
    error_diagnostic_variable("entr", turbconv_model)

function compute_entr!(
    out,
    state,
    cache,
    time,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    if isnothing(out)
        return copy(cache.precomputed.ᶜentrʲs.:1)
    else
        out .= cache.precomputed.ᶜentrʲs.:1
    end
end

add_diagnostic_variable!(
    short_name = "entr",
    long_name = "Entrainment rate",
    units = "s^-1",
    comments = "Entrainment rate of the first updraft",
    compute! = compute_entr!,
)

compute_turbentr!(out, state, cache, time) =
    compute_turbentr!(out, state, cache, time, cache.atmos.turbconv_model)
compute_turbentr!(_, _, _, _, turbconv_model::T) where {T} =
    error_diagnostic_variable("turbentr", turbconv_model)

function compute_turbentr!(
    out,
    state,
    cache,
    time,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    if isnothing(out)
        return copy(cache.precomputed.ᶜturb_entrʲs.:1)
    else
        out .= cache.precomputed.ᶜturb_entrʲs.:1
    end
end

add_diagnostic_variable!(
    short_name = "turbentr",
    long_name = "Turbulent entrainment rate",
    units = "s^-1",
    comments = "Turbulent entrainment rate of the first updraft",
    compute! = compute_turbentr!,
)

###
# Detrainment (3d)
###
compute_detr!(out, state, cache, time) =
    compute_detr!(out, state, cache, time, cache.atmos.turbconv_model)
compute_detr!(_, _, _, _, turbconv_model::T) where {T} =
    error_diagnostic_variable("detr", turbconv_model)

function compute_detr!(
    out,
    state,
    cache,
    time,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    if isnothing(out)
        return copy(cache.precomputed.ᶜdetrʲs.:1)
    else
        out .= cache.precomputed.ᶜdetrʲs.:1
    end
end

add_diagnostic_variable!(
    short_name = "detr",
    long_name = "Detrainment rate",
    units = "s^-1",
    comments = "Detrainment rate of the first updraft",
    compute! = compute_detr!,
)

###
# Environment area fraction (3d)
###
compute_aren!(out, state, cache, time) =
    compute_aren!(out, state, cache, time, cache.atmos.turbconv_model)
compute_aren!(_, _, _, _, turbconv_model::T) where {T} =
    error_diagnostic_variable("aren", turbconv_model)

function compute_aren!(out, state, cache, time, turbconv_model::PrognosticEDMFX)
    thermo_params = CAP.thermodynamics_params(cache.params)
    ᶜρa⁰ = @.lazy(ρa⁰(state.c))
    if isnothing(out)
        return draft_area.(
            ᶜρa⁰,
            TD.air_density.(thermo_params, cache.precomputed.ᶜts⁰),
        )
    else
        out .=
            draft_area.(
                ᶜρa⁰,
                TD.air_density.(thermo_params, cache.precomputed.ᶜts⁰),
            )
    end
end

function compute_aren!(out, state, cache, time, turbconv_model::DiagnosticEDMFX)
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return draft_area.(
            cache.precomputed.ᶜρaʲs.:1,
            TD.air_density.(thermo_params, cache.precomputed.ᶜtsʲs.:1),
        )
    else
        out .=
            draft_area.(
                cache.precomputed.ᶜρaʲs.:1,
                TD.air_density.(thermo_params, cache.precomputed.ᶜtsʲs.:1),
            )
    end
end

add_diagnostic_variable!(
    short_name = "aren",
    long_name = "Environment Area Fraction",
    units = "",
    compute! = compute_aren!,
)

###
# Environment density (3d)
###
compute_rhoaen!(out, state, cache, time) =
    compute_rhoaen!(out, state, cache, time, cache.atmos.turbconv_model)
compute_rhoaen!(_, _, _, _, turbconv_model::T) where {T} =
    error_diagnostic_variable("rhoaen", turbconv_model)

function compute_rhoaen!(
    out,
    state,
    cache,
    time,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return TD.air_density.(thermo_params, cache.precomputed.ᶜts⁰)
    else
        out .= TD.air_density.(thermo_params, cache.precomputed.ᶜts⁰)
    end
end

add_diagnostic_variable!(
    short_name = "rhoaen",
    long_name = "Environment Air Density",
    units = "kg m^-3",
    compute! = compute_rhoaen!,
)

###
# Environment w velocity (3d)
###
compute_waen!(out, state, cache, time) =
    compute_waen!(out, state, cache, time, cache.atmos.turbconv_model)
compute_waen!(_, _, _, _, turbconv_model::T) where {T} =
    error_diagnostic_variable("waen", turbconv_model)

function compute_waen!(
    out,
    state,
    cache,
    time,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    if isnothing(out)
        return copy(w_component.(Geometry.WVector.(cache.precomputed.ᶜu⁰)))
    else
        out .= w_component.(Geometry.WVector.(cache.precomputed.ᶜu⁰))
    end
end

add_diagnostic_variable!(
    short_name = "waen",
    long_name = "Environment Upward Air Velocity",
    units = "m s^-1",
    comments = "Vertical wind component of the environment",
    compute! = compute_waen!,
)

###
# Environment temperature (3d)
###
compute_taen!(out, state, cache, time) =
    compute_taen!(out, state, cache, time, cache.atmos.turbconv_model)
compute_taen!(_, _, _, _, turbconv_model::T) where {T} =
    error_diagnostic_variable("taen", turbconv_model)

function compute_taen!(out, state, cache, time, turbconv_model::PrognosticEDMFX)
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return TD.air_temperature.(thermo_params, cache.precomputed.ᶜts⁰)
    else
        out .= TD.air_temperature.(thermo_params, cache.precomputed.ᶜts⁰)
    end
end

add_diagnostic_variable!(
    short_name = "taen",
    long_name = "Environment Air Temperature",
    units = "K",
    compute! = compute_taen!,
)

###
# Environment potential temperature (3d)
###
compute_thetaaen!(out, state, cache, time) =
    compute_thetaaen!(out, state, cache, time, cache.atmos.turbconv_model)
compute_thetaaen!(_, _, _, _, turbconv_model::T) where {T} =
    error_diagnostic_variable("thetaaen", turbconv_model)

function compute_thetaaen!(
    out,
    state,
    cache,
    time,
    turbconv_model::PrognosticEDMFX,
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return TD.dry_pottemp.(thermo_params, cache.precomputed.ᶜts⁰)
    else
        out .= TD.dry_pottemp.(thermo_params, cache.precomputed.ᶜts⁰)
    end
end

add_diagnostic_variable!(
    short_name = "thetaaen",
    long_name = "Environment Air Potential Temperature",
    units = "K",
    compute! = compute_thetaaen!,
)

###
# Environment specific enthalpy (3d)
###
compute_haen!(out, state, cache, time) =
    compute_haen!(out, state, cache, time, cache.atmos.turbconv_model)
compute_haen!(_, _, _, _, turbconv_model::T) where {T} =
    error_diagnostic_variable("haen", turbconv_model)

function compute_haen!(out, state, cache, time, turbconv_model::PrognosticEDMFX)
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return TD.dry_pottemp.(thermo_params, cache.precomputed.ᶜts⁰)
    else
        out .= TD.dry_pottemp.(thermo_params, cache.precomputed.ᶜts⁰)
    end
end

add_diagnostic_variable!(
    short_name = "haen",
    long_name = "Environment Air Specific Enthalpy",
    units = "K",
    compute! = compute_haen!,
)

###
# Environment total specific humidity (3d)
###
compute_husen!(out, state, cache, time) = compute_husen!(
    out,
    state,
    cache,
    time,
    cache.atmos.moisture_model,
    cache.atmos.turbconv_model,
)
compute_husen!(
    _,
    _,
    _,
    _,
    moisture_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft specific humidity and with a moist model and with EDMFX",
)

function compute_husen!(
    out,
    state,
    cache,
    time,
    moisture_model::Union{EquilMoistModel, NonEquilMoistModel},
    turbconv_model::PrognosticEDMFX,
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return TD.total_specific_humidity.(
            thermo_params,
            cache.precomputed.ᶜts⁰,
        )
    else
        out .=
            TD.total_specific_humidity.(thermo_params, cache.precomputed.ᶜts⁰)
    end
end

add_diagnostic_variable!(
    short_name = "husen",
    long_name = "Environment Specific Humidity",
    units = "kg kg^-1",
    compute! = compute_husen!,
)

###
# Environment relative humidity (3d)
###
compute_huren!(out, state, cache, time) = compute_huren!(
    out,
    state,
    cache,
    time,
    cache.atmos.moisture_model,
    cache.atmos.turbconv_model,
)
compute_huren!(
    _,
    _,
    _,
    _,
    moisture_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft relative humidity and with a moist model and with EDMFX",
)

function compute_huren!(
    out,
    state,
    cache,
    time,
    moisture_model::Union{EquilMoistModel, NonEquilMoistModel},
    turbconv_model::PrognosticEDMFX,
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return TD.relative_humidity.(thermo_params, cache.precomputed.ᶜts⁰)
    else
        out .= TD.relative_humidity.(thermo_params, cache.precomputed.ᶜts⁰)
    end
end

add_diagnostic_variable!(
    short_name = "huren",
    long_name = "Environment Relative Humidity",
    units = "",
    compute! = compute_huren!,
)

###
# Environment liquid water specific humidity (3d)
###
compute_clwen!(out, state, cache, time) = compute_clwen!(
    out,
    state,
    cache,
    time,
    cache.atmos.moisture_model,
    cache.atmos.turbconv_model,
)
compute_clwen!(
    _,
    _,
    _,
    _,
    moisture_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft liquid water specific humidity and with a moist model and with EDMFX",
)

function compute_clwen!(
    out,
    state,
    cache,
    time,
    moisture_model::EquilMoistModel,
    turbconv_model::PrognosticEDMFX,
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return TD.liquid_specific_humidity.(
            thermo_params,
            cache.precomputed.ᶜts⁰,
        )
    else
        out .=
            TD.liquid_specific_humidity.(thermo_params, cache.precomputed.ᶜts⁰)
    end
end

function compute_clwen!(
    out,
    state,
    cache,
    time,
    moisture_model::NonEquilMoistModel,
    turbconv_model::PrognosticEDMFX,
)
    if isnothing(out)
        return specific_env_value(:q_liq, state.c, turbconv_model)
    else
        out .= specific_env_value(:q_liq, state.c, turbconv_model)
    end
end

add_diagnostic_variable!(
    short_name = "clwen",
    long_name = "Envrionment Mass Fraction of Cloud Liquid Water",
    units = "kg kg^-1",
    comments = """
    This is calculated as the mass of cloud liquid water in the environment divided by
    the mass of air (including the water in all phases) in the environment.
    """,
    compute! = compute_clwen!,
)

###
# Environment ice water specific humidity (3d)
###
compute_clien!(out, state, cache, time) = compute_clien!(
    out,
    state,
    cache,
    time,
    cache.atmos.moisture_model,
    cache.atmos.turbconv_model,
)
compute_clien!(
    _,
    _,
    _,
    _,
    moisture_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft ice water specific humidity and with a moist model and with EDMFX",
)

function compute_clien!(
    out,
    state,
    cache,
    time,
    moisture_model::EquilMoistModel,
    turbconv_model::PrognosticEDMFX,
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return TD.ice_specific_humidity.(thermo_params, cache.precomputed.ᶜts⁰)
    else
        out .= TD.ice_specific_humidity.(thermo_params, cache.precomputed.ᶜts⁰)
    end
end

function compute_clien!(
    out,
    state,
    cache,
    time,
    moisture_model::NonEquilMoistModel,
    turbconv_model::PrognosticEDMFX,
)
    if isnothing(out)
        return specific_env_value(:q_ice, state.c, turbconv_model)
    else
        out .= specific_env_value(:q_ice, state.c, turbconv_model)
    end
end

add_diagnostic_variable!(
    short_name = "clien",
    long_name = "Environment Mass Fraction of Cloud Ice",
    units = "kg kg^-1",
    comments = """
    This is calculated as the mass of cloud ice in the environment divided by
    the mass of air (including the water in all phases) in the environment.
    """,
    compute! = compute_clien!,
)

###
# Environment rain water specific humidity (3d)
###
compute_husraen!(out, state, cache, time) = compute_husraen!(
    out,
    state,
    cache,
    time,
    cache.atmos.precip_model,
    cache.atmos.turbconv_model,
)
compute_husraen!(
    _,
    _,
    _,
    _,
    precip_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft rain specific humidity and with a 1M model and with EDMFX",
)

function compute_husraen!(
    out,
    state,
    cache,
    time,
    precip_model_model::Microphysics1Moment,
    turbconv_model::PrognosticEDMFX,
)
    if isnothing(out)
        return specific_env_value(:q_rai, state.c, turbconv_model)
    else
        out .= specific_env_value(:q_rai, state.c, turbconv_model)
    end
end

add_diagnostic_variable!(
    short_name = "husraen",
    long_name = "Environment Mass Fraction of Rain",
    units = "kg kg^-1",
    comments = """
    This is calculated as the mass of rain in the environment divided by
    the mass of air (including the water in all phases) in the environment.
    """,
    compute! = compute_husraen!,
)

###
# Environment snow water specific humidity (3d)
###
compute_hussnen!(out, state, cache, time) = compute_hussnen!(
    out,
    state,
    cache,
    time,
    cache.atmos.precip_model,
    cache.atmos.turbconv_model,
)
compute_hussnen!(
    _,
    _,
    _,
    _,
    precip_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft snow specific humidity and with a 1M model and with EDMFX",
)

function compute_hussnen!(
    out,
    state,
    cache,
    time,
    precip_model_model::Microphysics1Moment,
    turbconv_model::PrognosticEDMFX,
)
    if isnothing(out)
        return specific_env_value(:q_sno, state.c, turbconv_model)
    else
        out .= specific_env_value(:q_sno, state.c, turbconv_model)
    end
end

add_diagnostic_variable!(
    short_name = "hussnen",
    long_name = "Environment Mass Fraction of Snow",
    units = "kg kg^-1",
    comments = """
    This is calculated as the mass of snow in the environment divided by
    the mass of air (including the water in all phases) in the environment.
    """,
    compute! = compute_hussnen!,
)

###
# Environment turbulent kinetic energy (3d)
###
compute_tke!(out, state, cache, time) =
    compute_tke!(out, state, cache, time, cache.atmos.turbconv_model)
compute_tke!(_, _, _, _, turbconv_model::T) where {T} =
    error_diagnostic_variable("tke", turbconv_model)

function compute_tke!(   # TODO need to fix this to remove ᶜtke⁰ from preconputed
    out,
    state,
    cache,
    time,
    turbconv_model::Union{EDOnlyEDMFX, PrognosticEDMFX, DiagnosticEDMFX},
)
    if isnothing(out)
        return copy(cache.precomputed.ᶜtke⁰)
    else
        out .= cache.precomputed.ᶜtke⁰
    end
end

add_diagnostic_variable!(
    short_name = "tke",
    long_name = "Environment Turbulent Kinetic Energy",
    units = "m^2 s^-2",
    compute! = compute_tke!,
)

###
# Environment Wall Constrained Mixing Length (3d)
###
compute_lmixw!(out, state, cache, time) =
    compute_lmixw!(out, state, cache, time, cache.atmos.turbconv_model)
compute_lmixw!(_, _, _, _, turbconv_model::T) where {T} =
    error_diagnostic_variable("lmixw", turbconv_model)

function compute_lmixw!(
    out,
    state,
    cache,
    time,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    if isnothing(out)
        return copy(cache.precomputed.ᶜmixing_length_tuple.wall)
    else
        out .= cache.precomputed.ᶜmixing_length_tuple.wall
    end
end

add_diagnostic_variable!(
    short_name = "lmixw",
    long_name = "Environment Wall Constrained Mixing Length",
    units = "m",
    compute! = compute_lmixw!,
)

###
# Environment TKE Balanced Mixing Length (3d)
###
compute_lmixtke!(out, state, cache, time) =
    compute_lmixtke!(out, state, cache, time, cache.atmos.turbconv_model)
compute_lmixtke!(_, _, _, _, turbconv_model::T) where {T} =
    error_diagnostic_variable("lmixtke", turbconv_model)

function compute_lmixtke!(
    out,
    state,
    cache,
    time,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    if isnothing(out)
        return copy(cache.precomputed.ᶜmixing_length_tuple.tke)
    else
        out .= cache.precomputed.ᶜmixing_length_tuple.tke
    end
end

add_diagnostic_variable!(
    short_name = "lmixtke",
    long_name = "Environment TKE Balanced Mixing Length",
    units = "m",
    compute! = compute_lmixtke!,
)

###
# Environment Stability Mixing Length (3d)
###
compute_lmixb!(out, state, cache, time) =
    compute_lmixb!(out, state, cache, time, cache.atmos.turbconv_model)
compute_lmixb!(_, _, _, _, turbconv_model::T) where {T} =
    error_diagnostic_variable("lmixb", turbconv_model)

function compute_lmixb!(
    out,
    state,
    cache,
    time,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    if isnothing(out)
        return copy(cache.precomputed.ᶜmixing_length_tuple.buoy)
    else
        out .= cache.precomputed.ᶜmixing_length_tuple.buoy
    end
end

add_diagnostic_variable!(
    short_name = "lmixb",
    long_name = "Environment Static Stability Mixing Length",
    units = "m",
    compute! = compute_lmixb!,
)

###
# Diffusivity of heat (3d)
###
compute_edt!(out, state, cache, time) = compute_edt!(
    out,
    state,
    cache,
    time,
    cache.atmos.vert_diff,
    cache.atmos.turbconv_model,
)
compute_edt!(_, _, _, _, vert_diff::T1, turbconv_model::T2) where {T1, T2} =
    error_diagnostic_variable(
        "Can only compute heat diffusivity with vertical diffusion or EDMFX",
    )

function compute_edt!(
    out,
    state,
    cache,
    time,
    vert_diff::Union{VerticalDiffusion, DecayWithHeightDiffusion},
    turbconv_model::Nothing,
)
    if isnothing(out)
        return copy(cache.precomputed.ᶜK_h)
    else
        out .= cache.precomputed.ᶜK_h
    end
end

function compute_edt!(
    out,
    state,
    cache,
    time,
    vert_diff::Nothing,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)

    turbconv_params = CAP.turbconv_params(cache.params)   
    (; ᶜmixing_length_tuple, ᶜlinear_buoygrad, ᶜstrain_rate_norm) = cache.precomputed  

    ᶜtke⁰ = @.lazy(specific_sgs(@name(tke), state.c.sgs⁰, state.c, turbconv_model))
    ᶜprandtl_nvec = @. lazy(turbulent_prandtl_number(params, ᶜlinear_buoygrad, ᶜstrain_rate_norm))

    ᶜmixing_length = ᶜmixing_length_tuple.master

    ᶜK_u = @. lazy(eddy_viscosity(turbconv_params, ᶜtke⁰, ᶜmixing_length))
    ᶜK_h = @. lazy(eddy_diffusivity(ᶜK_u, ᶜprandtl_nvec))
    if isnothing(out)
        return copy(ᶜK_h)
    else
        out .= ᶜK_h
    end
end

add_diagnostic_variable!(
    short_name = "edt",
    long_name = "Eddy Diffusivity Coefficient for Temperature",
    standard_name = "atmosphere_heat_diffusivity",
    units = "m^2 s^-1",
    comments = "Vertical diffusion coefficient for temperature due to parameterized eddies",
    compute! = compute_edt!,
)

###
# Diffusivity of momentum (3d)
###
compute_evu!(out, state, cache, time) = compute_evu!(
    out,
    state,
    cache,
    time,
    cache.atmos.vert_diff,
    cache.atmos.turbconv_model,
)
compute_evu!(_, _, _, _, vert_diff::T1, turbconv_model::T2) where {T1, T2} =
    error_diagnostic_variable(
        "Can only compute momentum diffusivity with vertical diffusion or EDMFX",
    )

function compute_evu!(
    out,
    state,
    cache,
    time,
    vert_diff::Union{VerticalDiffusion, DecayWithHeightDiffusion},
    turbconv_model::Nothing,
)
    if isnothing(out)
        return copy(cache.precomputed.ᶜK_u)
    else
        out .= cache.precomputed.ᶜK_u
    end
end

function compute_evu!(
    out,
    state,
    cache,
    time,
    vert_diff::Nothing,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    turbconv_params = CAP.turbconv_params(cache.params)   
    (; ᶜmixing_length_tuple, ᶜtke⁰) = cache.precomputed  # TODO need to remove ᶜtke⁰
    ᶜmixing_length = ᶜmixing_length_tuple.master
    ᶜK_u = @. lazy(eddy_viscosity(turbconv_params, ᶜtke⁰, ᶜmixing_length))

    if isnothing(out)
        return copy(ᶜK_u)
    else
        out .= ᶜK_u
    end
end

add_diagnostic_variable!(
    short_name = "evu",
    long_name = "Eddy Viscosity Coefficient for Momentum",
    standard_name = "atmosphere_momentum_diffusivity",
    units = "m^2 s^-1",
    comments = "Vertical diffusion coefficient for momentum due to parameterized eddies",
    compute! = compute_evu!,
)
