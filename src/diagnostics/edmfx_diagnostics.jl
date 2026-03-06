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
    if isnothing(out)
        return draft_area.(
            (state.c.sgsʲs.:1).ρa,
            cache.precomputed.ᶜρʲs.:1,
        )
    else
        out .=
            draft_area.(
                (state.c.sgsʲs.:1).ρa,
                cache.precomputed.ᶜρʲs.:1,
            )
    end
end

function compute_arup!(out, state, cache, time, turbconv_model::DiagnosticEDMFX)
    if isnothing(out)
        return draft_area.(
            cache.precomputed.ᶜρaʲs.:1,
            cache.precomputed.ᶜρʲs.:1,
        )
    else
        out .=
            draft_area.(
                cache.precomputed.ᶜρaʲs.:1,
                cache.precomputed.ᶜρʲs.:1,
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
    if isnothing(out)
        return copy(cache.precomputed.ᶜρʲs.:1)
    else
        out .= cache.precomputed.ᶜρʲs.:1
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
    if isnothing(out)
        return copy(cache.precomputed.ᶜTʲs.:1)
    else
        out .= cache.precomputed.ᶜTʲs.:1
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
    (; ᶜTʲs, ᶜp, ᶜρʲs, ᶜq_tot_safeʲs, ᶜq_liq_raiʲs, ᶜq_ice_snoʲs) = cache.precomputed
    if isnothing(out)
        return TD.potential_temperature.(
            thermo_params,
            ᶜTʲs.:1,
            ᶜρʲs.:1,
            ᶜq_tot_safeʲs.:1,
            ᶜq_liq_raiʲs.:1,
            ᶜq_ice_snoʲs.:1,
        )
    else
        out .=
            TD.potential_temperature.(
                thermo_params,
                ᶜTʲs.:1,
                ᶜρʲs.:1,
                ᶜq_tot_safeʲs.:1,
                ᶜq_liq_raiʲs.:1,
                ᶜq_ice_snoʲs.:1,
            )
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
    (; ᶜTʲs, ᶜq_tot_safeʲs, ᶜq_liq_raiʲs, ᶜq_ice_snoʲs) = cache.precomputed
    if isnothing(out)
        return TD.enthalpy.(
            thermo_params,
            ᶜTʲs.:1,
            ᶜq_tot_safeʲs.:1,
            ᶜq_liq_raiʲs.:1,
            ᶜq_ice_snoʲs.:1,
        )
    else
        out .=
            TD.enthalpy.(
                thermo_params,
                ᶜTʲs.:1,
                ᶜq_tot_safeʲs.:1,
                ᶜq_liq_raiʲs.:1,
                ᶜq_ice_snoʲs.:1,
            )
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
    cache.atmos.microphysics_model,
    cache.atmos.turbconv_model,
)
compute_husup!(
    _,
    _,
    _,
    _,
    microphysics_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft specific humidity with a moist model and with EDMFX",
)

# TODO: use the actual q_tot
function compute_husup!(
    out,
    state,
    cache,
    time,
    microphysics_model::MoistMicrophysics,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    if isnothing(out)
        return copy(cache.precomputed.ᶜq_tot_safeʲs.:1)
    else
        out .= cache.precomputed.ᶜq_tot_safeʲs.:1
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
    cache.atmos.microphysics_model,
    cache.atmos.turbconv_model,
)
compute_hurup!(
    _,
    _,
    _,
    _,
    microphysics_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft relative humidity and with a moist model and with EDMFX",
)

function compute_hurup!(
    out,
    state,
    cache,
    time,
    microphysics_model::MoistMicrophysics,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    (; ᶜTʲs, ᶜp, ᶜq_tot_safeʲs, ᶜq_liq_raiʲs, ᶜq_ice_snoʲs) = cache.precomputed
    if isnothing(out)
        return TD.relative_humidity.(
            thermo_params,
            ᶜTʲs.:1,
            ᶜp,
            ᶜq_tot_safeʲs.:1,
            ᶜq_liq_raiʲs.:1,
            ᶜq_ice_snoʲs.:1,
        )
    else
        out .=
            TD.relative_humidity.(
                thermo_params,
                ᶜTʲs.:1,
                ᶜp,
                ᶜq_tot_safeʲs.:1,
                ᶜq_liq_raiʲs.:1,
                ᶜq_ice_snoʲs.:1,
            )
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
# Updraft liquid water specific humidity and number mixing ratio (3d)
###
compute_clwup!(out, state, cache, time) = compute_clwup!(
    out,
    state,
    cache,
    time,
    cache.atmos.microphysics_model,
    cache.atmos.turbconv_model,
)
compute_clwup!(
    _,
    _,
    _,
    _,
    microphysics_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft liquid water specific humidity and with a moist model and with EDMFX",
)

function compute_clwup!(
    out,
    state,
    cache,
    time,
    microphysics_model::EquilibriumMicrophysics0M,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    if isnothing(out)
        return copy(cache.precomputed.ᶜq_liq_raiʲs.:1)
    else
        out .= cache.precomputed.ᶜq_liq_raiʲs.:1
    end
end
function compute_clwup!(
    out,
    state,
    cache,
    time,
    microphysics_model::NonEquilibriumMicrophysics,
    turbconv_model::PrognosticEDMFX,
)
    if isnothing(out)
        return (state.c.sgsʲs.:1).q_liq
    else
        out .= (state.c.sgsʲs.:1).q_liq
    end
end
function compute_clwup!(
    out,
    state,
    cache,
    time,
    microphysics_model::NonEquilibriumMicrophysics,
    turbconv_model::DiagnosticEDMFX,
)
    if isnothing(out)
        return (cache.precomputed.ᶜq_liqʲs.:1)
    else
        out .= (cache.precomputed.ᶜq_liqʲs.:1)
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

compute_cdncup!(out, state, cache, time) = compute_cdncup!(
    out,
    state,
    cache,
    time,
    cache.atmos.microphysics_model,
    cache.atmos.turbconv_model,
)
compute_cdncup!(
    _,
    _,
    _,
    _,
    microphysics_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft rain number mixing ratio with a 2M precip model and with EDMFX",
)

function compute_cdncup!(
    out,
    state,
    cache,
    time,
    microphysics_model_model::NonEquilibriumMicrophysics2M,
    turbconv_model::PrognosticEDMFX,
)
    if isnothing(out)
        return (state.c.sgsʲs.:1).n_liq
    else
        out .= (state.c.sgsʲs.:1).n_liq
    end
end

add_diagnostic_variable!(
    short_name = "cdncup",
    long_name = "Updraft Number Mixing Ratio of Cloud Liquid Water",
    units = "kg^-1",
    comments = """
    This is calculated as the number of cloud water droplets in the updraft divided by
    the mass of air (including the water in all phases) in the updraft.
    """,
    compute! = compute_cdncup!,
)

###
# Updraft ice water specific humidity (3d)
###
compute_cliup!(out, state, cache, time) = compute_cliup!(
    out,
    state,
    cache,
    time,
    cache.atmos.microphysics_model,
    cache.atmos.turbconv_model,
)
compute_cliup!(
    _,
    _,
    _,
    _,
    microphysics_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft ice water specific humidity and with a moist model and with EDMFX",
)

function compute_cliup!(
    out,
    state,
    cache,
    time,
    microphysics_model::EquilibriumMicrophysics0M,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    if isnothing(out)
        return copy(cache.precomputed.ᶜq_ice_snoʲs.:1)
    else
        out .= cache.precomputed.ᶜq_ice_snoʲs.:1
    end
end
function compute_cliup!(
    out,
    state,
    cache,
    time,
    microphysics_model::NonEquilibriumMicrophysics,
    turbconv_model::PrognosticEDMFX,
)
    if isnothing(out)
        return (state.c.sgsʲs.:1).q_ice
    else
        out .= (state.c.sgsʲs.:1).q_ice
    end
end
function compute_cliup!(
    out,
    state,
    cache,
    time,
    microphysics_model::NonEquilibriumMicrophysics,
    turbconv_model::DiagnosticEDMFX,
)
    if isnothing(out)
        return (cache.precomputed.ᶜq_iceʲs.:1)
    else
        out .= (cache.precomputed.ᶜq_iceʲs.:1)
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
# Updraft rain water specific humidity and number mixing ratio (3d)
###
compute_husraup!(out, state, cache, time) = compute_husraup!(
    out,
    state,
    cache,
    time,
    cache.atmos.microphysics_model,
    cache.atmos.turbconv_model,
)
compute_husraup!(
    _,
    _,
    _,
    _,
    microphysics_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft rain water specific humidity with a 1M or 2M precip model and with EDMFX",
)

function compute_husraup!(
    out,
    state,
    cache,
    time,
    microphysics_model::Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M},
    turbconv_model::PrognosticEDMFX,
)
    if isnothing(out)
        return (state.c.sgsʲs.:1).q_rai
    else
        out .= (state.c.sgsʲs.:1).q_rai
    end
end
function compute_husraup!(
    out,
    state,
    cache,
    time,
    microphysics_model::NonEquilibriumMicrophysics1M,
    turbconv_model::DiagnosticEDMFX,
)
    if isnothing(out)
        return (cache.precomputed.ᶜq_raiʲs.:1)
    else
        out .= (cache.precomputed.ᶜq_raiʲs.:1)
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

compute_ncraup!(out, state, cache, time) = compute_ncraup!(
    out,
    state,
    cache,
    time,
    cache.atmos.microphysics_model,
    cache.atmos.turbconv_model,
)
compute_ncraup!(
    _,
    _,
    _,
    _,
    microphysics_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft rain number mixing ratio with a 2M precip model and with EDMFX",
)

function compute_ncraup!(
    out,
    state,
    cache,
    time,
    microphysics_model_model::NonEquilibriumMicrophysics2M,
    turbconv_model::PrognosticEDMFX,
)
    if isnothing(out)
        return (state.c.sgsʲs.:1).n_rai
    else
        out .= (state.c.sgsʲs.:1).n_rai
    end
end

add_diagnostic_variable!(
    short_name = "ncraup",
    long_name = "Updraft Number Mixing Ratio of Rain",
    units = "kg^-1",
    comments = """
    This is calculated as the number of raindrops in the updraft divided by
    the mass of air (including the water in all phases) in the updraft.
    """,
    compute! = compute_ncraup!,
)

###
# Updraft snow specific humidity (3d)
###
compute_hussnup!(out, state, cache, time) = compute_hussnup!(
    out,
    state,
    cache,
    time,
    cache.atmos.microphysics_model,
    cache.atmos.turbconv_model,
)
compute_hussnup!(
    _,
    _,
    _,
    _,
    microphysics_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft snow specific humidity with a 1M or 2M precip model and with EDMFX",
)

function compute_hussnup!(
    out,
    state,
    cache,
    time,
    microphysics_model::Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M},
    turbconv_model::PrognosticEDMFX,
)
    if isnothing(out)
        return (state.c.sgsʲs.:1).q_sno
    else
        out .= (state.c.sgsʲs.:1).q_sno
    end
end
function compute_hussnup!(
    out,
    state,
    cache,
    time,
    microphysics_model::NonEquilibriumMicrophysics1M,
    turbconv_model::DiagnosticEDMFX,
)
    if isnothing(out)
        return (cache.precomputed.ᶜq_snoʲs.:1)
    else
        out .= (cache.precomputed.ᶜq_snoʲs.:1)
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
    ᶜρa⁰ = @. lazy(ρa⁰(state.c.ρ, state.c.sgsʲs, turbconv_model))
    (; ᶜT⁰, ᶜp, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰) = cache.precomputed
    ᶜρ⁰ = @. lazy(
        TD.air_density(thermo_params, ᶜT⁰, ᶜp, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰),
    )
    if isnothing(out)
        return draft_area.(ᶜρa⁰, ᶜρ⁰)
    else
        out .= draft_area.(ᶜρa⁰, ᶜρ⁰)
    end
end

function compute_aren!(out, state, cache, time, turbconv_model::DiagnosticEDMFX)
    if isnothing(out)
        return 1 .-
               draft_area.(
            cache.precomputed.ᶜρaʲs.:1,
            cache.precomputed.ᶜρʲs.:1,
        )
    else
        out .=
            1 -
            draft_area.(
                cache.precomputed.ᶜρaʲs.:1,
                cache.precomputed.ᶜρʲs.:1,
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
    turbconv_model::PrognosticEDMFX,
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    (; ᶜT⁰, ᶜp, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰) = cache.precomputed
    if isnothing(out)
        return TD.air_density.(
            thermo_params,
            ᶜT⁰,
            ᶜp,
            ᶜq_tot_safe⁰,
            ᶜq_liq_rai⁰,
            ᶜq_ice_sno⁰,
        )
    else
        out .=
            TD.air_density.(thermo_params, ᶜT⁰, ᶜp, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰)
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
    if isnothing(out)
        return copy(cache.precomputed.ᶜT⁰)
    else
        out .= cache.precomputed.ᶜT⁰
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
    (; ᶜT⁰, ᶜp, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰) = cache.precomputed
    ᶜρ⁰ = @. lazy(
        TD.air_density(thermo_params, ᶜT⁰, ᶜp, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰),
    )
    if isnothing(out)
        return TD.potential_temperature.(
            thermo_params,
            ᶜT⁰,
            ᶜρ⁰,
            ᶜq_tot_safe⁰,
            ᶜq_liq_rai⁰,
            ᶜq_ice_sno⁰,
        )
    else
        out .=
            TD.potential_temperature.(
                thermo_params,
                ᶜT⁰,
                ᶜρ⁰,
                ᶜq_tot_safe⁰,
                ᶜq_liq_rai⁰,
                ᶜq_ice_sno⁰,
            )
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
    (; ᶜT⁰, ᶜp, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰) = cache.precomputed
    ᶜρ⁰ = @. lazy(
        TD.air_density(thermo_params, ᶜT⁰, ᶜp, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰),
    )
    if isnothing(out)
        return TD.enthalpy.(
            thermo_params,
            ᶜT⁰,
            ᶜq_tot_safe⁰,
            ᶜq_liq_rai⁰,
            ᶜq_ice_sno⁰,
        )
    else
        out .= TD.enthalpy.(thermo_params, ᶜT⁰, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰)
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
    cache.atmos.microphysics_model,
    cache.atmos.turbconv_model,
)
compute_husen!(
    _,
    _,
    _,
    _,
    microphysics_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft specific humidity and with a moist model and with EDMFX",
)

function compute_husen!(
    out,
    state,
    cache,
    time,
    microphysics_model::MoistMicrophysics,
    turbconv_model::PrognosticEDMFX,
)
    if isnothing(out)
        return copy(cache.precomputed.ᶜq_tot_safe⁰)
    else
        out .= cache.precomputed.ᶜq_tot_safe⁰
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
    cache.atmos.microphysics_model,
    cache.atmos.turbconv_model,
)
compute_huren!(
    _,
    _,
    _,
    _,
    microphysics_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft relative humidity and with a moist model and with EDMFX",
)

function compute_huren!(
    out,
    state,
    cache,
    time,
    microphysics_model::MoistMicrophysics,
    turbconv_model::PrognosticEDMFX,
)
    thermo_params = CAP.thermodynamics_params(cache.params)
    (; ᶜT⁰, ᶜp, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰) = cache.precomputed
    if isnothing(out)
        return TD.relative_humidity.(
            thermo_params,
            ᶜT⁰,
            ᶜp,
            ᶜq_tot_safe⁰,
            ᶜq_liq_rai⁰,
            ᶜq_ice_sno⁰,
        )
    else
        out .=
            TD.relative_humidity.(
                thermo_params,
                ᶜT⁰,
                ᶜp,
                ᶜq_tot_safe⁰,
                ᶜq_liq_rai⁰,
                ᶜq_ice_sno⁰,
            )
    end
end

add_diagnostic_variable!(
    short_name = "huren",
    long_name = "Environment Relative Humidity",
    units = "",
    compute! = compute_huren!,
)

###
# Environment liquid water specific humidity and number mixing ratio (3d)
###
compute_clwen!(out, state, cache, time) = compute_clwen!(
    out,
    state,
    cache,
    time,
    cache.atmos.microphysics_model,
    cache.atmos.turbconv_model,
)
compute_clwen!(
    _,
    _,
    _,
    _,
    microphysics_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft liquid water specific humidity and with a moist model and with EDMFX",
)

function compute_clwen!(
    out,
    state,
    cache,
    time,
    microphysics_model::EquilibriumMicrophysics0M,
    turbconv_model::PrognosticEDMFX,
)
    if isnothing(out)
        return copy(cache.precomputed.ᶜq_liq_rai⁰)
    else
        out .= cache.precomputed.ᶜq_liq_rai⁰
    end
end

function compute_clwen!(
    out,
    state,
    cache,
    time,
    microphysics_model::NonEquilibriumMicrophysics,
    turbconv_model::PrognosticEDMFX,
)
    ᶜq_liq⁰ = ᶜspecific_env_value(@name(q_liq), state, cache)
    if isnothing(out)
        return Base.materialize(ᶜq_liq⁰)
    else
        out .= ᶜq_liq⁰
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

compute_cdncen!(out, state, cache, time) = compute_cdncen!(
    out,
    state,
    cache,
    time,
    cache.atmos.microphysics_model,
    cache.atmos.turbconv_model,
)
compute_cdncen!(
    _,
    _,
    _,
    _,
    microphysics_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft cloud liquid water number mixing ratio with a 2M model and with EDMFX",
)

function compute_cdncen!(
    out,
    state,
    cache,
    time,
    microphysics_model_model::NonEquilibriumMicrophysics2M,
    turbconv_model::PrognosticEDMFX,
)
    ᶜn_liq⁰ = ᶜspecific_env_value(@name(n_liq), state, cache)
    if isnothing(out)
        return Base.materialize(ᶜn_liq⁰)
    else
        out .= ᶜn_liq⁰
    end
end

add_diagnostic_variable!(
    short_name = "cdncen",
    long_name = "Environment Number Mixing Ratio of Cloud Liquid Water",
    units = "kg^-1",
    comments = """
    This is calculated as the number of cloud liquid droplets in the environment divided by
    the mass of air (including the water in all phases) in the environment.
    """,
    compute! = compute_cdncen!,
)

###
# Environment ice water specific humidity (3d)
###
compute_clien!(out, state, cache, time) = compute_clien!(
    out,
    state,
    cache,
    time,
    cache.atmos.microphysics_model,
    cache.atmos.turbconv_model,
)
compute_clien!(
    _,
    _,
    _,
    _,
    microphysics_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft ice water specific humidity and with a moist model and with EDMFX",
)

function compute_clien!(
    out,
    state,
    cache,
    time,
    microphysics_model::EquilibriumMicrophysics0M,
    turbconv_model::PrognosticEDMFX,
)
    if isnothing(out)
        return copy(cache.precomputed.ᶜq_ice_sno⁰)
    else
        out .= cache.precomputed.ᶜq_ice_sno⁰
    end
end

function compute_clien!(
    out,
    state,
    cache,
    time,
    microphysics_model::NonEquilibriumMicrophysics,
    turbconv_model::PrognosticEDMFX,
)
    ᶜq_ice⁰ = ᶜspecific_env_value(@name(q_ice), state, cache)
    if isnothing(out)
        return Base.materialize(ᶜq_ice⁰)
    else
        out .= ᶜq_ice⁰
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
# Environment rain water specific humidity and number mixing ratio (3d)
###
compute_husraen!(out, state, cache, time) = compute_husraen!(
    out,
    state,
    cache,
    time,
    cache.atmos.microphysics_model,
    cache.atmos.turbconv_model,
)
compute_husraen!(
    _,
    _,
    _,
    _,
    microphysics_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft rain specific humidity with a 1M or 2M model and with EDMFX",
)

function compute_husraen!(
    out,
    state,
    cache,
    time,
    microphysics_model_model::Union{
        NonEquilibriumMicrophysics1M,
        NonEquilibriumMicrophysics2M,
    },
    turbconv_model::PrognosticEDMFX,
)
    ᶜq_rai⁰ = ᶜspecific_env_value(@name(q_rai), state, cache)
    if isnothing(out)
        return Base.materialize(ᶜq_rai⁰)
    else
        out .= ᶜq_rai⁰
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

compute_ncraen!(out, state, cache, time) = compute_ncraen!(
    out,
    state,
    cache,
    time,
    cache.atmos.microphysics_model,
    cache.atmos.turbconv_model,
)
compute_ncraen!(
    _,
    _,
    _,
    _,
    microphysics_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft rain number mixing ratio with a 2M model and with EDMFX",
)

function compute_ncraen!(
    out,
    state,
    cache,
    time,
    microphysics_model_model::NonEquilibriumMicrophysics2M,
    turbconv_model::PrognosticEDMFX,
)
    ᶜn_rai⁰ = ᶜspecific_env_value(@name(n_rai), state, cache)
    if isnothing(out)
        return Base.materialize(ᶜn_rai⁰)
    else
        out .= ᶜn_rai⁰
    end
end

add_diagnostic_variable!(
    short_name = "ncraen",
    long_name = "Environment Number Mixing Ratio of Rain",
    units = "kg^-1",
    comments = """
    This is calculated as the number of raindrops in the environment divided by
    the mass of air (including the water in all phases) in the environment.
    """,
    compute! = compute_ncraen!,
)

###
# Environment snow water specific humidity (3d)
###
compute_hussnen!(out, state, cache, time) = compute_hussnen!(
    out,
    state,
    cache,
    time,
    cache.atmos.microphysics_model,
    cache.atmos.turbconv_model,
)
compute_hussnen!(
    _,
    _,
    _,
    _,
    microphysics_model::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute updraft snow specific humidity with a 1M or 2M model and with EDMFX",
)

function compute_hussnen!(
    out,
    state,
    cache,
    time,
    microphysics_model_model::Union{
        NonEquilibriumMicrophysics1M,
        NonEquilibriumMicrophysics2M,
    },
    turbconv_model::PrognosticEDMFX,
)
    ᶜq_sno⁰ = ᶜspecific_env_value(@name(q_sno), state, cache)
    if isnothing(out)
        return Base.materialize(ᶜq_sno⁰)
    else
        out .= ᶜq_sno⁰
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
# Turbulent kinetic energy (3d)
###
compute_tke!(out, state, cache, time) =
    compute_tke!(out, state, cache, time, cache.atmos.turbconv_model)
compute_tke!(_, _, _, _, turbconv_model::T) where {T} =
    error_diagnostic_variable("tke", turbconv_model)

function compute_tke!(
    out,
    state,
    cache,
    time,
    turbconv_model::Union{EDOnlyEDMFX, PrognosticEDMFX, DiagnosticEDMFX},
)
    if turbconv_model isa PrognosticEDMFX
        sgsʲs = state.c.sgsʲs
        ᶜρa⁰ = @. lazy(ρa⁰(state.c.ρ, sgsʲs, turbconv_model))
    else
        ᶜρa⁰ = state.c.ρ
    end

    ᶜtke = @. lazy(
        specific(state.c.ρtke, state.c.ρ),
    )
    if isnothing(out)
        return Base.materialize(ᶜtke)
    else
        out .= ᶜtke
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
    ᶜwall_mixing_length = ᶜmixing_length(state, cache, Val(:wall))

    if isnothing(out)
        return Base.materialize(ᶜwall_mixing_length)
    else
        out .= ᶜwall_mixing_length
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

    ᶜtke_mixing_length = ᶜmixing_length(state, cache, Val(:tke))

    if isnothing(out)
        return Base.materialize(ᶜtke_mixing_length)
    else
        out .= ᶜtke_mixing_length
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
    ᶜbuoy_mixing_length = ᶜmixing_length(state, cache, Val(:buoy))

    if isnothing(out)
        return Base.materialize(ᶜbuoy_mixing_length)
    else
        out .= ᶜbuoy_mixing_length
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
    cache.atmos.vertical_diffusion,
    cache.atmos.turbconv_model,
)
compute_edt!(
    _,
    _,
    _,
    _,
    vertical_diffusion::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute heat diffusivity with vertical diffusion or EDMFX",
)

function compute_edt!(
    out,
    state,
    cache,
    time,
    vertical_diffusion::Union{VerticalDiffusion, DecayWithHeightDiffusion},
    turbconv_model::Nothing,
)
    (; vertical_diffusion) = cache.atmos
    (; ᶜp) = cache.precomputed

    if vertical_diffusion isa DecayWithHeightDiffusion
        ᶜK_h =
            ᶜcompute_eddy_diffusivity_coefficient(state.c.ρ, vertical_diffusion)
    elseif vertical_diffusion isa VerticalDiffusion
        ᶜK_h = ᶜcompute_eddy_diffusivity_coefficient(
            state.c.uₕ,
            ᶜp,
            vertical_diffusion,
        )
    end
    if isnothing(out)
        return copy(ᶜK_h)
    else
        out .= ᶜK_h
    end
end

function compute_edt!(
    out,
    state,
    cache,
    time,
    vertical_diffusion::Nothing,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    turbconv_params = CAP.turbconv_params(cache.params)
    (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = cache.precomputed
    (; params) = cache

    ᶜtke = @. lazy(
        specific(state.c.ρtke, state.c.ρ),
    )
    ᶜmixing_length_field = ᶜmixing_length(state, cache)
    ᶜK_u = @. lazy(eddy_viscosity(turbconv_params, ᶜtke, ᶜmixing_length_field))
    ᶜprandtl_nvec = @. lazy(
        turbulent_prandtl_number(params, ᶜlinear_buoygrad, ᶜstrain_rate_norm),
    )
    ᶜK_h = @. lazy(eddy_diffusivity(ᶜK_u, ᶜprandtl_nvec))
    if isnothing(out)
        return Base.materialize(ᶜK_h)
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
    cache.atmos.vertical_diffusion,
    cache.atmos.turbconv_model,
)
compute_evu!(
    _,
    _,
    _,
    _,
    vertical_diffusion::T1,
    turbconv_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute momentum diffusivity with vertical diffusion or EDMFX",
)

function compute_evu!(
    out,
    state,
    cache,
    time,
    vertical_diffusion::Union{VerticalDiffusion, DecayWithHeightDiffusion},
    turbconv_model::Nothing,
)
    (; vertical_diffusion) = cache.atmos
    (; ᶜp) = cache.precomputed

    # this setup assumes ᶜK_u = ᶜK_h
    if vertical_diffusion isa DecayWithHeightDiffusion
        ᶜK_u =
            ᶜcompute_eddy_diffusivity_coefficient(state.c.ρ, vertical_diffusion)
    elseif vertical_diffusion isa VerticalDiffusion
        ᶜK_u = ᶜcompute_eddy_diffusivity_coefficient(
            state.c.uₕ,
            ᶜp,
            vertical_diffusion,
        )
    end
    if isnothing(out)
        return copy(ᶜK_u)
    else
        out .= ᶜK_u
    end
end

function compute_evu!(
    out,
    state,
    cache,
    time,
    vertical_diffusion::Nothing,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    turbconv_params = CAP.turbconv_params(cache.params)

    ᶜtke = @. lazy(
        specific(state.c.ρtke, state.c.ρ),
    )
    ᶜmixing_length_field = ᶜmixing_length(state, cache)
    ᶜK_u = @. lazy(eddy_viscosity(turbconv_params, ᶜtke, ᶜmixing_length_field))

    if isnothing(out)
        return Base.materialize(ᶜK_u)
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
