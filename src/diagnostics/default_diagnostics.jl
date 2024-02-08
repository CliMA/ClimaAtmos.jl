# This file is included by Diagnostics.jl and defines all the defaults for various models. A
# model here is either a global AtmosModel, or small (sub)models (e.g., DryModel()).
#
# If you are developing new models, add your defaults here. If you want to add more high
# level interfaces, add them here. Feel free to include extra files.

"""
    default_diagnostics(model, t_end; output_writer)

Return a list of `ScheduledDiagnostic`s associated with the given `model` that use
`output_write` to write to disk. `t_end` is the expected simulation end time and it is used
to choose the most reasonable output frequency.

The logic is as follows:

If `t_end < 1 day` take hourly means,
if `t_end < 30 days` take daily means,
if `t_end < 120 days` take means over ten days,
If `t_end >= 120 year` take monthly means.

One month is defined as 30 days.
"""
function default_diagnostics(model::AtmosModel, t_end::Real; output_writer)
    # Unfortunately, [] is not treated nicely in a map (we would like it to be "excluded"),
    # so we need to manually filter out the submodels that don't have defaults associated
    # to
    non_empty_fields = filter(
        x ->
            default_diagnostics(getfield(model, x), t_end; output_writer) != [],
        fieldnames(AtmosModel),
    )

    # We use a map because we want to ensure that diagnostics is a well defined type, not
    # Any. This reduces latency.
    return vcat(
        core_default_diagnostics(output_writer, t_end),
        map(non_empty_fields) do field
            default_diagnostics(
                getfield(model, field),
                t_end;
                output_writer,
            )
        end...,
    )
end

# Base case: if we call default_diagnostics on something that we don't have information
# about, we get nothing back (to be specific, we get an empty list, so that we can assume
# that all the default_diagnostics return the same type). This is used by
# default_diagnostics(model::AtmosModel; output_writer), so that we can ignore defaults for
# submodels that have no given defaults.
default_diagnostics(submodel, t_end; output_writer) = []

"""
    produce_common_diagnostic_function(period, reduction)

Helper function to define functions like `daily_max`.
"""
function common_diagnostics(
    period,
    reduction,
    output_writer,
    short_names...;
    pre_output_hook! = nothing,
)
    return [
        ScheduledDiagnosticTime(
            variable = get_diagnostic_variable(short_name),
            compute_every = :timestep,
            output_every = period, # seconds
            reduction_time_func = reduction,
            output_writer = output_writer,
            pre_output_hook! = pre_output_hook!,
        ) for short_name in short_names
    ]
end

function average_pre_output_hook!(accum, counter)
    @. accum = accum / counter
    nothing
end

include("standard_diagnostic_frequencies.jl")

"""
    frequency_averages(t_end::Real)

Return the correct averaging function depending on the total simulation time.

If `t_end < 1 day` take hourly means,
if `t_end < 30 days` take daily means,
if `t_end < 120 days` take means over ten days,
If `t_end >= 120 year` take monthly means.

One month is defined as 30 days.
"""
function frequency_averages(t_end::Real)
    if t_end > 120 * 86400
        return monthly_averages
    elseif t_end >= 30 * 86400
        return tendaily_averages
    elseif t_end >= 86400
        return daily_averages
    else
        return hourly_averages
    end
end

# Include all the subdefaults

########
# Core #
########
function core_default_diagnostics(output_writer, t_end)
    core_diagnostics =
        ["ts", "ta", "thetaa", "ha", "pfull", "rhoa", "ua", "va", "wa", "hfes"]

    average_func = frequency_averages(t_end)

    if t_end > 120 * 86400
        min_func = monthly_min
        max_func = monthly_max
    elseif t_end >= 30 * 86400
        min_func = tendaily_min
        max_func = tendaily_max
    elseif t_end >= 86400
        min_func = daily_min
        max_func = daily_max
    else
        min_func = hourly_min
        max_func = hourly_max
    end

    return [
        # We need to compute the topography at the beginning of the simulation (and only at
        # the beginning), so we set output_every = 0 (it still called at the first timestep)
        ScheduledDiagnosticIterations(;
            variable = get_diagnostic_variable("orog"),
            output_every = 0,
            output_writer,
        ),
        average_func(core_diagnostics...; output_writer)...,
        min_func("ts"; output_writer),
        max_func("ts"; output_writer),
    ]
end

##################
# Moisture model #
##################
function default_diagnostics(
    ::T,
    t_end;
    output_writer,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    moist_diagnostics = ["hur", "hus", "cl", "clw", "cli", "hussfc", "evspsbl"]

    average_func = frequency_averages(t_end)
    return [average_func(moist_diagnostics...; output_writer)...]
end

#######################
# Precipitation model #
#######################
function default_diagnostics(::Microphysics0Moment, t_end; output_writer)
    precip_diagnostics = ["pr"]

    average_func = frequency_averages(t_end)

    return [average_func(precip_diagnostics...; output_writer)...]
end

##################
# Radiation mode #
##################
function default_diagnostics(::RRTMGPI.AbstractRRTMGPMode, t_end; output_writer)
    rad_diagnostics = [
        "rsd",
        "rsdt",
        "rsds",
        "rsu",
        "rsut",
        "rsus",
        "rld",
        "rlds",
        "rlu",
        "rlut",
        "rlus",
    ]

    average_func = frequency_averages(t_end)

    return [average_func(rad_diagnostics...; output_writer)...]
end


function default_diagnostics(
    ::RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics,
    t_end;
    output_writer,
)
    rad_diagnostics = [
        "rsd",
        "rsdt",
        "rsds",
        "rsu",
        "rsut",
        "rsus",
        "rld",
        "rlds",
        "rlu",
        "rlut",
        "rlus",
    ]
    rad_clearsky_diagnostics = [
        "rsdcs",
        "rsdscs",
        "rsucs",
        "rsutcs",
        "rsuscs",
        "rldcs",
        "rldscs",
        "rlucs",
        "rlutcs",
    ]

    average_func = frequency_averages(t_end)

    return [
        average_func(rad_diagnostics...; output_writer)...,
        average_func(rad_clearsky_diagnostics...; output_writer)...,
    ]
end

##################
# Turbconv model #
##################
function default_diagnostics(::PrognosticEDMFX, t_end; output_writer)
    edmfx_tenmin_diagnostics = [
        "ts",
        "ta",
        "thetaa",
        "ha",
        "pfull",
        "rhoa",
        "ua",
        "va",
        "wa",
        "hur",
        "hus",
        "cl",
        "clw",
        "cli",
        "hussfc",
        "evspsbl",
        "arup",
        "waup",
        "taup",
        "thetaaup",
        "haup",
        "husup",
        "hurup",
        "clwup",
        "cliup",
        "rhoaen",
        "waen",
        "taen",
        "thetaaen",
        "haen",
        "husen",
        "huren",
        "clwen",
        "clien",
        "tke",
        "lmix",
    ]

    edmfx_draft_diagnostics = [
        "arup",
        "rhoaup",
        "waup",
        "taup",
        "thetaaup",
        "haup",
        "husup",
        "hurup",
        "clwup",
        "cliup",
    ]
    edmfx_env_diagnostics = [
        "aren",
        "rhoaen",
        "waen",
        "taen",
        "thetaaen",
        "haen",
        "husen",
        "huren",
        "clwen",
        "clien",
        "tke",
        "lmix",
    ]

    average_func = frequency_averages(t_end)

    return [
        thirtymin_insts(edmfx_tenmin_diagnostics...; output_writer)...,
        average_func(edmfx_draft_diagnostics...; output_writer)...,
        average_func(edmfx_env_diagnostics...; output_writer)...,
    ]
end


function default_diagnostics(::DiagnosticEDMFX, t_end; output_writer)
    diagnostic_edmfx_tenmin_diagnostics = [
        "ts",
        "ta",
        "thetaa",
        "ha",
        "pfull",
        "rhoa",
        "ua",
        "va",
        "wa",
        "hur",
        "hus",
        "cl",
        "clw",
        "cli",
        "hussfc",
        "evspsbl",
        "arup",
        "waup",
        "taup",
        "thetaaup",
        "haup",
        "husup",
        "hurup",
        "clwup",
        "cliup",
        "waen",
        "tke",
        "lmix",
    ]

    diagnostic_edmfx_draft_diagnostics = [
        "arup",
        "rhoaup",
        "waup",
        "taup",
        "thetaaup",
        "haup",
        "husup",
        "hurup",
        "clwup",
        "cliup",
    ]
    diagnostic_edmfx_env_diagnostics = ["waen", "tke", "lmix"]

    average_func = frequency_averages(t_end)

    return [
        thirtymin_insts(
            diagnostic_edmfx_tenmin_diagnostics...;
            output_writer,
        )...,
        average_func(diagnostic_edmfx_draft_diagnostics...; output_writer)...,
        average_func(diagnostic_edmfx_env_diagnostics...; output_writer)...,
    ]
end
