"""
Shared SOCRATES experiment helpers: cases, z_bounds, N_CCN, paths, config loaders.
"""

using Dates: Dates
using YAML: YAML
using SOCRATESSingleColumnForcings: SOCRATESSingleColumnForcings as SSCF

const EXPERIMENT_ROOT = dirname(@__DIR__)

"""Prescribed cloud droplet number [m^-3] by flight (Atlas Nd)."""
const N_CCN_DEFAULT = Dict(
    1 => 75.0e6,
    9 => 190.0e6,
    10 => 55.0e6,
    11 => 115.0e6,
    12 => 210.0e6,
    13 => 180.0e6,
)

"""
Vertical scoring bounds [m] by forcing family and flight.
Obs tops from user config; ERA copies Obs where available; RF11 provisional (0, 4000).
"""
const Z_BOUNDS = Dict(
    :Obs => Dict{Int, Tuple{Float64, Float64}}(
        1 => (0.0, 3000.0),
        9 => (0.0, 4000.0),
        10 => (0.0, 3500.0),
        12 => (0.0, 2000.0),
        13 => (0.0, 2000.0),
    ),
    :ERA5 => Dict{Int, Tuple{Float64, Float64}}(
        1 => (0.0, 3000.0),
        9 => (0.0, 4000.0),
        10 => (0.0, 3500.0),
        11 => (0.0, 4000.0), # provisional; refine to LES cloud-top + ~400 m
        12 => (0.0, 2000.0),
        13 => (0.0, 2000.0),
    ),
)

const T_END_SEC = Dict(:Obs => 12 * 3600.0, :ERA5 => 14 * 3600.0)

struct SOCRATESCase
    flight_number::Int
    forcing_type::SSCF.AbstractForcingType # ObsForcing() or ERA5Forcing()
end
name(case::SOCRATESCase) = "RF$(lpad(case.flight_number, 2, '0'))_$(forcing_symbol(case))"

function sscf_forcing(ft::Symbol)
    ft === :Obs && return SSCF.ObsForcing()
    ft === :ERA5 && return SSCF.ERA5Forcing()
    error("Unknown forcing_type=$ft; expected :Obs or :ERA5")
end
sscf_forcing(case::SOCRATESCase) = case.forcing_type
sscf_forcing(forcing_type::SSCF.AbstractForcingType) = forcing_type

forcing_symbol(ft::SSCF.AbstractForcingType) = SSCF.symbol(ft)
forcing_symbol(ft::Symbol) = ft
forcing_symbol(case::SOCRATESCase) = forcing_symbol(case.forcing_type)

function load_experiment_config(path = joinpath(EXPERIMENT_ROOT, "configs", "experiment_config.yml"))
    return YAML.load_file(path)
end

function load_cases(exp_cfg = load_experiment_config())
    cases = SOCRATESCase[]
    for c in exp_cfg["cases"]
        push!(
            cases,
            SOCRATESCase(Int(c["flight_number"]), sscf_forcing(Symbol(c["forcing_type"]))),
        )
    end
    return cases
end

function validate_case!(case::SOCRATESCase)
    ft = case.forcing_type
    SSCF.is_valid_flight_number(ft, case.flight_number) || error(
        "Invalid SOCRATES case $(case.name): SSCF has no $(forcing_symbol(ft)) artifact for flight $(case.flight_number)",
    )
    return nothing
end

forcing_tag(ft::SSCF.ObsForcing) = "Obs"
forcing_tag(ft::SSCF.ERA5Forcing) = "ERA"
forcing_tag(ft::Symbol) = forcing_tag(sscf_forcing(ft))
forcing_tag(case::SOCRATESCase) = forcing_tag(case.forcing_type)

climacolumn_filename(case::SOCRATESCase) =
    "RF$(lpad(case.flight_number, 2, '0'))_$(forcing_tag(case)).nc"
climacolumn_path(case::SOCRATESCase) =
    joinpath(EXPERIMENT_ROOT, "forcing", "climacolumn", climacolumn_filename(case))

function z_bounds(case::SOCRATESCase)
    key = forcing_symbol(case)
    haskey(Z_BOUNDS, key) || error("No z_bounds for $key")
    d = Z_BOUNDS[key]
    haskey(d, case.flight_number) ||
        error("No z_bounds for flight $(case.flight_number) / $key")
    return d[case.flight_number]
end

t_end_seconds(case::SOCRATESCase) = T_END_SEC[forcing_symbol(case)]
t_end_string(case::SOCRATESCase) = string(Int(t_end_seconds(case) ÷ 3600), "hours")

function score_window_sec(case::SOCRATESCase, exp_cfg)
    if case.forcing_type isa SSCF.ObsForcing
        return (Float64(exp_cfg["obs_y_t_start_sec"]), Float64(exp_cfg["obs_y_t_end_sec"]))
    else
        return (Float64(exp_cfg["era_y_t_start_sec"]), Float64(exp_cfg["era_y_t_end_sec"]))
    end
end

"""
Fixed Atmos `start_date` epoch (`yyyymmdd` only — stock Atmos warn_if parses
that format). Simulation times are seconds since this epoch; SSCF samples and
insolation use the true LES wall clock when building forcing arrays.
"""
start_date_string(::SOCRATESCase) = "19700101"

"""True LES start (Atlas reference − 12 h). Used for SSCF/insolation sampling."""
function start_datetime(case::SOCRATESCase)
    return SSCF.get_socrates_initial_time(case.flight_number)
end

"""Simulation-time epoch (`1970-01-01`); matches `start_date_string`."""
start_datetime_epoch(::SOCRATESCase) = Dates.DateTime(1970, 1, 1)

"""Domain top [m] from Atlas LES vertical grid for this flight."""
function z_max_meters(case::SOCRATESCase)
    return Float64(maximum(SSCF.default_new_z(case.flight_number)))
end

function n_ccn(case::SOCRATESCase)
    return N_CCN_DEFAULT[case.flight_number]
end

"""Write a tiny TOML overriding prescribed cloud droplet number for this case."""
function write_n_ccn_toml(case::SOCRATESCase, path::AbstractString)
    open(path, "w") do io
        println(io, "[prescribed_cloud_droplet_number_concentration]")
        println(io, "value = ", n_ccn(case))
    end
    return path
end

