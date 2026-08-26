"""
SOCRATES case identity and the per-case constants every layer keys off.

A `SocratesCase` names one Atlas LES configuration: a flight number and a forcing source.
Everything a case *is* — domain top, run length, prescribed droplet number, the hours Atlas
averaged over — is derived here so the model, the scoring, and the calibration cannot disagree
about it.
"""

using Dates: Dates
using NCDatasets: NCDatasets as NC
using SOCRATESSingleColumnForcings: SOCRATESSingleColumnForcings as SSCF

"""
    SocratesCase(flight_number, forcing_type)

One Atlas LES case: SOCRATES research flight `flight_number` forced by `forcing_type`
(`SSCF.ObsForcing()` or `SSCF.ERA5Forcing()`).

Construct from a name with [`socrates_case`](@ref), or get all of them with
[`socrates_cases`](@ref).
"""
struct SocratesCase{FT_TYPE <: SSCF.AbstractForcingType}
    flight_number::Int
    forcing_type::FT_TYPE
end

"""
    forcing_type(sym::Symbol)

The `SSCF.AbstractForcingType` for `:Obs` or `:ERA5`.
"""
forcing_type(::Val{:Obs}) = SSCF.ObsForcing()
forcing_type(::Val{:ERA5}) = SSCF.ERA5Forcing()
forcing_type(sym::Symbol) =
    sym in (:Obs, :ERA5) ? forcing_type(Val(sym)) :
    error("Unknown SOCRATES forcing type `:$sym`; expected `:Obs` or `:ERA5`")
forcing_type(ft::SSCF.AbstractForcingType) = ft
forcing_type(case::SocratesCase) = case.forcing_type

"""Short forcing label, `:Obs` or `:ERA5` (`SSCF.symbol`)."""
forcing_label(case::SocratesCase) = SSCF.symbol(case.forcing_type)

"""
    case_name(case)

The canonical case name, e.g. `"RF09_Obs"` or `"RF09_ERA5"`. Used for output subdirectories
and as the `EKP.Observation` name.
"""
case_name(case::SocratesCase) =
    string("RF", lpad(case.flight_number, 2, '0'), "_", forcing_label(case))

Base.show(io::IO, case::SocratesCase) = print(io, "SocratesCase(", case_name(case), ")")

"""
    socrates_case(name)

Parse a case name such as `"RF09_Obs"` or `"RF09_ERA5"` into a [`SocratesCase`](@ref).

The result is [`validate`](@ref)d, so a flight with no artifact for that forcing — a typo, or a real
flight such as RF11 that has no `Obs` forcing — fails here rather than at the first run.
"""
function socrates_case(name::AbstractString)
    m = match(r"^RF(\d{1,2})_(Obs|ERA5)$", name)
    isnothing(m) && error(
        "Cannot parse SOCRATES case name `$name`; expected e.g. \"RF09_Obs\" or \"RF09_ERA5\"",
    )
    return validate(SocratesCase(parse(Int, m[1]), forcing_type(Symbol(m[2]))))
end

socrates_case(case::SocratesCase) = case
socrates_case(flight_number::Integer, ft) =
    validate(SocratesCase(Int(flight_number), forcing_type(ft)))

"""
    socrates_cases()

Every valid (flight, forcing) Atlas LES case: the 5 Obs cases and the 6 ERA5 cases. Flight 11
has no Obs artifact, so it appears only under ERA5.
"""
function socrates_cases()
    cases = SocratesCase[]
    for ft in SSCF.forcing_types, flight in SSCF.flight_numbers
        SSCF.is_valid_flight_number(ft, flight) && push!(cases, SocratesCase(flight, ft))
    end
    return cases
end

"""
    validate(case)

Error unless SSCF has an Atlas artifact for this (flight, forcing) pair. Flight 11 has ERA5
forcing only.
"""
function validate(case::SocratesCase)
    SSCF.is_valid_flight_number(case.forcing_type, case.flight_number) || error(
        "No SOCRATES $(forcing_label(case)) artifact for flight $(case.flight_number) \
         (case $(case_name(case))). Valid Obs flights: \
         $(filter(f -> SSCF.is_valid_flight_number(SSCF.ObsForcing(), f), SSCF.flight_numbers)); \
         valid ERA5 flights: \
         $(filter(f -> SSCF.is_valid_flight_number(SSCF.ERA5Forcing(), f), SSCF.flight_numbers)).",
    )
    return case
end

# --- run length ---------------------------------------------------------------------------- #

"""
Atlas LES run length [s] by forcing source: Obs cases run 12 h, ERA5 cases 14 h
(Atlas et al. 2020).
"""
const RUN_DURATION_SECONDS = Base.ImmutableDict(:Obs => 12 * 3600.0, :ERA5 => 14 * 3600.0)

"""
    t_end(case)

Simulation end time [s] for `case`, matching the Atlas LES run length.
"""
t_end(case::SocratesCase) = RUN_DURATION_SECONDS[forcing_label(case)]

# --- prescribed cloud droplet number ------------------------------------------------------- #

"""
Prescribed cloud droplet number concentration [m^-3] by flight, from the Atlas LES `Nd` used
for each case. Enters the parameter set as `prescribed_cloud_droplet_number_concentration`.
"""
const N_CCN = Base.ImmutableDict(
    1 => 75.0e6,
    9 => 190.0e6,
    10 => 55.0e6,
    11 => 115.0e6,
    12 => 210.0e6,
    13 => 180.0e6,
)

"""
    n_ccn(case)

Prescribed cloud droplet number concentration [m^-3] for this case's flight.
"""
function n_ccn(case::SocratesCase)
    haskey(N_CCN, case.flight_number) ||
        error("No prescribed N_CCN for flight $(case.flight_number)")
    return N_CCN[case.flight_number]
end

# --- clocks -------------------------------------------------------------------------------- #

"""
    les_start_datetime(case)

True wall-clock start of the Atlas LES run (the Atlas reference time minus 12 h). Used when
sampling SSCF forcing and when computing insolation, which both need the real date.
"""
les_start_datetime(case::SocratesCase) =
    SSCF.get_socrates_initial_time(case.flight_number)

"""
    simulation_start_date(case)

Epoch for the simulation clock. The model runs on `t` seconds since this date; it is a fixed
epoch rather than the true LES date so that `t = 0` means "start of the run" everywhere.
Insolation is driven by `coszen`/`rsdt` series computed from [`les_start_datetime`](@ref), so
the fixed epoch does not desynchronize the solar cycle.
"""
simulation_start_date(::SocratesCase) = Dates.DateTime(1970, 1, 1)

# --- scoring window ------------------------------------------------------------------------ #

"""
Obs-case scoring window [s]: hours 10–12 of the 12 h run, the interval Atlas et al. (2020)
average over when comparing to observations.
"""
const OBS_SCORE_WINDOW_SECONDS = (10 * 3600.0, 12 * 3600.0)

"""
    score_window(case)

The `(t_start, t_end)` window in seconds over which observations and model output are
time-averaged for this case.

Obs cases use hours 10–12 (Atlas et al. 2020). ERA5 cases use each flight's own window from
the Atlas Table 2 metadata in `SOCRATES_summary.nc`, expressed relative to that flight's
reference time (which is hour 12 of the run) — i.e.
`(time_bnds - reference_time) + 12 h`.
"""
function score_window(case::SocratesCase)
    window =
        case.forcing_type isa SSCF.ObsForcing ? OBS_SCORE_WINDOW_SECONDS :
        era5_score_window(case.flight_number)
    return check_score_window(case, window)
end

"""
    check_score_window(case, (t0, t1))

Error unless the window is increasing and lies inside the case's run, `0 ≤ t0 < t1 ≤ t_end`.
A window outside the run silently scores nothing (or extrapolates), so this is checked rather
than assumed.
"""
function check_score_window(case::SocratesCase, window)
    t0, t1 = window
    duration = t_end(case)
    (0 <= t0 < t1 <= duration) || error(
        "Scoring window ($t0, $t1) s for $(case_name(case)) is not within the run \
         [0, $duration] s (hours $(t0 / 3600) to $(t1 / 3600) of a \
         $(duration / 3600) h run).",
    )
    return (t0, t1)
end

"""
    era5_score_window(flight_number)

`(t_start, t_end)` in seconds for an ERA5 case, from the Atlas Table 2 metadata in
`SOCRATES_summary.nc`.

`time_bnds` and `reference_time` carry *different* CF epochs in that file, so both are read
through NCDatasets' CF decoding and subtracted as `DateTime`s. The reference time is hour 12 of
the run, so the window is `(time_bnds - reference_time) + 12 h`.
"""
function era5_score_window(flight_number::Integer)
    path = SSCF.atlas_socrates_summary_file(Int(flight_number))
    isfile(path) || error("Atlas SOCRATES summary file not found: $path")
    return NC.NCDataset(path, "r") do ds
        flights = vec(Array(ds["flight_number"]))
        i = findfirst(==(flight_number), flights)
        isnothing(i) && error(
            "Flight $flight_number not present in $path (has flights $(collect(flights)))",
        )
        # Julia sees the CDL `time_bnds(flight_number, nbnds)` transposed, as (nbnds, flight).
        bnds_var = ds["time_bnds"]
        size(bnds_var) == (2, length(flights)) || error(
            "$path `time_bnds` has size $(size(bnds_var)); expected \
             (2, $(length(flights))) as (nbnds, flight_number).",
        )
        bnds = bnds_var[:, i]
        reference = ds["reference_time"][i]
        offsets = (Float64(Dates.value(Dates.Second(b - reference))) for b in bnds)
        t0, t1 = (o + 12 * 3600.0 for o in offsets)
        (t0, t1)
    end
end