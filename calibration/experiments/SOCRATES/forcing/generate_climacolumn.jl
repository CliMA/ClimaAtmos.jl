"""
SSCF Atlas LES inputs → in-memory ClimaAtmos forcing arrays.

Usage (from experiment root):
  julia --project=. forcing/generate_climacolumn.jl
  julia --project=. forcing/generate_climacolumn.jl RF01_Obs

Default path is RAM (`generate_socrates_forcing`). Optional NetCDF dump via
`write_socrates_forcing` is for debugging only (slow; not used by the driver).
"""

using ClimaAtmos: ClimaAtmos
using ClimaParams: ClimaParams as CP
using Dates: Dates
using Insolation: Insolation
using SOCRATESSingleColumnForcings: SOCRATESSingleColumnForcings as SSCF
using Thermodynamics: Thermodynamics as TD

# When loaded from model_interface.jl, utils.jl is already included.
isdefined(@__MODULE__, :SOCRATESCase) ||
    include(joinpath(@__DIR__, "..", "model_interface", "utils.jl"))

const FT = Float64

const COLUMN_KEYS = (
    :dTdt_hadv,
    :dqtdt_hadv,
    :T_nudge,
    :qt_nudge,
    :subsidence,
    :u_nudge,
    :v_nudge,
)

"""Sample SSCF level interpolants onto a (z, time) matrix."""
function sample_column(itps, z, times)
    nz, nt = length(z), length(times)
    length(itps) == nz || error("interpolant length $(length(itps)) ≠ nz=$nz")
    out = Matrix{FT}(undef, nz, nt)
    for k in 1:nz
        for (j, t) in enumerate(times)
            out[k, j] = FT(itps[k](t))
        end
    end
    return out
end

function sample_surface(itp, times)
    return FT[FT(itp(t)) for t in times]
end

function insolation_series(times_dt, lat, lon)
    coszen = Vector{FT}(undef, length(times_dt))
    rsdt = Vector{FT}(undef, length(times_dt))
    params = Insolation.Parameters.InsolationParameters(FT)
    for (i, date) in enumerate(times_dt)
        F, _, μ, _ = Insolation.insolation(date, FT(lat), FT(lon), params)
        coszen[i] = μ
        rsdt[i] = F
    end
    return coszen, rsdt
end

"""
    generate_socrates_forcing(case; dt_sec=300.0)

Build in-memory ClimaAtmos forcing arrays for `case`.

Simulation time axis is `1970-01-01 + t_rel` (matches `start_date_string`).
SSCF profiles and insolation use the true LES wall clock.
"""
function generate_socrates_forcing(case::SOCRATESCase; dt_sec::Float64 = 300.0, z::ZT = nothing) where {ZT <: Union{Nothing, AbstractVector}}
    validate_case!(case)

    ft = sscf_forcing(case.forcing_type)
    flight = case.flight_number
    tp = TD.Parameters.ThermodynamicsParameters(CP.create_toml_dict(FT))

    inp = SSCF.open_atlas_les_input(flight, ft)
    lat = FT(Array(inp.data["lat"])[1])
    lon = FT(Array(inp.data["lon"])[1])
    lon = mod(lon + 180, 360) - 180

    if isnothing(z)
        z = collect(FT, SSCF.default_new_z(flight))
    end
    t_end = t_end_seconds(case)
    t_rel = collect(0.0:dt_sec:t_end)
    t0_les = start_datetime(case)
    t0_epoch = start_datetime_epoch(case)
    times_dt = [t0_epoch + Dates.Second(round(Int, t)) for t in t_rel]
    times_dt_insol = [t0_les + Dates.Second(round(Int, t)) for t in t_rel]

    forcing = SSCF.get_column_forcing(
        flight,
        ft,
        COLUMN_KEYS;
        new_z = z,
        thermodynamics_backend = tp,
    )
    surf = SSCF.get_surface_forcing(flight, ft; thermodynamics_backend = tp)

    ta = sample_column(forcing.T_nudge, z, t_rel)
    hus = sample_column(forcing.qt_nudge, z, t_rel)
    ua = sample_column(forcing.u_nudge, z, t_rel)
    va = sample_column(forcing.v_nudge, z, t_rel)
    wa = sample_column(forcing.subsidence, z, t_rel)
    tntha = sample_column(forcing.dTdt_hadv, z, t_rel)
    tnhusha = sample_column(forcing.dqtdt_hadv, z, t_rel)
    tntva = zeros(FT, size(ta))
    tnhusva = zeros(FT, size(hus))

    les = SSCF.open_atlas_les_output(flight, ft)
    z_les = FT.(vec(Array(les.data["z"])))
    ρ_les = FT.(Array(les.data["RHO"])[:, 1])
    ρ_on_z = similar(z)
    for (k, zk) in enumerate(z)
        ρ_on_z[k] = ρ_les[argmin(abs.(z_les .- zk))]
    end
    rho = repeat(reshape(ρ_on_z, :, 1), 1, length(t_rel))

    ts = sample_surface(surf.Tsfc, t_rel)
    coszen, rsdt = insolation_series(times_dt_insol, lat, lon)

    column_vars = Dict(
        "ta" => ta,
        "hus" => hus,
        "ua" => ua,
        "va" => va,
        "wa" => wa,
        "rho" => rho,
        "tntha" => tntha,
        "tnhusha" => tnhusha,
        "tntva" => tntva,
        "tnhusva" => tnhusva,
    )
    surface_vars = Dict("ts" => ts, "coszen" => coszen, "rsdt" => rsdt)

    return (;
        path = climacolumn_path(case),
        z,
        times_dt,
        lat,
        lon,
        column_vars,
        surface_vars,
    )
end

"""
    write_socrates_forcing(forcing; overwrite=false)

Optional debug dump of ClimaColumn NetCDF. Not used by the calibration driver.
"""
function write_socrates_forcing(forcing::NamedTuple; overwrite::Bool = false)
    out = forcing.path
    if !overwrite && isfile(out) &&
       ClimaAtmos.ColumnDatasets.ClimaColumnFiles.is_conforming(out)
        @info "Reusing conforming ClimaColumn" out
        return out
    end
    mkpath(dirname(out))
    ClimaAtmos.ColumnDatasets.ClimaColumnFiles.write_column_forcing_file(
        out,
        FT;
        z = forcing.z,
        time = forcing.times_dt,
        time_attrib = [
            "units" => "seconds since 1970-01-01T00:00:00",
            "calendar" => "proleptic_gregorian",
        ],
        forcing.column_vars,
        forcing.surface_vars,
        site_latitude = forcing.lat,
        site_longitude = forcing.lon,
    )
    @info "Wrote ClimaColumn (debug)" out
    return out
end

function write_socrates_climacolumn(
    case::SOCRATESCase;
    overwrite::Bool = false,
    dt_sec::Float64 = 300.0,
)
    forcing = generate_socrates_forcing(case; dt_sec)
    return write_socrates_forcing(forcing; overwrite)
end

function generate_all(; overwrite = false, names = nothing)
    cases = load_cases()
    if !isnothing(names)
        cases = [case_by_name(n, cases) for n in names]
    end
    paths = String[]
    for case in cases
        push!(paths, write_socrates_climacolumn(case; overwrite))
    end
    return paths
end

if abspath(PROGRAM_FILE) == @__FILE__
    overwrite = "--overwrite" in ARGS
    names = filter(a -> !startswith(a, "-"), ARGS)
    generate_all(; overwrite, names = isempty(names) ? nothing : names)
end
