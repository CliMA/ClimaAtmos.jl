"""
    VaranalFiles

Converter from the ARM VARANAL (Variational Analysis) product to the native
ClimaColumn schema. VARANAL files store the column state and forcing tendencies
on pressure levels, with non-CMIP names, mixed units (K/hr, g/kg, hPa/hr, degC),
`-9999` fill values, and a `base_time` time axis that the ClimaColumn reader does
not consume directly.

[`to_climacolumn`](@ref) reads a VARANAL file once and writes an equivalent
ClimaColumn file via `ClimaColumnFiles.write_column_forcing_file`, which the
standard `ColumnDataset` path then reads like any other ClimaColumn
file. Same pattern as the ERA5 generator: one converter per source.
"""
module VaranalFiles

import NCDatasets as NC
import Dates
import Statistics: mean
import Thermodynamics.Parameters as TDP

import ..ClimaColumnFiles: write_column_forcing_file, is_conforming

# Missing-data fill value (-9999) used throughout the ARM VARANAL product.
const FILL_VALUE = -9999.0
fill_to_nan(x) = x == FILL_VALUE ? NaN : Float64(x)

# VARANAL products may store longitude in [0, 360], the ClimaColumn schema
# (and its validator) require [-180, 180].
to_pm180(lon) = mod(lon + 180.0, 360.0) - 180.0

"""
    to_climacolumn(path; thermo_params, dir = dirname(path), overwrite = false)

Read the ARM VARANAL file `path` and write an equivalent ClimaColumn file into
`dir`, returning the written path. `dir` defaults to the source file's
directory, so the converted file persists next to the source and is reused
across runs; a conforming file already at the target path is reused unless
`overwrite = true`. Pass a writable `dir` if the source directory is read-only.

`thermo_params` supplies the physical constants (g, R_d, R_v) used to map
pressure levels to geometric height, convert `omega` to a subsidence velocity,
and derive density. The written file carries the canonical `(z, time)` column
variables (`ta`, `hus`, `ua`, `va`, `wa`, `rho`, `tntha`, `tnhusha`), the
`(time,)` surface variables (`ts`, plus `hfls`/`hfss` when present), and the
`site_latitude`/`site_longitude` global attributes.

VARANAL's vertical-advection tendencies (`T_adv_v`, `q_adv_v`) are not carried
over. Vertical transport comes from the `Subsidence` term acting on the model's
evolving profiles.
"""
function to_climacolumn(path; thermo_params, dir = dirname(path), overwrite = false)
    out = joinpath(dir, canonical_name(path))
    if !overwrite && isfile(out) && is_conforming(out)
        return out
    end

    NC.NCDataset(path, "r") do ds
        lev_hPa = Float64.(vec(ds["lev"][:]))
        nlev = length(lev_hPa)
        times = read_times(ds, path)
        ntime = length(times)

        # (lev, time) fields in the file's level order, fill values -> NaN.
        ta = read_lev_time(ds, "T", nlev, ntime)
        hus = read_lev_time(ds, "q", nlev, ntime) ./ 1000.0        # g/kg -> kg/kg
        ua = read_lev_time(ds, "u", nlev, ntime)
        va = read_lev_time(ds, "v", nlev, ntime)
        tntha = read_lev_time(ds, "T_adv_h", nlev, ntime) ./ 3600.0  # K/hr -> K/s
        tnhusha =
            read_lev_time(ds, "q_adv_h", nlev, ntime) ./ 1000.0 ./ 3600.0  # g/kg/hr -> kg/kg/s
        omega = read_lev_time(ds, "omega", nlev, ntime)              # hPa/hr

        # Height axis from the time-mean profile (one grid for all times),
        # reordered to ascending z.
        g = TDP.grav(thermo_params)
        R_d = TDP.R_d(thermo_params)
        R_v = TDP.R_v(thermo_params)
        T_mean = vec(mean(ta; dims = 2))
        q_mean = vec(mean(hus; dims = 2))
        z = pressure_to_height(lev_hPa .* 100.0, T_mean, q_mean, g, R_d, R_v)
        order = sortperm(z)

        p_col = lev_hPa .* 100.0                                      # Pa, per level
        Tv = ta .* (1.0 .+ (R_v / R_d - 1.0) .* hus)                  # (lev, time)
        rho = p_col ./ (R_d .* Tv)                                    # broadcast p over time
        wa = omega_to_w(omega, p_col, Tv, g, R_d)                    # hPa/hr -> m/s, upward

        reorder(A) = A[order, :]
        column_vars = Dict(
            "ta" => reorder(ta),
            "hus" => reorder(hus),
            "ua" => reorder(ua),
            "va" => reorder(va),
            "wa" => reorder(wa),
            "rho" => reorder(rho),
            "tntha" => reorder(tntha),
            "tnhusha" => reorder(tnhusha),
        )

        surface_vars = Dict{String, Vector{Float64}}("ts" => read_ts(ds, ntime))
        haskey(ds, "LH") && (surface_vars["hfls"] = read_series(ds, "LH", ntime))
        haskey(ds, "SH") && (surface_vars["hfss"] = read_series(ds, "SH", ntime))

        write_column_forcing_file(
            out,
            Float64;
            z = z[order],
            time = times,
            time_attrib = [
                "units" => "seconds since 1970-01-01T00:00:00",
                "calendar" => "proleptic_gregorian",
            ],
            column_vars,
            surface_vars,
            site_latitude = Float64(first(ds["lat"][:])),
            site_longitude = to_pm180(Float64(first(ds["lon"][:]))),
        )
    end
    return out
end

canonical_name(path) =
    replace(basename(path), r"\.(nc|cdf)$"i => "") * "_climacolumn.nc"

# The time axis as DateTimes: a decoded DateTime coordinate, or seconds offset
# from the `base_time` (unix) attribute.
function read_times(ds, path)
    time_raw = vec(ds["time"][:])
    eltype(time_raw) <: Dates.AbstractDateTime && return collect(time_raw)
    haskey(ds.attrib, "base_time") || error(
        "ARM VARANAL file $path has a numeric time axis but no `base_time` \
         attribute; cannot build a calendar time axis.",
    )
    base = Dates.unix2datetime(ds.attrib["base_time"])
    return [base + Dates.Millisecond(round(Int, 1000 * Float64(t))) for t in time_raw]
end

# A (lev, time) field in the file's level order, fill values replaced with NaN.
function read_lev_time(ds, name, nlev, ntime)
    haskey(ds, name) ||
        error("ARM VARANAL file is missing required column variable `$name`")
    raw = Array(ds[name])
    ndims(raw) == 2 || error("`$name` must be a 2D (lev, time) field")
    nfill = count(==(FILL_VALUE), raw)
    nfill == 0 || @warn "ARM VARANAL `$name` has $nfill fill values ($FILL_VALUE) \
                         converted to NaN; may propagate into the forcing/initial state."
    data = fill_to_nan.(raw)
    size(data) == (nlev, ntime) && return data
    size(data) == (ntime, nlev) && return permutedims(data)
    error(
        "`$name` has shape $(size(data)); expected (lev=$nlev, time=$ntime) \
         in some order.",
    )
end

# A (time,) surface series, fill values replaced with NaN.
function read_series(ds, name, ntime)
    raw = vec(ds[name][:])
    nfill = count(==(FILL_VALUE), raw)
    nfill == 0 || @warn "ARM VARANAL `$name` has $nfill fill values ($FILL_VALUE) \
                         converted to NaN; may propagate into the forcing/initial state."
    data = fill_to_nan.(raw)
    length(data) == ntime ||
        error("surface variable `$name` has length $(length(data)); expected $ntime")
    return data
end

# Surface skin temperature (degC in the file) -> K, fill values -> NaN.
function read_ts(ds, ntime)
    name =
        haskey(ds, "T_skin") ? "T_skin" :
        haskey(ds, "T_srf") ? "T_srf" :
        error("ARM VARANAL file has neither `T_skin` nor `T_srf` for surface temperature")
    return read_series(ds, name, ntime) .+ 273.15
end

# Hypsometric integration: pressure [Pa] -> geometric height [m], z(surface) = 0.
function pressure_to_height(p_pa, T, q_kgkg, g, R_d, R_v)
    idx = sortperm(p_pa; rev = true)  # surface (highest pressure) first
    ps, Ts, qs = p_pa[idx], T[idx], q_kgkg[idx]
    Tv = Ts .* (1.0 .+ (R_v / R_d - 1.0) .* qs)
    z = zeros(length(ps))
    for i in 2:length(ps)
        Tv_mean = 0.5 * (Tv[i - 1] + Tv[i])
        z[i] = z[i - 1] + R_d * Tv_mean / g * log(ps[i - 1] / ps[i])
    end
    return z[invperm(idx)]
end

# Pressure vertical velocity omega [hPa/hr] -> geometric velocity w [m/s]
# (positive upward), hydrostatic: w = -omega / (rho g), rho = p / (R_d Tv).
function omega_to_w(omega_hPa_hr, p_pa, Tv, g, R_d)
    omega_Pa_s = omega_hPa_hr .* 100.0 ./ 3600.0
    rho = p_pa ./ (R_d .* Tv)
    return .-omega_Pa_s ./ (rho .* g)
end

end # module
