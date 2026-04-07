# Shen-style GCM forcing NetCDF for `GCMForcing` / `GCMDriven` from CloudBench `sounding.csv`.
# Phase A: zero large-scale tendencies; documented limitation (see README).

import CSV
import Downloads
import NCDatasets
import Statistics

"""HTTPS URL for `sounding.csv` (same directory layout as `data.zarr`)."""
function va_googleles_sounding_url(site_id::Int, month::Int, experiment::AbstractString)
    seg = replace(string(experiment), '_' => '-')
    return "https://storage.googleapis.com/cloudbench-simulation-output/" *
           join([string(site_id), string(month), seg, "sounding.csv"], '/')
end

function _va_read_cloudbench_sounding(path::AbstractString)
    f = CSV.File(path)
    z = collect(f.z)
    T = collect(f.temperature)
    q_t = collect(f.q_t)
    u = collect(f.u)
    v = collect(f.v)
    rho = collect(f.rho)
    return z, T, q_t, u, v, rho
end

"""
    va_write_googleles_shen_forcing_nc!(out_path, sounding_path, site_group; nt=48)

Write a minimal Shen-compatible NetCDF for `external_forcing_file` + `cfsite_number` = `site_group`.
Tendencies `tntha`, `tnhusha`, `tntva`, `tnhusva` and `wap` are set to **zero** (exploratory).
`rsdt` and `coszen` use placeholder constants.
"""
function va_write_googleles_shen_forcing_nc!(
    out_path::AbstractString,
    sounding_path::AbstractString,
    site_group::AbstractString;
    nt::Int = 48,
    rsdt_const::Float64 = 400.0,
    coszen_const::Float64 = 0.85,
)
    z, T, q_t, u, v, rho = _va_read_cloudbench_sounding(sounding_path)
    nz = length(z)
    nz >= 2 || error("sounding must have at least 2 levels")

    alpha = 1.0 ./ rho
    # Surface temperature: lowest model level (first row is z=0 in CloudBench sample)
    ts0 = T[1]

    mkpath(dirname(out_path))
    NCDatasets.NCDataset(out_path, "c") do ds
        g = NCDatasets.defGroup(ds, site_group)
        NCDatasets.defDim(g, "z", nz)
        NCDatasets.defDim(g, "time", nt)

        zg_a = zeros(nz, nt)
        ta_a = zeros(nz, nt)
        hus_a = zeros(nz, nt)
        ua_a = zeros(nz, nt)
        va_a = zeros(nz, nt)
        alpha_a = zeros(nz, nt)
        tn0 = zeros(nz, nt)
        wap_a = zeros(nz, nt)
        for j in 1:nt
            zg_a[:, j] .= z
            ta_a[:, j] .= T
            hus_a[:, j] .= q_t
            ua_a[:, j] .= u
            va_a[:, j] .= v
            alpha_a[:, j] .= alpha
        end

        for name in ("zg", "ta", "hus", "ua", "va", "alpha", "tntha", "tnhusha", "tntva", "tnhusva", "wap")
            data = name == "zg" ? zg_a :
                   name == "ta" ? ta_a :
                   name == "hus" ? hus_a :
                   name == "ua" ? ua_a :
                   name == "va" ? va_a :
                   name == "alpha" ? alpha_a :
                   name == "wap" ? wap_a : tn0
            NCDatasets.defVar(g, name, Float64, ("z", "time"))[:] = data
        end

        NCDatasets.defVar(g, "ts", Float64, ("time",))[:] = fill(ts0, nt)
        NCDatasets.defVar(g, "rsdt", Float64, ("time",))[:] = fill(rsdt_const, nt)
        NCDatasets.defVar(g, "coszen", Float64, ("time",))[:] = fill(coszen_const, nt)
    end
    @info "Wrote synthetic Shen-style GoogleLES forcing NetCDF" out_path site_group
    return out_path
end

"""Download `sounding.csv` once into `simulation_output/_googleles_cache/`."""
function va_ensure_googleles_sounding_local!(
    experiment_dir::AbstractString,
    site_id::Int,
    month::Int,
    experiment::AbstractString,
)
    cache = joinpath(
        experiment_dir,
        "simulation_output",
        "_googleles_cache",
        "$(site_id)_$(month)_$(replace(string(experiment), '/' => '_'))",
    )
    dest = joinpath(cache, "sounding.csv")
    isfile(dest) && return dest
    mkpath(cache)
    url = va_googleles_sounding_url(site_id, month, experiment)
    @info "Downloading CloudBench sounding" url
    Downloads.download(url, dest)
    return dest
end

"""Build forcing NetCDF next to reference output if missing; used for GoogleLES experiment YAMLs."""
function va_ensure_googleles_shen_forcing!(experiment_dir::AbstractString, expc)
    lt = get(expc, "les_truth", nothing)
    lt isa AbstractDict || return nothing
    string(get(lt, "source", "")) == "googleles_cloudbench" || return nothing

    site_id = Int(lt["site_id"])
    month = Int(lt["month"])
    experiment = string(lt["experiment"])
    site_group = string(get(lt, "shen_site_group", get(expc, "googleles_shen_site_group", "site_googleles")))

    out_rel = get(expc, "googleles_forcing_path", nothing)
    if out_rel === nothing || isempty(strip(string(out_rel)))
        ref = va_reference_output_dir(experiment_dir, expc)
        out_path = joinpath(ref, "googleles_shen_forcing.nc")
    else
        s = string(out_rel)
        out_path = isabspath(s) ? s : joinpath(experiment_dir, s)
    end
    isfile(out_path) && return out_path

    sounding_path = va_ensure_googleles_sounding_local!(experiment_dir, site_id, month, experiment)
    return va_write_googleles_shen_forcing_nc!(out_path, sounding_path, site_group)
end
