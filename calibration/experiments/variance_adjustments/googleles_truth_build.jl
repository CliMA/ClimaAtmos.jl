# Included from `les_truth_build.jl` after `experiment_common.jl`.
# Reads CloudBench `data.zarr` (Swirl-LM); see MLCD `docs/googleles_zarr_layout.md` for axis order.

import YAML
import Statistics
import Interpolations
import Zarr
import Thermodynamics as TD
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP

"""Swirl-LM linear liquid fraction (see MLCD `GoogleLES.jl`)."""
@inline function _va_googleles_liquid_frac(T::FT) where {FT <: AbstractFloat}
    t_icenuc = FT(233.0)
    t_freeze = FT(273.15)
    f_l = (T - t_icenuc) / (t_freeze - t_icenuc)
    return clamp(f_l, FT(0), FT(1))
end

@inline function _va_googleles_partition_qc(q_c::FT, T::FT) where {FT}
    f = _va_googleles_liquid_frac(T)
    q_l = q_c * f
    q_i = q_c * (FT(1) - f)
    return q_l, q_i
end

function _va_googleles_zarr_open(site_id::Int, month::Int, experiment::AbstractString)
    # YAML `experiment` (e.g. `amip` or `amip_p4k`) maps to the GCS path segment (`amip`, `amip-p4k`): underscores → hyphens.
    seg = replace(string(experiment), '_' => '-')
    root = strip(get(ENV, "VA_GOOGLELES_ZARR_ROOT", ""))
    path = if isempty(root)
        "https://storage.googleapis.com/cloudbench-simulation-output/" *
        join([string(site_id), string(month), seg, "data.zarr"], '/')
    else
        joinpath(root, string(site_id), string(month), seg, "data.zarr")
    end
    @info "Opening GoogleLES Zarr" path
    try
        return Zarr.zopen(path; consolidated = true)
    catch
        return Zarr.zopen(path)
    end
end

"""Horizontal mean over Julia dims 2 and 3 (y, x) for `(z,y,x,t)` storage."""
function _va_googleles_horiz_mean_slice(
    zg,
    varname::AbstractString,
    it::Int,
)::Vector{Float64}
    v = zg[varname]
    a = Array(v[:, :, :, it])
    m = dropdims(Statistics.mean(a, dims = (2, 3)), dims = (2, 3))
    return vec(m)
end

function _va_googleles_time_indices(t_vec::Vector{Float64}, ti::Float64, tf::Float64)
    dt = length(t_vec) > 1 ? Statistics.mean(diff(t_vec)) : 0.0
    if isempty(t_vec)
        return 1:0
    end
    t_lo, t_hi = extrema(t_vec)
    ti_c = clamp(ti, t_lo, t_hi)
    tf_c = clamp(tf, t_lo, t_hi)
    if tf_c < t_lo || ti_c > t_hi
        return 1:0
    end
    _, ti_i = findmin(abs.(t_vec .- ti_c))
    _, tf_i = findmin(abs.(t_vec .- tf_c))
    lo = min(ti_i, tf_i)
    hi = max(ti_i, tf_i)
    if t_vec[end] < tf_c - dt
        return 1:0
    end
    return lo:hi
end

function _va_googleles_interp_z(z_src::Vector{Float64}, prof::Vector{Float64}, z_tgt::Vector{Float64})
    if length(z_src) < 2
        return fill(prof[1], length(z_tgt))
    end
    order = sortperm(z_src)
    zs = z_src[order]
    ps = prof[order]
    spl = Interpolations.extrapolate(
        Interpolations.interpolate((zs,), ps, Interpolations.Gridded(Interpolations.Linear())),
        Interpolations.Flat(),
    )
    return [spl(z) for z in z_tgt]
end

function _va_googleles_thermo_params(experiment_dir::AbstractString, expc)
    cfg = va_load_merged_case_yaml_dict(experiment_dir, expc["model_config_path"])
    atmos_config = CA.AtmosConfig(cfg)
    params = CA.ClimaAtmosParameters(atmos_config)
    return CAP.thermodynamics_params(params)
end

"""
    va_build_y_vector_from_googleles_zarr(experiment_dir, expc) -> Vector{Float64}

Build observation vector from CloudBench `data.zarr` using horizontal domain means and
`Thermodynamics.potential_temperature` for `thetaa` (same definition as Clima diagnostics).
"""
function va_build_y_vector_from_googleles_zarr(
    experiment_dir::AbstractString,
    expc,
)
    lt = get(expc, "les_truth", Dict())
    site_id = Int(lt["site_id"])
    month = Int(lt["month"])
    experiment = string(lt["experiment"])
    ti = Float64(get(lt, "y_t_start_sec", 475200.0))
    tf = Float64(get(lt, "y_t_end_sec", 518400.0))

    zg = _va_googleles_zarr_open(site_id, month, experiment)
    t_arr = vec(Array(zg["t"]))
    tr = _va_googleles_time_indices(t_arr, ti, tf)
    isempty(tr) && error(
        "GoogleLES time window [$ti, $tf] s is outside zarr `t` range " *
            "[$(t_arr[1]), $(t_arr[end])]] or simulation too short.",
    )

    thermo_params = _va_googleles_thermo_params(experiment_dir, expc)
    z_scm = va_z_centers_column(experiment_dir, expc)
    specs = va_field_specs(expc)

    # Accumulate time-mean profiles on LES z-levels
    zcoord = zg["z"]
    za = Array(zcoord)
    z_les = ndims(za) == 1 ? vec(za) : vec(za[:, 1])
    nz = length(z_les)

    n_t = length(tr)
    acc_theta = zeros(nz)
    acc_hus = zeros(nz)
    acc_clw = zeros(nz)
    acc_cli = zeros(nz)
    acc_clw_cli = zeros(nz)

    for it in tr
        T_m = _va_googleles_horiz_mean_slice(zg, "T", it)
        rho_m = _va_googleles_horiz_mean_slice(zg, "rho", it)
        qt_m = _va_googleles_horiz_mean_slice(zg, "q_t", it)
        qc_m = _va_googleles_horiz_mean_slice(zg, "q_c", it)
        length(T_m) == nz || error("GoogleLES vertical length mismatch for T")
        @inbounds for iz in 1:nz
            q_l, q_i = _va_googleles_partition_qc(qc_m[iz], T_m[iz])
            θ = TD.potential_temperature(
                thermo_params,
                T_m[iz],
                rho_m[iz],
                qt_m[iz],
                q_l,
                q_i,
            )
            acc_theta[iz] += θ
            acc_hus[iz] += qt_m[iz]
            acc_clw[iz] += q_l
            acc_cli[iz] += q_i
            acc_clw_cli[iz] += q_l + q_i
        end
    end
    acc_theta ./= n_t
    acc_hus ./= n_t
    acc_clw ./= n_t
    acc_cli ./= n_t
    acc_clw_cli ./= n_t

    y = Float64[]
    for spec in specs
        sn = string(spec["short_name"])
        prof =
            if sn == "thetaa"
                _va_googleles_interp_z(z_les, acc_theta, z_scm)
            elseif sn == "hus"
                _va_googleles_interp_z(z_les, acc_hus, z_scm)
            elseif sn == "clw"
                _va_googleles_interp_z(z_les, acc_clw, z_scm)
            elseif sn == "cli"
                _va_googleles_interp_z(z_les, acc_cli, z_scm)
            elseif sn == "clw_plus_cli"
                _va_googleles_interp_z(z_les, acc_clw_cli, z_scm)
            else
                error(
                    "GoogleLES observations: unsupported short_name $(repr(sn)). " *
                        "Use thetaa, hus, clw, cli, or clw_plus_cli.",
                )
            end
        append!(y, prof)
    end
    expected = va_expected_obs_length(experiment_dir, expc)
    length(y) == expected || error(
        "GoogleLES observation length $(length(y)) != expected $expected (check observation_fields vs column z_elem).",
    )
    return y
end
