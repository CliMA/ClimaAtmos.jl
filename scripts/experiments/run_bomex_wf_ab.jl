using ClimaAtmos: ClimaAtmos as CA
using CairoMakie: CairoMakie
using Statistics: mean


# Diagnostics: instantaneous at 600 s for CF, cloud liquid, and height
DIAGS = CA.Diagnostics.DiagnosticsConfig(;
    default = false,
    additional = [
        (; short_name = "cl",  period = "600s", reduction = "inst", writer = "netcdf", output_name = "cl_inst_600s"),
        (; short_name = "clw", period = "600s", reduction = "inst", writer = "netcdf", output_name = "clw_inst_600s"),
        (; short_name = "zg",  period = "600s", reduction = "inst", writer = "netcdf", output_name = "zg_inst_600s"),
    ],
)

function make_setup(FT::Type{<:AbstractFloat} = Float32; z_elem::Int = 60, z_max = (3000), xmax = (1e5), ymax = (1e5), dz_bottom = (500))
    params = CA.ClimaAtmosParameters(FT)
    grid = CA.BoxGrid(FT; x_elem=1, y_elem=1, nh_poly=1,
                      x_max=xmax, y_max=ymax,
                      z_elem=z_elem, z_max=z_max,
                      z_stretch=false, dz_bottom=dz_bottom,
                      bubble=false, periodic_x=true, periodic_y=true)
    setup = CA.Setups.Bomex(; prognostic_tke=true, thermo_params=params.thermodynamics_params)
    return (; params, grid, setup)
end

function plot_profiles_panels(out_on::String, out_off::String, base_out::AbstractString)
    pick(base, names) = begin
        for n in names
            p = netcdf_path(base, n)
            isfile(p) && return p
        end
        error("None of the expected NetCDF files exist in $(base): $(join(names, ", "))")
    end
    # Source files (prefer 10m, fallback 600s)
    f_on_cl   = pick(out_on,  ("cl_inst_10m",  "cl_inst_600s"))
    f_on_clw  = pick(out_on,  ("clw_inst_10m", "clw_inst_600s"))
    f_on_zg   = pick(out_on,  ("zg_inst_10m",  "zg_inst_600s"))
    f_on_tke  = pick(out_on,  ("tke_inst_10m",))
    f_on_lmix = pick(out_on,  ("lmix_inst_10m",))
    f_on_p    = pick(out_on,  ("pfull_inst_10m",))
    f_on_rho  = pick(out_on,  ("rhoa_inst_10m",))

    f_off_cl   = pick(out_off, ("cl_inst_10m",  "cl_inst_600s"))
    f_off_clw  = pick(out_off, ("clw_inst_10m", "clw_inst_600s"))
    f_off_zg   = pick(out_off, ("zg_inst_10m",  "zg_inst_600s"))
    f_off_tke  = pick(out_off, ("tke_inst_10m",))
    f_off_lmix = pick(out_off, ("lmix_inst_10m",))
    f_off_p    = pick(out_off, ("pfull_inst_10m",))
    f_off_rho  = pick(out_off, ("rhoa_inst_10m",))

    # Load profiles
    z_on  = profile_from(f_on_zg,  "zg")
    z_off = profile_from(f_off_zg, "zg")
    cl_on,  cl_off   = profile_from(f_on_cl,  "cl"),   profile_from(f_off_cl,  "cl")
    clw_on, clw_off  = profile_from(f_on_clw, "clw"),  profile_from(f_off_clw, "clw")
    tke_on, tke_off  = profile_from(f_on_tke, "tke"),  profile_from(f_off_tke, "tke")
    lmx_on, lmx_off  = profile_from(f_on_lmix, "lmix"), profile_from(f_off_lmix, "lmix")
    p_on,   p_off    = profile_from(f_on_p,   "pfull"), profile_from(f_off_p,   "pfull")
    rho_on, rho_off  = profile_from(f_on_rho, "rhoa"),  profile_from(f_off_rho, "rhoa")

    outpng = joinpath(base_out, "bomex_ab_panels.png")
    CairoMakie.activate!(type = "png")
    f = CairoMakie.Figure(size = (1000, 1200))
    # Row 1: cl, clw
    ax11 = CairoMakie.Axis(f[1, 1], xlabel = "Cloud fraction", ylabel = "z (m)")
    CairoMakie.lines!(ax11, cl_on,  z_on,  color = :blue,   label = "WF ON")
    CairoMakie.lines!(ax11, cl_off, z_off, color = :orange, linestyle = :dash, label = "WF OFF")
    CairoMakie.axislegend(ax11)
    ax12 = CairoMakie.Axis(f[1, 2], xlabel = "q_liq (kg/kg)")
    CairoMakie.lines!(ax12, clw_on,  z_on,  color = :blue)
    CairoMakie.lines!(ax12, clw_off, z_off, color = :orange, linestyle = :dash)
    # Row 2: tke, lmix
    ax21 = CairoMakie.Axis(f[2, 1], xlabel = "tke (m^2/s^2)", ylabel = "z (m)")
    CairoMakie.lines!(ax21, tke_on,  z_on,  color = :blue)
    CairoMakie.lines!(ax21, tke_off, z_off, color = :orange, linestyle = :dash)
    ax22 = CairoMakie.Axis(f[2, 2], xlabel = "lmix (m)")
    CairoMakie.lines!(ax22, lmx_on,  z_on,  color = :blue)
    CairoMakie.lines!(ax22, lmx_off, z_off, color = :orange, linestyle = :dash)
    # Row 3: pfull, rhoa
    ax31 = CairoMakie.Axis(f[3, 1], xlabel = "pfull (Pa)", ylabel = "z (m)")
    CairoMakie.lines!(ax31, p_on,  z_on,  color = :blue)
    CairoMakie.lines!(ax31, p_off, z_off, color = :orange, linestyle = :dash)
    ax32 = CairoMakie.Axis(f[3, 2], xlabel = "rhoa (kg/m^3)")
    CairoMakie.lines!(ax32, rho_on,  z_on,  color = :blue)
    CairoMakie.lines!(ax32, rho_off, z_off, color = :orange, linestyle = :dash)

    thrs = last_time_hours(f_on_cl)
    title = isnothing(thrs) ? "BOMEX NEQ 1M" : "BOMEX NEQ 1M — last t ≈ $(round(thrs; digits=2)) h"
    CairoMakie.Label(f[0, 1:2], title)
    CairoMakie.save(outpng, f)
    println("Saved panel plot: ", outpng)
    return outpng
end

# YAML-driven equivalent using the canonical diagnostic_edmfx_bomex_box.yml
# Mirrors stability knobs (implicit diffusion, EDMF closures, etc.), while
# overriding microphysics to 1M and enabling SGS quadrature. dt/t_end follow
# the YAML (dt=100s) unless we override below.
function run_case_yaml(
    job_id::AbstractString, α::FT2, base_out::AbstractString;
    dt::FT2 = 100.,
    t_end_secs::FT2 = FT2(3600. * 12),
    z_elem::Int = 60,
    z_max::FT2 = FT2(3000),
    z_stretch::Bool = false,
    xmax::FT2 = FT2(1e5),
    ymax::FT2 = FT2(1e5),
    turbconv::Symbol = :diagnostic_edmfx,
    edmfx_vertical_diffusion::Bool = true,
) where {FT2}
    outdir = joinpath(base_out, job_id)
    Base.Filesystem.mkpath(outdir)
    cfg_file = joinpath(@__DIR__, "..", "..", "config", "model_configs", "diagnostic_edmfx_bomex_box.yml")
    yaml_cfg = CA.YAML.load_file(cfg_file)
    # Build overrides: 1M microphysics + SGS quadrature and our output_dir/job_id.
    overrides = Dict(
        "job_id" => job_id,
        "output_dir" => outdir,
        "microphysics_model" => "1M",
        "dt" => string(dt) * "secs",
        # Use canonical BOMEX box tiling (2x2) for stability of horizontal operators
        "x_elem" => 2,
        "y_elem" => 2,
        "z_stretch" => z_stretch,
        "z_elem" => z_elem,
        "z_max" => z_max,
        "xmax" => xmax,
        "ymax" => ymax,
        "turbconv" => string(turbconv),
        "tracer_upwinding" => "first_order",
        "implicit_diffusion" => true,
        "approximate_linear_solve_iters" => 2,
        "max_newton_iters_ode_subproblem" => 3,
        "max_newton_iters_ode" => 1,
        "edmfx_entr_model" => "PiGroups",
        "edmfx_detr_model" => "SmoothArea",
        "edmfx_sgs_mass_flux" => true,
        "edmfx_sgs_diffusive_flux" => true,
        "edmfx_nh_pressure" => true,
        "edmfx_vertical_diffusion" => edmfx_vertical_diffusion,
        "edmfx_filter" => false,
        "prognostic_tke" => true,
        "cloud_model" => "quadrature",
        "use_sgs_quadrature" => true,
        "sgs_distribution" => "gaussian",
        "quadrature_order" => 3,
        # Water-filling strength (0 = partition off, 1 = full); must match run_case_yaml `α` arg
        "water_filling_max_alpha" => α,
        # Disable state checkpointing to HDF5 during short experiments
        "dt_save_state_to_disk" => "Inf",
        # Write NetCDF on a single horizontal point to avoid 8x8 dims in files
        "netcdf_interpolation_num_points" => [1, 1, 60],
        # Disable default (broad) diagnostics and specify a minimal explicit set
        "output_default_diagnostics" => false,
        "enable_diagnostics" => true,
        "diagnostics" => Dict{String, String}[
            Dict("short_name" => "cl",  "period" => "10mins", "reduction" => "inst", "writer" => "netcdf", "output_name" => "cl_inst_10m"),
            Dict("short_name" => "clw", "period" => "10mins", "reduction" => "inst", "writer" => "netcdf", "output_name" => "clw_inst_10m"),
            Dict("short_name" => "cli", "period" => "10mins", "reduction" => "inst", "writer" => "netcdf", "output_name" => "cli_inst_10m"),
            Dict("short_name" => "ta",  "period" => "10mins", "reduction" => "inst", "writer" => "netcdf", "output_name" => "ta_inst_10m"),
            Dict("short_name" => "hus", "period" => "10mins", "reduction" => "inst", "writer" => "netcdf", "output_name" => "hus_inst_10m"),
            Dict("short_name" => "zg",  "period" => "10mins", "reduction" => "inst", "writer" => "netcdf", "output_name" => "zg_inst_10m"),
            Dict("short_name" => "pfull",  "period" => "10mins", "reduction" => "inst", "writer" => "netcdf", "output_name" => "pfull_inst_10m"),
            Dict("short_name" => "rhoa",   "period" => "10mins", "reduction" => "inst", "writer" => "netcdf", "output_name" => "rhoa_inst_10m"),
            Dict("short_name" => "tke",   "period" => "10mins", "reduction" => "inst", "writer" => "netcdf", "output_name" => "tke_inst_10m"),
            Dict("short_name" => "lmix",  "period" => "10mins", "reduction" => "inst", "writer" => "netcdf", "output_name" => "lmix_inst_10m"),
        ],
    )
    # Respect requested t_end if provided explicitly
    if !isnothing(t_end_secs)
        overrides["t_end"] = string(round(Int, t_end_secs)) * "secs"
    end


    # Construct config from YAML + overrides
    config = CA.AtmosConfig((yaml_cfg, overrides); job_id = job_id, config_files = (cfg_file,))
    sim = CA.get_simulation(config)
    res = CA.solve_atmos!(sim)
    println("DONE job_id=$(job_id) t_end=", res.sol === nothing ? "<crashed>" : res.sol.t[end], " output_dir=", sim.output_dir)
    return String(sim.output_dir)
end

function run_case(
    job_id::AbstractString, α::FT2, base_out::AbstractString, commons,
    ; FT::Type{<:AbstractFloat} = Float32,
      dt::FT3 = FT3(100.),
      t_end_secs::FT3 = FT3(3600. * 12),
      z_elem::Int = 60,
      z_max::FT2 = FT2(3000),
      xmax::FT2 = FT2(1e5),
      ymax::FT2 = FT2(1e5),
      dz_bottom::FT2 = FT2(500),
      diagnostics::CA.Diagnostics.DiagnosticsConfig = DIAGS,
      edmf::Symbol = :diagnostic,
) where {FT2, FT3}
    (; params, grid, setup) = commons
    quad = CA.SGSQuadrature(FT; quadrature_order=3, distribution=CA.GaussianSGS(), α_max=FT(α))
    model = if edmf === :diagnostic
        CA.Presets.diagnostic_edmf(FT;
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            microphysics_tendency_timestepping = CA.Explicit(), # is this right?
            sgs_quadrature = quad,
        )
    elseif edmf === :prognostic
        CA.Presets.prognostic_edmf_1m(FT; sgs_quadrature = quad)
    else
        error("Unsupported edmf=$(edmf); use :diagnostic or :prognostic")
    end
    outdir = joinpath(base_out, job_id)
    Base.Filesystem.mkpath(outdir)
    # IMEX ARS343 with modest Newton iterations, matching SCM config defaults
    ode = CA.CTS.IMEXAlgorithm(
        CA.CTS.ARS343(),
        CA.CTS.NewtonsMethod(; max_iters = 2, update_j = CA.CTS.UpdateEvery(CA.CTS.NewNewtonIteration)),
    )
    sim = CA.AtmosSimulation{FT}(;
        job_id,
        grid,
        setup,
        model,
        params,
        dt = dt,
        t_end = t_end_secs,
        z_elem = z_elem,
        z_max = z_max,
        xmax = xmax,
        ymax = ymax,
        dz_bottom = dz_bottom,
        ode_config = ode,
        log_to_file = true,
        diagnostics = diagnostics,
        output_dir = outdir,
    )
    res = CA.solve_atmos!(sim)
    println("DONE job_id=$(job_id) t_end=", res.sol.t[end], " output_dir=", sim.output_dir)
    # Return the concrete diagnostics directory (e.g., .../output_0000)
    return String(sim.output_dir)
end

netcdf_path(dir::AbstractString, base::AbstractString) = joinpath(dir, string(base, ".nc"))

last_time_hours(nc_path::AbstractString) = begin
    ds = CA.NC.Dataset(nc_path, "r")
    hrs = nothing
    if haskey(ds, "time")
        t = ds["time"][:]
        if !isempty(t)
            hrs = float(t[end]) / 3600
        end
    end
    close(ds)
    return hrs
end

"""
    collapse_profile_from_var(var)

Deterministically extract a 1D vertical profile from a NetCDF variable by:
- identifying time and vertical axes by name ("time" and either "z" or "lev"),
- slicing the last time index if present,
- moving the vertical axis first, and averaging across any remaining axes.
"""
function collapse_profile_from_var(var)
    timelike = ("time",)
    zlike = ("z", "lev")
    dn = try
        CA.NC.dimnames(var)
    catch
        nothing
    end
    dn === nothing && error("NetCDF variable lacks dimension names; cannot extract profile deterministically")
    nd = length(dn)
    ti = findfirst(n -> n in timelike, dn)
    zi = findfirst(n -> n in zlike, dn)
    zi === nothing && error("No vertical axis among $(dn) for variable")
    # Build index tuple to slice last time if needed
    idx = ntuple(i -> i == ti ? size(var, i) : Colon(), nd)
    A = var[idx...]
    # After slicing, recompute vertical axis index in the reduced array
    zi_new = zi
    if ti !== nothing
        ti == zi && error("Variable uses time as its only vertical axis; cannot extract z profile")
        zi_new = zi - (ti < zi ? 1 : 0)
    end
    ndA = ndims(A)
    ndA == 0 && error("Variable reduced to scalar after time slicing; missing vertical dimension")
    if ndA == 1
        @assert zi_new == 1
        return vec(A)
    else
        perm = (zi_new, Tuple(i for i in 1:ndA if i != zi_new)...)
        Ap = PermutedDimsArray(A, perm)
        M = reshape(Ap, size(Ap, 1), :)
        return size(M, 2) == 1 ? vec(M) : vec(mean(M; dims = 2))
    end
end

profile_from(nc_path::AbstractString, vname::AbstractString) = begin
    ds = CA.NC.Dataset(nc_path, "r")
    var = ds[vname]
    prof = collapse_profile_from_var(var)
    close(ds)
    return prof
end

function load_profiles(nc_cl, nc_clw, nc_zg)
    ds1 = CA.NC.Dataset(nc_cl, "r");  v1 = ds1["cl"]
    ds2 = CA.NC.Dataset(nc_clw, "r"); v2 = ds2["clw"]
    ds3 = CA.NC.Dataset(nc_zg, "r");  v3 = ds3["zg"]

    # z from zg using its named vertical axis
    z = collapse_profile_from_var(v3)
    cl  = collapse_profile_from_var(v1)
    clw = collapse_profile_from_var(v2)
    close(ds1); close(ds2); close(ds3)
    return (cl = cl, clw = clw, z = z)
end

function plot_profiles(out_on::String, out_off::String, base_out::AbstractString)
    pick(base, names) = begin
        for n in names
            p = netcdf_path(base, n)
            isfile(p) && return p
        end
        error("None of the expected NetCDF files exist in $(base): $(join(names, ", "))")
    end
    # Prefer canonical 10m names; fall back to legacy 600s if present
    nc_on_cl  = pick(out_on,  ("cl_inst_10m",  "cl_inst_600s"))
    nc_on_clw = pick(out_on,  ("clw_inst_10m", "clw_inst_600s"))
    nc_on_zg  = pick(out_on,  ("zg_inst_10m",  "zg_inst_600s"))
    prof_on = load_profiles(nc_on_cl, nc_on_clw, nc_on_zg)
    prof_off = load_profiles(
        pick(out_off, ("cl_inst_10m",  "cl_inst_600s")),
        pick(out_off, ("clw_inst_10m", "clw_inst_600s")),
        pick(out_off, ("zg_inst_10m",  "zg_inst_600s")),
    )
    outpng = joinpath(base_out, "bomex_ab_profiles.png")

    CairoMakie.activate!(type = "png")
    f = CairoMakie.Figure(size = (800, 600))
    ax1 = CairoMakie.Axis(f[1, 1], xlabel = "Cloud fraction", ylabel = "z (m)")
    CairoMakie.lines!(ax1, prof_on.cl, prof_on.z, color = :blue, label = "WF ON")
    CairoMakie.lines!(ax1, prof_off.cl, prof_off.z, color = :orange, linestyle = :dash, label = "WF OFF")
    CairoMakie.axislegend(ax1)
    ax2 = CairoMakie.Axis(f[1, 2], xlabel = "q_liq (kg/kg)")
    CairoMakie.lines!(ax2, prof_on.clw, prof_on.z, color = :blue, label = "WF ON")
    CairoMakie.lines!(ax2, prof_off.clw, prof_off.z, color = :orange, linestyle = :dash, label = "WF OFF")
    thrs = last_time_hours(nc_on_cl)
    title = isnothing(thrs) ? "BOMEX NEQ 1M" : "BOMEX NEQ 1M — last t ≈ $(round(thrs; digits=2)) h"
    CairoMakie.Label(f[0, 1:2], title)
    CairoMakie.save(outpng, f)
    println("Saved plot: ", outpng)
    return outpng
end

function run_bomex_wf_ab(
    base_out::AbstractString; 
    t_end_secs::FT = 3600. * 12,
    dt::FT = 10.,
    turbconv::Symbol = :diagnostic_edmfx,
    edmfx_vertical_diffusion::Bool = true,
    ) where {FT}
    Base.Filesystem.mkpath(base_out)
    # Use YAML-driven run to mirror canonical diagnostic-EDMFX BOMEX (box) numerics
    out_on  = run_case_yaml("bomex_neq1m_wf_on",  1.0, base_out; t_end_secs=t_end_secs, turbconv=turbconv, edmfx_vertical_diffusion=edmfx_vertical_diffusion, dt = dt)
    out_off = run_case_yaml("bomex_neq1m_wf_off", 0.0, base_out; t_end_secs=t_end_secs, turbconv=turbconv, edmfx_vertical_diffusion=edmfx_vertical_diffusion, dt = dt)
    plot_profiles(out_on, out_off, base_out)
end


# Choose the concrete diagnostics directory for an existing case without rerunning
function latest_output_dir(case_dir::AbstractString)
    active = joinpath(case_dir, "output_active")
    ispath(active) && return String(active)
    names = readdir(case_dir)
    outs = filter(n -> occursin(r"^output_\d{4}$", n) && isdir(joinpath(case_dir, n)), names)
    isempty(outs) && error("No diagnostics directory found in $(case_dir)")
    latest = sort(outs; rev = true)[1]
    return joinpath(case_dir, latest)
end

function replot_bomex_wf_ab(base_out::AbstractString)
    on_case = joinpath(base_out, "bomex_neq1m_wf_on")
    off_case = joinpath(base_out, "bomex_neq1m_wf_off")
    on_dir = latest_output_dir(on_case)
    off_dir = latest_output_dir(off_case)
    plot_profiles(on_dir, off_dir, base_out)
end

function run_bomex_wf_ab_now()
    out = mktempdir()
    run_bomex_wf_ab(out)
end

