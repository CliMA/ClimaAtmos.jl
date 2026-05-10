# One-command offline quadrature workflow API (single Julia process).

using Dates
using JLD2: @load

using Pkg: Pkg
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import Thermodynamics as TD

Pkg.activate(joinpath(@__DIR__, "..", "..")) # activate ClimaAtmos.jl/calibration/experiments/variance_adjustments

# Load helper function APIs once in the current module.
if !isdefined(@__MODULE__, :run_highres_truth_profile!)
    include(joinpath(@__DIR__, "run_highres_truth.jl"))
end
if !isdefined(@__MODULE__, :ForwardSweepConfig)
    include(joinpath(@__DIR__, "..", "..", "scripts", "resolution_ladder.jl"))
    include(joinpath(@__DIR__, "..", "..", "lib", "forward_sweep_grid.jl"))
end
if !isdefined(@__MODULE__, :Distributed)
    import Distributed
end
if !isdefined(@__MODULE__, :ClimaCore)
    import ClimaCore
end
if !isdefined(@__MODULE__, :run_offline_quadrature_from_netcdf!)
    include(joinpath(@__DIR__, "offline_quadrature.jl"))
end
if !isdefined(@__MODULE__, :run_plot_quadrature_results!)
    include(joinpath(@__DIR__, "plot_quadrature_results.jl"))
end

function _offline_quadrature_forward_cfg()
    cfg = va_forward_sweep_config_from_env()
    if cfg.registry_path === nothing || isempty(strip(cfg.registry_path))
        cfg.registry_path = joinpath("offline_quadrature", "registries", "forward_sweep_cases.yml")
    end
    return cfg
end



function _run_case_truth_and_offline!(
    experiment_dir::AbstractString,
    row,
    forward_cfg::ForwardSweepConfig;
    outdir::String,
    truth_quadrature_order::Int = first(forward_cfg.quadrature_orders),
    quadrature_orders::Vector{Int} = copy(forward_cfg.quadrature_orders),
    sgs_distributions::Vector{String} = String["lognormal"],
    regrid_methods::Vector{Symbol} = Symbol[:linear, :block_average],
    skip_truth::Bool = false,
)
    case_outdir = joinpath(outdir, row.slug)
    truth_dir = joinpath(case_outdir, "truth_simulation_output")
    mkpath(truth_dir)
    truth_profile_nc = joinpath(truth_dir, "highres_profile.nc")

    println("[offline-quadrature] truth case=", row.slug)
    if skip_truth && isfile(truth_profile_nc)
        println("  -> Skipping truth run, found cached profile: ", truth_profile_nc)
        truth_profile = read_highres_truth_profile(truth_profile_nc)
        # Load canonical thermodynamics parameters sidecar
        jld2_path = joinpath(truth_dir, "thermo_params.jld2")
        if !isfile(jld2_path)
            error("Cached truth profile found but thermo_params sidecar missing (" * jld2_path * "). Re-run without `skip_truth` to regenerate canonical params.")
        end
        @load jld2_path thp
    else
        tp, thp = compute_highres_truth_profile_from_case_layers(
            row.model_config_layers,
            String(row.scm_toml);
            case_name = row.slug,
            quadrature_order = truth_quadrature_order,
            truth_sgs_distribution = "lognormal_vertical_profile_full_cubature",
            experiment_dir = experiment_dir,
            output_root = truth_dir,
        )
        write_highres_truth_profile_netcdf!(tp, truth_profile_nc)
        # Persist the canonical thermodynamics parameters alongside the profile.
        # The downstream offline sweep must be able to reload the exact same params
        # when `skip_truth=true`, so the truth bundle on disk has to be complete.
        @save joinpath(truth_dir, "thermo_params.jld2") thp
        truth_profile = tp
    end

    mkpath(case_outdir)

    FT = eltype(thp)
    z_max = FT(row.case_dict["z_max"])
    z_stretch = va_forward_sweep_case_z_stretch(row.case_dict)
    yaml_dz = va_forward_sweep_case_dz_bottom(row.case_dict)

    tiers = va_resolution_tiers_for_forward(row.case_dict, forward_cfg)
    target_grids = Tuple{String, Vector{FT}, Vector{FT}}[]
    for tier in tiers
        res_seg = va_tier_path_segment(tier, z_stretch, yaml_dz)
        z_elem = tier.z_elem
        db = FT(tier.dz_bottom_written === nothing ? yaml_dz : tier.dz_bottom_written)
        stretch = z_stretch ? ClimaCore.Meshes.HyperbolicTangentStretching{FT}(FT(db)) : ClimaCore.Meshes.Uniform()
        mesh = ClimaCore.CommonGrids.DefaultZMesh(
            FT;
            z_min = FT(0),
            z_max = FT(z_max),
            z_elem = z_elem,
            stretch = stretch,
        )
        faces = mesh.faces
        centers = zeros(FT, z_elem)
        for i in 1:z_elem
            z_lo = ClimaCore.Geometry.component(faces[i], 1)
            z_hi = ClimaCore.Geometry.component(faces[i+1], 1)
            centers[i] = (z_lo + z_hi) / 2
        end
        faces_z = zeros(FT, z_elem + 1)
        for i in 1:(z_elem+1)
            faces_z[i] = ClimaCore.Geometry.component(faces[i], 1)
        end
        push!(target_grids, (res_seg, centers, faces_z))
    end
    unique!(target_grids)

    # Use the canonical thermodynamics parameters `thp` produced by the
    # high-resolution truth simulation (or loaded from the sidecar). Do not
    # perform any runtime conversions of parameter element types here — that
    # would mask upstream configuration errors.

    println("[offline-quadrature] offline case=", row.slug, " target_grids=", [g[1] for g in target_grids])
    for regrid_method in regrid_methods
        run_offline_quadrature_from_netcdf!(truth_profile_nc; outdir=case_outdir, target_grids=target_grids, quadrature_orders=quadrature_orders, sgs_distributions=sgs_distributions, skip_existing=true, regrid_method=regrid_method, thermo_params=thp)
    end
    
    # Note: plotting is deferred to main process after distributed work completes so CairoMakie is available

    return (case=row.slug, profile_nc=truth_profile_nc, results_dir=case_outdir)
end

function _setup_distributed_workers!(project_dir::AbstractString, init_script_dir::AbstractString, experiment_dir::AbstractString)
    worker_init = joinpath(init_script_dir, "run_end_to_end_worker_init.jl")
    for p in Distributed.workers()
        Distributed.remotecall_eval(Main, p, :(include($worker_init)))
    end
    return nothing
end

function run_offline_quadrature_end_to_end!(;
    forward_cfg::ForwardSweepConfig = _offline_quadrature_forward_cfg(),
    outdir::String = joinpath(@__DIR__, "..", "outputs"),
    truth_quadrature_order::Int = 5,
    quadrature_orders::Vector{Int} = [1, 2, 3, 4, 5],
    sgs_distributions::Vector{String} = String[
        "lognormal",
        "gaussian",
        "lognormal_vertical_profile_inner_bracketed",
        "lognormal_vertical_profile_full_cubature",
        "lognormal_vertical_profile_lhs_z",
        "lognormal_vertical_profile_principal_axis",
        "lognormal_vertical_profile_voronoi",
        "lognormal_vertical_profile_barycentric",
        "lognormal_vertical_profile_inner_halley",
        "lognormal_vertical_profile_inner_chebyshev",
        "gaussian_vertical_profile_inner_bracketed",
        "gaussian_vertical_profile_full_cubature",
        "gaussian_vertical_profile_lhs_z",
        "gaussian_vertical_profile_principal_axis",
        "gaussian_vertical_profile_voronoi",
        "gaussian_vertical_profile_barycentric",
        "gaussian_vertical_profile_inner_halley",
        "gaussian_vertical_profile_inner_chebyshev",
    ],
    parallel::Symbol = :sequential,
    distributed_workers::Int = 0,
    distributed_worker_threads::Int = 1,
    plot_after::Bool = true,
    plot_include_cases = nothing,
    plot_exclude_cases = String[],
    plot_color_map = Dict{String,Any}(),
    regrid_methods::Vector{Symbol} = Symbol[:linear, :block_average],
    skip_truth::Bool = false,
)
    experiment_dir = normpath(joinpath(@__DIR__, "..", ".."))
    mkpath(outdir)
    rows = va_load_forward_sweep_case_rows(experiment_dir, forward_cfg)
    isempty(rows) && error("Forward sweep registry produced no cases for the selected config.")


    outputs = Dict{String, Any}()
    if parallel === :distributed
        if  ((nworkers = Distributed.nworkers()) < distributed_workers)
            project_dir = dirname(Base.active_project())
            exeflags = "--project=" * project_dir * " -t" * string(max(1, distributed_worker_threads))
            Distributed.addprocs(
                distributed_workers - nworkers;
                exeflags = exeflags,
            )
        end
            init_script_dir = @__DIR__
            project_dir = dirname(Base.active_project())
            _setup_distributed_workers!(project_dir, init_script_dir, experiment_dir)
            results = Distributed.pmap(rows) do row
                _run_case_truth_and_offline!(
                    experiment_dir,
                    row,
                    forward_cfg;
                    outdir = outdir,
                    truth_quadrature_order = truth_quadrature_order,
                    quadrature_orders = quadrature_orders,
                    sgs_distributions = sgs_distributions,
                    regrid_methods = regrid_methods,
                    skip_truth = skip_truth,
                )
            end
            for result in results
                outputs[result.case] = result
            end
    else
        for row in rows
            outputs[row.slug] = _run_case_truth_and_offline!(
                experiment_dir,
                row,
                forward_cfg;
                outdir = outdir,
                truth_quadrature_order = truth_quadrature_order,
                quadrature_orders = quadrature_orders,
                sgs_distributions = sgs_distributions,
                regrid_methods = regrid_methods,
                skip_truth = skip_truth,
            )
        end
    end

    println("[offline-quadrature] done")
    println("  cases: ", join(sort(collect(keys(outputs))), ", "))
    # Produce registry-driven plots for the cases we ran if requested.
    if plot_after
        if isdefined(Main, :run_plot_registry_cases!)
            run_plot_registry_cases!(; outputs_root = outdir, include_cases = plot_include_cases, exclude_cases = plot_exclude_cases, color_map = plot_color_map, quadrature_orders = quadrature_orders, sgs_distributions = sgs_distributions, regrid_methods = regrid_methods)
        else
            # attempt to include plotting helpers if available locally
            p = joinpath(@__DIR__, "plot_quadrature_results.jl")
            isfile(p) && include(p)
            if isdefined(Main, :run_plot_registry_cases!)
                run_plot_registry_cases!(; outputs_root = outdir, include_cases = plot_include_cases, exclude_cases = plot_exclude_cases, color_map = plot_color_map, quadrature_orders = quadrature_orders, sgs_distributions = sgs_distributions, regrid_methods = regrid_methods)
            else
                @warn "Plotting helper `run_plot_registry_cases!` not available; skipping plots"
            end
        end
    end

    return outputs
end

