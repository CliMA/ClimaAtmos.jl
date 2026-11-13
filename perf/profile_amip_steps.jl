#=
Temporary profiling driver (comparison branch, not for merging) for flagship AMIP coupling
steps. Takes the arguments of experiments/AMIP/run_simulation.jl, and reads
- COUPLER_DIR: the ClimaCoupler checkout,
- PROFILE_OUT: the output directory,
- PROFILE_MODE: `trace` (default) writes the host, device, and NVTX tables of CUDA.jl's
  profile of PROFILE_STEPS (default 20) coupling steps to TSV files; `ncu` runs one coupling
  step between cuProfilerStart and cuProfilerStop, for `ncu --profile-from-start off`, and
  `ncu_rad` runs the PROFILE_STEPS steps of the trace between them, for an `ncu` filtered to
  the RRTMGP kernels of the one radiation call in those steps.
All modes first run two coupling steps, which compile the step.
=#
import CUDA
# PROFILE_SROA=1 (experiment): run SROA and instcombine after GPUCompiler's last
# AlwaysInlinerPass, at the start of its cleanup pipeline, which no promotion pass follows, so
# that the stack copies of the by-value kernel arguments are promoted to registers. The
# original pipeline is called in the world before this method replaces it.
if get(ENV, "PROFILE_SROA", "0") == "1"
    let GPUC = only(m for (k, m) in Base.loaded_modules if k.name == "GPUCompiler"), world = Base.get_world_counter()
        FPM = isdefined(GPUC, :NewPMFunctionPassManager) ? :NewPMFunctionPassManager : :FunctionPassManager
        @eval GPUC function buildCleanupPipeline(mpm, @nospecialize(job::CompilerJob), opt_level)
            add!(mpm, $FPM()) do fpm
                add!(fpm, SROAPass())
                add!(fpm, instcombine_pass(job))
            end
            Base.invoke_in_world($world, buildCleanupPipeline, mpm, job, opt_level)
        end
    end
    @info "GPUCompiler.buildCleanupPipeline runs SROA and instcombine first"
end
include(joinpath(ENV["COUPLER_DIR"], "experiments", "AMIP", "code_loading.jl"))
# PROFILE_REGCAP=n (experiment): compile every slice-loop kernel that leaves its register cap
# at :auto with at most n registers per thread, to find the cap that suits each kernel.
if haskey(ENV, "PROFILE_REGCAP")
    let core = only(m for (k, m) in Base.loaded_modules if k.name == "ClimaCore"),
        cap = parse(Int, ENV["PROFILE_REGCAP"])

        @eval Base.get_extension(core, :ClimaCoreCUDAExt) register_cap(::Val{:auto}, _) = $cap
        @info "Slice-loop kernels without a register cap are compiled with $cap registers per thread"
    end
end
const NVTX = only(m for (k, m) in Base.loaded_modules if k.name == "NVTX")

table_cell(x::Missing) = ""
table_cell(x::AbstractString) = replace(x, r"[\t\n]" => " ")
table_cell(x::Real) = string(x)
table_cell(x::Symbol) = string(x)
table_cell(x) = hasproperty(x, :z) ? string(x.x * x.y * x.z) : string(x)

function write_table(file, table, columns)
    open(file, "w") do io
        println(io, join(columns, '\t'))
        for i in eachindex(table.id)
            println(io, join((table_cell(getproperty(table, c)[i]) for c in columns), '\t'))
        end
    end
end

# Ranges of NVTX.@annotate and NVTX.@range, recorded on the clock of CUPTI's activity
# records, since CUDA.jl's profiler does not record NVTX ranges with every CUPTI version
const RANGE_ROWS = Tuple{Int, Float64, Symbol, String}[]
function cupti_seconds()
    t = Ref{UInt64}()
    CUDA.CUPTI.cuptiGetTimestamp(t)
    return t[] / 1e9
end
function record_range_start(message)
    id = length(RANGE_ROWS) + 1
    name = message isa NVTX.StringHandle ? message.string : string(message)
    push!(RANGE_ROWS, (id, cupti_seconds(), :start, name))
    return reinterpret(NVTX.RangeId, UInt64(id))
end
record_range_end(range) =
    push!(RANGE_ROWS, (Int(reinterpret(UInt64, range)), cupti_seconds(), :end, ""))
mode = get(ENV, "PROFILE_MODE", "trace")
if mode == "trace"
    @eval NVTX begin
        init!(domain::Domain) = domain
        init!(sh::StringHandle) = sh
        range_start(domain::Domain; message = nothing, kwargs...) =
            Main.record_range_start(message)
        range_end(range::RangeId) = (Main.record_range_end(range); nothing)
    end
    NVTX.NVTX_ACTIVE[] = true
end

config_file = Input.parse_commandline(Input.argparse_settings())["config_file"]
cs = CoupledSimulation(config_file)
coupling_step! = getfield(parentmodule(run!), :step!)
out = ENV["PROFILE_OUT"]
mkpath(out)

compile_time = @elapsed begin
    coupling_step!(cs)
    coupling_step!(cs)
    CUDA.synchronize()
end
@info "Two coupling steps (with compilation)" compile_time

if mode == "ncu"
    CUDA.@profile external = true begin
        NVTX.@range "coupling step" begin
            coupling_step!(cs)
            CUDA.synchronize()
        end
    end
elseif mode == "ncu_rad"
    # The PROFILE_STEPS coupling steps of the trace, which contain one radiation call, for
    # `ncu --kernel-name` filtered to the RRTMGP kernels
    n_steps = parse(Int, get(ENV, "PROFILE_STEPS", "20"))
    CUDA.@profile external = true begin
        for _ in 1:n_steps
            coupling_step!(cs)
        end
        CUDA.synchronize()
    end
else
    n_steps = parse(Int, get(ENV, "PROFILE_STEPS", "20"))
    empty!(RANGE_ROWS)
    res = CUDA.@profile external = false begin
        for _ in 1:n_steps
            NVTX.@range "coupling step" begin
                coupling_step!(cs)
                CUDA.synchronize()
            end
        end
    end
    write_table(joinpath(out, "host.tsv"), res.host, (:id, :start, :stop, :name, :tid))
    write_table(
        joinpath(out, "device.tsv"),
        res.device,
        (:id, :start, :stop, :name, :stream, :grid, :block, :registers),
    )
    ranges = (;
        id = first.(RANGE_ROWS),
        start = getindex.(RANGE_ROWS, 2),
        type = getindex.(RANGE_ROWS, 3),
        tid = zeros(Int, length(RANGE_ROWS)),
        name = last.(RANGE_ROWS),
    )
    write_table(joinpath(out, "nvtx.tsv"), ranges, (:id, :start, :type, :tid, :name))
    write_table(joinpath(out, "cupti_nvtx.tsv"), res.nvtx, (:id, :start, :type, :tid, :name))
    (; start, stop) = res.device
    @info "Profiled $n_steps coupling steps" n_device_events = length(start) n_ranges =
        length(RANGE_ROWS) n_cupti_nvtx_events = length(res.nvtx.id) device_time_per_step =
        sum(stop .- start) / n_steps
end
