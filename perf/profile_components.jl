#=
Temporary profiling driver (not for merging) for the AMIP target configuration of
`benchmark_components.jl`: runs that script's `main()` (timings, and with `save` or
`compare` the GPU equivalence check), then profiles one call of every top-level
component.

    CLIMACOMMS_DEVICE=CUDA julia --project=.buildkite perf/profile_components.jl Float32 [run|save|compare] [dir] [cuda|ncu]

- `cuda` (default): print CUDA.jl's profile (host and device activity) of each component.
- `ncu`: skip the timings, and wrap one call of each component in an NVTX range between
  `cuProfilerStart` and `cuProfilerStop`, for `ncu --profile-from-start off --nvtx`.
=#
const PROFILE_MODE = length(ARGS) >= 4 ? ARGS[4] : "cuda"
let file = joinpath(@__DIR__, "benchmark_components.jl")
    include_string(Main, replace(read(file, String), r"\nmain\(\)\s*$" => "\n"), file)
end
import CUDA

function profile_components(mode)
    device = ClimaComms.device()
    println("GPU: ", CUDA.name(CUDA.device()))
    Random.seed!(1234)
    state = case_setup()
    (; Y, Yₜ, p, t) = state
    jac_alg = CA.ManualSparseJacobian()
    jac_cache = CA.jacobian_cache(jac_alg, Y, p.atmos)
    Y₀ = save_state(Y)
    CA.set_implicit_precomputed_quantities!(Y, p, t)
    CA.set_explicit_precomputed_quantities!(Y, p, t)
    items = get_bench_items(
        Yₜ,
        similar(Yₜ),
        Y,
        p,
        t,
        p.scratch.ᶠtemp_CT3,
        CA.CAP.thermodynamics_params(p.params),
        jac_alg,
        jac_cache,
    )
    warmup_items(items, Y, Y₀, device)
    for (name, fn) in items
        startswith(name, " ") && continue
        restore_state!(Y, Y₀)
        CUDA.synchronize()
        if mode == "ncu"
            CA.NVTX.range_push(; message = name)
            CUDA.@profile external = true begin
                fn()
                CUDA.synchronize()
            end
            CA.NVTX.range_pop()
        else
            res = CUDA.@profile external = false begin
                fn()
                CUDA.synchronize()
            end
            println("\n=== CUDA profile of one call: $name ===")
            show(IOContext(stdout, :limit => false), res)
            println()
        end
        flush(stdout)
    end
    return nothing
end

if PROFILE_MODE == "cuda"
    main()
    GC.gc(true)
    CUDA.reclaim()
end
profile_components(PROFILE_MODE)
