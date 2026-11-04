#=
Temporary (comparison branch, not for merging): compares how three fused loops read
their inputs (`CA.LAB_MODE`: `base`, `inv`, `reg`, `reg_struct`) in the AMIP target
configuration of `benchmark_components.jl`. Each mode runs in its own process, so that the
first call of each loop (compilation and one run) starts from the same compiled state.

    CLIMACOMMS_DEVICE=CUDA julia --project=.buildkite perf/register_vs_invariant.jl <mode>

The setup runs with the loops skipped. The script then times the first call of each loop's
function, updates the cache with all loops in `mode`, times each function (`bench_pass`),
lists its kernels and their device times on GPUs, and compares its outputs with those of
`base` from the same state. `LAB_H_ELEM` sets a smaller horizontal resolution (CPU checks).
=#
const LAB_MODE_ARG = Symbol(ARGS[1])
const LAB_OVERRIDES =
    haskey(ENV, "LAB_H_ELEM") ? ("h_elem" => parse(Int, ENV["LAB_H_ELEM"]),) : ()
import CUDA
empty!(ARGS)
append!(ARGS, ["Float32", "run"])
let file = joinpath(@__DIR__, "benchmark_components.jl")
    code = replace(
        read(file, String),
        r"\nmain\(\)\s*$" => "\n",
        "Dict(\"FLOAT_TYPE\" => string(FT))" =>
            "Dict(\"FLOAT_TYPE\" => string(FT), LAB_OVERRIDES...)",
    )
    include_string(Main, code, file)
end
import ClimaCore: Fields

const IS_GPU = ClimaComms.device() isa ClimaComms.CUDADevice

# Host copies of the parent arrays of the given fields
snapshot(fields) = map(field -> Array(parent(field)), fields)

function max_rel_diffs(as, bs)
    return map(as, bs) do a, b
        all(((x, y),) -> isequal(x, y), zip(a, b)) ? 0.0 : max_rel_diff(a, b)
    end
end

function kernel_times(f!, n)
    IS_GPU || return nothing
    res = CUDA.@profile external = false begin
        for _ in 1:n
            f!()
        end
        CUDA.synchronize()
    end
    totals = Dict{String, Tuple{Float64, Int}}()
    (; name, start, stop) = res.device
    for i in eachindex(name)
        t, c = get(totals, name[i], (0.0, 0))
        totals[name[i]] = (t + stop[i] - start[i], c + 1)
    end
    return sort!(collect(totals); by = x -> -x[2][1])
end

function lab_main()
    device = ClimaComms.device()
    mode = LAB_MODE_ARG
    println("=== register_vs_invariant: mode $mode, device $(typeof(device))")
    CA.LAB_MODE[] = :skip
    setup_time = @elapsed begin
        Random.seed!(1234)
        state = case_setup()
    end
    (; Y, Yₜ, p, t) = state
    (; microphysics_model, turbconv_model) = p.atmos
    pc = p.precomputed
    zero_tendency!() = (parent(Yₜ.c) .= 0; parent(Yₜ.f) .= 0)
    loops = (
        (
            "set_precipitation_velocities! (1M, PrognosticEDMFX)",
            () -> CA.set_precipitation_velocities!(
                Y, p, microphysics_model, turbconv_model,
            ),
            () -> (pc.ᶜwₗ, pc.ᶜwᵢ, pc.ᶜwᵣ, pc.ᶜwₛ, pc.ᶜwₕhₜ, pc.ᶜwₗʲs, pc.ᶜwᵢʲs, pc.ᶜwᵣʲs, pc.ᶜwₛʲs),
        ),
        (
            "edmfx_entr_detr_tendency! (PrognosticEDMFX)",
            () -> CA.edmfx_entr_detr_tendency!(Yₜ, Y, p, t, turbconv_model),
            () -> (Yₜ.c, Yₜ.f),
        ),
        (
            "set_covariance_cache_and_cloud_fraction!",
            () -> CA.set_covariance_cache_and_cloud_fraction!(Y, p),
            () -> (
                pc.ᶜbuoygrad, pc.ᶜl_mix, pc.ᶜq′q′, pc.ᶜT′T′, pc.ᶜcorr_Tq, pc.ᶜcloud_fraction,
            ),
        ),
    )
    @printf("setup (loops skipped): %.1f s\n", setup_time)
    CA.LAB_MODE[] = mode
    println("\n| Loop function | first call (s) |")
    println("|---|---|")
    for (name, f!, _) in loops
        first_time = @elapsed begin
            f!()
            sync_device(device)
        end
        @printf("| %s | %.2f |\n", name, first_time)
        flush(stdout)
    end

    # The cache of every loop in `mode`, so that the timed calls read finite inputs
    p.precomputed.ᶜcloud_fraction .= 0
    CA.set_implicit_precomputed_quantities!(Y, p, t)
    CA.set_explicit_precomputed_quantities!(Y, p, t)
    Y₀ = save_state(Y)
    GC.gc(true)

    println("\n| Loop function | min (µs) | median (µs) |")
    println("|---|---|---|")
    for (name, f!, _) in loops
        restore_state!(Y, Y₀)
        r = bench_pass(f!, device; nwarmup = 5, nsamples = IS_GPU ? 100 : 5)
        @printf("| %s | %.1f | %.1f |\n", name, r.min_us, r.med_us)
        flush(stdout)
    end

    if IS_GPU
        for (name, f!, _) in loops
            restore_state!(Y, Y₀)
            println("\nKernels of $name (10 calls):")
            println("| kernel | calls | mean device time (µs) |")
            println("|---|---|---|")
            for (kname, (total, count)) in kernel_times(f!, 10)
                @printf("| %s | %d | %.1f |\n", first(kname, 160), count, 1e6 * total / count)
            end
            flush(stdout)
        end
    end

    println("\n| Loop function | max relative difference from base, per output |")
    println("|---|---|")
    for (name, f!, outputs) in loops
        restore_state!(Y, Y₀)
        zero_tendency!()
        f!()
        sync_device(device)
        new = snapshot(outputs())
        CA.LAB_MODE[] = :base
        restore_state!(Y, Y₀)
        zero_tendency!()
        f!()
        sync_device(device)
        base = snapshot(outputs())
        CA.LAB_MODE[] = mode
        diffs = max_rel_diffs(new, base)
        println("| ", name, " | ", join((@sprintf("%.2e", d) for d in diffs), ", "), " |")
        flush(stdout)
    end
    return nothing
end

lab_main()
