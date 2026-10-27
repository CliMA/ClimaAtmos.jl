#=
Component-level GPU timing and equivalence harness for the 3D cubed-sphere AMIP
target configuration (`numerics_sphere_he16ze63.yml` + `amip_target.yml`:
`HELEM = 16`, `Nq = 4`, `Nv = 63`, `NonEquilibriumMicrophysics1M`,
`PrognosticEDMFX`, `ManualSparseJacobian`).

Usage:
    CLIMACOMMS_DEVICE=CUDA julia --project=.buildkite perf/benchmark_components.jl [Float32|Float64] [run|save|compare] [dir]

Modes:
  - `run` (default): time all tendency, precomputed-cache, and sparse-Jacobian
    passes and print a Markdown table of minimum and median times (µs).
  - `save [dir]`: compute reference outputs and baseline timings and serialize
    them to `refs_<FT>.jls` and `timings_<FT>.jls` in `dir` (default: tempdir).
  - `compare [dir]`: check relative differences across 25 state, tendency,
    Jacobian-solve, and precomputed quantities against `refs_<FT>.jls` in `dir`,
    then time all passes and print speedups against `timings_<FT>.jls`.
=#

using Printf
using Statistics
using Serialization
import Random

import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore
import ClimaAtmos as CA

const FT_ARG = length(ARGS) >= 1 ? ARGS[1] : "Float32"
const MODE = length(ARGS) >= 2 ? ARGS[2] : "run"
const DATA_DIR = length(ARGS) >= 3 ? ARGS[3] : tempdir()
const BENCH_FT = FT_ARG == "Float64" ? Float64 : Float32

include(joinpath(@__DIR__, "run-and-time", "models", "implicit_tendency.jl"))

function atmos_config_for(::Type{FT}) where {FT}
    config_path = joinpath(pkgdir(CA), "config")
    config_files = [
        joinpath(config_path, "common_configs", "numerics_sphere_he16ze63.yml"),
        joinpath(config_path, "longrun_configs", "amip_target.yml"),
    ]
    configs = map(config_files) do f
        CA.strip_help_messages(CA.load_yaml_file(f))
    end
    configs = (
        CA.strip_help_messages(CA.load_yaml_file(CA.default_config_file)),
        configs...,
        Dict("FLOAT_TYPE" => string(FT)),
    )
    return CA.AtmosConfig(configs; config_files, job_id = "bench_components")
end

get_base_atmos_config() = atmos_config_for(BENCH_FT)

function elapsed_us(f!, ::ClimaComms.CUDADevice)
    return 1e6 * (CUDA.@elapsed f!())
end
function elapsed_us(f!, ::ClimaComms.AbstractDevice)
    return 1e6 * (@elapsed f!())
end

function sync_device(::ClimaComms.CUDADevice)
    CUDA.synchronize()
    return nothing
end
sync_device(::ClimaComms.AbstractDevice) = nothing

# Minimum and median wall-clock time (µs) of `f!` over `nsamples` calls.
function bench_pass(f!, device; nwarmup = 3, nsamples = 15)
    for _ in 1:nwarmup
        f!()
    end
    sync_device(device)
    times = Vector{Float64}(undef, nsamples)
    for i in 1:nsamples
        times[i] = elapsed_us(f!, device)
    end
    return (; min_us = minimum(times), med_us = median(times))
end

function get_bench_items(
    Yₜ,
    Yₜ_lim,
    Y,
    p,
    t,
    ᶠuₕ³,
    thermo_params,
    jac_alg,
    jac_cache,
)
    (; topography_flag, diffusion_flag) = jac_cache.derivative_flags
    (; matrix) = jac_cache
    eddy_diffs =
        CA.use_derivative(diffusion_flag) ?
        CA.eddy_diffusivity_coefficients!(Y, p) : nothing
    return [
        (
            "implicit_tendency! (total)",
            () -> CA.implicit_tendency!(Yₜ, Y, p, t),
        ),
        (
            "  implicit_vertical_advection_tendency!",
            () -> CA.implicit_vertical_advection_tendency!(Yₜ, Y, p, t),
        ),
        (
            "    vertical_advection_of_water_tendency!",
            () -> CA.vertical_advection_of_water_tendency!(Yₜ, Y, p, t),
        ),
        (
            "  microphysics_tendency!",
            () -> CA.microphysics_tendency!(
                Yₜ,
                Y,
                p,
                t,
                p.atmos.microphysics_model,
                p.atmos.turbconv_model,
            ),
        ),
        (
            "  edmfx_sgs_vertical_advection_tendency!",
            () -> CA.edmfx_sgs_vertical_advection_tendency!(
                Yₜ,
                Y,
                p,
                t,
                p.atmos.turbconv_model,
            ),
        ),
        (
            "  edmfx_sgs_diffusive_flux_tendency!",
            () -> CA.edmfx_sgs_diffusive_flux_tendency!(
                Yₜ,
                Y,
                p,
                t,
                p.atmos.turbconv_model,
            ),
        ),
        (
            "  edmfx_entr_detr_tendency!",
            () -> CA.edmfx_entr_detr_tendency!(
                Yₜ,
                Y,
                p,
                t,
                p.atmos.turbconv_model,
            ),
        ),
        (
            "  edmfx_sgs_mass_flux_tendency!",
            () -> CA.edmfx_sgs_mass_flux_tendency!(
                Yₜ,
                Y,
                p,
                t,
                p.atmos.turbconv_model,
            ),
        ),
        (
            "  edmfx_boundary_condition_tendency!",
            () -> CA.edmfx_boundary_condition_tendency!(
                Yₜ,
                Y,
                p,
                t,
                p.atmos.turbconv_model,
            ),
        ),
        (
            "  sgs_u₃ + sgs_ρa implicit_tendency!",
            () -> begin
                CA.sgs_u₃_implicit_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
                CA.sgs_ρa_implicit_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
            end,
        ),
        (
            "set_implicit_precomputed_quantities! (total)",
            () -> CA.set_implicit_precomputed_quantities!(Y, p, t),
        ),
        (
            "  set_prognostic_edmf_precomputed_quantities_draft!",
            () -> CA.set_prognostic_edmf_precomputed_quantities_draft!(
                Y,
                p,
                ᶠuₕ³,
                t,
            ),
        ),
        (
            "  set_prognostic_edmf_precomputed_quantities_environment!",
            () -> CA.set_prognostic_edmf_precomputed_quantities_environment!(
                Y,
                p,
                ᶠuₕ³,
                t,
            ),
        ),
        (
            "  set_precipitation_velocities!",
            () -> CA.set_precipitation_velocities!(
                Y,
                p,
                p.atmos.microphysics_model,
                p.atmos.turbconv_model,
            ),
        ),
        (
            "  set_precipitation_surface_fluxes!",
            () -> CA.set_precipitation_surface_fluxes!(
                Y,
                p,
                p.atmos.microphysics_model,
            ),
        ),
        (
            "set_explicit_precomputed_quantities! (total)",
            () -> CA.set_explicit_precomputed_quantities!(Y, p, t),
        ),
        (
            "  set_prognostic_edmf_precomputed_quantities_explicit_closures!",
            () ->
                CA.set_prognostic_edmf_precomputed_quantities_explicit_closures!(
                    Y,
                    p,
                    t,
                ),
        ),
        (
            "  set_covariance_cache_and_cloud_fraction!",
            () -> CA.set_covariance_cache_and_cloud_fraction!(Y, p),
        ),
        (
            "    set_buoyancy_gradient_inputs!",
            () -> CA.set_buoyancy_gradient_inputs!(Y, p, thermo_params),
        ),
        ("    set_precip_fraction!", () -> CA.set_precip_fraction!(Y, p)),
        ("  set_face_diffusivities!", () -> CA.set_face_diffusivities!(Y, p)),
        (
            "  set_microphysics_tendency_cache!",
            () -> CA.set_microphysics_tendency_cache!(
                Y,
                p,
                p.atmos.microphysics_model,
                p.atmos.turbconv_model,
            ),
        ),
        (
            "remaining_tendency! (total)",
            () -> CA.remaining_tendency!(Yₜ, Yₜ_lim, Y, p, t),
        ),
        (
            "  horizontal_tracer_advection_tendency!",
            () -> CA.horizontal_tracer_advection_tendency!(Yₜ_lim, Y, p, t),
        ),
        (
            "  hyperdiffusion_tendency!",
            () -> CA.hyperdiffusion_tendency!(Yₜ, Yₜ_lim, Y, p, t),
        ),
        (
            "  explicit_vertical_advection_tendency!",
            () -> CA.explicit_vertical_advection_tendency!(Yₜ, Y, p, t),
        ),
        (
            "  viscous_sponge_tendency!",
            () -> CA.viscous_sponge_tendency!(Yₜ, Y, p),
        ),
        (
            "  surface_flux_tendency!",
            () -> CA.surface_flux_tendency!(Yₜ, Y, p, t),
        ),
        (
            "  edmfx_tke_tendency!",
            () -> CA.edmfx_tke_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model),
        ),
        (
            "initialize_implicit_stage_problem!",
            () -> CA.initialize_implicit_stage_problem!(Y, p, p.dt),
        ),
        (
            "correct_implicit_advection_tendency!",
            () -> CA.correct_implicit_advection_tendency!(Yₜ, Y, p, t),
        ),
        (
            "update_jacobian! (total)",
            () -> CA.update_jacobian!(jac_alg, jac_cache, Y, p, p.dt, t),
        ),
        (
            "  update_advection_jacobian!",
            () -> CA.update_advection_jacobian!(
                matrix,
                Y,
                p,
                p.dt,
                topography_flag,
            ),
        ),
        (
            "  update_sedimentation_jacobian!",
            () -> CA.update_sedimentation_jacobian!(matrix, Y, p, p.dt),
        ),
        (
            "  update_diffusion_jacobian!",
            () -> CA.update_diffusion_jacobian!(
                matrix,
                Y,
                p,
                p.dt,
                diffusion_flag,
                eddy_diffs,
            ),
        ),
        (
            "  update_sgs_advection_jacobian!",
            () -> CA.update_sgs_advection_jacobian!(matrix, Y, p, p.dt),
        ),
        (
            "  update_sgs_diffusion_jacobian!",
            () -> CA.update_sgs_diffusion_jacobian!(
                matrix,
                Y,
                p,
                p.dt,
                diffusion_flag,
            ),
        ),
        (
            "  update_sgs_entr_detr_jacobian!",
            () -> CA.update_sgs_entr_detr_jacobian!(matrix, Y, p, p.dt),
        ),
        (
            "  update_sgs_boundary_condition_jacobian!",
            () ->
                CA.update_sgs_boundary_condition_jacobian!(matrix, Y, p, p.dt),
        ),
        (
            "  update_sgs_massflux_jacobian!",
            () -> CA.update_sgs_massflux_jacobian!(
                matrix,
                Y,
                p,
                p.dt,
                diffusion_flag,
            ),
        ),
    ]
end

@noinline snap(f) = Array(parent(f))
save_state(Y) = (; c = copy(parent(Y.c)), f = copy(parent(Y.f)))
function restore_state!(Y, Y₀)
    copyto!(parent(Y.c), Y₀.c)
    copyto!(parent(Y.f), Y₀.f)
    return nothing
end

# Largest relative difference between two host arrays, normalized by the
# maximum finite magnitude across both arrays.
function max_rel_diff(av::Array, bv::Array)
    scale = max(
        maximum(x -> isfinite(x) ? abs(x) : zero(x), av),
        maximum(x -> isfinite(x) ? abs(x) : zero(x), bv),
        eps(eltype(av)),
    )
    diff = map((a, b) -> a == b ? zero(a) : abs(a - b), av, bv)
    return maximum(diff) / scale
end

function compute_refs(state, jac_alg, jac_cache)
    (; Y, Yₜ, p, t) = state
    p.precomputed.ᶜcloud_fraction .= 0
    CA.set_implicit_precomputed_quantities!(Y, p, t)
    CA.set_explicit_precomputed_quantities!(Y, p, t)
    CA.initialize_implicit_stage_problem!(Y, p, p.dt)
    Y_after_init_c = snap(Y.c.sgsʲs)
    Y_after_init_f = snap(Y.f.sgsʲs)
    CA.implicit_tendency!(Yₜ, Y, p, t)
    Yₜ_imp_c = snap(Yₜ.c)
    Yₜ_imp_f = snap(Yₜ.f)
    CA.update_jacobian!(jac_alg, jac_cache, Y, p, p.dt, t)
    ΔY_jac = similar(Yₜ)
    CA.invert_jacobian!(jac_alg, jac_cache, ΔY_jac, Yₜ)
    ΔY_jac_c = snap(ΔY_jac.c)
    ΔY_jac_f = snap(ΔY_jac.f)
    Yₜ_rem = similar(Yₜ)
    Yₜ_lim = similar(Yₜ)
    CA.remaining_tendency!(Yₜ_rem, Yₜ_lim, Y, p, t)
    Yₜ_rem_c = snap(Yₜ_rem.c)
    Yₜ_rem_f = snap(Yₜ_rem.f)
    Yₜ_lim_c = snap(Yₜ_lim.c)
    Yₜ_corr = similar(Yₜ)
    CA.correct_implicit_advection_tendency!(Yₜ_corr, Y, p, t)
    Yₜ_corr_c = snap(Yₜ_corr.c)
    return (;
        Y_after_init_c,
        Y_after_init_f,
        Yₜ_imp_c,
        Yₜ_imp_f,
        ΔY_jac_c,
        ΔY_jac_f,
        Yₜ_rem_c,
        Yₜ_rem_f,
        Yₜ_lim_c,
        Yₜ_corr_c,
        ᶠK_h = snap(p.precomputed.ᶠK_h),
        ᶠK_u = snap(p.precomputed.ᶠK_u),
        ᶜgradᵥ_θ = snap(p.precomputed.ᶜgradᵥ_θ_liq_ice),
        ᶜgradᵥ_q = snap(p.precomputed.ᶜgradᵥ_q_tot),
        ᶜcloud_fraction = snap(p.precomputed.ᶜcloud_fraction),
        ᶜprecip_frac = snap(p.precomputed.ᶜprecip_frac),
        ᶜsgs_moments = snap(p.precomputed.ᶜsgs_moments),
        ᶜl_mix = snap(p.precomputed.ᶜl_mix),
        ᶜT′T′ = snap(p.precomputed.ᶜT′T′),
        ᶜq′q′ = snap(p.precomputed.ᶜq′q′),
        ᶜcorr_Tq = snap(p.precomputed.ᶜcorr_Tq),
        ᶜα = snap(p.precomputed.ᶜα),
        ᶜmp_tendency⁰ = snap(p.precomputed.ᶜmp_tendency⁰),
        rain_flux = snap(p.precomputed.surface_rain_flux),
        snow_flux = snap(p.precomputed.surface_snow_flux),
    )
end

function report_equivalence(ref, new)
    checks = (
        ("Y.c.sgsʲs after init", ref.Y_after_init_c, new.Y_after_init_c),
        ("Y.f.sgsʲs after init", ref.Y_after_init_f, new.Y_after_init_f),
        ("implicit Yₜ.c", ref.Yₜ_imp_c, new.Yₜ_imp_c),
        ("implicit Yₜ.f", ref.Yₜ_imp_f, new.Yₜ_imp_f),
        ("invert_jacobian! ΔY.c", ref.ΔY_jac_c, new.ΔY_jac_c),
        ("invert_jacobian! ΔY.f", ref.ΔY_jac_f, new.ΔY_jac_f),
        ("remaining Yₜ.c", ref.Yₜ_rem_c, new.Yₜ_rem_c),
        ("remaining Yₜ.f", ref.Yₜ_rem_f, new.Yₜ_rem_f),
        ("remaining Yₜ_lim.c", ref.Yₜ_lim_c, new.Yₜ_lim_c),
        ("correct_implicit Yₜ.c", ref.Yₜ_corr_c, new.Yₜ_corr_c),
        ("ᶠK_h", ref.ᶠK_h, new.ᶠK_h),
        ("ᶠK_u", ref.ᶠK_u, new.ᶠK_u),
        ("ᶜgradᵥ_θ_liq_ice", ref.ᶜgradᵥ_θ, new.ᶜgradᵥ_θ),
        ("ᶜgradᵥ_q_tot", ref.ᶜgradᵥ_q, new.ᶜgradᵥ_q),
        ("ᶜcloud_fraction", ref.ᶜcloud_fraction, new.ᶜcloud_fraction),
        ("ᶜprecip_frac", ref.ᶜprecip_frac, new.ᶜprecip_frac),
        ("ᶜsgs_moments", ref.ᶜsgs_moments, new.ᶜsgs_moments),
        ("ᶜl_mix", ref.ᶜl_mix, new.ᶜl_mix),
        ("ᶜT′T′", ref.ᶜT′T′, new.ᶜT′T′),
        ("ᶜq′q′", ref.ᶜq′q′, new.ᶜq′q′),
        ("ᶜcorr_Tq", ref.ᶜcorr_Tq, new.ᶜcorr_Tq),
        ("ᶜα", ref.ᶜα, new.ᶜα),
        ("ᶜmp_tendency⁰", ref.ᶜmp_tendency⁰, new.ᶜmp_tendency⁰),
        ("surface_rain_flux", ref.rain_flux, new.rain_flux),
        ("surface_snow_flux", ref.snow_flux, new.snow_flux),
    )
    println("| Quantity | max relative difference |")
    println("|---|---|")
    worst = 0.0
    for (name, a, b) in checks
        d = max_rel_diff(a, b)
        worst = max(worst, d)
        @printf("| %s | %.2e |\n", name, d)
    end
    return worst
end

function warmup_items(items, Y, Y₀, device)
    for (_, fn) in items
        restore_state!(Y, Y₀)
        fn()
    end
    sync_device(device)
    return nothing
end

function bench_items(items, Y, Y₀, device)
    res = Dict{String, NamedTuple}()
    for (name, fn) in items
        restore_state!(Y, Y₀)
        res[name] = bench_pass(fn, device)
    end
    return res
end

function print_timings(items, res)
    println("| Function | min_us | med_us |")
    println("|---|---|---|")
    for (name, _) in items
        r = res[name]
        @printf("| %s | %.1f | %.1f |\n", name, r.min_us, r.med_us)
    end
    flush(stdout)
end

function print_comparison(items, res_base, res_new)
    println(
        "| Function | Base min / med (µs) | New min / med (µs) | Speedup (min / med) |",
    )
    println("|---|---|---|---|")
    for (name, _) in items
        rb = res_base[name]
        r = res_new[name]
        @printf(
            "| %s | %.1f / %.1f | %.1f / %.1f | %.2fx / %.2fx |\n",
            name,
            rb.min_us,
            rb.med_us,
            r.min_us,
            r.med_us,
            rb.min_us / r.min_us,
            rb.med_us / r.med_us,
        )
    end
    flush(stdout)
end

function main()
    device = ClimaComms.device()
    println(
        "=== Setting up ClimaAtmos (HELEM=16, Nv=63, FT=$BENCH_FT, device=$(typeof(device))) ===",
    )
    flush(stdout)
    Random.seed!(1234)
    state = case_setup()
    (; Y, Yₜ, p, t) = state
    @assert eltype(Y) == BENCH_FT
    ᶠuₕ³ = p.scratch.ᶠtemp_CT3
    thermo_params = CA.CAP.thermodynamics_params(p.params)
    jac_alg = CA.ManualSparseJacobian()
    jac_cache = CA.jacobian_cache(jac_alg, Y, p.atmos)
    Y₀ = save_state(Y)

    refs_file = joinpath(DATA_DIR, "refs_$(BENCH_FT).jls")
    timings_file = joinpath(DATA_DIR, "timings_$(BENCH_FT).jls")

    if MODE == "save"
        mkpath(DATA_DIR)
        refs = compute_refs(state, jac_alg, jac_cache)
        serialize(refs_file, refs)
        println("Saved reference outputs to $(refs_file).")
    elseif MODE == "compare"
        refs = deserialize(refs_file)
        new_refs = compute_refs(state, jac_alg, jac_cache)
        println("\n=== Equivalence vs $(refs_file) ===")
        worst = report_equivalence(refs, new_refs)
        tol = BENCH_FT == Float64 ? 1e-12 : 1e-5
        worst <= tol || @warn "Largest relative difference $worst exceeds $tol"
        refs = nothing
        new_refs = nothing
        GC.gc(true)
    else
        # Populate the precomputed cache once before timing individual passes.
        CA.set_implicit_precomputed_quantities!(Y, p, t)
        CA.set_explicit_precomputed_quantities!(Y, p, t)
    end

    Yₜ_lim = similar(Yₜ)
    items = get_bench_items(
        Yₜ,
        Yₜ_lim,
        Y,
        p,
        t,
        ᶠuₕ³,
        thermo_params,
        jac_alg,
        jac_cache,
    )
    warmup_items(items, Y, Y₀, device)
    GC.gc(true)
    res = bench_items(items, Y, Y₀, device)

    if MODE == "save"
        print_timings(items, res)
        serialize(timings_file, res)
        println("Saved baseline timings to $(timings_file).")
    elseif MODE == "compare" && isfile(timings_file)
        res_base = deserialize(timings_file)
        println("\n=== Timing comparison (FT = $BENCH_FT) ===")
        print_comparison(items, res_base, res)
    else
        println("\n=== Component timings (FT = $BENCH_FT) ===")
        print_timings(items, res)
    end
    return nothing
end

main()
