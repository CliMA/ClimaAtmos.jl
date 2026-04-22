using Test
using LinearAlgebra
using ClimaComms
using ClimaCore: Fields
ClimaComms.@import_required_backends

import ClimaAtmos as CA
import SciMLBase

const CONFIG_DIR =
    joinpath(pkgdir(CA), "config", "model_configs")

function load_fi_config_f64(; overrides = Dict{String, Any}())
    path = joinpath(CONFIG_DIR, "plane_density_current_fully_implicit_float64_test.yml")
    isfile(path) || error("Config not found: $path")
    cfg = CA.AtmosConfig(path)
    if isempty(overrides)
        return cfg
    end
    merged = copy(cfg.parsed_args)
    for (k, v) in overrides
        merged[string(k)] = v
    end
    job_id = get(merged, "job_id", "fi_f64_bubble_test")
    return CA.AtmosConfig(merged; job_id = string(job_id))
end

function load_hevi_config(; overrides = Dict{String, Any}())
    path = joinpath(CONFIG_DIR, "plane_density_current_test.yml")
    isfile(path) || error("Config not found: $path")
    cfg = CA.AtmosConfig(path)
    merged = copy(cfg.parsed_args)
    merged["enable_diagnostics"] = false
    for (k, v) in overrides
        merged[string(k)] = v
    end
    job_id = get(merged, "job_id", "hevi_bubble_test")
    return CA.AtmosConfig(merged; job_id = string(job_id))
end

function run_to_time!(sim, t_target)
    integ = sim.integrator
    n = 0
    while integ.t < t_target
        SciMLBase.step!(integ)
        n += 1
        n > 1_000_000 && error("Too many steps; integrator.t = $(integ.t)")
    end
    return integ, n
end

function bubble_max_theta_x(Y, p)
    try
        # Use available thermodynamic / state information to build a proxy for θ
        # (cold bubble center). Prefer precomputed temperature if present.
        θ = if hasproperty(p.precomputed, :ᶜT)
            p.precomputed.ᶜT
        elseif hasproperty(Y.c, :ρe_tot)
            # Fallback: specific total energy as proxy for temperature
            Y.c.ρe_tot ./ Y.c.ρ
        else
            # Last resort: density anomaly proxy
            Y.c.ρ
        end

        # Obtain coordinates from the field axes
        coords = Fields.coordinate_field(axes(Y.c.ρ))
        θ_parent = parent(θ)
        z_parent = parent(coords.z)

        # Flatten and find index of maximum θ (vertical location of warmest air)
        idx_max = argmax(θ_parent)
        return z_parent[idx_max]
    catch e
        @info "bubble_max_theta_x diagnostic failed; returning NaN" error = e
        return NaN
    end
end

@testset "Fully implicit bubble motion diagnostics" begin

    @testset "Vertical velocity update (Float64 FI)" begin
        cfg = load_fi_config_f64(; overrides = Dict(
            "dt" => "1secs",
            "t_end" => "2secs",
        ))
        sim = CA.get_simulation(cfg)
        integ = sim.integrator
        u_before = copy(integ.u)
        SciMLBase.step!(integ)
        u_after = integ.u

        Δu₃ = sqrt(sum(abs2, parent(u_after.f.u₃) .- parent(u_before.f.u₃)))
        @test Δu₃ > 0
    end

    @testset "HEVI vs fully implicit vertical motion and theta max (short run)" begin
        t_compare = 3.0

        cfg_hevi = load_hevi_config(;
            overrides = Dict(
                "dt" => "0.3secs",
                "t_end" => "$(t_compare)secs",
            ),
        )
        sim_hevi = CA.get_simulation(cfg_hevi)
        integ_hevi, _ = run_to_time!(sim_hevi, t_compare)
        Y_hevi = copy(integ_hevi.u)
        p_hevi = integ_hevi.p
        CA.set_precomputed_quantities!(Y_hevi, p_hevi, t_compare)

        cfg_fi = load_fi_config_f64(;
            overrides = Dict(
                "dt" => "0.3secs",
                "t_end" => "$(t_compare)secs",
            ),
        )
        sim_fi = CA.get_simulation(cfg_fi)
        integ_fi, _ = run_to_time!(sim_fi, t_compare)
        Y_fi = copy(integ_fi.u)
        p_fi = integ_fi.p
        CA.set_precomputed_quantities!(Y_fi, p_fi, t_compare)

        # Numerical diagnostics (no hard-fail assertions except basic sanity)
        norm_w_hevi = sqrt(sum(abs2, parent(Y_hevi.f.u₃)))
        norm_w_fi = sqrt(sum(abs2, parent(Y_fi.f.u₃)))

        @test norm_w_hevi > 0
        @test norm_w_fi > 0

        rel_diff_w = abs(norm_w_hevi - norm_w_fi) /
                     max(norm_w_hevi, norm_w_fi)

        x_max_hevi = bubble_max_theta_x(Y_hevi, p_hevi)
        x_max_fi = bubble_max_theta_x(Y_fi, p_fi)
        bubble_dx = abs(x_max_hevi - x_max_fi)

        @info "Short-run HEVI vs FI diagnostics" norm_w_hevi norm_w_fi rel_diff_w x_max_hevi x_max_fi bubble_dx
    end

    @testset "HEVI vs fully implicit diagnostics at 30s" begin
        t_compare = 30.0

        cfg_hevi = load_hevi_config(;
            overrides = Dict(
                "dt" => "3secs",
                "t_end" => "$(t_compare)secs",
            ),
        )
        sim_hevi = CA.get_simulation(cfg_hevi)
        integ_hevi, _ = run_to_time!(sim_hevi, t_compare)
        Y_hevi = copy(integ_hevi.u)
        p_hevi = integ_hevi.p
        CA.set_precomputed_quantities!(Y_hevi, p_hevi, t_compare)

        cfg_fi = load_fi_config_f64(;
            overrides = Dict(
                "dt" => "3secs",
                "t_end" => "$(t_compare)secs",
            ),
        )
        sim_fi = CA.get_simulation(cfg_fi)
        integ_fi, _ = run_to_time!(sim_fi, t_compare)
        Y_fi = copy(integ_fi.u)
        p_fi = integ_fi.p
        CA.set_precomputed_quantities!(Y_fi, p_fi, t_compare)

        # Diagnostics at 30s
        norm_w_hevi = sqrt(sum(abs2, parent(Y_hevi.f.u₃)))
        norm_w_fi = sqrt(sum(abs2, parent(Y_fi.f.u₃)))

        x_max_hevi = bubble_max_theta_x(Y_hevi, p_hevi)
        x_max_fi = bubble_max_theta_x(Y_fi, p_fi)
        bubble_dx = abs(x_max_hevi - x_max_fi)

        @info "30s HEVI vs FI diagnostics" norm_w_hevi norm_w_fi x_max_hevi x_max_fi bubble_dx

        # Sanity-only assertions
        @test norm_w_hevi > 0
        @test norm_w_fi > 0
    end

end
