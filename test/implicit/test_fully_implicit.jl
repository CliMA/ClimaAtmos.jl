#=
Automated tests for fully implicit timestepping (DIRK + JFNK).

Tests validate:
1. Tendency decomposition: fully_implicit_tendency! = remaining_tendency! + implicit_tendency!
2. State update: one step actually changes the state
3. Mass conservation over short integration
4. Energy conservation over short integration
5. HEVI vs fully implicit agreement at small dt
6. dt sweep: solver completes at multiple dt values

Run: julia --project=. test/implicit/test_fully_implicit.jl
=#

using Test
using LinearAlgebra
using ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import SciMLBase

const CONFIG_DIR =
    joinpath(pkgdir(CA), "config", "model_configs")

function load_fi_config(; overrides = Dict{String, Any}())
    path = joinpath(CONFIG_DIR, "plane_density_current_fully_implicit_test.yml")
    isfile(path) || error("Config not found: $path")
    cfg = CA.AtmosConfig(path)
    if isempty(overrides)
        return cfg
    end
    merged = copy(cfg.parsed_args)
    for (k, v) in overrides
        merged[string(k)] = v
    end
    job_id = get(merged, "job_id", "fi_test")
    return CA.AtmosConfig(merged; job_id = string(job_id))
end

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
    job_id = get(merged, "job_id", "fi_f64_test")
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
    job_id = get(merged, "job_id", "hevi_test")
    return CA.AtmosConfig(merged; job_id = string(job_id))
end

function run_steps!(sim, n_steps)
    integ = sim.integrator
    for _ in 1:n_steps
        SciMLBase.step!(integ)
    end
    return integ
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

@testset "Fully implicit timestepping" begin

    @testset "Tendency decomposition consistency" begin
        # At initial state, fully_implicit_tendency! should equal
        # remaining_tendency! + implicit_tendency! + limiter contributions
        cfg = load_fi_config(; overrides = Dict(
            "dt" => "1secs",
            "t_end" => "2secs",
        ))
        sim = CA.get_simulation(cfg)
        Y = copy(sim.integrator.u)
        p = sim.integrator.p
        t = Float64(sim.integrator.t)

        # Ensure precomputed quantities are set
        CA.set_precomputed_quantities!(Y, p, t)

        # Full tendency
        Yₜ_full = similar(Y)
        CA.fully_implicit_tendency!(Yₜ_full, Y, p, t)

        # Decomposed tendencies
        Yₜ_exp = similar(Y)
        Yₜ_lim = similar(Y)
        CA.remaining_tendency!(Yₜ_exp, Yₜ_lim, Y, p, t)

        Yₜ_imp = similar(Y)
        CA.implicit_tendency!(Yₜ_imp, Y, p, t)

        # Sum should equal full
        sum_parts = Yₜ_exp .+ Yₜ_imp .+ Yₜ_lim
        diff = Yₜ_full .- sum_parts

        l2_diff = sqrt(sum(abs2, parent(diff)))
        l2_full = sqrt(sum(abs2, parent(Yₜ_full)))
        rel_err = l2_full > 0 ? l2_diff / l2_full : 0.0

        @test rel_err < 1e-10
        @info "Tendency decomposition" rel_err l2_diff l2_full
    end

    @testset "State update (smoke test)" begin
        # One step should produce nonzero state change
        cfg = load_fi_config(; overrides = Dict(
            "dt" => "1secs",
            "t_end" => "2secs",
        ))
        sim = CA.get_simulation(cfg)
        integ = sim.integrator
        u_before = copy(integ.u)
        SciMLBase.step!(integ)
        u_after = integ.u

        diff_vec = parent(u_after) .- parent(u_before)
        l2_change = sqrt(sum(abs2, diff_vec))
        linf_change = maximum(abs, diff_vec)

        @test l2_change > 1e-12
        @info "State update" l2_change linf_change

        # Check individual components
        Δρ = sqrt(sum(abs2, parent(u_after.c.ρ) .- parent(u_before.c.ρ)))
        Δu₃ = sqrt(sum(abs2, parent(u_after.f.u₃) .- parent(u_before.f.u₃)))
        @test Δρ > 0
        @test Δu₃ > 0
        @info "Component changes" Δρ Δu₃
    end

    @testset "Mass conservation (30s integration)" begin
        cfg = load_fi_config(; overrides = Dict(
            "dt" => "3secs",
            "t_end" => "30secs",
        ))
        sim = CA.get_simulation(cfg)
        integ = sim.integrator
        mass_0 = sum(integ.u.c.ρ)

        run_to_time!(sim, 30.0)
        mass_end = sum(integ.u.c.ρ)

        rel_mass_err = abs(mass_end - mass_0) / abs(mass_0)
        @test rel_mass_err < 1e-5
        @info "Mass conservation" mass_0 mass_end rel_mass_err
    end

    @testset "Energy conservation (30s integration)" begin
        cfg = load_fi_config(; overrides = Dict(
            "dt" => "3secs",
            "t_end" => "30secs",
        ))
        sim = CA.get_simulation(cfg)
        integ = sim.integrator
        energy_0 = sum(integ.u.c.ρe_tot)

        run_to_time!(sim, 30.0)
        energy_end = sum(integ.u.c.ρe_tot)

        rel_energy_err = abs(energy_end - energy_0) / abs(energy_0)
        @test rel_energy_err < 1e-5
        @info "Energy conservation" energy_0 energy_end rel_energy_err
    end

    @testset "HEVI vs fully implicit at small dt" begin
        # Both run at dt=0.3s to t=3s — different ODE algorithms (ARS343 vs ARS222)
        # but solutions should be qualitatively similar
        t_compare = 3.0

        cfg_hevi = load_hevi_config(;
            overrides = Dict(
                "dt" => "0.3secs",
                "t_end" => "$(t_compare)secs",
            ),
        )
        sim_hevi = CA.get_simulation(cfg_hevi)
        run_to_time!(sim_hevi, t_compare)
        Y_hevi = copy(sim_hevi.integrator.u)

        cfg_fi = load_fi_config(;
            overrides = Dict(
                "dt" => "0.3secs",
                "t_end" => "$(t_compare)secs",
            ),
        )
        sim_fi = CA.get_simulation(cfg_fi)
        run_to_time!(sim_fi, t_compare)
        Y_fi = copy(sim_fi.integrator.u)

        diff_vec = parent(Y_hevi) .- parent(Y_fi)
        linf_diff = maximum(abs, diff_vec)
        l2_diff = sqrt(sum(abs2, diff_vec))

        # Different algorithms (ARS343 HEVI vs ARS222 fully implicit), so not exact;
        # but should be qualitatively similar (same order of magnitude)
        linf_scale = max(maximum(abs, parent(Y_hevi)), maximum(abs, parent(Y_fi)))
        rel_linf = linf_diff / linf_scale
        @test rel_linf < 0.1
        @info "HEVI vs FI agreement" linf_diff l2_diff

        # Both should have produced nonzero state change from initial
        cfg_init = load_fi_config(;
            overrides = Dict(
                "dt" => "0.3secs",
                "t_end" => "$(t_compare)secs",
            ),
        )
        sim_init = CA.get_simulation(cfg_init)
        Y_init = copy(sim_init.integrator.u)

        change_hevi = maximum(abs, parent(Y_hevi) .- parent(Y_init))
        change_fi = maximum(abs, parent(Y_fi) .- parent(Y_init))
        @test change_hevi > 1e-6
        @test change_fi > 1e-6
        @info "State evolution" change_hevi change_fi
    end

    @testset "dt sweep (stability)" begin
        # Run at multiple dt values — all should complete without crash
        for dt_val in [1.0, 3.0, 9.0]
            t_end_val = max(30.0, 3 * dt_val)
            @testset "dt = $(dt_val)s" begin
                cfg = load_fi_config(;
                    overrides = Dict(
                        "dt" => "$(dt_val)secs",
                        "t_end" => "$(t_end_val)secs",
                    ),
                )
                sim = CA.get_simulation(cfg)
                integ, n = run_to_time!(sim, t_end_val)
                @test integ.t >= t_end_val
                @test !any(isnan, parent(integ.u))
                @info "dt sweep" dt_val t_end_val n_steps = n
            end
        end
    end

    @testset "Float64 fully implicit" begin

        @testset "Tendency decomposition (Float64)" begin
            cfg = load_fi_config_f64(; overrides = Dict(
                "dt" => "1secs",
                "t_end" => "2secs",
            ))
            sim = CA.get_simulation(cfg)
            Y = copy(sim.integrator.u)
            p = sim.integrator.p
            t = Float64(sim.integrator.t)

            CA.set_precomputed_quantities!(Y, p, t)

            Yₜ_full = similar(Y)
            CA.fully_implicit_tendency!(Yₜ_full, Y, p, t)

            Yₜ_exp = similar(Y)
            Yₜ_lim = similar(Y)
            CA.remaining_tendency!(Yₜ_exp, Yₜ_lim, Y, p, t)

            Yₜ_imp = similar(Y)
            CA.implicit_tendency!(Yₜ_imp, Y, p, t)

            sum_parts = Yₜ_exp .+ Yₜ_imp .+ Yₜ_lim
            diff = Yₜ_full .- sum_parts

            l2_diff = sqrt(sum(abs2, parent(diff)))
            l2_full = sqrt(sum(abs2, parent(Yₜ_full)))
            rel_err = l2_full > 0 ? l2_diff / l2_full : 0.0

            @test rel_err < 1e-12
            @info "Tendency decomposition (Float64)" rel_err l2_diff l2_full
        end

        @testset "Mass conservation 30s (Float64)" begin
            cfg = load_fi_config_f64(; overrides = Dict(
                "dt" => "3secs",
                "t_end" => "30secs",
            ))
            sim = CA.get_simulation(cfg)
            integ = sim.integrator
            mass_0 = sum(integ.u.c.ρ)

            run_to_time!(sim, 30.0)
            mass_end = sum(integ.u.c.ρ)

            rel_mass_err = abs(mass_end - mass_0) / abs(mass_0)
            @test rel_mass_err < 1e-10
            @info "Mass conservation (Float64)" mass_0 mass_end rel_mass_err
        end

        @testset "Energy conservation 30s (Float64)" begin
            cfg = load_fi_config_f64(; overrides = Dict(
                "dt" => "3secs",
                "t_end" => "30secs",
            ))
            sim = CA.get_simulation(cfg)
            integ = sim.integrator
            energy_0 = sum(integ.u.c.ρe_tot)

            run_to_time!(sim, 30.0)
            energy_end = sum(integ.u.c.ρe_tot)

            rel_energy_err = abs(energy_end - energy_0) / abs(energy_0)
            @test rel_energy_err < 1e-8
            @info "Energy conservation (Float64)" energy_0 energy_end rel_energy_err
        end

        @testset "State update smoke test (Float64)" begin
            cfg = load_fi_config_f64(; overrides = Dict(
                "dt" => "1secs",
                "t_end" => "2secs",
            ))
            sim = CA.get_simulation(cfg)
            integ = sim.integrator
            u_before = copy(integ.u)
            SciMLBase.step!(integ)
            u_after = integ.u

            diff_vec = parent(u_after) .- parent(u_before)
            l2_change = sqrt(sum(abs2, diff_vec))

            @test l2_change > 1e-12
            @test !any(isnan, parent(u_after))
            @info "State update (Float64)" l2_change
        end

    end
end
