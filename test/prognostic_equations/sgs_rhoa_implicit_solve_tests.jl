#=
Unit tests for the analytic implicit-stage solve of the updraft `ρa`
(`solve_sgs_ρa_implicit_stage_analytic!`), focused on the microphysics
mass source `ρa · dq_tot_dt`:
  - the source enters the implicit stage value and the cached tendency
    forwarded by `implicit_tendency!` (implicit microphysics timestepping),
  - a strong sink at large `dtγ` keeps `ρa ≥ 0`,
  - under explicit microphysics timestepping the source is applied by the
    implicit-stage solve, not by `microphysics_tendency!`.
=#
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA

include("../test_helpers.jl")

function create_sgs_ρa_config(job_id; implicit_microphysics)
    config_dict = Dict(
        "config" => "column",
        "initial_condition" => "DecayingProfile",
        "turbconv" => "prognostic_edmfx",
        "microphysics_model" => "0M",
        "FLOAT_TYPE" => "Float32",
        "implicit_microphysics" => implicit_microphysics,
        "output_default_diagnostics" => false,
    )
    return CA.AtmosConfig(config_dict; job_id)
end

# Zero all recurrence inputs except the microphysics mass source, so the
# solve reduces to `ρa_new = ρa_old / (1 - dtγ · dq_tot_dt)` per cell.
function isolate_microphysics_source!(Y, p)
    n = CA.n_mass_flux_subdomains(p.atmos.turbconv_model)
    for j in 1:n
        parent(Y.f.sgsʲs.:($j).u₃) .= 0
        parent(p.precomputed.ᶜentr_vel_scaleʲs.:($j)) .= 0
        parent(p.precomputed.ᶜarea_bounding_entr_detrʲs.:($j)) .= 0
        parent(p.precomputed.ᶜρ_diffʲs.:($j)) .= 0
        parent(p.precomputed.sfc_mass_flux_sourceʲs.:($j)) .= 0
    end
    return nothing
end

function set_updraft_state!(Y, p; area, rate)
    FT = eltype(Y)
    n = CA.n_mass_flux_subdomains(p.atmos.turbconv_model)
    for j in 1:n
        parent(Y.c.sgsʲs.:($j).ρa) .= FT(area) .* parent(Y.c.ρ)
        parent(p.precomputed.ᶜmp_tendencyʲs.:($j).dq_tot_dt) .= FT(rate)
    end
    return nothing
end

@testset "Microphysics ρa source in the implicit stage solve" begin
    config = create_sgs_ρa_config(
        "sgs_rhoa_implicit_micro";
        implicit_microphysics = true,
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    FT = eltype(Y)
    t = simulation.integrator.t
    n = CA.n_mass_flux_subdomains(p.atmos.turbconv_model)
    dtγ = FT(100)
    rate = FT(-0.005) # sink: 1 - dtγ · rate = 1.5

    isolate_microphysics_source!(Y, p)
    set_updraft_state!(Y, p; area = 0.1, rate)
    ρa_old = copy(Y.c.sgsʲs)

    CA.initialize_implicit_stage_problem!(Y, p, dtγ)

    (; ᶜρa_tendencyʲs) = p.precomputed
    for j in 1:n
        @test parent(Y.c.sgsʲs.:($j).ρa) ≈
              parent(ρa_old.:($j).ρa) ./ (1 - dtγ * rate)
        @test parent(ᶜρa_tendencyʲs.:($j)) ≈
              (parent(Y.c.sgsʲs.:($j).ρa) .- parent(ρa_old.:($j).ρa)) ./ dtγ
    end

    # `implicit_tendency!` forwards the cached tendency, so the microphysics
    # ρa source is not dropped by the overwrite in `sgs_ρa_implicit_tendency!`.
    Yₜ = similar(Y)
    CA.implicit_tendency!(Yₜ, Y, p, t)
    for j in 1:n
        @test parent(Yₜ.c.sgsʲs.:($j).ρa) ≈ parent(ᶜρa_tendencyʲs.:($j))
    end
end

@testset "Sign safety for a strong microphysics sink at large dtγ" begin
    config = create_sgs_ρa_config(
        "sgs_rhoa_sign_safety";
        implicit_microphysics = true,
    )
    (; Y, p) = generate_test_simulation(config)
    FT = eltype(Y)
    n = CA.n_mass_flux_subdomains(p.atmos.turbconv_model)

    set_updraft_state!(Y, p; area = 0.1, rate = -1e3)
    CA.solve_sgs_ρa_implicit_stage_analytic!(Y, p, FT(1e4))

    for j in 1:n
        @test all(isfinite, parent(Y.c.sgsʲs.:($j).ρa))
        @test minimum(parent(Y.c.sgsʲs.:($j).ρa)) >= 0
    end
end

@testset "Explicit microphysics timestepping: ρa source location" begin
    config = create_sgs_ρa_config(
        "sgs_rhoa_explicit_micro";
        implicit_microphysics = false,
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    FT = eltype(Y)
    t = simulation.integrator.t
    n = CA.n_mass_flux_subdomains(p.atmos.turbconv_model)
    dtγ = FT(100)
    rate = FT(-0.005)

    set_updraft_state!(Y, p; area = 0.1, rate)

    # `microphysics_tendency!` does not apply the updraft ρa source; the
    # grid-mean sources are unaffected.
    Yₜ = similar(Y)
    Yₜ .= zero(eltype(Yₜ))
    CA.microphysics_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.microphysics_model,
        p.atmos.turbconv_model,
    )
    for j in 1:n
        @test all(iszero, parent(Yₜ.c.sgsʲs.:($j).ρa))
    end
    @test !all(iszero, parent(Yₜ.c.ρq_tot))

    # The implicit-stage solve applies the source under explicit microphysics
    # timestepping as well.
    isolate_microphysics_source!(Y, p)
    ρa_old = copy(Y.c.sgsʲs)
    CA.solve_sgs_ρa_implicit_stage_analytic!(Y, p, dtγ)
    for j in 1:n
        @test parent(Y.c.sgsʲs.:($j).ρa) ≈
              parent(ρa_old.:($j).ρa) ./ (1 - dtγ * rate)
    end
end
