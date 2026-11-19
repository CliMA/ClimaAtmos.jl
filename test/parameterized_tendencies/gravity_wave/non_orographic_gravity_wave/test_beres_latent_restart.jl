# ============================================================================
# Beres applied latent heating across a restart (`nogw_beres_heating_latent`).
#
# The Beres latent heating is the mean of the heating the implicit stages applied
# since the last NOGW call. Its integrals (`gw_latent_*_acc`) live in the NOGW cache,
# are written to every checkpoint and read back on restart. The window is reset by the
# first implicit stage after a NOGW call (`gw_latent_window_used`), so a checkpoint
# written at a NOGW call holds the window that call used.
#
# A deep-convection column (TRMM-LBA, 1M, prognostic EDMF) runs 4 h with dt = 120 s,
# dt_nogw = 600 s and checkpoints every hour (6 NOGW intervals), then runs again from
# the 3 h checkpoint to 4 h. The solver is the production one: 3 Newton iterations with
# the Jacobian updated at every linear solve. The updrafts first heat above
# `nogw_beres_z_bot_floor` after about 3 h, so the Beres source is active over the
# restarted hour; the rain threshold is lowered to 0.01 mm/day so that the rain gate
# does not hold the source back while the convection is young. The restarted run must
# give the same state, drag, Beres source and flux profiles, and latent-heating
# integrals, bit for bit. Then the window
# bookkeeping is checked: the integrals at the end cover one NOGW interval, and the
# next stage starts a new window.
#
# Run on the CPU (the checks read fields back with scalar indexing):
#   CLIMACOMMS_DEVICE=CPU julia --project=.buildkite \
#       test/parameterized_tendencies/gravity_wave/non_orographic_gravity_wave/test_beres_latent_restart.jl
# ============================================================================

using Test
import ClimaAtmos as CA
import ClimaComms
import YAML
pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends

const comms_ctx = ClimaComms.context(ClimaComms.CPUSingleThreaded())
ClimaComms.init(comms_ctx)

repo = joinpath(@__DIR__, "..", "..", "..", "..")
output_dir = mktempdir()
config_dict = merge(
    YAML.load_file(
        joinpath(repo, "config", "model_configs", "prognostic_edmfx_trmm_column.yml"),
    ),
    Dict(
        "non_orographic_gravity_wave" => true,
        "nogw_beres_source" => true,
        "nogw_beres_heating_latent" => true,
        "nogw_beres_rain_threshold" => 0.01,
        "nogw_beres_detailed_diagnostics" => true,
        "max_newton_iters_ode" => 3,
        "update_jacobian_every" => "solve",
        "dt" => "120secs",
        "dt_nogw" => "600secs",
        "dt_save_state_to_disk" => "3600secs",
        "t_end" => "14400secs",
        "reproducible_restart" => true,
        "reproducibility_test" => false,
        "debug_tendency_diagnostics" => false,
        "enable_diagnostics" => false,
        "log_progress" => false,
        "output_dir" => output_dir,
        "toml" => [
            joinpath(repo, "toml", "prognostic_edmfx_1M.toml"),
            joinpath(repo, "toml", "nogw_beres_test.toml"),
        ],
    ),
)
job_id = "beres_latent_restart"

simulation = CA.get_simulation(CA.AtmosConfig(config_dict; job_id, comms_ctx))
CA.solve_atmos!(simulation)
restart_file = joinpath(simulation.output_dir, "day0.10800.hdf5")
@test isfile(restart_file)
restarted = CA.get_simulation(
    CA.AtmosConfig(
        merge(config_dict, Dict("restart_file" => restart_file));
        job_id,
        comms_ctx,
    ),
)
CA.solve_atmos!(restarted)

nogw = simulation.integrator.p.non_orographic_gravity_wave
nogw_r = restarted.integrator.p.non_orographic_gravity_wave
column(field) = vec(Array(parent(field)))

@testset "Beres latent heating: continuous and restarted runs agree" begin
    @test simulation.integrator.t == restarted.integrator.t
    # The test has power: the updrafts condensed, the Beres source is active and
    # launched waves, and its flux profiles start at the launched flux.
    @test maximum(abs, column(nogw.gw_latent_heat_acc)) > 0
    @test maximum(abs, column(nogw.uforcing)) > 0
    @test maximum(column(nogw.gw_beres_active)) > 0
    @test maximum(column(nogw.gw_launch_flux)) > 0
    east = column(nogw.gw_beres_flux_east_u)
    west = column(nogw.gw_beres_flux_west_u)
    @test all(>=(0), east) && all(>=(0), west)
    @test maximum(east .+ west) > 0
    launch_level = Int(column(nogw.beres_source_ρ_z_u_v_level.:5)[1])
    @test east[launch_level] + west[launch_level] ≈ column(nogw.gw_launch_flux)[1] rtol =
        1e-5
    @info "Beres at the end: active $(column(nogw.gw_beres_active)), launched flux $(column(nogw.gw_launch_flux)) Pa, E + W at launch level $launch_level = $(east[launch_level] + west[launch_level]) Pa; max |uforcing| = $(maximum(abs, column(nogw.uforcing)))"
    @test parent(simulation.integrator.u.c) == parent(restarted.integrator.u.c)
    @test parent(simulation.integrator.u.f) == parent(restarted.integrator.u.f)
    for name in (
        :uforcing,
        :vforcing,
        :gw_latent_heat_acc,
        :gw_latent_mass_acc,
        :gw_latent_area_acc,
        :gw_Q_conv,
        :gw_Q_conv_ic,
        :gw_Q0,
        :gw_beres_active,
        :gw_launch_flux,
        :gw_beres_flux_east_u,
        :gw_beres_flux_west_u,
    )
        @test column(getproperty(nogw, name)) == column(getproperty(nogw_r, name)) ||
              (@info "differs: $name"; false)
    end
    for j in 1:length(nogw.gw_latent_lin)
        @test parent(nogw.gw_latent_lin[j]) == parent(nogw_r.gw_latent_lin[j])
    end
    @test nogw.gw_latent_time_acc[] == nogw_r.gw_latent_time_acc[]
    @test nogw.gw_latent_window_used[] == nogw_r.gw_latent_window_used[]
end

@testset "Beres latent heating: the window covers one NOGW interval, then resets" begin
    # The run ends on a NOGW call (14400 s = 24 × 600 s), which used the window and marked
    # it used; the window covered the five steps since the previous call.
    @test nogw.gw_latent_window_used[]
    @test nogw.gw_latent_time_acc[] ≈ 600 rtol = 1e-12
    # The next implicit stage resets the window and adds its own share only.
    Y = simulation.integrator.u
    p = simulation.integrator.p
    FT = eltype(Y)
    CA.accumulate_beres_latent_heating!(Y, p)
    k = p.scratch.precip_stage_idx[]
    w = FT(p.scratch.b_imp_stages[k]) * FT(p.dt)
    @test !nogw.gw_latent_window_used[]
    @test nogw.gw_latent_time_acc[] == w
    ᶜρʲ = p.precomputed.ᶜρʲs.:(1)
    ᶜρa = Y.c.sgsʲs.:(1).ρa
    expected_mass = @. ifelse(ᶜρʲ > eps(FT), w * max(ᶜρa, FT(0)), FT(0))
    @test column(nogw.gw_latent_mass_acc) == column(expected_mass)
    @test maximum(column(expected_mass)) > 0
end

@testset "Beres latent heating needs the stage accumulator" begin
    # update_jacobian_every = "dt" disables the stage-weighted accumulator, so the
    # model must refuse to build rather than read the stored rates at every call.
    no_acc = merge(
        config_dict,
        Dict("update_jacobian_every" => "dt", "output_dir" => mktempdir()),
    )
    message = try
        CA.get_simulation(CA.AtmosConfig(no_acc; job_id = "beres_latent_no_acc", comms_ctx))
        ""
    catch err
        sprint(showerror, err)
    end
    @test occursin("stage-weighted implicit accumulator", message)
end
