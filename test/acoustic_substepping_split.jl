using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaTimeSteppers as CTS

# On a dry box the outer operator of the inner/outer implicit split (the full
# implicit tendency minus the vertical grid-mean acoustic subset) vanishes, so
# the split sub-cycle reproduces the unsplit one: `ρ`, `ρe_tot`, and `u₃` are
# bitwise identical, and `uₕ` agrees to machine precision (the outer no-op re-applies
# the covariant-vector DSS). Both outer orders are covered; order 1 also exercises
# the first-order `outer_half!` timestep path.

function acoustic_box_config(; order, implicit_split)
    return Dict{String, Any}(
        "initial_condition" => "DryDensityCurrentProfile",
        "config" => "box",
        "FLOAT_TYPE" => "Float64",
        "hyperdiff" => nothing,
        "smagorinsky_lilly" => nothing,
        "x_max" => 6400.0,
        "y_max" => 6400.0,
        "z_max" => 6400.0,
        "x_elem" => 2,
        "y_elem" => 2,
        "z_elem" => 8,
        "z_stretch" => false,
        "dt" => "0.5secs",
        "t_end" => "60secs",
        "disable_surface_flux_tendency" => true,
        "output_default_diagnostics" => false,
        "dt_save_state_to_disk" => "Inf",
        "log_progress" => false,
        "acoustic_substeps" => "3",
        "acoustic_substep_vertical" => "implicit",
        "acoustic_substep_order" => order,
        "acoustic_substep_damping" => 1.5,
        "acoustic_substep_implicit_split" => implicit_split,
    )
end

# Step a fresh simulation and snapshot the state at the requested step counts.
function stepped_states(; order, implicit_split, checkpoints)
    config = acoustic_box_config(; order, implicit_split)
    config["output_dir"] = mktempdir()
    job_id = "acoustic_o$(order)_$(implicit_split ? "split" : "unsplit")"
    integrator = CA.get_simulation(CA.AtmosConfig(config; job_id)).integrator
    states = Dict{Int, Any}()
    for step in 1:maximum(checkpoints)
        CTS.step!(integrator)
        step in checkpoints && (states[step] = deepcopy(integrator.u))
    end
    return states
end

@testset "Acoustic substepping: implicit split reproduces the unsplit sub-cycle" begin
    checkpoints = (2, 4)
    uₕ_relative_tolerance = 1e-12
    for order in (1, 2)
        @testset "outer order $order" begin
            unsplit = stepped_states(; order, implicit_split = false, checkpoints)
            split = stepped_states(; order, implicit_split = true, checkpoints)
            for n in checkpoints
                u_unsplit, u_split = unsplit[n], split[n]
                # ρ, ρe_tot, u₃ are bitwise identical.
                @test parent(u_unsplit.c.ρ) == parent(u_split.c.ρ)
                @test parent(u_unsplit.c.ρe_tot) == parent(u_split.c.ρe_tot)
                @test parent(u_unsplit.f.u₃) == parent(u_split.f.u₃)
                @test all(isfinite, parent(u_unsplit.c.ρe_tot))
                # uₕ agrees to machine precision.
                uₕ_unsplit = parent(u_unsplit.c.uₕ)
                uₕ_split = parent(u_split.c.uₕ)
                relative_difference =
                    maximum(abs, uₕ_unsplit .- uₕ_split) /
                    maximum(abs, uₕ_unsplit)
                @test relative_difference < uₕ_relative_tolerance
            end
        end
    end
end
