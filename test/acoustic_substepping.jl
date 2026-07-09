using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaTimeSteppers as CTS

# On a dry box the outer operator of the inner/outer implicit split (the full
# implicit tendency minus the vertical grid-mean acoustic subset) vanishes, so
# the split sub-cycle reproduces the unsplit one: `ρ`, `ρe_tot`, and `u₃` are
# bit-identical, and `uₕ` agrees to machine precision (the outer no-op re-applies
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

# Box config carrying hyperdiffusion, for the coefficient-scaling checks.
function hyperdiff_scaling_config(; acoustic_substeps, scaling, dt)
    return Dict{String, Any}(
        "initial_condition" => "DryDensityCurrentProfile",
        "config" => "box",
        "FLOAT_TYPE" => "Float64",
        "hyperdiff" => "Hyperdiffusion",
        "smagorinsky_lilly" => nothing,
        "x_max" => 6400.0,
        "y_max" => 6400.0,
        "z_max" => 6400.0,
        "x_elem" => 2,
        "y_elem" => 2,
        "z_elem" => 8,
        "z_stretch" => false,
        "dt" => dt,
        "t_end" => "60secs",
        "acoustic_substeps" => acoustic_substeps,
        "acoustic_substep_hyperdiffusion_scaling" => scaling,
    )
end

# Build the model and grid the way `get_simulation` does, without the cache.
function model_and_grid(config_dict; job_id)
    config = CA.AtmosConfig(config_dict; job_id)
    pa = config.parsed_args
    params = CA.ClimaAtmosParameters(config)
    setup = CA.get_setup_type(pa, CA.CAP.thermodynamics_params(params))
    model = CA.get_atmos(config, params; setup_type = setup)
    grid = CA.get_grid(pa, params, config.comms_ctx)
    return (; model, grid, parsed_args = pa)
end

@testset "Acoustic substepping: hyperdiffusion coefficient scaling" begin
    hyperdiff = CA.Hyperdiffusion{Float64}(;
        ν₄_vorticity_coeff = 0.1857,
        divergence_damping_factor = 5.0,
        prandtl_number = 0.2,
    )
    Δx_node = 113.0
    F = max(5.0, inv(0.2))
    Δt_hd_limit = 2 * Δx_node / (F * 0.1857 * 4.0^4)

    @testset "auto arithmetic and real pass-through" begin
        # A real factor is returned unchanged (unclamped), independent of the limit.
        @test CA.acoustic_hyperdiffusion_scale(0.5, hyperdiff, Δx_node, 1.0) == 0.5
        @test CA.acoustic_hyperdiffusion_scale(1.0, hyperdiff, Δx_node, 1.0) == 1.0
        @test CA.acoustic_hyperdiffusion_scale(2.0, hyperdiff, Δx_node, 1e-6) == 2.0

        # auto clamps to 1 when the outer step is below the limit.
        @test CA.acoustic_hyperdiffusion_scale("auto", hyperdiff, Δx_node, 1e-4) ==
              1.0

        # auto matches the hand-computed factor at a large outer step.
        dt_outer = 2.0
        expected = min(1.0, Δt_hd_limit / (2 * dt_outer))
        s = CA.acoustic_hyperdiffusion_scale("auto", hyperdiff, Δx_node, dt_outer)
        @test s ≈ expected
        @test s < 1.0
    end

    @testset "config accepts auto or a real factor" begin
        # The key takes either the "auto" sentinel or a number, so it must bypass
        # scalar coercion to the String default (EXCEPTED_KEYS).
        for scaling in ("auto", 1.0, 0.5, 2)
            config = CA.AtmosConfig(
                hyperdiff_scaling_config(;
                    acoustic_substeps = "4",
                    scaling,
                    dt = "10secs",
                );
                job_id = "hyperdiff_scaling_coerce",
            )
            @test config.parsed_args["acoustic_substep_hyperdiffusion_scaling"] ==
                  scaling
        end
    end

    @testset "recovery identities" begin
        configured = 0.1857
        coeff(model) = model.numerics.hyperdiff.ν₄_vorticity_coeff
        (; model, grid, parsed_args) = model_and_grid(
            hyperdiff_scaling_config(;
                acoustic_substeps = "4",
                scaling = "auto",
                dt = "10secs",
            );
            job_id = "hyperdiff_scaling",
        )

        # acoustic_substeps = 0: the scaling key is ignored, the model is untouched.
        parsed_off = copy(parsed_args)
        parsed_off["acoustic_substeps"] = "0"
        off = CA.scale_hyperdiffusion_under_acoustic_substepping(
            model, grid, parsed_off,
        )
        @test off === model
        @test coeff(off) == configured

        # Real 1.0 under substepping leaves the coefficient unchanged.
        parsed_one = copy(parsed_args)
        parsed_one["acoustic_substep_hyperdiffusion_scaling"] = 1.0
        one_scaled = CA.scale_hyperdiffusion_under_acoustic_substepping(
            model, grid, parsed_one,
        )
        @test one_scaled === model
        @test coeff(one_scaled) == configured

        # auto under substepping folds the reduction into the stored coefficient.
        Δx = CA.Spaces.node_horizontal_length_scale(
            CA.Spaces.horizontal_space(CA.get_spaces(grid).center_space),
        )
        s = CA.acoustic_hyperdiffusion_scale("auto", model.numerics.hyperdiff, Δx, 10.0)
        auto_scaled = CA.scale_hyperdiffusion_under_acoustic_substepping(
            model, grid, parsed_args,
        )
        @test s < 1
        @test auto_scaled !== model
        @test coeff(auto_scaled) ≈ s * configured
        # Only the vorticity coefficient is scaled; the Prandtl number and
        # divergence-damping factor are unchanged.
        @test auto_scaled.numerics.hyperdiff.prandtl_number == 0.2
        @test auto_scaled.numerics.hyperdiff.divergence_damping_factor == 5.0
    end
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
                # ρ, ρe_tot, u₃ are bit-identical.
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
