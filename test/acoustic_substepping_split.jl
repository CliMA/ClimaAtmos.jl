using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaTimeSteppers as CTS
import ClimaCore: Fields, Geometry, Spaces
import LinearAlgebra
import LinearAlgebra: I

# On a dry box the outer operator of the inner/outer implicit split (the full
# implicit tendency minus the vertical grid-mean acoustic subset) vanishes, so
# the split sub-cycle reproduces the unsplit one: `ρ`, `ρe_tot`, and `u₃` are
# bitwise identical, and `uₕ` agrees to machine precision (the outer no-op re-applies
# the covariant-vector DSS). Both outer orders are covered; order 1 also exercises
# the first-order `outer_half!` timestep path.

function acoustic_box_config(; order, implicit_split, scheme = "imex")
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
        "acoustic_substep_scheme" => scheme,
    )
end

# Step a fresh simulation and snapshot the state at the requested step counts.
function stepped_states(; order, implicit_split, checkpoints, scheme = "imex")
    config = acoustic_box_config(; order, implicit_split, scheme)
    config["output_dir"] = mktempdir()
    job_id = "acoustic_$(scheme)_o$(order)_$(implicit_split ? "split" : "unsplit")"
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
    for scheme in ("imex", "forward_backward"), order in (1, 2)
        @testset "$scheme outer order $order" begin
            unsplit =
                stepped_states(; order, implicit_split = false, checkpoints, scheme)
            split =
                stepped_states(; order, implicit_split = true, checkpoints, scheme)
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

# The outer implicit complement pairs its residual (the full implicit tendency
# minus the vertical grid-mean acoustic subset) with the matching
# `AcousticComplementJacobian`. The tests below run on a box with implicit
# vertical diffusion, so the complement has a nontrivial residual, and a
# sheared horizontal wind so the momentum diffusion is active. See #4661.

function diffusive_split_integrator()
    config = acoustic_box_config(; order = 1, implicit_split = true)
    config["vert_diff"] = "VerticalDiffusion"
    config["implicit_diffusion"] = true
    config["output_dir"] = mktempdir()
    integrator =
        CA.get_simulation(
            CA.AtmosConfig(config; job_id = "acoustic_complement"),
        ).integrator
    Y = integrator.u
    ᶜz = Fields.coordinate_field(Y.c).z
    z_max = Spaces.z_max(axes(Y.f))
    @. Y.c.uₕ += CA.C12(Geometry.UVVector(5 * sin(ᶜz / z_max * pi), 0 * ᶜz))
    return integrator
end

# Assert that the first-column block arrays satisfy
# `full = acoustic + complement`, plus the identity shift on diagonal keys.
# Entries can be dense matrices or `UniformScaling`s; blocks absent from a
# matrix are structurally zero.
function assert_block_identity(full, acoustic, complement)
    for key in union(keys(full), keys(acoustic), keys(complement))
        shift = key[1] == key[2] ? 1.0 : 0.0
        blocks =
            map(dict -> get(dict, key, 0.0 * I), (full, acoustic, complement))
        matrix_blocks = filter(block -> block isa AbstractMatrix, collect(blocks))
        if isempty(matrix_blocks)
            @test abs(blocks[1].λ - blocks[2].λ - blocks[3].λ - shift) < 1e-10
        else
            dims = size(first(matrix_blocks))
            to_matrix(block) =
                block isa AbstractMatrix ? Matrix{Float64}(block) :
                Matrix{Float64}(block, dims...)
            residual =
                to_matrix(blocks[1]) - to_matrix(blocks[2]) -
                to_matrix(blocks[3]) - Matrix{Float64}(shift * I, dims...)
            scale = max(1.0, maximum(abs, to_matrix(blocks[1])))
            @test maximum(abs, residual) < 1e-10 * scale
        end
    end
end

@testset "Acoustic substepping: outer complement Jacobian" begin
    integrator = diffusive_split_integrator()
    Y, p, t, f = integrator.u, integrator.p, integrator.t, integrator.cache.f
    f.cache!(Y, p, t)
    FT = eltype(Y)
    # Linearization step well beyond the box's horizontal acoustic CFL (~3 s),
    # the regime the implicit split targets.
    dtγ = FT(20)

    @testset "wiring: complement residual paired with complement Jacobian" begin
        complement = integrator.cache.cts_cache.outercache.outer.complement
        outer_T_imp! = complement.outer_integ.sol.prob.f.T_imp!
        @test outer_T_imp!.f isa CA.OuterImplicitTendency
        @test outer_T_imp!.jac_prototype.alg isa CA.AcousticComplementJacobian
        @test outer_T_imp!.jac_prototype.alg.approximate_solve_iters ==
              f.T_imp!.jac_prototype.alg.approximate_solve_iters
    end

    @testset "acoustic + complement blocks reproduce the full blocks" begin
        # With W = dtγ ∂T/∂Y - I and T_full = T_acoustic + T_complement, the
        # blocks satisfy W_full = W_acoustic + W_complement + I on the diagonal
        # and W_full = W_acoustic + W_complement off it. Exact for
        # configurations without sedimenting condensate masses, whose
        # pressure-gradient couplings `AcousticJacobian` omits.
        full = CA.first_column_block_arrays(
            CA.ManualSparseJacobian(; approximate_solve_iters = 2), Y, p, dtγ, t,
        )
        acoustic =
            CA.first_column_block_arrays(CA.AcousticJacobian(), Y, p, dtγ, t)
        complement = CA.first_column_block_arrays(
            CA.AcousticComplementJacobian(; approximate_solve_iters = 2),
            Y, p, dtγ, t,
        )
        assert_block_identity(full, acoustic, complement)
    end

    @testset "complement Newton update quality" begin
        # One Newton update of the complement stage equation
        # U = u0 + dtγ T_complement(U), from the predictor U = u0. The
        # complement Jacobian must reduce the residual at least as much as a
        # pairing with the full-tendency Jacobian, whose acoustic blocks the
        # residual excludes. See #4661.
        u0 = deepcopy(Y)
        T_complement! = CA.OuterImplicitTendency(CA.implicit_tendency!, zero(Y))
        function complement_residual(U)
            CA.set_implicit_precomputed_quantities!(U, p, t)
            r = zero(U)
            T_complement!(r, U, p, t)
            @. r = u0 + dtγ * r - U
            return r
        end
        function newton_update(alg)
            jacobian = CA.Jacobian(alg, u0, p.atmos)
            CA.set_implicit_precomputed_quantities!(u0, p, t)
            CA.update_jacobian!(jacobian, u0, p, dtγ, t)
            ΔU = zero(u0)
            LinearAlgebra.ldiv!(ΔU, jacobian, r0)
            U1 = zero(u0)
            @. U1 = u0 - ΔU
            return U1, ΔU
        end

        r0 = complement_residual(u0)
        U1_full, ΔU_full = newton_update(CA.ManualSparseJacobian())
        U1_comp, ΔU_comp = newton_update(CA.AcousticComplementJacobian())
        r1_full = complement_residual(U1_full)
        r1_comp = complement_residual(U1_comp)

        # Scalar rows: the update reduces the residual by at least as much as
        # the full-Jacobian pairing, up to rounding-level differences.
        for name in (:uₕ, :ρe_tot)
            residual_norm(r) = maximum(abs, parent(getproperty(r.c, name)))
            r0_norm = residual_norm(r0)
            @test r0_norm > 0
            @test residual_norm(r1_comp) < r0_norm
            @test residual_norm(r1_comp) <= residual_norm(r1_full) * 1.001
        end
        # The complement residual has no u₃ component, so the consistent
        # pairing leaves u₃ unchanged and its residual zero; the full-Jacobian
        # pairing perturbs u₃ and leaves a residual there.
        @test maximum(abs, parent(ΔU_comp.f.u₃)) == 0
        @test maximum(abs, parent(r1_comp.f.u₃)) == 0
        @test maximum(abs, parent(ΔU_full.f.u₃)) > 0
        @test maximum(abs, parent(r1_full.f.u₃)) > 0
    end
end

@testset "Acoustic substepping: implicit split completes a timestep sweep" begin
    for dt in ("0.5secs", "1secs", "2secs")
        config = acoustic_box_config(; order = 1, implicit_split = true)
        config["dt"] = dt
        config["acoustic_substeps"] = "auto"
        config["output_dir"] = mktempdir()
        integrator =
            CA.get_simulation(
                CA.AtmosConfig(config; job_id = "split_sweep_$dt"),
            ).integrator
        t_start = integrator.t
        for _ in 1:5
            CTS.step!(integrator)
        end
        @test integrator.t > t_start
        @test all(isfinite, parent(integrator.u.c.ρ))
        @test all(isfinite, parent(integrator.u.f.u₃))
    end
end
