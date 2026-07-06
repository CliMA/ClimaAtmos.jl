#=
Unit tests comparing the sparse autodiff Jacobian (AutoSparseJacobian) against
the exact dense autodiff Jacobian (AutoDenseJacobian) on the first column of a
single-column EDMF simulation.

The sparse algorithm recovers Jacobian entries from dual number partials that
are shared between same-colored columns of the sparsity structure, so entries
outside of the structure can alias into ("poison") the recovered entries. The
recovery error is isolated by comparing entries only on the stored bands of
each block: differences off the stored bands are the sparsity structure's
deliberate truncation of the Jacobian, which is identical for every
AutoSparseJacobian variant and is not what these tests measure (it is reported
alongside the aliasing error for context).

The tests lock in two guarantees:
1. With the default padding bands, aliasing errors on every stored band are
   small compared to the significance threshold of the implicit solver
   (normalized magnitudes of order 1/dt).
2. With seed scaling (`seed_scaling = :static`) and no padding bands at all,
   the same bound holds: scaling each column's seed by its state variable's
   typical increment magnitude suppresses aliasing errors wherever the
   increment-weighted criterion for dropping entries from the sparsity
   structure applies.
They also verify that the seed scaling plumbing reproduces the unscaled
entries exactly when all scales are 1 (which requires the deterministic
coloring guaranteed by AutoSparseJacobian), and they report the aliasing error
of the naive unscaled and unpadded variant for calibration.

To iterate on this file without repeatedly compiling a fresh package test
environment, run it directly in the .buildkite environment:
    julia +release --project=.buildkite -e 'include("test/implicit/jacobian_comparison.jl")'
=#

using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA

include("../test_helpers.jl")

# LinearAlgebra and ForwardDiff are reached through ClimaAtmos so that this
# file also runs in environments that do not declare them (e.g. .buildkite).
const LA = CA.LinearAlgebra

rms(block) = sqrt(sum(abs2, block) / length(block))

# Boolean mask of the stored band positions of a block, inferred from the
# manual Jacobian's block array: band d is stored if any entry on it is
# nonzero. The diagonal of ∂R/∂Y is always nonzero (from the -I term), and
# the other stored bands are generically nonzero after a few time steps; a
# stored band that is identically zero at this state is excluded, which only
# makes the aliasing metric less strict at that band.
function stored_band_mask(manual_block)
    mask = falses(size(manual_block))
    for band in (1 - size(manual_block, 1)):(size(manual_block, 2) - 1)
        any(!iszero, LA.diag(manual_block, band)) || continue
        mask[LA.diagind(manual_block, band)] .= true
    end
    return mask
end

# Errors of the given sparse Jacobian algorithm's blocks relative to the dense
# reference blocks, maximized over all non-constant blocks in the sparsity
# structure. Each block's entrywise difference is normalized by typical
# increment ratios [s^-1] and multiplied by dt to make it dimensionless;
# max_aliasing_error is the RMS restricted to stored band positions (pure
# recovery error), while max_full_error also includes the truncation of the
# sparsity structure itself.
function block_errors(alg, Y, p, dtγ, t, dense_blocks, manual_blocks, rescalings)
    FT = eltype(Y)
    dt = FT(float(p.dt))
    blocks = CA.first_column_block_arrays(alg, Y, p, dtγ, t)
    max_aliasing_error = zero(FT)
    max_full_error = zero(FT)
    for (block_key, block) in blocks
        block isa LA.UniformScaling && continue
        manual_blocks[block_key] isa LA.UniformScaling && continue
        difference =
            (dense_blocks[block_key] .- block) .* rescalings[block_key] .* dt
        mask = stored_band_mask(manual_blocks[block_key])
        max_aliasing_error = max(max_aliasing_error, rms(difference .* mask))
        max_full_error = max(max_full_error, rms(difference))
    end
    return (; max_aliasing_error, max_full_error)
end

@testset "sparse vs dense autodiff Jacobian agreement" begin
    # Single-column GABLS with prognostic EDMF and 1M microphysics, mirroring
    # config/model_configs/prognostic_edmfx_gabls_column_sparse_autodiff.yml;
    # this exercises most Jacobian blocks, including the SGS advection,
    # diffusion, and mass flux blocks.
    config = CA.AtmosConfig(
        Dict(
            "initial_condition" => "GABLS",
            "turbconv" => "prognostic_edmfx",
            "implicit_diffusion" => true,
            "approximate_linear_solve_iters" => 2,
            "ode_algo" => "ARS222",
            "edmfx_entr_model" => "Generalized",
            "edmfx_detr_model" => "Generalized",
            "edmfx_sgs_mass_flux" => true,
            "edmfx_sgs_diffusive_flux" => true,
            "edmfx_nh_pressure" => true,
            "edmfx_vertical_diffusion" => true,
            "edmfx_filter" => true,
            "prognostic_tke" => true,
            "microphysics_model" => "1M",
            "config" => "column",
            "hyperdiff" => nothing,
            "z_max" => 400,
            "x_elem" => 2,
            "y_elem" => 2,
            "z_elem" => 8,
            "z_stretch" => false,
            "dt" => "10secs",
            "t_end" => "2mins", # enough steps to reach a non-degenerate state
            "perturb_initstate" => false,
            "toml" => ["toml/prognostic_edmfx.toml"],
            "FLOAT_TYPE" => "Float64",
            "output_default_diagnostics" => false,
        ),
        job_id = "jacobian_comparison_gabls",
    )
    (; simulation) = generate_test_simulation(config)
    CA.solve_atmos!(simulation)
    (; integrator) = simulation
    Y = integrator.u
    (; p, t) = integrator
    FT = eltype(Y)
    dtγ = FT(float(p.dt))

    dense_blocks =
        CA.first_column_block_arrays(CA.AutoDenseJacobian(), Y, p, dtγ, t)
    manual_blocks = CA.first_column_block_arrays(
        CA.ManualSparseJacobian(; approximate_solve_iters = 2),
        Y,
        p,
        dtγ,
        t,
    )
    rescalings = CA.first_column_rescaling_arrays(Y, p, t)

    errors(alg) = block_errors(
        alg,
        Y,
        p,
        dtγ,
        t,
        dense_blocks,
        manual_blocks,
        rescalings,
    )
    sparse_alg(; kwargs...) =
        CA.AutoSparseJacobian(; approximate_solve_iters = 2, kwargs...)

    # Provisional bound on the normalized aliasing error; entries become
    # significant to the implicit solver when this metric approaches 1.
    tolerance = 1e-2

    # The current default configuration: unscaled seeds, default padding.
    errors_padded = errors(sparse_alg())
    @info "Errors with unscaled seeds and default padding: $errors_padded"
    @test errors_padded.max_aliasing_error < tolerance

    # Seed scaling should make all padding bands unnecessary.
    errors_scaled_unpadded = errors(
        sparse_alg(; padding_bands_per_block = 0, seed_scaling = :static),
    )
    @info "Errors with scaled seeds and no padding: $errors_scaled_unpadded"
    @test errors_scaled_unpadded.max_aliasing_error < tolerance

    # Calibration report for the naive variant, which is expected to violate
    # the bound whenever the coloring assigns the same color to a column with
    # large out-of-pattern entries and a column of a small in-pattern block.
    # Scaling must never do worse than the naive variant, but the naive error
    # is not asserted against the bound, since it depends on the specific
    # coloring; add that assertion once the error proves stably large.
    errors_naive = errors(sparse_alg(; padding_bands_per_block = 0))
    @info "Errors with unscaled seeds and no padding: $errors_naive"
    @test errors_scaled_unpadded.max_aliasing_error <=
          errors_naive.max_aliasing_error

    # With all seed scales set to 1, the seed scaling code path reproduces the
    # unscaled entries exactly. This relies on the coloring being
    # deterministic across cache constructions (RandomOrder is excluded from
    # the coloring order sweep in AutoSparseJacobian).
    blocks_unscaled = CA.first_column_block_arrays(sparse_alg(), Y, p, dtγ, t)
    blocks_unit_scaled = CA.first_column_block_arrays(
        sparse_alg(; seed_scaling = Returns(1)),
        Y,
        p,
        dtγ,
        t,
    )
    @test all(keys(blocks_unscaled)) do block_key
        blocks_unit_scaled[block_key] == blocks_unscaled[block_key]
    end

    # Dropping the padding bands reduces the number of colors, and therefore
    # the number of ε components per dual number (on CPU devices, the number
    # of ε components is the number of colors).
    n_εs(alg) = CA.ForwardDiff.npartials(
        eltype(
            parent(
                CA.jacobian_cache(alg, Y, p.atmos; verbose = false).Y_dual.c.ρ,
            ),
        ),
    )
    n_εs_padded = n_εs(sparse_alg())
    n_εs_scaled_unpadded =
        n_εs(sparse_alg(; padding_bands_per_block = 0, seed_scaling = :static))
    @info "ε components per dual number: $n_εs_padded with default padding, \
           $n_εs_scaled_unpadded without padding"
    @test n_εs_scaled_unpadded <= n_εs_padded
end
