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
1. With unscaled seeds and the default padding bands, aliasing errors on
   every stored band are small compared to the significance threshold of the
   implicit solver (normalized magnitudes of order 1/dt).
2. With seed scaling (`seed_scaling = :static`) and the default padding
   bands, the same bound holds. Scaling suppresses cross-field aliasing
   wherever the increment-weighted criterion for dropping entries from the
   sparsity structure applies; within-field band deficits (whose seed scale
   ratios are 1) remain the padding bands' job, so the zero-padding scaled
   variant is reported for calibration but not asserted against the bound.
They also verify that the seed scaling plumbing reproduces the unscaled
entries exactly when all scales are 1 (which requires the deterministic
coloring guaranteed by AutoSparseJacobian), report the aliasing error of the
naive unscaled and unpadded variant, and, for any variant that exceeds the
bound, rank the same-color contributors into its worst block to identify the
field whose seed scale or padding needs adjustment.

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

# Report any non-finite entries so that non-finite error metrics can be
# traced to their source (the dense reference, the rescalings, or a sparse
# variant's recovered entries).
function report_nonfinite_entries(blocks, label)
    for (block_key, block) in blocks
        block isa LA.UniformScaling && continue
        nonfinite_count = count(!isfinite, block)
        nonfinite_count == 0 && continue
        indices = findall(!isfinite, block)
        @warn "$label block $block_key has $nonfinite_count non-finite \
               entries in a block of size $(size(block)), at indices \
               $(first(indices, min(4, length(indices))))"
    end
end

# Errors of the given sparse Jacobian algorithm's blocks relative to the dense
# reference blocks, maximized over all non-constant blocks in the sparsity
# structure. Each block's entrywise difference is normalized by typical
# increment ratios [s^-1] and multiplied by dt to make it dimensionless;
# max_aliasing_error is the RMS restricted to stored band positions (pure
# recovery error), while max_full_error also includes the truncation of the
# sparsity structure itself. Entries where the dense reference or the
# rescaling is non-finite cannot be compared (they indicate a
# non-differentiable point or a non-finite tendency value, rather than an
# aliasing error), so they are excluded from both metrics and reported by
# report_nonfinite_entries; non-finite recovered entries at comparable
# positions are kept, making the metrics non-finite by design.
function block_errors(
    alg,
    label,
    Y,
    p,
    dtγ,
    t,
    dense_blocks,
    manual_blocks,
    rescalings,
)
    FT = eltype(Y)
    dt = FT(float(p.dt))
    blocks = CA.first_column_block_arrays(alg, Y, p, dtγ, t)
    report_nonfinite_entries(blocks, label)
    max_aliasing_error = zero(FT)
    max_full_error = zero(FT)
    block_aliasing_errors = Pair{Any, FT}[]
    for (block_key, block) in blocks
        block isa LA.UniformScaling && continue
        manual_blocks[block_key] isa LA.UniformScaling && continue
        valid =
            isfinite.(dense_blocks[block_key]) .&
            isfinite.(rescalings[block_key])
        difference =
            ifelse.(
                valid,
                (dense_blocks[block_key] .- block) .* rescalings[block_key] .* dt,
                zero(FT),
            )
        mask = stored_band_mask(manual_blocks[block_key])
        block_aliasing_error = rms(difference .* mask)
        push!(block_aliasing_errors, block_key => block_aliasing_error)
        max_aliasing_error = max(max_aliasing_error, block_aliasing_error)
        max_full_error = max(max_full_error, rms(difference))
    end
    # Report the dominant blocks so that scale or padding adjustments can be
    # targeted (NaN sorts first, so poisoned blocks are always listed).
    worst_blocks = sort(block_aliasing_errors; by = last, rev = true)
    @info "$label: largest aliasing errors per block: \
           $(first(worst_blocks, min(3, length(worst_blocks))))"
    @info "$label: max_aliasing_error = $max_aliasing_error, \
           max_full_error = $max_full_error"
    worst_block_keys =
        map(first, first(worst_blocks, min(3, length(worst_blocks))))
    return (; max_aliasing_error, max_full_error, worst_block_keys)
end

# For each stored column of the victim block, rank the other Y columns that
# share its color by their largest increment-weighted dense-Jacobian magnitude
# in the victim's rows, aggregated by field. These are the sources of the
# victim block's recovery (aliasing) error: a contributor from the same field
# (or from a field with the same seed scale) indicates missing bands that
# require padding or a pattern extension, while a contributor from another
# field indicates a seed scale ratio that needs adjustment.
function report_aliasing_contributors(
    alg,
    Y,
    p,
    victim_key,
    dense_blocks,
    label;
    measurement = nothing,
)
    FT = eltype(Y)
    column_Y = CA.first_column_view(Y)
    cache = CA.column_jacobian_cache(alg, Y, p.atmos; measurement)
    # The coloring metadata lives behind the cache's `rebuildable` Ref (it is
    # rebuilt as a unit when the runtime re-measure grows the mask).
    colors = cache.rebuildable[].jacobian_column_colors
    scalar_names = CA.scalar_field_names(column_Y)
    flat_indices = collect(CA.field_vector_index_iterator(column_Y))
    (victim_row_name, victim_column_name) = victim_key
    uₕ_scale = CA.uₕ_seed_scale(Y)
    victim_scale =
        CA.seed_scale(FT, victim_column_name, alg.seed_scaling, uₕ_scale)
    contributions = Dict{Any, FT}()
    for (j, (scalar_index, _)) in enumerate(flat_indices)
        scalar_names[scalar_index] == victim_column_name || continue
        color = colors[j]
        color == 0 && continue
        for (j2, (scalar_index2, level2)) in enumerate(flat_indices)
            (j2 == j || colors[j2] != color) && continue
            contributor_name = scalar_names[scalar_index2]
            contributor_block = dense_blocks[(victim_row_name, contributor_name)]
            contribution =
                maximum(abs, view(contributor_block, :, level2)) *
                CA.seed_scale(FT, contributor_name, alg.seed_scaling, uₕ_scale) /
                victim_scale
            contributions[contributor_name] =
                max(get(contributions, contributor_name, zero(FT)), contribution)
        end
    end
    ranked = sort(collect(contributions); by = last, rev = true)
    @info "$label: same-color contributors into $victim_key (max dense \
           magnitude × seed scale ratio, by field): \
           $(first(ranked, min(6, length(ranked))))"
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
    report_nonfinite_entries(dense_blocks, "dense")
    report_nonfinite_entries(manual_blocks, "manual sparse")
    report_nonfinite_entries(rescalings, "rescaling")

    errors(alg, label) = block_errors(
        alg,
        label,
        Y,
        p,
        dtγ,
        t,
        dense_blocks,
        manual_blocks,
        rescalings,
    )
    # Pin the pre-default-flip behavior (unscaled seeds, manual-rules padding)
    # as this helper's defaults, so each test variant names exactly the mode it
    # checks and stays fixed even though the AutoSparseJacobian defaults are now
    # seed_scaling = :static and padding_mode = :measured.
    sparse_alg(; seed_scaling = nothing, padding_mode = :manual_rules, kwargs...) =
        CA.AutoSparseJacobian(;
            approximate_solve_iters = 2,
            seed_scaling,
            padding_mode,
            kwargs...,
        )

    # Provisional bound on the normalized aliasing error; entries become
    # significant to the implicit solver when this metric approaches 1.
    tolerance = 1e-2

    # The current default configuration: unscaled seeds, default padding.
    errors_padded = errors(sparse_alg(), "unscaled + default padding")
    @test errors_padded.max_aliasing_error < tolerance
    # Attribute near-misses as well as failures, so that a variant that
    # passes with little margin still gets a contributor ranking in the log.
    if errors_padded.max_aliasing_error > tolerance / 2
        report_aliasing_contributors(
            sparse_alg(),
            Y,
            p,
            first(errors_padded.worst_block_keys),
            dense_blocks,
            "unscaled + default padding",
        )
    end

    # The practical scaled configuration: scaling suppresses cross-field
    # aliasing, while the default padding continues to cover within-field
    # band deficits (which scaling cannot suppress, since the seed scale
    # ratio within a field is 1).
    scaled_alg = sparse_alg(; seed_scaling = :static)
    errors_scaled = errors(scaled_alg, "scaled + default padding")
    @test errors_scaled.max_aliasing_error < tolerance
    if errors_scaled.max_aliasing_error > tolerance / 2
        report_aliasing_contributors(
            scaled_alg,
            Y,
            p,
            first(errors_scaled.worst_block_keys),
            dense_blocks,
            "scaled + default padding",
        )
    end

    # Constant-band mode (padding_mode = :constant): every padding rule
    # is active in every scaling mode, and every present block gets at least
    # 4 padding bands, so entries recovered on the stored bands should agree
    # with the dense reference far below the tolerance, at the cost of more
    # colors. Note that this mode does not widen the stored bands: the linear
    # solver still uses the manual structure's truncation of the Jacobian.
    constant_alg = sparse_alg(; padding_mode = :constant)
    errors_constant = errors(constant_alg, "constant bands + unscaled")
    @test errors_constant.max_aliasing_error < tolerance
    # No ordering assertion against the default-padding variant: both
    # unscaled variants sit at the dense-comparison floor (~1e-7 on this
    # configuration), and the wider mask re-rolls the coloring, which moves
    # the error within that floor in either direction (measured: 3.2e-7
    # default vs 6.7e-7 constant in gate run 243282).
    if errors_constant.max_aliasing_error > tolerance / 2
        report_aliasing_contributors(
            constant_alg,
            Y,
            p,
            first(errors_constant.worst_block_keys),
            dense_blocks,
            "constant bands + unscaled",
        )
    end
    constant_scaled_alg =
        sparse_alg(; seed_scaling = :static, padding_mode = :constant)
    errors_constant_scaled = errors(constant_scaled_alg, "constant bands + scaled")
    @test errors_constant_scaled.max_aliasing_error < tolerance
    @test errors_constant_scaled.max_aliasing_error <=
          errors_scaled.max_aliasing_error
    if errors_constant_scaled.max_aliasing_error > tolerance / 2
        report_aliasing_contributors(
            constant_scaled_alg,
            Y,
            p,
            first(errors_constant_scaled.worst_block_keys),
            dense_blocks,
            "constant bands + scaled",
        )
    end

    # Measured padding mode (padding_mode = :measured): the coloring mask is
    # derived from a one-time dense-AD measurement pass (stencil support plus
    # increment-weighted magnitudes) with no hand-maintained padding rules.
    # Aliasing on the stored bands must stay below the tolerance in both
    # scaling modes. The scaled variant should approach the dense-comparison
    # floor of the constant variant, since within-field support is captured
    # exactly (its seed scale ratio is 1) and every cross-field coupling whose
    # increment-weighted magnitude exceeds the threshold is kept. The color
    # count must not exceed the constant variant's, since the measured mask pads
    # present blocks to their true support (not a blanket floor) and keeps only
    # the significant cross-field blocks.
    measured_alg = sparse_alg(; padding_mode = :measured)
    errors_measured = errors(measured_alg, "measured bands + unscaled")
    @test errors_measured.max_aliasing_error < tolerance
    if errors_measured.max_aliasing_error > tolerance / 2
        report_aliasing_contributors(
            measured_alg,
            Y,
            p,
            first(errors_measured.worst_block_keys),
            dense_blocks,
            "measured bands + unscaled";
            measurement = (; p, dtγ, t),
        )
    end
    measured_scaled_alg =
        sparse_alg(; seed_scaling = :static, padding_mode = :measured)
    errors_measured_scaled =
        errors(measured_scaled_alg, "measured bands + scaled")
    @test errors_measured_scaled.max_aliasing_error < tolerance
    if errors_measured_scaled.max_aliasing_error > tolerance / 2
        report_aliasing_contributors(
            measured_scaled_alg,
            Y,
            p,
            first(errors_measured_scaled.worst_block_keys),
            dense_blocks,
            "measured bands + scaled";
            measurement = (; p, dtγ, t),
        )
    end

    # Calibration reports for the zero-padding variants. The scaled variant
    # is not asserted against the bound: within-field band deficits (e.g.,
    # the upwind-widened ∂/∂u₃ stencils) are invisible to seed scaling by
    # construction, so removing ALL padding is not expected to pass. The
    # naive variant additionally loses the cross-field suppression, and
    # scaling must never do worse than it.
    scaled_unpadded_alg =
        sparse_alg(; padding_bands_per_block = 0, seed_scaling = :static)
    errors_scaled_unpadded = errors(scaled_unpadded_alg, "scaled + no padding")
    for victim_key in first(errors_scaled_unpadded.worst_block_keys, 2)
        report_aliasing_contributors(
            scaled_unpadded_alg,
            Y,
            p,
            victim_key,
            dense_blocks,
            "scaled + no padding",
        )
    end
    errors_naive = errors(
        sparse_alg(; padding_bands_per_block = 0),
        "unscaled + no padding",
    )
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
    measurement = (; p, dtγ, t)
    n_εs(alg) = CA.ForwardDiff.npartials(
        eltype(
            parent(
                CA.jacobian_cache(
                    alg,
                    Y,
                    p.atmos;
                    verbose = false,
                    measurement,
                ).rebuildable[].Y_dual.c.ρ,
            ),
        ),
    )
    n_εs_padded = n_εs(sparse_alg())
    n_εs_scaled_default = n_εs(sparse_alg(; seed_scaling = :static))
    n_εs_scaled_unpadded =
        n_εs(sparse_alg(; padding_bands_per_block = 0, seed_scaling = :static))
    n_εs_constant = n_εs(sparse_alg(; padding_mode = :constant))
    n_εs_constant_scaled =
        n_εs(sparse_alg(; seed_scaling = :static, padding_mode = :constant))
    n_εs_measured = n_εs(sparse_alg(; padding_mode = :measured))
    n_εs_measured_scaled =
        n_εs(sparse_alg(; seed_scaling = :static, padding_mode = :measured))
    @info "ε components per dual number: $n_εs_padded with default padding, \
           $n_εs_scaled_default with scaled default padding, \
           $n_εs_scaled_unpadded without padding, $n_εs_constant with constant \
           bands (unscaled), $n_εs_constant_scaled with constant bands (scaled), \
           $n_εs_measured with measured bands (unscaled), \
           $n_εs_measured_scaled with measured bands (scaled)"
    @test n_εs_scaled_unpadded <= n_εs_padded
    # The measured mask never costs more colors than the constant-band mask.
    @test n_εs_measured <= n_εs_constant
    @test n_εs_measured_scaled <= n_εs_constant_scaled
end
