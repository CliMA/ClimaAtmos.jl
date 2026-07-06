import SparseMatrixColorings
import Random

# Minimum number of padding bands added to every present block when
# padding_mode = :exact. The measured within-field band deficits (upwind
# advection stencils that extend past the stored bands) are all covered by 4
# bands (±2 levels); see the padding rule comments in jacobian_cache.
const exact_recovery_padding_bands = 4

# Parameters of the measured padding mode (padding_mode = :measured); see
# validation/design_measured_coloring.md and measure_block_support_and_magnitude.
# A cross-field block a—b is kept in the coloring mask only if its
# increment-weighted magnitude dtγ * max|∂Yₜ| * s_b / s_a exceeds this
# threshold for the worst (smallest-scale) victim a in its row. 1e-3 is an
# order below the unit-test aliasing bound (1e-2) and orders above the
# ~1e-6-7 dense-comparison noise floor.
const measured_cross_field_threshold = 1e-3
# A band of a block counts as "supported" when its increment-weighted
# magnitude dtγ * |∂Yₜ| exceeds this floor at some measured state. Real
# stencil entries are O(1e-2-1); anything below this floor is negligible
# compared to solver significance, so excluding it from the mask is safe.
const measured_support_floor = 1e-6
# Number of randomly perturbed states (in addition to the base state) whose
# support is unioned. Perturbations break accidental zeros so that the full
# stencil support is detected; the union covers state-dependent branches
# (e.g. the sign of the advecting velocity).
const measured_n_random_states = 2
# Size of the state perturbation, in multiples of each field's typical
# increment (the seed-scale table); a few increments break accidental zeros
# while keeping the state physically plausible.
const measured_perturbation_multiplier = 4
# Fixed RNG seed for the perturbations, so that the measured mask (and hence
# the coloring and the recovered Jacobian) is reproducible across cache
# constructions, matching the deterministic coloring guarantee.
const measured_rng_seed = 0
# Whether to symmetrize each block's detected band range in the coloring mask
# (fill ±max(|lower|, |upper|)). Symmetrizing covers the opposite branch of
# sign-dependent upwind stencils that a given state does not exercise, at the
# cost of a few extra colors on one-sided advection blocks.
const measured_symmetrize_support = true


"""
    AutoSparseJacobian(sparse_jacobian_alg, [padding_bands_per_block], [seed_scaling])

A [`JacobianAlgorithm`](@ref) that computes the Jacobian using forward-mode
automatic differentiation, assuming that the Jacobian's sparsity structure is
given by `sparse_jacobian_alg`.

Only entries that are expected to be nonzero according to the sparsity
structure are updated, but any other entries that are nonzero can introduce
errors to the updated entries. This issue can be avoided by adding padding bands
to blocks that are likely to introduce errors. In cases where the default
padding bands are insufficient, `padding_bands_per_block` can be specified to
add a fixed number of padding bands to every block.

The errors introduced by entries outside of the sparsity structure can also be
reduced by rescaling the dual number seed of every Jacobian column by the
typical increment magnitude of its state variable, which is enabled by setting
`seed_scaling` to `:static`. When the seed of the column for ``Y_b`` is a
typical increment ``s_b`` instead of 1, an entry ``∂Yₜ_i/∂Y_b`` outside of the
sparsity structure modifies an entry ``∂Yₜ_i/∂Y_a`` that has the same color by
``∂Yₜ_i/∂Y_b * s_b / s_a``, which is negligible compared to ``∂Yₜ_i/∂Y_a``
whenever the increment-weighted criterion used to drop blocks from the sparsity
structure applies. Entries inside the sparsity structure are unaffected because
dual number arithmetic is linear in the seeds. Since this suppression makes the
default padding for missing cross-field blocks redundant, `:static` disables
those padding rules (reducing the number of colors); rules for blocks that are
present but narrower than their true stencils are kept in every mode, because
their extra entries have seed scale ratios of 1. Setting `seed_scaling` to a
function that maps scalar field names to positive numbers overrides the default
scales used by `:static`, but keeps the full default padding.

The `padding_mode` selects how the coloring mask is padded beyond the stored
bands. `:manual_rules` (the default) applies the hand-maintained padding
rules: they cover the blocks whose out-of-pattern entries have been measured
or reasoned to be significant, and they are only applied in the scaling modes
where those measurements were made. `:exact` instead pads the coloring mask
for exact recovery: every padding rule is applied in every scaling mode, and
every block that is present in the sparsity structure gets at least
`$exact_recovery_padding_bands` padding bands, so that all entries the true
Jacobian has near the stored bands receive their own colors and the recovered
entries on the stored bands are exact. This costs additional colors.
`:measured` derives the mask from a one-time dense-AD measurement pass instead
of hand-maintained rules (see [`measure_block_support_and_magnitude`](@ref)):
each block's coloring bands are its detected stencil support if the block is
present or if an absent cross-field block's increment-weighted magnitude
`dtγ * max|∂Yₜ| * s_b / s_a` exceeds a threshold for the worst victim in its
row. This requires a `measurement` context (the atmos cache, `dtγ`, and `t`),
which is threaded in from `get_jacobian` and the debug comparison path.
No padding mode changes which entries are stored — the linear solver always
uses the manual sparsity structure's truncation of the Jacobian.

For more information about this algorithm, see [Implicit Solver](@ref).
"""
struct AutoSparseJacobian{A <: SparseJacobian, P, S} <: SparseJacobian
    sparse_jacobian_alg::A
    padding_bands_per_block::P
    seed_scaling::S
    padding_mode::Symbol
    # When true (only meaningful with padding_mode = :measured), the coloring
    # mask is re-measured on a fixed early step schedule and grown monotonically
    # to what the developed flow reveals, so a from-rest configuration reaches a
    # correct-and-lean mask without a representative measurement state. Off by
    # default, so the static build is unchanged. See
    # remeasure_and_maybe_rebuild! and validation/design_measured_coloring.md.
    runtime_remeasure::Bool
end

"""
    AutoSparseJacobian(sparse_jacobian_alg, [padding_bands_per_block], [seed_scaling])

Construct an [`AutoSparseJacobian`](@ref) that reuses the sparsity structure
of the given `sparse_jacobian_alg`.
"""
AutoSparseJacobian(sparse_jacobian_alg) =
    AutoSparseJacobian(sparse_jacobian_alg, nothing, nothing, :manual_rules, false)
AutoSparseJacobian(sparse_jacobian_alg, padding_bands_per_block) =
    AutoSparseJacobian(
        sparse_jacobian_alg,
        padding_bands_per_block,
        nothing,
        :manual_rules,
        false,
    )
AutoSparseJacobian(sparse_jacobian_alg, padding_bands_per_block, seed_scaling) =
    AutoSparseJacobian(
        sparse_jacobian_alg,
        padding_bands_per_block,
        seed_scaling,
        :manual_rules,
        false,
    )

"""
    AutoSparseJacobian(; approximate_solve_iters = 1, padding_bands_per_block = nothing, seed_scaling = nothing, padding_mode = :manual_rules, runtime_remeasure = false)

Construct an [`AutoSparseJacobian`](@ref) that reuses the sparsity structure
of an inner [`ManualSparseJacobian`](@ref) built from `approximate_solve_iters`.
"""
function AutoSparseJacobian(;
    approximate_solve_iters::Int = 1,
    padding_bands_per_block = nothing,
    seed_scaling = nothing,
    padding_mode::Symbol = :manual_rules,
    runtime_remeasure::Bool = false,
)
    padding_mode in (:manual_rules, :exact, :measured) || error(
        "Unknown padding_mode :$padding_mode (supported modes are \
         :manual_rules, :exact, and :measured)",
    )
    runtime_remeasure && padding_mode != :measured &&
        error(
            "runtime_remeasure = true is only supported with padding_mode = \
             :measured (it re-measures the measured mode's coloring mask on a \
             schedule); got padding_mode = :$padding_mode",
        )
    return AutoSparseJacobian(
        ManualSparseJacobian(; approximate_solve_iters),
        padding_bands_per_block,
        seed_scaling,
        padding_mode,
        runtime_remeasure,
    )
end

"""
    uₕ_seed_scale(Y)

Typical increment magnitude of the covariant horizontal wind components,
derived from the grid of `Y`: a physical wind increment of ~0.05 m/s times the
horizontal node length scale, which approximates the metric factor that
converts physical wind components to covariant components (the same idiom
hyperdiffusion uses to set grid-dependent coefficients). On column spaces the
horizontal space is a point space whose length scale is 1, so the scale
reduces to the physical increment, matching the identity metric that covariant
components have in column models.

The first-column view used by the debug comparison also types as a column
space, so caches built on column views of a full grid must be given the full
grid's scale explicitly (see [`column_jacobian_cache`](@ref)) to stay
consistent with the simulation's Jacobian.
"""
function uₕ_seed_scale(Y)
    FT = eltype(Y)
    h_space = Spaces.horizontal_space(axes(Y.c))
    return FT(0.05) * FT(Spaces.node_horizontal_length_scale(h_space))
end

"""
    seed_scale(::Type{FT}, scalar_name, seed_scaling, uₕ_scale)

The scale of the dual number seed for the Jacobian column that corresponds to
the scalar field `scalar_name`, given the `seed_scaling` of an
[`AutoSparseJacobian`](@ref). When `seed_scaling` is `nothing`, every seed is 1;
when it is `:static`, the seed is the field's typical increment magnitude from
[`default_jacobian_seed_scale`](@ref), with the geometry-dependent `uₕ_scale`
(see [`uₕ_seed_scale`](@ref)) used for the covariant horizontal wind
components; when it is a function, the seed is `FT(seed_scaling(scalar_name))`.
"""
seed_scale(::Type{FT}, scalar_name, ::Nothing, uₕ_scale) where {FT} = one(FT)
seed_scale(
    ::Type{FT},
    scalar_name,
    seed_scaling::Symbol,
    uₕ_scale,
) where {FT} =
    seed_scaling == :static ?
    default_jacobian_seed_scale(FT, scalar_name, uₕ_scale) :
    error("Unknown seed_scaling mode :$seed_scaling (the only mode is :static)")
seed_scale(
    ::Type{FT},
    scalar_name,
    seed_scaling::F,
    uₕ_scale,
) where {FT, F <: Function} = FT(seed_scaling(scalar_name))

"""
    default_jacobian_seed_scale(::Type{FT}, scalar_name, uₕ_scale)

Typical magnitude of the change in the scalar field `scalar_name` over an
implicit stage of the model, used to scale dual number seeds when an
[`AutoSparseJacobian`](@ref) is constructed with `seed_scaling = :static`.

These are order-of-magnitude estimates, and only the ratios between scales
matter, since rescaling all seeds by a constant has no effect on aliasing
errors. The scales make the increment-weighted comparisons in the padding band
comments below executable; e.g., ``‖δᶜρ‖ ≪ ‖δᶜρe_tot‖`` becomes
``s_ρ ≪ s_ρe_tot``. Scalar fields that are not listed here default to a scale
of 1, which is safe for fields whose columns have no nonzero entries in the
sparsity structure (their colors are 0, so their seeds are never set), but
which may allow aliasing errors from new prognostic variables; the seed scales
are logged when the Jacobian is constructed with `verbose = true`.
"""
function default_jacobian_seed_scale(::Type{FT}, scalar_name, uₕ_scale) where {FT}
    uₕ_component_names =
        (@name(c.uₕ.components.data.:(1)), @name(c.uₕ.components.data.:(2)))
    grid_mean_condensate_names =
        (@name(c.ρq_lcl), @name(c.ρq_icl), @name(c.ρq_rai), @name(c.ρq_sno))
    sgs_condensate_names = (
        @name(c.sgsʲs.:(1).q_lcl),
        @name(c.sgsʲs.:(1).q_icl),
        @name(c.sgsʲs.:(1).q_rai),
        @name(c.sgsʲs.:(1).q_sno),
    )
    return if scalar_name == @name(c.ρ)
        FT(1e-5) # δρ ~ dt * ρₜ, much smaller than δρe_tot / (cᵥ * T)
    elseif scalar_name in uₕ_component_names
        # δu ~ 0.05 m/s, times the grid's horizontal metric factor; this
        # covariant component scale is geometry-dependent (~1e4-1e5 m on
        # global meshes, 1 in column models), so it is resolved from the grid
        # by uₕ_seed_scale instead of being hardcoded here.
        FT(uₕ_scale)
    elseif scalar_name == @name(c.ρe_tot)
        FT(1e2) # δT ~ 0.03-0.1 K, times ρ * cᵥ ~ 1e3 J/(K * m^3)
    elseif scalar_name == @name(f.u₃.components.data.:(1)) ||
           scalar_name == @name(f.sgsʲs.:(1).u₃.components.data.:(1))
        FT(1e1) # δw ~ 0.01-0.1 m/s, times a vertical metric factor of ~1e2 m
    elseif scalar_name == @name(c.ρq_tot)
        FT(1e-6) # δq_tot ~ 1e-6 kg/kg, times ρ ~ 1 kg/m^3
    elseif scalar_name in grid_mean_condensate_names
        FT(1e-7) # δq ~ 1e-7 kg/kg for condensate and precipitation species
    elseif scalar_name == @name(c.ρtke)
        FT(1e-3) # δtke ~ 1e-3 m^2/s^2, times ρ ~ 1 kg/m^3
    elseif scalar_name == @name(c.sgsʲs.:(1).ρa)
        FT(1e-4) # δaʲ ~ 1e-4, times ρ ~ 1 kg/m^3
    elseif scalar_name == @name(c.sgsʲs.:(1).mse)
        FT(1e2) # δT ~ 0.03-0.1 K, times cₚ ~ 1e3 J/(K * kg)
    elseif scalar_name == @name(c.sgsʲs.:(1).q_tot)
        FT(1e-6) # δq_tot ~ 1e-6 kg/kg
    elseif scalar_name in sgs_condensate_names
        FT(1e-7) # δq ~ 1e-7 kg/kg for condensate and precipitation species
    else
        # All other fields (e.g., surface fields, gas tracers, and number
        # concentrations) keep unscaled seeds. This is always safe for fields
        # whose columns lie outside the sparsity structure, since their seeds
        # are never set.
        one(FT)
    end
end

# Additive random perturbation of one column's state, used by the measured
# padding mode to break accidental zeros so that a block's full stencil support
# is detected. Each scalar value whose field has a typical-increment entry in
# the seed-scale table is shifted by `multiplier` of its typical increments;
# values whose field has no entry (surface fields, tracers, number
# concentrations) are left unperturbed, since an additive shift of size 1 would
# be meaningless for them. The shift is additive rather than multiplicative so
# that it also reveals support that depends on values which are exactly zero at
# the base state, and independent per level so that it covers both signs of a
# sign-dependent stencil. Each value is written with the same `get_field`/
# `point` idiom used by update_column_matrices!, so it addresses exactly its own
# scalar component regardless of the field's parent layout.
function perturb_column_state!(
    Yk,
    Y_base,
    scalar_names,
    field_vector_indices,
    column_index,
    uₕ_scale,
    multiplier,
    rng,
)
    FT = eltype(Y_base)
    Yk .= Y_base
    for (scalar_index, level_index) in field_vector_indices
        scale =
            default_jacobian_seed_scale(FT, scalar_names[scalar_index], uₕ_scale)
        scale == one(FT) && continue
        shift = FT(multiplier) * scale * randn(rng, FT)
        unrolled_applyat(scalar_index, scalar_names) do name
            field = MatrixFields.get_field(Yk, name)
            @inbounds point(field, level_index, column_index...)[] += shift
        end
    end
    return Yk
end

"""
    measure_block_support_and_magnitude(Y, atmos, p, dtγ, t; ...)

One-time dense-AD measurement pass on the first column, used by the measured
padding mode. Returns an `n_scalars × n_scalars` matrix of NamedTuples, one per
block of the scalarized tendency matrix, with fields:

  - `has_support`: whether any band of the block is above the support floor;
  - `support_lower`, `support_upper`: the band index range (column minus row,
    in block-local level indices) of the detected stencil support, unioned
    over the base state and `n_random_states` perturbed states;
  - `magnitude`: `dtγ * max|∂Yₜ|` over the block on the base (representative)
    state, i.e. the increment-unweighted aliasing magnitude before the seed
    scale ratio is applied. Only the base state is used: the perturbed states
    over-activate the flow, so maximizing the magnitude over them keeps far
    more cross-field blocks than a real developed flow needs.

The pass reuses the [`AutoDenseJacobian`](@ref) machinery: it builds one dense
column cache and evaluates the exact dense Jacobian `∂Yₜ/∂Y` of the implicit
tendency on each state (the precomputed quantities are recomputed from each
state inside `update_column_matrices!`, so only `Y` needs to be perturbed).
Support comes from the union over states (perturbations break accidental
zeros, e.g. at a degenerate initial state); the magnitude comes from the base
state alone (a perturbed state's magnitudes are not physical). The measurement
is deterministic (fixed RNG seed), so the derived mask, coloring, and recovered
Jacobian are reproducible across cache constructions.
"""
function measure_block_support_and_magnitude(
    Y,
    atmos,
    p,
    dtγ,
    t;
    uₕ_scale = uₕ_seed_scale(Y),
    verbose = false,
    n_random_states = measured_n_random_states,
    multiplier = measured_perturbation_multiplier,
    support_floor = measured_support_floor,
    rng_seed = measured_rng_seed,
)
    FT = eltype(Y)
    dense_alg = AutoDenseJacobian()
    column_Y = first_column_view(Y)
    column_p = first_column_view(p)
    scalar_names = scalar_field_names(column_Y)
    n_scalars = length(scalar_names)
    field_vector_indices = collect(field_vector_index_iterator(column_Y))
    column_index = first(column_index_iterator(column_Y))
    column_cache = jacobian_cache(dense_alg, column_Y, atmos)

    # Positions of each scalar field's rows/columns in the N × N column matrix.
    positions = [Int[] for _ in 1:n_scalars]
    for (k, (scalar_index, _)) in enumerate(field_vector_indices)
        push!(positions[scalar_index], k)
    end

    support_lower = fill(typemax(Int), n_scalars, n_scalars)
    support_upper = fill(typemin(Int), n_scalars, n_scalars)
    magnitude = zeros(FT, n_scalars, n_scalars)

    Yk = similar(column_Y)
    rng = Random.MersenneTwister(rng_seed)
    n_states = 1 + n_random_states
    for state_index in 1:n_states
        state =
            state_index == 1 ? column_Y :
            perturb_column_state!(
                Yk,
                column_Y,
                scalar_names,
                field_vector_indices,
                column_index,
                uₕ_scale,
                multiplier,
                rng,
            )
        try
            update_column_matrices!(dense_alg, column_cache, state, column_p, t)
        catch err
            # The base state is the real, physically valid state, so a failure
            # there is a real problem; a perturbed state may leave the physical
            # domain (e.g. produce a non-finite tendency), in which case the
            # other states and the support floor still cover the block.
            state_index == 1 && rethrow(err)
            verbose && @warn "Measured padding: skipping perturbed state \
                              $state_index (tendency evaluation failed)"
            continue
        end
        M = Array(column_cache.column_matrices) # N × N × 1, this state's ∂Yₜ/∂Y
        for ci in 1:n_scalars, ri in 1:n_scalars
            rows = positions[ri]
            cols = positions[ci]
            for (a, i) in enumerate(rows), (b, j) in enumerate(cols)
                v = @inbounds M[i, j, 1]
                isfinite(v) || continue
                m = dtγ * abs(v)
                # Magnitude is taken from the base (representative) state only:
                # the perturbed states over-activate the flow, so maximizing
                # over them inflates cross-field magnitudes and keeps far more
                # blocks than a real developed flow needs (measured on the
                # quiescent trmm t=0 state, that over-keeping pushes the color
                # count past :exact). Cross-field significance is inherently a
                # magnitude property of the actual state; a representative
                # measurement state or the runtime probe (Rung C) is what makes
                # it correct on a from-rest configuration.
                state_index == 1 &&
                    (magnitude[ri, ci] = max(magnitude[ri, ci], m))
                if m > support_floor
                    band = b - a
                    support_lower[ri, ci] = min(support_lower[ri, ci], band)
                    support_upper[ri, ci] = max(support_upper[ri, ci], band)
                end
            end
        end
    end

    block_measurement = map(
        Iterators.product(1:n_scalars, 1:n_scalars),
    ) do (ri, ci)
        has_support = support_lower[ri, ci] <= support_upper[ri, ci]
        (;
            has_support,
            support_lower = has_support ? support_lower[ri, ci] : 0,
            support_upper = has_support ? support_upper[ri, ci] : 0,
            magnitude = magnitude[ri, ci],
        )
    end
    if verbose
        n_with_support = count(m -> m.has_support, block_measurement)
        @info "Measured padding: dense-AD support/magnitude over $n_states \
               states ($n_random_states perturbed); $n_with_support of \
               $(n_scalars^2) blocks have support above the floor \
               (dtγ * |∂Yₜ| > $support_floor)"
    end
    return block_measurement
end

"""
    measured_coloring_band_range(block_key, lower_band, upper_band, is_present, meas, s_column, s_victim_min, verbose)

The band index range of a block in the coloring mask under the measured
padding mode, given its stored band range (`lower_band:upper_band`), whether
the block is present in the sparsity structure, its measurement `meas`, the
seed scale `s_column` of its column field, and the smallest seed scale
`s_victim_min` among the present blocks in its row (or `nothing` if the row has
no present block). Present blocks include their stored bands, their measured
support, and the structural within-field floor (within-field / same-increment
aliasing has a seed scale ratio of 1 and cannot be suppressed by scaling, and
its width is a state-free property of the discretization); absent cross-field
blocks are included only when their increment-weighted magnitude exceeds the
threshold for the worst victim.
"""
function measured_coloring_band_range(
    block_key,
    lower_band,
    upper_band,
    is_present,
    meas,
    s_column,
    s_victim_min,
    verbose,
)
    FT = typeof(s_column)
    (support_lo, support_hi) = if !meas.has_support
        (0, 0)
    elseif measured_symmetrize_support
        width = max(abs(meas.support_lower), abs(meas.support_upper))
        (-width, width)
    else
        (meas.support_lower, meas.support_upper)
    end

    if is_present
        # Present (recovered) block: the stored bands plus its measured support
        # (symmetrized to cover the opposite branch of a sign-dependent
        # advection stencil). Within-field aliasing has a seed scale ratio of 1
        # and so cannot be suppressed by scaling; its width is resolved from the
        # measurement at the measured state. A state-free per-operator-metadata
        # width (design Rung A) would additionally cover advection stencils that
        # are inactive at a quiescent measurement state (a from-rest config such
        # as trmm_0M measured at t = 0); that, and the runtime probe (Rung C),
        # are the completing pieces documented in
        # validation/design_measured_coloring.md. A blanket ±2-level floor was
        # rejected: it exceeds the :exact color count and still does not cover
        # the state-dependent cross-field couplings, so it is not a substitute.
        return (min(Int(lower_band), support_lo), max(Int(upper_band), support_hi))
    end

    # Absent cross-field block: keep it only if its increment-weighted
    # magnitude exceeds the threshold for the worst (smallest-scale) victim in
    # its row; then separate the colliding columns over its measured support.
    keep =
        meas.has_support &&
        !isnothing(s_victim_min) &&
        meas.magnitude * s_column / s_victim_min >
        FT(measured_cross_field_threshold)
    keep || return (1, 0)
    if verbose
        metric = meas.magnitude * s_column / s_victim_min
        @info "Measured mask: cross-field block $block_key kept (metric \
               dtγ * max|∂Yₜ| * s_b / s_a = $metric > \
               τ = $measured_cross_field_threshold); coloring bands \
               $support_lo:$support_hi"
    end
    return (support_lo, support_hi)
end

# The first-column view of Y types as a column space, whose horizontal length
# scale is 1, so the geometry-dependent uₕ seed scale must be resolved from
# the full grid before slicing; otherwise the debug comparison would measure
# aliasing with a uₕ seed that differs from the simulation's by the
# horizontal metric factor. The measurement context (the atmos cache, dtγ, and
# t needed by the :measured padding mode) is threaded through unchanged; its
# Y and cache are the full grid's, which the measurement reduces to the first
# column itself, matching the per-column coloring mask.
column_jacobian_cache(alg::AutoSparseJacobian, Y, atmos; measurement = nothing) =
    jacobian_cache(
        alg,
        first_column_view(Y),
        atmos;
        verbose = false,
        uₕ_scale = uₕ_seed_scale(Y),
        measurement,
    )

function jacobian_cache(
    alg::AutoSparseJacobian,
    Y,
    atmos;
    verbose = true,
    uₕ_scale = uₕ_seed_scale(Y),
    measurement = nothing,
)
    (; sparse_jacobian_alg, padding_bands_per_block, seed_scaling) = alg
    (; padding_mode) = alg

    FT = eltype(Y)
    DA = ClimaComms.array_type(Y)
    device = ClimaComms.device(Y.c)

    column_indices = column_index_iterator(Y) # iterator of (i, j, h)
    field_vector_indices = field_vector_index_iterator(Y) # iterator of (f, v)
    scalar_names = scalar_field_names(Y) # iterator of names corresponding to f

    precomputed = implicit_precomputed_quantities(Y, atmos)
    scratch = implicit_temporary_quantities(Y, atmos)

    # Allocate ∂R/∂Y and its corresponding linear solver.
    # TODO: Add FieldNameTree(Y) to the matrix in FieldMatrixWithSolver. The
    # tree is needed to evaluate scalar_tendency_matrix[autodiff_matrix_keys].
    # (; matrix) = jacobian_cache(sparse_jacobian_alg, Y, atmos)
    matrix_without_tree = jacobian_cache(sparse_jacobian_alg, Y, atmos).matrix
    tree = MatrixFields.FieldNameTree(Y)
    inner_matrix =
        MatrixFields.replace_name_tree(matrix_without_tree.matrix, tree)
    # Zero-initialize all block storage: update_jacobian! only writes the band
    # entries whose colors are in the current partition, so entries outside
    # the vertical range of a band (near the domain boundaries) would
    # otherwise keep whatever memory `similar` returned, which makes the
    # Jacobian blocks non-deterministic across cache constructions.
    foreach(values(inner_matrix)) do block
        block isa Fields.Field && fill!(parent(block), zero(FT))
    end
    matrix = MatrixFields.FieldMatrixWithSolver(
        inner_matrix,
        matrix_without_tree.solver,
    )

    # Allocate ∂Yₜ/∂Y and create a view of the scalar components of its blocks.
    tendency_matrix = matrix .+ one(matrix)
    scalar_tendency_matrix = MatrixFields.scalar_field_matrix(tendency_matrix)

    # Find scalar keys that correspond to the scalar components of the blocks.
    # When we approximate a tensor derivative as a scalar multiple of the
    # identity tensor, we compute the scalar quantity using the top-left
    # component of the tensor. For example, the derivative of Yₜ.c.uₕ with
    # respect to Y.c.uₕ is computed as the derivative of
    # Yₜ.c.uₕ.components.data.:1 with respect to Y.c.uₕ.components.data.:1.
    scalar_block_keys =
        map(keys(scalar_tendency_matrix)) do (block_row_name, block_column_name)
            scalar_block_row_name =
                block_row_name in scalar_names ? block_row_name :
                unrolled_argfirst(scalar_names) do scalar_name
                    MatrixFields.is_child_name(scalar_name, block_row_name)
                end
            scalar_block_column_name =
                block_column_name in scalar_names ? block_column_name :
                unrolled_argfirst(scalar_names) do scalar_name
                    MatrixFields.is_child_name(scalar_name, block_column_name)
                end
            (scalar_block_row_name, scalar_block_column_name)
        end

    # Resolve the entry that the autodiff machinery can read band structure
    # from and write recovered values into, for each scalarized key: the
    # scalar view itself when it is a Field, or the parent block of
    # tendency_matrix when the scalar view is an unwritable lazy wrapper but
    # the parent is a Field whose entries occupy a single scalar component.
    # The latter case covers the covector-valued advection blocks (∂χₜ/∂u₃)
    # and single-component tensor blocks (∂u₃ʲₜ/∂u₃ʲ), whose scalar views are
    # not Fields; without the parent fallback, they are misclassified as
    # constant blocks and silently dropped from the Jacobian, leaving zeros
    # in their place. Constant blocks resolve to `nothing` and are excluded.
    parent_block_keys = map(identity, keys(tendency_matrix))
    function autodiff_block_entry(scalar_block_key)
        scalar_view = scalar_tendency_matrix[scalar_block_key]
        scalar_view isa Fields.Field && return scalar_view
        (scalar_row_name, scalar_column_name) = scalar_block_key
        matching_parent_keys = unrolled_filter(parent_block_keys) do parent_key
            (parent_row_name, parent_column_name) = parent_key
            (
                scalar_row_name == parent_row_name ||
                MatrixFields.is_child_name(scalar_row_name, parent_row_name)
            ) && (
                scalar_column_name == parent_column_name ||
                MatrixFields.is_child_name(
                    scalar_column_name,
                    parent_column_name,
                )
            )
        end
        isempty(matching_parent_keys) && return nothing
        parent_block = tendency_matrix[matching_parent_keys[1]]
        parent_block isa Fields.Field || return nothing
        # Every band entry must contain exactly one FT-sized scalar, since
        # update_jacobian! writes recovered scalars into it by scaling the
        # block's unit entry (see autodiff_block_unit_entries).
        sizeof(eltype(eltype(parent_block))) == sizeof(FT) || return nothing
        return parent_block
    end

    # Find keys of non-constant blocks, which are represented by matrix fields.
    non_constant_scalar_block_keys =
        unrolled_filter(scalar_block_keys) do scalar_block_key
            !isnothing(autodiff_block_entry(scalar_block_key))
        end

    # Create a new view of ∂Yₜ/∂Y that has scalar keys and non-constant blocks.
    autodiff_matrix = MatrixFields.FieldNameDict(
        MatrixFields.FieldMatrixKeys(non_constant_scalar_block_keys),
        unrolled_map(autodiff_block_entry, non_constant_scalar_block_keys),
    )

    # A unit entry for each block of autodiff_matrix, used by update_jacobian!
    # to convert recovered scalars to the block's entry type with a scalar
    # multiplication. The scalar `reinterpret` used previously performs padding
    # reflection (Base.padding) that cannot be compiled for GPU kernels, so
    # the bitcast is done once here on the CPU instead: every entry type is an
    # isbits struct with a single FT-sized leaf, so `scalar * unit_entry`
    # places exactly the bits of `scalar` in that leaf (multiplication by 1
    # is exact for all values, including NaN, Inf, and signed zero).
    autodiff_block_unit_entries =
        unrolled_map(values(autodiff_matrix)) do matrix_field
            reinterpret(eltype(eltype(matrix_field)), one(FT))
        end
    if verbose
        @info "Scalar names of Y: $(join(collect(scalar_names), ", "))"
        @info "Scalar block keys of the tendency matrix view: \
               $(join(collect(keys(scalar_tendency_matrix)), ", "))"
        @info "Autodiff matrix scalar block keys: \
               $(join(collect(non_constant_scalar_block_keys), ", "))"
        foreach(non_constant_scalar_block_keys) do scalar_block_key
            scalar_tendency_matrix[scalar_block_key] isa Fields.Field ||
                @info "Block $scalar_block_key uses its parent matrix block \
                       for autodiff (scalar view type: \
                       $(typeof(scalar_tendency_matrix[scalar_block_key])))"
        end
        dropped_scalar_block_keys =
            unrolled_filter(scalar_block_keys) do scalar_block_key
                !(scalar_block_key in non_constant_scalar_block_keys)
            end
        foreach(dropped_scalar_block_keys) do scalar_block_key
            @info "Block $scalar_block_key was dropped from the autodiff \
                   view; its scalar view has type \
                   $(typeof(scalar_tendency_matrix[scalar_block_key]))"
        end
    end

    # For the measured padding mode, run a one-time dense-AD measurement pass
    # on the first column to detect each block's true stencil support and its
    # increment-weighted magnitude, then derive the coloring mask from those
    # (below) instead of the hand-maintained padding rules. The measurement
    # needs a full atmos cache to evaluate the implicit tendency, plus dtγ and
    # t; these are threaded in through the `measurement` keyword (from
    # get_jacobian in production and from the debug comparison path).
    block_measurement = nothing
    victim_min_scale = nothing
    if padding_mode == :measured
        isnothing(measurement) && error(
            "padding_mode :measured requires a `measurement` context (the \
             atmos cache p, dtγ, and t) to run its dense-AD measurement pass; \
             it is threaded in from get_jacobian and the debug comparison \
             path. See validation/design_measured_coloring.md.",
        )
        block_measurement = measure_block_support_and_magnitude(
            Y,
            atmos,
            measurement.p,
            measurement.dtγ,
            measurement.t;
            uₕ_scale,
            verbose,
        )
        # Smallest victim seed scale per row field, taken over the present
        # blocks in that row (the recovered blocks whose entries a dropped
        # cross-field coupling could poison). A cross-field block a—b is kept
        # only if dtγ * max|∂Yₜ| * s_b / s_a exceeds τ for the worst (smallest
        # s_a) victim in row a; see measured_coloring_band_range.
        victim_min_scale = Dict{Any, FT}()
        for (present_row_name, present_column_name) in
            non_constant_scalar_block_keys
            s = seed_scale(FT, present_column_name, seed_scaling, uₕ_scale)
            victim_min_scale[present_row_name] =
                min(get(victim_min_scale, present_row_name, FT(Inf)), s)
        end
    end

    (sparsity_mask, padded_sparsity_mask) = build_sparsity_masks(
        alg,
        Y,
        autodiff_matrix,
        non_constant_scalar_block_keys,
        block_measurement,
        victim_min_scale;
        uₕ_scale,
        verbose,
    )

    # The type-varying buffers (dual-eltype fields and the coloring metadata,
    # whose types depend on n_εs) are built from the padded coloring mask and
    # kept behind a Ref so the runtime re-measure can swap in a rebuilt set
    # without changing the Jacobian object's identity (which ClimaTimeSteppers
    # holds); see build_jacobian_dual_buffers and remeasure_and_maybe_rebuild!.
    rebuildable = build_jacobian_dual_buffers(
        alg,
        Y,
        precomputed,
        scratch,
        autodiff_matrix,
        sparsity_mask,
        padded_sparsity_mask;
        uₕ_scale,
        verbose,
    )

    return (;
        matrix,
        tendency_matrix,
        autodiff_matrix,
        autodiff_block_unit_entries,
        rebuildable = Ref{Any}(rebuildable),
        # Runtime re-measure bookkeeping (only used when the algorithm's
        # runtime_remeasure flag is set); the stored bands never change, so
        # sparsity_mask stays fixed and only padded_sparsity_mask grows.
        sparsity_mask,
        padded_sparsity_mask = Ref(padded_sparsity_mask),
        remeasure_frozen = Ref(false),
        remeasure_no_change_count = Ref(0),
        remeasure_step = Ref(0),
        remeasure_next_step = Ref(1),
        remeasure_dtγ = isnothing(measurement) ? nothing : measurement.dtγ,
    )
end

# Build the coloring masks for the sparse autodiff Jacobian: `sparsity_mask`
# marks the stored (structural) bands of every present block, and
# `padded_sparsity_mask` additionally marks the padding bands that the current
# padding_mode adds beyond them. Factored out of `jacobian_cache` so the
# runtime re-measure (remeasure_and_maybe_rebuild!) rebuilds the mask through
# the same path. `block_measurement` and `victim_min_scale` are only used by
# padding_mode = :measured (both `nothing` otherwise).
function build_sparsity_masks(
    alg::AutoSparseJacobian,
    Y,
    autodiff_matrix,
    non_constant_scalar_block_keys,
    block_measurement,
    victim_min_scale;
    uₕ_scale = uₕ_seed_scale(Y),
    verbose = false,
)
    (; padding_bands_per_block, seed_scaling, padding_mode) = alg
    FT = eltype(Y)
    field_vector_indices = field_vector_index_iterator(Y)
    scalar_names = scalar_field_names(Y)

    # Construct a mask for nonzero entries in autodiff_matrix as a dense array,
    # and a similar mask with additional padding bands in each block.
    # TODO: Improve performance by only adding bands where they are necessary,
    # or by rescaling blocks instead of adding new bands.
    N = length(Fields.column(Y, 1, 1, 1))
    sparsity_mask = Array{Bool}(undef, N, N)
    sparsity_mask .= false
    padded_sparsity_mask = copy(sparsity_mask)
    for block_key in Iterators.product(scalar_names, scalar_names)
        (block_row_name, block_column_name) = block_key

        # Get a view of this block's sparsity masks with its row/column indices.
        block_jacobian_row_index_to_Yₜ_index_map =
            Iterators.filter(enumerate(field_vector_indices)) do index_pair
                (_, (scalar_index, _)) = index_pair
                scalar_names[scalar_index] == block_row_name
            end
        block_jacobian_column_index_to_Y_index_map =
            Iterators.filter(enumerate(field_vector_indices)) do index_pair
                (_, (scalar_index, _)) = index_pair
                scalar_names[scalar_index] == block_column_name
            end
        block_view_indices = (
            map(first, block_jacobian_row_index_to_Yₜ_index_map),
            map(first, block_jacobian_column_index_to_Y_index_map),
        )
        block_sparsity_mask = view(sparsity_mask, block_view_indices...)
        padded_block_sparsity_mask =
            view(padded_sparsity_mask, block_view_indices...)

        # Compute the lower and upper band indices of this block, with empty
        # blocks corresponding to index ranges whose length is -1 (centered
        # around 0 for square blocks and around ±1/2 for non-square blocks).
        # The membership test uses exact comparison against the tuple of
        # scalarized keys that the autodiff matrix view was built from, so
        # that no block of the view can be silently dropped from the mask.
        (n_rows_in_block, n_columns_in_block) = size(block_sparsity_mask)
        if block_key in non_constant_scalar_block_keys
            (_, _, lower_band, upper_band) =
                MatrixFields.band_matrix_info(autodiff_matrix[block_key])
        else
            (lower_band, upper_band) =
                n_rows_in_block == n_columns_in_block ? (1 / 2, -1 / 2) :
                (n_rows_in_block < n_columns_in_block ? (1, 0) : (0, -1))
        end
        verbose &&
            lower_band <= upper_band &&
            @info "Sparsity structure of $block_key has bands \
                   $(Int(lower_band)):$(Int(upper_band))"

        # Symmetrically expand the range of band indices, with the number of
        # new bands either limited by padding_bands_per_block, or hardcoded
        # for each block when padding_bands_per_block is not specified.
        mass_names = (@name(c.ρ), @name(c.sgsʲs.:(1).ρa))
        uₕ_component_names =
            (@name(c.uₕ.components.data.:(1)), @name(c.uₕ.components.data.:(2)))
        condensate_names =
            (@name(c.ρq_lcl), @name(c.ρq_icl), @name(c.ρq_rai), @name(c.ρq_sno))
        # Padding for missing cross-field blocks is only required with
        # unscaled seeds: the aliasing it prevents is exactly the kind that
        # seed scaling suppresses through the increment-weighted criterion, so
        # with `seed_scaling = :static` these rules return 0 bands and the
        # coloring needs fewer colors. Rules for blocks that are present but
        # narrower than their true stencils are kept in every mode, since the
        # extra entries come from the same field (or a field with the same
        # seed scale) and no scale ratio can suppress them. Custom function
        # scalings keep the full padding, since their scales are not
        # guaranteed to satisfy the increment-weighted criterion.
        static_scaling = seed_scaling == :static
        # With padding_mode = :exact, the mask is padded for exact recovery
        # instead of relying on the measured rules: every rule below is active
        # in every scaling mode, and every present block additionally gets at
        # least exact_recovery_padding_bands bands (applied after the rules).
        exact_recovery = padding_mode == :exact
        cross_field_padding_active = !static_scaling || exact_recovery
        rule_padding_bands =
            if (
                cross_field_padding_active &&
                (
                    block_row_name in uₕ_component_names &&
                    block_column_name in (mass_names..., @name(c.ρtke)) ||
                    block_row_name == @name(c.ρe_tot) &&
                    block_column_name in (
                        mass_names...,
                        condensate_names...,
                        @name(c.ρtke),
                        @name(c.sgsʲs.:(1).q_tot),
                    ) ||
                    block_row_name == @name(c.sgsʲs.:(1).ρa) &&
                    block_column_name == @name(c.ρq_tot) ||
                    block_row_name == @name(f.sgsʲs.:(1).u₃.components.data.:(1)) &&
                    block_column_name in uₕ_component_names
                ) &&
                !(block_key in non_constant_scalar_block_keys) &&
                (block_row_name, block_row_name) in non_constant_scalar_block_keys
            )
                # Missing off-diagonal blocks whose entries typically have
                # magnitudes that are larger than (or similar to) diagonal blocks in
                # the same rows:
                # - ‖∂ᶜuᵢₜ/∂ᶜρ‖, ‖∂ᶜuᵢₜ/∂ᶜρaʲ‖ ≳ ‖∂ᶜuᵢₜ/∂ᶜuᵢ‖, and ‖∂ᶜuᵢₜ/∂ᶜρtke‖
                #   where uᵢ is either u₁ or u₂, as long as ‖δᶜρ‖, ‖δᶜρaʲ‖ and
                #   ‖δᶜρtke‖ are relatively smaller than ‖δᶜuᵢ‖.
                # - ‖∂ᶜρe_totₜ/∂ᶜρ‖, ‖∂ᶜρe_totₜ/∂ᶜχ‖, and ‖∂ᶜρe_totₜ/∂ᶜρχ‖ ≳
                #   ‖∂ᶜρe_totₜ/∂ᶜρe_tot‖ when χ is any scalar of order unity, as
                #   long as ‖δᶜρ‖ and ‖δᶜχ‖ are relatively smaller than ‖δᶜρe_tot‖.
                # - ‖∂ᶜρaʲₜ/∂ᶜρq_tot‖ ≳ ‖∂ᶜρaʲₜ/∂ᶜρaʲ‖, as long as ‖δᶜρq_tot‖ is
                #   relatively smaller than ‖δᶜρaʲ‖.
                # - ‖∂ᶠu₃ʲₜ/∂ᶜu₁‖ and ‖∂ᶠu₃ʲₜ/∂ᶜu₁‖ ≳ ‖∂ᶠu₃ʲₜ/∂ᶠu₃ʲ‖, as long as
                #   ‖δᶜu₁‖ and ‖δᶜu₂‖ are relatively smaller than ‖δᶠu₃ʲ‖.
                # Diagonal blocks are critical for conservation and stability, so
                # these potential errors from off-diagonal blocks should be avoided.
                3
            elseif (
                cross_field_padding_active &&
                block_row_name == @name(c.sgsʲs.:(1).ρa) &&
                block_column_name == @name(c.ρ) &&
                !(block_key in non_constant_scalar_block_keys) &&
                (block_row_name, @name(c.sgsʲs.:(1).mse)) in non_constant_scalar_block_keys
            )
                # ‖∂ᶜρaʲₜ/∂ᶜρ‖ ≳ ‖∂ᶜρaʲₜ/∂ᶜmseʲ‖, as long as ‖δᶜρ‖ is relatively
                # smaller than ‖δᶜmseʲ‖. The ∂ᶜρaʲₜ/∂ᶜmseʲ block is important for
                # stability in some simulations with turbulence, so this potential
                # error from the ∂ᶜρaʲₜ/∂ᶜρ block should be avoided.
                3
            elseif (
                cross_field_padding_active &&
                block_row_name == @name(f.u₃.components.data.:(1)) &&
                block_column_name in condensate_names &&
                !(block_key in non_constant_scalar_block_keys) &&
                (block_row_name, uₕ_component_names[1]) in non_constant_scalar_block_keys
            )
                # ‖∂ᶜu₃ₜ/∂ᶜρχ‖ ≳ ‖∂ᶜu₃ₜ/∂ᶜuₕ‖ when χ is any specific humidity, as
                # long as ‖δᶜρχ‖ is relatively smaller than ‖δᶜuₕ‖. The ∂ᶜu₃ₜ/∂ᶜuₕ
                # block is important for stability in some simulations with
                # topography, so this potential error from ∂ᶜu₃ₜ/∂ᶜρχ blocks should
                # be avoided.
                2
            elseif (
                block_row_name in
                (@name(c.ρ), @name(c.ρe_tot), @name(c.ρq_tot)) &&
                block_column_name == @name(f.u₃.components.data.:(1)) &&
                block_key in non_constant_scalar_block_keys
            )
                # Present blocks whose true bandwidth exceeds the stored bandwidth:
                # the implicit advection of ρe_tot and ρq_tot includes upwind
                # corrections whose derivatives with respect to u₃ extend beyond
                # the stored bidiagonal, and the grid-mean continuity and energy
                # rows also couple to u₃ at nearby levels through EDMF mass flux
                # terms. These entries come from the same column field (or from
                # u₃ʲ, which has the same typical increment), so no seed scale can
                # suppress them; padding separates the colors of nearby u₃ columns
                # instead.
                4
            elseif (
                block_row_name == @name(c.ρe_tot) &&
                block_column_name in (@name(c.ρq_tot), condensate_names...) &&
                block_key in non_constant_scalar_block_keys
            )
                # Present blocks whose true bandwidth exceeds the stored bandwidth:
                # ‖∂ᶜρe_totₜ/∂ᶜρχ‖ entries outside the stored bands of these
                # blocks (e.g., from sedimentation and precipitation stencils) are
                # much larger than ‖∂ᶜρe_totₜ/∂ᶜρe_tot‖ in raw units, so with
                # unscaled seeds they can alias into the energy diagonal whenever
                # a ρχ column shares a color with a ρe_tot column. Seed scaling
                # suppresses this by ‖δᶜρχ‖ / ‖δᶜρe_tot‖, but the energy diagonal
                # is critical for conservation, so the bands are padded regardless
                # of the scaling mode.
                4
            elseif (
                (static_scaling || exact_recovery) &&
                block_row_name == @name(c.sgsʲs.:(1).mse) &&
                block_column_name == @name(c.sgsʲs.:(1).mse) &&
                block_key in non_constant_scalar_block_keys
            )
                # Present block whose true bandwidth exceeds the stored bandwidth:
                # the upwind advection of mseʲ has derivatives with respect to
                # mseʲ at nearby levels that extend beyond the stored bands (the
                # calibration variant measures a self-contribution of order 1
                # whenever nearby mseʲ columns share a color). These entries come
                # from the same field, so their seed scale ratio is 1 and no
                # scaling can suppress them; padding separates the colors of
                # nearby mseʲ columns instead. Only the reduced-color coloring of
                # the `:static` mode packs mseʲ columns close enough to collide,
                # so the bands are only added in that mode — with unscaled seeds
                # they would perturb the default coloring, whose aliasing
                # behavior is validated as-is (changing the unscaled mask
                # reshuffles every color pair and can expose new cross-field
                # offenders, which is measured and documented in the unit test
                # logs).
                4
            else
                0
            end
        max_padding_bands = if !isnothing(padding_bands_per_block)
            padding_bands_per_block
        elseif exact_recovery && block_key in non_constant_scalar_block_keys
            # Every present block gets at least this many padding bands, so
            # that within-field stencil deficits (whose seed scale ratios are
            # 1) cannot alias regardless of which blocks the measured rules
            # cover; see the (uₕ, uₕ) deficit in the unit test logs.
            max(rule_padding_bands, exact_recovery_padding_bands)
        else
            rule_padding_bands
        end
        padded_lower_band = ceil(Int, lower_band - max_padding_bands / 2)
        padded_upper_band = floor(Int, upper_band + max_padding_bands / 2)

        # In the measured padding mode, override the rule-derived band range
        # with one derived from the dense-AD measurement pass: the detected
        # stencil support for present blocks (whose within-field aliasing has a
        # seed scale ratio of 1 and cannot be suppressed by scaling), plus the
        # support of absent cross-field blocks whose increment-weighted
        # magnitude survives the threshold. block_measurement is indexed by the
        # scalar field position, which matches scalar_names by construction.
        if padding_mode == :measured
            block_row_index = findfirst(==(block_row_name), scalar_names)
            block_column_index = findfirst(==(block_column_name), scalar_names)
            (padded_lower_band, padded_upper_band) =
                measured_coloring_band_range(
                    block_key,
                    lower_band,
                    upper_band,
                    block_key in non_constant_scalar_block_keys,
                    block_measurement[block_row_index, block_column_index],
                    seed_scale(FT, block_column_name, seed_scaling, uₕ_scale),
                    get(victim_min_scale, block_row_name, nothing),
                    verbose,
                )
        end

        if verbose
            n_padding_bands =
                length(padded_lower_band:padded_upper_band) -
                length(lower_band:upper_band)
            n_padding_bands > 0 &&
                @info "Adding $n_padding_bands padding bands for $block_key"
        end

        # Update the sparsity mask entries corresponding to bands in this block.
        for band in padded_lower_band:padded_upper_band
            is_not_padding_band = band in lower_band:upper_band
            level_index_min = band < 0 ? 1 - band : 1
            level_index_max =
                band < n_columns_in_block - n_rows_in_block ? n_rows_in_block :
                n_columns_in_block - band
            for level_index in level_index_min:level_index_max
                block_mask_index = (level_index, level_index + band)
                block_sparsity_mask[block_mask_index...] = is_not_padding_band
                padded_block_sparsity_mask[block_mask_index...] = true
            end
        end
    end

    return (sparsity_mask, padded_sparsity_mask)
end

# Build the type-varying autodiff buffers (dual-eltype FieldVectors and cached
# quantities, the identity-matrix partitions, and the coloring metadata) from a
# coloring mask. This is the "rebuild from mask" step: it is the only part of
# the cache whose types depend on n_εs, so both the initial `jacobian_cache`
# build and the runtime re-measure (remeasure_and_maybe_rebuild!) call it, and
# its result is stored behind the cache's `rebuildable` Ref. The stored-band
# structures (matrix, tendency_matrix, autodiff_matrix) are unchanged by a mask
# grow, so they are built once in `jacobian_cache` and passed back in here.
function build_jacobian_dual_buffers(
    alg::AutoSparseJacobian,
    Y,
    precomputed,
    scratch,
    autodiff_matrix,
    sparsity_mask,
    padded_sparsity_mask;
    uₕ_scale = uₕ_seed_scale(Y),
    verbose = false,
)
    (; seed_scaling) = alg
    FT = eltype(Y)
    DA = ClimaComms.array_type(Y)
    device = ClimaComms.device(Y.c)
    column_indices = column_index_iterator(Y)
    field_vector_indices = field_vector_index_iterator(Y)
    scalar_names = scalar_field_names(Y)

    # Find a coloring that minimizes the number of required colors. Exclude
    # RandomOrder from the candidate orders to make the coloring deterministic
    # across cache constructions; when entries outside of the sparsity
    # structure are nonzero, different colorings produce different aliasing
    # errors, so a random coloring would make the Jacobian non-reproducible.
    all_coloring_orders = filter(
        order -> !(order isa SparseMatrixColorings.RandomOrder),
        SparseMatrixColorings.all_orders(),
    )
    jacobian_column_colorings = map(all_coloring_orders) do coloring_order
        SparseMatrixColorings.coloring(
            SparseMatrixColorings.sparse(padded_sparsity_mask),
            SparseMatrixColorings.ColoringProblem(),
            SparseMatrixColorings.GreedyColoringAlgorithm(coloring_order),
        )
    end
    best_order_index =
        findmin(SparseMatrixColorings.ncolors, jacobian_column_colorings)[2]
    best_jacobian_column_coloring = jacobian_column_colorings[best_order_index]
    n_colors = SparseMatrixColorings.ncolors(best_jacobian_column_coloring)

    # When running on GPU devices, divide n_colors into partitions that are each
    # guaranteed to fit in the memory that is currently free (adding a factor of
    # 2 to account for potential future garbage collection).
    n_partitions = if device isa ClimaComms.AbstractCPUDevice
        1
    else
        free_memory = ClimaComms.free_memory(device)
        max_memory = 2 * free_memory
        memory_for_I_matrix = n_colors * parent_memory(Y)
        memory_per_ε =
            (parent_memory(precomputed) + parent_memory(scratch)) +
            2 * parent_memory(Y)
        # Find the smallest possible integer n_partitions and some other integer
        # n_εs such that n_partitions * n_εs >= n_colors and
        # (n_εs + 1) * memory_per_ε + memory_for_I_matrix <= max_memory, where
        # (n_εs + 1) * memory_per_ε is the memory required to store
        # precomputed_dual, scratch_dual, Y_dual, and Yₜ_dual, and where
        # memory_for_I_matrix is an approximation of the memory required to
        # store I_matrix_partitions. The actual memory_for_I_matrix is given by
        # (n_colors + n_partitions) * parent_memory(Y), but we can ignore the
        # value of n_partitions if it is negligible compared to n_colors. When
        # max_memory is too small to fit any εs, try using one ε per partition.
        # TODO: Replace the fields in I_matrix_partitions with column fields,
        # making memory_for_I_matrix negligible compared to max_memory.
        n_εs_max = (max_memory - memory_for_I_matrix) ÷ memory_per_ε - 1
        cld(n_colors, max(n_εs_max, 1))
    end
    n_εs = cld(n_colors, n_partitions)

    if verbose
        @info "Using coloring order $(all_coloring_orders[best_order_index])"
        if n_partitions == 1
            @info "Updating Jacobian using $n_εs ε components per dual number"
        else
            @info "Updating Jacobian using $n_partitions partitions of \
                   $n_colors colors, with $n_εs ε components per dual number"
        end
    end

    # FieldVectors and cached fields with dual numbers instead of real numbers,
    # with dual numbers using the tag "Jacobian" for specialized dispatch
    # TODO: Refactor FieldVector broadcasting so that performance does not
    # deteriorate if we only store one column of each partition_εs.
    FT_dual = ForwardDiff.Dual{Jacobian, FT, n_εs}
    precomputed_dual = replace_parent_eltype(precomputed, FT_dual)
    scratch_dual = replace_parent_eltype(scratch, FT_dual)
    Y_dual = replace_parent_eltype(Y, FT_dual)
    Yₜ_dual = similar(Y_dual)
    I_matrix_partitions = ntuple(_ -> similar(Y_dual), n_partitions)

    # iterator of colors for each of the values in Y that require autodiff, and
    # 0 for values that do not require it (by not initializing dual number
    # components when they are unneeded, we might be able to avoid some of the
    # errors introduced by our sparsity approximation)
    jacobian_column_colors =
        SparseMatrixColorings.column_colors(best_jacobian_column_coloring)
    for Y_index_and_sparsity_mask in enumerate(eachcol(sparsity_mask))
        (Y_index, jacobian_column_sparsity_mask) = Y_index_and_sparsity_mask
        if !any(jacobian_column_sparsity_mask)
            jacobian_column_colors[Y_index] = 0
        end
    end

    # Seed scale of every scalar field, used to rescale the dual number seed of
    # each Jacobian column (and to unscale the entries recovered from the
    # output partials). Rescaling the seed of the column for Y_b by s_b makes
    # the aliasing error it introduces into a same-colored column for Y_a
    # proportional to s_b / s_a, which is negligible whenever the
    # increment-weighted criterion for dropping blocks from the sparsity
    # structure applies to the entries outside of that structure.
    seed_scales = unrolled_map(
        name -> seed_scale(FT, name, seed_scaling, uₕ_scale),
        scalar_names,
    )
    if verbose && !isnothing(seed_scaling)
        seed_scale_pairs = unrolled_map(
            name -> name => seed_scale(FT, name, seed_scaling, uₕ_scale),
            scalar_names,
        )
        @info "Scaling dual number seeds by typical increment magnitudes: \
               $(join(seed_scale_pairs, ", "))"
    end

    # iterator of tuples ((f, v), c, s), where the color c identifies a
    # component of the dual number in row (f, v) of Y_dual that corresponds to
    # the diagonal entry in the same row of the matrix ∂Y/∂Y (or 0 if the
    # corresponding value in Y does not require autodiff), and s is the seed
    # scale of the scalar field that contains this row
    Y_index_seed_scales = Iterators.map(
        ((scalar_index, _),) -> seed_scales[scalar_index],
        field_vector_indices,
    )
    Y_index_to_diagonal_color_map =
        zip(field_vector_indices, jacobian_column_colors, Y_index_seed_scales)

    # Set the dual numbers in each FieldVector partition_εs so that the ε
    # components correspond to partitions of the N × N identity matrix ∂Y/∂Y.
    # Specifically, every column of partition_εs is a vector of N dual numbers,
    # each of which is stored as a combination of a value and n_εs partial
    # derivatives. The ε components can be interpreted as representing N × n_εs
    # slices of a sparse N × n_colors representation of ∂Y/∂Y. Convert n_εs to
    # a Val and Y_index_to_diagonal_color_map to a DA for GPU compatibility, and
    # drop spatial information from every Field to ensure that this kernel stays
    # below the GPU parameter memory limit.
    n_εs_val = Val(n_εs)
    I_matrix_partitions_data = unrolled_map(I_matrix_partitions) do partition_εs
        unrolled_map(Fields.field_values, Fields._values(partition_εs))
    end
    ClimaComms.@threaded device begin
        # On multithreaded devices, use one thread for each dual number.
        for (partition_index, partition_εs_data) in
            enumerate(I_matrix_partitions_data),
            column_index in column_indices,
            index_pair in DA(collect(Y_index_to_diagonal_color_map))

            ((scalar_index, level_index), diagonal_entry_color, entry_scale) =
                index_pair
            ε_offset = (partition_index - 1) * n_εs
            diagonal_entry_ε_index =
                ε_offset < diagonal_entry_color <= ε_offset + n_εs ?
                diagonal_entry_color - ε_offset : 0
            ε_coefficients =
                ntuple(i -> (i == diagonal_entry_ε_index) * entry_scale, n_εs_val)
            unrolled_applyat(scalar_index, scalar_names) do name
                field = MatrixFields.get_field(partition_εs_data, name)
                @inbounds point(field, level_index, column_index...)[] =
                    ForwardDiff.Dual{Jacobian}(0, ε_coefficients)
            end
        end
    end

    # number of colors needed to represent a band matrix row in any block
    colors_per_band_matrix_row =
        maximum(values(autodiff_matrix)) do matrix_field
            (_, _, lower_band, upper_band) =
                MatrixFields.band_matrix_info(matrix_field)
            upper_band - lower_band + 1
        end

    # iterator of pairs ((b, v), (f, cs, s⁻¹)), where (b, v) is the index of a
    # band matrix row in autodiff_matrix, (f, v) is the index of a dual number
    # in Yₜ_dual, cs is a tuple that contains the colors of the band matrix row
    # entries, which is padded to have a constant size for GPU compatibility,
    # and s⁻¹ is the inverse of the seed scale of the block's column field
    # TODO: Use an iterator of pairs ((f, v), ((b1, cs1), (b2, cs2), ...)), so
    # that each pair corresponds to a dual number instead of a band matrix row.
    band_matrix_row_index_to_colors_map = Iterators.flatmap(
        enumerate(pairs(autodiff_matrix)),
    ) do (block_index, (block_key, matrix_field))
        (block_row_name, block_column_name) = block_key
        (n_rows_in_block, n_columns_in_block, lower_band, upper_band) =
            MatrixFields.band_matrix_info(matrix_field)
        inv_column_seed_scale =
            inv(seed_scale(FT, block_column_name, seed_scaling, uₕ_scale))

        block_Yₜ_indices =
            Iterators.filter(field_vector_indices) do (scalar_index, _)
                scalar_names[scalar_index] == block_row_name
            end
        block_Y_index_to_color_map =
            Iterators.filter(Y_index_to_diagonal_color_map) do index_pair
                ((scalar_index, _), _) = index_pair
                scalar_names[scalar_index] == block_column_name
            end
        block_colors = getindex.(block_Y_index_to_color_map, 2)

        map(block_Yₜ_indices) do (scalar_index, level_index)
            entry_colors = ntuple(colors_per_band_matrix_row) do band_index
                band = lower_band + band_index - 1
                level_index_min = band < 0 ? 1 - band : 1
                level_index_max =
                    band < n_columns_in_block - n_rows_in_block ?
                    n_rows_in_block : n_columns_in_block - band
                is_color_at_index =
                    band <= upper_band &&
                    level_index_min <= level_index <= level_index_max
                is_color_at_index ? block_colors[level_index + band] : 0
            end
            (
                (block_index, level_index),
                (scalar_index, entry_colors, inv_column_seed_scale),
            )
        end
    end

    # Convert the lazy iterator to a DA for GPU compatibility.
    band_matrix_row_index_to_colors_map =
        DA(collect(band_matrix_row_index_to_colors_map))


    return (;
        precomputed_dual,
        scratch_dual,
        Y_dual,
        Yₜ_dual,
        I_matrix_partitions,
        band_matrix_row_index_to_colors_map,
        jacobian_column_colors, # CPU-resident; only used for diagnostics
        n_colors, # total colors in the mask; reported by the runtime re-measure
    )
end

function update_jacobian!(::AutoSparseJacobian, cache, Y, p, dtγ, t)
    (; matrix, tendency_matrix, autodiff_matrix) = cache
    (; autodiff_block_unit_entries) = cache

    # The dual-eltype buffers and coloring metadata live behind the Ref so the
    # runtime re-measure can swap in a rebuilt (and differently n_εs-typed) set
    # without changing the cache's identity. Read them through a function
    # barrier so the compute kernel below stays type-stable and simply
    # recompiles for the new n_εs when a rebuild changes the type; a direct
    # `cache.rebuildable[]` read here would type the whole kernel as `Any`.
    apply_autodiff_partials!(
        cache.rebuildable[],
        autodiff_matrix,
        autodiff_block_unit_entries,
        Y,
        p,
        t,
    )

    # Update the matrix for ∂R/∂Y using the new values of ∂Yₜ/∂Y.
    matrix .= dtγ .* tendency_matrix .- one(matrix)
end

# Function barrier for the autodiff compute kernel: specialized on the concrete
# type of `rebuildable` (which changes with n_εs), so the hot loop is
# monomorphic. Mutates the blocks of `autodiff_matrix` in place.
function apply_autodiff_partials!(
    rebuildable,
    autodiff_matrix,
    autodiff_block_unit_entries,
    Y,
    p,
    t,
)
    (; precomputed_dual, scratch_dual, Y_dual, Yₜ_dual) = rebuildable
    (; I_matrix_partitions, band_matrix_row_index_to_colors_map) = rebuildable

    device = ClimaComms.device(Y.c)
    column_indices = column_index_iterator(Y)
    scalar_names = scalar_field_names(Y)
    p_dual = append_to_atmos_cache(p, precomputed_dual, scratch_dual)

    for (partition_index, partition_εs) in enumerate(I_matrix_partitions)
        # Set the εs in Y_dual to represent a partition of the identity matrix.
        Y_dual .= Y .+ partition_εs

        # Compute ∂p/∂Y * I_matrix_partition and ∂Yₜ/∂Y * I_matrix_partition.
        set_implicit_precomputed_quantities!(Y_dual, p_dual, t)
        implicit_tendency!(Yₜ_dual, Y_dual, p_dual, t)

        # Move the entries of ∂Yₜ/∂Y * I_matrix_partition from Yₜ_dual into the
        # blocks of autodiff_matrix. Drop spatial information from every Field
        # to ensure that this kernel stays below the GPU parameter memory limit.
        Yₜ_dual_data =
            unrolled_map(Fields.field_values, Fields._values(Yₜ_dual))
        matrix_fields_data =
            unrolled_map(Fields.field_values, values(autodiff_matrix))
        matrix_fields_data_and_units =
            map(tuple, matrix_fields_data, autodiff_block_unit_entries)
        ClimaComms.@threaded device begin
            # On multithreaded devices, use one thread for each band matrix row.
            # TODO: Modify the map and use one thread for each dual number.
            for column_index in column_indices,
                index_pair in band_matrix_row_index_to_colors_map

                (
                    (block_index, level_index),
                    (scalar_index, entry_colors, inv_column_seed_scale),
                ) = index_pair
                dual_number =
                    unrolled_applyat(scalar_index, scalar_names) do name
                        data = MatrixFields.get_field(Yₜ_dual_data, name)
                        @inbounds point(data, level_index, column_index...)[]
                    end
                ε_coefficients = ForwardDiff.partials(dual_number)
                n_εs = length(ε_coefficients)
                ε_offset = (partition_index - 1) * n_εs
                unrolled_applyat(
                    block_index,
                    matrix_fields_data_and_units,
                ) do (block_data, unit_entry)
                    @inbounds entries_data =
                        point(block_data, level_index, column_index...).entries
                    entries_data[] =
                        map(entry_colors, entries_data[]) do entry_color, entry
                            # If the entry has a color in the current partition,
                            # set the entry to the ε coefficient for that color,
                            # unscaled by the seed scale of the block's column
                            # field, and converted to the entry type (which is
                            # a single-scalar-component covector or tensor for
                            # blocks that use their parent matrix blocks) by
                            # scaling the block's unit entry — a bitcast
                            # through Base.reinterpret is not GPU-compilable.
                            # Otherwise, keep the block's current value.
                            ε_offset < entry_color <= ε_offset + n_εs ?
                            (@inbounds ε_coefficients[entry_color - ε_offset]) *
                            inv_column_seed_scale *
                            unit_entry : entry
                        end # TODO: Why does unrolled_map break GPU compilation?
                end
            end
        end
    end
    # TODO: Figure out why this is currently 2--3 orders of magnitude more
    # expensive than any other kernel we are launching on GPUs.
    return nothing
end

invert_jacobian!(alg::AutoSparseJacobian, cache, ΔY, R) =
    invert_jacobian!(alg.sparse_jacobian_alg, cache, ΔY, R)

# The fixed geometric schedule of steps after which the measured coloring mask
# is re-measured (runtime_remeasure = true). A from-rest configuration has an
# inactive SGS updraft/turbulence at t = 0, so the couplings that develop by
# the first steps are not in the Jacobian to measure yet; re-measuring on the
# good state between these early steps grows the mask before the coupling that
# would otherwise crash the next step, with a one-step lag and no rollback. See
# validation/design_measured_coloring.md (Rung C) and the handover.
const jacobian_remeasure_max_step = 32

"""
    remeasure_and_maybe_rebuild!(jacobian::Jacobian, Y, p, t)

Re-measure the measured padding mode's coloring mask at the current state `Y`
and, if the mask grew, rebuild the type-varying autodiff buffers behind the
cache's `rebuildable` Ref. The mask is grown monotonically (unioned), so colors
never decrease; after two consecutive re-measures produce no change the mask is
frozen and further re-measures are skipped. No-op once frozen.

This is the runtime-adaptive rung of the measured mode: it lets a from-rest
configuration reach a correct-and-lean coloring mask without a representative
(spun-up) measurement state. It detects new coupling structure with high
probability at the sampled states, not with certainty.
"""
function remeasure_and_maybe_rebuild!(jacobian, Y, p, t)
    alg = jacobian.alg
    cache = jacobian.cache
    cache.remeasure_frozen[] && return nothing
    atmos = p.atmos
    dtγ = cache.remeasure_dtγ
    uₕ_scale = uₕ_seed_scale(Y)

    # Re-measure block support and magnitude at the current (developed) state.
    # Reuses the same dense-AD measurement path as the initial build (no second
    # measurement code path), at the dtγ the initial build used, so the support
    # floor and cross-field threshold stay comparable across re-measures.
    block_measurement = measure_block_support_and_magnitude(
        Y,
        atmos,
        p,
        dtγ,
        t;
        uₕ_scale,
        verbose = false,
    )

    # Smallest victim seed scale per row (same recipe as jacobian_cache's
    # measured branch), needed by the cross-field significance test.
    FT = eltype(Y)
    non_constant_scalar_block_keys = keys(cache.autodiff_matrix)
    victim_min_scale = Dict{Any, FT}()
    for (present_row_name, present_column_name) in
        non_constant_scalar_block_keys
        s = seed_scale(FT, present_column_name, alg.seed_scaling, uₕ_scale)
        victim_min_scale[present_row_name] =
            min(get(victim_min_scale, present_row_name, FT(Inf)), s)
    end

    # Build a fresh candidate mask from the new measurement, then grow the
    # current mask monotonically (union). sparsity_mask (stored bands) is
    # state-free and never changes, so only padded_sparsity_mask grows.
    (_, candidate_padded_mask) = build_sparsity_masks(
        alg,
        Y,
        cache.autodiff_matrix,
        non_constant_scalar_block_keys,
        block_measurement,
        victim_min_scale;
        uₕ_scale,
        verbose = false,
    )
    current_padded_mask = cache.padded_sparsity_mask[]
    union_mask = current_padded_mask .| candidate_padded_mask

    if union_mask == current_padded_mask
        cache.remeasure_no_change_count[] += 1
        if cache.remeasure_no_change_count[] >= 2
            cache.remeasure_frozen[] = true
            @info "Runtime Jacobian re-measure: mask converged (no change for \
                   two consecutive re-measures); freezing at \
                   $(cache.rebuildable[].n_colors) colors."
        end
        return nothing
    end

    # The mask grew: rebuild the type-varying buffers from the unioned mask and
    # swap them in behind the Ref. ClimaTimeSteppers reads cache.rebuildable[]
    # fresh on the next Wfact, so the swap takes effect on the next step. The
    # rebuild changes n_εs, hence the dual-number type, hence recompiles the
    # autodiff kernel; this is amortized over the run (rebuilds only happen
    # while the mask is still growing during spin-up, then it freezes).
    cache.padded_sparsity_mask[] = union_mask
    cache.remeasure_no_change_count[] = 0
    precomputed = implicit_precomputed_quantities(Y, atmos)
    scratch = implicit_temporary_quantities(Y, atmos)
    cache.rebuildable[] = build_jacobian_dual_buffers(
        alg,
        Y,
        precomputed,
        scratch,
        cache.autodiff_matrix,
        cache.sparsity_mask,
        union_mask;
        uₕ_scale,
        verbose = false,
    )
    n_added = count(union_mask) - count(current_padded_mask)
    @info "Runtime Jacobian re-measure: mask grew by $n_added band entries; \
           rebuilt coloring buffers ($(cache.rebuildable[].n_colors) colors)."
    return nothing
end

# Callback affect: fires after every step, but only acts on the fixed geometric
# schedule (steps 1, 2, 4, …, jacobian_remeasure_max_step) and only when the
# implicit Jacobian is an AutoSparseJacobian with runtime_remeasure enabled.
# No-op for every other configuration, so adding this callback unconditionally
# keeps non-adaptive runs bit-identical.
function jacobian_runtime_remeasure_affect!(integrator)
    jacobian = integrator_jacobian(integrator)
    isnothing(jacobian) && return nothing
    (jacobian.alg isa AutoSparseJacobian && jacobian.alg.runtime_remeasure) ||
        return nothing
    cache = jacobian.cache
    cache.remeasure_frozen[] && return nothing
    step = (cache.remeasure_step[] += 1)
    if step == cache.remeasure_next_step[]
        cache.remeasure_next_step[] =
            step >= jacobian_remeasure_max_step ? typemax(Int) : 2 * step
        remeasure_and_maybe_rebuild!(
            jacobian,
            integrator.u,
            integrator.p,
            integrator.t,
        )
    end
    return nothing
end
