"""
    ExactJacobian()

A `JacobianAlgorithm` that computes the `ImplicitEquationJacobian` using
forward-mode automatic differentiation and inverts it using LU factorization.
"""
struct ExactJacobian <: JacobianAlgorithm end

exact_jacobian_batch_size() = 32 # number of derivatives computed simultaneously

replace_parent_type(x::Fields.FieldVector, ::Type{T}) where {T} = similar(x, T)
replace_parent_type(x::Fields.Field, ::Type{T}) where {T} =
    Fields.Field(replace_parent_type(Fields.field_values(x), T), axes(x))
replace_parent_type(x::DataLayouts.AbstractData, ::Type{T}) where {T} =
    DataLayouts.replace_basetype(x, T)
replace_parent_type(x::Union{Tuple, NamedTuple}, ::Type{T}) where {T} =
    unrolled_map(Base.Fix2(replace_parent_type, T), x)

function jacobian_cache(::ExactJacobian, Y, atmos)
    FT = eltype(Y)
    DA = ClimaComms.array_type(Y)

    FT_dual = ForwardDiff.Dual{ExactJacobian, FT, exact_jacobian_batch_size()}
    Y_dual = replace_parent_type(Y, FT_dual)
    Yₜ_dual = similar(Y_dual)
    precomputed_dual =
        replace_parent_type(implicit_precomputed_quantities(Y, atmos), FT_dual)
    scratch_dual = replace_parent_type(temporary_quantities(Y, atmos), FT_dual)

    n_columns = Fields.ncolumns(Y.c)
    n_εs = length(first_column(Y))
    column_matrices = DA{FT}(undef, n_columns, n_εs, n_εs)
    column_lu_factors = copy(column_matrices)
    column_lu_solve_vectors = DA{FT}(undef, n_columns, n_εs)
    lu_cache = DA{FT}(undef, n_columns)

    # LinearAlgebra.I does not support broadcasting, so we need a workaround.
    I_column_matrix = DA{FT}(undef, n_εs, n_εs)
    I_column_matrix .= 0
    I_column_matrix[diagind(I_column_matrix)] .= 1
    I_matrix = reshape(I_column_matrix, 1, n_εs, n_εs)

    return (;
        Y_dual,
        Yₜ_dual,
        precomputed_dual,
        scratch_dual,
        column_matrices,
        column_lu_factors,
        column_lu_solve_vectors,
        lu_cache,
        I_matrix,
    )
end

function update_jacobian_skip_factorizing!(::ExactJacobian, cache, Y, p, dtγ, t)
    (; Y_dual, Yₜ_dual, precomputed_dual, scratch_dual, column_matrices) = cache
    batch_size = exact_jacobian_batch_size()

    p_dual_args = ntuple(Val(fieldcount(typeof(p)))) do cache_field_index
        cache_field_name = fieldname(typeof(p), cache_field_index)
        if cache_field_name == :precomputed
            (; p.precomputed..., precomputed_dual...)
        elseif cache_field_name == :scratch
            scratch_dual
        else
            getfield(p, cache_field_index)
        end
    end
    p_dual = AtmosCache(p_dual_args...)

    Y_dual_scalar_levels = scalar_level_iterator(Y_dual)
    level_batches =
        Iterators.partition(enumerate(Y_dual_scalar_levels), batch_size)

    Y_dual .= Y
    for level_batch in level_batches
        for (partial_index, (_, Y_dual_scalar_level)) in enumerate(level_batch)
            partials = ntuple(i -> i == partial_index ? 1 : 0, Val(batch_size))
            parent(Y_dual_scalar_level) .+=
                ForwardDiff.Dual{ExactJacobian}(0, partials...)
        end # Add a unique ε to Y_dual for each combination of scalar and level.
        set_implicit_precomputed_quantities!(Y_dual, p_dual, t) # Compute ∂p/∂Y.
        implicit_tendency!(Yₜ_dual, Y_dual, p_dual, t) # Compute ∂Yₜ/∂Y.
        for (partial_index, (ε_index, _)) in enumerate(level_batch)
            ε_column_vectors = view(column_matrices, :, :, ε_index)
            column_vectors_to_field_vector(ε_column_vectors, Y) .=
                getindex.(ForwardDiff.partials.(Yₜ_dual), partial_index)
        end # Copy the new values of ∂Yₜ/∂Y into column_matrices.
        Y_dual .= ForwardDiff.value.(Y_dual) # Drop this batch's εs from Y_dual.
    end
end

function update_jacobian!(::ExactJacobian, cache, Y, p, dtγ, t)
    (; column_matrices, column_lu_factors, lu_cache, I_matrix) = cache
    update_jacobian_skip_factorizing!(ExactJacobian(), cache, Y, p, dtγ, t)
    column_lu_factors .= dtγ .* column_matrices .- I_matrix
    parallel_lu_factorize!(column_lu_factors, lu_cache)
end

function invert_jacobian!(::ExactJacobian, cache, ΔY, R)
    (; column_lu_solve_vectors, column_lu_factors, lu_cache) = cache
    column_vectors_to_field_vector(column_lu_solve_vectors, R) .= R
    parallel_lu_solve!(column_lu_solve_vectors, column_lu_factors, lu_cache)
    ΔY .= column_vectors_to_field_vector(column_lu_solve_vectors, ΔY)
end

function save_jacobian!(::ExactJacobian, cache, Y, dtγ, t)
    (; column_matrices, column_matrix) = cache
    (n_columns, n_εs, _) = size(column_matrices)

    column_matrix .= view(column_matrices, 1, :, :)
    file_name = "exact_jacobian_first"
    title = "Exact ∂Yₜ/∂Y$(n_columns == 1 ? "" : " at $(first_column_str(Y))")"
    save_cached_column_matrix_and_vector!(cache, file_name, title, t)

    if n_columns > 1
        maximum!(abs, reshape(column_matrix, 1, n_εs, n_εs), column_matrices)
        file_name = "exact_jacobian_max"
        title = "Exact ∂Yₜ/∂Y, max over all columns"
        save_cached_column_matrix_and_vector!(cache, file_name, title, t)

        sum!(abs, reshape(column_matrix, 1, n_εs, n_εs), column_matrices)
        column_matrix ./= n_columns
        file_name = "exact_jacobian_avg"
        title = "Exact ∂Yₜ/∂Y, avg over all columns"
        save_cached_column_matrix_and_vector!(cache, file_name, title, t)
    end
end

# Set the derivative of `sqrt(x)` to `iszero(x) ? zero(x) : inv(2 * sqrt(x))` in
# order to properly handle derivatives of `x * sqrt(x)`. Without this change,
# the derivative of `x * sqrt(x)` is `NaN` when `x` is zero. This method
# specializes on the tag `ExactJacobian` because not specializing on any tag
# overwrites the generic method for `Dual` in `ForwardDiff` and breaks
# precompilation, while specializing on the default tag `Nothing` causes the
# type piracy Aqua test to fail.
@inline function Base.sqrt(d::ForwardDiff.Dual{ExactJacobian})
    tag = Val{ExactJacobian}()
    x = ForwardDiff.value(d)
    partials = ForwardDiff.partials(d)
    val = sqrt(x)
    deriv = iszero(x) ? zero(x) : inv(2 * val)
    return ForwardDiff.dual_definition_retval(tag, val, deriv, partials)
end

# TODO: Reshape column_matrices and turn this into a single kernel launch.
function parallel_lu_factorize!(column_matrices, temporary_vector)
    @assert ndims(column_matrices) == 3
    n = size(column_matrices, 2)
    @assert size(column_matrices, 3) == n
    @inbounds for k in 1:n
        all(!isnan, view(column_matrices, :, k, k)) ||
            error("LU error: NaN on diagonal")
        all(!iszero, view(column_matrices, :, k, k)) ||
            error("LU error: 0 on diagonal")
        temporary_vector .= inv.(view(column_matrices, :, k, k))
        for i in (k + 1):n
            view(column_matrices, :, i, k) .*= temporary_vector
        end
        for j in (k + 1):n
            for i in (k + 1):n
                view(column_matrices, :, i, j) .-=
                    view(column_matrices, :, i, k) .*
                    view(column_matrices, :, k, j)
            end
        end
    end
end

# TODO: Reshape column_matrices and turn this into a single kernel launch.
function parallel_lu_solve!(column_vectors, column_matrices, temporary_vector)
    @assert ndims(column_vectors) == 2 && ndims(column_matrices) == 3
    n = size(column_matrices, 2)
    @assert size(column_vectors, 2) == n && size(column_matrices, 3) == n
    @inbounds begin
        for i in 2:n
            temporary_vector .= zero(eltype(column_vectors))
            for j in 1:(i - 1)
                temporary_vector .+=
                    view(column_matrices, :, i, j) .* view(column_vectors, :, j)
            end
            view(column_vectors, :, i) .-= temporary_vector
        end
        view(column_vectors, :, n) ./= view(column_matrices, :, n, n)
        for i in (n - 1):-1:1
            temporary_vector .= zero(eltype(column_vectors))
            for j in (i + 1):n
                temporary_vector .+=
                    view(column_matrices, :, i, j) .* view(column_vectors, :, j)
            end
            view(column_vectors, :, i) .-= temporary_vector
            view(column_vectors, :, i) ./= view(column_matrices, :, i, i)
        end
    end
end
