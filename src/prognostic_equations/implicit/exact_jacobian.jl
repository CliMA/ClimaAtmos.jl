"""
    ExactJacobian()

A `JacobianAlgorithm` that computes the `ImplicitEquationJacobian` using
forward-mode automatic differentiation and inverts it using LU factorization.
"""
struct ExactJacobian <: JacobianAlgorithm end

function jacobian_cache(alg::ExactJacobian, Y, atmos)
    FT = eltype(Y)
    DA = ClimaComms.array_type(Y)
    Y_columns = column_iterator(Y)
    n_columns = length(Y_columns)
    n_εs = length(first(Y_columns))
    dual_type = ForwardDiff.Dual{ExactJacobian, FT, 1}

    similar_with_dual_fields(named_tuple) =
        Fields._values(similar(Fields.FieldVector(; named_tuple...), dual_type))

    Y_dual = similar(Y, dual_type)
    Y_err_dual = similar(Y_dual)
    precomputed_original = implicit_precomputed_quantities(Y, atmos)
    precomputed_dual = similar_with_dual_fields(precomputed_original)
    scratch_dual = similar_with_dual_fields(temporary_quantities(Y, atmos))

    column_vectors = DA{FT}(undef, n_columns, n_εs)
    column_matrices = DA{FT}(undef, n_columns, n_εs, n_εs)
    column_factorized_matrices = copy(column_matrices)

    # LinearAlgebra.I does not support broadcasting, so we need a workaround.
    I_matrix = DA{FT}(undef, 1, n_εs, n_εs)
    I_matrix .= 0
    view(I_matrix, 1, :, :)[LinearAlgebra.diagind(view(I_matrix, 1, :, :))] .= 1

    lu_cache = DA{FT}(undef, n_columns)

    return (;
        Y_dual,
        Y_err_dual,
        precomputed_dual,
        scratch_dual,
        column_vectors,
        column_matrices,
        column_factorized_matrices,
        I_matrix,
        lu_cache,
    )
end

function update_jacobian!(alg::ExactJacobian, cache, Y, p, dtγ, t)
    (; Y_dual, Y_err_dual, precomputed_dual, scratch_dual) = cache
    (; column_matrices, column_factorized_matrices, I_matrix, lu_cache) = cache

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
    Y_err_dual_columns = column_iterator(Y_err_dual)
    ε = ForwardDiff.Dual{ExactJacobian}(0, 1)

    Y_dual .= Y
    for (ε_index, Y_dual_scalar_level) in enumerate(Y_dual_scalar_levels)
        dual_array = parent(Y_dual_scalar_level)

        dual_array .+= ε # add ε to this level of Y_dual
        set_implicit_precomputed_quantities!(Y_dual, p_dual, t)
        implicit_tendency!(Y_err_dual, Y_dual, p_dual, t)
        dual_array .= ForwardDiff.value.(dual_array) # remove ε from this level

        for (col_index, Y_err_dual_column) in enumerate(Y_err_dual_columns)
            column_matrix_slice = view(column_matrices, col_index, :, ε_index)
            vector_to_fieldvector(column_matrix_slice, Y_err_dual_column) .=
                first.(ForwardDiff.partials.(Y_err_dual_column))
        end # TODO: Parallelize this loop.
    end # TODO: Make this loop use batches of up to 32 values of ε at a time.

    column_factorized_matrices .= dtγ .* column_matrices .- I_matrix
    parallel_lu_factorize!(column_factorized_matrices, lu_cache)
end

function invert_jacobian!(alg::ExactJacobian, cache, ΔY, R)
    (; column_vectors, column_factorized_matrices, lu_cache) = cache
    for (col_index, R_column) in enumerate(column_iterator(R))
        column_vector = view(column_vectors, col_index, :)
        vector_to_fieldvector(column_vector, R_column) .= R_column
    end # TODO: Parallelize this loop.
    parallel_lu_solve!(column_vectors, column_factorized_matrices, lu_cache)
    for (col_index, ΔY_column) in enumerate(column_iterator(ΔY))
        column_vector = view(column_vectors, col_index, :)
        ΔY_column .= vector_to_fieldvector(column_vector, ΔY_column)
    end # TODO: Parallelize this loop.
end

function save_jacobian(alg::ExactJacobian, cache, Y, dtγ, t)
    (; column_matrices, column_matrix) = cache
    (n_columns, n_εs, _) = size(column_matrices)
    one_column = n_columns == 1

    column_matrix .= view(column_matrices, 1, :, :)
    file_name = "exact_jacobian" * (one_column ? "" : "_first")
    description =
        "Exact ∂Yₜ/∂Y matrix" *
        (one_column ? "" : " at $(first_column_coordinate_string(Y))")
    save_column_matrix(cache, file_name, description, Y, t)

    if !one_column
        reshaped_column_matrix = reshape(column_matrix, 1, n_εs, n_εs)

        maximum!(abs, reshaped_column_matrix, column_matrices)
        file_name = "exact_jacobian_max"
        description = "Maximum of exact ∂Yₜ/∂Y matrix over all columns"
        save_column_matrix(cache, file_name, description, Y, t)

        sum!(abs, reshaped_column_matrix, column_matrices)
        column_matrix ./= n_columns
        file_name = "exact_jacobian_avg"
        description = "Average of exact ∂Yₜ/∂Y matrix over all columns"
        save_column_matrix(cache, file_name, description, Y, t)
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
