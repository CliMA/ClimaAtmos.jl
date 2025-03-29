"""
    ExactJacobian()

A `JacobianAlgorithm` that computes the `ImplicitEquationJacobian` using
forward-mode automatic differentiation and inverts it using LU factorization.
"""
@kwdef struct ExactJacobian <: JacobianAlgorithm
    only_first_column::Bool = false # TODO: Consider making this type-stable
    always_update_exact_jacobian::Bool = true # TODO: Remove this flag
end

function jacobian_cache(alg::ExactJacobian, Y, atmos)
    FT = eltype(Y)
    DA = ClimaComms.array_type(Y)
    Y_columns = column_iterator(Y)
    n_columns = alg.only_first_column ? 1 : length(Y_columns)
    n_εs = length(first(Y_columns))
    dual_type = ForwardDiff.Dual{ExactJacobian, FT, 1}

    similar_with_dual_fields(named_tuple) =
        Fields._values(similar(Fields.FieldVector(; named_tuple...), dual_type))

    Y_dual = similar(Y, dual_type)
    Y_dual_copy = similar(Y_dual)
    Y_err_dual = similar(Y_dual)
    precomputed_original = implicit_precomputed_quantities(Y, atmos)
    precomputed_dual = similar_with_dual_fields(precomputed_original)
    scratch_dual = similar_with_dual_fields(temporary_quantities(Y, atmos))

    column_vectors = DA{FT}(undef, n_columns, n_εs)
    column_matrices = DA{FT}(undef, n_columns, n_εs, n_εs)
    column_factorized_matrices = copy(column_matrices)
    lu_cache = DA{FT}(undef, n_columns)
    dtγ_ref = Ref{FT}()

    return (;
        Y_dual,
        Y_dual_copy,
        Y_err_dual,
        precomputed_dual,
        scratch_dual,
        column_vectors,
        column_matrices,
        column_factorized_matrices,
        lu_cache,
        dtγ_ref,
    )
end

always_update_exact_jacobian(alg::ExactJacobian) =
    alg.always_update_exact_jacobian

function factorize_exact_jacobian!(alg::ExactJacobian, cache, Y, p, dtγ, t)
    (; Y_dual, Y_dual_copy, Y_err_dual, precomputed_dual, scratch_dual) = cache
    (; column_matrices, column_factorized_matrices, lu_cache) = cache

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
        Y_dual_copy .= Y_dual # make a copy since Y_dual is changed at boundary
        set_implicit_precomputed_quantities!(Y_dual, p_dual, t)
        implicit_tendency!(Y_err_dual, Y_dual, p_dual, t) # tendency derivative
        @. Y_err_dual = dtγ * Y_err_dual - Y_dual_copy # residual derivative
        dual_array .= ForwardDiff.value.(dual_array) # remove ε from this level
        # TODO: Don't compute Y_err_dual in every column if it isn't needed.
        for (col_index, Y_err_dual_column) in enumerate(Y_err_dual_columns)
            alg.only_first_column && col_index > 1 && break
            column_matrix_slice = view(column_matrices, col_index, :, ε_index)
            vector_to_fieldvector(column_matrix_slice, Y_err_dual_column) .=
                first.(ForwardDiff.partials.(Y_err_dual_column))
        end # TODO: Parallelize this loop.
    end # TODO: Make this loop use batches of up to 32 values of ε at a time.

    # TODO: We only need to copy if the unfactorized matrix will be plotted.
    column_factorized_matrices .= column_matrices
    cache.dtγ_ref[] = dtγ

    parallel_lu_factorize!(column_factorized_matrices, lu_cache)
end

approximate_jacobian!(::ExactJacobian, _, _, _, _, _) = nothing

function invert_jacobian!(alg::ExactJacobian, cache, x, b)
    (; column_vectors, column_factorized_matrices, lu_cache) = cache
    for (col_index, b_column) in enumerate(column_iterator(b))
        alg.only_first_column && col_index > 1 && break
        column_vector = view(column_vectors, col_index, :)
        vector_to_fieldvector(column_vector, b_column) .= b_column
    end # TODO: Parallelize this loop.
    parallel_lu_solve!(column_vectors, column_factorized_matrices, lu_cache)
    for (col_index, x_column) in enumerate(column_iterator(x))
        alg.only_first_column && col_index > 1 && break
        column_vector = view(column_vectors, col_index, :)
        x_column .= vector_to_fieldvector(column_vector, x_column)
    end # TODO: Parallelize this loop.
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

function parallel_lu_factorize!(As, temporary_vector)
    @assert ndims(As) == 3
    n = size(As, 2)
    @assert size(As, 3) == n
    @inbounds for k in 1:n
        all(!isnan, view(As, :, k, k)) || error("LU error: NaN on diagonal")
        all(!iszero, view(As, :, k, k)) || error("LU error: 0 on diagonal")
        temporary_vector .= inv.(view(As, :, k, k))
        for i in (k + 1):n
            view(As, :, i, k) .*= temporary_vector
        end
        for j in (k + 1):n
            for i in (k + 1):n
                view(As, :, i, j) .-= view(As, :, i, k) .* view(As, :, k, j)
            end
        end
    end
end

function parallel_lu_solve!(xs, As, temporary_vector)
    @assert ndims(xs) == 2 && ndims(As) == 3
    n = size(As, 2)
    @assert size(xs, 2) == n && size(As, 3) == n
    @inbounds begin
        for i in 2:n
            temporary_vector .= zero(eltype(xs))
            for j in 1:(i - 1)
                temporary_vector .+= view(As, :, i, j) .* view(xs, :, j)
            end
            view(xs, :, i) .-= temporary_vector
        end
        view(xs, :, n) ./= view(As, :, n, n)
        for i in (n - 1):-1:1
            temporary_vector .= zero(eltype(xs))
            for j in (i + 1):n
                temporary_vector .+= view(As, :, i, j) .* view(xs, :, j)
            end
            view(xs, :, i) .-= temporary_vector
            view(xs, :, i) ./= view(As, :, i, i)
        end
    end
end
