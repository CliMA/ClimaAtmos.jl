@kwdef struct ExactJacobian <: JacobianAlgorithm
    always_update_exact_jacobian::Bool = true
    preserve_unfactorized_jacobian::Bool = false # only true for DebugJacobian
end

function jacobian_cache(alg::ExactJacobian, Y, p)
    FT = eltype(Y)
    DA = ClimaComms.array_type(Y)
    Y_columns = column_iterator(Y)
    n_εs = length(first(Y_columns))
    dual_type = ForwardDiff.Dual{ExactJacobian, FT, n_εs}

    similar_with_dual_fields(named_tuple) =
        Fields._values(similar(Fields.FieldVector(; named_tuple...), dual_type))

    Y_dual = similar(Y, dual_type)
    Y_dual_copy = similar(Y_dual)
    Y_err_dual = similar(Y_dual)
    precomputed_dual = similar_with_dual_fields(p.precomputed)
    scratch_dual = similar_with_dual_fields(p.scratch)

    column_vectors = map(_ -> DA{FT}(undef, n_εs), Y_columns)
    column_partials_vectors =
        map(_ -> DA{ForwardDiff.Partials{n_εs, FT}}(undef, n_εs), Y_columns)
    column_matrices = map(_ -> DA{FT}(I, n_εs, n_εs), Y_columns)
    column_factorized_matrices =
        alg.preserve_unfactorized_jacobian ? deepcopy(column_matrices) :
        column_matrices

    return (;
        Y_dual,
        Y_dual_copy,
        Y_err_dual,
        precomputed_dual,
        scratch_dual,
        column_vectors,
        column_partials_vectors,
        column_matrices,
        column_factorized_matrices,
    )
end

always_update_exact_jacobian(alg::ExactJacobian) =
    alg.always_update_exact_jacobian

function factorize_exact_jacobian!(alg::ExactJacobian, cache, Y, p, dtγ, t)
    (; preserve_unfactorized_jacobian) = alg
    (; Y_dual, Y_dual_copy, Y_err_dual, precomputed_dual, scratch_dual) = cache

    p_dual_args = ntuple(Val(fieldcount(typeof(p)))) do cache_field_index
        cache_field_name = fieldname(typeof(p), cache_field_index)
        if cache_field_name == :precomputed
            precomputed_dual
        elseif cache_field_name == :scratch
            scratch_dual
        else
            getfield(p, cache_field_index)
        end
    end
    p_dual = AtmosCache(p_dual_args...)

    FT = eltype(Y)
    Y_dual_scalar_levels = scalar_level_iterator(Y_dual)
    Y_err_dual_columns = column_iterator(Y_err_dual)
    n_εs = length(first(Y_err_dual_columns))

    Y_dual .= Y
    zeros_tuple = ntuple(_ -> 0, n_εs)
    for (ε_index, Y_dual_scalar_level) in enumerate(Y_dual_scalar_levels)
        ε_partials = Base.setindex(zeros_tuple, 1, ε_index)
        ε = ForwardDiff.Dual{ExactJacobian}(0, ε_partials...)
        parent(Y_dual_scalar_level) .+= ε
    end # add a unique infinitesimal ε to the scalar values on each level of Y
    Y_dual_copy .= Y_dual # need a copy because Y_dual gets changed at boundary
    set_precomputed_quantities!(Y_dual, p_dual, t) # changes Y_dual at boundary
    implicit_tendency!(Y_err_dual, Y_dual, p_dual, t) # the tendency Jacobian
    @. Y_err_dual = dtγ * Y_err_dual - Y_dual_copy # the residual Jacobian

    for (col_index, Y_err_dual_column) in enumerate(Y_err_dual_columns)
        column_partials_vector = cache.column_partials_vectors[col_index]
        column_matrix = cache.column_matrices[col_index]
        column_factorized_matrix = cache.column_factorized_matrices[col_index]

        column_partials_vector .= ForwardDiff.partials.(Y_err_dual_column)
        column_matrix .= reinterpret(reshape, FT, column_partials_vector)'
        if preserve_unfactorized_jacobian
            column_factorized_matrix .= column_matrix
        end
        lu_factorize!(column_factorized_matrix)
    end # TODO: Parallelize this loop
end

approximate_jacobian!(::ExactJacobian, _, _, _, _, _) = nothing

invert_jacobian!(::ExactJacobian, cache, x, b) =
    for (col_index, (x_column, b_column)) in enumerate(column_iterator(x, b))
        column_vector = cache.column_vectors[col_index]
        column_factorized_matrix = cache.column_factorized_matrices[col_index]

        column_vector .= b_column
        lu_solve!(column_vector, column_factorized_matrix)
        x_column .= column_vector
    end # TODO: Parallelize this loop

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

function scalar_level_iterator(field_vector)
    Iterators.flatmap(scalar_field_names(field_vector)) do name
        field = MatrixFields.get_field(field_vector, name)
        if field isa Fields.SpectralElementField
            (field,)
        else
            Iterators.map(1:Spaces.nlevels(axes(field))) do v
                Fields.level(field, v - 1 + Operators.left_idx(axes(field)))
            end
        end
    end
end

import ClimaCore.Fields.CUDA: @allowscalar
function lu_factorize!(A)
    n = size(A, 1)
    @assert size(A) == (n, n)
    @allowscalar @inbounds for k in 1:n
        A_kk = A[k, k]
        (A_kk != 0 && isfinite(A_kk)) ||
            error("Cannot get LU factors due to $A_kk on the matrix diagonal")
        A_kk_inv = inv(A_kk)
        for i in (k + 1):n
            A[i, k] *= A_kk_inv
        end
        for j in (k + 1):n
            for i in (k + 1):n
                A[i, j] -= A[i, k] * A[k, j]
            end
        end
    end
end
function lu_solve!(x, A)
    n = size(A, 1)
    @assert size(A) == (n, n) && size(x) == (n,)
    @allowscalar @inbounds begin
        for i in 2:n
            Δx_i = zero(eltype(x))
            for j in 1:(i - 1)
                Δx_i += A[i, j] * x[j]
            end
            x[i] -= Δx_i
        end
        x[n] /= A[n, n]
        for i in (n - 1):-1:1
            Δx_i = zero(eltype(x))
            for j in (i + 1):n
                Δx_i += A[i, j] * x[j]
            end
            x[i] -= Δx_i
            x[i] /= A[i, i]
        end
    end
end
