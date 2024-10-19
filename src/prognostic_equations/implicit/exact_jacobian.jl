@kwdef struct ExactJacobian <: JacobianAlgorithm
    differentiate_all_precomputed::Bool = false
    always_update_exact_jacobian::Bool = true
    preserve_unfactorized_jacobian::Bool = false # also records the value of dtγ
end

function jacobian_cache(alg::ExactJacobian, Y, atmos)
    FT = eltype(Y)
    DA = ClimaComms.array_type(Y)
    Y_columns = column_iterator(Y)
    n_columns = length(Y_columns)
    n_εs = length(first(Y_columns))
    dual_type = ForwardDiff.Dual{ExactJacobian, FT, 1}
    (; differentiate_all_precomputed, preserve_unfactorized_jacobian) = alg

    similar_with_dual_fields(named_tuple) =
        Fields._values(similar(Fields.FieldVector(; named_tuple...), dual_type))

    Y_dual = similar(Y, dual_type)
    Y_dual_copy = similar(Y_dual)
    Y_err_dual = similar(Y_dual)
    precomputed_original =
        differentiate_all_precomputed ? precomputed_quantities(Y, atmos) :
        implicit_precomputed_quantities(Y, atmos)
    precomputed_dual = similar_with_dual_fields(precomputed_original)
    scratch_dual = similar_with_dual_fields(temporary_quantities(Y, atmos))

    column_partials_vector = DA{ForwardDiff.Partials{1, FT}}(undef, n_εs)
    column_vectors = DA{FT}(undef, n_columns, n_εs)
    column_matrices = DA{FT}(undef, n_columns, n_εs, n_εs)
    column_factorized_matrices =
        preserve_unfactorized_jacobian ? copy(column_matrices) : column_matrices
    lu_cache = DA{FT}(undef, n_columns)
    dtγ_kwarg = preserve_unfactorized_jacobian ? (; dtγ_ref = Ref{FT}()) : (;)

    return (;
        Y_dual,
        Y_dual_copy,
        Y_err_dual,
        precomputed_dual,
        scratch_dual,
        column_partials_vector,
        column_vectors,
        column_matrices,
        column_factorized_matrices,
        lu_cache,
        dtγ_kwarg...,
    )
end

always_update_exact_jacobian(alg::ExactJacobian) =
    alg.always_update_exact_jacobian

# function factorize_exact_jacobian!(alg::ExactJacobian, cache, Y, p, dtγ, t)
#     ...
#     Y_dual .= Y
#     zeros_tuple = ntuple(_ -> 0, n_εs)
#     for (ε_index, Y_dual_scalar_level) in enumerate(Y_dual_scalar_levels)
#         ε_partials = Base.setindex(zeros_tuple, 1, ε_index)
#         ε = ForwardDiff.Dual{ExactJacobian}(0, ε_partials...)
#         parent(Y_dual_scalar_level) .+= ε
#     end # add a unique infinitesimal ε to the scalar values on each level of Y
#     Y_dual_copy .= Y_dual # need a copy because Y_dual gets changed at boundary
#     set_implicit_precomputed_quantities!(Y_dual, p_dual, t) # changes Y_dual
#     implicit_tendency!(Y_err_dual, Y_dual, p_dual, t) # the tendency Jacobian
#     @. Y_err_dual = dtγ * Y_err_dual - Y_dual_copy # the residual Jacobian
#     for (col_index, Y_err_dual_column) in enumerate(Y_err_dual_columns)
#         column_partials_vector = cache.column_partials_vectors[col_index]
#         column_matrix = cache.column_matrices[col_index]
#         column_factorized_matrix = cache.column_factorized_matrices[col_index]
#         column_partials_vector .= ForwardDiff.partials.(Y_err_dual_column)
#         column_matrix .= reinterpret(reshape, FT, column_partials_vector)'
#     end # TODO: Parallelize this loop
#     ...
# end
function factorize_exact_jacobian!(alg::ExactJacobian, cache, Y, p, dtγ, t)
    (; differentiate_all_precomputed, preserve_unfactorized_jacobian) = alg
    (; Y_dual, Y_dual_copy, Y_err_dual) = cache
    (; precomputed_dual, scratch_dual, column_partials_vector) = cache
    (; column_matrices, column_factorized_matrices, lu_cache) = cache

    differentiate_all_precomputed || set_precomputed_quantities!(Y, p, t)
    p_dual_args = ntuple(Val(fieldcount(typeof(p)))) do cache_field_index
        cache_field_name = fieldname(typeof(p), cache_field_index)
        if cache_field_name == :precomputed
            differentiate_all_precomputed ? precomputed_dual :
            (; p.precomputed..., precomputed_dual...)
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
    ε = ForwardDiff.Dual{ExactJacobian}(0, 1)

    Y_dual .= Y
    for (ε_index, Y_dual_scalar_level) in enumerate(Y_dual_scalar_levels)
        dual_array = parent(Y_dual_scalar_level)
        dual_array .+= ε # add ε to this level of Y_dual
        Y_dual_copy .= Y_dual # make a copy since Y_dual is changed at boundary
        differentiate_all_precomputed ?
        set_precomputed_quantities!(Y_dual, p_dual, t) :
        set_implicit_precomputed_quantities!(Y_dual, p_dual, t)
        implicit_tendency!(Y_err_dual, Y_dual, p_dual, t) # tendency derivative
        @. Y_err_dual = dtγ * Y_err_dual - Y_dual_copy # residual derivative
        dual_array .= ForwardDiff.value.(dual_array) # remove ε from this level
        for (col_index, Y_err_dual_column) in enumerate(Y_err_dual_columns)
            column_partials_vector .= ForwardDiff.partials.(Y_err_dual_column)
            view(column_matrices, col_index, :, ε_index) .=
                reinterpret(reshape, FT, column_partials_vector)
        end # TODO: Parallelize this loop.
    end # TODO: Make this loop use batches of up to 32 values of ε at a time.

    if preserve_unfactorized_jacobian
        column_factorized_matrices .= column_matrices
        cache.dtγ_ref[] = dtγ
    end
    parallel_lu_factorize!(column_factorized_matrices, lu_cache)
end

approximate_jacobian!(::ExactJacobian, _, _, _, _, _) = nothing

function invert_jacobian!(::ExactJacobian, cache, x, b)
    (; column_vectors, column_factorized_matrices, lu_cache) = cache
    for (col_index, b_column) in enumerate(column_iterator(b))
        view(column_vectors, col_index, :) .= b_column
    end # TODO: Parallelize this loop.
    parallel_lu_solve!(column_vectors, column_factorized_matrices, lu_cache)
    for (col_index, x_column) in enumerate(column_iterator(x))
        x_column .= view(column_vectors, col_index, :)
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

#=
struct DenseFieldMatrix{C, N}
    column_matrices::C
    name_map::N
end
function DenseFieldMatrix(field_vector)
    FT = eltype(field_vector)
    DA = ClimaComms.array_type(field_vector)
    column_field_vectors = column_iterator(field_vector)
    n_columns = length(column_field_vectors)
    matrix_size = length(first(column_field_vectors))
    column_matrices = DA{FT}(undef, n_columns, matrix_size, matrix_size)
    name_map = FieldNameDict(
        FieldNameSet{Any}(scalar_field_names(field_vector)),
        scalar_field_index_ranges(field_vector)
    )
    return DenseFieldMatrix(column_matrices, name_map)
end
function Base.copyto!(dest::DenseFieldMatrix, src::FieldMatrix)
    (; column_matrices, name_map)
    FT = eltype(column_matrices)
    all_scalar_names = keys(name_map)
    fill!(column_matrices, zero(FT))
    unrolled_foreach(pairs(src)) do (row_block_name, col_block_name), block_data
        is_row_in_block = Base.Fix2(is_child_name, row_block_name)
        is_col_in_block = Base.Fix2(is_child_name, col_block_name)
        scalar_row_names = unrolled_filter(is_row_in_block, all_scalar_names)
        scalar_col_names = unrolled_filter(is_col_in_block, all_scalar_names)
        for column_index in axes(column_matrices, 1) # TODO: Parallelize this.
            for (block_row, row_name) in enumerate(scalar_row_names)
                for (block_col, col_name) in enumerate(scalar_col_names)
                    scalar_matrix = view(
                        column_matrices,
                        column_index,
                        name_map[row_name],
                        name_map[col_name],
                    )
                    if block_data isa UniformScaling && block_row == block_col
                        diagonal_indices = LinearAlgebra.diagind(scalar_matrix)
                        view(scalar_matrix, diagonal_indices) .= block_data.λ
                    elseif eltype(block_data) <: FT
                        scalar_matrix
                    end
                end
            end
        end
    end
    return dest
end
=#
