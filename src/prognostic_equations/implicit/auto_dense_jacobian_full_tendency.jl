import ForwardDiff
import ClimaComms
import ClimaCore.Fields as Fields

"""
    AutoDenseJacobianFullTendency([max_simultaneous_derivatives])

Dense Jacobian for the **full** tendency (remaining + implicit), for use with
DIRK timesteppers. Same as `AutoDenseJacobian` but uses `full_tendency!` and
full precomputed/scratch from `p`. The cache is built from `p` so that
`set_precomputed_quantities!` and `full_tendency!` see the full state.
"""
struct AutoDenseJacobianFullTendency{S} <: JacobianAlgorithm end
AutoDenseJacobianFullTendency(max_simultaneous_derivatives = 32) =
    AutoDenseJacobianFullTendency{max_simultaneous_derivatives}()

max_simultaneous_derivatives(::AutoDenseJacobianFullTendency{S}) where {S} = S

function jacobian_cache(alg::AutoDenseJacobianFullTendency, Y, p)
    FT = eltype(Y)
    DA = ClimaComms.array_type(Y)
    FT_dual = ForwardDiff.Dual{Jacobian, FT, max_simultaneous_derivatives(alg)}

    precomputed_dual = replace_parent_eltype(p.precomputed, FT_dual)
    scratch_dual = replace_parent_eltype(p.scratch, FT_dual)
    hyperdiff_dual = isnothing(p.hyperdiff) ? p.hyperdiff : replace_parent_eltype(p.hyperdiff, FT_dual)
    core_dual = replace_parent_eltype(p.core, FT_dual)
    Y_dual = replace_parent_eltype(Y, FT_dual)
    Yₜ_dual = similar(Y_dual)

    N = length(Fields.column(Y, 1, 1, 1))
    n_columns = Fields.ncolumns(Y.c)
    column_matrices = DA{FT}(undef, N, N, n_columns)
    column_lu_factors = copy(column_matrices)
    column_lu_vectors = DA{FT}(undef, N, n_columns)

    I_column_matrix = DA{FT}(undef, N, N)
    I_column_matrix .= 0
    I_column_matrix[diagind(I_column_matrix)] .= 1
    I_matrix = reshape(I_column_matrix, N, N, 1)

    return (;
        precomputed_dual,
        scratch_dual,
        hyperdiff_dual,
        core_dual,
        Y_dual,
        Yₜ_dual,
        column_matrices,
        column_lu_factors,
        column_lu_vectors,
        I_matrix,
        N_val = Val(N),
    )
end

function update_column_matrices!(alg::AutoDenseJacobianFullTendency, cache, Y, p, t)
    (; precomputed_dual, scratch_dual, hyperdiff_dual, core_dual, Y_dual, Yₜ_dual, column_matrices) = cache
    device = ClimaComms.device(Y.c)
    column_indices = column_index_iterator(Y)
    scalar_names = scalar_field_names(Y)
    field_vector_indices = field_vector_index_iterator(Y)
    p_dual = append_to_atmos_cache(p, precomputed_dual, scratch_dual, hyperdiff_dual, core_dual)

    jacobian_index_to_Y_index_map_partitions = Iterators.partition(
        enumerate(field_vector_indices),
        max_simultaneous_derivatives(alg),
    )
    for jacobian_index_to_Y_index_map_partition in
        ClimaComms.threadable(device, jacobian_index_to_Y_index_map_partitions)

        Y_dual .= Y
        ClimaComms.@threaded device begin
            for column_index in column_indices,
                (diagonal_entry_ε_index, (_, (scalar_index, level_index))) in
                enumerate(jacobian_index_to_Y_index_map_partition)

                n_εs_val = Val(max_simultaneous_derivatives(alg))
                ε_coefficients = ntuple(==(diagonal_entry_ε_index), n_εs_val)
                unrolled_applyat(scalar_index, scalar_names) do name
                    field = MatrixFields.get_field(Y_dual, name)
                    @inbounds point(field, level_index, column_index...)[] +=
                        ForwardDiff.Dual{Jacobian}(0, ε_coefficients)
                end
            end
        end

        set_precomputed_quantities!(Y_dual, p_dual, t)
        full_tendency!(Yₜ_dual, Y_dual, p_dual, t)

        ClimaComms.@threaded device begin
            for (matrix_index, column_index) in enumerate(column_indices),
                (jacobian_row_index, (scalar_index, level_index)) in
                enumerate(field_vector_indices)

                dual_number =
                    unrolled_applyat(scalar_index, scalar_names) do name
                        field = MatrixFields.get_field(Yₜ_dual, name)
                        @inbounds point(field, level_index, column_index...)[]
                    end
                ε_coefficients = ForwardDiff.partials(dual_number)
                for (jacobian_column_ε_index, (jacobian_column_index, _)) in
                    enumerate(jacobian_index_to_Y_index_map_partition)
                    cartesian_index = (
                        jacobian_row_index,
                        jacobian_column_index,
                        matrix_index,
                    )
                    @inbounds column_matrices[cartesian_index...] =
                        ε_coefficients[jacobian_column_ε_index]
                end
            end
        end
    end
end

function update_jacobian!(alg::AutoDenseJacobianFullTendency, cache, Y, p, dtγ, t)
    (; column_matrices, column_lu_factors, I_matrix, N_val) = cache
    device = ClimaComms.device(Y.c)
    update_column_matrices!(alg, cache, Y, p, t)
    column_lu_factors .= dtγ .* column_matrices .- I_matrix
    parallel_lu_factorize!(device, column_lu_factors, N_val)
end

function invert_jacobian!(::AutoDenseJacobianFullTendency, cache, ΔY, R)
    (; column_lu_vectors, column_lu_factors, N_val) = cache
    device = ClimaComms.device(ΔY.c)
    column_indices = column_index_iterator(ΔY)
    scalar_names = scalar_field_names(ΔY)
    vector_index_to_field_vector_index_map =
        enumerate(field_vector_index_iterator(ΔY))

    ClimaComms.@threaded device begin
        for (vector_index, column_index) in enumerate(column_indices),
            (scalar_level_index, (scalar_index, level_index)) in
            vector_index_to_field_vector_index_map

            number = unrolled_applyat(scalar_index, scalar_names) do name
                field = MatrixFields.get_field(R, name)
                @inbounds point(field, level_index, column_index...)[]
            end
            @inbounds column_lu_vectors[scalar_level_index, vector_index] =
                number
        end
    end

    parallel_lu_solve!(device, column_lu_vectors, column_lu_factors, N_val)

    ClimaComms.@threaded device begin
        for (vector_index, column_index) in enumerate(column_indices),
            (scalar_level_index, (scalar_index, level_index)) in
            vector_index_to_field_vector_index_map

            @inbounds number =
                column_lu_vectors[scalar_level_index, vector_index]
            unrolled_applyat(scalar_index, scalar_names) do name
                field = MatrixFields.get_field(ΔY, name)
                @inbounds point(field, level_index, column_index...)[] = number
            end
        end
    end
end

# Constructor for full-tendency Jacobian (cache built from p, not Y and atmos).
function Jacobian(alg::AutoDenseJacobianFullTendency, Y, p; verbose = false)
    krylov_cache = (; ΔY_krylov = similar(Y), R_krylov = similar(Y))
    cache = (; jacobian_cache(alg, Y, p)..., krylov_cache...)
    return Jacobian(alg, cache)
end
