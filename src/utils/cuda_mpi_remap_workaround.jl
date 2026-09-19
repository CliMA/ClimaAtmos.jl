# Cray CUDA-aware MPI_Reduce of large device buffers fails with MPI_ERR_TRUNCATE
# during NetCDF remapping (`ClimaCore.Remapping._collect_interpolated_values!`).
# Instantaneous lat-lon dumps still interpolate on the GPU; only the gather to
# rank 0 is done on the host. See ClimaCore `distributed_remapping.jl`.

const _CUDA_MPI_REMAP_HOST_REDUCE_LOGGED = Ref(false)
const _CUDA_MPI_REMAP_HOST_REDUCE_INSTALLED = Ref(false)

function _install_cuda_mpi_remap_host_reduce!()
    # `@eval` into Remapping is forbidden while compiling ClimaAtmos. Only
    # `__init__` should call this (load time, after the image exists).
    ccall(:jl_generating_output, Cint, ()) == 1 && return nothing
    _CUDA_MPI_REMAP_HOST_REDUCE_INSTALLED[] && return nothing
    _CUDA_MPI_REMAP_HOST_REDUCE_INSTALLED[] = true
    logged = _CUDA_MPI_REMAP_HOST_REDUCE_LOGGED
    @eval ClimaCore.Remapping begin
        function interpolate(remapper::Remapper, fields)
            ArrayType = ClimaComms.array_type(remapper.space)
            FT = Spaces.undertype(remapper.space)
            only_one_field = fields isa Fields.Field

            interpolated_values_dim..., _buffer_length =
                size(remapper._interpolated_values)

            allocate_extra = only_one_field ? () : (length(fields),)
            dest = if ClimaComms.iamroot(remapper.comms_ctx)
                ArrayType(
                    zeros(FT, interpolated_values_dim..., allocate_extra...),
                )
            else
                nothing
            end

            interpolate!(dest, remapper, fields)
            return dest
        end

        function _collect_interpolated_values!(
            dest,
            remapper::Remapper,
            index_field_begin::Int,
            index_field_end::Int;
            only_one_field,
        )
            cuda_synchronize(ClimaComms.device(remapper.comms_ctx))
            ctx = remapper.comms_ctx
            device = ClimaComms.device(ctx)

            sendbuf = if only_one_field
                view(remapper._interpolated_values, remapper.colons..., 1)
            else
                num_fields = 1 + index_field_end - index_field_begin
                view(
                    remapper._interpolated_values,
                    remapper.colons...,
                    1:num_fields,
                )
            end

            recvbuf = if isnothing(dest)
                nothing
            elseif only_one_field
                dest
            else
                view(
                    dest,
                    remapper.colons...,
                    index_field_begin:index_field_end,
                )
            end

            if device isa ClimaComms.CUDADevice &&
               ctx isa ClimaComms.MPICommsContext
                if !$logged[]
                    $logged[] = true
                    @info "NetCDF remapping: host MPI_Reduce (CUDA+MPI)"
                end
                send_h = Array(sendbuf)
                if ClimaComms.iamroot(ctx)
                    recv_h = similar(send_h)
                    ClimaComms.reduce!(ctx, send_h, recv_h, +)
                    isnothing(recvbuf) || copyto!(recvbuf, recv_h)
                else
                    ClimaComms.reduce!(ctx, send_h, nothing, +)
                end
            else
                ClimaComms.reduce!(ctx, sendbuf, recvbuf, +)
            end
            return nothing
        end
    end
    return nothing
end
