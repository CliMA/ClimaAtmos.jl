import ClimaCore: MatrixFields
import LinearAlgebra: UniformScaling

###
### edmfx advection test 
###

# Turn off all momentum tendencies in the advection test.
function zero_velocity_tendency!(Yₜ, Y, p, t)
    p.atmos.advection_test || return nothing
    @. Yₜ.c.uₕ = C12(0, 0)
    @. Yₜ.f.u₃ = C3(0)
    if p.atmos.turbconv_model isa PrognosticEDMFX
        for j in 1:n_mass_flux_subdomains(p.atmos.turbconv_model)
            @. Yₜ.f.sgsʲs.:($$j).u₃ = C3(0)
        end
    end
    return nothing
end

# Turn off all momentum tendency derivatives in the advection test.
function zero_velocity_jacobian!(∂Yₜ_err_∂Y, Y, p, t)
    p.atmos.advection_test || return nothing
    for ((row_name, col_name), matrix_entry) in pairs(∂Yₜ_err_∂Y)
        matrix_entry isa Fields.Field || continue
        if row_name in (MatrixFields.@name(c.uₕ), MatrixFields.@name(f.u₃))
            set_identity_matrix_entry!(matrix_entry, row_name, col_name)
        end
        if p.atmos.turbconv_model isa PrognosticEDMFX
            for j in 1:n_mass_flux_subdomains(p.atmos.turbconv_model)
                if row_name == MatrixFields.FieldName(:f, :sgsʲs, j, :u₃)
                    set_identity_matrix_entry!(matrix_entry, row_name, col_name)
                end
            end
        end
    end
end

function set_identity_matrix_entry!(matrix_entry, row_name, col_name)
    identity_matrix_entry_value = if row_name == col_name
        # TODO: Add a method for one(::Axis2Tensor) to simplify this.
        T = eltype(eltype(matrix_entry))
        tensor_data = UniformScaling(one(eltype(T)))
        -DiagonalMatrixRow(Geometry.AxisTensor(axes(T), tensor_data))
    else
        zero(eltype(matrix_entry))
    end
    matrix_entry .= (identity_matrix_entry_value,)
end
