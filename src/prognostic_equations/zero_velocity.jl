###
### EDMFX advection test (zero velocity)
###

import ClimaCore: MatrixFields
import LinearAlgebra: UniformScaling

"""
    zero_velocity_tendency!(Yₜ, Y, p, t)

Forcibly sets all velocity-related tendencies in `Yₜ` to zero if the
simulation is configured for an "advection test" (`p.atmos.advection_test == true`).

This includes:
- Grid-mean horizontal velocity (`Yₜ.c.uₕ`).
- Grid-mean vertical velocity (`Yₜ.f.u₃`).
- EDMFX updraft vertical velocities (`Yₜ.f.sgsʲs.:(j).u₃`) if using `PrognosticEDMFX`.

This function is called at the end of the tendency calculation pipeline
during an advection test to ensure that velocities do not evolve, effectively
keeping them prescribed or frozen for the purpose of the test.

Arguments:
- `Yₜ`: The tendency state vector, modified in place.
- `Y`: The current state vector (not directly used but part of standard signature).
- `p`: Cache containing parameters and atmospheric model configurations (e.g.,
       `p.atmos.advection_test`, `p.atmos.turbconv_model`).
- `t`: Current simulation time (not directly used but part of standard signature).
"""
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

"""
    zero_velocity_jacobian!(∂Yₜ_err_∂Y, Y, p, t)

Modifies the Jacobian matrix `∂Yₜ_err_∂Y` for an "advection test" scenario
(`p.atmos.advection_test == true`).

For matrix rows corresponding to velocity variables (grid-mean `uₕ`, `u₃`, and
EDMFX updraft `u₃ʲ` if applicable):
- Diagonal blocks (e.g., `∂(uₕ_tendency)/∂uₕ`) are set to represent `-I` (negative identity).
- Off-diagonal blocks (e.g., `∂(uₕ_tendency)/∂(tracer)`) are set to zero.

This effectively decouples the velocity evolution from other variables in the
linearized system used by an implicit solver, or simplifies their implicit
contribution to behave like a strong relaxation to zero tendency (if the Jacobian
is for `Y' - Δt J Y'` and `J = -I/Δt`). This is useful if velocities
are intended to be "frozen" or follow a prescribed path during the test.

Arguments:
- `∂Yₜ_err_∂Y`: The Jacobian matrix (a `MatrixFields.FieldMatrix`), modified in place.
- `Y`: The current state vector.
- `p`: Cache containing parameters and model configurations.
- `t`: Current simulation time.
"""
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

"""
    set_identity_matrix_entry!(matrix_entry, row_name, col_name)

Helper function to set a `matrix_entry` (a block in a Jacobian `Field`)
to either a scaled negative identity matrix (if `row_name == col_name`, i.e.,
a diagonal block) or zero (if `row_name != col_name`, i.e., an off-diagonal block).

Specifically, for diagonal blocks, it sets the entry to represent `-I` (negative
identity), where `I` is an `AxisTensor` identity of the appropriate type and structure.
This is used within `zero_velocity_jacobian!` to modify Jacobian contributions
related to velocity variables during an advection test.

Arguments:
- `matrix_entry`: A `ClimaCore.Fields.Field` representing a block of the Jacobian matrix.
                  It is modified in place.
- `row_name`: `MatrixFields.FieldName` identifying the row variable of this block.
- `col_name`: `MatrixFields.FieldName` identifying the column variable of this block.
"""
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
