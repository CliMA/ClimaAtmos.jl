###
### EDMFX advection test (zero velocity)
###

import ClimaCore: MatrixFields
import LinearAlgebra: UniformScaling

"""
    zero_velocity_tendency!(Yₜ, Y, p, t)

Overwrite all velocity tendencies in `Yₜ` with zero, freezing the velocities of an
advection test.

Returns immediately unless `p.atmos.advection_test` is `true`. Otherwise sets
`Yₜ.c.uₕ`, `Yₜ.f.u₃`, and, for `PrognosticEDMFX`, every updraft
`Yₜ.f.sgsʲs.:(j).u₃` to zero. Unlike the other tendency functions, this one
assigns rather than accumulates, so it must be called last in
`additional_tendency!`: any velocity tendency added afterwards would survive.
`Y` and `t` are unused. Returns `nothing`.
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

Strip the implicit velocity coupling out of the Jacobian `∂Yₜ_err_∂Y`, the
counterpart of `zero_velocity_tendency!` for an advection test.

Returns immediately unless `p.atmos.advection_test` is `true`. Otherwise, for
every row belonging to a velocity variable (`c.uₕ`, `f.u₃`, and every updraft
`f.sgsʲs.:(j).u₃` under `PrognosticEDMFX`), the diagonal block is set to `-I` and
the off-diagonal blocks to zero, via `set_identity_matrix_entry!`. In this
Jacobian's convention a `-I` diagonal block means the variable has no implicit
contribution to its own error, so the velocities are left entirely to the
(zeroed) explicit tendency and stay frozen.

# Arguments

  - `∂Yₜ_err_∂Y`: Jacobian of the implicit residual, a `MatrixFields.FieldMatrix`,
    modified in place.
  - `Y`: Current state vector; unused.
  - `p`: Cache; only `p.atmos` is read.
  - `t`: Current simulation time; unused.

Returns `nothing`.
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

Fill a Jacobian block `matrix_entry` with `-I` when it is diagonal
(`row_name == col_name`) and with zero otherwise.

`matrix_entry` is a `ClimaCore.Fields.Field` of matrix rows, modified in place;
`row_name` and `col_name` are the `MatrixFields.FieldName`s of the block's row and
column variables. Called from `zero_velocity_jacobian!`.
"""
function set_identity_matrix_entry!(matrix_entry, row_name, col_name)
    identity_matrix_entry_value = if row_name == col_name
        # TODO: Add a method for one(::Axis2Tensor) to simplify this.
        T = eltype(eltype(matrix_entry))
        tensor_data = UniformScaling(one(eltype(T)))
        -DiagonalMatrixRow(one(eltype(eltype(matrix_entry))))
    else
        zero(eltype(matrix_entry))
    end
    matrix_entry .= (identity_matrix_entry_value,)
end
