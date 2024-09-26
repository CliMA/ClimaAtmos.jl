###
### Zero grid-scale or subgrid-scale tendencies
###

zero_tendency!(Yₜ, Y, p, t, _, _) = nothing

function zero_tendency!(Yₜ, Y, p, t, ::NoGridScaleTendency, _)
    # turn off all grid-scale tendencies
    @. Yₜ.c.ρ = 0
    @. Yₜ.c.uₕ = C12(0, 0)
    @. Yₜ.f.u₃ = C3(0)
    @. Yₜ.c.ρe_tot = 0
    for ρχ_name in filter(is_tracer_var, propertynames(Y.c))
        @. Yₜ.c.:($$ρχ_name) = 0
    end
    return nothing
end

function zero_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::NoSubgridScaleTendency,
    ::PrognosticEDMFX,
)
    # turn off all subgrid-scale tendencies
    n = n_prognostic_mass_flux_subdomains(p.atmos.turbconv_model)
    for j in 1:n
        @. Yₜ.c.sgsʲs.:($$j).ρa = 0
        @. Yₜ.f.sgsʲs.:($$j).u₃ = C3(0)
        @. Yₜ.c.sgsʲs.:($$j).mse = 0
        @. Yₜ.c.sgsʲs.:($$j).q_tot = 0
    end
    return nothing
end

import ClimaCore.MatrixFields: MatrixFields, @name

zero_jacobian!(_, _, _) = nothing

function zero_jacobian!(matrix, ::NoGridScaleTendency, _)
    sgs_names = (@name(c.sgs⁰), @name(c.sgsʲs), @name(f.sgsʲs))
    for ((row_name, col_name), matrix_block) in matrix
        any(name -> MatrixFields.is_child_name(row_name, name), sgs_names) &&
            continue
        matrix_block isa LinearAlgebra.UniformScaling && continue
        matrix_block .=
            row_name == col_name ?
            (DiagonalMatrixRow(-one(eltype(eltype(matrix_block)))),) :
            (zero(eltype(matrix_block)),)
    end
end

function zero_jacobian!(matrix, ::NoSubgridScaleTendency, ::PrognosticEDMFX)
    sgs_names = (@name(c.sgsʲs), @name(f.sgsʲs))
    for ((row_name, col_name), matrix_block) in matrix
        !any(name -> MatrixFields.is_child_name(row_name, name), sgs_names) &&
            continue
        matrix_block isa LinearAlgebra.UniformScaling && continue
        matrix_block .=
            row_name == col_name ?
            (DiagonalMatrixRow(-one(eltype(eltype(matrix_block)))),) :
            (zero(eltype(matrix_block)),)
    end
end

# TODO: Make the Jacobian adjustment consistent with the tendency adjustment.

# TODO: Move this to ClimaCore.
Base.one(::Type{T}) where {T′, N, A, S, T <: Geometry.AxisTensor{T′, N, A, S}} =
    T(axes(T), S(one(T′)))
