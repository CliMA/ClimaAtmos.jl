const compose = Operators.ComposeStencils()
const apply = Operators.ApplyStencil()

struct SchurComplementW{P, F, FT, J1, J2, J3, J4, S, A}
    # model parameters
    params::P

    # whether this struct is used to compute Wfact_t or Wfact
    transform::Bool

    # flags for computing the Jacobian
    flags::F

    # reference to dtγ, which is specified by the ODE solver
    dtγ_ref::FT

    # nonzero blocks of the Jacobian
    ∂dρ∂M::J1
    ∂dE∂M::J2
    ∂dM∂E::J3
    ∂dM∂ρ::J3
    ∂dM∂M::J4

    # cache for the Schur complement linear solve
    S::S
    S_column_array::A

    # whether to test the Jacobian and linear solver
    test::Bool
end

function SchurComplementW(
    space_center,
    space_face,
    thermo_style,
    params,
    transform,
    flags,
    test = false,
)
    FT = Spaces.undertype(space_center)
    dtγ_ref = Ref(zero(FT))

    # TODO: Automate this.
    J_eltype1 = Operators.StencilCoefs{-half, half, NTuple{2, FT}}
    J_eltype2 =
        flags.∂dE∂M_mode == :exact && thermo_style isa TotalEnergy ?
        Operators.StencilCoefs{-(1 + half), 1 + half, NTuple{4, FT}} : J_eltype1
    J_eltype3 = Operators.StencilCoefs{-1, 1, NTuple{3, FT}}
    ∂dρ∂M = Fields.Field(J_eltype1, space_center)
    ∂dE∂M = Fields.Field(J_eltype2, space_center)
    ∂dM∂E = Fields.Field(J_eltype1, space_face)
    ∂dM∂ρ = Fields.Field(J_eltype1, space_face)
    ∂dM∂M = Fields.Field(J_eltype3, space_face)

    # TODO: Automate this.
    S_eltype = Operators.StencilCoefs{-1, 1, NTuple{3, FT}}
    S = Fields.Field(S_eltype, space_face)
    N = Spaces.nlevels(space_face)
    S_column_array = Tridiagonal(
        Array{FT}(undef, N - 1),
        Array{FT}(undef, N),
        Array{FT}(undef, N - 1),
    )

    SchurComplementW{
        typeof(params),
        typeof(flags),
        typeof(dtγ_ref),
        typeof(∂dρ∂M),
        typeof(∂dE∂M),
        typeof(∂dM∂ρ),
        typeof(∂dM∂M),
        typeof(S),
        typeof(S_column_array),
    }(
        params,
        transform,
        flags,
        dtγ_ref,
        ∂dρ∂M,
        ∂dE∂M,
        ∂dM∂E,
        ∂dM∂ρ,
        ∂dM∂M,
        S,
        S_column_array,
        test,
    )
end

# We only use Wfact, but the implicit/IMEX solvers require us to pass
# jac_prototype, then call similar(jac_prototype) to obtain J and Wfact. Here
# is a temporary workaround to avoid unnecessary allocations.
Base.similar(w::SchurComplementW) = w

#=
A = [-I         0          dtγ ∂dρ∂M    ;
     0          -I         dtγ ∂dE∂M    ;
     dtγ ∂dM∂ρ  dtγ ∂dM∂E  dtγ ∂dM∂M - I] =
    [-I   0    A13    ;
     0    -I   A23    ;
     A31  A32  A33 - I]
b = [b1; b2; b3]
x = [x1; x2; x3]
Solving A x = b:
    -x1 + A13 x3 = b1 ==> x1 = -b1 + A13 x3  (1)
    -x2 + A23 x3 = b2 ==> x2 = -b2 + A23 x3  (2)
    A31 x1 + A32 x2 + (A33 - I) x3 = b3  (3)
Substitute (1) and (2) into (3):
    A31 (-b1 + A13 x3) + A32 (-b2 + A23 x3) + (A33 - I) x3 = b3 ==>
    (A31 A13 + A32 A23 + A33 - I) x3 = b3 + A31 b1 + A32 b2 ==>
    x3 = (A31 A13 + A32 A23 + A33 - I) \ (b3 + A31 b1 + A32 b2)
Finally, use (1) and (2) to get x1 and x2.
Note: The matrix S = A31 A13 + A32 A23 + A33 - I is the "Schur complement" of
[-I 0; 0 -I] (the top-left 4 blocks) in A.
=#
function linsolve!(::Type{Val{:init}}, f, u0; kwargs...)
    function _linsolve!(x, A, b, update_matrix = false; kwargs...)
        (; dtγ_ref, ∂dρ∂M, ∂dE∂M, ∂dM∂E, ∂dM∂ρ, ∂dM∂M) = A
        (; S, S_column_array) = A
        dtγ = dtγ_ref[]
        FT = typeof(dtγ)

        xρ = x.base.ρ
        bρ = b.base.ρ
        if :ρθ in propertynames(x.thermodynamics)
            xE = x.thermodynamics.ρθ
            bE = b.thermodynamics.ρθ
        elseif :ρe_tot in propertynames(x.thermodynamics)
            xE = x.thermodynamics.ρe_tot
            bE = b.thermodynamics.ρe_tot
        elseif :ρe_int in propertynames(x.thermodynamics)
            xE = x.thermodynamics.ρe_int
            bE = b.thermodynamics.ρe_int
        end
        if :ρw in propertynames(x.base)
            xM = x.base.ρw.components.data.:1
            bM = b.base.ρw.components.data.:1
        elseif :w in propertynames(x.base)
            xM = x.base.w.components.data.:1
            bM = b.base.w.components.data.:1
        end

        # TODO: Extend LinearAlgebra.I to work with stencil fields.
        FT = eltype(eltype(S))
        I = Ref(Operators.StencilCoefs{-1, 1}((zero(FT), one(FT), zero(FT))))
        if Operators.bandwidths(eltype(∂dE∂M)) != (-half, half)
            str = "The linear solver cannot yet be run with the given ∂dE/∂M \
                block, since it has more than 2 diagonals. So, ∂dE/∂M will \
                be set to 0 for the Schur complement computation. Consider \
                changing the ∂dE∂M_mode or the energy variable."
            @warn str maxlog = 1
            @. S = dtγ^2 * compose(∂dM∂ρ, ∂dρ∂M) + dtγ * ∂dM∂M - I
        else
            @. S =
                dtγ^2 * compose(∂dM∂ρ, ∂dρ∂M) +
                dtγ^2 * compose(∂dM∂E, ∂dE∂M) +
                dtγ * ∂dM∂M - I
        end

        @. xM = bM + dtγ * (apply(∂dM∂ρ, bρ) + apply(∂dM∂E, bE))

        # TODO: Do this with stencil_solve!.
        Ni, Nj, _, _, Nh = size(Spaces.local_geometry_data(axes(xρ)))
        for h in 1:Nh, j in 1:Nj, i in 1:Ni
            xM_column_view = parent(Spaces.column(xM, i, j, h))
            S_column = Spaces.column(S, i, j, h)
            @views S_column_array.dl .= parent(S_column.coefs.:1)[2:end]
            S_column_array.d .= parent(S_column.coefs.:2)
            @views S_column_array.du .= parent(S_column.coefs.:3)[1:(end - 1)]
            ldiv!(lu!(S_column_array), xM_column_view)
        end

        @. xρ = -bρ + dtγ * apply(∂dρ∂M, xM)
        @. xE = -bE + dtγ * apply(∂dE∂M, xM)

        if A.test && Operators.bandwidths(eltype(∂dE∂M)) == (-half, half)
            Ni, Nj, _, Nv, Nh = size(Spaces.local_geometry_data(axes(xρ)))
            ∂Yₜ∂Y = Array{FT}(undef, 3 * Nv + 1, 3 * Nv + 1)
            ΔY = Array{FT}(undef, 3 * Nv + 1)
            ΔΔY = Array{FT}(undef, 3 * Nv + 1)
            for h in 1:Nh, j in 1:Nj, i in 1:Ni
                ∂Yₜ∂Y .= zero(FT)
                ∂Yₜ∂Y[1:Nv, (2 * Nv + 1):(3 * Nv + 1)] .=
                    matrix_column(∂dρ∂M, axes(xM), i, j, h)
                ∂Yₜ∂Y[(Nv + 1):(2 * Nv), (2 * Nv + 1):(3 * Nv + 1)] .=
                    matrix_column(∂dE∂M, axes(xM), i, j, h)
                ∂Yₜ∂Y[(2 * Nv + 1):(3 * Nv + 1), 1:Nv] .=
                    matrix_column(∂dM∂ρ, axes(xρ), i, j, h)
                ∂Yₜ∂Y[(2 * Nv + 1):(3 * Nv + 1), (Nv + 1):(2 * Nv)] .=
                    matrix_column(∂dM∂E, axes(xE), i, j, h)
                ∂Yₜ∂Y[(2 * Nv + 1):(3 * Nv + 1), (2 * Nv + 1):(3 * Nv + 1)] .=
                    matrix_column(∂dM∂M, axes(xM), i, j, h)
                ΔY[1:Nv] .= vector_column(xρ, i, j, h)
                ΔY[(Nv + 1):(2 * Nv)] .= vector_column(xE, i, j, h)
                ΔY[(2 * Nv + 1):(3 * Nv + 1)] .= vector_column(xM, i, j, h)
                ΔΔY[1:Nv] .= vector_column(bρ, i, j, h)
                ΔΔY[(Nv + 1):(2 * Nv)] .= vector_column(bE, i, j, h)
                ΔΔY[(2 * Nv + 1):(3 * Nv + 1)] .= vector_column(bM, i, j, h)
                @assert (-LinearAlgebra.I + dtγ * ∂Yₜ∂Y) * ΔY ≈ ΔΔY
            end
        end

        if :ρuₕ in propertynames(x.base)
            @. x.base.ρuₕ = -b.base.ρuₕ
        elseif :uₕ in propertynames(x.base)
            @. x.base.uₕ = -b.base.uₕ
        end

        if A.transform
            x .*= dtγ
        end
    end
end

transform_wfact(stepper) =
    !(stepper isa Rosenbrock23 || stepper isa Rosenbrock32)
