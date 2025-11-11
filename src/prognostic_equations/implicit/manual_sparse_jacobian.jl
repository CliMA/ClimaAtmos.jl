import LinearAlgebra: I, Adjoint

using ClimaCore.MatrixFields
import ClimaCore.MatrixFields: @name

abstract type DerivativeFlag end
struct UseDerivative <: DerivativeFlag end
struct IgnoreDerivative <: DerivativeFlag end

DerivativeFlag(value) = value ? UseDerivative() : IgnoreDerivative()
DerivativeFlag(mode::AbstractTimesteppingMode) =
    DerivativeFlag(mode == Implicit())

use_derivative(::UseDerivative) = true
use_derivative(::IgnoreDerivative) = false

"""
    ManualSparseJacobian(
        topography_flag,
        diffusion_flag,
        sgs_advection_flag,
        sgs_entr_detr_flag,
        sgs_mass_flux_flag,
        sgs_nh_pressure_flag,
        sgs_vertdiff_flag,
        approximate_solve_iters,
    )

A [`JacobianAlgorithm`](@ref) that approximates the Jacobian using analytically
derived tendency derivatives and inverts it using a specialized nested linear
solver. Certain groups of derivatives can be toggled on or off by setting their
`DerivativeFlag`s to either `UseDerivative` or `IgnoreDerivative`.

# Arguments

- `topography_flag::DerivativeFlag`: whether the derivative of vertical
  contravariant velocity with respect to horizontal covariant velocity should be
  computed
- `diffusion_flag::DerivativeFlag`: whether the derivatives of the grid-scale
  diffusion tendency should be computed
- `sgs_advection_flag::DerivativeFlag`: whether the derivatives of the
  subgrid-scale advection tendency should be computed
- `sgs_entr_detr_flag::DerivativeFlag`: whether the derivatives of the
  subgrid-scale entrainment and detrainment tendencies should be computed
- `sgs_mass_flux_flag::DerivativeFlag`: whether the derivatives of the
  subgrid-scale mass flux tendency should be computed
- `sgs_nh_pressure_flag::DerivativeFlag`: whether the derivatives of the
  subgrid-scale non-hydrostatic pressure drag tendency should be computed
- `sgs_vertdiff_flag::DerivativeFlag`: whether the derivatives of the
  subgrid-scale vertical diffusion tendency should be computed
- `approximate_solve_iters::Int`: number of iterations to take for the
  approximate linear solve required when the `diffusion_flag` is `UseDerivative`
"""
struct ManualSparseJacobian{F1, F2, F3, F4, F5, F6, F7} <: SparseJacobian
    topography_flag::F1
    diffusion_flag::F2
    sgs_advection_flag::F3
    sgs_entr_detr_flag::F4
    sgs_mass_flux_flag::F5
    sgs_nh_pressure_flag::F6
    sgs_vertdiff_flag::F7
    approximate_solve_iters::Int
end

function jacobian_cache(alg::ManualSparseJacobian, Y, atmos)
    (;
        topography_flag,
        diffusion_flag,
        sgs_advection_flag,
        sgs_mass_flux_flag,
        approximate_solve_iters,
    ) = alg
    FT = Spaces.undertype(axes(Y.c))
    CTh = CTh_vector_type(axes(Y.c))

    DiagonalRow = DiagonalMatrixRow{FT}
    TridiagonalRow = TridiagonalMatrixRow{FT}
    BidiagonalRow_C3 = BidiagonalMatrixRow{C3{FT}}
    TridiagonalRow_ACTh = TridiagonalMatrixRow{Adjoint{FT, CTh{FT}}}
    BidiagonalRow_ACT3 = BidiagonalMatrixRow{Adjoint{FT, CT3{FT}}}
    BidiagonalRow_C3xACTh =
        BidiagonalMatrixRow{typeof(zero(C3{FT}) * zero(CTh{FT})')}
    DiagonalRow_C3xACT3 =
        DiagonalMatrixRow{typeof(zero(C3{FT}) * zero(CT3{FT})')}
    TridiagonalRow_C3xACT3 =
        TridiagonalMatrixRow{typeof(zero(C3{FT}) * zero(CT3{FT})')}

    is_in_Y(name) = MatrixFields.has_field(Y, name)

    ρq_tot_if_available = is_in_Y(@name(c.ρq_tot)) ? (@name(c.ρq_tot),) : ()
    ρatke_if_available =
        is_in_Y(@name(c.sgs⁰.ρatke)) ? (@name(c.sgs⁰.ρatke),) : ()
    sfc_if_available = is_in_Y(@name(sfc)) ? (@name(sfc),) : ()

    condensate_names = (
        @name(c.ρq_liq),
        @name(c.ρq_ice),
        @name(c.ρq_rai),
        @name(c.ρq_sno),
        @name(c.ρn_liq),
        @name(c.ρn_rai),
        # P3 frozen
        @name(c.ρn_ice), @name(c.ρq_rim), @name(c.ρb_rim),
    )
    available_condensate_names =
        MatrixFields.unrolled_filter(is_in_Y, condensate_names)
    available_tracer_names =
        (ρq_tot_if_available..., available_condensate_names...)

    # we define the list of condensate masses separately because ρa and q_tot
    # depend on the masses via sedimentation
    sgs_condensate_mass_names = (
        @name(c.sgsʲs.:(1).q_liq),
        @name(c.sgsʲs.:(1).q_ice),
        @name(c.sgsʲs.:(1).q_rai),
        @name(c.sgsʲs.:(1).q_sno),
    )
    available_sgs_condensate_mass_names =
        MatrixFields.unrolled_filter(is_in_Y, sgs_condensate_mass_names)

    sgs_condensate_names =
        (sgs_condensate_mass_names..., @name(c.sgsʲs.:(1).n_liq), @name(c.sgsʲs.:(1).n_rai))
    available_sgs_condensate_names =
        MatrixFields.unrolled_filter(is_in_Y, sgs_condensate_names)

    sgs_scalar_names =
        (
            sgs_condensate_names...,
            @name(c.sgsʲs.:(1).q_tot),
            @name(c.sgsʲs.:(1).mse),
            @name(c.sgsʲs.:(1).ρa)
        )
    available_sgs_scalar_names =
        MatrixFields.unrolled_filter(is_in_Y, sgs_scalar_names)

    sgs_u³_if_available =
        is_in_Y(@name(f.sgsʲs.:(1).u₃)) ? (@name(f.sgsʲs.:(1).u₃),) : ()

    # Note: We have to use FT(-1) * I instead of -I because inv(-1) == -1.0,
    # which means that multiplying inv(-1) by a Float32 will yield a Float64.
    identity_blocks = MatrixFields.unrolled_map(
        name -> (name, name) => FT(-1) * I,
        (@name(c.ρ), sfc_if_available...),
    )

    active_scalar_names = (@name(c.ρ), @name(c.ρe_tot), ρq_tot_if_available...)
    advection_blocks = (
        (
            use_derivative(topography_flag) ?
            MatrixFields.unrolled_map(
                name ->
                    (name, @name(c.uₕ)) =>
                        similar(Y.c, TridiagonalRow_ACTh),
                active_scalar_names,
            ) : ()
        )...,
        MatrixFields.unrolled_map(
            name -> (name, @name(f.u₃)) => similar(Y.c, BidiagonalRow_ACT3),
            active_scalar_names,
        )...,
        MatrixFields.unrolled_map(
            name -> (@name(f.u₃), name) => similar(Y.f, BidiagonalRow_C3),
            active_scalar_names,
        )...,
        (@name(f.u₃), @name(c.uₕ)) => similar(Y.f, BidiagonalRow_C3xACTh),
        (@name(f.u₃), @name(f.u₃)) => similar(Y.f, TridiagonalRow_C3xACT3),
    )

    diffused_scalar_names = (@name(c.ρe_tot), available_tracer_names...)
    diffusion_blocks = if use_derivative(diffusion_flag)
        (
            MatrixFields.unrolled_map(
                name -> (name, @name(c.ρ)) => similar(Y.c, TridiagonalRow),
                (diffused_scalar_names..., ρatke_if_available...),
            )...,
            MatrixFields.unrolled_map(
                name -> (name, name) => similar(Y.c, TridiagonalRow),
                (diffused_scalar_names..., ρatke_if_available...),
            )...,
            (
                is_in_Y(@name(c.ρq_tot)) ?
                (
                    (@name(c.ρe_tot), @name(c.ρq_tot)) =>
                        similar(Y.c, TridiagonalRow),
                ) : ()
            )...,
            (@name(c.uₕ), @name(c.uₕ)) =>
                !isnothing(atmos.turbconv_model) ||
                    !disable_momentum_vertical_diffusion(
                        atmos.vertical_diffusion,
                    ) ? similar(Y.c, TridiagonalRow) : FT(-1) * I,
        )
    elseif atmos.moisture_model isa DryModel
        MatrixFields.unrolled_map(
            name -> (name, name) => FT(-1) * I,
            (diffused_scalar_names..., ρatke_if_available..., @name(c.uₕ)),
        )
    else
        (
            MatrixFields.unrolled_map(
                name -> (name, name) => similar(Y.c, TridiagonalRow),
                diffused_scalar_names,
            )...,
            (@name(c.ρe_tot), @name(c.ρq_tot)) =>
                similar(Y.c, TridiagonalRow),
            MatrixFields.unrolled_map(
                name -> (name, name) => FT(-1) * I,
                (ρatke_if_available..., @name(c.uₕ)),
            )...,
        )
    end

    sgs_advection_blocks = if atmos.turbconv_model isa PrognosticEDMFX
        if use_derivative(sgs_advection_flag)
            (
                MatrixFields.unrolled_map(
                    name -> (name, name) => similar(Y.c, TridiagonalRow),
                    available_sgs_scalar_names,
                )...,
                MatrixFields.unrolled_map(
                    name ->
                        (@name(c.sgsʲs.:(1).q_tot), name) =>
                            similar(Y.c, TridiagonalRow),
                    available_sgs_condensate_mass_names,
                )...,
                MatrixFields.unrolled_map(
                    name ->
                        (@name(c.sgsʲs.:(1).ρa), name) => similar(Y.c, TridiagonalRow),
                    available_sgs_condensate_mass_names,
                )...,
                MatrixFields.unrolled_map(
                    name ->
                        (name, @name(f.sgsʲs.:(1).u₃)) =>
                            similar(Y.c, BidiagonalRow_ACT3),
                    available_sgs_scalar_names,
                )...,
                (@name(c.sgsʲs.:(1).mse), @name(c.sgsʲs.:(1).q_tot)) =>
                    similar(Y.c, DiagonalRow),
                (@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).q_tot)) =>
                    similar(Y.c, TridiagonalRow),
                (@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).mse)) =>
                    similar(Y.c, TridiagonalRow),
                (@name(f.sgsʲs.:(1).u₃), @name(c.sgsʲs.:(1).q_tot)) =>
                    similar(Y.f, BidiagonalRow_C3),
                (@name(f.sgsʲs.:(1).u₃), @name(c.sgsʲs.:(1).mse)) =>
                    similar(Y.f, BidiagonalRow_C3),
                (@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)) =>
                    similar(Y.f, TridiagonalRow_C3xACT3),
            )
        else
            (
                MatrixFields.unrolled_map(
                    name -> (name, name) => FT(-1) * I,
                    available_sgs_scalar_names,
                )...,
                (@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)) =>
                    !isnothing(atmos.rayleigh_sponge) ?
                    similar(Y.f, DiagonalRow_C3xACT3) : FT(-1) * I,
            )
        end
    else
        ()
    end

    sgs_massflux_blocks = if atmos.turbconv_model isa PrognosticEDMFX
        if use_derivative(sgs_mass_flux_flag)
            (
                MatrixFields.unrolled_map(
                    name ->
                        (name, get_χʲ_name_from_ρχ_name(name)) =>
                            similar(Y.c, TridiagonalRow),
                    available_tracer_names,
                )...,
                MatrixFields.unrolled_map(
                    name ->
                        (name, @name(c.sgsʲs.:(1).ρa)) =>
                            similar(Y.c, TridiagonalRow),
                    available_tracer_names,
                )...,
                MatrixFields.unrolled_map(
                    name ->
                        (name, @name(f.sgsʲs.:(1).u₃)) =>
                            similar(Y.c, BidiagonalRow_ACT3),
                    available_tracer_names,
                )...,
                (@name(c.ρe_tot), @name(c.sgsʲs.:(1).mse)) =>
                    similar(Y.c, TridiagonalRow),
                (@name(c.ρe_tot), @name(f.sgsʲs.:(1).u₃)) =>
                    similar(Y.c, BidiagonalRow_ACT3),
                (@name(c.ρe_tot), @name(c.sgsʲs.:(1).ρa)) =>
                    similar(Y.c, TridiagonalRow),
            )
        else
            ()
        end
    else
        ()
    end

    matrix = MatrixFields.FieldMatrix(
        identity_blocks...,
        sgs_advection_blocks...,
        advection_blocks...,
        diffusion_blocks...,
        sgs_massflux_blocks...,
    )

    mass_and_surface_names = (@name(c.ρ), sfc_if_available...)
    available_scalar_names = (
        mass_and_surface_names...,
        available_tracer_names...,
        @name(c.ρe_tot),
        ρatke_if_available...,
        available_sgs_scalar_names...,
    )

    velocity_alg = MatrixFields.BlockLowerTriangularSolve(
        @name(c.uₕ),
        sgs_u³_if_available...,
    )
    full_alg =
        if use_derivative(diffusion_flag) ||
           use_derivative(sgs_advection_flag) ||
           !(atmos.moisture_model isa DryModel)
            gs_scalar_subalg = if !(atmos.moisture_model isa DryModel)
                MatrixFields.BlockLowerTriangularSolve(@name(c.ρq_tot))
            else
                MatrixFields.BlockDiagonalSolve()
            end
            scalar_subalg =
                if atmos.turbconv_model isa PrognosticEDMFX &&
                   use_derivative(sgs_advection_flag)
                    MatrixFields.BlockLowerTriangularSolve(
                        available_sgs_condensate_names...;
                        alg₂ = MatrixFields.BlockLowerTriangularSolve(
                            @name(c.sgsʲs.:(1).q_tot);
                            alg₂ = MatrixFields.BlockLowerTriangularSolve(
                                @name(c.sgsʲs.:(1).mse);
                                alg₂ = MatrixFields.BlockLowerTriangularSolve(
                                    @name(c.sgsʲs.:(1).ρa);
                                    alg₂ = gs_scalar_subalg,
                                ),
                            ),
                        ),
                    )
                else
                    gs_scalar_subalg
                end
            scalar_alg = MatrixFields.BlockLowerTriangularSolve(
                mass_and_surface_names...;
                alg₂ = scalar_subalg,
            )
            MatrixFields.ApproximateBlockArrowheadIterativeSolve(
                available_scalar_names...;
                alg₁ = scalar_alg,
                alg₂ = velocity_alg,
                P_alg₁ = MatrixFields.MainDiagonalPreconditioner(),
                n_iters = approximate_solve_iters,
            )
        else
            MatrixFields.BlockArrowheadSolve(
                available_scalar_names...;
                alg₂ = velocity_alg,
            )
        end

    return (; matrix = MatrixFields.FieldMatrixWithSolver(matrix, Y, full_alg))
end

function update_jacobian!(alg::ManualSparseJacobian, cache, Y, p, dtγ, t)
    (;
        topography_flag,
        diffusion_flag,
        sgs_advection_flag,
        sgs_entr_detr_flag,
        sgs_nh_pressure_flag,
        sgs_mass_flux_flag,
        sgs_vertdiff_flag,
    ) = alg
    (; matrix) = cache
    (; params) = p
    (; ᶜΦ, ᶠgradᵥ_ᶜΦ) = p.core
    (; ᶠu³, ᶜK, ᶜts, ᶜp) = p.precomputed
    (;
        ∂ᶜK_∂ᶜuₕ,
        ∂ᶜK_∂ᶠu₃,
        ᶠp_grad_matrix,
        ᶜadvection_matrix,
        ᶜdiffusion_h_matrix,
        ᶜdiffusion_u_matrix,
        ᶜtridiagonal_matrix_scalar,
        ᶠbidiagonal_matrix_ct3,
        ᶠbidiagonal_matrix_ct3_2,
        ᶠtridiagonal_matrix_c3,
    ) = p.scratch
    rs = p.atmos.rayleigh_sponge

    FT = Spaces.undertype(axes(Y.c))
    CTh = CTh_vector_type(axes(Y.c))
    one_C3xACT3 = C3(FT(1)) * CT3(FT(1))'

    cv_d = FT(CAP.cv_d(params))
    Δcv_v = FT(CAP.cv_v(params)) - cv_d
    T_0 = FT(CAP.T_0(params))
    R_d = FT(CAP.R_d(params))
    ΔR_v = FT(CAP.R_v(params)) - R_d
    cp_d = FT(CAP.cp_d(params))
    Δcp_v = FT(CAP.cp_v(params)) - cp_d
    # This term appears a few times in the Jacobian, and is technically
    # minus ∂e_int_∂q_tot
    thermo_params = CAP.thermodynamics_params(params)
    ∂e_int_∂q_tot = T_0 * (Δcv_v - R_d) - FT(CAP.e_int_v0(params))
    ᶜh_tot = @. lazy(
        TD.total_specific_enthalpy(
            thermo_params,
            ᶜts,
            specific(Y.c.ρe_tot, Y.c.ρ),
        ),
    )

    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠu₃ = Y.f.u₃
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    ᶜgⁱʲ = Fields.local_geometry_field(Y.c).gⁱʲ
    ᶠgⁱʲ = Fields.local_geometry_field(Y.f).gⁱʲ
    ᶠz = Fields.coordinate_field(Y.f).z
    zmax = z_max(axes(Y.f))

    ᶜkappa_m = p.scratch.ᶜtemp_scalar
    @. ᶜkappa_m =
        TD.gas_constant_air(thermo_params, ᶜts) / TD.cv_m(thermo_params, ᶜts)

    ᶜ∂kappa_m∂q_tot = p.scratch.ᶜtemp_scalar_2
    # Using abs2 because ^2 results in allocation
    @. ᶜ∂kappa_m∂q_tot =
        (
            ΔR_v * TD.cv_m(thermo_params, ᶜts) -
            Δcv_v * TD.gas_constant_air(thermo_params, ᶜts)
        ) / abs2(TD.cv_m(thermo_params, ᶜts))

    if use_derivative(topography_flag)
        @. ∂ᶜK_∂ᶜuₕ = DiagonalMatrixRow(
            adjoint(CTh(ᶜuₕ)) + adjoint(ᶜinterp(ᶠu₃)) * g³ʰ(ᶜgⁱʲ),
        )
    else
        @. ∂ᶜK_∂ᶜuₕ = DiagonalMatrixRow(adjoint(CTh(ᶜuₕ)))
    end
    @. ∂ᶜK_∂ᶠu₃ =
        ᶜinterp_matrix() ⋅ DiagonalMatrixRow(adjoint(CT3(ᶠu₃))) +
        DiagonalMatrixRow(adjoint(CT3(ᶜuₕ))) ⋅ ᶜinterp_matrix()

    @. ᶠp_grad_matrix = DiagonalMatrixRow(-1 / ᶠinterp(ᶜρ)) ⋅ ᶠgradᵥ_matrix()

    @. ᶜadvection_matrix =
        -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ)

    if use_derivative(topography_flag)
        ∂ᶜρ_err_∂ᶜuₕ = matrix[@name(c.ρ), @name(c.uₕ)]
        @. ∂ᶜρ_err_∂ᶜuₕ =
            dtγ * ᶜadvection_matrix ⋅ ᶠwinterp_matrix(ᶜJ * ᶜρ) ⋅
            DiagonalMatrixRow(g³ʰ(ᶜgⁱʲ))
    end
    ∂ᶜρ_err_∂ᶠu₃ = matrix[@name(c.ρ), @name(f.u₃)]
    @. ∂ᶜρ_err_∂ᶠu₃ = dtγ * ᶜadvection_matrix ⋅ DiagonalMatrixRow(g³³(ᶠgⁱʲ))

    tracer_info = (@name(c.ρe_tot), @name(c.ρq_tot))

    MatrixFields.unrolled_foreach(tracer_info) do ρχ_name
        MatrixFields.has_field(Y, ρχ_name) || return
        ᶜχ = if ρχ_name === @name(c.ρe_tot)
            @. lazy(
                TD.total_specific_enthalpy(
                    thermo_params,
                    ᶜts,
                    specific(Y.c.ρe_tot, Y.c.ρ),
                ),
            )
        else
            @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
        end

        if use_derivative(topography_flag)
            ∂ᶜρχ_err_∂ᶜuₕ = matrix[ρχ_name, @name(c.uₕ)]
            @. ∂ᶜρχ_err_∂ᶜuₕ =
                dtγ * ᶜadvection_matrix ⋅ DiagonalMatrixRow(ᶠinterp(ᶜχ)) ⋅
                ᶠwinterp_matrix(ᶜJ * ᶜρ) ⋅ DiagonalMatrixRow(g³ʰ(ᶜgⁱʲ))
        end

        ∂ᶜρχ_err_∂ᶠu₃ = matrix[ρχ_name, @name(f.u₃)]
        @. ∂ᶜρχ_err_∂ᶠu₃ =
            dtγ * ᶜadvection_matrix ⋅ DiagonalMatrixRow(ᶠinterp(ᶜχ) * g³³(ᶠgⁱʲ))
    end

    ∂ᶠu₃_err_∂ᶜρ = matrix[@name(f.u₃), @name(c.ρ)]
    ∂ᶠu₃_err_∂ᶜρe_tot = matrix[@name(f.u₃), @name(c.ρe_tot)]

    ᶜθ_v = @. lazy(theta_v(thermo_params, ᶜts))
    ᶜΠ = @. lazy(dry_exner_function(thermo_params, ᶜts))
    # In implicit tendency, we use the new pressure-gradient formulation (PGF) and gravitational acceleration: 
    #              grad(p) / ρ + grad(Φ)  =  cp_d * θ_v * grad(Π) + grad(Φ).
    # Here below, we use the old formulation of (grad(Φ) + grad(p) / ρ).
    # This is because the new formulation would require computing the derivative of θ_v.
    # The only exception is:
    # We are rewriting grad(p) / ρ from the expansion of ∂ᶠu₃_err_∂ᶜρ with the new PGF.
    @. ∂ᶠu₃_err_∂ᶜρ =
        dtγ * (
            ᶠp_grad_matrix ⋅
            DiagonalMatrixRow(ᶜkappa_m * (T_0 * cp_d - ᶜK - ᶜΦ)) +
            DiagonalMatrixRow(cp_d * ᶠinterp(ᶜθ_v) * ᶠgradᵥ(ᶜΠ) / ᶠinterp(ᶜρ)) ⋅
            ᶠinterp_matrix()
        )
    @. ∂ᶠu₃_err_∂ᶜρe_tot = dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(ᶜkappa_m)
    ᶜe_tot = @. lazy(specific(Y.c.ρe_tot, Y.c.ρ))
    if MatrixFields.has_field(Y, @name(c.ρq_tot))
        ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
        ∂ᶠu₃_err_∂ᶜρq_tot = matrix[@name(f.u₃), @name(c.ρq_tot)]
        @. ∂ᶠu₃_err_∂ᶜρq_tot =
            dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow((
                ᶜkappa_m * ∂e_int_∂q_tot +
                ᶜ∂kappa_m∂q_tot *
                (cp_d * T_0 + ᶜe_tot - ᶜK - ᶜΦ + ∂e_int_∂q_tot * ᶜq_tot)
            ))
    end

    ∂ᶠu₃_err_∂ᶜuₕ = matrix[@name(f.u₃), @name(c.uₕ)]
    ∂ᶠu₃_err_∂ᶠu₃ = matrix[@name(f.u₃), @name(f.u₃)]
    I_u₃ = DiagonalMatrixRow(one_C3xACT3)
    @. ∂ᶠu₃_err_∂ᶜuₕ =
        dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(-(ᶜkappa_m) * ᶜρ) ⋅ ∂ᶜK_∂ᶜuₕ
    if rs isa RayleighSponge
        @. ∂ᶠu₃_err_∂ᶠu₃ =
            dtγ * (
                ᶠp_grad_matrix ⋅ DiagonalMatrixRow(-(ᶜkappa_m) * ᶜρ) ⋅
                ∂ᶜK_∂ᶠu₃ +
                DiagonalMatrixRow(-β_rayleigh_w(rs, ᶠz, zmax) * (one_C3xACT3,))
            ) - (I_u₃,)
    else
        @. ∂ᶠu₃_err_∂ᶠu₃ =
            dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(-(ᶜkappa_m) * ᶜρ) ⋅
            ∂ᶜK_∂ᶠu₃ - (I_u₃,)
    end

    α_vert_diff_tracer = CAP.α_vert_diff_tracer(params)
    tracer_info = (
        (@name(c.ρq_liq), @name(ᶜwₗ), FT(1)),
        (@name(c.ρq_ice), @name(ᶜwᵢ), FT(1)),
        (@name(c.ρq_rai), @name(ᶜwᵣ), α_vert_diff_tracer),
        (@name(c.ρq_sno), @name(ᶜwₛ), α_vert_diff_tracer),
        (@name(c.ρn_liq), @name(ᶜwₙₗ), FT(1)),
        (@name(c.ρn_rai), @name(ᶜwₙᵣ), α_vert_diff_tracer),
        (@name(c.ρn_ice), @name(ᶜwnᵢ), FT(1)),
        (@name(c.ρq_rim), @name(ᶜwᵢ), FT(1)),
        (@name(c.ρb_rim), @name(ᶜwᵢ), FT(1)),
    )
    if !(p.atmos.moisture_model isa DryModel) || use_derivative(diffusion_flag)
        ∂ᶜρe_tot_err_∂ᶜρe_tot = matrix[@name(c.ρe_tot), @name(c.ρe_tot)]
        @. ∂ᶜρe_tot_err_∂ᶜρe_tot = zero(typeof(∂ᶜρe_tot_err_∂ᶜρe_tot)) - (I,)
    end

    if !(p.atmos.moisture_model isa DryModel)
        #TODO: tetsing explicit vs implicit
        #@. ∂ᶜρe_tot_err_∂ᶜρe_tot +=
        #    dtγ * -(ᶜprecipdivᵥ_matrix()) ⋅
        #    DiagonalMatrixRow(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ) ⋅ ᶠright_bias_matrix() ⋅
        #    DiagonalMatrixRow(
        #        -(1 + ᶜkappa_m) / ᶜρ * ifelse(
        #            ᶜh_tot == 0,
        #            (Geometry.WVector(FT(0)),),
        #            p.precomputed.ᶜwₕhₜ / ᶜh_tot,
        #        ),
        #    )

        ∂ᶜρe_tot_err_∂ᶜρq_tot = matrix[@name(c.ρe_tot), @name(c.ρq_tot)]
        @. ∂ᶜρe_tot_err_∂ᶜρq_tot = zero(typeof(∂ᶜρe_tot_err_∂ᶜρq_tot))
        #TODO: tetsing explicit vs implicit
        #@. ∂ᶜρe_tot_err_∂ᶜρq_tot =
        #    dtγ * -(ᶜprecipdivᵥ_matrix()) ⋅
        #    DiagonalMatrixRow(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ) ⋅ ᶠright_bias_matrix() ⋅
        #    DiagonalMatrixRow(
        #        -(ᶜkappa_m) * ∂e_int_∂q_tot / ᶜρ * ifelse(
        #            ᶜh_tot == 0,
        #            (Geometry.WVector(FT(0)),),
        #            p.precomputed.ᶜwₕhₜ / ᶜh_tot,
        #        ),
        #    )

        ∂ᶜρq_tot_err_∂ᶜρq_tot = matrix[@name(c.ρq_tot), @name(c.ρq_tot)]
        @. ∂ᶜρq_tot_err_∂ᶜρq_tot = zero(typeof(∂ᶜρq_tot_err_∂ᶜρq_tot)) - (I,)
        #TODO: testing explicit vs implicit
        #@. ∂ᶜρq_tot_err_∂ᶜρq_tot =
        #    dtγ * -(ᶜprecipdivᵥ_matrix()) ⋅
        #    DiagonalMatrixRow(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ) ⋅ ᶠright_bias_matrix() ⋅
        #    DiagonalMatrixRow(
        #        -1 / ᶜρ * ifelse(
        #           ᶜq_tot == 0,
        #            (Geometry.WVector(FT(0)),),
        #            p.precomputed.ᶜwₜqₜ / ᶜq_tot,
        #        ),
        #    ) - (I,)

        MatrixFields.unrolled_foreach(tracer_info) do (ρχₚ_name, wₚ_name, _)
            MatrixFields.has_field(Y, ρχₚ_name) || return
            ∂ᶜρχₚ_err_∂ᶜρχₚ = matrix[ρχₚ_name, ρχₚ_name]
            ᶜwₚ = MatrixFields.get_field(p.precomputed, wₚ_name)
            @. ∂ᶜρχₚ_err_∂ᶜρχₚ =
                dtγ * -(ᶜprecipdivᵥ_matrix()) ⋅
                DiagonalMatrixRow(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ) ⋅
                ᶠright_bias_matrix() ⋅
                DiagonalMatrixRow(-Geometry.WVector(ᶜwₚ) / ᶜρ) - (I,)
        end

    end

    if use_derivative(diffusion_flag)
        (; turbconv_model) = p.atmos
        turbconv_params = CAP.turbconv_params(params)
        FT = eltype(params)
        (; vertical_diffusion, smagorinsky_lilly) = p.atmos
        (; ᶜp) = p.precomputed
        ᶜK_u = p.scratch.ᶜtemp_scalar_4
        ᶜK_h = p.scratch.ᶜtemp_scalar_5
        if vertical_diffusion isa DecayWithHeightDiffusion
            ᶜK_h .= ᶜcompute_eddy_diffusivity_coefficient(Y.c.ρ, vertical_diffusion)
            ᶜK_u = ᶜK_h
        elseif vertical_diffusion isa VerticalDiffusion
            ᶜK_h .= ᶜcompute_eddy_diffusivity_coefficient(Y.c.uₕ, ᶜp, vertical_diffusion)
            ᶜK_u = ᶜK_h
        elseif is_smagorinsky_vertical(smagorinsky_lilly)
            set_smagorinsky_lilly_precomputed_quantities!(Y, p, smagorinsky_lilly)
            ᶜK_u = p.precomputed.ᶜνₜ_v
            ᶜK_h = p.precomputed.ᶜD_v
        elseif turbconv_model isa AbstractEDMF
            (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = p.precomputed
            ᶜtke⁰ = @. lazy(specific(Y.c.sgs⁰.ρatke, Y.c.ρ))
            ᶜmixing_length_field = p.scratch.ᶜtemp_scalar_3
            ᶜmixing_length_field .= ᶜmixing_length(Y, p)
            @. ᶜK_u = eddy_viscosity(turbconv_params, ᶜtke⁰, ᶜmixing_length_field)

            ᶜprandtl_nvec = @. lazy(
                turbulent_prandtl_number(params, ᶜlinear_buoygrad, ᶜstrain_rate_norm),
            )
            @. ᶜK_h = eddy_diffusivity(ᶜK_u, ᶜprandtl_nvec)
        end

        @. ᶜadvection_matrix =
            ᶜadvdivᵥ_matrix() ⋅ DiagonalMatrixRow(ᶠinterp(ᶜρ) * ᶠinterp(ᶜK_h))
        @. ᶜdiffusion_h_matrix = ᶜadvection_matrix ⋅ ᶠgradᵥ_matrix()
        if (
            MatrixFields.has_field(Y, @name(c.sgs⁰.ρatke)) ||
            !isnothing(p.atmos.turbconv_model) ||
            !disable_momentum_vertical_diffusion(p.atmos.vertical_diffusion)
        )
            @. ᶜadvection_matrix =
                ᶜadvdivᵥ_matrix() ⋅ DiagonalMatrixRow(ᶠinterp(ᶜρ) * ᶠinterp(ᶜK_u))
            @. ᶜdiffusion_u_matrix = ᶜadvection_matrix ⋅ ᶠgradᵥ_matrix()
        end

        ∂ᶜρe_tot_err_∂ᶜρ = matrix[@name(c.ρe_tot), @name(c.ρ)]
        @. ∂ᶜρe_tot_err_∂ᶜρ = zero(typeof(∂ᶜρe_tot_err_∂ᶜρ))
        @. ∂ᶜρe_tot_err_∂ᶜρe_tot +=
            dtγ * ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow((1 + ᶜkappa_m) / ᶜρ)

        if MatrixFields.has_field(Y, @name(c.ρq_tot))
            ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
            ∂ᶜρe_tot_err_∂ᶜρq_tot = matrix[@name(c.ρe_tot), @name(c.ρq_tot)]
            ∂ᶜρq_tot_err_∂ᶜρ = matrix[@name(c.ρq_tot), @name(c.ρ)]
            @. ∂ᶜρe_tot_err_∂ᶜρq_tot +=
                dtγ * ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow((
                    ᶜkappa_m * ∂e_int_∂q_tot / ᶜρ +
                    ᶜ∂kappa_m∂q_tot *
                    (cp_d * T_0 + ᶜe_tot - ᶜK - ᶜΦ + ∂e_int_∂q_tot * ᶜq_tot)
                ))
            @. ∂ᶜρq_tot_err_∂ᶜρ = zero(typeof(∂ᶜρq_tot_err_∂ᶜρ))
            @. ∂ᶜρq_tot_err_∂ᶜρq_tot +=
                dtγ * ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow(1 / ᶜρ)
        end

        MatrixFields.unrolled_foreach(tracer_info) do (ρχ_name, _, α)
            MatrixFields.has_field(Y, ρχ_name) || return
            ∂ᶜρχ_err_∂ᶜρ = matrix[ρχ_name, @name(c.ρ)]
            ∂ᶜρχ_err_∂ᶜρχ = matrix[ρχ_name, ρχ_name]
            @. ∂ᶜρχ_err_∂ᶜρ = zero(typeof(∂ᶜρχ_err_∂ᶜρ))
            @. ∂ᶜρχ_err_∂ᶜρχ +=
                dtγ * α * ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow(1 / ᶜρ)
        end

        if MatrixFields.has_field(Y, @name(c.sgs⁰.ρatke))
            turbconv_params = CAP.turbconv_params(params)
            c_d = CAP.tke_diss_coeff(turbconv_params)
            (; dt) = p
            turbconv_model = p.atmos.turbconv_model
            ᶜtke⁰ = @. lazy(specific(Y.c.sgs⁰.ρatke, Y.c.ρ))
            ᶜρatke⁰ = Y.c.sgs⁰.ρatke

            # scratch to prevent GPU Kernel parameter memory error
            ᶜmixing_length_field = p.scratch.ᶜtemp_scalar_3
            ᶜmixing_length_field .= ᶜmixing_length(Y, p)

            @inline tke_dissipation_rate_tendency(tke⁰, mixing_length) =
                tke⁰ >= 0 ? c_d * sqrt(tke⁰) / mixing_length : 1 / typeof(tke⁰)(dt)
            @inline ∂tke_dissipation_rate_tendency_∂tke⁰(tke⁰, mixing_length) =
                tke⁰ > 0 ? c_d / (2 * mixing_length * sqrt(tke⁰)) :
                typeof(tke⁰)(0)

            ᶜdissipation_matrix_diagonal = p.scratch.ᶜtemp_scalar
            @. ᶜdissipation_matrix_diagonal =
                ᶜρatke⁰ * ∂tke_dissipation_rate_tendency_∂tke⁰(
                    ᶜtke⁰,
                    ᶜmixing_length_field,
                )

            ∂ᶜρatke⁰_err_∂ᶜρ = matrix[@name(c.sgs⁰.ρatke), @name(c.ρ)]
            ∂ᶜρatke⁰_err_∂ᶜρatke⁰ =
                matrix[@name(c.sgs⁰.ρatke), @name(c.sgs⁰.ρatke)]
            @. ∂ᶜρatke⁰_err_∂ᶜρ =
                dtγ * (
                    DiagonalMatrixRow(ᶜdissipation_matrix_diagonal)
                ) ⋅ DiagonalMatrixRow(ᶜtke⁰ / Y.c.ρ)
            @. ∂ᶜρatke⁰_err_∂ᶜρatke⁰ =
                dtγ * (
                    (
                        ᶜdiffusion_u_matrix -
                        DiagonalMatrixRow(ᶜdissipation_matrix_diagonal)
                    ) ⋅ DiagonalMatrixRow(1 / Y.c.ρ) - DiagonalMatrixRow(
                        tke_dissipation_rate_tendency(
                            ᶜtke⁰,
                            ᶜmixing_length_field,
                        ),
                    )
                ) - (I,)
        end

        if (
            !isnothing(p.atmos.turbconv_model) ||
            !disable_momentum_vertical_diffusion(p.atmos.vertical_diffusion)
        )
            ∂ᶜuₕ_err_∂ᶜuₕ = matrix[@name(c.uₕ), @name(c.uₕ)]
            @. ∂ᶜuₕ_err_∂ᶜuₕ =
                dtγ * DiagonalMatrixRow(1 / ᶜρ) ⋅ ᶜdiffusion_u_matrix - (I,)
        end

    end

    if p.atmos.turbconv_model isa PrognosticEDMFX
        if use_derivative(sgs_advection_flag)
            (; ᶜgradᵥ_ᶠΦ) = p.core
            (; ᶜρʲs, ᶠu³ʲs, ᶜtsʲs, ᶜKʲs, bdmr_l, bdmr_r, bdmr) = p.precomputed

            # upwinding options for q_tot and mse
            is_third_order =
                p.atmos.numerics.edmfx_mse_q_tot_upwinding == Val(:third_order)
            ᶠupwind = is_third_order ? ᶠupwind3 : ᶠupwind1
            ᶠset_upwind_bcs = Operators.SetBoundaryOperator(;
                top = Operators.SetValue(zero(CT3{FT})),
                bottom = Operators.SetValue(zero(CT3{FT})),
            ) # Need to wrap ᶠupwind in this for well-defined boundaries.
            UpwindMatrixRowType =
                is_third_order ? QuaddiagonalMatrixRow : BidiagonalMatrixRow
            ᶠupwind_matrix = is_third_order ? ᶠupwind3_matrix : ᶠupwind1_matrix
            ᶠset_upwind_matrix_bcs = Operators.SetBoundaryOperator(;
                top = Operators.SetValue(zero(UpwindMatrixRowType{CT3{FT}})),
                bottom = Operators.SetValue(zero(UpwindMatrixRowType{CT3{FT}})),
            ) # Need to wrap ᶠupwind_matrix in this for well-defined boundaries.

            # upwinding options for other tracers
            is_tracer_upwinding_third_order =
                p.atmos.numerics.edmfx_tracer_upwinding == Val(:third_order)
            ᶠtracer_upwind = is_tracer_upwinding_third_order ? ᶠupwind3 : ᶠupwind1
            ᶠset_tracer_upwind_bcs = Operators.SetBoundaryOperator(;
                top = Operators.SetValue(zero(CT3{FT})),
                bottom = Operators.SetValue(zero(CT3{FT})),
            ) # Need to wrap ᶠtracer_upwind in this for well-defined boundaries.
            TracerUpwindMatrixRowType =
                is_tracer_upwinding_third_order ? QuaddiagonalMatrixRow :
                BidiagonalMatrixRow
            ᶠtracer_upwind_matrix =
                is_tracer_upwinding_third_order ? ᶠupwind3_matrix : ᶠupwind1_matrix
            ᶠset_tracer_upwind_matrix_bcs = Operators.SetBoundaryOperator(;
                top = Operators.SetValue(zero(TracerUpwindMatrixRowType{CT3{FT}})),
                bottom = Operators.SetValue(zero(TracerUpwindMatrixRowType{CT3{FT}})),
            ) # Need to wrap ᶠtracer_upwind_matrix in this for well-defined boundaries.

            ᶠu³ʲ_data = ᶠu³ʲs.:(1).components.data.:1

            ᶜkappa_mʲ = p.scratch.ᶜtemp_scalar
            @. ᶜkappa_mʲ =
                TD.gas_constant_air(thermo_params, ᶜtsʲs.:(1)) /
                TD.cv_m(thermo_params, ᶜtsʲs.:(1))

            # Note this is the derivative of R_m / cp_m with respect to ᶜq_tot
            # but we call it ∂kappa_m∂q_totʲ
            ᶜ∂kappa_m∂q_totʲ = p.scratch.ᶜtemp_scalar_2
            @. ᶜ∂kappa_m∂q_totʲ =
                (
                    ΔR_v * TD.cp_m(thermo_params, ᶜtsʲs.:(1)) -
                    Δcp_v * TD.gas_constant_air(thermo_params, ᶜtsʲs.:(1))
                ) / abs2(TD.cp_m(thermo_params, ᶜtsʲs.:(1)))

            ∂ᶜq_totʲ_err_∂ᶜq_totʲ =
                matrix[@name(c.sgsʲs.:(1).q_tot), @name(c.sgsʲs.:(1).q_tot)]
            @. ∂ᶜq_totʲ_err_∂ᶜq_totʲ =
                dtγ * (
                    DiagonalMatrixRow(ᶜadvdivᵥ(ᶠu³ʲs.:(1))) -
                    ᶜadvdivᵥ_matrix() ⋅
                    ᶠset_upwind_matrix_bcs(ᶠupwind_matrix(ᶠu³ʲs.:(1)))
                ) - (I,)
            ∂ᶜq_totʲ_err_∂ᶠu₃ʲ =
                matrix[@name(c.sgsʲs.:(1).q_tot), @name(f.sgsʲs.:(1).u₃)]
            @. ∂ᶜq_totʲ_err_∂ᶠu₃ʲ =
                dtγ * (
                    -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(
                        ᶠset_upwind_bcs(
                            ᶠupwind(CT3(sign(ᶠu³ʲ_data)), Y.c.sgsʲs.:(1).q_tot),
                        ) * adjoint(C3(sign(ᶠu³ʲ_data))),
                    ) +
                    DiagonalMatrixRow(Y.c.sgsʲs.:(1).q_tot) ⋅ ᶜadvdivᵥ_matrix()
                ) ⋅ DiagonalMatrixRow(g³³(ᶠgⁱʲ))

            if p.atmos.moisture_model isa NonEquilMoistModel && (
                p.atmos.microphysics_model isa Microphysics1Moment ||
                p.atmos.microphysics_model isa Microphysics2Moment
            )

                ᶜa = (@. lazy(draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1))))
                ᶜ∂a∂z =
                    @. lazy(
                        ᶜprecipdivᵥ(ᶠinterp(ᶜJ) / ᶠJ * ᶠright_bias(Geometry.WVector(ᶜa))),
                    )
                ᶜinv_ρ̂ = (@. lazy(
                    specific(
                        FT(1),
                        Y.c.sgsʲs.:(1).ρa,
                        FT(0),
                        ᶜρʲs.:(1),
                        turbconv_model,
                    ),
                ))
                sgs_microphysics_tracers = (
                    (@name(c.sgsʲs.:(1).q_liq), @name(ᶜwₗʲs.:(1))),
                    (@name(c.sgsʲs.:(1).q_ice), @name(ᶜwᵢʲs.:(1))),
                    (@name(c.sgsʲs.:(1).q_rai), @name(ᶜwᵣʲs.:(1))),
                    (@name(c.sgsʲs.:(1).q_sno), @name(ᶜwₛʲs.:(1))),
                    (@name(c.sgsʲs.:(1).n_liq), @name(ᶜwₙₗʲs.:(1))),
                    (@name(c.sgsʲs.:(1).n_rai), @name(ᶜwₙᵣʲs.:(1))),
                )
                MatrixFields.unrolled_foreach(
                    sgs_microphysics_tracers,
                ) do (χʲ_name, wʲ_name)
                    MatrixFields.has_field(Y, χʲ_name) || return
                    ᶜχʲ = MatrixFields.get_field(Y, χʲ_name)
                    ᶜwʲ = MatrixFields.get_field(p.precomputed, wʲ_name)

                    # advection
                    ∂ᶜχʲ_err_∂ᶜχʲ = matrix[χʲ_name, χʲ_name]
                    @. ∂ᶜχʲ_err_∂ᶜχʲ =
                        dtγ * (
                            DiagonalMatrixRow(ᶜadvdivᵥ(ᶠu³ʲs.:(1))) -
                            ᶜadvdivᵥ_matrix() ⋅
                            ᶠset_tracer_upwind_matrix_bcs(
                                ᶠtracer_upwind_matrix(ᶠu³ʲs.:(1)),
                            )
                        ) - (I,)
                    ∂ᶜχʲ_err_∂ᶠu₃ʲ =
                        matrix[χʲ_name, @name(f.sgsʲs.:(1).u₃)]
                    @. ∂ᶜχʲ_err_∂ᶠu₃ʲ =
                        dtγ * (
                            -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(
                                ᶠset_tracer_upwind_bcs(
                                    ᶠtracer_upwind(CT3(sign(ᶠu³ʲ_data)), ᶜχʲ),
                                ) * adjoint(C3(sign(ᶠu³ʲ_data))),
                            ) +
                            DiagonalMatrixRow(ᶜχʲ) ⋅ ᶜadvdivᵥ_matrix()
                        ) ⋅ DiagonalMatrixRow(g³³(ᶠgⁱʲ))

                    # precipitation
                    @. ᶜtridiagonal_matrix_scalar =
                        dtγ * -(ᶜprecipdivᵥ_matrix()) ⋅
                        DiagonalMatrixRow(ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ) ⋅
                        ᶠright_bias_matrix() ⋅
                        DiagonalMatrixRow(-Geometry.WVector(ᶜwʲ) * ᶜa)
                    # precipitation detrainment
                    @. ᶜtridiagonal_matrix_scalar +=
                        dtγ * -DiagonalMatrixRow(ᶜρʲs.:(1) * ᶜwʲ * max(0, ᶜ∂a∂z))

                    @. ∂ᶜχʲ_err_∂ᶜχʲ +=
                        DiagonalMatrixRow(ᶜinv_ρ̂) ⋅ ᶜtridiagonal_matrix_scalar

                    if χʲ_name in (
                        @name(c.sgsʲs.:(1).q_liq),
                        @name(c.sgsʲs.:(1).q_ice),
                        @name(c.sgsʲs.:(1).q_rai),
                        @name(c.sgsʲs.:(1).q_sno),
                    )
                        ∂ᶜq_totʲ_err_∂ᶜχʲ =
                            matrix[@name(c.sgsʲs.:(1).q_tot), χʲ_name]
                        @. ∂ᶜq_totʲ_err_∂ᶜχʲ =
                            DiagonalMatrixRow(ᶜinv_ρ̂) ⋅ ᶜtridiagonal_matrix_scalar

                        ∂ᶜρaʲ_err_∂ᶜχʲ =
                            matrix[@name(c.sgsʲs.:(1).ρa), χʲ_name]
                        @. ∂ᶜρaʲ_err_∂ᶜχʲ = ᶜtridiagonal_matrix_scalar
                    end

                end
            end

            ∂ᶜmseʲ_err_∂ᶜq_totʲ =
                matrix[@name(c.sgsʲs.:(1).mse), @name(c.sgsʲs.:(1).q_tot)]
            @. ∂ᶜmseʲ_err_∂ᶜq_totʲ =
                dtγ * (
                    -DiagonalMatrixRow(
                        adjoint(ᶜinterp(ᶠu³ʲs.:(1))) * ᶜgradᵥ_ᶠΦ * Y.c.ρ / ᶜp *
                        (
                            (ᶜkappa_mʲ / (ᶜkappa_mʲ + 1) * ∂e_int_∂q_tot) +
                            ᶜ∂kappa_m∂q_totʲ * (
                                Y.c.sgsʲs.:(1).mse - ᶜΦ +
                                cp_d * T_0 +
                                ∂e_int_∂q_tot * Y.c.sgsʲs.:(1).q_tot
                            )
                        ),
                    )
                )
            ∂ᶜmseʲ_err_∂ᶜmseʲ =
                matrix[@name(c.sgsʲs.:(1).mse), @name(c.sgsʲs.:(1).mse)]
            @. ∂ᶜmseʲ_err_∂ᶜmseʲ =
                dtγ * (
                    DiagonalMatrixRow(ᶜadvdivᵥ(ᶠu³ʲs.:(1))) -
                    ᶜadvdivᵥ_matrix() ⋅
                    ᶠset_upwind_matrix_bcs(ᶠupwind_matrix(ᶠu³ʲs.:(1))) -
                    DiagonalMatrixRow(
                        adjoint(ᶜinterp(ᶠu³ʲs.:(1))) *
                        ᶜgradᵥ_ᶠΦ *
                        Y.c.ρ *
                        ᶜkappa_mʲ / ((ᶜkappa_mʲ + 1) * ᶜp),
                    )
                ) - (I,)
            ∂ᶜmseʲ_err_∂ᶠu₃ʲ =
                matrix[@name(c.sgsʲs.:(1).mse), @name(f.sgsʲs.:(1).u₃)]
            @. ∂ᶜmseʲ_err_∂ᶠu₃ʲ =
                dtγ * (
                    -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(
                        ᶠset_upwind_bcs(
                            ᶠupwind(CT3(sign(ᶠu³ʲ_data)), Y.c.sgsʲs.:(1).mse),
                        ) * adjoint(C3(sign(ᶠu³ʲ_data))),
                    ) +
                    DiagonalMatrixRow(Y.c.sgsʲs.:(1).mse) ⋅ ᶜadvdivᵥ_matrix()
                ) ⋅ DiagonalMatrixRow(g³³(ᶠgⁱʲ))

            ∂ᶜρaʲ_err_∂ᶜq_totʲ =
                matrix[@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).q_tot)]
            @. ᶠbidiagonal_matrix_ct3 =
                DiagonalMatrixRow(
                    ᶠset_upwind_bcs(
                        ᶠupwind(
                            ᶠu³ʲs.:(1),
                            draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)),
                        ),
                    ) / ᶠJ,
                ) ⋅ ᶠinterp_matrix() ⋅ DiagonalMatrixRow(
                    ᶜJ * (ᶜρʲs.:(1))^2 / ᶜp * (
                        ᶜkappa_mʲ / (ᶜkappa_mʲ + 1) * ∂e_int_∂q_tot +
                        ᶜ∂kappa_m∂q_totʲ * (
                            Y.c.sgsʲs.:(1).mse - ᶜΦ +
                            cp_d * T_0 +
                            ∂e_int_∂q_tot * Y.c.sgsʲs.:(1).q_tot
                        )
                    ),
                )
            @. ᶠbidiagonal_matrix_ct3_2 =
                DiagonalMatrixRow(ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ) ⋅
                ᶠset_upwind_matrix_bcs(ᶠupwind_matrix(ᶠu³ʲs.:(1))) ⋅
                DiagonalMatrixRow(
                    Y.c.sgsʲs.:(1).ρa * ᶜkappa_mʲ / ((ᶜkappa_mʲ + 1) * ᶜp) *
                    ∂e_int_∂q_tot,
                )
            @. ∂ᶜρaʲ_err_∂ᶜq_totʲ =
                dtγ * ᶜadvdivᵥ_matrix() ⋅
                (ᶠbidiagonal_matrix_ct3 - ᶠbidiagonal_matrix_ct3_2)

            ∂ᶜρaʲ_err_∂ᶜmseʲ =
                matrix[@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).mse)]
            @. ᶠbidiagonal_matrix_ct3 =
                DiagonalMatrixRow(
                    ᶠset_upwind_bcs(
                        ᶠupwind(
                            ᶠu³ʲs.:(1),
                            draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)),
                        ),
                    ) / ᶠJ,
                ) ⋅ ᶠinterp_matrix() ⋅ DiagonalMatrixRow(
                    ᶜJ * ᶜkappa_mʲ * (ᶜρʲs.:(1))^2 / ((ᶜkappa_mʲ + 1) * ᶜp),
                )
            @. ᶠbidiagonal_matrix_ct3_2 =
                DiagonalMatrixRow(ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ) ⋅
                ᶠset_upwind_matrix_bcs(ᶠupwind_matrix(ᶠu³ʲs.:(1))) ⋅
                DiagonalMatrixRow(
                    Y.c.sgsʲs.:(1).ρa * ᶜkappa_mʲ / ((ᶜkappa_mʲ + 1) * ᶜp),
                )
            @. ∂ᶜρaʲ_err_∂ᶜmseʲ =
                dtγ * ᶜadvdivᵥ_matrix() ⋅
                (ᶠbidiagonal_matrix_ct3 - ᶠbidiagonal_matrix_ct3_2)

            ∂ᶜρaʲ_err_∂ᶜρaʲ =
                matrix[@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).ρa)]
            @. ᶜadvection_matrix =
                -(ᶜadvdivᵥ_matrix()) ⋅
                DiagonalMatrixRow(ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ)
            @. ∂ᶜρaʲ_err_∂ᶜρaʲ =
                dtγ * ᶜadvection_matrix ⋅
                ᶠset_upwind_matrix_bcs(ᶠupwind_matrix(ᶠu³ʲs.:(1))) ⋅
                DiagonalMatrixRow(1 / ᶜρʲs.:(1)) - (I,)

            ∂ᶜρaʲ_err_∂ᶠu₃ʲ =
                matrix[@name(c.sgsʲs.:(1).ρa), @name(f.sgsʲs.:(1).u₃)]
            @. ∂ᶜρaʲ_err_∂ᶠu₃ʲ =
                dtγ * -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(
                    ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ *
                    ᶠset_upwind_bcs(
                        ᶠupwind(
                            CT3(sign(ᶠu³ʲ_data)),
                            draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)),
                        ),
                    ) *
                    adjoint(C3(sign(ᶠu³ʲ_data))) *
                    g³³(ᶠgⁱʲ),
                )

            turbconv_params = CAP.turbconv_params(params)
            α_b = CAP.pressure_normalmode_buoy_coeff1(turbconv_params)
            ∂ᶠu₃ʲ_err_∂ᶜq_totʲ =
                matrix[@name(f.sgsʲs.:(1).u₃), @name(c.sgsʲs.:(1).q_tot)]
            @. ∂ᶠu₃ʲ_err_∂ᶜq_totʲ =
                dtγ * DiagonalMatrixRow(
                    (1 - α_b) * ᶠgradᵥ_ᶜΦ * ᶠinterp(Y.c.ρ) /
                    (ᶠinterp(ᶜρʲs.:(1)))^2,
                ) ⋅ ᶠinterp_matrix() ⋅ DiagonalMatrixRow(
                    (ᶜρʲs.:(1))^2 / ᶜp * (
                        ᶜkappa_mʲ / (ᶜkappa_mʲ + 1) * ∂e_int_∂q_tot +
                        ᶜ∂kappa_m∂q_totʲ * (
                            Y.c.sgsʲs.:(1).mse - ᶜΦ +
                            cp_d * T_0 +
                            ∂e_int_∂q_tot * Y.c.sgsʲs.:(1).q_tot
                        )
                    ),
                )
            ∂ᶠu₃ʲ_err_∂ᶜmseʲ =
                matrix[@name(f.sgsʲs.:(1).u₃), @name(c.sgsʲs.:(1).mse)]
            @. ∂ᶠu₃ʲ_err_∂ᶜmseʲ =
                dtγ * DiagonalMatrixRow(
                    (1 - α_b) * ᶠgradᵥ_ᶜΦ * ᶠinterp(Y.c.ρ) /
                    (ᶠinterp(ᶜρʲs.:(1)))^2,
                ) ⋅ ᶠinterp_matrix() ⋅ DiagonalMatrixRow(
                    ᶜkappa_mʲ * (ᶜρʲs.:(1))^2 / ((ᶜkappa_mʲ + 1) * ᶜp),
                )

            ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ =
                matrix[@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)]
            ᶜu₃ʲ = p.scratch.ᶜtemp_C3
            @. ᶜu₃ʲ = ᶜinterp(Y.f.sgsʲs.:(1).u₃)
            @. bdmr_l = convert(BidiagonalMatrixRow{FT}, ᶜleft_bias_matrix())
            @. bdmr_r = convert(BidiagonalMatrixRow{FT}, ᶜright_bias_matrix())
            @. bdmr = ifelse(ᶜu₃ʲ.components.data.:1 > 0, bdmr_l, bdmr_r)
            @. ᶠtridiagonal_matrix_c3 = -(ᶠgradᵥ_matrix()) ⋅ bdmr
            if rs isa RayleighSponge
                @. ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ =
                    dtγ * (
                        ᶠtridiagonal_matrix_c3 ⋅
                        DiagonalMatrixRow(adjoint(CT3(Y.f.sgsʲs.:(1).u₃))) -
                        DiagonalMatrixRow(
                            β_rayleigh_w(rs, ᶠz, zmax) * (one_C3xACT3,),
                        )
                    ) - (I_u₃,)
            else
                @. ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ =
                    dtγ * ᶠtridiagonal_matrix_c3 ⋅
                    DiagonalMatrixRow(adjoint(CT3(Y.f.sgsʲs.:(1).u₃))) - (I_u₃,)
            end

            # vertical diffusion of updrafts
            if use_derivative(sgs_vertdiff_flag)
                α_vert_diff_tracer = CAP.α_vert_diff_tracer(params)
                @. ᶜdiffusion_h_matrix =
                    ᶜadvdivᵥ_matrix() ⋅
                    DiagonalMatrixRow(ᶠinterp(ᶜρʲs.:(1)) * ᶠinterp(ᶜK_h)) ⋅
                    ᶠgradᵥ_matrix()

                @. ∂ᶜmseʲ_err_∂ᶜmseʲ +=
                    dtγ * DiagonalMatrixRow(1 / ᶜρʲs.:(1)) ⋅ ᶜdiffusion_h_matrix
                @. ∂ᶜq_totʲ_err_∂ᶜq_totʲ +=
                    dtγ * DiagonalMatrixRow(1 / ᶜρʲs.:(1)) ⋅ ᶜdiffusion_h_matrix
                if p.atmos.moisture_model isa NonEquilMoistModel && (
                    p.atmos.microphysics_model isa Microphysics1Moment ||
                    p.atmos.microphysics_model isa Microphysics2Moment
                )
                    sgs_microphysics_tracers = (
                        (@name(c.sgsʲs.:(1).q_liq), FT(1)),
                        (@name(c.sgsʲs.:(1).q_ice), FT(1)),
                        (@name(c.sgsʲs.:(1).q_rai), α_vert_diff_tracer),
                        (@name(c.sgsʲs.:(1).q_sno), α_vert_diff_tracer),
                        (@name(c.sgsʲs.:(1).n_liq), FT(1)),
                        (@name(c.sgsʲs.:(1).n_rai), α_vert_diff_tracer),
                    )
                    MatrixFields.unrolled_foreach(
                        sgs_microphysics_tracers,
                    ) do (χʲ_name, α)
                        MatrixFields.has_field(Y, χʲ_name) || return
                        ∂ᶜχʲ_err_∂ᶜχʲ = matrix[χʲ_name, χʲ_name]
                        @. ∂ᶜχʲ_err_∂ᶜχʲ +=
                            dtγ * α * DiagonalMatrixRow(1 / ᶜρʲs.:(1)) ⋅
                            ᶜdiffusion_h_matrix
                    end
                end
            end
            # entrainment and detrainment (rates are treated explicitly)
            if use_derivative(sgs_entr_detr_flag)
                (; ᶜentrʲs, ᶜdetrʲs, ᶜturb_entrʲs) = p.precomputed
                @. ∂ᶜq_totʲ_err_∂ᶜq_totʲ -=
                    dtγ * DiagonalMatrixRow(ᶜentrʲs.:(1) + ᶜturb_entrʲs.:(1))
                @. ∂ᶜmseʲ_err_∂ᶜmseʲ -=
                    dtγ * DiagonalMatrixRow(ᶜentrʲs.:(1) + ᶜturb_entrʲs.:(1))
                @. ∂ᶜρaʲ_err_∂ᶜρaʲ +=
                    dtγ * DiagonalMatrixRow(ᶜentrʲs.:(1) - ᶜdetrʲs.:(1))
                @. ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ -=
                    dtγ * (DiagonalMatrixRow(
                        (ᶠinterp(ᶜentrʲs.:(1) + ᶜturb_entrʲs.:(1))) *
                        (one_C3xACT3,),
                    ))
                if p.atmos.moisture_model isa NonEquilMoistModel && (
                    p.atmos.microphysics_model isa Microphysics1Moment ||
                    p.atmos.microphysics_model isa Microphysics2Moment
                )
                    sgs_microphysics_tracers = (
                        (@name(c.sgsʲs.:(1).q_liq)),
                        (@name(c.sgsʲs.:(1).q_ice)),
                        (@name(c.sgsʲs.:(1).q_rai)),
                        (@name(c.sgsʲs.:(1).q_sno)),
                    )
                    MatrixFields.unrolled_foreach(
                        sgs_microphysics_tracers,
                    ) do (qʲ_name)
                        MatrixFields.has_field(Y, qʲ_name) || return

                        ∂ᶜqʲ_err_∂ᶜqʲ = matrix[qʲ_name, qʲ_name]
                        @. ∂ᶜqʲ_err_∂ᶜqʲ -=
                            dtγ * DiagonalMatrixRow(ᶜentrʲs.:(1) + ᶜturb_entrʲs.:(1))
                    end
                end
            end

            # non-hydrostatic pressure drag
            # (quadratic drag term treated implicitly, buoyancy term explicitly)
            if use_derivative(sgs_nh_pressure_flag)
                (; ᶠu₃⁰) = p.precomputed
                α_d = CAP.pressure_normalmode_drag_coeff(turbconv_params)
                scale_height =
                    CAP.R_d(params) * CAP.T_surf_ref(params) / CAP.grav(params)
                H_up_min = CAP.min_updraft_top(turbconv_params)
                @. ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ -=
                    dtγ * (DiagonalMatrixRow(
                        2 * α_d * norm(Y.f.sgsʲs.:(1).u₃ - ᶠu₃⁰) /
                        max(scale_height, H_up_min) * (one_C3xACT3,),
                    ))
            end

            # add updraft mass flux contributions to grid-mean
            if use_derivative(sgs_mass_flux_flag)
                # Jacobian contributions of updraft massflux to grid-mean
                ∂ᶜupdraft_mass_flux_∂ᶜscalar = ᶠbidiagonal_matrix_ct3
                @. ∂ᶜupdraft_mass_flux_∂ᶜscalar =
                    DiagonalMatrixRow(
                        (ᶠinterp(ᶜρ * ᶜJ) / ᶠJ) * (ᶠu³ʲs.:(1) - ᶠu³),
                    ) ⋅ ᶠinterp_matrix() ⋅
                    DiagonalMatrixRow(Y.c.sgsʲs.:(1).ρa / ᶜρʲs.:(1))

                # Derivative of total energy tendency with respect to updraft MSE
                ## grid-mean ρe_tot
                ᶜkappa_m = p.scratch.ᶜtemp_scalar
                @. ᶜkappa_m =
                    TD.gas_constant_air(thermo_params, ᶜts) /
                    TD.cv_m(thermo_params, ᶜts)

                ᶜ∂kappa_m∂q_tot = p.scratch.ᶜtemp_scalar_2
                @. ᶜ∂kappa_m∂q_tot =
                    (
                        ΔR_v * TD.cv_m(thermo_params, ᶜts) -
                        Δcv_v * TD.gas_constant_air(thermo_params, ᶜts)
                    ) / abs2(TD.cv_m(thermo_params, ᶜts))

                ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))

                @. ∂ᶜρe_tot_err_∂ᶜρ +=
                    dtγ * ᶜadvdivᵥ_matrix() ⋅ ∂ᶜupdraft_mass_flux_∂ᶜscalar ⋅
                    DiagonalMatrixRow(
                        (
                            -(1 + ᶜkappa_m) * ᶜe_tot -
                            ᶜkappa_m * ∂e_int_∂q_tot * ᶜq_tot
                        ) / ᶜρ,
                    )

                @. ∂ᶜρe_tot_err_∂ᶜρq_tot +=
                    dtγ * ᶜadvdivᵥ_matrix() ⋅ ∂ᶜupdraft_mass_flux_∂ᶜscalar ⋅
                    DiagonalMatrixRow((
                        ᶜkappa_m * ∂e_int_∂q_tot / ᶜρ +
                        ᶜ∂kappa_m∂q_tot * (
                            cp_d * T_0 + ᶜe_tot - ᶜK - ᶜΦ +
                            ∂e_int_∂q_tot * ᶜq_tot
                        )
                    ))

                @. ∂ᶜρe_tot_err_∂ᶜρe_tot +=
                    dtγ * ᶜadvdivᵥ_matrix() ⋅ ∂ᶜupdraft_mass_flux_∂ᶜscalar ⋅
                    DiagonalMatrixRow((1 + ᶜkappa_m) / ᶜρ)

                ∂ᶜρe_tot_err_∂ᶜmseʲ =
                    matrix[@name(c.ρe_tot), @name(c.sgsʲs.:(1).mse)]
                @. ∂ᶜρe_tot_err_∂ᶜmseʲ =
                    -(dtγ * ᶜadvdivᵥ_matrix() ⋅ ∂ᶜupdraft_mass_flux_∂ᶜscalar)

                ## grid-mean ρq_tot
                @. ∂ᶜρq_tot_err_∂ᶜρ +=
                    dtγ * ᶜadvdivᵥ_matrix() ⋅ ∂ᶜupdraft_mass_flux_∂ᶜscalar ⋅
                    DiagonalMatrixRow(-(ᶜq_tot) / ᶜρ)

                @. ∂ᶜρq_tot_err_∂ᶜρq_tot +=
                    dtγ * ᶜadvdivᵥ_matrix() ⋅ ∂ᶜupdraft_mass_flux_∂ᶜscalar ⋅
                    DiagonalMatrixRow(1 / ᶜρ)

                ∂ᶜρq_tot_err_∂ᶜq_totʲ =
                    matrix[@name(c.ρq_tot), @name(c.sgsʲs.:(1).q_tot)]
                @. ∂ᶜρq_tot_err_∂ᶜq_totʲ =
                    -(dtγ * ᶜadvdivᵥ_matrix() ⋅ ∂ᶜupdraft_mass_flux_∂ᶜscalar)

                # grid-mean ∂/∂(u₃ʲ)
                ∂ᶜρe_tot_err_∂ᶠu₃ = matrix[@name(c.ρe_tot), @name(f.u₃)]
                @. ∂ᶜρe_tot_err_∂ᶠu₃ +=
                    dtγ * ᶜadvdivᵥ_matrix() ⋅ DiagonalMatrixRow(
                        ᶠinterp(
                            (Y.c.sgsʲs.:(1).mse + ᶜKʲs.:(1) - ᶜh_tot) *
                            ᶜρʲs.:(1) *
                            ᶜJ *
                            draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)),
                        ) / ᶠJ * (g³³(ᶠgⁱʲ)),
                    )

                ∂ᶜρe_tot_err_∂ᶠu₃ʲ =
                    matrix[@name(c.ρe_tot), @name(f.sgsʲs.:(1).u₃)]
                @. ∂ᶜρe_tot_err_∂ᶠu₃ʲ =
                    dtγ * -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(
                        ᶠinterp(
                            (Y.c.sgsʲs.:(1).mse + ᶜKʲs.:(1) - ᶜh_tot) *
                            ᶜρʲs.:(1) *
                            ᶜJ *
                            draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)),
                        ) / ᶠJ * (g³³(ᶠgⁱʲ)),
                    )

                ∂ᶜρq_tot_err_∂ᶠu₃ = matrix[@name(c.ρq_tot), @name(f.u₃)]
                @. ∂ᶜρq_tot_err_∂ᶠu₃ +=
                    dtγ * ᶜadvdivᵥ_matrix() ⋅ DiagonalMatrixRow(
                        ᶠinterp(
                            (Y.c.sgsʲs.:(1).q_tot - ᶜq_tot) *
                            ᶜρʲs.:(1) *
                            ᶜJ *
                            draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)),
                        ) / ᶠJ * (g³³(ᶠgⁱʲ)),
                    )

                ∂ᶜρq_tot_err_∂ᶠu₃ʲ =
                    matrix[@name(c.ρq_tot), @name(f.sgsʲs.:(1).u₃)]
                @. ∂ᶜρq_tot_err_∂ᶠu₃ʲ =
                    dtγ * -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(
                        ᶠinterp(
                            (Y.c.sgsʲs.:(1).q_tot - ᶜq_tot) *
                            ᶜρʲs.:(1) *
                            ᶜJ *
                            draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)),
                        ) / ᶠJ * (g³³(ᶠgⁱʲ)),
                    )

                # grid-mean ∂/∂(rho*a)
                ∂ᶜρe_tot_err_∂ᶜρa =
                    matrix[@name(c.ρe_tot), @name(c.sgsʲs.:(1).ρa)]
                @. ∂ᶜρe_tot_err_∂ᶜρa =
                    dtγ * -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(
                        (ᶠu³ʲs.:(1) - ᶠu³) *
                        ᶠinterp((Y.c.sgsʲs.:(1).mse + ᶜKʲs.:(1) - ᶜh_tot)) / ᶠJ,
                    ) ⋅ ᶠinterp_matrix() ⋅ DiagonalMatrixRow(ᶜJ)

                ∂ᶜρq_tot_err_∂ᶜρa =
                    matrix[@name(c.ρq_tot), @name(c.sgsʲs.:(1).ρa)]
                @. ∂ᶜρq_tot_err_∂ᶜρa =
                    dtγ * -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(
                        (ᶠu³ʲs.:(1) - ᶠu³) *
                        ᶠinterp((Y.c.sgsʲs.:(1).q_tot - ᶜq_tot)) / ᶠJ,
                    ) ⋅ ᶠinterp_matrix() ⋅ DiagonalMatrixRow(ᶜJ)

                # grid-mean tracers
                if p.atmos.moisture_model isa NonEquilMoistModel && (
                    p.atmos.microphysics_model isa Microphysics1Moment ||
                    p.atmos.microphysics_model isa Microphysics2Moment
                )

                    microphysics_tracers = (
                        (@name(c.ρq_liq), @name(c.sgsʲs.:(1).q_liq)),
                        (@name(c.ρq_ice), @name(c.sgsʲs.:(1).q_ice)),
                        (@name(c.ρq_rai), @name(c.sgsʲs.:(1).q_rai)),
                        (@name(c.ρq_sno), @name(c.sgsʲs.:(1).q_sno)),
                        (@name(c.ρn_liq), @name(c.sgsʲs.:(1).n_liq)),
                        (@name(c.ρn_rai), @name(c.sgsʲs.:(1).n_rai)),
                    )
                    MatrixFields.unrolled_foreach(
                        microphysics_tracers,
                    ) do (ρχ_name, χʲ_name)
                        MatrixFields.has_field(Y, ρχ_name) || return
                        ᶜχʲ = MatrixFields.get_field(Y, χʲ_name)

                        ∂ᶜρχ_err_∂ᶜχʲ =
                            matrix[ρχ_name, χʲ_name]
                        @. ∂ᶜρχ_err_∂ᶜχʲ =
                            dtγ *
                            -(ᶜadvdivᵥ_matrix()) ⋅
                            DiagonalMatrixRow(ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ) ⋅
                            ᶠset_tracer_upwind_matrix_bcs(
                                ᶠtracer_upwind_matrix(ᶠu³ʲs.:(1)),
                            ) ⋅
                            DiagonalMatrixRow(draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)))

                        ∂ᶜρχ_err_∂ᶜρa =
                            matrix[ρχ_name, @name(c.sgsʲs.:(1).ρa)]
                        @. ∂ᶜρχ_err_∂ᶜρa =
                            dtγ *
                            -(ᶜadvdivᵥ_matrix()) ⋅
                            DiagonalMatrixRow(ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ) ⋅
                            ᶠset_tracer_upwind_matrix_bcs(
                                ᶠtracer_upwind_matrix(ᶠu³ʲs.:(1)),
                            ) ⋅
                            DiagonalMatrixRow(ᶜχʲ / ᶜρʲs.:(1))

                        ∂ᶜρχ_err_∂ᶠu₃ʲ =
                            matrix[ρχ_name, @name(f.sgsʲs.:(1).u₃)]
                        @. ∂ᶜρχ_err_∂ᶠu₃ʲ =
                            dtγ * (
                                -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(
                                    ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ *
                                    ᶠset_tracer_upwind_bcs(
                                        ᶠtracer_upwind(CT3(sign(ᶠu³ʲ_data)),
                                            draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)) * ᶜχʲ,
                                        ),
                                    ) * adjoint(C3(sign(ᶠu³ʲ_data))),
                                )) ⋅ DiagonalMatrixRow(g³³(ᶠgⁱʲ))

                    end
                end
            end
        elseif rs isa RayleighSponge
            ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ =
                matrix[@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)]
            @. ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ =
                dtγ *
                -DiagonalMatrixRow(
                    β_rayleigh_w(rs, ᶠz, zmax) * (one_C3xACT3,),
                ) - (I_u₃,)
        end
    end

    # NOTE: All velocity tendency derivatives should be set BEFORE this call.
    zero_velocity_jacobian!(matrix, Y, p, t)
end

invert_jacobian!(::ManualSparseJacobian, cache, ΔY, R) =
    LinearAlgebra.ldiv!(ΔY, cache.matrix, R)
