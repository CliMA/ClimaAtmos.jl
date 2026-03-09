import LinearAlgebra: I

using ClimaCore.MatrixFields
import ClimaCore.MatrixFields: @name

function jacobian_sgs_u₃_cache(alg, cache, Y, p, dtγ, t)
    error("jacobian_sgs_u₃_cache is not implemented for the given Jacobian algorithm!!!")
end
function jacobian_sgs_u₃_cache(alg::ManualSparseJacobian, Y, atmos)
    (;
        sgs_advection_flag,
    ) = alg
    FT = Spaces.undertype(axes(Y.c))

    DiagonalRow_C3xACT3 =
        DiagonalMatrixRow{typeof(zero(C3{FT}) * zero(CT3{FT})')}
    TridiagonalRow_C3xACT3 =
        TridiagonalMatrixRow{typeof(zero(C3{FT}) * zero(CT3{FT})')}

    is_in_Y(name) = MatrixFields.has_field(Y, name)

    gs_scalar_names = (
        @name(c.ρ),
        @name(sfc),
        @name(c.ρtke),
        @name(c.ρe_tot),
        @name(c.ρq_tot),
        @name(c.ρq_liq),
        @name(c.ρq_ice),
        @name(c.ρq_rai),
        @name(c.ρq_sno),
        @name(c.ρn_liq),
        @name(c.ρn_rai),
        @name(c.ρn_ice), @name(c.ρq_rim), @name(c.ρb_rim),
        @name(c.uₕ), @name(f.u₃),
    )
    available_gs_scalar_names =
        MatrixFields.unrolled_filter(is_in_Y, gs_scalar_names)

    sgs_scalar_names =
        (
            @name(c.sgsʲs.:(1).q_liq),
            @name(c.sgsʲs.:(1).q_ice),
            @name(c.sgsʲs.:(1).q_rai),
            @name(c.sgsʲs.:(1).q_sno),
            @name(c.sgsʲs.:(1).n_liq),
            @name(c.sgsʲs.:(1).n_rai),
            @name(c.sgsʲs.:(1).q_tot),
            @name(c.sgsʲs.:(1).mse),
            @name(c.sgsʲs.:(1).ρa)
        )
    available_sgs_scalar_names =
        MatrixFields.unrolled_filter(is_in_Y, sgs_scalar_names)

    gs_blocks = MatrixFields.unrolled_map(
        name -> (name, name) => FT(-1) * I,
        available_gs_scalar_names,
    )

    sgs_blocks = (
        MatrixFields.unrolled_map(
            name -> (name, name) => FT(-1) * I,
            available_sgs_scalar_names,
        )...,
        if use_derivative(sgs_advection_flag)
            (@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)) =>
                similar(Y.f, TridiagonalRow_C3xACT3)
        else
            (@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)) =>
                !isnothing(atmos.rayleigh_sponge) ?
                similar(Y.f, DiagonalRow_C3xACT3) : FT(-1) * I
        end,
    )

    matrix = MatrixFields.FieldMatrix(
        gs_blocks...,
        sgs_blocks...,
    )

    alg = MatrixFields.BlockDiagonalSolve(
    )

    return (; matrix = MatrixFields.FieldMatrixWithSolver(matrix, Y, alg))
end

function update_jacobian_sgs_u₃!(alg, cache, Y, p, dtγ, t)
    error("update_jacobian_sgs_u₃ is not implemented for the given Jacobian algorithm!!!")
end
function update_jacobian_sgs_u₃!(alg::ManualSparseJacobian, cache, Y, p, dtγ, t)
    (;
        sgs_advection_flag,
        sgs_entr_detr_flag,
        sgs_nh_pressure_flag,
    ) = alg
    (; matrix) = cache
    (; params) = p
    (; ᶠtridiagonal_matrix_c3,) = p.scratch
    rs = p.atmos.rayleigh_sponge
    FT = Spaces.undertype(axes(Y.c))
    turbconv_params = CAP.turbconv_params(params)
    one_C3xACT3 = C3(FT(1)) * CT3(FT(1))'
    I_u₃ = DiagonalMatrixRow(one_C3xACT3)
    ᶠz = Fields.coordinate_field(Y.f).z
    zmax = z_max(axes(Y.f))

    if p.atmos.turbconv_model isa PrognosticEDMFX
        if use_derivative(sgs_advection_flag)
            ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ =
                matrix[@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)]
            ᶜu₃ʲ = p.scratch.ᶜtemp_C3
            @. ᶜu₃ʲ = ᶜinterp(Y.f.sgsʲs.:(1).u₃)
            @. p.scratch.ᶜtemp_bdmr = convert(BidiagonalMatrixRow{FT}, ᶜleft_bias_matrix())
            @. p.scratch.ᶜtemp_bdmr_2 =
                convert(BidiagonalMatrixRow{FT}, ᶜright_bias_matrix())
            @. p.scratch.ᶜtemp_bdmr_3 = ifelse(
                ᶜu₃ʲ.components.data.:1 > 0,
                p.scratch.ᶜtemp_bdmr,
                p.scratch.ᶜtemp_bdmr_2,
            )
            @. ᶠtridiagonal_matrix_c3 = -(ᶠgradᵥ_matrix()) ⋅ p.scratch.ᶜtemp_bdmr_3
            if rs isa RayleighSponge
                @. ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ =
                    dtγ * (
                        ᶠtridiagonal_matrix_c3 ⋅
                        DiagonalMatrixRow(adjoint(CT3(Y.f.sgsʲs.:(1).u₃))) -
                        DiagonalMatrixRow(β_rayleigh_u₃(rs, ᶠz, zmax) * (one_C3xACT3,))
                    ) - (I_u₃,)
            else
                @. ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ =
                    dtγ * ᶠtridiagonal_matrix_c3 ⋅
                    DiagonalMatrixRow(adjoint(CT3(Y.f.sgsʲs.:(1).u₃))) - (I_u₃,)
            end

            # entrainment and detrainment (rates are treated explicitly)
            if use_derivative(sgs_entr_detr_flag)
                (; ᶜentrʲs, ᶜturb_entrʲs) = p.precomputed
                @. ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ -=
                    dtγ * (DiagonalMatrixRow(
                        (ᶠinterp(ᶜentrʲs.:(1) + ᶜturb_entrʲs.:(1))) *
                        (one_C3xACT3,),
                    ))
            end

            # non-hydrostatic pressure drag
            # (quadratic drag term treated implicitly, buoyancy term explicitly)
            if use_derivative(sgs_nh_pressure_flag)
                (; ᶠu₃⁰) = p.precomputed
                α_d = CAP.pressure_normalmode_drag_coeff(turbconv_params)
                scale_height =
                    CAP.R_d(params) * CAP.T_surf_ref(params) / CAP.grav(params)
                H_up_min = CAP.min_updraft_top(turbconv_params)
                ᶠlg = Fields.local_geometry_field(Y.f)
                @. ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ -=
                    dtγ * (DiagonalMatrixRow(
                        2 * α_d * CC.Geometry._norm(Y.f.sgsʲs.:(1).u₃ - ᶠu₃⁰, ᶠlg) /
                        max(scale_height, H_up_min) * (one_C3xACT3,),
                    ))
            end

        elseif rs isa RayleighSponge
            ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ =
                matrix[@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)]
            @. ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ =
                dtγ *
                -DiagonalMatrixRow(β_rayleigh_u₃(rs, ᶠz, zmax) * (one_C3xACT3,)) - (I_u₃,)
        end
    end

    # NOTE: All velocity tendency derivatives should be set BEFORE this call.
    zero_velocity_jacobian!(matrix, Y, p, t)
end
