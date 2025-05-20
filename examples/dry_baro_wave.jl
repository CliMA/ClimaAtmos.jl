#=
       .-.      Welcome to ClimaAtmos!
      (   ).    ----------------------
     (___(__)   A state-of-the-art Julia model for
    ⌜^^^^^^^⌝   simulating atmospheric dynamics.
   ⌜  ~  ~  ⌝
  ⌜ ~  ~  ~  ⌝  This example: *Dry Baroclinic Wave*
 ⌜  ~   ~  ~ ⌝
⌜~~~~~~~~~~~~~⌝  ⚡ Harnessing GPU acceleration with CUDA.jl
    “““““““      🌎 Pushing the frontiers of climate science!

Run with
```
julia +1.11 --project=.buildkite
using Revise; include("examples/dry_baro_wave.jl")
=#

import ClimaComms
ClimaComms.@import_required_backends
using ClimaCore.CommonSpaces
import ClimaAtmos as CA
using LazyBroadcast: lazy
using LinearAlgebra: ×
import ClimaAtmos.Parameters as CAP
import Thermodynamics as TD
import SciMLBase
import ClimaTimeSteppers as CTS
import ClimaCore.Geometry
import ClimaCore.MatrixFields: @name, ⋅
import ClimaCore.MatrixFields: DiagonalMatrixRow, BidiagonalMatrixRow
import LinearAlgebra: Adjoint
import LinearAlgebra: adjoint
import LinearAlgebra as LA
import ClimaCore.MatrixFields
import ClimaCore.Spaces
import ClimaCore.Fields

import ClimaAtmos: C1, C2, C12, C3, C123, CT1, CT2, CT12, CT3, CT123, UVW
import ClimaAtmos:
    divₕ, wdivₕ, gradₕ, wgradₕ, curlₕ, wcurlₕ, ᶜinterp, ᶜdivᵥ, ᶜgradᵥ
import ClimaAtmos: ᶠinterp, ᶠgradᵥ, ᶠcurlᵥ, ᶜinterp_matrix, ᶠgradᵥ_matrix
import ClimaAtmos: ᶜadvdivᵥ_matrix, ᶠwinterp, ᶠinterp_matrix

# Define tendency functions
function implicit_tendency!(Yₜ, Y, p, t)
    # Yₜ .= zero(eltype(Yₜ))
    set_precomputed_quantities!(Y, p, t)
    (; rayleigh_sponge, params, dt) = p
    (; ᶜh_tot, ᶠu³, ᶜp) = p.precomputed
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠz = Fields.coordinate_field(Y.f).z
    grav = FT(CAP.grav(params))
    zmax = CA.z_max(axes(Y.f))

    @. Yₜ.c.ρ -= ᶜdivᵥ(ᶠwinterp(ᶜJ, Y.c.ρ) * ᶠu³)
    # Central advection of active tracers (e_tot and q_tot)
    Yₜ.c.ρe_tot .+= CA.vertical_transport(Y.c.ρ, ᶠu³, ᶜh_tot, dt, Val(:none))
    @. Yₜ.f.u₃ -= ᶠgradᵥ(ᶜp) / ᶠinterp(Y.c.ρ) + ᶠgradᵥ(Φ(grav, ᶜz))

    @. Yₜ.f.u₃ -= CA.β_rayleigh_w(rayleigh_sponge, ᶠz, zmax) * Y.f.u₃
    return nothing
end

function ImplicitEquationJacobian(
    Y::Fields.FieldVector;
    approximate_solve_iters = 1,
    transform_flag = false,
)
    FT = Spaces.undertype(axes(Y.c))
    CTh = CA.CTh_vector_type(axes(Y.c))

    BidiagonalRow_C3 = MatrixFields.BidiagonalMatrixRow{CA.C3{FT}}
    BidiagonalRow_ACT3 =
        MatrixFields.BidiagonalMatrixRow{LA.Adjoint{FT, CA.CT3{FT}}}
    BidiagonalRow_C3xACTh = MatrixFields.BidiagonalMatrixRow{
        typeof(zero(CA.C3{FT}) * zero(CTh{FT})'),
    }
    TridiagonalRow_C3xACT3 = MatrixFields.TridiagonalMatrixRow{
        typeof(zero(CA.C3{FT}) * zero(CA.CT3{FT})'),
    }

    is_in_Y(name) = MatrixFields.has_field(Y, name)

    sfc_if_available = is_in_Y(@name(sfc)) ? (@name(sfc),) : ()


    # Note: We have to use FT(-1) * I instead of -I because inv(-1) == -1.0,
    # which means that multiplying inv(-1) by a Float32 will yield a Float64.
    identity_blocks = MatrixFields.unrolled_map(
        name -> (name, name) => FT(-1) * LA.I,
        (@name(c.ρ), sfc_if_available...),
    )

    active_scalar_names = (@name(c.ρ), @name(c.ρe_tot))
    advection_blocks = (
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

    diffused_scalar_names = (@name(c.ρe_tot),)
    diffusion_blocks = MatrixFields.unrolled_map(
        name -> (name, name) => FT(-1) * LA.I,
        (diffused_scalar_names..., @name(c.uₕ)),
    )

    matrix = MatrixFields.FieldMatrix(
        identity_blocks...,
        advection_blocks...,
        diffusion_blocks...,
    )

    names₁_group₁ = (@name(c.ρ), sfc_if_available...)
    names₁_group₃ = (@name(c.ρe_tot),)
    names₁ = (names₁_group₁..., names₁_group₃...)

    alg₂ = MatrixFields.BlockLowerTriangularSolve(@name(c.uₕ))
    alg = MatrixFields.BlockArrowheadSolve(names₁...; alg₂)

    return CA.ImplicitEquationJacobian(
        matrix,
        MatrixFields.FieldMatrixSolver(alg, matrix, Y),
        CA.IgnoreDerivative(), # diffusion_flag
        CA.IgnoreDerivative(), # topography_flag
        CA.IgnoreDerivative(), # sgs_advection_flag
        similar(Y),
        similar(Y),
        transform_flag,
        Ref{FT}(),
    )
end

function Wfact!(A, Y, p, dtγ, t)
    FT = Spaces.undertype(axes(Y.c))
    dtγ′ = FT(float(dtγ))
    A.dtγ_ref[] = dtγ′
    update_implicit_equation_jacobian!(A, Y, p, dtγ′)
end

Φ(grav, z) = grav * z

function update_implicit_equation_jacobian!(A, Y, p, dtγ)
    (; matrix) = A
    (; ᶜK, ᶜts, ᶜp, ᶜh_tot) = p.precomputed
    (; ∂ᶜK_∂ᶜuₕ, ∂ᶜK_∂ᶠu₃, ᶠp_grad_matrix, ᶜadvection_matrix) = p
    (; params) = p

    FT = Spaces.undertype(axes(Y.c))
    CTh = CA.CTh_vector_type(axes(Y.c))
    one_C3xACT3 = C3(FT(1)) * CT3(FT(1))'
    rs = p.rayleigh_sponge
    ᶠz = Fields.coordinate_field(Y.f).z
    zmax = CA.z_max(axes(Y.f))

    T_0 = FT(CAP.T_0(params))
    cp_d = FT(CAP.cp_d(params))
    thermo_params = CAP.thermodynamics_params(params)
    ᶜz = Fields.coordinate_field(Y.c).z
    grav = FT(CAP.grav(params))

    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠu₃ = Y.f.u₃
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠgⁱʲ = Fields.local_geometry_field(Y.f).gⁱʲ

    ᶜkappa_m = p.ᶜtemp_scalar
    @. ᶜkappa_m =
        TD.gas_constant_air(thermo_params, ᶜts) / TD.cv_m(thermo_params, ᶜts)

    @. ∂ᶜK_∂ᶜuₕ = DiagonalMatrixRow(adjoint(CTh(ᶜuₕ)))
    @. ∂ᶜK_∂ᶠu₃ =
        ᶜinterp_matrix() ⋅ DiagonalMatrixRow(adjoint(CT3(ᶠu₃))) +
        DiagonalMatrixRow(adjoint(CT3(ᶜuₕ))) ⋅ ᶜinterp_matrix()

    @. ᶠp_grad_matrix = DiagonalMatrixRow(-1 / ᶠinterp(ᶜρ)) ⋅ ᶠgradᵥ_matrix()

    @. ᶜadvection_matrix =
        -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(ᶠwinterp(ᶜJ, ᶜρ))

    ∂ᶜρ_err_∂ᶠu₃ = matrix[@name(c.ρ), @name(f.u₃)]
    @. ∂ᶜρ_err_∂ᶠu₃ = dtγ * ᶜadvection_matrix ⋅ DiagonalMatrixRow(CA.g³³(ᶠgⁱʲ))

    ∂ᶜρχ_err_∂ᶠu₃ = matrix[@name(c.ρe_tot), @name(f.u₃)]
    @. ∂ᶜρχ_err_∂ᶠu₃ =
        dtγ * ᶜadvection_matrix ⋅
        DiagonalMatrixRow(ᶠinterp(ᶜh_tot) * CA.g³³(ᶠgⁱʲ))

    ∂ᶠu₃_err_∂ᶜρ = matrix[@name(f.u₃), @name(c.ρ)]
    ∂ᶠu₃_err_∂ᶜρe_tot = matrix[@name(f.u₃), @name(c.ρe_tot)]

    @. ∂ᶠu₃_err_∂ᶜρ =
        dtγ * (
            ᶠp_grad_matrix ⋅
            DiagonalMatrixRow(ᶜkappa_m * (T_0 * cp_d - ᶜK - Φ(grav, ᶜz))) +
            DiagonalMatrixRow(ᶠgradᵥ(ᶜp) / abs2(ᶠinterp(ᶜρ))) ⋅
            ᶠinterp_matrix()
        )
    @. ∂ᶠu₃_err_∂ᶜρe_tot = dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(ᶜkappa_m)

    ∂ᶠu₃_err_∂ᶜuₕ = matrix[@name(f.u₃), @name(c.uₕ)]
    ∂ᶠu₃_err_∂ᶠu₃ = matrix[@name(f.u₃), @name(f.u₃)]
    I_u₃ = DiagonalMatrixRow(one_C3xACT3)
    @. ∂ᶠu₃_err_∂ᶜuₕ =
        dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(-(ᶜkappa_m) * ᶜρ) ⋅ ∂ᶜK_∂ᶜuₕ

    @. ∂ᶠu₃_err_∂ᶠu₃ =
        dtγ * (
            ᶠp_grad_matrix ⋅ DiagonalMatrixRow(-(ᶜkappa_m) * ᶜρ) ⋅ ∂ᶜK_∂ᶠu₃ +
            DiagonalMatrixRow(-CA.β_rayleigh_w(rs, ᶠz, zmax) * (one_C3xACT3,))
        ) - (I_u₃,)

end

function set_precomputed_quantities!(Y, p, t)
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜu, ᶠu³, ᶠu, ᶜK, ᶜts, ᶜp) = p.precomputed

    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶜz = Fields.coordinate_field(Y.c).z
    grav = FT(CAP.grav(params))
    ᶠu₃ = Y.f.u₃
    @. ᶜu = C123(ᶜuₕ) + ᶜinterp(C123(ᶠu₃))
    ᶠu³ .= CA.compute_ᶠuₕ³(ᶜuₕ, ᶜρ) .+ CT3.(ᶠu₃)
    ᶜK .= CA.compute_kinetic(ᶜuₕ, ᶠu₃)

    @. ᶜts = TD.PhaseDry_ρe(
        thermo_params,
        Y.c.ρ,
        Y.c.ρe_tot / Y.c.ρ - ᶜK - Φ(grav, ᶜz),
    )
    @. ᶜp = TD.air_pressure(thermo_params, ᶜts)

    (; ᶜh_tot) = p.precomputed
    @. ᶜh_tot =
        TD.total_specific_enthalpy(thermo_params, ᶜts, Y.c.ρe_tot / Y.c.ρ)
    return nothing
end

function dss!(Y, p, t)
    Spaces.weighted_dss!(Y.c => p.ghost_buffer.c, Y.f => p.ghost_buffer.f)
    return nothing
end

function remaining_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    # Yₜ_lim .= zero(eltype(Yₜ_lim))
    Yₜ .= zero(eltype(Yₜ))
    (; dt, params, rayleigh_sponge) = p
    (; ᶜh_tot) = p.precomputed
    (; ᶠu³, ᶜu, ᶜK, ᶜp) = p.precomputed
    (; ᶜf³, ᶠf¹²) = p.precomputed
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶜJ = Fields.local_geometry_field(Y.c).J
    grav = FT(CAP.grav(params))
    ᶜuₕ = Y.c.uₕ
    ᶠu₃ = Y.f.u₃
    ᶜρ = Y.c.ρ

    @. Yₜ.c.ρ -= wdivₕ(ᶜρ * ᶜu)
    @. Yₜ.c.ρe_tot -= wdivₕ(ᶜρ * ᶜh_tot * ᶜu)
    @. Yₜ.c.uₕ -= C12(gradₕ(ᶜp) / ᶜρ + gradₕ(ᶜK + Φ(grav, ᶜz)))

    ᶜω³ = p.scratch.ᶜtemp_CT3
    ᶠω¹² = p.scratch.ᶠtemp_CT12

    point_type = eltype(Fields.coordinate_field(Y.c))
    if point_type <: Geometry.Abstract3DPoint
        @. ᶜω³ = curlₕ(ᶜuₕ)
    elseif point_type <: Geometry.Abstract2DPoint
        @. ᶜω³ = zero(ᶜω³)
    end

    @. ᶠω¹² = ᶠcurlᵥ(ᶜuₕ)
    @. ᶠω¹² += CT12(curlₕ(ᶠu₃))
    # Without the CT12(), the right-hand side would be a CT1 or CT2 in 2D space.

    ᶠω¹²′ = if isnothing(ᶠf¹²)
        ᶠω¹² # shallow atmosphere
    else
        @. lazy(ᶠf¹² + ᶠω¹²) # deep atmosphere
    end

    @. Yₜ.c.uₕ -=
        ᶜinterp(ᶠω¹²′ × (ᶠinterp(ᶜρ * ᶜJ) * ᶠu³)) / (ᶜρ * ᶜJ) +
        (ᶜf³ + ᶜω³) × CT12(ᶜu)
    @. Yₜ.f.u₃ -= ᶠω¹²′ × ᶠinterp(CT12(ᶜu)) + ᶠgradᵥ(ᶜK)

    Yₜ.c.uₕ .+= CA.rayleigh_sponge_tendency_uₕ(ᶜuₕ, rayleigh_sponge)

    return Yₜ
end

# This block:
@time if !@isdefined(integrator)
    FT = Float64;
    ᶜspace = ExtrudedCubedSphereSpace(
        FT;
        z_elem = 63,
        z_min = 0,
        z_max = 30000.0,
        radius = 6.371e6,
        h_elem = 30,
        n_quad_points = 4,
        staggering = CellCenter(),
    );
    ᶠspace = Spaces.face_space(ᶜspace);
    cnt = (; ρ = zero(FT), uₕ = zero(CA.C12{FT}), ρe_tot = zero(FT));
    Yc = Fields.fill(cnt, ᶜspace);
    Yf = Fields.fill((; u₃ = zero(CA.C3{FT})), ᶠspace);
    Y = Fields.FieldVector(; c = Yc, f = Yf);

    A = ImplicitEquationJacobian(
        Y;
        approximate_solve_iters = 2,
        transform_flag = false, # assumes use_transform returns false
    )

    implicit_func = SciMLBase.ODEFunction(
        implicit_tendency!;
        jac_prototype = A,
        Wfact = Wfact!, # assumes use_transform returns false
        tgrad = (∂Y∂t, Y, p, t) -> (∂Y∂t .= 0),
    )

    func = CTS.ClimaODEFunction(;
        T_exp_T_lim! = remaining_tendency!,
        T_imp! = implicit_func,
        # Can we just pass implicit_tendency! and jac_prototype etc.?
        lim! = (Y, p, t, ref_Y) -> nothing, # limiters_func!
        dss!,
        cache! = set_precomputed_quantities!,
        cache_imp! = set_precomputed_quantities!,
    )

    newtons_method = CTS.NewtonsMethod(; max_iters = 2)
    params = CA.ClimaAtmosParameters(FT)
    ᶠcoord = Fields.coordinate_field(ᶠspace);
    ᶜcoord = Fields.coordinate_field(ᶜspace);
    (; ᶜf³, ᶠf¹²) = CA.compute_coriolis(ᶜcoord, ᶠcoord, params);
    scratch = (;
        ᶜtemp_CT3 = Fields.Field(CT3{FT}, ᶜspace),
        ᶠtemp_CT12 = Fields.Field(CT12{FT}, ᶠspace),
    )
    precomputed = (;
        ᶜh_tot = Fields.Field(FT, ᶜspace),
        ᶠu³ = Fields.Field(CA.CT3{FT}, ᶠspace),
        ᶜf³,
        ᶠf¹²,
        ᶜp = Fields.Field(FT, ᶜspace),
        ᶜK = Fields.Field(FT, ᶜspace),
        ᶜts = Fields.Field(TD.PhaseDry{FT}, ᶜspace),
        ᶠu = Fields.Field(C123{FT}, ᶠspace),
        ᶜu = Fields.Field(C123{FT}, ᶜspace),
    )
    dt = FT(0.1)

    ghost_buffer =
        !CA.do_dss(axes(Y.c)) ? (;) :
        (; c = Spaces.create_dss_buffer(Y.c), f = Spaces.create_dss_buffer(Y.f))

    CTh = CA.CTh_vector_type(axes(Y.c))
    p = (;
        rayleigh_sponge = CA.RayleighSponge{FT}(;
            zd = params.zd_rayleigh,
            α_uₕ = params.alpha_rayleigh_uh,
            α_w = params.alpha_rayleigh_w,
        ),
        params,
        ∂ᶜK_∂ᶜuₕ = Fields.Field(DiagonalMatrixRow{Adjoint{FT, CTh{FT}}}, ᶜspace),
        ∂ᶜK_∂ᶠu₃ = Fields.Field(BidiagonalMatrixRow{Adjoint{FT, CT3{FT}}}, ᶜspace),
        ᶜadvection_matrix = Fields.Field(
            BidiagonalMatrixRow{Adjoint{FT, C3{FT}}},
            ᶜspace,
        ),
        ᶜtemp_scalar = Fields.Field(FT, ᶜspace),
        ᶠp_grad_matrix = Fields.Field(BidiagonalMatrixRow{C3{FT}}, ᶠspace),
        scratch,
        ghost_buffer,
        dt,
        precomputed,
    )
    ode_algo = CTS.IMEXAlgorithm(CTS.ARS343(), newtons_method)
    problem = SciMLBase.ODEProblem(func, Y, (FT(0), FT(1)), p)
    integrator = SciMLBase.init(problem, ode_algo; dt)
    Yₜ = similar(integrator.u);
end

function main!(integrator, Yₜ, n)
    for _ in 1:n
        # @time SciMLBase.step!(integrator)
        @time implicit_tendency!(Yₜ, integrator.u, integrator.p, integrator.t)
    end
    return nothing
end
if ClimaComms.device() isa ClimaComms.CUDADevice
    println(CUDA.@profile begin
        # SciMLBase.step!(integrator)
        implicit_tendency!(Yₜ, integrator.u, integrator.p, integrator.t)
        implicit_tendency!(Yₜ, integrator.u, integrator.p, integrator.t)
        implicit_tendency!(Yₜ, integrator.u, integrator.p, integrator.t)
        implicit_tendency!(Yₜ, integrator.u, integrator.p, integrator.t)
    end)
else
    @info "Compiling main loop"
    @time main!(integrator, Yₜ, 1)
    @info "Running main loop"
    @time main!(integrator, Yₜ, 10)
end

nothing