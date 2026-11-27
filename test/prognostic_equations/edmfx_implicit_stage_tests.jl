#=
Unit tests of the PrognosticEDMFX implicit stage on a stretched Bomex column
with an ascending, buoyant draft:

- Newton predictor (`initialize_implicit_stage_problem!`): the analytic `u₃`
  solve satisfies its stage equation for the physical velocity on a stretched
  grid and never returns a downward velocity; the `ρa` sweep stays in
  `[0, ρʲ a_max]` from a stage-entering ρa outside those bounds; the
  draft-scalar sweep keeps a uniform scalar uniform and stays non-negative;
  the grid-mean warm start keeps the environment.
- Jacobian (`ManualSparseJacobian`) against centred finite differences of
  `implicit_tendency!`: the blocks that are exact by design match to
  round-off, the diagonal barriers only add damping, the SGS mass-flux blocks
  match to within 1%, the `u₃` rows have no buoyancy blocks, and the
  stable-draft buoyancy feedback in the `u₃` block is the Schur complement
  built from the residual's operators.
- Newton iterations in a stably stratified draft, from the bottom of a polar
  AMIP column where the iterates swung between rest and launch without the
  stable-draft feedback: the draft velocity converges monotonically.
- Split upwinding of the draft's own velocity and the horizontal wind's
  component across sloping levels, for a hard-coded component of either
  sign.

The entrainment that scales with `|w|` and the buoyancy detrainment are turned
off, because the Jacobian treats their rates as frozen.
=#
using Test
using LinearAlgebra: I
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
const TD = CA.TD
import ClimaCore: Fields, MatrixFields
import ClimaCore.MatrixFields: @name

include("../test_helpers.jl")

column(f) = vec(Array(parent(Base.materialize(f))))

# Buoyancy detrainment depends on `w` through its time scale, which the
# Jacobian freezes.
override_toml = joinpath(mktempdir(), "edmfx_implicit_stage_overrides.toml")
write(
    override_toml,
    """
    [detr_buoy_coeff]
    value = 0
    type = "float"
    """,
)
config = CA.AtmosConfig(
    Dict{String, Any}(
        "initial_condition" => "Bomex",
        "FLOAT_TYPE" => "Float64",
        "config" => "column",
        "turbconv" => "prognostic_edmfx",
        "edmfx_entr_model" => "Generalized",
        "edmfx_detr_model" => "Generalized",
        "edmfx_sgs_mass_flux" => true,
        "edmfx_sgs_diffusive_flux" => false,
        "edmfx_nh_pressure" => false,
        "implicit_microphysics" => false,
        "microphysics_model" => "1M",
        "z_max" => 3000.0,
        "z_elem" => 20,
        "z_stretch" => true,
        "dz_bottom" => 50.0,
        "dz_top" => 400.0,
        "dt" => "10secs",
        "t_end" => "10secs",
        "toml" => [override_toml],
        "output_default_diagnostics" => false,
    );
    job_id = "edmfx_implicit_stage_test",
)
(; Y, p, simulation) = generate_test_simulation(config)
t = simulation.integrator.t
FT = eltype(Y)
dtγ = FT(p.dtγ)
turbconv_params = CAP.turbconv_params(p.params)
a_max = CAP.max_area(turbconv_params)
ᶜz = Fields.coordinate_field(Y.c).z
ᶠz = Fields.coordinate_field(Y.f).z
ᶠlg = Fields.local_geometry_field(Y.f)
z_top = FT(3000)
Y_initial = copy(Y)

"""
    set_ascending_draft!(Y, p, t)

An ascending, buoyant draft with a non-uniform area, and the cache at that
state. The `|w|`-scaled entrainment rate, which the Jacobian freezes, is
zeroed after the cache update.
"""
function set_ascending_draft!(Y, p, t)
    Y .= Y_initial
    sgsʲ = Y.c.sgsʲs.:(1)
    @. sgsʲ.ρa = Y.c.ρ * FT(0.08) * (1 + FT(0.3) * sin(FT(π) * ᶜz / z_top))
    @. Y.f.sgsʲs.:(1).u₃ =
        CA.C3(CA.WVec(FT(1.5) * sin(FT(π) * ᶠz / z_top)^FT(0.7)), ᶠlg)
    CA.set_precomputed_quantities!(Y, p, t)
    @. sgsʲ.mse += FT(800) * (1 + ᶜz / z_top)
    @. sgsʲ.q_tot += FT(5e-4)
    CA.set_precomputed_quantities!(Y, p, t)
    fill!(parent(p.precomputed.ᶜentr_vel_scaleʲs), 0)
    return nothing
end

physical_w(Y) = column(@. CA.w_component(CA.WVec(Y.f.sgsʲs.:(1).u₃, ᶠlg)))

@testset "Newton predictor" begin
    @testset "u₃ solve on a stretched grid" begin
        set_ascending_draft!(Y, p, t)
        # No buoyancy and no drag: the stage equation of the physical velocity
        # on face i is (w − w_old)/dtγ + (w² − w[i−1]²)/(2 Δz[i]) = 0.
        fill!(parent(p.precomputed.ᶜρ_diffʲs), 0)
        fill!(parent(p.precomputed.ᶜentr_nonvel_rateʲs), 0)
        fill!(parent(p.precomputed.ᶜturb_entrʲs), 0)
        w_old = physical_w(Y)
        CA.predict_sgs_u₃!(Y, p, dtγ)
        w = physical_w(Y)
        ᶠdz = column(Fields.Δz_field(axes(Y.f)))
        @test ᶠdz[2] != ᶠdz[end - 1]
        residual = [
            (w[i] - w_old[i]) / dtγ + (w[i]^2 - w[i - 1]^2) / (2 * ᶠdz[i]) for
            i in 2:(length(w) - 1)
        ]
        @test maximum(abs, residual) <= 1e-10 * maximum(abs, w_old) / dtγ
        @test all(>(0), w[2:(end - 1)])

        # Negative buoyancy and a downward stage-entering velocity never give
        # a downward velocity.
        set_ascending_draft!(Y, p, t)
        @. p.precomputed.ᶜρ_diffʲs.:(1) = FT(0.05)
        @. Y.f.sgsʲs.:(1).u₃ = -Y.f.sgsʲs.:(1).u₃
        CA.predict_sgs_u₃!(Y, p, FT(100))
        @test all(>=(0), physical_w(Y))
    end

    @testset "ρa sweep stays in [0, ρʲ a_max]" begin
        set_ascending_draft!(Y, p, t)
        # Stage-entering ρa below 0 and above ρʲ a_max in places, which the
        # stage initializer passes to the sweep unchanged, and a long step.
        ᶜρʲ = p.precomputed.ᶜρʲs.:(1)
        @. Y.c.sgsʲs.:(1).ρa =
            ᶜρʲ * a_max * (FT(0.5) + FT(0.6) * sin(FT(4π) * ᶜz / z_top))
        @test minimum(column(Y.c.sgsʲs.:(1).ρa)) < 0
        @test maximum(column(Y.c.sgsʲs.:(1).ρa) ./ column(ᶜρʲ)) > a_max
        CA.predict_sgs_u₃!(Y, p, FT(1000))
        CA.predict_sgs_ρa!(Y, p, FT(1000))
        ρa = column(Y.c.sgsʲs.:(1).ρa)
        @test all(>=(0), ρa)
        @test all(ρa .<= a_max .* column(ᶜρʲ) .* (1 + 1e-14))
    end

    @testset "draft-scalar sweep" begin
        set_ascending_draft!(Y, p, t)
        sgsʲ = Y.c.sgsʲs.:(1)
        # A uniform scalar, with matching surface values, stays uniform.
        q_tot_value = FT(0.012)
        mse_value = FT(3.1e5)
        @. sgsʲ.q_tot = q_tot_value
        @. sgsʲ.mse = mse_value
        @. p.precomputed.sfc_q_tot_buoyantʲs.:(1) = q_tot_value
        @. p.precomputed.sfc_mse_buoyantʲs.:(1) = mse_value
        # The stage-entering drafts, which the stage initializer saves.
        @. p.precomputed.ᶜsgsʲs_stage_snapshot = Y.c.sgsʲs
        CA.predict_sgs_u₃!(Y, p, dtγ)
        CA.predict_sgs_tracers!(Y, p, dtγ)
        @test column(sgsʲ.q_tot) ≈ fill(q_tot_value, length(column(sgsʲ.q_tot))) rtol =
            1e-14
        @test column(sgsʲ.mse) ≈ fill(mse_value, length(column(sgsʲ.mse))) rtol =
            1e-14

        # Negative stage-entering values are floored, so a long step stays
        # non-negative and below the largest floored or surface value.
        set_ascending_draft!(Y, p, t)
        @. sgsʲ.q_tot = FT(0.01) * sin(FT(6π) * ᶜz / z_top)
        @. sgsʲ.q_lcl = FT(1e-4) * sin(FT(5π) * ᶜz / z_top)
        q_tot_sfc = column(p.precomputed.sfc_q_tot_buoyantʲs.:(1))[1]
        @. p.precomputed.ᶜsgsʲs_stage_snapshot = Y.c.sgsʲs
        CA.predict_sgs_u₃!(Y, p, FT(1000))
        CA.predict_sgs_tracers!(Y, p, FT(1000))
        @test all(>=(0), column(sgsʲ.q_tot))
        @test all(<=(max(FT(0.01), q_tot_sfc)), column(sgsʲ.q_tot))
        @test all(>=(0), column(sgsʲ.q_lcl))
        @test all(<=(FT(1e-4)), column(sgsʲ.q_lcl))
    end

    @testset "warm start keeps the environment" begin
        set_ascending_draft!(Y, p, t)
        tc = p.atmos.turbconv_model
        (; ᶜh_tot, ᶜK) = p.precomputed
        p.scratch.precip_stage_idx[] = p.scratch.n_imp_stages
        CA.set_implicit_precomputed_quantities_core!(Y, p, t)
        env_q_tot() = column(CA.ᶜspecific_env_value(@name(q_tot), Y, p))
        # `mse⁰` from `ρe_tot = ρa⁰ mse⁰ + Σⱼ ρaʲ mseʲ + (ρK − p)` at the
        # stage-entering `ρK − p`, through the shift of `h_tot`.
        ρe_tot_old = copy(Y.c.ρe_tot)
        env_mse() = column(
            @. CA.specific_env_mse(
                Y.c.ρ, Y.c.sgsʲs, ᶜh_tot + (Y.c.ρe_tot - ρe_tot_old) / Y.c.ρ,
                ᶜK, CA.ρa⁰(Y.c.ρ, Y.c.sgsʲs, tc), tc,
            )
        )
        (q_tot⁰_old, mse⁰_old) = (env_q_tot(), env_mse())
        ρa_old = column(Y.c.sgsʲs.:(1).ρa)
        CA.initialize_implicit_stage_problem!(Y, p, dtγ)
        @test column(Y.c.sgsʲs.:(1).ρa) != ρa_old
        @test env_q_tot() ≈ q_tot⁰_old rtol = 1e-12
        @test env_mse() ≈ mse⁰_old rtol = 1e-12
    end
end

"""
    dense_block(block)

The first column of a Jacobian block as a dense matrix of numbers.
"""
function dense_block(block)
    B = MatrixFields.column_field2array(Base.materialize(block))
    (data, l, u) = (getfield(B, :data), getfield(B, :l), getfield(B, :u))
    A = zeros(size(B))
    for i in axes(B, 1), j in axes(B, 2)
        -l <= j - i <= u || continue
        x = data[u + i - j + 1, j]
        A[i, j] = x isa Number ? x : x[1]
    end
    return A
end

"""
    finite_difference_blocks(Y, p, t, dtγ, row_names, col_name)

`dtγ ∂T/∂Y − δ I` from centred differences of `implicit_tendency!`, with the
implicit cache refreshed at each perturbed state, for the rows `row_names` and
the column `col_name`.
"""
function finite_difference_blocks(Y, p, t, dtγ, row_names, col_name)
    function tendency(Y′)
        CA.set_implicit_precomputed_quantities!(Y′, p, t)
        Yₜ = similar(Y′)
        CA.implicit_tendency!(Yₜ, Y′, p, t)
        return Yₜ
    end
    x = parent(MatrixFields.get_field(Y, col_name))
    ε = 1e-6 * maximum(abs, x)
    columns = map(1:length(x)) do k
        Y⁺ = copy(Y)
        Y⁻ = copy(Y)
        parent(MatrixFields.get_field(Y⁺, col_name))[k] += ε
        parent(MatrixFields.get_field(Y⁻, col_name))[k] -= ε
        (tendency(Y⁺) .- tendency(Y⁻)) ./ (2ε)
    end
    CA.set_implicit_precomputed_quantities!(Y, p, t)
    return map(row_names) do row_name
        fd = dtγ .* hcat(
            (column(MatrixFields.get_field(c, row_name)) for c in columns)...,
        )
        row_name == col_name ? fd - I : fd
    end
end

relative_error(A, B) = maximum(abs, A - B) / maximum(abs, B)
off_diagonal(A) = [i == j ? 0.0 : A[i, j] for i in axes(A, 1), j in axes(A, 2)]
diagonal(A) = [A[i, i] for i in 1:minimum(size(A))]

"""
    stable_draft_u₃_block(Y, p, dtγ)

The stable-draft buoyancy feedback of the `u₃` block as a dense matrix, built
from the column's metric and draft profiles: the Schur complement
`-∂(u₃ₜ)/∂mse (mse block)⁻¹ ∂(mseₜ)/∂u₃` over the cells whose mse rises from
the cell below, with the face buoyancy the mean of the two adjacent cells,
each cell's upwinded mse driven by the velocity at its bottom face, and the
mse block replaced by its diagonal `-(1 + C)` at the Jacobian velocity.
Boundary rows are zero.
"""
function stable_draft_u₃_block(Y, p, dtγ)
    thermo_params = CAP.thermodynamics_params(p.params)
    α_b = CAP.pressure_normalmode_buoy_coeff1(CAP.turbconv_params(p.params))
    g = CAP.grav(p.params)
    ᶠJ = column(ᶠlg.J)
    ᶜJ = column(Fields.local_geometry_field(Y.c).J)
    ᶠg³³ = column(CA.g³³_field(axes(Y.f)))
    ᶠ∂Φ = column(p.core.ᶠgradᵥ_ᶜΦ.components.data.:1)
    mse = column(Y.c.sgsʲs.:(1).mse)
    β = column(
        @. CA.draft_buoyancy_per_mse(
            thermo_params, p.precomputed.ᶜTʲs.:(1), p.precomputed.ᶜρʲs.:(1),
            Y.c.sgsʲs.:(1).q_lcl, Y.c.sgsʲs.:(1).q_icl,
        )
    )
    ᶠu³_jac = column(
        @. CA.u³_component(
            CA.jacobian_updraft_u³(p.precomputed.ᶠwʲ_maxs.:(1), ᶠlg),
        )
    )
    n_faces = length(ᶠJ)
    S = zeros(n_faces, n_faces)
    # Cell c lies between faces c (bottom) and c + 1 (top).
    for c in 2:(n_faces - 1)
        Δmse = mse[c] - mse[c - 1]
        Δmse > 0 || continue
        C = dtγ * ᶠJ[c] * ᶠu³_jac[c] / ᶜJ[c]
        ∂mseₜ_∂u₃ = -1 * ᶠJ[c] * ᶠg³³[c] * Δmse / ᶜJ[c]
        for i in (c, c + 1)
            1 < i < n_faces || continue
            ∂u₃ₜ_∂mse = (1 - α_b) * ᶠ∂Φ[i] / g * β[c] / 2
            # -(dtγ ∂u₃ₜ/∂mse) (-(1 + C))⁻¹ (dtγ ∂mseₜ/∂u₃), negative: damping
            S[i, c] += dtγ^2 * ∂u₃ₜ_∂mse * ∂mseₜ_∂u₃ / (1 + C)
        end
    end
    return S
end

@testset "Jacobian against finite differences" begin
    set_ascending_draft!(Y, p, t)
    alg = CA.ManualSparseJacobian(; approximate_solve_iters = 1)
    cache = CA.jacobian_cache(alg, Y, p.atmos)
    CA.update_jacobian!(alg, cache, Y, p, dtγ, t)
    matrix = cache.matrix.matrix

    # The Jacobian advects the updraft equations with a velocity at least as
    # large as the iterate's.
    w = max.(physical_w(Y), 0)
    @test all(column(p.precomputed.ᶠwʲ_maxs.:(1)) .>= w .* (1 - 1e-12))

    u₃ = @name(f.sgsʲs.:(1).u₃)
    ρa = @name(c.sgsʲs.:(1).ρa)
    mse = @name(c.sgsʲs.:(1).mse)
    q_tot = @name(c.sgsʲs.:(1).q_tot)
    (ρe_tot, ρq_tot) = (@name(c.ρe_tot), @name(c.ρq_tot))

    # The `u₃` rows have no buoyancy blocks.
    for χ in (mse, q_tot, @name(c.sgsʲs.:(1).q_lcl))
        @test !((u₃, χ) in keys(matrix))
    end

    # Blocks with respect to `u₃`, at the iterate's velocity.
    (∂ρa, ∂mse, ∂q_tot, ∂ρe_tot, ∂ρq_tot) = finite_difference_blocks(
        Y, p, t, dtγ, (ρa, mse, q_tot, ρe_tot, ρq_tot), u₃,
    )
    @test relative_error(dense_block(matrix[ρa, u₃]), ∂ρa) < 1e-8
    @test relative_error(dense_block(matrix[q_tot, u₃]), ∂q_tot) < 1e-8
    # Approximate blocks, which agree to within 1%: the `(mse, u₃)` block and
    # the SGS mass-flux blocks of the grid mean (whose `ρe_tot` rows omit the
    # dependence of the draft kinetic energy on `u₃`). The tolerance catches
    # a wrong upwinding, sign, or area factor.
    @test relative_error(dense_block(matrix[mse, u₃]), ∂mse) < 1e-2
    @test relative_error(dense_block(matrix[ρe_tot, u₃]), ∂ρe_tot) < 1e-2
    @test relative_error(dense_block(matrix[ρq_tot, u₃]), ∂ρq_tot) < 1e-2

    # Blocks with respect to `ρa`.
    (∂ρa_ρa, ∂ρe_tot_ρa, ∂ρq_tot_ρa) =
        finite_difference_blocks(Y, p, t, dtγ, (ρa, ρe_tot, ρq_tot), ρa)
    @test relative_error(dense_block(matrix[ρe_tot, ρa]), ∂ρe_tot_ρa) < 2e-2
    @test relative_error(dense_block(matrix[ρq_tot, ρa]), ∂ρq_tot_ρa) < 2e-2

    # With the Jacobian velocity set to the iterate's, the advection blocks
    # are exact, and the `ρa` and `u₃` diagonals differ only by the barriers,
    # which add damping.
    @. p.precomputed.ᶠwʲ_maxs.:(1) =
        max(CA.w_component(CA.WVec(p.precomputed.ᶠu³ʲs.:(1), ᶠlg)), 0)
    CA.update_sgs_advection_jacobian!(matrix, Y, p, dtγ)
    CA.update_sgs_u₃_and_ρa_jacobian!(matrix, Y, p, dtγ)
    flag = cache.derivative_flags.diffusion_flag
    CA.update_sgs_diffusion_and_entr_detr_jacobian!(matrix, Y, p, dtγ, flag)
    CA.update_sgs_boundary_condition_jacobian!(matrix, Y, p, dtγ)
    CA.update_sgs_massflux_jacobian!(matrix, Y, p, dtγ, flag)
    CA.update_implicit_microphysics_jacobian!(matrix, Y, p, dtγ)
    for (χ, ∂χ) in zip(
        (mse, q_tot),
        (
            finite_difference_blocks(Y, p, t, dtγ, (mse,), mse)[1],
            finite_difference_blocks(Y, p, t, dtγ, (q_tot,), q_tot)[1],
        ),
    )
        @test relative_error(dense_block(matrix[χ, χ]), ∂χ) < 1e-8
    end
    (∂u₃_u₃,) = finite_difference_blocks(Y, p, t, dtγ, (u₃,), u₃)
    # The `u₃` block also carries the stable-draft buoyancy feedback, on its
    # diagonal and sub-diagonal, which the residual's `u₃` rows do not have.
    S = stable_draft_u₃_block(Y, p, dtγ)
    @test maximum(abs, off_diagonal(S)) > 0
    for (A, ∂) in (
        (dense_block(matrix[u₃, u₃]) - S, ∂u₃_u₃),
        (dense_block(matrix[ρa, ρa]), ∂ρa_ρa),
    )
        @test maximum(abs, off_diagonal(A - ∂)) < 1e-8 * maximum(abs, ∂)
        @test all(diagonal(A) .<= diagonal(∂) .+ 1e-8 * maximum(abs, ∂))
    end
end

# The bottom six levels of a polar column of an AMIP run (77.1°S, 96.8°E, on
# the plateau at 3375 m) at the start of an implicit stage in which Newton
# swung the draft between rest and launch when the `u₃` Jacobian had no
# stable-draft buoyancy feedback. A cold, buoyant draft fills 71% of the
# bottom cell under an empty draft at the environment's mse. Levels are
# 27 m apart in the run; covariant `u₃ = w Δz` there.
const antarctic_column = (;
    z = 3374.8955 .+ (0:5) .* 27.14,
    Δz = 27.14,
    ρ = [1.0222586, 0.95928276, 0.94837064, 0.9404487, 0.9352885, 0.9291215],
    ρe_tot = [-85952.54, -71237.766, -68843.38, -67021.55, -66094.37, -65032.562],
    ρq_tot = [
        2.2089414e-5,
        3.812567e-5,
        4.7362708e-5,
        6.587729e-5,
        8.0273436e-5,
        8.900241e-5,
    ],
    ρq_icl = [
        2.4200295e-7,
        2.6959617e-8,
        1.5735985e-8,
        4.7282813e-9,
        3.6291803e-9,
        1.8913426e-9,
    ],
    ρq_sno = [
        1.8366343e-7,
        1.7553617e-7,
        1.8939065e-7,
        2.0387488e-7,
        2.1911976e-7,
        2.3351357e-7,
    ],
    u₃ = [-0.015822621, 0.061286204, 0.18181342, 0.20834102, 0.2009222, 0.1908955],
    ρaʲ = [0.7091001, -5.2312016e-6, -5.032332e-8, 8.178056e-6, 4.2154068e-7, 3.0367034e-6],
    mseʲ = [-19997.133, -7751.8994, -5312.0435, -3932.1763, -3026.249, -2239.4858],
    q_totʲ = [
        2.3736031e-5,
        3.9742376e-5,
        4.996303e-5,
        6.766877e-5,
        8.488129e-5,
        9.483113e-5,
    ],
    q_iclʲ = [
        1.7818928e-7,
        2.8266626e-8,
        1.7038184e-8,
        1.3801608e-8,
        6.8660166e-9,
        2.183741e-9,
    ],
    q_snoʲ = [
        1.7976971e-7,
        1.8302141e-7,
        1.9977924e-7,
        1.7659326e-7,
        7.987332e-8,
        7.711134e-8,
    ],
    u₃ʲ = [-0.015668003, 0.050501637, 0.16599698, 0.22402912, 0.23867992, 0.23289326],
    # The cached `ᶜρ_diffʲs` of the run at the start of the stage.
    ρ_diffʲ = [
        -0.005073372,
        0.001225489,
        2.762087e-6,
        0.0010400531,
        8.956912e-5,
        0.0001760792,
    ],
)

"""
    set_antarctic_draft!(Y, p, t)

The state of `antarctic_column` on the bottom six levels of the test column,
and the cache at that state. The grid mean keeps the run's temperature and
moisture, continues above with the lapse rate of the top two levels, and has
the density of discrete hydrostatic balance; the run's horizontal wind is not
known and is set to zero. The draft keeps the run's area, moisture, and
velocity (`w = u₃/Δz`), and its mse is set so that its density difference
matches the run's cached `ᶜρ_diffʲs`.
"""
function set_antarctic_draft!(Y, p, t)
    Y .= Y_initial
    c = antarctic_column
    thp = CAP.thermodynamics_params(p.params)
    g = CAP.grav(p.params)
    cp_d = TD.Parameters.cp_d(thp)
    zc = column(ᶜz)
    n = length(zc)
    q_tot_a = c.ρq_tot ./ c.ρ
    q_ice_a = (c.ρq_icl .+ c.ρq_sno) ./ c.ρ
    T_a = map(1:6) do k
        TD.air_temperature(thp, c.ρe_tot[k] / c.ρ[k] - g * c.z[k], q_tot_a[k], 0, q_ice_a[k])
    end
    lapse_rate = (T_a[6] - T_a[5]) / c.Δz
    T = [k <= 6 ? T_a[k] : T_a[6] + lapse_rate * (zc[k] - zc[6]) for k in 1:n]
    q_tot = [q_tot_a[min(k, 6)] for k in 1:n]
    q_icl = [k <= 6 ? c.ρq_icl[k] / c.ρ[k] : 0.0 for k in 1:n]
    q_sno = [c.ρq_sno[min(k, 6)] / c.ρ[min(k, 6)] for k in 1:n]
    R_m = [TD.gas_constant_air(thp, q_tot[k], 0, q_icl[k] + q_sno[k]) for k in 1:n]
    ρ = similar(zc)
    ρ[1] = c.ρ[1]
    for k in 2:n
        ρ[k] = ρ[k - 1]
        for _ in 1:20
            p_k =
                ρ[k - 1] * R_m[k - 1] * T[k - 1] -
                g * (ρ[k - 1] + ρ[k]) / 2 * (zc[k] - zc[k - 1])
            ρ[k] = p_k / (R_m[k] * T[k])
        end
    end
    e_int = [TD.internal_energy(thp, T[k], q_tot[k], 0, q_icl[k] + q_sno[k]) for k in 1:n]
    parent(Y.c.ρ) .= ρ
    parent(Y.c.ρe_tot) .= ρ .* (e_int .+ g .* zc)
    parent(Y.c.ρq_tot) .= ρ .* q_tot
    parent(Y.c.ρq_lcl) .= 0
    parent(Y.c.ρq_icl) .= ρ .* q_icl
    parent(Y.c.ρq_rai) .= 0
    parent(Y.c.ρq_sno) .= ρ .* q_sno
    parent(Y.c.uₕ) .= 0
    face_w(u₃) = [1 < f <= 6 ? u₃[f] / c.Δz : 0.0 for f in 1:(n + 1)]
    ᶠw = similar(Y.f, FT)
    parent(ᶠw) .= face_w(c.u₃)
    @. Y.f.u₃ = CA.C3(CA.WVec(ᶠw), ᶠlg)
    parent(ᶠw) .= face_w(c.u₃ʲ)
    @. Y.f.sgsʲs.:(1).u₃ = CA.C3(CA.WVec(ᶠw), ᶠlg)
    sgsʲ = Y.c.sgsʲs.:(1)
    h = [TD.enthalpy(thp, T[k], q_tot[k], 0, q_icl[k] + q_sno[k]) for k in 1:n]
    parent(sgsʲ.ρa) .= [k <= 6 ? c.ρaʲ[k] : c.ρaʲ[6] / c.ρ[6] * ρ[k] for k in 1:n]
    parent(sgsʲ.mse) .=
        [k <= 6 ? c.mseʲ[k] - g * c.z[k] + g * zc[k] : h[k] + g * zc[k] for k in 1:n]
    parent(sgsʲ.q_tot) .= [k <= 6 ? c.q_totʲ[k] : q_tot[k] for k in 1:n]
    parent(sgsʲ.q_lcl) .= 0
    parent(sgsʲ.q_icl) .= [k <= 6 ? c.q_iclʲ[k] : 0.0 for k in 1:n]
    parent(sgsʲ.q_rai) .= 0
    parent(sgsʲ.q_sno) .= [k <= 6 ? c.q_snoʲ[k] : q_sno[k] for k in 1:n]
    CA.set_precomputed_quantities!(Y, p, t)
    for _ in 1:4
        ρ_diffʲ = column(p.precomputed.ᶜρ_diffʲs.:(1))[1:6]
        Tʲ = column(p.precomputed.ᶜTʲs.:(1))[1:6]
        parent(sgsʲ.mse)[1:6] .+= (ρ_diffʲ .- c.ρ_diffʲ) .* cp_d .* Tʲ
        CA.set_precomputed_quantities!(Y, p, t)
    end
    return nothing
end

"""
    newton_draft_velocities(Y, p, t, dtγ, n_iters)

The draft's physical velocity at each face after the Newton predictor and
after each of `n_iters` Newton iterations of the implicit stage equation
`U = U₀ + dtγ T(U)` from `U₀ = Y`, as ClimaTimeSteppers takes them: the
implicit cache at each iterate, the residual `U₀ + dtγ T(U) − U`, the Jacobian
at the iterate, and `U −= W⁻¹ R`. Mutates `Y` into the last iterate.
"""
function newton_draft_velocities(Y, p, t, dtγ, n_iters)
    alg = CA.ManualSparseJacobian(; approximate_solve_iters = 2)
    cache = CA.jacobian_cache(alg, Y, p.atmos)
    Y₀ = copy(Y)
    R = similar(Y)
    ΔY = similar(Y)
    CA.initialize_implicit_stage_problem!(Y, p, dtγ)
    velocities = [physical_w(Y)]
    for _ in 1:n_iters
        CA.set_implicit_precomputed_quantities!(Y, p, t)
        CA.implicit_tendency!(R, Y, p, t)
        @. R = Y₀ + dtγ * R - Y
        CA.update_jacobian!(alg, cache, Y, p, dtγ, t)
        CA.invert_jacobian!(alg, cache, ΔY, R)
        Y .-= ΔY
        push!(velocities, physical_w(Y))
    end
    return velocities
end

@testset "Newton iterations in a stably stratified draft" begin
    # Moving the draft lifts cold bottom-cell air into the empty cell above,
    # which turns its buoyancy negative; at rest that cell is warm and the
    # draft is pushed up again. Without the stable-draft feedback in the `u₃`
    # Jacobian, the iterates swing between rest and launch and do not
    # converge.
    set_antarctic_draft!(Y, p, t)
    @test column(p.precomputed.ᶜρ_diffʲs.:(1))[1:6] ≈ antarctic_column.ρ_diffʲ rtol = 1e-3
    w = map(v -> v[2], newton_draft_velocities(Y, p, t, FT(53), 6))
    increments = diff(w)
    @test w[1] > 0.1
    @test all(Δ -> Δ < 0, increments)
    @test abs(increments[end]) < 1e-2 * abs(increments[1])
    @test all(>(0), column(Y.c.ρ))
end

@testset "Split upwinding of the draft velocity and the horizontal wind" begin
    set_ascending_draft!(Y, p, t)
    ᶠu₃ʲ = Y.f.sgsʲs.:(1).u₃
    ᶜmseʲ = Y.c.sgsʲs.:(1).mse
    ᶜρaʲ = Y.c.sgsʲs.:(1).ρa
    # The horizontal wind's component across sloping levels, which a flat
    # column does not have: upward in the lower half of the column, downward
    # and faster than the draft near the top, zero on the boundary faces.
    ᶠuₕ³ = @. CA.CT3(CA.WVec(FT(2) * sin(FT(2π) * ᶠz / z_top)), ᶠlg)
    u_draft = column(@. CA.u³_component(CA.CT3(ᶠu₃ʲ)))
    u_wind = column(@. CA.u³_component(ᶠuₕ³))
    interior = 2:(length(u_wind) - 1)
    @test all(>=(0), u_draft)
    @test any(>(0), u_wind[interior]) && any(<(0), u_wind[interior])
    @test any(u_draft[interior] .+ u_wind[interior] .< 0)

    # Each part is upwinded from its own side: the draft's from below, the
    # wind's from whichever side it comes from.
    ᶠJ = column(ᶠlg.J)
    ᶜJ = column(Fields.local_geometry_field(Y.c).J)
    mse = column(ᶜmseʲ)
    ρa = column(ᶜρaʲ)
    upwind(v, χ, f) = v > 0 ? χ[f - 1] : χ[f]
    ᶠρa_flux = CA.updraft_ρa_flux(ᶠu₃ʲ, ᶠuₕ³, ᶜρaʲ)
    ρa_flux = column(@. CA.u³_component(ᶠρa_flux))
    expected_ρa_flux = [
        u_draft[f] * ρa[f - 1] + u_wind[f] * upwind(u_wind[f], ρa, f) for
        f in interior
    ]
    @test ρa_flux[interior] ≈ expected_ρa_flux rtol = 1e-12

    # Advective form `-(∂(J u³ χ) - χ ∂(J u³)) / J` of each part in cell c,
    # with no flux through the boundary faces.
    face_term(v, f, c) =
        1 < f < length(v) ? ᶠJ[f] * v[f] * (upwind(v[f], mse, f) - mse[c]) :
        zero(eltype(v))
    advection = column(
        CA.updraft_vertical_advection(ᶠu₃ʲ, ᶠuₕ³, ᶜmseʲ, Val(:first_order)),
    )
    expected_advection = map(eachindex(mse)) do c
        sum(
            -(face_term(v, c + 1, c) - face_term(v, c, c)) / ᶜJ[c] for
            v in (u_draft, u_wind)
        )
    end
    @test advection ≈ expected_advection rtol = 1e-10
end
