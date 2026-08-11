#=
KE-budget verification of the vector-invariant face sets
(docs/vi_kep_face_terms.md): with face_set = :kep the horizontal advective
KE production closes to roundoff on the flat sphere AND the terrain-warped
Hughes2023 grid, the velocity penalties are sign-definite sinks, and the
unstabilized core integrates stably; :kg violates the same budget by
orders of magnitude.
=#
import ClimaCore: Spaces, Fields, Geometry
import ClimaCore.Operators as _O
import LinearAlgebra

# Short spin-up with the unstabilized KEP core, then evaluate both budgets
# on the same final state (the :kg model is rebuilt on an identical grid and
# the state copied across at the parent-array level).
function budget_pair(; topography, nsteps)
    mk(face_set) = DG.BaroclinicWaveDG(;
        helem = 4,
        zelem = 10,
        dt = 60.0,
        t_end = nsteps * 60.0,
        topography,
        face_set,
        sponge_τ = Inf,
    )
    sim = DG.DGSimulation(mk(:kep))
    result = DG.run!(sim)
    Y = result.sol.u[end]
    @test !any(isnan, parent(Y.c))
    m_kep = sim.model
    m_kg = DG.DGModel(mk(:kg))
    Y_kg = DG.initial_state(m_kg)
    parent(Y_kg.c) .= parent(Y.c)
    parent(Y_kg.f) .= parent(Y.f)
    b_kep = DG.horizontal_ke_budget(Y, m_kep)
    b_kg = DG.horizontal_ke_budget(Y_kg, m_kg)
    Δh = Spaces.node_horizontal_length_scale(m_kep.spaces.horzspace)
    # roundoff anchor: KE × the fastest budget rate (acoustic crossing)
    P_ref = b_kep.KE * 350 / Δh
    KE0 = DG.horizontal_ke_budget(sim.Y₀, m_kep).KE
    return (; b_kep, b_kg, P_ref, KE0)
end

@testset "VI KEP budget: flat sphere" begin
    (; b_kep, b_kg, P_ref, KE0) = budget_pair(topography = :none, nsteps = 120)
    @info "flat" b_kep b_kg P_ref
    # exact budget: roundoff-level closure
    @test abs(b_kep.P_adv) < 1e-10 * P_ref
    # exact sign-definite dissipation
    @test b_kep.P_pen <= 0
    # the legacy set measurably violates the budget on the same state
    @test abs(b_kg.P_adv) > 10 * abs(b_kep.P_adv)
    # unstabilized stability sanity: KE bounded over the spin-up
    @test b_kep.KE < 1.05 * KE0
end

# Isolate the :es interface dissipation as (full :es) − (central-only)
# and contract with the entropy variable v = ∂S/∂ρe|ρ = −ρ/p.
central_only_scalars(normal, (y⁻,), (y⁺,)) =
    _O.vi_kep_scalars_flux(normal, normal, y⁻, y⁺)
function es_entropy_budget(Y, m)
    c = m.c
    (; Ic) = m.ops
    ρ = Y.c.ρ
    ρe = Y.c.ρe
    uₕ = Y.c.uₕ
    w = Y.f.w
    lg = Fields.local_geometry_field(m.spaces.hv_center_space)
    uv = Geometry.UVVector.(uₕ)
    Kf = @. (
        LinearAlgebra.dot(DG.C123(uₕ), DG.CT123(uₕ)) +
        Ic(LinearAlgebra.dot(DG.C123(w), DG.CT123(w))) +
        2 * LinearAlgebra.dot(DG.CT123(uₕ), Ic(DG.C123(w)))
    ) / 2
    p = @. DG.pres_ρe(c, ρe, Kf, m.fields.ᶜΦ, ρ)
    λ_c = @. sqrt(LinearAlgebra.norm_sqr(uv)) + sqrt(c.γ * p / ρ)
    y = map(
        (a, b, cc, d, ee, f) ->
            (; ρ = a, ρe = b, p = cc, λ = d, uv = ee, e = f),
        ρ,
        ρe,
        p,
        λ_c,
        uv,
        ρe ./ ρ,
    )
    dy_es = map(_ -> (ρ = 0.0, ρe = 0.0), ρ)
    dy_c = map(_ -> (ρ = 0.0, ρe = 0.0), ρ)
    _O.add_numerical_flux_internal!(
        _O.VIESInterfaceScalars(c.γ - 1),
        dy_es,
        y,
    )
    _O.add_numerical_flux_internal!(central_only_scalars, dy_c, y)
    dρe_diss = @. (dy_es.ρe - dy_c.ρe) / lg.WJ
    mass_diff = maximum(abs, parent(@. dy_es.ρ - dy_c.ρ))
    P_S = sum(@. -(ρ / p) * dρe_diss)
    return (; P_S, mass_diff)
end

@testset "VI ES interface: entropy dissipation (docs §9)" begin
    prob = DG.BaroclinicWaveDG(;
        helem = 4,
        zelem = 10,
        dt = 60.0,
        t_end = 120 * 60.0,
        face_set = :es,
        sponge_τ = Inf,
    )
    sim = DG.DGSimulation(prob)
    Y = DG.run!(sim).sol.u[end]
    m = sim.model
    @test !any(isnan, parent(Y.c))
    (; P_S, mass_diff) = es_entropy_budget(Y, m)
    K = DG.horizontal_ke_budget(Y, m)
    @info "ES entropy production" P_S K.P_adv
    @test mass_diff == 0                 # mass flux stays central
    @test P_S < 0                        # provable entropy dissipation
    # kinetic-energy budget unchanged for :es (dissipation is KE-inert)
    P_ref =
        K.KE * 350 /
        Spaces.node_horizontal_length_scale(m.spaces.horzspace)
    @test abs(K.P_adv) < 1e-10 * P_ref
end

@testset "VI KEP budget: Hughes2023 double mountain (warped grid)" begin
    # The budget is a STATE FUNCTIONAL identity — it must close for any
    # state on the warped grid. Spin up on the flat sphere (stable
    # unstabilized) to develop element-boundary jumps, then transplant the
    # state onto the double-mountain grid (identical layout) and evaluate
    # both budgets there. (Coarse time-stepping over the barely-resolved
    # analytic mountains crashes on the separate well-balancedness
    # residual — docs/vi_kep_face_terms.md distinguishes the two — so the
    # transplant is also the cleaner metric-transparency test.)
    mk(face_set, topography) = DG.BaroclinicWaveDG(;
        helem = 4,
        zelem = 10,
        dt = 60.0,
        t_end = 120 * 60.0,
        topography,
        face_set,
        sponge_τ = Inf,
    )
    sim = DG.DGSimulation(mk(:kep, :none))
    Y = DG.run!(sim).sol.u[end]
    budgets = map((:kep, :kg)) do fs
        m = DG.DGModel(mk(fs, :hughes2023))
        Yw = DG.initial_state(m)
        parent(Yw.c) .= parent(Y.c)
        parent(Yw.f) .= parent(Y.f)
        DG.horizontal_ke_budget(Yw, m)
    end
    b_kep, b_kg = budgets
    P_ref =
        b_kep.KE * 350 /
        Spaces.node_horizontal_length_scale(sim.model.spaces.horzspace)
    @info "hughes2023 (transplanted state)" b_kep b_kg P_ref
    # metric-transparency: exactness must survive the terrain warp
    @test abs(b_kep.P_adv) < 1e-10 * P_ref
    @test b_kep.P_pen <= 0
    @test abs(b_kg.P_adv) > 10 * abs(b_kep.P_adv)
end
