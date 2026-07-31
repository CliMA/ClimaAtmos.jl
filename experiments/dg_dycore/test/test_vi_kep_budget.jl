#=
KE-budget verification of the vector-invariant face sets
(docs/vi_kep_face_terms.md): with face_set = :kep the horizontal advective
KE production closes to roundoff — on the flat sphere AND on the
terrain-warped Hughes & Jablonowski (2023) double-mountain grid (the ledger
is metric-transparent) — the velocity penalties are sign-definite sinks,
and the unstabilized core (κ₄ = 0, filter_Nc = 0, the :kep defaults)
integrates stably. The legacy :kg set, evaluated on the SAME state,
violates the ledger at a measurably larger level.
=#
import ClimaCore: Spaces

# Short spin-up with the unstabilized KEP core, then evaluate both ledgers
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
    # roundoff anchor: KE × the fastest ledger rate (acoustic crossing)
    P_ref = b_kep.KE * 350 / Δh
    KE0 = DG.horizontal_ke_budget(sim.Y₀, m_kep).KE
    return (; b_kep, b_kg, P_ref, KE0)
end

@testset "VI KEP budget: flat sphere" begin
    (; b_kep, b_kg, P_ref, KE0) = budget_pair(topography = :none, nsteps = 120)
    @info "flat" b_kep b_kg P_ref
    # exact ledger: roundoff-level closure
    @test abs(b_kep.P_adv) < 1e-10 * P_ref
    # exact sign-definite dissipation
    @test b_kep.P_pen <= 0
    # the legacy set measurably violates the ledger on the same state
    @test abs(b_kg.P_adv) > 10 * abs(b_kep.P_adv)
    # unstabilized stability sanity: KE bounded over the spin-up
    @test b_kep.KE < 1.05 * KE0
end

@testset "VI KEP budget: Hughes2023 double mountain (warped grid)" begin
    # The ledger is a STATE FUNCTIONAL identity — it must close for any
    # state on the warped grid. Spin up on the flat sphere (stable
    # unstabilized) to develop element-boundary jumps, then transplant the
    # state onto the double-mountain grid (identical layout) and evaluate
    # both ledgers there. (Coarse time-stepping over the barely-resolved
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
