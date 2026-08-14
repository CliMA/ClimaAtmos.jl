#=
:es2 face set (VIES2InterfaceScalars): :es plus acoustic-selective Roe
dissipation on ([[p′]], [[uₙ]]) with the CONTACT wave kept central.

Testset 1 (flux-level, pure algebra): consistency (zero dissipation at
equal states), contact invariance (a pure density jump — Δp′ = Δuₙ = 0 —
receives bitwise NO mass dissipation, the property Rusanov/full-Roe
violate), and down-gradient acoustic transport.

Testset 2 (well-balance): the U₀ = 0 Agnesi rest state must keep its
roundoff dw with :es2 — p′ ≈ 0 there by construction of ᶜp_ref, so the
acoustic channel must not disturb the balance.

Testset 3 (stability smoke): short unstabilized :es2 integration over the
double mountain — no NaN, bounded KE.
=#

import ClimaCore.Operators as _O2
import ClimaCore: Geometry as _G2

@testset ":es2 flux level" begin
    fn = _O2.VIES2InterfaceScalars(0.4)
    fn_es = _O2.VIESInterfaceScalars(0.4)
    central(n, ya, yb) = _O2.vi_kep_scalars_flux(n, n, ya, yb)
    mk(; ρ, p, un, p′ = p) = (;
        ρ,
        p,
        p′,
        e = 2.5 * p / ρ + 1000.0,
        λ = sqrt(1.4 * p / ρ) + abs(un),
        uv = _G2.UVVector(un, 0.0),
    )
    n̂ = _G2.UVVector(1.0, 0.0)

    # consistency: equal states ⇒ dissipation-free (equals the central)
    y = mk(ρ = 1.0, p = 9e4, un = 10.0)
    F = fn(n̂, (y,), (y,))
    Fc = central(n̂, y, y)
    @test F.ρ == Fc.ρ
    @test F.ρe == Fc.ρe

    # contact invariance: pure [[ρ]] jump (Δp′ = Δuₙ = 0) ⇒ mass flux
    # BITWISE central (α± = 0); the ρe channel sees it only through the
    # provably-signed entropy term (shared with :es)
    y⁻ = mk(ρ = 1.0, p = 9e4, un = 10.0)
    y⁺ = mk(ρ = 1.2, p = 9e4, un = 10.0)
    F2 = fn(n̂, (y⁻,), (y⁺,))
    Fc2 = central(n̂, y⁻, y⁺)
    Fes = fn_es(n̂, (y⁻,), (y⁺,))
    @test F2.ρ == Fc2.ρ
    @test F2.ρe == Fes.ρe   # reduces exactly to :es when Δp′ = Δuₙ = 0

    # acoustic down-gradient: Δp′ > 0 at Δuₙ = 0 ⇒ mass dissipation
    # transports toward the low-p′ side (F.ρ decreases for n̂ toward +)
    y⁺p = mk(ρ = 1.0, p = 9.2e4, un = 10.0, p′ = 2e3)
    y⁻p = mk(ρ = 1.0, p = 9e4, un = 10.0, p′ = 0.0)
    F3 = fn(n̂, (y⁻p,), (y⁺p,))
    Fc3 = central(n̂, y⁻p, y⁺p)
    @test F3.ρ < Fc3.ρ
end

@testset ":es2 rest-state well-balance (Agnesi, U₀ = 0)" begin
    prob = DG.MountainWaveDG(;
        helem = 20,
        zelem = 20,
        xmax = 600e3,
        h₀ = 1000.0,
        a = 25e3,
        U₀ = 0.0,
        t_end = 3600.0,
        face_set = :es2,
        perturb = false,
    )
    sim = DG.DGSimulation(prob)
    dY = similar(sim.Y₀)
    DG.rhs_vi!(dY, sim.Y₀, sim.model, 0.0)
    # the acoustic channel must not disturb the discrete balance
    @test maximum(abs, parent(dY.f.w)) < 1e-8
end

@testset ":es2 stability smoke (double mountain, unstabilized)" begin
    mk(face_set) = DG.BaroclinicWaveDG(;
        helem = 4,
        zelem = 10,
        dt = 60.0,
        t_end = 120 * 60.0,
        topography = :hughes2023,
        face_set,
        sponge_τ = Inf,
    )
    sim = DG.DGSimulation(mk(:es2))
    Y = DG.run!(sim).sol.u[end]
    @test !any(isnan, parent(Y.c))
    KE0 = DG.horizontal_ke_budget(sim.Y₀, sim.model).KE
    KE = DG.horizontal_ke_budget(Y, sim.model).KE
    @info ":es2 smoke" KE0 KE
    @test KE < 1.05 * KE0
end
