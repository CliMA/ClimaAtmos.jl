#=
FDDG `wb_gravity` knob (well-balanced geopotential fluctuation in the
horizontal volume kernel, Waruszewski et al. 2022 Eq. 76 — ClimaCore
`kennedy_gruber_gravity_cartesian_flux`): on flat grids the fluctuation is
identically zero (Φ level-constant), so tendencies are BITWISE unchanged;
over terrain it activates and stays finite. The operator-level exactness
and cross-term attribution live in ClimaCore
test/Operators/spectralelement/wb_gravity_flux.jl.
=#

@testset "FDDG wb_gravity: flat no-op + terrain smoke" begin
    rhs(wb, topo) = begin
        prob = DG.BaroclinicWaveFDDG(;
            helem = 2,
            zelem = 5,
            dt = 60.0,
            topography = topo,
            wb_gravity = wb,
            perturb = false,
        )
        sim = DG.DGSimulation(prob)
        dY = similar(sim.Y₀)
        DG.rhs_fddg!(dY, sim.Y₀, sim.model, 0.0)
        dY
    end
    # flat sphere: the fluctuation is exactly zero — bitwise identical
    d0 = rhs(false, :none)
    d1 = rhs(true, :none)
    for name in (:ρ, :ρe, :ρu1, :ρu2, :ρu3)
        @test parent(getproperty(d1.c, name)) ==
              parent(getproperty(d0.c, name))
    end
    # terrain: activates (momentum differs), stays finite
    t0 = rhs(false, :hughes2023)
    t1 = rhs(true, :hughes2023)
    @test !any(isnan, parent(t1.c))
    @test parent(t1.c.ρu1) != parent(t0.c.ρu1)
    @test parent(t1.c.ρ) == parent(t0.c.ρ)   # mass flux untouched
end
