#=
FDDG Waruszewski volume flux — the moisture-0M-mpi replacement for the removed
standalone `wb_gravity` KG-gravity operator. Waruszewski (2022) is entropy-
conservative and machine-precision well-balanced over terrain, carrying gravity
as the ½ρ̂⟦φ⟧ fluctuation; it requires the conservative-perturbation momentum
pressure (pgf = :conservative_pert). Here: it runs finite over terrain and
produces a momentum tendency distinct from the plain KG volume flux. Operator-
level exactness lives in ClimaCore test/Operators/spectralelement.
The legacy `wb_gravity` knob is now rejected by `validate`.
=#

@testset "FDDG waruszewski: terrain smoke + differs from KG" begin
    rhs(vf) = begin
        prob = DG.BaroclinicWaveFDDG(;
            helem = 2,
            zelem = 5,
            dt = 60.0,
            topography = :hughes2023,
            volume_flux = vf,
            pgf = vf == :kg ? :conservative : :conservative_pert,
            interface_flux = :roe,
            perturb = false,
        )
        sim = DG.DGSimulation(prob)
        dY = similar(sim.Y₀)
        DG.rhs_fddg!(dY, sim.Y₀, sim.model, 0.0)
        dY
    end
    dkg = rhs(:kg)
    dw = rhs(:waruszewski)
    @test !any(isnan, parent(dw.c))
    # the entropy-conservative flux changes the momentum tendency over terrain
    @test parent(dw.c.ρu1) != parent(dkg.c.ρu1)
end

@testset "FDDG wb_gravity knob is deprecated" begin
    @test_throws ErrorException DG.validate(
        DG.BaroclinicWaveFDDG(; wb_gravity = true),
    )
end
