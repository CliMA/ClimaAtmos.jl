#=
Agnesi mountain-wave slab (MountainWaveDG): the CPU-cheap terrain testbed.

Testset 1 (well-balance): U₀ = 0 over the ridge — any motion is spurious.
Verifies the exact isothermal discrete hydrostatics (vertical residual at
roundoff), the O(Δz²) convergence of the horizontal PGF residual, and that
one simulated hour stays quiet.

Testset 2 (linear wave, smoke): U₀ = 20 m/s develops a bounded wave
response with the KEP/ES face machinery on the periodic plane.
=#

import ClimaCore: Geometry

@testset "Mountain wave: U₀ = 0 well-balance" begin
    resid = map((20, 40)) do ze
        prob = DG.MountainWaveDG(;
            helem = 20,
            zelem = ze,
            xmax = 600e3,
            h₀ = 1000.0,
            a = 25e3,
            U₀ = 0.0,
            t_end = 3600.0,
            face_set = :es,
            perturb = false,
        )
        sim = DG.DGSimulation(prob)
        dY = similar(sim.Y₀)
        DG.rhs_vi!(dY, sim.Y₀, sim.model, 0.0)
        # vertical balance at roundoff (isothermal_discrete_hydrostatic!)
        @test maximum(abs, parent(dY.f.w)) < 1e-8
        r = maximum(abs, parent(Geometry.UVVector.(dY.c.uₕ)))
        if ze == 20
            Y = DG.run!(sim).sol.u[end]
            w_spur = maximum(abs, parent(Geometry.WVector.(Y.f.w)))
            @info "well-balance drift (1 h)" w_spur
            @test w_spur < 0.5   # m/s; measured ~0.03 (h₀ = 1 km ridge)
        end
        r
    end
    # horizontal PGF residual converges O(Δz²)
    @info "PGF residual" resid
    @test resid[1] / resid[2] > 3
end

@testset "Mountain wave: linear wave response (smoke)" begin
    prob = DG.MountainWaveDG(;
        helem = 20,
        zelem = 20,
        xmax = 600e3,
        h₀ = 250.0,
        a = 25e3,
        U₀ = 20.0,
        t_end = 3600.0,
        face_set = :es,
    )
    Y = DG.run!(DG.DGSimulation(prob)).sol.u[end]
    @test !any(isnan, parent(Y.c))
    w_max = maximum(abs, parent(Geometry.WVector.(Y.f.w)))
    # linear scale U₀h₀/a = 0.2 m/s; transient overshoot bounded
    @info "wave response (1 h)" w_max
    @test 0.01 < w_max < 2.0
end
