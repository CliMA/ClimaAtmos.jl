# The ClimaAtmos-Setups-sourced IC must equal the examples' own formulas
# under the parity TOML (they are the same Ullrich et al. expressions;
# only the constants source differs).
@testset "IC parity: Setups ≡ formulas" begin
    for perturb in (false, true)
        mk(ic_source) = DG.initial_state_fddg(
            DG.DGModel(
                BaroclinicWaveFDDG(;
                    helem = 2,
                    zelem = 5,
                    dt = 60.0,
                    perturb,
                    ic_source,
                ),
            ),
        )
        Y_s = mk(:setups)
        Y_f = mk(:formulas)
        for name in (:ρ, :ρe, :ρu1, :ρu2, :ρu3)
            a = parent(getproperty(Y_s.c, name))
            b = parent(getproperty(Y_f.c, name))
            scale = maximum(abs, b)
            @test maximum(abs, a .- b) ≤ 1e-11 * max(scale, 1)
        end
    end
end
