# Vector-invariant core: state construction, RHS evaluation (including the
# tendency cutoff filter — the path broken by the DataLayouts migration
# until the Nv-generic CPU tensor_product! was restored), and HEVI split.
@testset "vector-invariant core" begin
    m = DG.DGModel(
        BaroclinicWaveDG(; helem = 2, zelem = 5, dt = 60.0, perturb = true),
    )
    Y = DG.initial_state(m)
    @test all(isfinite, parent(Y.c))
    dY = similar(Y)
    DG.rhs_vi!(dY, Y, m, 0.0)   # exercises filter (filter_Nc = npoly)
    @test all(isfinite, parent(dY.c))
    @test all(isfinite, parent(dY.f))
    # HEVI split: rhs − imp − rem = F(imp) − imp (filter convention);
    # exactly zero for dρ at t = 0 since the implicit mass flux is 0 (w = 0)
    dY2 = similar(Y)
    dY3 = similar(Y)
    DG.implicit_tendency_vi!(dY2, Y, m, 0.0)
    DG.remaining_tendency_vi!(dY3, Y, m, 0.0)
    @test maximum(
        abs,
        parent(dY.c.ρ) .- parent(dY2.c.ρ) .- parent(dY3.c.ρ),
    ) == 0
end
