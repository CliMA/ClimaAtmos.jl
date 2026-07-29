# Under the parity TOML, the ClimaParams-derived constants must equal the
# ClimaCore examples' literals (which :parity-mode DGConstants carries).
@testset "constants parity" begin
    c_lit = DG.DGConstants{Float64}(; mode = :parity)
    params = DG.dg_params(Float64, :parity)
    c_par = DG.DGConstants(params)
    for f in fieldnames(typeof(c_lit))
        @test getfield(c_par, f) ≈ getfield(c_lit, f) rtol = 1e-14
    end
    # exact where the TOML was chosen to round-trip
    @test c_par.R_d == 287.0
    @test c_par.p_0 == 1.0e5
    @test c_par.grav == 9.80616
end
