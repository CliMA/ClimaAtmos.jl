# The closed-form pres_ρe (kept for the HEVI Jacobian's analytic
# derivatives) must be thermodynamically consistent with Thermodynamics.jl.
# NOTE the convention difference: our ρe uses e_int = cv_d(T − T_tri);
# TD uses e_int = cv_d(T − T_0) − R_d·T_0. Both give p = ρ R_d T for a
# self-consistently constructed state — that invariant is what we test.
import Thermodynamics as TD
@testset "pressure ≡ Thermodynamics" begin
    params = DG.dg_params(Float64, :parity)
    c = DG.DGConstants(params)
    tps = CAP.thermodynamics_params(params)
    @test CAP.T_0(params) == c.T_tri == 273.16
    for (ρ, T, K, Φ) in
        ((1.2, 287.0, 0.0, 0.0), (0.7, 250.0, 450.0, 9.8e4), (0.01, 210.0, 50.0, 2.9e5))
        # build ρe in OUR convention, invert with pres_ρe
        ρe = ρ * (c.cv_d * (T - c.T_tri) + K + Φ)
        p_ours = DG.pres_ρe(c, ρe, K, Φ, ρ)
        p_td = TD.air_pressure(tps, T, ρ)  # TD 1.x API (no state types)
        @test p_ours ≈ ρ * c.R_d * T rtol = 1e-14
        @test p_ours ≈ p_td rtol = 1e-14
    end
end
