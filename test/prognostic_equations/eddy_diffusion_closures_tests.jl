#=
Unit tests for the pointwise closures in eddy_diffusion_closures.jl:

1. `buoyancy_gradient_coefficients` + `blended_N²` — the cloud-fraction-
   independent factoring of the moist buoyancy gradient:
   ∂b/∂z = (Cθ_unsat + cf·ΔCθ)·∂θli/∂z + (Cq_unsat + cf·ΔCq)·∂qt/∂z.
   Tests: exact linearity in cf, the dry unsaturated limit against an
   independently computed g/θᵥ·∂θᵥ/∂z, equivalence with the
   `buoyancy_gradients` reference path, and physical sign conventions.

2. `interface_effective_N²` — N²_eff = N² + [(Δb)₊]²/(c_b·κ):
   resolved-limit consistency (correction quadratically small in Δz),
   exactness of the jump term, inertness for unstable/neutral gradients,
   and monotonicity in Δz.

3. `interface_entrainment_diffusivity` — K_e = γ·w_e·Δz:
   zero for A = 0, unstable faces, or absent turbulence; cubic decay
   K_e ∝ Δz³ in the resolved limit; resolution-independent entrainment
   flux coefficient K_e/Δz = γ·w_e at a sheet interface (fixed Δb,
   Δz ≫ ℓ_p); and the w_e ≤ A√κ efficiency bound.

4. `horizontal_filter_scale` + `resolvability_filter_scale` + the grid cap
   in `mixing_length_lopez_gomez_2020`: single columns are uncapped
   (Δx_h = Inf), extruded spaces return the spectral-element node scale,
   Δ_f = max(Δx_h, Δz) pointwise and as a field over a space, the cap
   binds exactly at Δ_f when the physical scales exceed it, and the mixing
   length is monotonically nondecreasing in Δ_f.
=#

using Test
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaParams as CP
import Thermodynamics as TD
import ClimaCore
import ClimaCore: Fields, Spaces
import ClimaCore.CommonSpaces

@testset "Eddy diffusion closures (pointwise)" begin
    for FT in (Float32, Float64)
        toml_dict = CP.create_toml_dict(FT)
        thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
        grav = TD.Parameters.grav(thp)

        @testset "FT = $FT" begin
            @testset "blended_N²: exact linearity in cloud fraction" begin
                # Warm, moist, partly cloudy state
                T, ρ = FT(288), FT(1.1)
                q_tot, q_liq, q_ice = FT(8e-3), FT(2e-4), FT(0)
                coeffs = CA.buoyancy_gradient_coefficients(
                    thp, T, ρ, q_tot, q_liq, q_ice,
                )
                ∂θli∂z, ∂qt∂z = FT(3e-3), FT(-1e-6)
                b0 = CA.blended_N²(coeffs, FT(0), ∂θli∂z, ∂qt∂z)
                b1 = CA.blended_N²(coeffs, FT(1), ∂θli∂z, ∂qt∂z)
                for cf in FT[0, 0.25, 0.5, 0.75, 1]
                    b = CA.blended_N²(coeffs, cf, ∂θli∂z, ∂qt∂z)
                    @test b ≈ (1 - cf) * b0 + cf * b1 rtol = 4 * eps(FT)
                end
            end

            @testset "blended_N²: dry unsaturated limit = g/θᵥ · ∂θᵥ/∂z" begin
                # Dry air: θli = θ = θᵥ, so with ∂qt∂z = 0 and cf = 0 the
                # closure must reduce to the classic dry buoyancy gradient.
                T, ρ = FT(280), FT(1.0)
                coeffs = CA.buoyancy_gradient_coefficients(
                    thp, T, ρ, FT(0), FT(0), FT(0),
                )
                θv = TD.virtual_pottemp(thp, T, ρ, FT(0), FT(0), FT(0))
                ∂θ∂z = FT(4e-3)
                @test CA.blended_N²(coeffs, FT(0), ∂θ∂z, FT(0)) ≈
                      grav / θv * ∂θ∂z rtol = sqrt(eps(FT))
                # Dry air admits no saturated branch difference in ∂b/∂θli
                # through θᵥ: the two θ-coefficients stay within a few percent.
                @test coeffs.Cθ_unsat > 0
            end

            @testset "blended_N² ∘ coefficients ≡ buoyancy_gradients" begin
                # The factored path must reproduce the reference closure for
                # arbitrary states and cloud fractions.
                states = (
                    (
                        T = FT(288),
                        ρ = FT(1.1),
                        q_tot = FT(8e-3),
                        q_liq = FT(2e-4),
                        q_ice = FT(0),
                    ),
                    (
                        T = FT(298),
                        ρ = FT(1.15),
                        q_tot = FT(16e-3),
                        q_liq = FT(0),
                        q_ice = FT(0),
                    ),
                    (
                        T = FT(250),
                        ρ = FT(0.7),
                        q_tot = FT(1e-3),
                        q_liq = FT(0),
                        q_ice = FT(5e-5),
                    ),
                )
                for s in states, cf in FT[0, 0.3, 1]
                    ∂θli∂z, ∂qt∂z = FT(2.5e-3), FT(-8e-7)
                    coeffs = CA.buoyancy_gradient_coefficients(
                        thp, s.T, s.ρ, s.q_tot, s.q_liq, s.q_ice,
                    )
                    ref = CA.buoyancy_gradients(
                        CA.BuoyGradMean(),
                        thp,
                        CA.EnvBuoyGradVars(
                            s.T, s.ρ, s.q_tot, s.q_liq, s.q_ice, cf,
                            (; ∂θli∂z, ∂qt∂z),
                        ),
                    )
                    @test CA.blended_N²(coeffs, cf, ∂θli∂z, ∂qt∂z) ≈ ref rtol =
                        10 * eps(FT)
                end
            end

            @testset "blended_N²: physical sign conventions" begin
                T, ρ = FT(288), FT(1.1)
                coeffs = CA.buoyancy_gradient_coefficients(
                    thp, T, ρ, FT(8e-3), FT(2e-4), FT(0),
                )
                # Stable θli profile, uniform moisture: N² > 0 for any cf.
                for cf in FT[0, 0.5, 1]
                    @test CA.blended_N²(coeffs, cf, FT(3e-3), FT(0)) > 0
                end
                # Warm saturated air: the moisture coefficient exceeds its
                # unsaturated (virtual-effect) counterpart because latent-heat
                # release dominates (ΔCq > 0), while the θli coefficient is
                # reduced because part of a θli perturbation is buffered by
                # condensation/evaporation (ΔCθ < 0).
                @test coeffs.ΔCq > 0
                @test coeffs.ΔCθ < 0
            end

            @testset "interface_effective_N²" begin
                c_b = FT(0.4)
                N², κ = FT(1e-4), FT(0.3)
                # Exact jump term for a stable gradient
                for Δz in FT[10, 50, 200]
                    Δb = N² * Δz
                    @test CA.interface_effective_N²(N², Δz, κ, c_b) ≈
                          N² + Δb^2 / (c_b * κ) rtol = 4 * eps(FT)
                end
                # Resolved limit: correction relatively O((Δz/l_N)²) ≪ 1
                l_N = sqrt(c_b * κ) / sqrt(N²)   # ≈ 35 m here
                Δz = l_N / 100
                rel = CA.interface_effective_N²(N², Δz, κ, c_b) / N² - 1
                @test 0 < rel < FT(2e-4)
                # Unstable and neutral gradients pass through unchanged
                @test CA.interface_effective_N²(-N², Δz, κ, c_b) == -N²
                @test CA.interface_effective_N²(FT(0), Δz, κ, c_b) == 0
                # Monotonically increasing in Δz for stable gradients
                vals = [
                    CA.interface_effective_N²(N², dz, κ, c_b) for
                    dz in FT[5, 25, 100, 400]
                ]
                @test issorted(vals)
                # κ → 0 stays finite (eps guard)
                @test isfinite(CA.interface_effective_N²(N², FT(50), FT(0), c_b))
            end

            @testset "interface_entrainment_diffusivity" begin
                c_b, A = FT(0.4), FT(0.4)
                κ, ℓ_e = FT(0.3), FT(100)
                # Inert cases
                @test CA.interface_entrainment_diffusivity(
                    FT(1e-4), FT(50), κ, ℓ_e, c_b, FT(0),
                ) == 0
                @test CA.interface_entrainment_diffusivity(
                    FT(-1e-4), FT(50), κ, ℓ_e, c_b, A,
                ) == 0
                @test CA.interface_entrainment_diffusivity(
                    FT(0), FT(50), κ, ℓ_e, c_b, A,
                ) == 0
                # Absent turbulence: K_e ≈ 0 (κ enters as κ^{3/2}/Δb via the
                # Ri_b branch, so the eps-guarded value is negligible)
                @test CA.interface_entrainment_diffusivity(
                    FT(1e-4), FT(50), FT(0), ℓ_e, c_b, A,
                ) < 10 * eps(FT)
                # Resolved limit: γ ∝ Δz² and Ri_b < 1, so K_e ∝ Δz³
                N² = FT(1e-4)
                K(dz) = CA.interface_entrainment_diffusivity(
                    N², dz, κ, ℓ_e, c_b, A,
                )
                @test K(FT(2)) / K(FT(1)) ≈ 8 rtol = FT(0.05)
                @test K(FT(4)) / K(FT(2)) ≈ 8 rtol = FT(0.05)
                # Sheet interface: fixed jump Δb, Δz ≫ ℓ_p = c_b κ / Δb.
                # The entrainment flux coefficient K_e/Δz = γ·w_e approaches a
                # Δz-independent limit.
                Δb = FT(0.25)                     # DYCOMS-like jump [m/s²]
                ℓ_p = c_b * κ / Δb                # ≈ 0.5 m
                flux_coeff(dz) =
                    CA.interface_entrainment_diffusivity(
                        Δb / dz, dz, κ, ℓ_e, c_b, A,
                    ) / dz
                f1, f2 = flux_coeff(FT(100)), flux_coeff(FT(200))
                @test f1 ≈ f2 rtol = FT(0.01)
                # ... and equals the closure's w_e (γ → 1, Ri_b > 1 here)
                Ri_b = ℓ_e * Δb / κ
                w_e = A * sqrt(κ) / max(Ri_b, FT(1))
                @test f2 ≈ w_e rtol = FT(0.01)
                # Efficiency bound: w_e ≤ A√κ ⟹ K_e ≤ A√κ·Δz
                for dz in FT[10, 100, 500], n² in FT[1e-5, 1e-3, 1e-1]
                    @test CA.interface_entrainment_diffusivity(
                        n², dz, κ, ℓ_e, c_b, A,
                    ) <= A * sqrt(κ) * dz * (1 + 4 * eps(FT))
                end
            end

            @testset "horizontal_filter_scale: space dispatch" begin
                # Single column: no horizontal discretization — the filter
                # scale is unbounded (cap inert).
                col_space = CommonSpaces.ColumnSpace(
                    FT;
                    z_min = 0,
                    z_max = 1000,
                    z_elem = 10,
                    staggering = CommonSpaces.CellCenter(),
                )
                @test CA.horizontal_filter_scale(col_space) == FT(Inf)
                @test CA.horizontal_filter_scale(col_space) isa FT
                # Extruded space: the spectral-element node length scale of
                # the horizontal space.
                slice_space = CommonSpaces.SliceXZSpace(
                    FT;
                    z_min = 0,
                    z_max = 1000,
                    z_elem = 5,
                    x_min = 0,
                    x_max = 10_000,
                    x_elem = 4,
                    periodic_x = true,
                    n_quad_points = 4,
                    staggering = CommonSpaces.CellCenter(),
                )
                Δx_h = CA.horizontal_filter_scale(slice_space)
                @test Δx_h isa FT
                @test Δx_h ≈ FT(
                    Spaces.node_horizontal_length_scale(
                        Spaces.horizontal_space(slice_space),
                    ),
                )
                # 10 km / 4 elements / 3 unique nodes per element
                @test Δx_h ≈ FT(10_000) / 4 / 3 rtol = sqrt(eps(FT))

                # resolvability_filter_scale: pointwise max ...
                @test CA.resolvability_filter_scale(FT(100), FT(50)) ==
                      FT(100)
                @test CA.resolvability_filter_scale(FT(10), FT(50)) == FT(50)
                @test CA.resolvability_filter_scale(FT(Inf), FT(50)) ==
                      FT(Inf)
                # ... and as a field over a space: Inf for the column;
                # max(Δx_h, Δz) pointwise for the extruded slice, where
                # Δx_h ≈ 833 m > Δz = 200 m.
                col_Δf = Base.materialize(
                    CA.resolvability_filter_scale(col_space),
                )
                @test all(==(FT(Inf)), Array(parent(col_Δf)))
                slice_Δf = Base.materialize(
                    CA.resolvability_filter_scale(slice_space),
                )
                @test maximum(slice_Δf) ≈ Δx_h rtol = sqrt(eps(FT))
                @test minimum(slice_Δf) ≈ Δx_h rtol = sqrt(eps(FT))
                # Fine horizontal grid: Δz = 200 m governs.
                fine_space = CommonSpaces.SliceXZSpace(
                    FT;
                    z_min = 0,
                    z_max = 1000,
                    z_elem = 5,
                    x_min = 0,
                    x_max = 1200,
                    x_elem = 4,
                    periodic_x = true,
                    n_quad_points = 4,
                    staggering = CommonSpaces.CellCenter(),
                )
                fine_Δf = Base.materialize(
                    CA.resolvability_filter_scale(fine_space),
                )
                @test maximum(fine_Δf) ≈ FT(200) rtol = sqrt(eps(FT))
            end

            @testset "mixing length: resolvability filter-scale cap" begin
                params = CA.ClimaAtmosParameters(FT)
                turbconv_params = CAP.turbconv_params(params)
                sf_params = CAP.surface_fluxes_params(params)
                vkc = CAP.von_karman_const(params)
                # Neutral, well-mixed inputs chosen so all physical scales
                # (l_W, l_TKE, l_N = l_z) are several hundred meters or more:
                # the filter-scale cap, where finite, is the binding limit.
                z, z_sfc = FT(1000), FT(0)
                ustar, sfc_tke, tke = FT(0.3), FT(0.09), FT(1)
                N²_eff = N²_prod = FT(0)
                obukhov_length = FT(1e8)     # neutral surface layer
                strain, Pr = FT(1e-8), FT(1)
                ml(Δ_f) = CA.mixing_length_lopez_gomez_2020(
                    turbconv_params,
                    sf_params,
                    vkc,
                    ustar,
                    z,
                    z_sfc,
                    Δ_f,
                    sfc_tke,
                    N²_eff,
                    N²_prod,
                    tke,
                    obukhov_length,
                    strain,
                    Pr,
                    CA.SmoothMinimumBlending(),
                )
                # Uncapped (single-column) limit: purely physical, bounded by
                # the wall distance, and identical to any cap larger than the
                # physical scales.
                l_inf = ml(FT(Inf))
                @test isfinite(l_inf.master)
                @test FT(100) < l_inf.master <= z - z_sfc
                @test ml(FT(1e7)).master == l_inf.master
                @test l_inf.l_grid == FT(Inf)
                # Where the physical scales exceed Δ_f, the cap binds
                # exactly.
                for Δ_f in FT[50, 100]
                    l_capped = ml(Δ_f)
                    @test l_capped.l_grid == Δ_f
                    @test l_capped.master == Δ_f
                end
                # Monotonically nondecreasing in Δ_f, and the cap only ever
                # reduces the mixing length.
                masters = [ml(Δf).master for Δf in FT[50, 100, 300, 1e6, Inf]]
                @test issorted(masters)
                @test all(<=(l_inf.master), masters)
            end
        end
    end
end
