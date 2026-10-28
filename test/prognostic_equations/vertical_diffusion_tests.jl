# Verify the vertical diffusion tendency (Eqs. 30 and 33, Yatunin et al. 2026):
# - ρq_tot diffuses on q_tot_eff = q_tot - q_rai - q_sno (rain/snow excluded)
# - Suspended cloud species (ρq_lcl, ρq_icl, ρn_lcl, ρn_ice, and with 2MP3 the
#   rime mass and volume) diffuse down their own vertical gradients with ρ K_h
# - ρq_rai, ρq_sno, ρn_rai receive no diffusion tendency
# - Passive (non-microphysics) grid-scale tracers diffuse with the full K_h
# - Eq. (30) enthalpy-species identity with the heat capacity of the diffusing
#   mixture, the reversible 0M moist adiabat as a fixed point, and Kirchhoff
#   T_0 reference-state invariance hold.
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaParams as CP
import ClimaCore: Fields, Geometry
const TD = CA.TD

include("../test_helpers.jl")

@testset "Vertical diffusion tendency (Yatunin et al. 2026 Eq. 33)" begin
    FT = Float64
    params = CA.ClimaAtmosParameters(CP.create_toml_dict(FT))
    thermo_params = CAP.thermodynamics_params(params)

    (; cent_space, face_space) = get_cartesian_spaces(; FT)
    coords = Fields.coordinate_field(cent_space)
    ᶜz = coords.z

    tracer_names = (
        :ρq_tot,
        :ρq_lcl,
        :ρq_icl,
        :ρq_rai,
        :ρq_sno,
        :ρn_lcl,
        :ρn_rai,
        :ρq_gas_A,
    )
    NT = NamedTuple{
        (:ρ, :ρe_tot, tracer_names...),
        NTuple{2 + length(tracer_names), FT},
    }
    Yc = similar(coords, NT)
    @. Yc.ρ = FT(1.2)
    @. Yc.ρe_tot = Yc.ρ * FT(2.5e5)
    @. Yc.ρq_tot = Yc.ρ * (FT(1e-2) + FT(2e-3) * cos(ᶜz))
    @. Yc.ρq_lcl = Yc.ρ * (FT(2e-4) + FT(1e-4) * cos(ᶜz))
    @. Yc.ρq_icl = Yc.ρ * (FT(1e-4) + FT(5e-5) * cos(ᶜz))
    @. Yc.ρq_rai = Yc.ρ * (FT(1e-5) + FT(5e-6) * cos(ᶜz))
    @. Yc.ρq_sno = Yc.ρ * (FT(5e-6) + FT(2e-6) * cos(ᶜz))
    @. Yc.ρn_lcl = Yc.ρ * (FT(1e6) + FT(1e5) * cos(ᶜz))
    @. Yc.ρn_rai = Yc.ρ * (FT(1e3) + FT(1e2) * cos(ᶜz))
    @. Yc.ρq_gas_A = Yc.ρ * cos(ᶜz)
    Y = Fields.FieldVector(; c = Yc)

    ᶜu = @. Geometry.UVWVector(zero(ᶜz), zero(ᶜz), zero(ᶜz))
    ᶜp = @. FT(1e5) - FT(1e3) * ᶜz
    ᶜT = @. FT(280) - FT(20) * (ᶜz / FT(π))
    ᶜq_liq = @. Yc.ρq_lcl / Yc.ρ
    ᶜq_ice = @. Yc.ρq_icl / Yc.ρ
    ᶜq_tot_nonneg = @. Yc.ρq_tot / Yc.ρ
    p = (;
        atmos = (;
            vertical_diffusion = CA.DecayWithHeightDiffusion{FT}(;
                disable_momentum_vertical_diffusion = true,
                H = FT(1),
                D₀ = FT(1),
            ),
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
        ),
        params,
        precomputed = (; ᶜu, ᶜp, ᶜT, ᶜq_liq, ᶜq_ice, ᶜq_tot_nonneg),
        core = (; ᶜΦ = (@. CAP.grav(params) * ᶜz)),
        scratch = (;
            ᶜtemp_scalar = similar(ᶜz),
            ᶜtemp_scalar_2 = similar(ᶜz),
            ᶜtemp_scalar_3 = similar(ᶜz),
            ᶜtemp_scalar_4 = similar(ᶜz),
        ),
    )

    Yₜ = Fields.FieldVector(; c = zero(Yc))
    CA.vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, FT(0))

    # ρq_tot diffuses; mass conservation applies matching change to ρ.
    @test maximum(abs, parent(Yₜ.c.ρq_tot)) > 0
    @test parent(Yₜ.c.ρ) == parent(Yₜ.c.ρq_tot)

    # ρe_tot receives dry-static-energy + water-enthalpy contributions.
    @test maximum(abs, parent(Yₜ.c.ρe_tot)) > 0

    # Suspended cloud mass and number species diffuse down their own gradients.
    ᶜK_h = similar(ᶜz)
    ᶜK_h .= CA.ᶜcompute_eddy_diffusivity_coefficient(
        Yc.ρ,
        p.atmos.vertical_diffusion,
    )
    ᶠρK = similar(Fields.coordinate_field(face_space).z)
    @. ᶠρK = CA.ᶠinterp(Yc.ρ) / CA.ᶠinterp(1 / max(ᶜK_h, eps(FT)))
    ᶜexpected_lcl = similar(ᶜz)
    @. ᶜexpected_lcl = -CA.ᶜdiffdivᵥ(-(ᶠρK * CA.ᶠgradᵥ(Yc.ρq_lcl / Yc.ρ)))
    @test parent(Yₜ.c.ρq_lcl) ≈ parent(ᶜexpected_lcl) rtol = FT(1e-12)
    @test maximum(abs, parent(Yₜ.c.ρq_icl)) > 0
    @test maximum(abs, parent(Yₜ.c.ρn_lcl)) > 0

    # Sedimenting precipitation (rain, snow, rain number) does not diffuse.
    @test all(iszero, parent(Yₜ.c.ρq_rai))
    @test all(iszero, parent(Yₜ.c.ρq_sno))
    @test all(iszero, parent(Yₜ.c.ρn_rai))

    # Passive (non-microphysics) grid-scale tracer diffuses with full K_h.
    @test maximum(abs, parent(Yₜ.c.ρq_gas_A)) > 0

    # Eq. (30) identity (1M) — Yₜ.c.ρe_tot equals the sensible-heat and
    # geopotential diffusion, with the heat capacity of the diffusing mixture
    # (rain and snow counted at c_pd), plus the divergence of
    # Σ_μ (face-interpolated h_tot,μ) * F_qμ for vapor, liquid, and ice, where
    # F_qv = F_qt_eff - F_qlcl - F_qicl.
    ᶜq_tot_eff = @. (Yc.ρq_tot - Yc.ρq_rai - Yc.ρq_sno) / Yc.ρ
    ᶠF_qt = @. -(ᶠρK * CA.ᶠgradᵥ(ᶜq_tot_eff))
    ᶠF_ql = @. -(ᶠρK * CA.ᶠgradᵥ(Yc.ρq_lcl / Yc.ρ))
    ᶠF_qi = @. -(ᶠρK * CA.ᶠgradᵥ(Yc.ρq_icl / Yc.ρ))
    ᶠF_qv = @. ᶠF_qt - ᶠF_ql - ᶠF_qi
    ᶜh_tot_v = @. TD.enthalpy_vapor(thermo_params, ᶜT) + p.core.ᶜΦ
    ᶜh_tot_l = @. TD.enthalpy_liquid(thermo_params, ᶜT) + p.core.ᶜΦ
    ᶜh_tot_i = @. TD.enthalpy_ice(thermo_params, ᶜT) + p.core.ᶜΦ
    (; ᶜcp) = enthalpy_flux_coefficients(
        thermo_params,
        ᶜT,
        p.core.ᶜΦ,
        (ᶜq_tot_eff, (@. Yc.ρq_lcl / Yc.ρ), (@. Yc.ρq_icl / Yc.ρ)),
    )
    ᶜρe_totₜ_eq30 = similar(ᶜz)
    @. ᶜρe_totₜ_eq30 =
        -CA.ᶜdiffdivᵥ(
            -(ᶠρK * (CA.ᶠinterp(ᶜcp) * CA.ᶠgradᵥ(ᶜT) + CA.ᶠgradᵥ(p.core.ᶜΦ))) +
            CA.ᶠinterp(ᶜh_tot_v) * ᶠF_qv +
            CA.ᶠinterp(ᶜh_tot_l) * ᶠF_ql +
            CA.ᶠinterp(ᶜh_tot_i) * ᶠF_qi,
        )
    @test parent(Yₜ.c.ρe_tot) ≈ parent(ᶜρe_totₜ_eq30) rtol = FT(1e-12)

    # The sensible-heat flux carries the heat capacity of the diffusing
    # mixture: the tendency differs from the Eq. (33) form with ∇s_d by the
    # divergence of ρ K (c_p - c_pd) ∇T.
    ᶜs_d = @. TD.dry_static_energy(thermo_params, ᶜT, p.core.ᶜΦ)
    ᶜρe_totₜ_sd = similar(ᶜz)
    @. ᶜρe_totₜ_sd =
        -CA.ᶜdiffdivᵥ(
            -(ᶠρK * CA.ᶠgradᵥ(ᶜs_d)) +
            CA.ᶠinterp(ᶜh_tot_v) * ᶠF_qv +
            CA.ᶠinterp(ᶜh_tot_l) * ᶠF_ql +
            CA.ᶠinterp(ᶜh_tot_i) * ᶠF_qi,
        )
    cp_d = CAP.cp_d(params)
    @. ᶜρe_totₜ_sd -=
        CA.ᶜdiffdivᵥ(-(ᶠρK * CA.ᶠinterp(ᶜcp - cp_d) * CA.ᶠgradᵥ(ᶜT)))
    @test parent(Yₜ.c.ρe_tot) ≈ parent(ᶜρe_totₜ_sd) rtol = FT(1e-12)

    # Kirchhoff T_0 reference-state invariance — shifting T_0 by δT_0
    # (with Kirchhoff-adjusted e_int_v0 and e_int_i0) shifts ∂_t(ρe_tot) by
    # δh_water * ∂_t(ρq_tot) pointwise.
    δT_0 = FT(25.0)
    toml_shifted = CP.create_toml_dict(FT)
    toml_shifted.data["thermodynamics_temperature_reference"]["value"] += δT_0
    toml_shifted.data["latent_heat_vaporization_at_reference"]["value"] +=
        (CAP.cp_v(params) - CAP.cp_l(params)) * δT_0
    toml_shifted.data["latent_heat_sublimation_at_reference"]["value"] +=
        (CAP.cp_v(params) - CAP.cp_i(params)) * δT_0
    params_shifted = CA.ClimaAtmosParameters(toml_shifted)
    p_shifted = merge(p, (; params = params_shifted))
    Yₜ_shifted = Fields.FieldVector(; c = zero(Yc))
    CA.vertical_diffusion_boundary_layer_tendency!(
        Yₜ_shifted,
        Y,
        p_shifted,
        FT(0),
    )
    tp_0 = CAP.thermodynamics_params(params)
    tp_s = CAP.thermodynamics_params(params_shifted)
    # Shift in water enthalpy h_μ(T; T_0 + δT_0) - h_μ(T; T_0) is identical for
    # all three phases (vapor, liquid, ice) by Kirchhoff's law:
    δh_water =
        TD.enthalpy_vapor(tp_s, FT(280)) - TD.enthalpy_vapor(tp_0, FT(280))
    @test isapprox(
        TD.enthalpy_liquid(tp_s, FT(280)) - TD.enthalpy_liquid(tp_0, FT(280)),
        δh_water;
        rtol = FT(1e-12),
    )
    @test isapprox(
        TD.enthalpy_ice(tp_s, FT(280)) - TD.enthalpy_ice(tp_0, FT(280)),
        δh_water;
        rtol = FT(1e-12),
    )
    @. ᶜexpected_lcl = (Yₜ_shifted.c.ρe_tot - Yₜ.c.ρe_tot) - δh_water * Yₜ.c.ρq_tot
    @test maximum(abs, parent(ᶜexpected_lcl)) <=
          FT(1e-11) * maximum(abs, parent(Yₜ.c.ρe_tot))

    # 1M: with uniform q_tot_eff and vertically varying q_lcl,
    # Yₜ.c.ρq_tot == 0 while Yₜ.c.ρq_lcl ≈ +∇·(ρK∇q_lcl) (nonzero).
    @. Yc.ρq_rai = FT(0)
    @. Yc.ρq_sno = FT(0)
    @. Yc.ρq_icl = FT(0)
    @. Yc.ρq_tot = Yc.ρ * FT(1e-2)
    @. Yc.ρq_lcl = Yc.ρ * (FT(5e-4) + FT(2e-4) * cos(ᶜz))
    Yₜ_1m = Fields.FieldVector(; c = zero(Yc))
    CA.vertical_diffusion_boundary_layer_tendency!(Yₜ_1m, Y, p, FT(0))
    @test maximum(abs, parent(Yₜ_1m.c.ρq_tot)) <= FT(1e-15)
    @. ᶜexpected_lcl = -CA.ᶜdiffdivᵥ(-(ᶠρK * CA.ᶠgradᵥ(Yc.ρq_lcl / Yc.ρ)))
    @test maximum(abs, parent(ᶜexpected_lcl)) > 0
    @test parent(Yₜ_1m.c.ρq_lcl) ≈ parent(ᶜexpected_lcl) rtol = FT(1e-12)

    # 0M: a reversible saturated warm-cloud moist adiabat with uniform
    # MSE and q_tot is a fixed point: Yₜ.c.ρq_tot == 0, and Yₜ.c.ρe_tot
    # vanishes to the accuracy of the saturation adjustment, since the flux
    # bracket equals ∇(h + Φ) + (h_d + Φ) ∇q_tot and the discrete product
    # rule with face-interpolated coefficients is exact.
    q_tot_0 = FT(1.55e-2)
    mse_0 = TD.moist_static_energy(
        thermo_params,
        FT(293),
        FT(0),
        q_tot_0,
        FT(1.5e-3),
        FT(0),
    )
    @. p.core.ᶜΦ = FT(50) * CAP.grav(params) * ᶜz
    @. Yc.ρq_tot = Yc.ρ * q_tot_0
    @. ᶜq_tot_nonneg = q_tot_0
    for i in 1:length(parent(ᶜz))
        Φ_i = parent(p.core.ᶜΦ)[i]
        p_i = FT(9.4e4) * exp(-Φ_i / (CAP.R_d(params) * FT(290)))
        parent(ᶜp)[i] = p_i
        sa = TD.saturation_adjustment(
            thermo_params,
            TD.ph(),
            p_i,
            mse_0 - Φ_i,
            q_tot_0;
            maxiter = 40,
        )
        parent(ᶜT)[i] = sa.T
        parent(ᶜq_liq)[i] = sa.q_liq
        parent(ᶜq_ice)[i] = sa.q_ice
    end
    @test all(>(0), parent(ᶜq_liq))
    @test all(iszero, parent(ᶜq_ice))
    p_0m = merge(
        p,
        (;
            atmos = merge(
                p.atmos,
                (; microphysics_model = CA.EquilibriumMicrophysics0M()),
            ),
        ),
    )
    Yₜ_0m = Fields.FieldVector(; c = zero(Yc))
    CA.vertical_diffusion_boundary_layer_tendency!(Yₜ_0m, Y, p_0m, FT(0))
    ᶜsensible_only = similar(ᶜz)
    ᶜcp_0m = @. TD.cp_m(thermo_params, q_tot_0, ᶜq_liq, ᶜq_ice)
    @. ᶜsensible_only =
        -CA.ᶜdiffdivᵥ(
            -(
                ᶠρK * (
                    CA.ᶠinterp(ᶜcp_0m) * CA.ᶠgradᵥ(ᶜT) +
                    CA.ᶠgradᵥ(p.core.ᶜΦ)
                )
            ),
        )
    @test maximum(abs, parent(Yₜ_0m.c.ρq_tot)) <= FT(1e-15)
    @test maximum(abs, parent(ᶜsensible_only)) > 0
    @test maximum(abs, parent(Yₜ_0m.c.ρe_tot)) <=
          FT(1e-6) * maximum(abs, parent(ᶜsensible_only))
end

@testset "Vertical diffusion tendency with 2MP3 microphysics" begin
    # P3 takes the non-equilibrium path: rain and snow are excluded from the
    # diffusing water, and cloud ice carries its number and rime quantities,
    # which diffuse down their own gradients with the same ρ K_h.
    FT = Float64
    params = CA.ClimaAtmosParameters(CP.create_toml_dict(FT))
    thermo_params = CAP.thermodynamics_params(params)

    (; cent_space, face_space) = get_cartesian_spaces(; FT)
    coords = Fields.coordinate_field(cent_space)
    ᶜz = coords.z

    tracer_names = (
        :ρq_tot,
        :ρq_lcl,
        :ρq_icl,
        :ρq_rai,
        :ρq_sno,
        :ρn_lcl,
        :ρn_rai,
        :ρn_ice,
        :ρq_rim,
        :ρb_rim,
    )
    NT = NamedTuple{
        (:ρ, :ρe_tot, tracer_names...),
        NTuple{2 + length(tracer_names), FT},
    }
    Yc = similar(coords, NT)
    @. Yc.ρ = FT(1.2)
    @. Yc.ρe_tot = Yc.ρ * FT(2.5e5)
    @. Yc.ρq_tot = Yc.ρ * (FT(1e-2) + FT(2e-3) * cos(ᶜz))
    @. Yc.ρq_lcl = Yc.ρ * (FT(2e-4) + FT(1e-4) * cos(ᶜz))
    @. Yc.ρq_icl = Yc.ρ * (FT(1e-4) + FT(5e-5) * cos(ᶜz + FT(1)))
    @. Yc.ρq_rai = Yc.ρ * (FT(1e-5) + FT(5e-6) * cos(ᶜz + FT(2)))
    @. Yc.ρq_sno = Yc.ρ * (FT(5e-6) + FT(2e-6) * cos(ᶜz + FT(3)))
    @. Yc.ρn_lcl = Yc.ρ * (FT(1e6) + FT(1e5) * cos(ᶜz))
    @. Yc.ρn_rai = Yc.ρ * (FT(1e3) + FT(1e2) * cos(ᶜz))
    @. Yc.ρn_ice = Yc.ρ * (FT(1e5) + FT(1e4) * cos(ᶜz + FT(1)))
    @. Yc.ρq_rim = Yc.ρ * (FT(3e-5) + FT(1e-5) * cos(ᶜz + FT(0.5)))
    @. Yc.ρb_rim = Yc.ρ * (FT(5e-8) + FT(2e-8) * cos(ᶜz + FT(0.5)))
    Y = Fields.FieldVector(; c = Yc)

    ᶜu = @. Geometry.UVWVector(zero(ᶜz), zero(ᶜz), zero(ᶜz))
    ᶜp = @. FT(1e5) - FT(1e3) * ᶜz
    ᶜT = @. FT(280) - FT(20) * (ᶜz / FT(π))
    ᶜq_liq = @. (Yc.ρq_lcl + Yc.ρq_rai) / Yc.ρ
    ᶜq_ice = @. (Yc.ρq_icl + Yc.ρq_sno) / Yc.ρ
    ᶜq_tot_nonneg = @. Yc.ρq_tot / Yc.ρ
    p = (;
        atmos = (;
            vertical_diffusion = CA.DecayWithHeightDiffusion{FT}(;
                disable_momentum_vertical_diffusion = true,
                H = FT(1),
                D₀ = FT(1),
            ),
            microphysics_model = CA.NonEquilibriumMicrophysics2MP3(),
        ),
        params,
        precomputed = (; ᶜu, ᶜp, ᶜT, ᶜq_liq, ᶜq_ice, ᶜq_tot_nonneg),
        core = (; ᶜΦ = (@. CAP.grav(params) * ᶜz)),
        scratch = (;
            ᶜtemp_scalar = similar(ᶜz),
            ᶜtemp_scalar_2 = similar(ᶜz),
            ᶜtemp_scalar_3 = similar(ᶜz),
            ᶜtemp_scalar_4 = similar(ᶜz),
        ),
    )

    Yₜ = Fields.FieldVector(; c = zero(Yc))
    CA.vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, FT(0))

    ᶜK_h = similar(ᶜz)
    ᶜK_h .= CA.ᶜcompute_eddy_diffusivity_coefficient(
        Yc.ρ,
        p.atmos.vertical_diffusion,
    )
    ᶠρK = similar(Fields.coordinate_field(face_space).z)
    @. ᶠρK = CA.ᶠinterp(Yc.ρ) / CA.ᶠinterp(1 / max(ᶜK_h, eps(FT)))
    ᶜexpected = similar(ᶜz)

    # Total water diffuses on q_tot - q_rai - q_sno.
    @. ᶜexpected =
        -CA.ᶜdiffdivᵥ(
            -(ᶠρK * CA.ᶠgradᵥ((Yc.ρq_tot - Yc.ρq_rai - Yc.ρq_sno) / Yc.ρ)),
        )
    @test parent(Yₜ.c.ρq_tot) ≈ parent(ᶜexpected) rtol = FT(1e-12)

    # Each cloud species, the ice number and rime quantities included,
    # diffuses down its own gradient; precipitation does not diffuse.
    for name in (:ρq_lcl, :ρq_icl, :ρn_lcl, :ρn_ice, :ρq_rim, :ρb_rim)
        ᶜρχ = getproperty(Yc, name)
        @. ᶜexpected = -CA.ᶜdiffdivᵥ(-(ᶠρK * CA.ᶠgradᵥ(ᶜρχ / Yc.ρ)))
        @test maximum(abs, parent(ᶜexpected)) > 0
        @test parent(getproperty(Yₜ.c, name)) ≈ parent(ᶜexpected) rtol =
            FT(1e-12)
    end
    for name in (:ρq_rai, :ρq_sno, :ρn_rai)
        @test all(iszero, parent(getproperty(Yₜ.c, name)))
    end

    # The enthalpy flux carries the cloud liquid and cloud ice, not rain and
    # snow, as condensate.
    ᶠF_qt = @. -(ᶠρK * CA.ᶠgradᵥ((Yc.ρq_tot - Yc.ρq_rai - Yc.ρq_sno) / Yc.ρ))
    ᶠF_ql = @. -(ᶠρK * CA.ᶠgradᵥ(Yc.ρq_lcl / Yc.ρ))
    ᶠF_qi = @. -(ᶠρK * CA.ᶠgradᵥ(Yc.ρq_icl / Yc.ρ))
    ᶠF_qv = @. ᶠF_qt - ᶠF_ql - ᶠF_qi
    ᶜh_tot_v = @. TD.enthalpy_vapor(thermo_params, ᶜT) + p.core.ᶜΦ
    ᶜh_tot_l = @. TD.enthalpy_liquid(thermo_params, ᶜT) + p.core.ᶜΦ
    ᶜh_tot_i = @. TD.enthalpy_ice(thermo_params, ᶜT) + p.core.ᶜΦ
    (; ᶜcp) = enthalpy_flux_coefficients(
        thermo_params,
        ᶜT,
        p.core.ᶜΦ,
        (
            (@. (Yc.ρq_tot - Yc.ρq_rai - Yc.ρq_sno) / Yc.ρ),
            (@. Yc.ρq_lcl / Yc.ρ),
            (@. Yc.ρq_icl / Yc.ρ),
        ),
    )
    @. ᶜexpected =
        -CA.ᶜdiffdivᵥ(
            -(ᶠρK * (CA.ᶠinterp(ᶜcp) * CA.ᶠgradᵥ(ᶜT) + CA.ᶠgradᵥ(p.core.ᶜΦ))) +
            CA.ᶠinterp(ᶜh_tot_v) * ᶠF_qv +
            CA.ᶠinterp(ᶜh_tot_l) * ᶠF_ql +
            CA.ᶠinterp(ᶜh_tot_i) * ᶠF_qi,
        )
    @test parent(Yₜ.c.ρe_tot) ≈ parent(ᶜexpected) rtol = FT(1e-12)
end

@testset "Blended turbulent flux of suspended cloud species" begin
    # With prognostic TKE, a mixing length, and the SGS saturation-excess width
    # available, cloud liquid and ice take the blended flux
    #   F_qμ = (1 - λ_μ) (-ρK ∇q_μ) + λ_μ (c_qt F_qt + c_sl F_sl),
    # λ_μ = τ_e / (τ_e + τ_μ). λ_μ = 0 is the down-gradient self-diffusion;
    # λ_μ = 1 is the flux implied by the conserved-variable fluxes, which
    # vanishes in clear air and, in saturated air (C = 1), equals
    #   χ_μ [F_qt - (∂q_sat/∂T / c_p) F_sl] / (1 + γ).
    # The energy and total-water tendencies do not depend on λ_μ.
    FT = Float64
    params = CA.ClimaAtmosParameters(CP.create_toml_dict(FT))
    thermo_params = CAP.thermodynamics_params(params)

    (; cent_space, face_space) = get_cartesian_spaces(; FT)
    coords = Fields.coordinate_field(cent_space)
    ᶜz = coords.z

    NT = NamedTuple{
        (:ρ, :ρe_tot, :ρq_tot, :ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno, :ρtke),
        NTuple{8, FT},
    }
    Yc = similar(coords, NT)
    @. Yc.ρ = FT(1.2)
    @. Yc.ρe_tot = Yc.ρ * FT(2.5e5)
    @. Yc.ρq_lcl = Yc.ρ * (FT(2e-4) + FT(1e-4) * cos(ᶜz))
    @. Yc.ρq_icl = Yc.ρ * (FT(1e-4) + FT(5e-5) * cos(ᶜz))
    @. Yc.ρq_rai = FT(0)
    @. Yc.ρq_sno = FT(0)
    Y = Fields.FieldVector(; c = Yc)

    ᶜu = @. Geometry.UVWVector(zero(ᶜz), zero(ᶜz), zero(ᶜz))
    ᶜp = @. FT(1e5) - FT(1e3) * ᶜz
    ᶜT = @. FT(280) - FT(20) * (ᶜz / FT(π))
    ᶜq_liq = @. Yc.ρq_lcl / Yc.ρ
    ᶜq_ice = @. Yc.ρq_icl / Yc.ρ
    ᶜl_mix = similar(ᶜz)
    ᶜsgs_moments = similar(
        ᶜz,
        @NamedTuple{sigma_S::FT, λ_lagrange::FT, λ_eq::FT},
    )
    fill!(ᶜsgs_moments, (; sigma_S = FT(1e-9), λ_lagrange = FT(0), λ_eq = FT(0)))
    make_p(ᶜq_tot_nonneg) = (;
        atmos = (;
            vertical_diffusion = CA.DecayWithHeightDiffusion{FT}(;
                disable_momentum_vertical_diffusion = true,
                H = FT(1),
                D₀ = FT(1),
            ),
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
        ),
        params,
        precomputed = (;
            ᶜu, ᶜp, ᶜT, ᶜq_liq, ᶜq_ice, ᶜq_tot_nonneg, ᶜl_mix, ᶜsgs_moments,
        ),
        core = (; ᶜΦ = (@. CAP.grav(params) * ᶜz)),
        scratch = (;
            ᶜtemp_scalar = similar(ᶜz),
            ᶜtemp_scalar_2 = similar(ᶜz),
            ᶜtemp_scalar_3 = similar(ᶜz),
            ᶜtemp_scalar_4 = similar(ᶜz),
            ᶜtemp_scalar_8 = similar(ᶜz),
            ᶜtemp_scalar_9 = similar(ᶜz),
            ᶜtemp_scalar_10 = similar(ᶜz),
            ᶠtemp_C3 = Fields.Field(
                Geometry.Covariant3Vector{FT},
                face_space,
            ),
            ᶠtemp_C3_2 = Fields.Field(
                Geometry.Covariant3Vector{FT},
                face_space,
            ),
        ),
    )
    tendency(Y, p) = begin
        Yₜ = Fields.FieldVector(; c = zero(Y.c))
        CA.vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, FT(0))
        Yₜ
    end
    ᶜK_h = similar(ᶜz)
    ᶜK_h .= CA.ᶜcompute_eddy_diffusivity_coefficient(
        Yc.ρ,
        make_p(ᶜq_liq).atmos.vertical_diffusion,
    )
    ᶠρK = similar(Fields.coordinate_field(face_space).z)
    @. ᶠρK = CA.ᶠinterp(Yc.ρ) / CA.ᶠinterp(1 / max(ᶜK_h, eps(FT)))

    # Blend weight limits, with τ_e = l_mix / (c_d √tke). The liquid
    # relaxation time of the 1M microphysics is the constant
    # `condensation_evaporation_timescale`; the ice time of these parameters
    # (`ConstantTimescale`, the CloudMicrophysics default without a model) is
    # `sublimation_deposition_timescale`.
    c_d = CA.tke_dissipation_coefficient(CAP.turbconv_params(params))
    τ_l = CAP.condensation_evaporation_timescale(params)
    cmp = CAP.microphysics_1m_params(params)
    ρ_ref = TD.air_density(thermo_params, FT(280), FT(1e5), FT(0), FT(0), FT(0))
    @test CA.cloud_species_relaxation_time(
        cmp, thermo_params, FT(280), ρ_ref, FT(5e-3), FT(1e-4), FT(0),
        TD.Liquid(),
    ) == τ_l
    @test CA.cloud_species_relaxation_time(
        cmp, thermo_params, FT(260), ρ_ref, FT(2e-3), FT(0), FT(1e-4),
        TD.Ice(),
    ) == CAP.sublimation_deposition_timescale(params)
    @test CA.cloud_species_flux_blend(FT(50), FT(0.1), c_d, FT(0)) == 1
    @test CA.cloud_species_flux_blend(FT(0), FT(0.1), c_d, FT(100)) == 0
    @test CA.cloud_species_flux_blend(FT(50), FT(0.1), c_d, τ_l) ≈
          FT(50) / (FT(50) + τ_l * c_d * sqrt(FT(0.1)))
    # A disabled process gives self-diffusion at every TKE, with no NaN from
    # `Inf · 0` at zero TKE; the ramp-blended fidelity time drops an infinite
    # phase time of zero weight the same way.
    @test CA.cloud_species_flux_blend(FT(50), FT(0.1), c_d, FT(Inf)) == 0
    @test CA.cloud_species_flux_blend(FT(50), FT(0), c_d, FT(Inf)) == 0
    @test CA.sgs_microphysics_timescale(thermo_params, FT(280), τ_l, FT(Inf)) ==
          τ_l
    @test CA.sgs_variance_fidelity(FT(1), FT(Inf), c_d, FT(0), FT(50)) ==
          CA.sgs_variance_fidelity(FT(1), τ_l, c_d, FT(0), FT(50))

    # Clear, subsaturated air (q_t ≪ q_sat, so λ_eq ≪ -σ_S): λ = 0 (l_mix = 0)
    # recovers the down-gradient self-diffusion; λ = 1 (tke = 0) gives no
    # species flux at all because the equilibrium cloudy fraction vanishes.
    # The equilibrium multiplier is the mean saturation excess, as with the
    # untruncated Gaussian target.
    @. Yc.ρq_tot = Yc.ρ * (FT(1e-3) + FT(2e-4) * cos(ᶜz))
    ᶜq_tot_nonneg = @. Yc.ρq_tot / Yc.ρ
    @. ᶜsgs_moments.λ_eq =
        ᶜq_tot_nonneg - TD.q_vap_saturation(thermo_params, ᶜT, Yc.ρ)
    p = make_p(ᶜq_tot_nonneg)
    @. ᶜl_mix = FT(0)
    @. Yc.ρtke = Yc.ρ * FT(0.1)
    Yₜ_self = tendency(Y, p)
    ᶜexpected = similar(ᶜz)
    @. ᶜexpected = -CA.ᶜdiffdivᵥ(-(ᶠρK * CA.ᶠgradᵥ(Yc.ρq_lcl / Yc.ρ)))
    @test parent(Yₜ_self.c.ρq_lcl) ≈ parent(ᶜexpected) rtol = FT(1e-12)
    @. ᶜexpected = -CA.ᶜdiffdivᵥ(-(ᶠρK * CA.ᶠgradᵥ(Yc.ρq_icl / Yc.ρ)))
    @test parent(Yₜ_self.c.ρq_icl) ≈ parent(ᶜexpected) rtol = FT(1e-12)

    @. ᶜl_mix = FT(50)
    @. Yc.ρtke = FT(0)
    Yₜ_eq = tendency(Y, p)
    @test maximum(abs, parent(Yₜ_eq.c.ρq_lcl)) <=
          FT(1e-12) * maximum(abs, parent(Yₜ_self.c.ρq_lcl))
    @test maximum(abs, parent(Yₜ_eq.c.ρq_icl)) <=
          FT(1e-12) * maximum(abs, parent(Yₜ_self.c.ρq_icl))
    # Energy and total water do not depend on the blend.
    @test parent(Yₜ_eq.c.ρe_tot) == parent(Yₜ_self.c.ρe_tot)
    @test parent(Yₜ_eq.c.ρq_tot) == parent(Yₜ_self.c.ρq_tot)

    # Saturated air (q_t > q_sat everywhere, λ_eq ≫ σ_S so C = 1): λ = 1
    # gives the conserved-variable implied flux with the saturated-limit
    # coefficients, split between liquid and ice by the liquid fraction.
    ᶜq_sat = @. TD.q_vap_saturation(thermo_params, ᶜT, Yc.ρ)
    @. Yc.ρq_tot = Yc.ρ * (ᶜq_sat + FT(2e-3))
    ᶜq_tot_nonneg = @. Yc.ρq_tot / Yc.ρ
    @. ᶜsgs_moments.λ_eq = ᶜq_tot_nonneg - ᶜq_sat
    p = make_p(ᶜq_tot_nonneg)
    Yₜ_sat = tendency(Y, p)
    ᶠF_qt = @. -(ᶠρK * CA.ᶠgradᵥ(ᶜq_tot_nonneg))
    ᶜcp = @. TD.cp_m(thermo_params, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice)
    ᶜL_v = @. TD.latent_heat_vapor(thermo_params, ᶜT)
    ᶜL_s = @. TD.latent_heat_sublim(thermo_params, ᶜT)
    ᶠF_sl = @. -(
        ᶠρK * (
            CA.ᶠinterp(ᶜcp) * CA.ᶠgradᵥ(ᶜT) + CA.ᶠgradᵥ(p.core.ᶜΦ) -
            CA.ᶠinterp(ᶜL_v) * CA.ᶠgradᵥ(ᶜq_liq) -
            CA.ᶠinterp(ᶜL_s) * CA.ᶠgradᵥ(ᶜq_ice)
        )
    )
    ᶜχ_l = @. TD.liquid_fraction(thermo_params, ᶜT, ᶜq_liq, ᶜq_ice)
    # γ = (L / c_p) ∂q_sat/∂T for the phase mixture of Thermodynamics, with the
    # density of the diffusing mixture from the gas law
    ᶜρ_air = @. TD.air_density(
        thermo_params,
        ᶜT,
        ᶜp,
        ᶜq_tot_nonneg,
        ᶜq_liq,
        ᶜq_ice,
    )
    ᶜ∂q_sat_∂T = @. TD.∂q_vap_sat_∂T(thermo_params, ᶜT, ᶜρ_air)
    ᶜL = @. TD.latent_heat_mixed(
        thermo_params,
        ᶜT,
        TD.liquid_fraction_ramp(thermo_params, ᶜT),
    )
    ᶜγ = @. ᶜL / ᶜcp * ᶜ∂q_sat_∂T
    ᶜc_qt_l = @. ᶜχ_l / (1 + ᶜγ)
    ᶜc_sl_l = @. (-1) * ᶜc_qt_l * ᶜ∂q_sat_∂T / ᶜcp
    ᶜc_qt_i = @. (1 - ᶜχ_l) / (1 + ᶜγ)
    ᶜc_sl_i = @. (-1) * ᶜc_qt_i * ᶜ∂q_sat_∂T / ᶜcp
    @. ᶜexpected = -CA.ᶜdiffdivᵥ(
        CA.ᶠinterp(ᶜc_qt_l) * ᶠF_qt + CA.ᶠinterp(ᶜc_sl_l) * ᶠF_sl,
    )
    @test maximum(abs, parent(ᶜexpected)) > 0
    @test parent(Yₜ_sat.c.ρq_lcl) ≈ parent(ᶜexpected) rtol = FT(1e-10)
    @. ᶜexpected = -CA.ᶜdiffdivᵥ(
        CA.ᶠinterp(ᶜc_qt_i) * ᶠF_qt + CA.ᶠinterp(ᶜc_sl_i) * ᶠF_sl,
    )
    @test parent(Yₜ_sat.c.ρq_icl) ≈ parent(ᶜexpected) rtol = FT(1e-10)

    # Half-way blend is the average of the two limits (same state).
    @. Yc.ρtke = Yc.ρ * (FT(50) / (τ_l * c_d))^2
    Yₜ_mid = tendency(Y, p)
    @. ᶜl_mix = FT(0)
    Yₜ_sat_self = tendency(Y, p)
    @. ᶜexpected = (Yₜ_sat.c.ρq_lcl + Yₜ_sat_self.c.ρq_lcl) / 2
    @test parent(Yₜ_mid.c.ρq_lcl) ≈ parent(ᶜexpected) rtol = FT(1e-10)
end
