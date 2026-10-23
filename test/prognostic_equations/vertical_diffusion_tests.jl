# Verify the vertical diffusion tendency (Eqs. 30 and 33, Yatunin et al. 2026):
# - ρq_tot diffuses on q_tot_eff = q_tot - q_rai - q_sno (rain/snow excluded)
# - Suspended cloud species (ρq_lcl, ρq_icl, ρn_lcl, ρn_ice, and with 2MP3 the
#   rime mass and volume) diffuse down their own vertical gradients with ρ K_h
# - ρq_rai, ρq_sno, ρn_rai receive no diffusion tendency
# - Passive (non-microphysics) grid-scale tracers diffuse with the full K_h
# - Eq. (30) enthalpy-species identity, reversible 0M moist-adiabatic residual,
#   and Kirchhoff T_0 reference-state invariance hold.
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

    # Eq. (30) identity (1M) — Yₜ.c.ρe_tot equals the s_d diffusion plus
    # the divergence of Σ_μ (face-interpolated h_tot,μ) * F_qμ for vapor,
    # liquid, and ice, where F_qv = F_qt_eff - F_qlcl - F_qicl.
    ᶠF_qt = @. -(ᶠρK * CA.ᶠgradᵥ((Yc.ρq_tot - Yc.ρq_rai - Yc.ρq_sno) / Yc.ρ))
    ᶠF_ql = @. -(ᶠρK * CA.ᶠgradᵥ(Yc.ρq_lcl / Yc.ρ))
    ᶠF_qi = @. -(ᶠρK * CA.ᶠgradᵥ(Yc.ρq_icl / Yc.ρ))
    ᶠF_qv = @. ᶠF_qt - ᶠF_ql - ᶠF_qi
    ᶜh_tot_v = @. TD.enthalpy_vapor(thermo_params, ᶜT) + p.core.ᶜΦ
    ᶜh_tot_l = @. TD.enthalpy_liquid(thermo_params, ᶜT) + p.core.ᶜΦ
    ᶜh_tot_i = @. TD.enthalpy_ice(thermo_params, ᶜT) + p.core.ᶜΦ
    ᶜs_d = @. TD.dry_static_energy(thermo_params, ᶜT, p.core.ᶜΦ)
    ᶜρe_totₜ_eq30 = similar(ᶜz)
    @. ᶜρe_totₜ_eq30 =
        -CA.ᶜdiffdivᵥ(
            -(ᶠρK * CA.ᶠgradᵥ(ᶜs_d)) +
            CA.ᶠinterp(ᶜh_tot_v) * ᶠF_qv +
            CA.ᶠinterp(ᶜh_tot_l) * ᶠF_ql +
            CA.ᶠinterp(ᶜh_tot_i) * ᶠF_qi,
        )
    @test parent(Yₜ.c.ρe_tot) ≈ parent(ᶜρe_totₜ_eq30) rtol = FT(1e-12)

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
    # MSE and q_tot — Yₜ.c.ρq_tot == 0 and
    # |Yₜ.c.ρe_tot| <= 0.03 * |Yₜ.c.ρe_tot from s_d alone|.
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
    ᶜs_d_only = similar(ᶜz)
    @. ᶜs_d_only =
        -CA.ᶜdiffdivᵥ(
            -(ᶠρK * CA.ᶠgradᵥ(TD.dry_static_energy(thermo_params, ᶜT, p.core.ᶜΦ))),
        )
    @test maximum(abs, parent(Yₜ_0m.c.ρq_tot)) <= FT(1e-15)
    @test maximum(abs, parent(Yₜ_0m.c.ρe_tot)) <=
          FT(0.03) * maximum(abs, parent(ᶜs_d_only))
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
    ᶜs_d = @. TD.dry_static_energy(thermo_params, ᶜT, p.core.ᶜΦ)
    @. ᶜexpected =
        -CA.ᶜdiffdivᵥ(
            -(ᶠρK * CA.ᶠgradᵥ(ᶜs_d)) +
            CA.ᶠinterp(ᶜh_tot_v) * ᶠF_qv +
            CA.ᶠinterp(ᶜh_tot_l) * ᶠF_ql +
            CA.ᶠinterp(ᶜh_tot_i) * ᶠF_qi,
        )
    @test parent(Yₜ.c.ρe_tot) ≈ parent(ᶜexpected) rtol = FT(1e-12)
end
