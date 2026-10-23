using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaParams as CP
using NullBroadcasts: NullBroadcasted
using ClimaCore: Spaces, Fields, Geometry
using ClimaCore.CommonSpaces
using Base.Broadcast: materialize

@testset "Sponge Layers" begin
    FT = Float64
    z_max = FT(1)
    z_damping = FT(0.5)  # Damping starts at z = 0.5
    κ₂ = FT(2)  # Viscosity coefficient for viscous sponge

    # Create shared extruded cubed sphere space
    ᶜspace = ExtrudedCubedSphereSpace(
        FT;
        z_elem = 10,
        z_min = 0,
        z_max = z_max,
        radius = 10,
        h_elem = 4,
        n_quad_points = 4,
        staggering = CellCenter(),
    )
    ᶠspace = Spaces.face_space(ᶜspace)
    ᶜz = Fields.coordinate_field(ᶜspace).z
    ᶠz = Fields.coordinate_field(ᶠspace).z

    # Create shared test fields
    ᶜuₕ = Fields.Field(Geometry.Covariant12Vector{FT}, ᶜspace)
    @. ᶜuₕ = Geometry.Covariant12Vector(FT(1), FT(1))
    ᶠw = Fields.Field(Geometry.Covariant3Vector{FT}, ᶠspace)
    @. ᶠw = Geometry.Covariant3Vector(FT(1))
    ᶜρ = ones(ᶜspace)
    ᶜs_d = ones(ᶜspace)
    ᶜh_vap_plus_Φ = ones(ᶜspace)
    ᶜq_tot_eff = ones(ᶜspace)
    ᶜχ = ones(ᶜspace)
    ᶜχʲ = 2 .* ones(ᶜspace)

    # ========================================================================
    # Rayleigh Sponge Tests
    # ========================================================================
    @testset "Rayleigh Sponge" begin
        @testset "Damping coefficient β" begin
            rs = CA.RayleighSponge(;
                zd = z_damping,
                α_uₕ = FT(1),
                α_w = FT(2),
                α_tracer = FT(3),
            )

            # β = α * sin²(π/2 * (z - zd) / (zmax - zd)) for z > zd, else 0
            expected_uₕ = @. ifelse(
                ᶜz > z_damping,
                FT(1) * sin(FT(π) / 2 * (ᶜz - z_damping) / (z_max - z_damping))^2,
                FT(0),
            )
            expected_w = @. ifelse(
                ᶠz > z_damping,
                FT(2) * sin(FT(π) / 2 * (ᶠz - z_damping) / (z_max - z_damping))^2,
                FT(0),
            )

            @test CA.β_rayleigh_uₕ.(rs, ᶜz, z_max) ≈ expected_uₕ
            @test CA.β_rayleigh_u₃.(rs, ᶠz, z_max) ≈ expected_w
        end

        @testset "Tendency for horizontal velocity" begin
            rs = CA.RayleighSponge(;
                zd = z_damping,
                α_uₕ = FT(1),
                α_w = FT(1),
                α_tracer = FT(1),
            )

            # Tendency = -β * uₕ
            tendency = CA.rayleigh_sponge_tendency_uₕ(ᶜuₕ, rs)
            β = CA.β_rayleigh_uₕ.(rs, ᶜz, z_max)
            expected = @. -β * ᶜuₕ

            @test materialize(tendency) ≈ expected
        end

        @testset "Tendency for tracer (single argument)" begin
            rs = CA.RayleighSponge(;
                zd = z_damping,
                α_uₕ = FT(1),
                α_w = FT(1),
                α_tracer = FT(2),
            )

            # Tendency = -β * χ
            tendency = CA.rayleigh_sponge_tendency_tracer(ᶜχ, rs)
            β = CA.β_rayleigh_tracer.(rs, ᶜz, z_max)
            expected = @. -β * ᶜχ

            @test materialize(tendency) ≈ expected
        end

        @testset "Tendency for SGS tracer (updraft-environment difference)" begin
            rs = CA.RayleighSponge(;
                zd = z_damping,
                α_uₕ = FT(1),
                α_w = FT(1),
                α_tracer = FT(2),
            )

            # Tendency = -β * (χʲ - χ)
            tendency = CA.rayleigh_sponge_tendency_sgs_tracer(ᶜχʲ, ᶜχ, rs)
            β = CA.β_rayleigh_tracer.(rs, ᶜz, z_max)
            expected = @. -β * (ᶜχʲ - ᶜχ)

            @test materialize(tendency) ≈ expected
        end

        @testset "No sponge (nothing) returns NullBroadcasted" begin
            @test CA.rayleigh_sponge_tendency_uₕ(ᶜuₕ, nothing) isa NullBroadcasted
            @test CA.rayleigh_sponge_tendency_tracer(ᶜχ, nothing) isa NullBroadcasted
            @test CA.rayleigh_sponge_tendency_sgs_tracer(ᶜχʲ, ᶜχ, nothing) isa
                  NullBroadcasted
        end
    end

    # ========================================================================
    # Viscous Sponge Tests
    # ========================================================================
    @testset "Viscous Sponge" begin
        @testset "Damping coefficient β_viscous" begin
            s = CA.ViscousSponge(; zd = z_damping, κ₂ = κ₂)

            # β = κ₂ * sin²(π/2 * (z - zd) / (zmax - zd)) for z > zd, else 0
            expected = @. ifelse(
                ᶜz > z_damping,
                κ₂ * sin(FT(π) / 2 * (ᶜz - z_damping) / (z_max - z_damping))^2,
                FT(0),
            )

            @test CA.β_viscous.(s, ᶜz, z_max) ≈ expected

            # Test that damping is maximum at z_max (= κ₂)
            @test CA.β_viscous(s, z_max, z_max) ≈ κ₂
        end

        @testset "Tendency functions return finite values" begin
            s = CA.ViscousSponge(; zd = z_damping, κ₂ = κ₂)

            # Test viscous_sponge_tendency_uₕ returns finite values
            tendency_uₕ = CA.viscous_sponge_tendency_uₕ(ᶜuₕ, s)
            @test all(isfinite, parent(materialize(tendency_uₕ)))

            # Test viscous_sponge_tendency_u₃ returns finite values
            tendency_u₃ = CA.viscous_sponge_tendency_u₃(ᶠw, s)
            @test all(isfinite, parent(materialize(tendency_u₃)))

            # Split-form energy tendency: dry static energy + water enthalpy
            tendency_ρe_tot_dry =
                CA.viscous_sponge_tendency_ρe_tot_dry(ᶜρ, ᶜs_d, s)
            @test all(isfinite, parent(materialize(tendency_ρe_tot_dry)))
            tendency_ρe_tot_water = CA.viscous_sponge_tendency_ρe_tot_water(
                ᶜρ,
                ᶜh_vap_plus_Φ,
                ᶜq_tot_eff,
                s,
            )
            @test all(isfinite, parent(materialize(tendency_ρe_tot_water)))

            # Test viscous_sponge_tendency_tracer returns finite values
            tendency_tracer = CA.viscous_sponge_tendency_tracer(ᶜρ, ᶜχ, s)
            @test all(isfinite, parent(materialize(tendency_tracer)))
        end

        @testset "No sponge (nothing) returns NullBroadcasted" begin
            @test CA.viscous_sponge_tendency_uₕ(ᶜuₕ, nothing) isa NullBroadcasted
            @test CA.viscous_sponge_tendency_u₃(ᶠw, nothing) isa NullBroadcasted
            @test CA.viscous_sponge_tendency_ρe_tot_dry(
                ᶜρ,
                ᶜs_d,
                nothing,
            ) isa NullBroadcasted
            @test CA.viscous_sponge_tendency_ρe_tot_water(
                ᶜρ,
                ᶜh_vap_plus_Φ,
                ᶜq_tot_eff,
                nothing,
            ) isa NullBroadcasted
            @test CA.viscous_sponge_tendency_tracer(ᶜρ, ᶜχ, nothing) isa NullBroadcasted
        end

        @testset "viscous_sponge_tendency! water enthalpy and cloud species" begin
            s = CA.ViscousSponge(; zd = z_damping, κ₂ = κ₂)
            params = CA.ClimaAtmosParameters(CP.create_toml_dict(FT))
            thermo_params = CAP.thermodynamics_params(params)

            coords = Fields.coordinate_field(ᶜspace)
            ᶜlat = coords.lat
            ᶜlon = coords.long

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
            NTc = NamedTuple{
                (:ρ, :ρe_tot, :uₕ, tracer_names...),
                Tuple{
                    FT,
                    FT,
                    Geometry.Covariant12Vector{FT},
                    ntuple(_ -> FT, length(tracer_names))...,
                },
            }
            Yc = similar(coords, NTc)
            Yf = similar(
                Fields.coordinate_field(ᶠspace),
                NamedTuple{(:u₃,), Tuple{Geometry.Covariant3Vector{FT}}},
            )
            ᶜpattern = @. cosd(ᶜlat) * sind(ᶜlon)
            @. Yc.ρ = FT(1.2)
            @. Yc.ρe_tot = Yc.ρ * FT(2.5e5)
            @. Yc.uₕ = Geometry.Covariant12Vector(FT(1), FT(1))
            @. Yf.u₃ = Geometry.Covariant3Vector(FT(0.5))
            @. Yc.ρq_tot = Yc.ρ * (FT(1e-2) + FT(2e-3) * ᶜpattern)
            @. Yc.ρq_lcl = Yc.ρ * (FT(2e-4) + FT(1e-4) * ᶜpattern)
            @. Yc.ρq_icl = Yc.ρ * (FT(1e-4) - FT(5e-5) * ᶜpattern)
            @. Yc.ρq_rai = Yc.ρ * (FT(1e-5) + FT(5e-6) * ᶜpattern)
            @. Yc.ρq_sno = Yc.ρ * (FT(5e-6) - FT(2e-6) * ᶜpattern)
            @. Yc.ρn_lcl = Yc.ρ * (FT(1e6) + FT(1e5) * ᶜpattern)
            @. Yc.ρn_rai = Yc.ρ * (FT(1e3) + FT(1e2) * ᶜpattern)
            @. Yc.ρq_gas_A = Yc.ρ * (FT(1) + FT(0.1) * ᶜpattern)
            Y = Fields.FieldVector(; c = Yc, f = Yf)

            ᶜT = @. FT(270) + FT(5) * ᶜpattern - FT(10) * ᶜz
            ᶜΦ = @. CAP.grav(params) * ᶜz
            ᶜq_liq = @. Yc.ρq_lcl / Yc.ρ
            ᶜq_ice = @. Yc.ρq_icl / Yc.ρ
            ᶜq_tot_nonneg = @. Yc.ρq_tot / Yc.ρ
            p = (;
                atmos = (;
                    viscous_sponge = s,
                    microphysics_model = CA.NonEquilibriumMicrophysics1M(),
                    turbconv_model = nothing,
                ),
                params,
                precomputed = (; ᶜT, ᶜq_liq, ᶜq_ice, ᶜq_tot_nonneg),
                core = (; ᶜΦ),
                scratch = (;
                    ᶜtemp_scalar = similar(ᶜz),
                    ᶜtemp_scalar_2 = similar(ᶜz),
                ),
            )

            Yₜ = similar(Y)
            Yₜ .= zero(FT)
            CA.viscous_sponge_tendency!(Yₜ, Y, p)

            # Mass conservation: ρ tendency equals ρq_tot tendency on q_tot_eff.
            ᶜq_tot_eff_expected =
                @. (Yc.ρq_tot - Yc.ρq_rai - Yc.ρq_sno) / Yc.ρ
            expected_ρq_tot = materialize(
                CA.viscous_sponge_tendency_tracer(Yc.ρ, ᶜq_tot_eff_expected, s),
            )
            @test maximum(abs, parent(Yₜ.c.ρq_tot)) > 0
            @test parent(Yₜ.c.ρ) == parent(Yₜ.c.ρq_tot)
            @test parent(Yₜ.c.ρq_tot) ≈ parent(expected_ρq_tot)

            # Suspended cloud species diffuse down their own gradients.
            @test parent(Yₜ.c.ρq_lcl) ≈ parent(
                materialize(
                    CA.viscous_sponge_tendency_tracer(Yc.ρ, ᶜq_liq, s),
                ),
            )
            @test parent(Yₜ.c.ρq_icl) ≈ parent(
                materialize(
                    CA.viscous_sponge_tendency_tracer(Yc.ρ, ᶜq_ice, s),
                ),
            )
            @test parent(Yₜ.c.ρn_lcl) ≈ parent(
                materialize(
                    CA.viscous_sponge_tendency_tracer(
                        Yc.ρ,
                        (@. Yc.ρn_lcl / Yc.ρ),
                        s,
                    ),
                ),
            )

            # Rain, snow, and rain number density do not diffuse.
            @test all(iszero, parent(Yₜ.c.ρq_rai))
            @test all(iszero, parent(Yₜ.c.ρq_sno))
            @test all(iszero, parent(Yₜ.c.ρn_rai))

            # Passive tracer diffuses independently.
            @test parent(Yₜ.c.ρq_gas_A) ≈ parent(
                materialize(
                    CA.viscous_sponge_tendency_tracer(
                        Yc.ρ,
                        (@. Yc.ρq_gas_A / Yc.ρ),
                        s,
                    ),
                ),
            )

            # Energy matches the Yatunin et al. (2026, Eq. 33) 4-term split.
            ᶜs_d_expected =
                @. CA.TD.dry_static_energy(thermo_params, ᶜT, ᶜΦ)
            ᶜh_v_Φ = @. CA.TD.enthalpy_vapor(thermo_params, ᶜT) + ᶜΦ
            ᶜh_lv = @. CA.TD.enthalpy_liquid(thermo_params, ᶜT) -
               CA.TD.enthalpy_vapor(thermo_params, ᶜT)
            ᶜh_iv = @. CA.TD.enthalpy_ice(thermo_params, ᶜT) -
               CA.TD.enthalpy_vapor(thermo_params, ᶜT)
            expected_ρe_tot =
                materialize(
                    CA.viscous_sponge_tendency_ρe_tot_dry(
                        Yc.ρ,
                        ᶜs_d_expected,
                        s,
                    ),
                ) .+
                materialize(
                    CA.viscous_sponge_tendency_ρe_tot_water(
                        Yc.ρ,
                        ᶜh_v_Φ,
                        ᶜq_tot_eff_expected,
                        s,
                    ),
                ) .+
                materialize(
                    CA.viscous_sponge_tendency_ρe_tot_water(
                        Yc.ρ,
                        ᶜh_lv,
                        ᶜq_liq,
                        s,
                    ),
                ) .+
                materialize(
                    CA.viscous_sponge_tendency_ρe_tot_water(
                        Yc.ρ,
                        ᶜh_iv,
                        ᶜq_ice,
                        s,
                    ),
                )
            @test parent(Yₜ.c.ρe_tot) ≈ parent(expected_ρe_tot)

            # Kirchhoff reference-temperature shift invariance.
            δT₀ = FT(25)
            toml_shifted = CP.create_toml_dict(FT)
            toml_shifted.data["thermodynamics_temperature_reference"]["value"] +=
                δT₀
            toml_shifted.data["latent_heat_vaporization_at_reference"]["value"] +=
                (CAP.cp_v(params) - CAP.cp_l(params)) * δT₀
            toml_shifted.data["latent_heat_sublimation_at_reference"]["value"] +=
                (CAP.cp_v(params) - CAP.cp_i(params)) * δT₀
            params_shifted = CA.ClimaAtmosParameters(toml_shifted)
            p_shifted = merge(p, (; params = params_shifted))
            Yₜ_shifted = similar(Y)
            Yₜ_shifted .= zero(FT)
            CA.viscous_sponge_tendency!(Yₜ_shifted, Y, p_shifted)
            tp_s = CAP.thermodynamics_params(params_shifted)
            δh_water =
                CA.TD.enthalpy_vapor(tp_s, FT(280)) -
                CA.TD.enthalpy_vapor(thermo_params, FT(280))
            ᶜresidual = similar(ᶜz)
            @. ᶜresidual =
                (Yₜ_shifted.c.ρe_tot - Yₜ.c.ρe_tot) - δh_water * Yₜ.c.ρq_tot
            @test maximum(abs, parent(ᶜresidual)) <=
                  FT(1e-11) * maximum(abs, parent(Yₜ.c.ρe_tot))

            # At horizontally uniform q_tot_eff, ρq_tot and ρ tendencies vanish
            # while suspended cloud species still diffuse down their own
            # horizontal gradients and drive latent enthalpy transport.
            @. Yc.ρq_tot = Yc.ρ * FT(1e-2) + Yc.ρq_rai + Yc.ρq_sno
            @. ᶜq_tot_nonneg = Yc.ρq_tot / Yc.ρ
            @. ᶜT = FT(270) - FT(10) * ᶜz
            Yₜ .= zero(FT)
            CA.viscous_sponge_tendency!(Yₜ, Y, p)
            @test maximum(abs, parent(Yₜ.c.ρq_tot)) < FT(1e-15)
            @test maximum(abs, parent(Yₜ.c.ρq_lcl)) > 0
            @test maximum(abs, parent(Yₜ.c.ρq_icl)) > 0
            @test maximum(abs, parent(Yₜ.c.ρe_tot)) > 0
        end
    end
end
