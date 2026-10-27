using Test
import ClimaComms
ClimaComms.@import_required_backends
import Random
import ClimaAtmos as CA
import ClimaAtmos.RRTMGPInterface as RRTMGPI
import ClimaParams as CP
import RRTMGP.Parameters.RRTMGPParameters

Random.seed!(1234)

include("test_helpers.jl")

# Setup RRTMGP parameters for extrapolation tests
const RRTMGP_PARAMS = RRTMGPParameters(CP.create_toml_dict(Float64))

@testset "Array <-> Field conversion" begin
    (; bubble_space) = get_spherical_spaces()
    test_field = ones(bubble_space)
    idx = size(parent(test_field))
    npts = idx[2] * idx[3] * idx[5] # Ni * Nj * Nh (for a VIJFH layout)

    # Field -> Array
    test_array = Fields.field2array(test_field)
    @test test_array isa Array
    @test length(test_array) == npts

    # Array -> Field
    new_field = Fields.array2field(test_array, bubble_space)
    @test new_field == test_field
    @test size(parent(new_field)) == idx
end

@testset "Interpolation methods" for FT in (Float32, Float64)
    # Test data
    T, Tꜛ, Tꜜ = FT(1), FT(1), FT(2)
    p, p₁, p₂ = FT(0.5), FT(2), FT(3)
    XT, XP = [FT(0)], [FT(0)]

    @testset "ArithmeticMean" begin
        RRTMGPI.interp!(RRTMGPI.ArithmeticMean(), XP, XT, [p₁], [Tꜜ], [p₂], [Tꜛ])
        @test XT[1] == (Tꜜ + Tꜛ) / 2
        @test XP[1] == (p₁ + p₂) / 2
    end

    @testset "GeometricMean" begin
        RRTMGPI.interp!(RRTMGPI.GeometricMean(), XP, XT, [p₁], [Tꜜ], [p₂], [Tꜛ])
        @test XT[1] == sqrt(Tꜜ * Tꜛ)
        @test XP[1] == sqrt(p₁ * p₂)
    end

    @testset "UniformZ" begin
        RRTMGPI.interp!(RRTMGPI.UniformZ(), XP, XT, [p₁], [Tꜜ], [p₂], [Tꜛ])
        @test XT[1] == (Tꜜ + Tꜛ) / 2
        @test XP[1] == RRTMGPI.uniform_z_p(XT[1], p₁, Tꜜ, p₂, Tꜛ)
    end

    @testset "UniformP" begin
        RRTMGPI.interp!(RRTMGPI.UniformP(), XP, XT, [p₁], [Tꜜ], [p₂], [Tꜛ])
        @test XP[1] == (p₁ + p₂) / 2
        # Same function form as uniform_z_p with swapped arguments
        @test XT[1] == RRTMGPI.uniform_z_p(XP[1], Tꜜ, p₁, Tꜛ, p₂)
    end
end

@testset "Extrapolation methods" for FT in (Float32, Float64)
    # Test data
    Tꜜ, Tꜛ = FT(2), FT(1)
    p₁, p₂ = FT(2), FT(3)
    Tₛ = FT(5)
    XT, XP = [FT(0)], [FT(0)]

    @testset "ArithmeticMean" begin
        RRTMGPI.extrap!(
            RRTMGPI.ArithmeticMean(),
            XP,
            XT,
            [p₁],
            [Tꜜ],
            [p₂],
            [Tꜛ],
            [Tₛ],
            RRTMGP_PARAMS,
        )
        @test XP[1] == (3 * p₁ - p₂) / 2
        @test XT[1] == (3 * Tꜜ - Tꜛ) / 2
    end

    @testset "GeometricMean" begin
        RRTMGPI.extrap!(
            RRTMGPI.GeometricMean(),
            XP,
            XT,
            [p₁],
            [Tꜜ],
            [p₂],
            [Tꜛ],
            [Tₛ],
            RRTMGP_PARAMS,
        )
        @test XP[1] == sqrt(p₁^3 / p₂)
        @test XT[1] == sqrt(Tꜜ^3 / Tꜛ)
    end

    @testset "UniformZ" begin
        RRTMGPI.extrap!(
            RRTMGPI.UniformZ(),
            XP,
            XT,
            [p₁],
            [Tꜜ],
            [p₂],
            [Tꜛ],
            [Tₛ],
            RRTMGP_PARAMS,
        )
        # UniformZ: T is computed from uniform_z_p formula
        @test XT[1] == RRTMGPI.uniform_z_p(XP[1], Tꜜ, p₁, Tꜛ, p₂)
    end

    @testset "UniformP" begin
        RRTMGPI.extrap!(
            RRTMGPI.UniformP(),
            XP,
            XT,
            [p₁],
            [Tꜜ],
            [p₂],
            [Tꜛ],
            [Tₛ],
            RRTMGP_PARAMS,
        )
        # UniformP: P is computed from uniform_z_p formula (with swapped args)
        @test XP[1] == RRTMGPI.uniform_z_p(XT[1], p₁, Tꜜ, p₂, Tꜛ)
    end
end

@testset "RadiationDYCOMS inversion isoline and updraft heating" for FT in (
    Float32,
    Float64,
)
    (; cent_space, face_space) = get_cartesian_spaces(; FT)
    ᶜcoords = Fields.coordinate_field(cent_space)
    ᶠcoords = Fields.coordinate_field(face_space)
    sgs_nt = NamedTuple{(:mse,), Tuple{FT}}
    Y = Fields.FieldVector(;
        c = similar(
            ᶜcoords,
            NamedTuple{
                (:ρ, :ρe_tot, :ρq_tot, :sgsʲs),
                Tuple{FT, FT, FT, Tuple{sgs_nt}},
            },
        ),
        f = similar(ᶠcoords, NamedTuple{(), Tuple{}}),
    )
    radiation_mode = CA.RadiationDYCOMS{FT}()
    rad_cache = CA.radiation_model_cache(Y, radiation_mode)
    ᶜq_liq = similar(Y.c, FT)
    ᶜρʲ = similar(Y.c, FT)
    fill!(Y.c.ρ, FT(1.1))
    fill!(ᶜq_liq, FT(2e-4))
    fill!(ᶜρʲ, FT(1.05))
    p = (;
        params = CA.ClimaAtmosParameters(FT),
        atmos = (;
            microphysics_model = CA.EquilibriumMicrophysics0M(),
            turbconv_model = CA.PrognosticEDMFX(;
                n_updrafts = 1,
                prognostic_tke = false,
                area_fraction = FT(1e-3),
            ),
            radiation_mode,
        ),
        radiation = rad_cache,
        precomputed = (; ᶜq_liq, ᶜρʲs = (ᶜρʲ,)),
    )

    z_levels = sort(unique(Array(parent(ᶜcoords.z))))
    n_levels = length(z_levels)
    k_inv = cld(n_levels, 2)
    z_inv = z_levels[k_inv]
    z_above = z_levels[k_inv + 1]

    # Case 1: levels 1..k_inv have q_tot >= 0.008 (with level k_inv at 0.0088),
    # while level k_inv + 1 above the inversion has q_tot = 0.0079 (closer to
    # 0.008 in |q_tot - 0.008|, but drier than the 0.008 threshold). The highest
    # level with q_tot >= 0.008 (k_inv) must be selected.
    @. Y.c.ρq_tot =
        Y.c.ρ * ifelse(
            ᶜcoords.z < z_inv,
            FT(0.0092),
            ifelse(
                ᶜcoords.z == z_inv,
                FT(0.0088),
                ifelse(ᶜcoords.z == z_above, FT(0.0079), FT(0.002)),
            ),
        )

    Yₜ = similar(Y)
    fill!(Yₜ, FT(0))
    CA.radiation_tendency!(Yₜ, Y, p, FT(0), radiation_mode)

    zi_selected = Array(parent(rad_cache.isoline_z_ρ_ρq.z))[1]
    @test zi_selected == z_inv

    # Verify PrognosticEDMFX updraft mse receives the grid-mean radiative flux
    # convergence divided by updraft density.
    ρe_tend = Array(parent(Yₜ.c.ρe_tot))
    mse_up_tend = Array(parent(Yₜ.c.sgsʲs.:(1).mse))
    ρ_up = Array(parent(ᶜρʲ))
    @test maximum(abs, ρe_tend) > 0
    @test mse_up_tend ≈ ρe_tend ./ ρ_up rtol = 100 * eps(FT)

    # Case 2: entire column drier than 0.008 kg/kg falls back to the level
    # closest to 0.008 kg/kg (k_dry).
    k_dry = max(1, k_inv - 1)
    z_dry = z_levels[k_dry]
    @. Y.c.ρq_tot = Y.c.ρ * ifelse(ᶜcoords.z == z_dry, FT(0.0075), FT(0.003))
    fill!(Yₜ, FT(0))
    CA.radiation_tendency!(Yₜ, Y, p, FT(0), radiation_mode)
    zi_fallback = Array(parent(rad_cache.isoline_z_ρ_ρq.z))[1]
    @test zi_fallback == z_dry
end

@testset "Cloud inhomogeneity switch" begin
    FT = Float64
    (; cent_space, face_space) = get_cartesian_spaces(; FT)
    Y = Fields.FieldVector(; c = ones(cent_space), f = ones(face_space))
    parsed_args = Dict(
        "rad" => "allsky",
        "idealized_h2o" => false,
        "idealized_clouds" => false,
        "prescribe_clouds_in_radiation" => false,
        "cloud_inhomogeneity_in_sw_radiation" => false,
        "add_isothermal_boundary_layer" => true,
        "aerosol_radiation" => false,
        "radiation_reset_rng_seed" => false,
        "deep_atmosphere" => true,
    )
    setup_components = (; radiation_mode = nothing)
    mode = CA.get_radiation_mode(parsed_args, FT; setup_components)
    @test mode isa RRTMGPI.AllSkyRadiation
    @test !mode.sw_cloud_inhomogeneity
    @test RRTMGPI.AllSkyRadiation().sw_cloud_inhomogeneity
    @test RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics().sw_cloud_inhomogeneity

    # With `sw_cloud_inhomogeneity = false`, the cloud cache is empty, leaving
    # the shortwave water paths unscaled and the factor setter a no-op.
    for cloud in (CA.InteractiveCloudInRadiation(), nothing)
        @test CA.get_cloud_cache(cloud, Y, nothing, false) == (;)
        cache = CA.get_cloud_cache(cloud, Y, nothing, true)
        @test hasproperty(cache, :ᶜsw_cloud_optical_depth_factor)
        @test all(==(1), parent(cache.ᶜsw_cloud_optical_depth_factor))
    end
    @test isnothing(CA.set_sw_cloud_optical_depth_factor!(Y, (; radiation = (;))))
end
