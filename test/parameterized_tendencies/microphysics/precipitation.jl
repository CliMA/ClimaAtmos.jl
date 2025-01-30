using ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import SurfaceFluxes as SF
import ClimaAtmos.Parameters as CAP
import ClimaCore as CC
import Thermodynamics as TD
import CloudMicrophysics as CM

include("../../test_helpers.jl")

import Test

@testset "Equilibrium Moisture + 0-moment precipitation RHS terms" begin

    ### Boilerplate default integrator objects
    config = CA.AtmosConfig(
        Dict(
            "initial_condition" => "DYCOMS_RF02",
            "moist" => "equil",
            "precip_model" => "0M",
            "config" => "column",
            "output_default_diagnostics" => false,
        ),
        job_id = "equil_0M",
    )
    (; Y, p, params) = generate_test_simulation(config)

    FT = eltype(Y)
    ᶜYₜ = zero(Y)

    # Set all model choices
    (; turbconv_model, moisture_model, precip_model) = p.atmos

    # Test cache to verify expected variables exist in tendency function
    precip_cache = CA.precipitation_cache(Y, precip_model)
    test_varnames = (
        :ᶜS_ρq_tot,
        :ᶜS_ρe_tot,
        :ᶜ3d_rain,
        :ᶜ3d_snow,
        :surface_rain_flux,
        :surface_snow_flux,
    )
    for var_name in test_varnames
        @test var_name ∈ propertynames(precip_cache)
    end

    # No NaNs in cache
    CA.compute_precipitation_cache!(Y, p, precip_model, turbconv_model)
    @test maximum(abs.(p.precipitation.ᶜS_ρq_tot)) <= sqrt(eps(FT))

    # Test that tendencies result in correct water-mass budget,
    # and that the tendency modification corresponds exactly to the
    # cached source term.
    CA.precipitation_tendency!(
        ᶜYₜ,
        Y,
        p,
        FT(0),
        moisture_model,
        precip_model,
        turbconv_model,
    )
    @test ᶜYₜ.c.ρ == ᶜYₜ.c.ρq_tot
    @test ᶜYₜ.c.ρ == p.precipitation.ᶜS_ρq_tot

    # No cloud condensation tendency for the equilibrium model
    @test CA.cloud_condensate_tendency!(
        ᶜYₜ,
        p,
        moisture_model,
        precip_model,
    ) isa Nothing
end

@testset "NonEquilibrium Moisture + 1-moment precipitation RHS terms" begin

    ### Boilerplate default integrator objects
    config = CA.AtmosConfig(
        Dict(
            "initial_condition" => "PrecipitatingColumn",
            "moist" => "nonequil",
            "precip_model" => "1M",
            "config" => "column",
            "output_default_diagnostics" => false,
        ),
        job_id = "precipitation2",
    )
    (; Y, p, params) = generate_test_simulation(config)

    FT = eltype(Y)
    ᶜYₜ = zero(Y)

    # Set all model choices
    (; turbconv_model, moisture_model, precip_model) = p.atmos

    # Test cache to verify expected variables exist in tendency function
    precip_cache = CA.precipitation_cache(Y, precip_model)
    test_varnames =
        (:ᶜSqₗᵖ, :ᶜSqᵢᵖ, :ᶜSqᵣᵖ, :ᶜSqₛᵖ, :surface_rain_flux, :surface_snow_flux)
    for var_name in test_varnames
        @test var_name ∈ propertynames(precip_cache)
    end

    # test helper functions
    @test CA.limit(FT(10), FT(2), 5) == FT(1)

    # compute source terms based on the last model state
    CA.precipitation_tendency!(
        ᶜYₜ,
        Y,
        p,
        FT(0),
        moisture_model,
        precip_model,
        turbconv_model,
    )

    # check for nans
    @assert !any(isnan, ᶜYₜ.c.ρ)
    @assert !any(isnan, ᶜYₜ.c.ρq_tot)
    @assert !any(isnan, ᶜYₜ.c.ρe_tot)
    @assert !any(isnan, ᶜYₜ.c.ρq_liq)
    @assert !any(isnan, ᶜYₜ.c.ρq_ice)
    @assert !any(isnan, ᶜYₜ.c.ρq_rai)
    @assert !any(isnan, ᶜYₜ.c.ρq_sno)
    @assert !any(isnan, p.precomputed.ᶜwₗ)
    @assert !any(isnan, p.precomputed.ᶜwᵢ)
    @assert !any(isnan, p.precomputed.ᶜwᵣ)
    @assert !any(isnan, p.precomputed.ᶜwₛ)

    # test water budget
    @test ᶜYₜ.c.ρ == ᶜYₜ.c.ρq_tot
    @assert iszero(ᶜYₜ.c.ρ)

    # test nonequilibrium cloud condensate
    CA.cloud_condensate_tendency!(ᶜYₜ, p, moisture_model, precip_model)
    @assert !any(isnan, ᶜYₜ.c.ρq_liq)
    @assert !any(isnan, ᶜYₜ.c.ρq_ice)

    # test if terminal velocity is positive
    @test minimum(p.precomputed.ᶜwₗ) >= FT(0)
    @test minimum(p.precomputed.ᶜwᵢ) >= FT(0)
    @test minimum(p.precomputed.ᶜwᵣ) >= FT(0)
    @test minimum(p.precomputed.ᶜwₛ) >= FT(0)

    # test if cloud fraction diagnostics make sense
    @assert !any(isnan, p.precomputed.cloud_diagnostics_tuple.cf)
    @test minimum(p.precomputed.cloud_diagnostics_tuple.cf) >= FT(0)
    @test maximum(p.precomputed.cloud_diagnostics_tuple.cf) <= FT(1)
end
