
import ClimaAtmos as CA
import SurfaceFluxes as SF
import ClimaAtmos.Parameters as CAP
import ClimaCore as CC
import Thermodynamics as TD
import CloudMicrophysics as CM

include("../../test_helpers.jl")

### Common Objects ###
@testset begin
    "Precipitation tendency functions"
    ### Boilerplate default integrator objects
    config = CA.AtmosConfig(
        Dict(
            "initial_condition" => "PrecipitatingColumn",
            "moist" => "nonequil",
            "precip_model" => "0M",
            "config" => "column",
            "output_default_diagnostics" => false,
        ),
    )
    (; Y, p, params) = generate_test_simulation(config)

    FT = eltype(Y)
    ᶜYₜ = zero(Y)
    ### Component test begins here

    @info "0M Scheme"
    ### 0-Moment Scheme
    precip_model = CA.Microphysics0Moment()
    precip_cache = CA.precipitation_cache(Y, precip_model)
    # Test cache to verify expected variables exist in tendency function
    test_varnames = (
        :ᶜS_ρq_tot,
        :ᶜ3d_rain,
        :ᶜ3d_snow,
        :col_integrated_rain,
        :col_integrated_snow,
    )
    for var_name in test_varnames
        @test var_name ∈ propertynames(precip_cache)
    end
    turbconv_model = nothing # Extend to other turbulence convection models
    CA.compute_precipitation_cache!(Y, p, precip_model, turbconv_model)
    @test maximum(abs.(p.precipitation.ᶜS_ρq_tot)) <= sqrt(eps(FT))

    # Test that tendencies result in correct water-mass budget,
    # and that the tendency modification corresponds exactly to the
    # cached source term.
    CA.precipitation_tendency!(ᶜYₜ, Y, p, FT(0), precip_model, turbconv_model)
    @test ᶜYₜ.c.ρ == ᶜYₜ.c.ρq_tot
    @test ᶜYₜ.c.ρ == p.precipitation.ᶜS_ρq_tot

    ### 1-Moment Scheme
    @info "1M Scheme"
    config = CA.AtmosConfig(
        Dict(
            "initial_condition" => "PrecipitatingColumn",
            "moist" => "nonequil",
            "precip_model" => "1M",
            "config" => "column",
            "output_default_diagnostics" => false,
        ),
    )
    (; Y, p, params) = generate_test_simulation(config)
    precip_model = CA.Microphysics1Moment()
    (; turbconv_model) = p.atmos
    precip_cache = CA.precipitation_cache(Y, precip_model)
    ᶜYₜ = Y .* FT(0)
    test_varnames = (:ᶜSqₜᵖ, :ᶜSqᵣᵖ, :ᶜSqₛᵖ, :ᶜSeₜᵖ)
    for var_name in test_varnames
        @test var_name ∈ propertynames(precip_cache)
    end

    # test helper functions
    @test CA.qₚ(FT(10), FT(2)) == FT(5)
    @test CA.qₚ(FT(-10), FT(2)) == FT(0)
    @test CA.limit(FT(10), FT(2), 5) == FT(1)

    # compute source terms based on the last model state
    CA.precipitation_tendency!(ᶜYₜ, Y, p, FT(0), precip_model, turbconv_model)

    # check for nans
    @assert !any(isnan, ᶜYₜ.c.ρ)
    @assert !any(isnan, ᶜYₜ.c.ρq_tot)
    @assert !any(isnan, ᶜYₜ.c.ρe_tot)
    @assert !any(isnan, ᶜYₜ.c.ρq_rai)
    @assert !any(isnan, ᶜYₜ.c.ρq_sno)
    @assert !any(isnan, p.precomputed.ᶜwᵣ)
    @assert !any(isnan, p.precomputed.ᶜwₛ)

    # test water budget
    @test ᶜYₜ.c.ρ == ᶜYₜ.c.ρq_tot
    @test ᶜYₜ.c.ρ == Y.c.ρ .* p.precipitation.ᶜSqₜᵖ
    @test all(
        isapprox(
            .-p.precipitation.ᶜSqₛᵖ .- p.precipitation.ᶜSqᵣᵖ,
            p.precipitation.ᶜSqᵣᵖ,
            atol = eps(FT),
        ),
    )

    # test if terminal velocity is positive
    @test minimum(p.precomputed.ᶜwᵣ) >= FT(0)
    @test minimum(p.precomputed.ᶜwₛ) >= FT(0)

    # test if cloud fraction diagnostics make sense
    @assert !any(isnan, p.precomputed.cloud_diagnostics_tuple.cf)
    @test minimum(p.precomputed.cloud_diagnostics_tuple.cf) >= FT(0)
    @test maximum(p.precomputed.cloud_diagnostics_tuple.cf) <= FT(1)
end
