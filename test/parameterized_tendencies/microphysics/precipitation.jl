
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
    CC.Fields.bycolumn(axes(Y.c)) do colidx
        CA.compute_precipitation_cache!(
            Y,
            p,
            colidx,
            precip_model,
            turbconv_model,
        )
    end
    @test maximum(abs.(p.precipitation.ᶜS_ρq_tot)) <= sqrt(eps(FT))

    # Test that tendencies result in correct water-mass budget,
    # and that the tendency modification corresponds exactly to the 
    # cached source term. 
    CC.Fields.bycolumn(axes(Y.c)) do colidx
        CA.precipitation_tendency!(
            ᶜYₜ,
            Y,
            p,
            FT(0),
            colidx,
            precip_model,
            turbconv_model,
        )
    end
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
    @test CA.qₚ(FT(10), FT(2)) == FT(5)
    @test CA.qₚ(FT(-10), FT(2)) == FT(0)
    @test CA.limit(FT(10), FT(2)) == FT(1)
    CC.Fields.bycolumn(axes(Y.c)) do colidx
        CA.precipitation_tendency!(
            ᶜYₜ,
            Y,
            p,
            FT(0),
            colidx,
            precip_model,
            turbconv_model,
        )
    end
    @test ᶜYₜ.c.ρ == ᶜYₜ.c.ρq_tot
    @test ᶜYₜ.c.ρ == Y.c.ρ .* p.precipitation.ᶜSqₜᵖ
end
