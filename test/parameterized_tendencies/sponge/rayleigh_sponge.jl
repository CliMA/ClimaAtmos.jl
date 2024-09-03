using ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import SurfaceFluxes as SF
import ClimaAtmos.Parameters as CAP
import ClimaCore as CC

include("../../test_helpers.jl")
### Common Objects ###
@testset begin
    "Rayleigh-sponge functions"
    ### Boilerplate default integrator objects
    config = CA.AtmosConfig(
        Dict("initial_condition" => "DryBaroclinicWave");
        job_id = "sponge1",
    )
    (; Y) = generate_test_simulation(config)
    zmax = maximum(CC.Fields.coordinate_field(Y.f).z)
    z = CC.Fields.coordinate_field(Y.c).z
    Y.c.uₕ.components.data.:1 .= ones(axes(Y.c))
    Y.c.uₕ.components.data.:2 .= ones(axes(Y.c))
    FT = eltype(Y)
    ᶜYₜ = zero(Y)
    ### Component test begins here
    rs = CA.RayleighSponge(; zd = FT(0), α_uₕ = FT(1), α_w = FT(1))
    (; ᶜβ_rayleigh_uₕ) = CA.rayleigh_sponge_cache(Y, rs)
    @test ᶜβ_rayleigh_uₕ == @. sin(FT(π) / 2 * z / zmax)^2
    test_cache = (; rayleigh_sponge = (; ᶜβ_rayleigh_uₕ = ᶜβ_rayleigh_uₕ))
    CA.rayleigh_sponge_tendency!(ᶜYₜ, Y, test_cache, FT(0), rs)
    # Test that only required tendencies are affected
    for (var_name) in filter(x -> (x != :uₕ), propertynames(Y.c))
        @test ᶜYₜ.c.:($var_name) == zeros(axes(Y.c))
    end
    for (var_name) in propertynames(Y.f)
        @test ᶜYₜ.f.:($var_name) == zeros(axes(Y.f))
    end
    @test ᶜYₜ.c.uₕ.components.data.:1 == -1 .* (ᶜβ_rayleigh_uₕ)
    @test ᶜYₜ.c.uₕ.components.data.:2 == -1 .* (ᶜβ_rayleigh_uₕ)
end
