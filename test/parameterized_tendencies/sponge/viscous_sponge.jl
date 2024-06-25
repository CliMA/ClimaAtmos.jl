using ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
import ClimaAtmos as CA
import SurfaceFluxes as SF
import ClimaAtmos.Parameters as CAP
import ClimaCore as CC
include("../../test_helpers.jl")

### Common Objects ###
@testset begin
    "Rayleigh-sponge functions"
    ### Boilerplate default integrator objects
    config = CA.AtmosConfig(Dict("initial_condition" => "DryBaroclinicWave"))
    (; Y) = generate_test_simulation(config)
    z = CC.Fields.coordinate_field(Y.c).z
    zmax = maximum(CC.Fields.coordinate_field(Y.f).z)
    Y.c.uₕ.components.data.:1 .= ones(axes(Y.c))
    Y.c.uₕ.components.data.:2 .= ones(axes(Y.c))
    ᶜYₜ = Y .* FT(0)
    FT = eltype(Y)
    ### Component test begins here
    rs = CA.RayleighSponge(; zd = FT(0), α_uₕ = FT(1), α_w = FT(1))
    (; ᶜβ_rayleigh_uₕ) = CA.viscous_sponge_cache(Y, rs)
    @test ᶜβ_rayleigh_uₕ == @. sin(FT(π) / 2 * z / zmax)^2
    test_cache = (; viscous_sponge = (; ᶜβ_rayleigh_uₕ = ᶜβ_rayleigh_uₕ))
    CA.viscous_sponge_tendency!(ᶜYₜ, Y, test_cache, FT(0), rs)
    @test ᶜYₜ.c.uₕ.components.data.:1 == -1 .* (ᶜβ_rayleigh_uₕ)
    @test ᶜYₜ.c.uₕ.components.data.:2 == -1 .* (ᶜβ_rayleigh_uₕ)
end
