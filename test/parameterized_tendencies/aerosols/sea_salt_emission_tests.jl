using Test
import ClimaAtmos as CA
import ClimaCore.Fields as Fields

const CP = CA.CP
const SF = CA.SF

include(joinpath(@__DIR__, "..", "..", "test_helpers.jl"))
include(joinpath(@__DIR__, "sea_salt_fit_skill.jl"))

const FT = Float64
const toml_dict = CP.create_toml_dict(FT)

@testset "MOST wind at height" begin
    sfp = SF.Parameters.SurfaceFluxesParameters(toml_dict, CA.UF.GryanikParams)
    u10(u_star; L = FT(-50)) = CA.wind_at_height(FT(10), u_star, L, sfp)

    # wind is 0 at calm surface, monotone in u_star
    @test u10(FT(0)) == 0
    @test u10(FT(0.6)) > u10(FT(0.3)) > 0

    # near-neutral L is guarded, both signs
    @test isfinite(u10(FT(0.3); L = FT(1e-30)))
    @test isfinite(u10(FT(0.3); L = FT(-1e-30)))
end

@testset "Gong 3-mode lognormal fit skill" begin
    sp = SeaSaltFitSkill.stored_params(toml_dict)

    s = SeaSaltFitSkill.skill(sp)
    # pointwise fits of lognormal fit to full Gong expansion
    @test s.max_pointwise < 0.2
    @test maximum(abs, s.number) < 0.08
    @test maximum(abs, s.mass) < 0.08

    # Number emission dominated by the sub-micron bins, mass emission by super-micron bins.
    (; bin_number_scales, bin_mass_scales) = sp
    @test sum(bin_number_scales[1:2]) / sum(bin_number_scales) > 0.9
    @test sum(bin_mass_scales[4:5]) / sum(bin_mass_scales) > 0.8
end

# TODO: Need tendency tests beyond ci?
