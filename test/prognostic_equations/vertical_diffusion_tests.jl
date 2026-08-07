# Verify the vertical diffusion tendency:
# - ρq_tot diffuses on q_tot_eff = q_tot - q_rai - q_sno (rain/snow excluded)
# - ρq_lcl, ρq_icl inherit their share via clipped ratio min(q_μ/q_tot_eff, 1)
# - ρq_rai, ρq_sno, ρn_rai (and other sedimenting non-lcl/icl species) receive
#   no diffusion tendency
# - Passive (non-microphysics) grid-scale tracers diffuse with the full K_h
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaParams as CP
import ClimaCore: Fields, Geometry

include("../test_helpers.jl")

@testset "Vertical diffusion tendency (new formulation)" begin
    FT = Float64
    params = CA.ClimaAtmosParameters(CP.create_toml_dict(FT))

    (; cent_space) = get_cartesian_spaces(; FT)
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
    ᶜq_liq = @. FT(2e-4) + zero(ᶜz)
    ᶜq_ice = @. FT(1e-4) + zero(ᶜz)
    ᶜq_tot_nonneg = @. FT(1e-2) + zero(ᶜz)
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

    # Cloud mass species inherit a share of the q_tot diffusion.
    @test maximum(abs, parent(Yₜ.c.ρq_lcl)) > 0
    @test maximum(abs, parent(Yₜ.c.ρq_icl)) > 0

    # Cloud number densities scale proportionally when present.
    @test maximum(abs, parent(Yₜ.c.ρn_lcl)) > 0

    # Sedimenting species (rain, snow, rain number) do not diffuse.
    @test all(iszero, parent(Yₜ.c.ρq_rai))
    @test all(iszero, parent(Yₜ.c.ρq_sno))
    @test all(iszero, parent(Yₜ.c.ρn_rai))

    # Passive (non-microphysics) grid-scale tracer diffuses with full K_h.
    @test maximum(abs, parent(Yₜ.c.ρq_gas_A)) > 0
end
