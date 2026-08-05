# Sedimenting tracers in `gs_sedimenting_tracer_candidates` (including P3
# `ρn_ice`/`ρq_rim`/`ρb_rim`) receive the `α_vert_diff_tracer`-scaled diffusivity;
# passive tracers keep the unscaled diffusivity.
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaParams as CP
import ClimaCore: Fields, Geometry

include("../test_helpers.jl")

@testset "Vertical diffusion tracer diffusivity scaling" begin
    FT = Float64
    α = FT(0.4)
    override_file = Dict(
        "tracer_vertical_diffusion_factor" => Dict("value" => α, "type" => "float"),
    )
    params = CA.ClimaAtmosParameters(CP.create_toml_dict(FT; override_file))
    @test CAP.α_vert_diff_tracer(params) == α

    (; cent_space) = get_cartesian_spaces(; FT)
    coords = Fields.coordinate_field(cent_space)
    ᶜz = coords.z

    tracer_names = (:ρq_tot, :ρq_rai, :ρn_ice, :ρq_rim, :ρb_rim, :ρq_gas_A)
    NT = NamedTuple{
        (:ρ, :ρe_tot, tracer_names...),
        NTuple{2 + length(tracer_names), FT},
    }
    Yc = similar(coords, NT)
    @. Yc.ρ = FT(1.2)
    @. Yc.ρe_tot = Yc.ρ * FT(2.5e5)
    @. Yc.ρq_tot = Yc.ρ * FT(1e-2)
    ᶜρχ = @. Yc.ρ * cos(ᶜz)
    @. Yc.ρq_rai = ᶜρχ
    @. Yc.ρn_ice = ᶜρχ
    @. Yc.ρq_rim = ᶜρχ
    @. Yc.ρb_rim = ᶜρχ
    @. Yc.ρq_gas_A = ᶜρχ
    Y = Fields.FieldVector(; c = Yc)

    ᶜu = @. Geometry.UVWVector(zero(ᶜz), zero(ᶜz), zero(ᶜz))
    ᶜp = @. FT(1e5) - FT(1e3) * ᶜz
    ᶜT = @. FT(280) - FT(20) * (ᶜz / FT(π))
    ᶜq_liq = @. FT(1e-4) + zero(ᶜz)
    ᶜq_ice = @. FT(1e-5) + zero(ᶜz)
    ᶜq_tot_nonneg = @. FT(1e-2) + zero(ᶜz)
    p = (;
        atmos = (;
            vertical_diffusion = CA.DecayWithHeightDiffusion{FT}(;
                disable_momentum_vertical_diffusion = true,
                H = FT(1),
                D₀ = FT(1),
            ),
        ),
        params,
        precomputed = (; ᶜu, ᶜp, ᶜT, ᶜq_liq, ᶜq_ice, ᶜq_tot_nonneg),
        core = (; ᶜΦ = (@. CAP.grav(params) * ᶜz)),
        scratch = (;
            ᶜtemp_scalar = similar(ᶜz),
            ᶜtemp_scalar_2 = similar(ᶜz),
            ᶜtemp_scalar_3 = similar(ᶜz),
        ),
    )

    Yₜ = Fields.FieldVector(; c = zero(Yc))
    CA.vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, FT(0))

    @test maximum(abs, parent(Yₜ.c.ρq_rai)) > 0
    @test parent(Yₜ.c.ρn_ice) == parent(Yₜ.c.ρq_rai)
    @test parent(Yₜ.c.ρq_rim) == parent(Yₜ.c.ρq_rai)
    @test parent(Yₜ.c.ρb_rim) == parent(Yₜ.c.ρq_rai)
    @test parent(Yₜ.c.ρn_ice) ≈ α .* parent(Yₜ.c.ρq_gas_A)
    @test parent(Yₜ.c.ρq_gas_A) != parent(Yₜ.c.ρn_ice)
end
