using Test
import ClimaAtmos as CA
import ClimaCore.Fields as Fields
import ClimaCore.Spaces as Spaces
import ClimaCore.Operators as Operators
const TD = CA.TD

include(joinpath(@__DIR__, "..", "..", "test_helpers.jl"))

const FT = Float64

"""
    edmf_column_config(job_id)

Bomex `PrognosticEDMFX` column with 1-moment microphysics and prognostic sea
salt: the configuration every subdomain test runs on.
"""
edmf_column_config(job_id) = CA.AtmosConfig(
    Dict(
        "config" => "column",
        "initial_condition" => "Bomex",
        "turbconv" => "prognostic_edmfx",
        "implicit_diffusion" => true,
        "approximate_linear_solve_iters" => 2,
        "edmfx_entr_model" => "Generalized",
        "edmfx_detr_model" => "Generalized",
        "edmfx_sgs_mass_flux" => true,
        "edmfx_sgs_diffusive_flux" => true,
        "edmfx_nh_pressure" => true,
        "edmfx_vertical_diffusion" => true,
        "edmfx_filter" => true,
        "prognostic_tke" => true,
        "microphysics_model" => "1M",
        "z_max" => 4200,
        "z_elem" => 60,
        "z_stretch" => false,
        "perturb_initstate" => false,
        "dt" => "40secs",
        "prognostic_aerosols" => ["SSLT"],
        "toml" => ["toml/prognostic_edmfx_1M.toml"],
        "strict_params" => false,
        "output_default_diagnostics" => false,
    );
    job_id,
)

"""
    seed_sea_salt!(Y, p; χ = 1e-9, a_updraft = 0.1)

Give every sea salt bin a uniform specific concentration `χ` in the grid mean
and in each updraft, and every updraft the area fraction `a_updraft`, then
refresh the precomputed quantities so the subdomain states match. Returns the per-bin grid-mean tracer fields.
"""
function seed_sea_salt!(Y, p; χ = FT(1e-9), a_updraft = FT(0.1))
    sslt = p.atmos.seasalt
    n = CA.n_mass_flux_subdomains(p.atmos.turbconv_model)
    for j in 1:n
        @. Y.c.sgsʲs.:($$j).ρa = a_updraft * Y.c.ρ
    end
    ρχ_fields = map(name -> getproperty(Y.c, Symbol(:ρ, name)), CA.bin_names(sslt))
    for (name, ᶜρχ) in zip(CA.bin_names(sslt), ρχ_fields)
        @. ᶜρχ = χ * Y.c.ρ
        for j in 1:n
            ᶜχʲ = getproperty(Y.c.sgsʲs.:($j), name)
            @. ᶜχʲ = χ
        end
    end
    CA.set_implicit_precomputed_quantities!(Y, p, FT(0))
    return ρχ_fields
end

@testset "Subdomain settling (PrognosticEDMFX column)" begin
    (; Y, p) = generate_test_simulation(edmf_column_config("sslt_subdomain_settling"))
    FTc = eltype(Y)
    sslt = p.atmos.seasalt
    n_levels = Spaces.nlevels(axes(Y.c))
    ρχ_fields = seed_sea_salt!(Y, p)

    Yₜ = zero(Y)
    CA.aerosol_settling_tendency!(Yₜ, Y, p, FTc(0))
    for name in CA.bin_names(sslt)
        ᶜρχₜ = getproperty(Yₜ.c, Symbol(:ρ, name))
        ᶜχʲₜ = getproperty(Yₜ.c.sgsʲs.:(1), name)
        # Grid mean: net column loss through the free-outflow surface, top
        # cell loses mass, everything finite.
        @test sum(ᶜρχₜ) < 0
        @test all(<(0), parent(Fields.level(ᶜρχₜ, n_levels)))
        @test !any(isnan, parent(ᶜρχₜ))
        # Updraft: the within-updraft convergence is active and finite; with
        # a uniform concentration the top cell loses mass here too.
        @test !any(isnan, parent(ᶜχʲₜ))
        @test any(!iszero, parent(ᶜχʲₜ))
        @test all(<(0), parent(Fields.level(ᶜχʲₜ, n_levels)))
    end

    # Zero tracer everywhere ⇒ exactly zero tendency in both subdomains.
    for (name, ᶜρχ) in zip(CA.bin_names(sslt), ρχ_fields)
        ᶜχʲ = getproperty(Y.c.sgsʲs.:(1), name)
        @. ᶜρχ = FTc(0)
        @. ᶜχʲ = FTc(0)
    end
    Yₜ = zero(Y)
    CA.aerosol_settling_tendency!(Yₜ, Y, p, FTc(0))
    for name in CA.bin_names(sslt)
        @test all(iszero, parent(getproperty(Yₜ.c, Symbol(:ρ, name))))
        @test all(iszero, parent(getproperty(Yₜ.c.sgsʲs.:(1), name)))
    end
end
