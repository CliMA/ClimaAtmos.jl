using Test
import ClimaAtmos as CA
import ClimaCore.Fields as Fields
import ClimaCore.Spaces as Spaces
import ClimaCore.Operators as Operators
import LinearAlgebra: dot
const TD = CA.TD

include(joinpath(@__DIR__, "..", "..", "test_helpers.jl"))

const FT = Float64

"""
    edmf_column_config(job_id; microphysics = "1M")

Bomex `PrognosticEDMFX` column with 1-moment (or, on request, 0-moment)
microphysics and prognostic sea salt: the configuration every subdomain test
runs on.
"""
edmf_column_config(job_id; microphysics = "1M") = CA.AtmosConfig(
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
        "microphysics_model" => microphysics,
        "z_max" => 4200,
        "z_elem" => 60,
        "z_stretch" => false,
        "perturb_initstate" => false,
        "dt" => "40secs",
        "prognostic_aerosols" => ["SSLT"],
        "toml" => [
            microphysics == "1M" ? "toml/prognostic_edmfx_1M.toml" :
            "toml/prognostic_edmfx.toml",
        ],
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

@testset "Dry deposition surface sink (PrognosticEDMFX column)" begin
    (; Y, p) = generate_test_simulation(edmf_column_config("sslt_subdomain_drydep"))
    FTc = eltype(Y)
    sslt = p.atmos.seasalt
    ρχ_fields = seed_sea_salt!(Y, p)
    # Realistic surface conditions so the deposition velocity is active.
    @. p.precomputed.sfc_conditions.ustar = FTc(0.3)
    @. p.precomputed.sfc_conditions.obukhov_length = FTc(-50)

    CA.set_sslt_dry_deposition_fluxes!(Y, p, sslt)
    Yₜ = zero(Y)
    CA.aerosol_deposition_tendency!(Yₜ, Y, p, FTc(0))
    (; surface_ct3_unit) = p.core
    for name in CA.bin_names(sslt)
        ᶜρχₜ = getproperty(Yₜ.c, Symbol(:ρ, name))
        ᶜχʲₜ = getproperty(Yₜ.c.sgsʲs.:(1), name)
        # Grid mean and updraft: a sink confined to the lowest cell.
        @test all(<(0), parent(Fields.level(ᶜρχₜ, 1)))
        @test all(iszero, parent(Fields.level(ᶜρχₜ, 2)))
        @test all(<(0), parent(Fields.level(ᶜχʲₜ, 1)))
        @test all(iszero, parent(Fields.level(ᶜχʲₜ, 2)))
        @test !any(isnan, parent(ᶜρχₜ)) && !any(isnan, parent(ᶜχʲₜ))
        # Cached flux points downward (positive toward the surface).
        flux = getproperty(p.tracers.sslt_drydep_fluxes, Symbol(:ρ, name))
        @test all(>(0), parent(@. -dot(flux, surface_ct3_unit)))
    end
    # Zero tracer ⇒ exactly zero fluxes and tendencies.
    for ᶜρχ in ρχ_fields
        @. ᶜρχ = FTc(0)
    end
    CA.set_sslt_dry_deposition_fluxes!(Y, p, sslt)
    Yₜ = zero(Y)
    CA.aerosol_deposition_tendency!(Yₜ, Y, p, FTc(0))
    for name in CA.bin_names(sslt)
        @test all(
            iszero,
            parent(getproperty(p.tracers.sslt_drydep_fluxes, Symbol(:ρ, name))),
        )
        @test all(iszero, parent(getproperty(Yₜ.c, Symbol(:ρ, name))))
        @test all(iszero, parent(getproperty(Yₜ.c.sgsʲs.:(1), name)))
    end
end

"""
    seed_rain!(Y, p; q_rai_gm, q_rai_updraft)

Put rain into the grid mean and the (single) updraft so the environment
residual rain differs from the updraft rain, then refresh the implicit
precomputed quantities.
"""
function seed_rain!(Y, p; q_rai_gm = FT(1e-4), q_rai_updraft = FT(5e-4))
    @. Y.c.ρq_rai = q_rai_gm * Y.c.ρ
    @. Y.c.sgsʲs.:(1).q_rai = q_rai_updraft
    CA.set_implicit_precomputed_quantities!(Y, p, FT(0))
    return nothing
end

@testset "Subdomain below-cloud washout (PrognosticEDMFX column)" begin
    (; Y, p) = generate_test_simulation(edmf_column_config("sslt_subdomain_washout"))
    FTc = eltype(Y)
    sslt = p.atmos.seasalt
    @test p.atmos.microphysics_model isa CA.NonEquilibriumMicrophysics1M
    @test hasproperty(p.tracers, :sslt_wetdep_ratesʲs)
    ρχ_fields = seed_sea_salt!(Y, p)
    seed_rain!(Y, p)

    CA.set_sslt_wet_deposition_rates!(Y, p)
    cmp = CA.CAP.microphysics_1m_params(p.params)
    ap = CA.CAP.prognostic_aerosol_params(p.params)
    (; ᶜρʲs) = p.precomputed
    for (i, name) in enumerate(CA.bin_names(sslt))
        ᶜk = getproperty(p.tracers.sslt_wetdep_rates, Symbol(:ρ, name))
        ᶜkʲ = getproperty(p.tracers.sslt_wetdep_ratesʲs, Symbol(:ρ, name))[1]
        @test all(x -> isfinite(x) && x > 0, parent(ᶜk))
        @test all(x -> isfinite(x) && x > 0, parent(ᶜkʲ))
        # The updraft rate is the washout rate of the updraft's own rain.
        E = FTc(ap.ssa_E_coll[i])
        ᶜkʲ_ref = @. E * CA.rain_swept_collection_rate(
            Y.c.sgsʲs.:(1).q_rai, ᶜρʲs.:(1), cmp.precip.rain, cmp.terminal_velocity.rain,
        )
        @test parent(ᶜkʲ) ≈ parent(ᶜkʲ_ref)
        # The updraft holds more rain than the environment residual, so its
        # rate exceeds the mass-weighted grid-mean rate, which in turn exceeds
        # a rate evaluated on the (drier) environment residual alone.
        ᶜq_rai⁰ = CA.ᶜspecific_env_value(CA.MatrixFields.@name(q_rai), Y, p)
        ᶜk⁰_ref = @. E * CA.rain_swept_collection_rate(
            ᶜq_rai⁰, Y.c.ρ, cmp.precip.rain, cmp.terminal_velocity.rain,
        )
        @test all(parent(ᶜkʲ) .> parent(ᶜk))
        @test all(parent(ᶜk) .> FTc(0.999) .* parent(ᶜk⁰_ref))
    end

    # Tendencies: sinks in both subdomains, exponential form bounded by the
    # available mass per step.
    dt = float(p.dt)
    Yₜ = zero(Y)
    CA.aerosol_wet_deposition_tendency!(Yₜ, Y, p, FTc(0))
    for (name, ᶜρχ) in zip(CA.bin_names(sslt), ρχ_fields)
        ᶜρχₜ = getproperty(Yₜ.c, Symbol(:ρ, name))
        ᶜχʲ = getproperty(Y.c.sgsʲs.:(1), name)
        ᶜχʲₜ = getproperty(Yₜ.c.sgsʲs.:(1), name)
        @test all(<(0), parent(ᶜρχₜ))
        @test all(<(0), parent(ᶜχʲₜ))
        @test all(parent(ᶜρχₜ) .* dt .≥ -parent(ᶜρχ))
        @test all(parent(ᶜχʲₜ) .* dt .≥ -parent(ᶜχʲ))
    end

    # No rain anywhere ⇒ zero rates and zero tendencies in both subdomains.
    seed_rain!(Y, p; q_rai_gm = FTc(0), q_rai_updraft = FTc(0))
    CA.set_sslt_wet_deposition_rates!(Y, p)
    Yₜ = zero(Y)
    CA.aerosol_wet_deposition_tendency!(Yₜ, Y, p, FTc(0))
    for name in CA.bin_names(sslt)
        @test all(
            iszero,
            parent(getproperty(p.tracers.sslt_wetdep_rates, Symbol(:ρ, name))),
        )
        @test all(
            iszero,
            parent(getproperty(p.tracers.sslt_wetdep_ratesʲs, Symbol(:ρ, name))[1]),
        )
        @test all(iszero, parent(getproperty(Yₜ.c, Symbol(:ρ, name))))
        @test all(iszero, parent(getproperty(Yₜ.c.sgsʲs.:(1), name)))
    end
end

@testset "Subdomain washout under 0M (PrognosticEDMFX column)" begin
    (; Y, p) = generate_test_simulation(
        edmf_column_config("sslt_subdomain_washout_0m"; microphysics = "0M"),
    )
    FTc = eltype(Y)
    sslt = p.atmos.seasalt
    @test p.atmos.microphysics_model isa CA.EquilibriumMicrophysics0M
    ρχ_fields = seed_sea_salt!(Y, p)

    # Prescribe a warm precipitating layer in the cached 0M sink (the shadow
    # is a column quantity, so no subdomain rain is needed).
    n_levels = Spaces.nlevels(axes(Y.c))
    k_lo, k_hi = n_levels ÷ 2, n_levels ÷ 2 + 4
    (; ᶜT, ᶜρ_dq_tot_dt, surface_rain_flux) = p.precomputed
    @. ᶜT = FTc(290)
    ᶜz = Fields.coordinate_field(Y.c).z
    z_lo = parent(Fields.level(ᶜz, k_lo))[1]
    z_hi = parent(Fields.level(ᶜz, k_hi))[1]
    @. ᶜρ_dq_tot_dt = ifelse(z_lo <= ᶜz <= z_hi, -FTc(1e-6), FTc(0))
    Operators.column_integral_definite!(surface_rain_flux, ᶜρ_dq_tot_dt)
    @test parent(surface_rain_flux)[1] < 0

    CA.set_sslt_wet_deposition_rates!(Y, p)
    for name in CA.bin_names(sslt)
        ᶜk = getproperty(p.tracers.sslt_wetdep_rates, Symbol(:ρ, name))
        ᶜkʲ = getproperty(p.tracers.sslt_wetdep_ratesʲs, Symbol(:ρ, name))[1]
        k = vec(parent(ᶜk))
        @test all(x -> isfinite(x) && x >= 0, k)
        @test all(>(0), k[1:(k_lo - 1)]) && all(iszero, k[(k_hi + 1):end])
        # Every subdomain sees the same column shadow.
        @test parent(ᶜkʲ) == parent(ᶜk)
    end

    # Tendencies: grid mean and updraft both sink below the layer only,
    # bounded by the available mass per step.
    dt = float(p.dt)
    Yₜ = zero(Y)
    CA.aerosol_wet_deposition_tendency!(Yₜ, Y, p, FTc(0))
    for (name, ᶜρχ) in zip(CA.bin_names(sslt), ρχ_fields)
        ᶜρχₜ = getproperty(Yₜ.c, Symbol(:ρ, name))
        ᶜχʲ = getproperty(Y.c.sgsʲs.:(1), name)
        ᶜχʲₜ = getproperty(Yₜ.c.sgsʲs.:(1), name)
        @test all(<(0), parent(Fields.level(ᶜρχₜ, 1)))
        @test all(<(0), parent(Fields.level(ᶜχʲₜ, 1)))
        @test all(iszero, parent(Fields.level(ᶜρχₜ, n_levels)))
        @test all(parent(ᶜρχₜ) .* dt .≥ -parent(ᶜρχ))
        @test all(parent(ᶜχʲₜ) .* dt .≥ -parent(ᶜχʲ))
    end
    column_sink =
        -sum(
            name -> sum(parent(getproperty(Yₜ.c, Symbol(:ρ, name)))),
            CA.bin_names(sslt),
        )
    @test column_sink > 0
end
