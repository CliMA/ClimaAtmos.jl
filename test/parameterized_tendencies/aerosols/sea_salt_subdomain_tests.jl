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
    rain, vel = cmp.precip.rain, cmp.terminal_velocity.rain
    ap = CA.CAP.prognostic_aerosol_params(p.params)
    thp = CA.CAP.thermodynamics_params(p.params)
    grav = FTc(CA.CAP.grav(p.params))
    ρ_s = CA.CAP.prescribed_aerosol_params(p.params).seasalt_density
    (; ᶜρʲs, ᶜTʲs, ᶜp, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed
    (; sslt_air_state⁰, sslt_air_stateʲs, sslt_settling_radii, sslt_kelvin_coeffs) =
        p.tracers

    # Reference chain of each subdomain, as the rate setter evaluates it: the
    # representative drop and unit-efficiency swept rate of that subdomain's
    # rain, then the Slinn efficiency at the bin's wet settling radius on the
    # subdomain air state (filled by `set_sslt_wet_deposition_rates!`).
    ᶜq_raiʲ = @. max(FTc(0), Y.c.sgsʲs.:(1).q_rai)
    ᶜρʲ = ᶜρʲs.:(1)
    ᶜr_Dʲ = @. CA.rain_representative_drop_radius(ᶜq_raiʲ, ᶜρʲ, rain, vel)
    ᶜV_Dʲ = @. CA.rain_drop_fall_speed(ᶜr_Dʲ, ᶜρʲ, rain, vel)
    ᶜΛʲ = @. CA.rain_swept_collection_rate(ᶜq_raiʲ, ᶜρʲ, rain, vel)
    ᶜq_rai⁰ = CA.ᶜspecific_env_value(CA.MatrixFields.@name(q_rai), Y, p)
    ᶜρ⁰ = @. TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰)
    ᶜr_D⁰ = @. CA.rain_representative_drop_radius(ᶜq_rai⁰, ᶜρ⁰, rain, vel)
    ᶜV_D⁰ = @. CA.rain_drop_fall_speed(ᶜr_D⁰, ᶜρ⁰, rain, vel)
    ᶜΛ⁰ = @. CA.rain_swept_collection_rate(ᶜq_rai⁰, ᶜρ⁰, rain, vel)
    @test all(>(0), parent(ᶜr_Dʲ)) && all(>(0), parent(ᶜV_Dʲ))
    for (i, name) in enumerate(CA.bin_names(sslt))
        ᶜk = getproperty(p.tracers.sslt_wetdep_rates, Symbol(:ρ, name))
        ᶜkʲ = getproperty(p.tracers.sslt_wetdep_ratesʲs, Symbol(:ρ, name))[1]
        @test all(x -> isfinite(x) && x > 0, parent(ᶜk))
        @test all(x -> isfinite(x) && x > 0, parent(ᶜkʲ))
        r_settle, C_kelvin = sslt_settling_radii[i], sslt_kelvin_coeffs[i]
        # The updraft rate is the washout rate of the updraft's own rain.
        ᶜkʲ_ref = @. CA.sslt_bin_washout_rate(
            sslt_air_stateʲs[1], ᶜTʲs.:(1), r_settle, C_kelvin, ρ_s, ᶜρʲ,
            ᶜr_Dʲ, ᶜV_Dʲ, ᶜΛʲ, grav, (ap,),
        )
        @test parent(ᶜkʲ) ≈ parent(ᶜkʲ_ref)
        # E ≤ 1, so no rate exceeds the swept rate of its own rain.
        @test all(parent(ᶜkʲ) .<= parent(ᶜΛʲ))
        # The updraft holds more rain than the environment residual, so its
        # rate exceeds the mass-weighted grid-mean rate, which in turn lies
        # above the rate of the (drier) environment residual alone.
        ᶜk⁰_ref = @. CA.sslt_bin_washout_rate(
            sslt_air_state⁰, ᶜT⁰, r_settle, C_kelvin, ρ_s, ᶜρ⁰,
            ᶜr_D⁰, ᶜV_D⁰, ᶜΛ⁰, grav, (ap,),
        )
        @test all(parent(ᶜkʲ) .> parent(ᶜk))
        @test all(parent(ᶜk) .>= FTc(1 - 1e-6) .* parent(ᶜk⁰_ref))
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

@testset "Subdomain in-cloud scavenging (PrognosticEDMFX column)" begin
    (; Y, p) = generate_test_simulation(edmf_column_config("sslt_subdomain_incloud"))
    FTc = eltype(Y)
    sslt = p.atmos.seasalt
    a_updraft = FTc(0.1)
    ρχ_fields = seed_sea_salt!(Y, p; a_updraft)
    # Cloud liquid and rain in both subdomains, more in the updraft, so the
    # updraft is cloudy (binary indicator 1) and precipitating.
    @. Y.c.ρq_lcl = FTc(2e-4) * Y.c.ρ
    @. Y.c.sgsʲs.:(1).q_lcl = FTc(1e-3)
    seed_rain!(Y, p)
    # Full explicit precompute: cloud fraction, microphysics cache, and the
    # wet-removal rates, in production order.
    CA.set_explicit_precomputed_quantities!(Y, p, FTc(0))

    thp = CA.CAP.thermodynamics_params(p.params)
    cmp = CA.CAP.microphysics_1m_params(p.params)
    ap = CA.CAP.prognostic_aerosol_params(p.params)
    dt = float(p.dt)
    (; ᶜρʲs, ᶜTʲs, ᶜq_tot_nonnegʲs) = p.precomputed
    sgs = Y.c.sgsʲs.:(1)
    # Cloud indicator from cloud condensate only (rain is not cloud).
    ᶜFʲ = @. ifelse(
        TD.has_condensate(thp, max(FTc(0), sgs.q_lcl + sgs.q_icl)),
        FTc(1),
        FTc(0),
    )
    @test all(==(1), parent(ᶜFʲ))   # the seeded updraft is cloudy everywhere
    ᶜQʲ = @. CA.cloud_precip_formation_rate(
        cmp, thp, ᶜρʲs.:(1), ᶜTʲs.:(1), ᶜq_tot_nonnegʲs.:(1),
        sgs.q_lcl, sgs.q_icl, sgs.q_rai, sgs.q_sno,
    )
    @test all(>(0), parent(ᶜQʲ))
    rain, vel = cmp.precip.rain, cmp.terminal_velocity.rain
    grav = FTc(CA.CAP.grav(p.params))
    ρ_s = CA.CAP.prescribed_aerosol_params(p.params).seasalt_density
    (; sslt_air_stateʲs, sslt_settling_radii, sslt_kelvin_coeffs) = p.tracers
    ᶜq_raiʲ = @. max(FTc(0), sgs.q_rai)
    ᶜr_Dʲ = @. CA.rain_representative_drop_radius(ᶜq_raiʲ, ᶜρʲs.:(1), rain, vel)
    ᶜV_Dʲ = @. CA.rain_drop_fall_speed(ᶜr_Dʲ, ᶜρʲs.:(1), rain, vel)
    for (i, name) in enumerate(CA.bin_names(sslt))
        r_settle, C_kelvin = sslt_settling_radii[i], sslt_kelvin_coeffs[i]
        ᶜk = getproperty(p.tracers.sslt_wetdep_rates, Symbol(:ρ, name))
        ᶜkʲ = getproperty(p.tracers.sslt_wetdep_ratesʲs, Symbol(:ρ, name))[1]
        # Updraft: the shared formula on the updraft's own ingredients. A
        # cloudy updraft has no cloud-free washout term.
        ᶜkʲ_ref = @. CA.sslt_in_cloud_scavenging_rate(
            ᶜFʲ, FTc(1), ᶜQʲ, sgs.q_lcl + sgs.q_icl, dt,
        ) + CA.sslt_bin_washout_rate(
            sslt_air_stateʲs[1], ᶜTʲs.:(1), r_settle, C_kelvin, ρ_s, ᶜρʲs.:(1),
            ᶜr_Dʲ, ᶜV_Dʲ,
            CA.sslt_below_cloud_scavenging_rate(
                ᶜFʲ,
                FTc(1),
                CA.rain_swept_collection_rate(ᶜq_raiʲ, ᶜρʲs.:(1), rain, vel),
            ),
            grav, (ap,),
        )
        @test parent(ᶜkʲ) ≈ parent(ᶜkʲ_ref)
        @test all(x -> isfinite(x) && x > 0, parent(ᶜkʲ))
        # Grid mean: the rate is survival-weighted, so with a uniform
        # concentration the implied environment survival
        # (e^{−kΔt} − aʲe^{−kʲΔt})/a⁰ is a valid fraction — the exact
        # statement that the grid sink is the sum of the subdomain sinks.
        ᶜs⁰_implied = @. (
            exp(-(ᶜk * dt)) - a_updraft * exp(-(ᶜkʲ * dt))
        ) / (1 - a_updraft)
        @test all(x -> isfinite(x) && 0 < x ≤ 1 + eps(FTc), parent(ᶜs⁰_implied))
        @test all(x -> isfinite(x) && x ≥ 0, parent(ᶜk))
    end

    # Tendencies: sinks bounded by the available mass per step.
    Yₜ = zero(Y)
    CA.aerosol_wet_deposition_tendency!(Yₜ, Y, p, FTc(0))
    for (name, ᶜρχ) in zip(CA.bin_names(sslt), ρχ_fields)
        ᶜρχₜ = getproperty(Yₜ.c, Symbol(:ρ, name))
        ᶜχʲ = getproperty(Y.c.sgsʲs.:(1), name)
        ᶜχʲₜ = getproperty(Yₜ.c.sgsʲs.:(1), name)
        @test all(≤(0), parent(ᶜρχₜ)) && all(<(0), parent(ᶜχʲₜ))
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

"""
    sslt_0m_washout_reference(p, ᶜP, ᶜρ, ᶜT, ᶜair, r_settle, C_kelvin)

Reference unit-area (cloud-free) washout rate of one bin in one subdomain
under 0M, recomputed outside the rate setter: invert the column shadow `ᶜP`
to the 1M rain mixing ratio at the subdomain density `ᶜρ`
(`rain_mixing_ratio_from_flux`), then run the 1M chain (swept rate,
representative drop, Slinn efficiency at the bin's wet radius on the
subdomain air state `ᶜair`).
"""
function sslt_0m_washout_reference(p, ᶜP, ᶜρ, ᶜT, ᶜair, r_settle, C_kelvin)
    FTc = eltype(ᶜP)
    cmp = CA.CAP.microphysics_1m_params(p.params)
    rain, vel = cmp.precip.rain, cmp.terminal_velocity.rain
    ap = CA.CAP.prognostic_aerosol_params(p.params)
    grav = FTc(CA.CAP.grav(p.params))
    ρ_s = CA.CAP.prescribed_aerosol_params(p.params).seasalt_density
    ᶜq_rai = @. CA.rain_mixing_ratio_from_flux(ᶜP, ᶜρ, rain, vel)
    ᶜr_D = @. CA.rain_representative_drop_radius(ᶜq_rai, ᶜρ, rain, vel)
    ᶜV_D = @. CA.rain_drop_fall_speed(ᶜr_D, ᶜρ, rain, vel)
    ᶜΛ = @. CA.rain_swept_collection_rate(ᶜq_rai, ᶜρ, rain, vel)
    return @. CA.sslt_bin_washout_rate(
        ᶜair, ᶜT, r_settle, C_kelvin, ρ_s, ᶜρ, ᶜr_D, ᶜV_D, ᶜΛ, grav, (ap,),
    )
end

"""
    survival_weighted_rate(ᶜk⁰, ᶜkʲ, ᶜw⁰, dt)

Grid-mean rate whose survival `e^{−kΔt}` is the `ᶜw⁰`-weighted mean of the
environment and (single) updraft survivals, the definition
`sslt_mass_weighted_rate!` implements, for a uniform concentration where the
mass weights reduce to the subdomain air-mass fractions.
"""
survival_weighted_rate(ᶜk⁰, ᶜkʲ, ᶜw⁰, dt) = @. -log(
    ᶜw⁰ * exp(-(ᶜk⁰ * dt)) + (1 - ᶜw⁰) * exp(-(ᶜkʲ * dt)),
) / dt

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
    # Washout only first: no cloud or 0M sink in either subdomain.
    (; ᶜcloud_fraction, ᶜq_liq⁰, ᶜq_ice⁰, ᶜq_liqʲs, ᶜq_iceʲs) = p.precomputed
    (; ᶜmp_tendency⁰, ᶜmp_tendencyʲs) = p.precomputed
    @. ᶜcloud_fraction = FTc(0)
    @. ᶜq_liq⁰ = FTc(0)
    @. ᶜq_ice⁰ = FTc(0)
    @. ᶜq_liqʲs.:(1) = FTc(0)
    @. ᶜq_iceʲs.:(1) = FTc(0)
    @. ᶜmp_tendency⁰.dq_tot_dt = FTc(0)
    @. ᶜmp_tendencyʲs.:(1).dq_tot_dt = FTc(0)

    CA.set_sslt_wet_deposition_rates!(Y, p)
    # Reference ingredients: the same column shadow for every subdomain,
    # inverted at each subdomain's own density, on the air states the
    # setter just refilled.
    thp = CA.CAP.thermodynamics_params(p.params)
    dt = float(p.dt)
    (; ᶜp, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜρʲs, ᶜTʲs) = p.precomputed
    (; sslt_air_state⁰, sslt_air_stateʲs, sslt_settling_radii, sslt_kelvin_coeffs) =
        p.tracers
    ᶜP = similar(ᶜT⁰)
    CA.set_sslt_precipitation_shadow!(ᶜP, Y, p)
    @test all(>(0), parent(ᶜP)[1:k_hi])
    ᶜρ⁰ = @. TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰)
    ᶜw⁰ = @. (Y.c.ρ - Y.c.sgsʲs.:(1).ρa) / Y.c.ρ
    for (i, name) in enumerate(CA.bin_names(sslt))
        r_settle, C_kelvin = sslt_settling_radii[i], sslt_kelvin_coeffs[i]
        ᶜk = getproperty(p.tracers.sslt_wetdep_rates, Symbol(:ρ, name))
        ᶜkʲ = getproperty(p.tracers.sslt_wetdep_ratesʲs, Symbol(:ρ, name))[1]
        k = vec(parent(ᶜk))
        @test all(x -> isfinite(x) && x >= 0, k)
        @test all(>(0), k[1:(k_lo - 1)]) && all(iszero, k[(k_hi + 1):end])
        # Cloud-free everywhere, so each subdomain's rate is its Slinn-chain
        # washout of the shadow rain, and the grid mean is survival-weighted.
        ᶜkʲ_ref = sslt_0m_washout_reference(
            p, ᶜP, ᶜρʲs.:(1), ᶜTʲs.:(1), sslt_air_stateʲs[1], r_settle, C_kelvin,
        )
        ᶜk⁰_ref = sslt_0m_washout_reference(
            p, ᶜP, ᶜρ⁰, ᶜT⁰, sslt_air_state⁰, r_settle, C_kelvin,
        )
        @test parent(ᶜkʲ) ≈ parent(ᶜkʲ_ref)
        ᶜk_ref = survival_weighted_rate(ᶜk⁰_ref, ᶜkʲ_ref, ᶜw⁰, dt)
        @test isapprox(parent(ᶜk), parent(ᶜk_ref); rtol = 1e-6)
    end

    # In-cloud phase: in the layer the updraft is cloudy and precipitating
    # (binary indicator 1, so no washout there) and the environment is
    # half-cloudy with its own weaker sink; ᶜcloud_fraction = a⁰F⁰ + aʲ·1[condʲ].
    a_updraft = FTc(0.1)
    Qʲ, q_cldʲ = FTc(1e-7), FTc(1e-3)
    Q⁰, q_cld⁰, F⁰ = FTc(2e-8), FTc(2e-4), FTc(0.5)
    @. ᶜq_liqʲs.:(1) = ifelse(z_lo <= ᶜz <= z_hi, q_cldʲ, FTc(0))
    @. ᶜmp_tendencyʲs.:(1).dq_tot_dt = ifelse(z_lo <= ᶜz <= z_hi, -Qʲ, FTc(0))
    @. ᶜq_liq⁰ = ifelse(z_lo <= ᶜz <= z_hi, q_cld⁰, FTc(0))
    @. ᶜmp_tendency⁰.dq_tot_dt = ifelse(z_lo <= ᶜz <= z_hi, -Q⁰, FTc(0))
    @. ᶜcloud_fraction = ifelse(
        z_lo <= ᶜz <= z_hi,
        (1 - a_updraft) * F⁰ + a_updraft,
        FTc(0),
    )
    CA.set_sslt_wet_deposition_rates!(Y, p)
    # Environment ingredients as the setter recovers them: density from the
    # (now cloudy) environment state, and the cloud fraction F⁰ from the
    # area-weighted grid-mean value minus the binary updraft cloud area.
    ᶜρ⁰ = @. TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰)
    ᶜaʲ_cld = @. ifelse(
        z_lo <= ᶜz <= z_hi,
        CA.draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)),
        FTc(0),
    )
    ᶜa⁰ = @. CA.draft_area(Y.c.ρ - Y.c.sgsʲs.:(1).ρa, ᶜρ⁰)
    # Zero outside the layer, where ᶜcloud_fraction is zero.
    ᶜF⁰ = @. min(max((ᶜcloud_fraction - ᶜaʲ_cld) / ᶜa⁰, FTc(0)), FTc(1))
    ᶜc⁰ = @. ifelse(z_lo <= ᶜz <= z_hi, ᶜF⁰ * Q⁰ / q_cld⁰, FTc(0))
    for (i, name) in enumerate(CA.bin_names(sslt))
        r_settle, C_kelvin = sslt_settling_radii[i], sslt_kelvin_coeffs[i]
        ᶜk = getproperty(p.tracers.sslt_wetdep_rates, Symbol(:ρ, name))
        ᶜkʲ = getproperty(p.tracers.sslt_wetdep_ratesʲs, Symbol(:ρ, name))[1]
        k, kʲ = vec(parent(ᶜk)), vec(parent(ᶜkʲ))
        # Cloudy updraft: pure conversion rate, no washout.
        @test all(x -> x ≈ Qʲ / q_cldʲ, kʲ[k_lo:k_hi])
        # Below the layer the updraft is cloud-free under the shadow.
        ᶜwashʲ = sslt_0m_washout_reference(
            p, ᶜP, ᶜρʲs.:(1), ᶜTʲs.:(1), sslt_air_stateʲs[1], r_settle, C_kelvin,
        )
        @test kʲ[1:(k_lo - 1)] ≈ vec(parent(ᶜwashʲ))[1:(k_lo - 1)]
        @test all(>(0), k[1:(k_lo - 1)])
        # Grid mean: survival-weighted over the updraft's own rate and the
        # environment rate F⁰·Q⁰/q_cld⁰ + (1−F⁰)·washout⁰, with the
        # environment washout from the same shadow at the environment density.
        ᶜwash⁰ = sslt_0m_washout_reference(
            p, ᶜP, ᶜρ⁰, ᶜT⁰, sslt_air_state⁰, r_settle, C_kelvin,
        )
        ᶜk⁰_ref = @. ᶜc⁰ + (1 - ᶜF⁰) * ᶜwash⁰
        ᶜk_ref = survival_weighted_rate(ᶜk⁰_ref, ᶜkʲ, ᶜw⁰, dt)
        @test isapprox(parent(ᶜk), parent(ᶜk_ref); rtol = 1e-6)
        @test all(iszero, k[(k_hi + 1):end])
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
