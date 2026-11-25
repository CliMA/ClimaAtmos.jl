#=
Terminal velocity modes: selection from the config keys, and each (mode, species) closure
against its CloudMicrophysics reference. In-cloud sedimentation of cloud liquid and
cloud ice (`sgs_cloud_sedimentation_cf_floor`): the parameter, the in-cloud specific
humidity, and the velocities `set_precipitation_velocities!` and
`set_precomputed_quantities!` set for the grid mean and for the `PrognosticEDMFX`
environment.
=#

using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics.MicrophysicsNonEq as CMNe
import Thermodynamics as TD
import ClimaCore: Fields
import ClimaCore.MatrixFields: @name

include("../../test_helpers.jl")

@testset "Terminal velocity modes" begin
    for FT in (Float32, Float64)
        @testset "FT = $FT" begin
            base = Dict(
                "microphysics_model" => "1M",
                "FLOAT_TYPE" => string(FT),
                "output_default_diagnostics" => false,
            )
            make_water(extra) = begin
                config =
                    CA.AtmosConfig(merge(base, extra); job_id = "terminal_velocity_$FT")
                params = CA.ClimaAtmosParameters(config)
                (; water = CA.AtmosWater(config, params, FT), params)
            end

            @testset "Mode selection" begin
                (; water) = make_water(Dict())
                @test water.terminal_velocity_liquid isa CA.FixedTerminalVelocity
                @test water.terminal_velocity_ice isa CA.FixedTerminalVelocity
                @test water.terminal_velocity_rain isa CA.Chen2022TerminalVelocity
                @test water.terminal_velocity_snow isa CA.Chen2022TerminalVelocity

                (; water) = make_water(
                    Dict(
                        "cloud_liquid_terminal_velocity" => "Stokes",
                        "cloud_ice_terminal_velocity" => "Chen2022",
                        "rain_terminal_velocity" => "Chen2022",
                        "snow_terminal_velocity" => "Blk1M",
                    ),
                )
                @test water.terminal_velocity_liquid isa CA.StokesTerminalVelocity
                @test water.terminal_velocity_ice isa CA.Chen2022TerminalVelocity
                @test water.terminal_velocity_rain isa CA.Chen2022TerminalVelocity
                @test water.terminal_velocity_snow isa CA.Blk1MTerminalVelocity

                (; water) = make_water(
                    Dict(
                        "rain_terminal_velocity" => "Fixed",
                        "snow_terminal_velocity" => "Fixed",
                    ),
                )
                @test water.terminal_velocity_rain isa CA.FixedTerminalVelocity
                @test water.terminal_velocity_snow isa CA.FixedTerminalVelocity

                # closures a species does not support are rejected
                @test_throws ErrorException make_water(
                    Dict("cloud_liquid_terminal_velocity" => "Blk1M"),
                )
                @test_throws ErrorException make_water(
                    Dict("cloud_ice_terminal_velocity" => "Stokes"),
                )
            end

            @testset "Closures" begin
                (; water, params) = make_water(Dict())
                mm = water.microphysics_model
                cmc = CAP.microphysics_cloud_params(params)
                cmp = CAP.microphysics_1m_params(params)
                w(mode, name, ρ, q) =
                    CA.terminal_velocity(mm, mode, name, params, cmc, cmp, FT(ρ), FT(q))
                fixed = CA.FixedTerminalVelocity()
                blk = CA.Blk1MTerminalVelocity()
                chen = CA.Chen2022TerminalVelocity()
                stokes = CA.StokesTerminalVelocity()
                for ρ in (0.4, 1.2), q in (1e-6, 1e-3)
                    @test w(fixed, @name(q_lcl), ρ, q) ==
                          CAP.fixed_cloud_liquid_terminal_velocity(params)
                    @test w(fixed, @name(q_icl), ρ, q) ==
                          CAP.fixed_cloud_ice_terminal_velocity(params)
                    @test w(fixed, @name(q_rai), ρ, q) ==
                          CAP.fixed_rain_terminal_velocity(params)
                    @test w(fixed, @name(q_sno), ρ, q) ==
                          CAP.fixed_snow_terminal_velocity(params)
                    @test w(stokes, @name(q_lcl), ρ, q) ==
                          CMNe.terminal_velocity(cmc.liquid, cmc.stokes, FT(ρ), FT(q))
                    @test w(chen, @name(q_icl), ρ, q) ==
                          CMNe.terminal_velocity(
                        cmc.ice,
                        cmc.Ch2022.small_ice,
                        FT(ρ),
                        FT(q),
                    )
                    @test w(blk, @name(q_rai), ρ, q) == CM1.terminal_velocity(
                        cmp.precip.rain,
                        cmp.terminal_velocity.rain,
                        FT(ρ),
                        FT(q),
                    )
                    @test w(chen, @name(q_rai), ρ, q) ==
                          CM1.terminal_velocity(
                        cmp.precip.rain,
                        cmc.Ch2022.rain,
                        FT(ρ),
                        FT(q),
                    )
                    @test w(blk, @name(q_sno), ρ, q) == CM1.terminal_velocity(
                        cmp.precip.snow,
                        cmp.terminal_velocity.snow,
                        FT(ρ),
                        FT(q),
                    )
                    @test w(chen, @name(q_sno), ρ, q) == CM1.terminal_velocity(
                        cmp.precip.snow,
                        cmc.Ch2022.large_ice,
                        FT(ρ),
                        FT(q),
                    )
                end
            end

            @testset "In-cloud specific humidity" begin
                (; water, params) = make_water(
                    Dict(
                        "cloud_liquid_terminal_velocity" => "Stokes",
                        "cloud_ice_terminal_velocity" => "Chen2022",
                    ),
                )
                @test CAP.sgs_cloud_sedimentation_cf_floor(params) == one(FT)
                @test !CA.sedimentation_reads_cloud_fraction(water, params)
                for bad_floor in (zero(FT), FT(-0.1), FT(1.1))
                    @test_throws ArgumentError CA.SGSQuadratureParameters(
                        FT,
                        (; sgs_cloud_sedimentation_cf_floor = bad_floor),
                    )
                end
                cf_floor = FT(0.02)
                sgs_quad = CA.SGSQuadratureParameters(
                    FT,
                    (; sgs_cloud_sedimentation_cf_floor = cf_floor),
                )
                @test CAP.sgs_cloud_sedimentation_cf_floor(sgs_quad) == cf_floor

                @test CA.is_cloud_condensate(@name(q_lcl))
                @test CA.is_cloud_condensate(@name(q_icl))
                @test !CA.is_cloud_condensate(@name(q_rai))
                @test !CA.is_cloud_condensate(@name(q_sno))

                q = FT(2e-5)
                q_ic(cf) = CA.incloud_specific_humidity(q, FT(cf), cf_floor)
                @test q_ic(1) == q
                @test q_ic(0.5) == q / FT(0.5)
                # The floor bounds the concentration factor by 1 / cf_floor.
                @test q_ic(0.005) == q_ic(0) == q / cf_floor
                @test CA.incloud_specific_humidity(FT(-1e-5), FT(0.15), cf_floor) ==
                      zero(FT)
                # A floor of 1 returns the mean at any cloud fraction.
                @test CA.incloud_specific_humidity(q, FT(0.1), one(FT)) == q

                # The Stokes cloud liquid and Chen et al. (2022) cloud ice fall
                # speeds increase with the in-cloud specific humidity, down to
                # the cloud-fraction floor.
                mm = water.microphysics_model
                cmc = CAP.microphysics_cloud_params(params)
                cmp = CAP.microphysics_1m_params(params)
                for (mode, name) in (
                    (CA.StokesTerminalVelocity(), @name(q_lcl)),
                    (CA.Chen2022TerminalVelocity(), @name(q_icl)),
                )
                    w_ic(cf) = CA.terminal_velocity(
                        mm, mode, name, params, cmc, cmp, FT(0.35), q_ic(cf),
                    )
                    @test zero(FT) < w_ic(1) < w_ic(0.5) < w_ic(0.15) < w_ic(0.02)
                    @test w_ic(0.005) == w_ic(0.02)
                end
            end

            @testset "Shaft-weighted rain and snow velocities" begin
                (; water, params) = make_water(Dict())
                mm = water.microphysics_model
                cmc = CAP.microphysics_cloud_params(params)
                cmp = CAP.microphysics_1m_params(params)
                chen(name, ρ, q) = CA.terminal_velocity(
                    mm, CA.Chen2022TerminalVelocity(), name, params, cmc, cmp, ρ, q,
                )
                fixed(name, ρ, q) = CA.terminal_velocity(
                    mm, CA.FixedTerminalVelocity(), name, params, cmc, cmp, ρ, q,
                )
                w_shaft(tv, name, q, β_p, a_p) = CA.shaft_weighted_terminal_velocity(
                    tv, name, FT(1), FT(q), FT(β_p), FT(a_p), FT(0.1),
                )
                for name in (@name(q_rai), @name(q_sno)), q in (1e-5, 1e-3)
                    w_mean = chen(name, FT(1), FT(q))
                    # No placement, a full-cell shaft, or a constant velocity
                    # leave the velocity of the mean.
                    @test w_shaft(chen, name, q, 0, 0.3) ≈ w_mean
                    @test w_shaft(chen, name, q, 1, 1) ≈ w_mean
                    @test w_shaft(fixed, name, q, 1, 0.3) ≈ fixed(name, FT(1), FT(q))
                    # The concentrated shaft falls faster, more so for a
                    # narrower shaft, and the floor bounds the width.
                    @test w_shaft(chen, name, q, 1, 0.5) > w_mean
                    @test w_shaft(chen, name, q, 1, 0.2) > w_shaft(chen, name, q, 1, 0.5)
                    @test w_shaft(chen, name, q, 1, 0.01) == w_shaft(chen, name, q, 1, 0.1)
                    # Zone fluxes: shaft at `φ_in q`, remainder at `φ_out q`.
                    (β_p, a_s) = (FT(0.6), FT(0.25))
                    (φ_in, φ_out) = (1 - β_p + β_p / a_s, 1 - β_p)
                    flux_zones =
                        a_s * φ_in * FT(q) * chen(name, FT(1), φ_in * FT(q)) +
                        (1 - a_s) * φ_out * FT(q) * chen(name, FT(1), φ_out * FT(q))
                    @test FT(q) * w_shaft(chen, name, q, β_p, a_s) ≈ flux_zones
                    # Negative humidities count as zero.
                    @test w_shaft(chen, name, -q, 1, 0.3) == w_shaft(chen, name, 0, 1, 0.3)
                end
                @test CA.on_precip_shaft(@name(q_rai))
                @test CA.on_precip_shaft(@name(q_sno))
                @test !CA.on_precip_shaft(@name(q_lcl))
                @test !CA.on_precip_shaft(@name(q_icl))
            end
        end
    end
end

# A 1M column with state-dependent cloud liquid and cloud ice velocities, the
# cloud-fraction floor read from a toml file, and cloud condensate set by hand
# to a few percent of the total water, in proportions that vary with height.
function cloud_sedimentation_column(job_id, extra)
    cf_floor = 0.02
    toml = joinpath(mktempdir(), "cloud_sedimentation_cf_floor.toml")
    write(
        toml,
        "[sgs_cloud_sedimentation_cf_floor]\nvalue = $cf_floor\ntype = \"float\"\n",
    )
    config = CA.AtmosConfig(
        merge(
            Dict{String, Any}(
                "FLOAT_TYPE" => "Float64",
                "config" => "column",
                "microphysics_model" => "1M",
                "cloud_liquid_terminal_velocity" => "Stokes",
                "cloud_ice_terminal_velocity" => "Chen2022",
                "z_stretch" => false,
                "toml" =>
                    [joinpath(pkgdir(CA), "toml", "default_parameters.toml"), toml],
                "output_default_diagnostics" => false,
            ),
            extra,
        );
        job_id,
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    FT = eltype(Y)
    @test CAP.sgs_cloud_sedimentation_cf_floor(p.params) == FT(cf_floor)
    @test CA.sedimentation_reads_cloud_fraction(p.atmos, p.params)
    ᶜz = Fields.coordinate_field(Y.c).z
    z_top = maximum(parent(ᶜz))
    @. Y.c.ρq_lcl = FT(0.02) * (2 - ᶜz / z_top) * Y.c.ρq_tot
    @. Y.c.ρq_icl = FT(0.01) * (1 + ᶜz / z_top) * Y.c.ρq_tot
    return (; Y, p, t = simulation.integrator.t)
end

grid_mean_column(job_id) = cloud_sedimentation_column(
    job_id,
    Dict("initial_condition" => "PrecipitatingColumn"),
)
edmfx_column(job_id) = cloud_sedimentation_column(
    job_id,
    Dict(
        "initial_condition" => "Bomex",
        "turbconv" => "prognostic_edmfx",
        "prognostic_tke" => true,
        "z_max" => 3000.0,
        "z_elem" => 30,
        "dt" => "1secs",
        "t_end" => "10secs",
    ),
)

# A cloud fraction without cloud in the lower half of the column, where the
# floor applies, increasing to overcast at the top.
function set_cloud_fraction_profile!(ᶜcf, Y)
    ᶜz = Fields.coordinate_field(Y.c).z
    z_top = maximum(parent(ᶜz))
    @. ᶜcf = max(0, 2 * ᶜz / z_top - 1)
    return nothing
end

# Cloud liquid and cloud ice terminal velocities at the density `ᶜρ`, for the
# in-cloud specific humidity with the cloud fraction `ᶜcf` and its floor.
function expected_cloud_velocities(Y, p, ᶜρ, ᶜcf, cf_floor)
    params = p.params
    mm = p.atmos.microphysics_model
    cmc = CAP.microphysics_cloud_params(params)
    cmp = CAP.microphysics_1m_params(params)
    w(mode, name, ρ, q) =
        CA.terminal_velocity(mm, mode, name, params, cmc, cmp, ρ, q)
    w_liq(ρ, q) = w(CA.StokesTerminalVelocity(), @name(q_lcl), ρ, q)
    w_ice(ρ, q) = w(CA.Chen2022TerminalVelocity(), @name(q_icl), ρ, q)
    ᶜwₗ = @. w_liq(
        ᶜρ,
        CA.incloud_specific_humidity(Y.c.ρq_lcl / Y.c.ρ, ᶜcf, cf_floor),
    )
    ᶜwᵢ = @. w_ice(
        ᶜρ,
        CA.incloud_specific_humidity(Y.c.ρq_icl / Y.c.ρ, ᶜcf, cf_floor),
    )
    return (; ᶜwₗ, ᶜwᵢ)
end

# The cached velocities equal those at the in-cloud specific humidity for the
# cloud fraction `ᶜcf`, and differ from those for the cloud fraction `other_cf`.
function test_cloud_velocities(Y, p, ᶜρ, ᶜcf, other_cf)
    cf_floor = CAP.sgs_cloud_sedimentation_cf_floor(p.params)
    expected = expected_cloud_velocities(Y, p, ᶜρ, ᶜcf, cf_floor)
    other = expected_cloud_velocities(Y, p, ᶜρ, other_cf, cf_floor)
    for name in (:ᶜwₗ, :ᶜwᵢ)
        ᶜw = getproperty(p.precomputed, name)
        @test parent(ᶜw) ≈ parent(getproperty(expected, name))
        @test !(parent(ᶜw) ≈ parent(getproperty(other, name)))
    end
    return nothing
end

# Environment density of the `PrognosticEDMFX` cache.
function environment_density(p)
    (; ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed
    thp = CAP.thermodynamics_params(p.params)
    return @. TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰)
end

@testset "In-cloud velocities through `set_precipitation_velocities!`" begin
    @testset "Grid mean" begin
        (; Y, p) = grid_mean_column("cloud_sedimentation_grid_mean")
        (; ᶜcloud_fraction) = p.precomputed
        set_cloud_fraction_profile!(ᶜcloud_fraction, Y)
        CA.set_precipitation_velocities!(
            Y, p, p.atmos.microphysics_model, p.atmos.turbconv_model,
        )
        # An overcast cell has the velocities at the mean specific humidity.
        test_cloud_velocities(Y, p, Y.c.ρ, ᶜcloud_fraction, one(eltype(Y)))
    end

    @testset "PrognosticEDMFX environment" begin
        (; Y, p) = edmfx_column("cloud_sedimentation_edmfx")
        # Without updraft area, the environment holds all of the cloud
        # condensate, and the grid-mean velocity is the environment velocity.
        @. Y.c.sgsʲs.:(1).ρa = 0
        # The environment cloud fraction is the one that enters, not the
        # grid-mean cloud fraction, which is set to overcast.
        (; ᶜcloud_fraction, ᶜcloud_fraction⁰) = p.precomputed
        set_cloud_fraction_profile!(ᶜcloud_fraction⁰, Y)
        @. ᶜcloud_fraction = 1
        CA.set_precipitation_velocities!(
            Y, p, p.atmos.microphysics_model, p.atmos.turbconv_model,
        )
        test_cloud_velocities(
            Y, p, environment_density(p), ᶜcloud_fraction⁰, ᶜcloud_fraction,
        )
    end
end

# `set_precomputed_quantities!` sets the velocities from the cloud fraction it
# computes, so that the cache depends on `Y` alone: the cloud fraction that the
# cache held before the call, here one half, leaves no trace in the velocities.
@testset "In-cloud velocities through `set_precomputed_quantities!`" begin
    @testset "Grid mean" begin
        (; Y, p, t) = grid_mean_column("cloud_sedimentation_cache_grid_mean")
        FT = eltype(Y)
        (; ᶜcloud_fraction) = p.precomputed
        @. ᶜcloud_fraction = FT(0.5)
        CA.set_precomputed_quantities!(Y, p, t)
        test_cloud_velocities(Y, p, Y.c.ρ, ᶜcloud_fraction, FT(0.5))
    end

    @testset "PrognosticEDMFX environment" begin
        (; Y, p, t) = edmfx_column("cloud_sedimentation_cache_edmfx")
        FT = eltype(Y)
        @. Y.c.sgsʲs.:(1).ρa = 0
        (; ᶜcloud_fraction, ᶜcloud_fraction⁰) = p.precomputed
        @. ᶜcloud_fraction = FT(0.5)
        @. ᶜcloud_fraction⁰ = FT(0.5)
        CA.set_precomputed_quantities!(Y, p, t)
        test_cloud_velocities(
            Y, p, environment_density(p), ᶜcloud_fraction⁰, FT(0.5),
        )
    end
end
