using Test
import ClimaAtmos.COSP.COSPCloudSatOptics as CCO
import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics.Parameters as CMP
using ClimaCore: Domains, Meshes, Spaces, Fields, Geometry

function make_center_field(FT; value, nelems = 3)
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0),
        Geometry.ZPoint{FT}(1000);
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain, nelems = nelems)
    face_space = Spaces.FaceFiniteDifferenceSpace(z_mesh)
    center_space = Spaces.CenterFiniteDifferenceSpace(face_space)

    field = Fields.Field(FT, center_space)
    @. field = FT(value)
    return field
end

function make_center_profile_field(FT, profile)
    field = make_center_field(FT; value = 0, nelems = length(profile))
    for (ilev, value) in enumerate(profile)
        Fields.level(field, ilev) .= FT(value)
    end
    return field
end

function make_hydrometeor_fields(FT, nelems; value = 0)
    return (;
        q_lcl = make_center_field(FT; value, nelems),
        q_icl = make_center_field(FT; value, nelems),
        q_rai = make_center_field(FT; value, nelems),
        q_sno = make_center_field(FT; value, nelems),
    )
end

function make_grid_size_fields(FT, nelems; value = 0)
    return (;
        r_lcl = make_center_field(FT; value, nelems),
        lambda_inv_icl = make_center_field(FT; value, nelems),
        lambda_inv_rai = make_center_field(FT; value, nelems),
        lambda_inv_sno = make_center_field(FT; value, nelems),
    )
end

function make_thermo_state(FT)
    return (;
        p = make_center_profile_field(FT, [100000, 90000, 80000]),
        T = make_center_profile_field(FT, [290, 280, 270]),
        qv = make_center_profile_field(FT, [0.012, 0.008, 0.004]),
    )
end

make_rho_air(FT) = make_center_profile_field(FT, [1.2, 1.0, 0.8])

@testset "CloudSat Clima 1M PSD parameters" begin
    for FT in (Float32, Float64)
        microphysics_params = CMP.Microphysics1MParams(FT)
        for (class, hydrometeor, phase_type) in (
            (:lcl, microphysics_params.cloud.liquid, CCO.LiquidPhase),
            (:icl, microphysics_params.cloud.ice, CCO.IcePhase),
            (:rai, microphysics_params.precip.rain, CCO.LiquidPhase),
            (:sno, microphysics_params.precip.snow, CCO.IcePhase),
        )
            params = CCO._clima_1m_psd_parameters(
                microphysics_params,
                Val(class),
            )
            @test params.hydrometeor === hydrometeor
            @test params.phase isa phase_type
            @test isbitstype(typeof(params.phase))
        end

        for class in (:icl, :sno)
            params = CCO._clima_1m_psd_parameters(
                microphysics_params,
                Val(class),
            )
            mass_params = params.hydrometeor.mass
            r = FT(1e-4)
            mass =
                mass_params.χm *
                mass_params.m0 *
                (r / mass_params.r0)^(mass_params.me + mass_params.Δm)
            expected_diameter =
                cbrt(FT(6) * mass / (FT(pi) * CCO._rho_solid_ice(FT)))

            @test CCO._scattering_diameter(r, params) ≈ expected_diameter
        end
    end
end

@testset "CloudSat grid-mean 1M sizes and subcolumn mass" begin
    for FT in (Float32, Float64)
        nelems = 3
        rho_air = make_rho_air(FT)
        microphysics_params = CMP.Microphysics1MParams(FT)
        grid_mean = make_hydrometeor_fields(FT, nelems; value = 1e-4)
        grid_mean_sizes = make_grid_size_fields(FT, nelems)

        @test isnothing(
            CCO.cloudsat_grid_mean_sizes!(
                grid_mean_sizes,
                grid_mean,
                rho_air,
                microphysics_params,
            ),
        )

        liquid = microphysics_params.cloud.liquid
        for ilev in 1:nelems
            rho = Fields.level(rho_air, ilev)[]
            q_grid = Fields.level(grid_mean.q_lcl, ilev)[]
            radius = Fields.level(grid_mean_sizes.r_lcl, ilev)[]
            expected_radius = cbrt(
                FT(3) * rho * q_grid /
                (FT(4) * FT(pi) * liquid.ρw * liquid.N_0),
            )
            @test radius ≈ expected_radius

            q_sub = FT(2) * q_grid
            number_sub =
                CCO._clima_liquid_cloud_number(q_sub, radius, rho, liquid)
            particle_mass =
                FT(4) / FT(3) * FT(pi) * liquid.ρw * radius^3
            @test number_sub * particle_mass ≈ rho * q_sub
            @test number_sub ≈ FT(2) * liquid.N_0
        end

        for (class, size_name, q_name) in (
            (:icl, :lambda_inv_icl, :q_icl),
            (:rai, :lambda_inv_rai, :q_rai),
            (:sno, :lambda_inv_sno, :q_sno),
        )
            params =
                CCO._clima_1m_psd_parameters(microphysics_params, Val(class))
            q_grid = getproperty(grid_mean, q_name)
            lambda_inv_grid = getproperty(grid_mean_sizes, size_name)
            for ilev in 1:nelems
                rho = Fields.level(rho_air, ilev)[]
                q = Fields.level(q_grid, ilev)[]
                lambda_inv = Fields.level(lambda_inv_grid, ilev)[]
                expected_lambda_inv = CM1.lambda_inverse(
                    params.hydrometeor.pdf,
                    params.hydrometeor.mass,
                    q,
                    rho,
                )
                @test lambda_inv ≈ expected_lambda_inv

                q_sub = FT(2) * q
                n0_sub = CCO._discrete_psd_intercept(
                    q_sub,
                    lambda_inv,
                    rho,
                    params,
                )
                n0_grid = CCO._discrete_psd_intercept(
                    q,
                    lambda_inv,
                    rho,
                    params,
                )
                @test n0_sub ≈ FT(2) * n0_grid

                binned_mass = zero(FT)
                r_prev = FT(1e-7)
                log_step = exp(log(FT(5e-3) / r_prev) / FT(40))
                for _ in 1:40
                    r_next = r_prev * log_step
                    r_mid = sqrt(r_prev * r_next)
                    dr = r_next - r_prev
                    bin_number =
                        n0_sub * exp(-r_mid / lambda_inv) * dr
                    binned_mass +=
                        CCO._particle_mass(r_mid, params) * bin_number
                    r_prev = r_next
                end
                @test binned_mass ≈ rho * q_sub rtol = FT(2e-5)
            end
        end
    end
end

@testset "CloudSat streamed optics" begin
    for FT in (Float32, Float64)
        nelems = 3
        thermo_state = make_thermo_state(FT)
        rho_air = make_rho_air(FT)
        microphysics_params = CMP.Microphysics1MParams(FT)
        grid_mean = make_hydrometeor_fields(FT, nelems; value = 1e-4)
        grid_mean_sizes = make_grid_size_fields(FT, nelems)
        CCO.cloudsat_grid_mean_sizes!(
            grid_mean_sizes,
            grid_mean,
            rho_air,
            microphysics_params,
        )
        radar_cfg = CCO.CloudSatRadarConfig(FT; use_gas_abs = true)
        g_vol = make_center_field(FT; value = 999, nelems)

        @test isnothing(
            CCO.cloudsat_gas_attenuation!(
                g_vol,
                thermo_state.T,
                thermo_state.p,
                thermo_state.qv,
                radar_cfg,
            ),
        )
        @test all(isfinite, parent(g_vol))
        @test any(>(0), parent(g_vol))
        hydrometeor_optics_type = @NamedTuple{z_vol::FT, kr_vol::FT}
        hydrometeor_optics = similar(thermo_state.T, hydrometeor_optics_type)
        z_vol = hydrometeor_optics.z_vol
        kr_vol = hydrometeor_optics.kr_vol
        z_vol .= FT(999)
        kr_vol .= FT(999)
        @test eltype(hydrometeor_optics) == hydrometeor_optics_type
        hydrometeors = make_hydrometeor_fields(FT, nelems; value = 0)
        for q_name in keys(hydrometeors)
            getproperty(hydrometeors, q_name) .= FT(1e-4)
            @test isnothing(
                CCO.cloudsat_optics_subcolumn!(
                    hydrometeor_optics,
                    hydrometeors,
                    grid_mean_sizes,
                    thermo_state.T,
                    rho_air,
                    microphysics_params,
                    radar_cfg,
                ),
            )
            @test all(isfinite, parent(z_vol))
            @test all(isfinite, parent(kr_vol))
            @test any(>(0), parent(z_vol))
            @test any(>(0), parent(kr_vol))
            getproperty(hydrometeors, q_name) .= zero(FT)
        end

        # A clear subcolumn must overwrite the reusable work fields.
        CCO.cloudsat_optics_subcolumn!(
            hydrometeor_optics,
            hydrometeors,
            grid_mean_sizes,
            thermo_state.T,
            rho_air,
            microphysics_params,
            radar_cfg,
        )
        @test all(iszero, parent(z_vol))
        @test all(iszero, parent(kr_vol))

    end
end
