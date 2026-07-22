using Test
import ClimaAtmos.COSP.COSPCloudSatOptics as CCO
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

function make_thermo_state(FT)
    return (;
        p = make_center_profile_field(FT, [100000, 90000, 80000]),
        T = make_center_profile_field(FT, [290, 280, 270]),
        qv = make_center_profile_field(FT, [0.012, 0.008, 0.004]),
    )
end

make_rho_air(FT) = make_center_profile_field(FT, [1.2, 1.0, 0.8])

const COSP_MIE_REFERENCES = (;
    water_small = (;
        phase = :liquid,
        D_m = 1.9999999494757503E-005,
        T = 2.7314999389648438E+002,
        number_m3 = 1.0000000000000000E+000,
        m = ComplexF64(2.7999734878540039E+000, -1.3627929687500000E+000),
        qext = 1.4807554893195629E-002,
        qbsca = 4.1043065834855952E-007,
        z_vol = 5.8123325596959319E-011,
        kr_vol = 2.0203076545044496E-008,
        err = 0,
    ),
    water_medium = (;
        phase = :liquid,
        D_m = 1.0000000474974513E-003,
        T = 2.7314999389648438E+002,
        number_m3 = 1.0000000000000000E+000,
        m = ComplexF64(2.7999734878540039E+000, -1.3627929687500000E+000),
        qext = 3.2664134502410889E+000,
        qbsca = 1.4222202301025391E+000,
        z_vol = 5.0352096557617188E-001,
        kr_vol = 1.1141545139253139E-002,
        err = 0,
    ),
    ice_small = (;
        phase = :ice,
        D_m = 1.9999999494757503E-005,
        T = 2.5314999389648438E+002,
        number_m3 = 1.0000000000000000E+000,
        m = ComplexF64(1.7825161218643188E+000, -2.8795218095183372E-003),
        qext = 9.0639849076978862E-005,
        qbsca = 1.0656524551677649E-007,
        z_vol = 1.5091284125134941E-011,
        kr_vol = 1.2366685453457649E-010,
        err = 0,
    ),
    ice_medium = (;
        phase = :ice,
        D_m = 1.0000000474974513E-003,
        T = 2.5314999389648438E+002,
        number_m3 = 1.0000000000000000E+000,
        m = ComplexF64(1.7825161218643188E+000, -2.8795218095183372E-003),
        qext = 4.8764702677726746E-001,
        qbsca = 3.8374635577201843E-001,
        z_vol = 1.3586105406284332E-001,
        kr_vol = 1.6633353661745787E-003,
        err = 0,
    ),
)

function test_or_skip_isapprox(actual, expected; rtol, atol)
    if ismissing(expected)
        @test_skip "COSP scalar Mie reference value not filled yet"
    else
        @test isapprox(actual, expected; rtol, atol)
    end
end

@testset "COSP CloudSat scalar Mie references" begin
    FT = Float64
    radar_cfg = CCO.CloudSatRadarConfig(FT; use_gas_abs = false)

    for (case_name, ref) in pairs(COSP_MIE_REFERENCES)
        @testset "$(case_name)" begin
            D_m = FT(ref.D_m)
            T = FT(ref.T)
            number_m3 = FT(ref.number_m3)

            m =
                ref.phase === :liquid ?
                CCO._m_wat(radar_cfg.freq, T) :
                CCO._m_ice(radar_cfg.freq, T)
            qext, qbsca = CCO._mie_efficiencies(D_m, T, radar_cfg, ref.phase)
            z_vol, kr_vol = CCO._zeff_particle_integral(
                D_m,
                number_m3,
                T,
                radar_cfg,
                ref.phase,
            )

            test_or_skip_isapprox(m, ref.m; rtol = 1e-6, atol = 1e-12)
            test_or_skip_isapprox(qext, ref.qext; rtol = 1e-5, atol = 1e-12)
            test_or_skip_isapprox(qbsca, ref.qbsca; rtol = 1e-5, atol = 1e-12)
            test_or_skip_isapprox(z_vol, ref.z_vol; rtol = 1e-5, atol = 1e-12)
            test_or_skip_isapprox(kr_vol, ref.kr_vol; rtol = 1e-5, atol = 1e-12)
        end
    end
end

@testset "COSP CloudSat Clima 1M PSD parameters" begin
    for FT in (Float32, Float64)
        microphysics_params = CMP.Microphysics1MParams(FT)
        for (class, hydrometeor) in (
            (:icl, microphysics_params.cloud.ice),
            (:rai, microphysics_params.precip.rain),
            (:sno, microphysics_params.precip.snow),
        )
            params = CCO._clima_1m_psd_parameters(
                microphysics_params,
                Val(class),
            )
            @test params.hydrometeor === hydrometeor
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

@testset "COSP CloudSat gas absorption matches COSPv2" begin
    FT = Float64
    thermo_state = (;
        p = make_center_profile_field(FT, [100000, 50000, 20000]),
        T = make_center_profile_field(FT, [290, 260, 220]),
        qv = make_center_profile_field(FT, [0.012, 0.001, 1.0e-5]),
    )
    g_vol = make_center_field(FT; value = 999, nelems = 3)
    radar_cfg = CCO.CloudSatRadarConfig(FT; use_gas_abs = true)

    CCO.cloudsat_gas_attenuation!(
        g_vol,
        thermo_state.T,
        thermo_state.p,
        thermo_state.qv,
        radar_cfg,
    )

    # Reference values generated from COSPv2 quickbeam_optics.F90 `gases`.
    ref_g_vol = FT[
        1.381359726862E-4,
        2.007164393762E-5,
        5.921524172503E-6,
    ]
    @test all(isfinite, parent(g_vol))
    @test all(>=(0), parent(g_vol))
    @test isapprox(parent(g_vol), ref_g_vol; rtol = 1e-5, atol = 1e-12)

    radar_cfg = CCO.CloudSatRadarConfig(FT; use_gas_abs = false)
    CCO.cloudsat_gas_attenuation!(
        g_vol,
        thermo_state.T,
        thermo_state.p,
        thermo_state.qv,
        radar_cfg,
    )
    @test all(iszero, parent(g_vol))
end

@testset "COSP CloudSat streamed optics" begin
    for FT in (Float32, Float64)
        nelems = 3
        thermo_state = make_thermo_state(FT)
        rho_air = make_rho_air(FT)
        microphysics_params = CMP.Microphysics1MParams(FT)
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
        z_vol = make_center_field(FT; value = 999, nelems)
        kr_vol = make_center_field(FT; value = 999, nelems)
        hydrometeors = make_hydrometeor_fields(FT, nelems; value = 0)
        for q_name in keys(hydrometeors)
            getproperty(hydrometeors, q_name) .= FT(1e-4)
            @test isnothing(
                CCO.cloudsat_optics_subcolumn!(
                    z_vol,
                    kr_vol,
                    hydrometeors,
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
            z_vol,
            kr_vol,
            hydrometeors,
            thermo_state.T,
            rho_air,
            microphysics_params,
            radar_cfg,
        )
        @test all(iszero, parent(z_vol))
        @test all(iszero, parent(kr_vol))

    end
end
