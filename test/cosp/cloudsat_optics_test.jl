using Test
import ClimaAtmos as CA
import ClimaAtmos.COSP.COSPCloudSatOptics as CCO
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

make_subcolumn_fields(FT, nsubcolumns, nelems; value = 0) =
    ntuple(_ -> make_center_field(FT; value, nelems), nsubcolumns)

function make_hydrometeor_subcolumns(FT, nsubcolumns, nelems; value = 0)
    return (;
        q_lcl = make_subcolumn_fields(FT, nsubcolumns, nelems; value),
        q_icl = make_subcolumn_fields(FT, nsubcolumns, nelems; value),
        q_rai = make_subcolumn_fields(FT, nsubcolumns, nelems; value),
        q_sno = make_subcolumn_fields(FT, nsubcolumns, nelems; value),
    )
end

function make_cloudsat_outputs(FT, nsubcolumns, nelems)
    z_vol = make_subcolumn_fields(FT, nsubcolumns, nelems; value = 999)
    kr_vol = make_subcolumn_fields(FT, nsubcolumns, nelems; value = 999)
    g_vol = make_center_field(FT; value = 999, nelems)
    return z_vol, kr_vol, g_vol
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
        D_m =  1.9999999494757503E-005,
        T =  2.7314999389648438E+002,
        number_m3 =  1.0000000000000000E+000,
        m = ComplexF64( 2.7999734878540039E+000, -1.3627929687500000E+000),
        qext =  1.4807554893195629E-002,
        qbsca =  4.1043065834855952E-007,
        z_vol =  5.8123325596959319E-011,
        kr_vol =  2.0203076545044496E-008,
        err = 0,
    ),

    water_medium = (;
        phase = :liquid,
        D_m =  1.0000000474974513E-003,
        T =  2.7314999389648438E+002,
        number_m3 =  1.0000000000000000E+000,
        m = ComplexF64( 2.7999734878540039E+000, -1.3627929687500000E+000),
        qext =  3.2664134502410889E+000,
        qbsca =  1.4222202301025391E+000,
        z_vol =  5.0352096557617188E-001,
        kr_vol =  1.1141545139253139E-002,
        err = 0,
    ),

    ice_small = (;
        phase = :ice,
        D_m =  1.9999999494757503E-005,
        T =  2.5314999389648438E+002,
        number_m3 =  1.0000000000000000E+000,
        m = ComplexF64( 1.7825161218643188E+000, -2.8795218095183372E-003),
        qext =  9.0639849076978862E-005,
        qbsca =  1.0656524551677649E-007,
        z_vol =  1.5091284125134941E-011,
        kr_vol =  1.2366685453457649E-010,
        err = 0,
    ),

    ice_medium = (;
        phase = :ice,
        D_m =  1.0000000474974513E-003,
        T =  2.5314999389648438E+002,
        number_m3 =  1.0000000000000000E+000,
        m = ComplexF64( 1.7825161218643188E+000, -2.8795218095183372E-003),
        qext =  4.8764702677726746E-001,
        qbsca =  3.8374635577201843E-001,
        z_vol =  1.3586105406284332E-001,
        kr_vol =  1.6633353661745787E-003,
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

@testset "COSP CloudSat optics scaffold" begin
    FT = Float64
    nsubcolumns = 2
    nelems = 3

    hydrometeors =
        make_hydrometeor_subcolumns(FT, nsubcolumns, nelems; value = 0)
    thermo_state = make_thermo_state(FT)
    rho_air = make_rho_air(FT)

    @testset "zero hydrometeors with gas absorption off" begin
        z_vol, kr_vol, g_vol = make_cloudsat_outputs(FT, nsubcolumns, nelems)
        radar_cfg = CCO.CloudSatRadarConfig(FT; use_gas_abs = false)

        result = CCO.cloudsat_optics!(
            z_vol,
            kr_vol,
            g_vol,
            hydrometeors,
            thermo_state,
            rho_air,
            radar_cfg,
        )

        @test isnothing(result)
        for field in z_vol
            @test all(iszero, parent(field))
        end
        for field in kr_vol
            @test all(iszero, parent(field))
        end
        @test all(iszero, parent(g_vol))
    end

    @testset "zero hydrometeors with gas absorption on" begin
        z_vol, kr_vol, g_vol = make_cloudsat_outputs(FT, nsubcolumns, nelems)
        radar_cfg = CCO.CloudSatRadarConfig(FT; use_gas_abs = true)

        result = CCO.cloudsat_optics!(
            z_vol,
            kr_vol,
            g_vol,
            hydrometeors,
            thermo_state,
            rho_air,
            radar_cfg,
        )

        @test isnothing(result)
        for field in z_vol
            @test all(iszero, parent(field))
        end
        for field in kr_vol
            @test all(iszero, parent(field))
        end
        @test all(isfinite, parent(g_vol))
        @test all(>=(0), parent(g_vol))
        @test any(>(0), parent(g_vol))
    end

    @testset "single class activates only one subcolumn" begin
        radar_cfg = CCO.CloudSatRadarConfig(FT; use_gas_abs = false)

        for q_name in (:q_lcl, :q_icl, :q_rai, :q_sno)
            hydrometeors =
                make_hydrometeor_subcolumns(FT, nsubcolumns, nelems; value = 0)
            active_q = getproperty(hydrometeors, q_name)[1]
            @. active_q = FT(1e-4)
            z_vol, kr_vol, g_vol =
                make_cloudsat_outputs(FT, nsubcolumns, nelems)

            result = CCO.cloudsat_optics!(
                z_vol,
                kr_vol,
                g_vol,
                hydrometeors,
                thermo_state,
                rho_air,
                radar_cfg,
            )

            @test isnothing(result)
            @test all(isfinite, parent(z_vol[1]))
            @test all(isfinite, parent(kr_vol[1]))
            @test all(>=(0), parent(z_vol[1]))
            @test all(>=(0), parent(kr_vol[1]))
            @test any(>(0), parent(z_vol[1]))
            @test any(>(0), parent(kr_vol[1]))
            for isubcolumn in 2:nsubcolumns
                @test all(iszero, parent(z_vol[isubcolumn]))
                @test all(iszero, parent(kr_vol[isubcolumn]))
            end
            @test all(iszero, parent(g_vol))
        end
    end

    @testset "hydrometeor optics stay isolated by subcolumn" begin
        radar_cfg = CCO.CloudSatRadarConfig(FT; use_gas_abs = false)

        for active_subcolumn in 1:nsubcolumns
            hydrometeors =
                make_hydrometeor_subcolumns(FT, nsubcolumns, nelems; value = 0)
            @. hydrometeors.q_lcl[active_subcolumn] = FT(1e-4)
            z_vol, kr_vol, g_vol =
                make_cloudsat_outputs(FT, nsubcolumns, nelems)

            result = CCO.cloudsat_optics!(
                z_vol,
                kr_vol,
                g_vol,
                hydrometeors,
                thermo_state,
                rho_air,
                radar_cfg,
            )

            @test isnothing(result)
            @test all(isfinite, parent(z_vol[active_subcolumn]))
            @test all(isfinite, parent(kr_vol[active_subcolumn]))
            @test any(>(0), parent(z_vol[active_subcolumn]))
            @test any(>(0), parent(kr_vol[active_subcolumn]))
            for isubcolumn in 1:nsubcolumns
                if isubcolumn == active_subcolumn
                    @test all(>=(0), parent(z_vol[isubcolumn]))
                    @test all(>=(0), parent(kr_vol[isubcolumn]))
                else
                    @test all(iszero, parent(z_vol[isubcolumn]))
                    @test all(iszero, parent(kr_vol[isubcolumn]))
                end
            end
            @test all(iszero, parent(g_vol))
        end
    end

    @testset "gas absorption matches COSPv2 gases reference" begin
        thermo_state = (;
            p = make_center_profile_field(FT, [100000, 50000, 20000]),
            T = make_center_profile_field(FT, [290, 260, 220]),
            qv = make_center_profile_field(FT, [0.012, 0.001, 1.0e-5]),
        )
        rho_air = make_rho_air(FT)
        hydrometeors =
            make_hydrometeor_subcolumns(FT, nsubcolumns, nelems; value = 0)
        z_vol, kr_vol, g_vol = make_cloudsat_outputs(FT, nsubcolumns, nelems)

        radar_cfg = CCO.CloudSatRadarConfig(FT; use_gas_abs = true)

        CCO.cloudsat_optics!(
            z_vol,
            kr_vol,
            g_vol,
            hydrometeors,
            thermo_state,
            rho_air,
            radar_cfg,
        )

        # Reference values generated from COSPv2 quickbeam_optics.F90 `gases`.
        ref_g_vol = FT[
            1.381359726862E-4,
            2.007164393762E-5,
            5.921524172503E-6,
        ]

        for field in z_vol
            @test all(iszero, parent(field))
        end
        for field in kr_vol
            @test all(iszero, parent(field))
        end
        @test all(isfinite, parent(g_vol))
        @test all(>=(0), parent(g_vol))
        @test isapprox(parent(g_vol), ref_g_vol; rtol = 1e-5, atol = 1e-12)

        z_vol, kr_vol, g_vol = make_cloudsat_outputs(FT, nsubcolumns, nelems)
        radar_cfg = CCO.CloudSatRadarConfig(FT; use_gas_abs = false)

        CCO.cloudsat_optics!(
            z_vol,
            kr_vol,
            g_vol,
            hydrometeors,
            thermo_state,
            rho_air,
            radar_cfg,
        )

        @test all(iszero, parent(g_vol))
    end
end

@testset "COSP CloudSat optics Float32 smoke" begin
    FT = Float32
    nsubcolumns = 2
    nelems = 3

    thermo_state = make_thermo_state(FT)
    rho_air = make_rho_air(FT)

    @testset "zero hydrometeors with gas absorption on and off" begin
        for use_gas_abs in (false, true)
            hydrometeors =
                make_hydrometeor_subcolumns(FT, nsubcolumns, nelems; value = 0)
            z_vol, kr_vol, g_vol =
                make_cloudsat_outputs(FT, nsubcolumns, nelems)
            radar_cfg = CCO.CloudSatRadarConfig(FT; use_gas_abs)

            result = CCO.cloudsat_optics!(
                z_vol,
                kr_vol,
                g_vol,
                hydrometeors,
                thermo_state,
                rho_air,
                radar_cfg,
            )

            @test isnothing(result)
            for field in z_vol
                @test all(isfinite, parent(field))
                @test all(iszero, parent(field))
            end
            for field in kr_vol
                @test all(isfinite, parent(field))
                @test all(iszero, parent(field))
            end
            @test all(isfinite, parent(g_vol))
            if use_gas_abs
                @test all(>=(0), parent(g_vol))
                @test any(>(0), parent(g_vol))
            else
                @test all(iszero, parent(g_vol))
            end
        end
    end

    @testset "nonzero hydrometeors produce positive optics" begin
        hydrometeors =
            make_hydrometeor_subcolumns(FT, nsubcolumns, nelems; value = 0)
        @. hydrometeors.q_lcl[1] = FT(1e-4)
        z_vol, kr_vol, g_vol = make_cloudsat_outputs(FT, nsubcolumns, nelems)
        radar_cfg = CCO.CloudSatRadarConfig(FT; use_gas_abs = false)

        result = CCO.cloudsat_optics!(
            z_vol,
            kr_vol,
            g_vol,
            hydrometeors,
            thermo_state,
            rho_air,
            radar_cfg,
        )

        @test isnothing(result)
        @test all(isfinite, parent(z_vol[1]))
        @test all(isfinite, parent(kr_vol[1]))
        @test any(>(0), parent(z_vol[1]))
        @test any(>(0), parent(kr_vol[1]))
        @test all(isfinite, parent(g_vol))
        @test all(iszero, parent(g_vol))
    end
end
