using Test
using ClimaAtmos
using ClimaCore: Domains, Meshes, Spaces, Fields, Geometry

const CA = ClimaAtmos
const CNR = CA.COSP1MReffNpDiagnostics

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

function center_profile(field)
    return [Fields.level(field, ilev)[] for ilev in 1:Spaces.nlevels(axes(field))]
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

function make_reff_subcolumns(FT, nsubcolumns, nelems; value = -1)
    return (;
        Reff_lcl = make_subcolumn_fields(FT, nsubcolumns, nelems; value),
        Reff_icl = make_subcolumn_fields(FT, nsubcolumns, nelems; value),
        Reff_rai = make_subcolumn_fields(FT, nsubcolumns, nelems; value),
        Reff_sno = make_subcolumn_fields(FT, nsubcolumns, nelems; value),
    )
end

function make_number_subcolumns(FT, nsubcolumns, nelems; value = -1)
    return (;
        Np_lcl = make_subcolumn_fields(FT, nsubcolumns, nelems; value),
        Np_icl = make_subcolumn_fields(FT, nsubcolumns, nelems; value),
        Np_rai = make_subcolumn_fields(FT, nsubcolumns, nelems; value),
        Np_sno = make_subcolumn_fields(FT, nsubcolumns, nelems; value),
    )
end

function set_profile!(field, values)
    for (ilev, value) in enumerate(values)
        Fields.level(field, ilev) .= value
    end
end

function all_diagnostic_fields(reff, np)
    return (values(reff)..., values(np)...)
end

@testset "COSP 1M Reff and Np diagnostics" begin
    FT = Float64
    nsubcolumns = 2
    nelems = 3
    rho = make_center_profile_field(FT, [1.2, 1.0, 0.8])

    @test CNR.N_LCL == 100e6
    @test CNR.K_LCL == 0.8
    @test CNR.RHO_W == 1000.0

    @testset "zero hydrometeor gives zero diagnostics" begin
        hydrometeors =
            make_hydrometeor_subcolumns(FT, nsubcolumns, nelems; value = 0)
        reff = make_reff_subcolumns(FT, nsubcolumns, nelems)
        np = make_number_subcolumns(FT, nsubcolumns, nelems)

        result = CNR.set_1M_reff_np_subcolumns!(
            reff,
            np,
            hydrometeors,
            rho,
        )

        @test isnothing(result)
        for field_group in all_diagnostic_fields(reff, np), field in field_group
            @test all(iszero, parent(field))
        end
    end

    @testset "positive cloud liquid uses prescribed number" begin
        hydrometeors =
            make_hydrometeor_subcolumns(FT, nsubcolumns, nelems; value = 0)
        set_profile!(hydrometeors.q_lcl[1], FT[1e-4, 2e-4, 3e-4])
        reff = make_reff_subcolumns(FT, nsubcolumns, nelems)
        np = make_number_subcolumns(FT, nsubcolumns, nelems)

        CNR.set_1M_reff_np_subcolumns!(reff, np, hydrometeors, rho)

        @test center_profile(np.Np_lcl[1]) ==
              fill(FT(CNR.N_LCL), nelems)
        @test all(>(0), center_profile(reff.Reff_lcl[1]))
        @test all(iszero, center_profile(np.Np_lcl[2]))
    end

    @testset "Marshall-Palmer Reff and Np use lambda inverse and n0" begin
        hydrometeors =
            make_hydrometeor_subcolumns(FT, nsubcolumns, nelems; value = 0)
        set_profile!(hydrometeors.q_rai[1], FT[1e-3, 0, 0])
        set_profile!(hydrometeors.q_icl[1], FT[2e-4, 0, 0])
        set_profile!(hydrometeors.q_sno[1], FT[4e-4, 0, 0])
        reff = make_reff_subcolumns(FT, nsubcolumns, nelems)
        np = make_number_subcolumns(FT, nsubcolumns, nelems)

        CNR.set_1M_reff_np_subcolumns!(reff, np, hydrometeors, rho)
        checks = (
            (
                q_field = hydrometeors.q_rai[1],
                reff_field = reff.Reff_rai[1],
                np_field = np.Np_rai[1],
                lambda_inverse = CNR._rain_lambda_inverse,
                n0 = (q, rho) -> FT(CNR.N0_RAIN),
            ),
            (
                q_field = hydrometeors.q_icl[1],
                reff_field = reff.Reff_icl[1],
                np_field = np.Np_icl[1],
                lambda_inverse = CNR._ice_lambda_inverse,
                n0 = (q, rho) -> FT(CNR.N0_ICE),
            ),
            (
                q_field = hydrometeors.q_sno[1],
                reff_field = reff.Reff_sno[1],
                np_field = np.Np_sno[1],
                lambda_inverse = CNR._snow_lambda_inverse,
                n0 = CNR._snow_n0,
            ),
        )

        for check in checks
            q = center_profile(check.q_field)[1]
            rho_level = center_profile(rho)[1]
            lambda_inv = check.lambda_inverse(q, rho_level)
            n0 = check.n0(q, rho_level)

            @test isapprox(center_profile(check.reff_field)[1], 3 * lambda_inv)
            @test isapprox(center_profile(check.np_field)[1], n0 * lambda_inv)
        end
    end

    @testset "snow uses variable n0" begin
        hydrometeors =
            make_hydrometeor_subcolumns(FT, nsubcolumns, nelems; value = 0)
        set_profile!(hydrometeors.q_sno[1], FT[1e-4, 8e-4, 0])
        reff = make_reff_subcolumns(FT, nsubcolumns, nelems)
        np = make_number_subcolumns(FT, nsubcolumns, nelems)

        CNR.set_1M_reff_np_subcolumns!(reff, np, hydrometeors, rho)

        q1, q2 = center_profile(hydrometeors.q_sno[1])[1:2]
        rho1, rho2 = center_profile(rho)[1:2]
        n0_1 = CNR._snow_n0(q1, rho1)
        n0_2 = CNR._snow_n0(q2, rho2)

        @test n0_1 != n0_2
        @test isapprox(
            center_profile(np.Np_sno[1])[1],
            n0_1 * CNR._snow_lambda_inverse(q1, rho1),
        )
        @test isapprox(
            center_profile(np.Np_sno[1])[2],
            n0_2 * CNR._snow_lambda_inverse(q2, rho2),
        )
    end

    @testset "tiny positive values stay finite" begin
        tiny = sqrt(eps(FT))
        hydrometeors =
            make_hydrometeor_subcolumns(FT, nsubcolumns, nelems; value = tiny)
        tiny_rho = make_center_field(FT; value = tiny, nelems)
        reff = make_reff_subcolumns(FT, nsubcolumns, nelems)
        np = make_number_subcolumns(FT, nsubcolumns, nelems)

        CNR.set_1M_reff_np_subcolumns!(
            reff,
            np,
            hydrometeors,
            tiny_rho,
        )

        for field_group in all_diagnostic_fields(reff, np), field in field_group
            @test all(isfinite, parent(field))
        end
    end
end
