function instantiate_column(FT)
    domain = Column(FT, zlim = (0.0, 1.0), nelements = 2)
    check1 = domain.zlim == (0.0, 1.0)
    check2 = domain.nelements == 2

    return check1 && check2
end

function instantiate_plane(FT)
    domain = Plane(
        FT,
        xlim = (0.0, 1.0),
        ylim = (0.0, 2.0),
        nelements = (3, 2),
        npolynomial = 16,
        periodic = (false, false),
    )
    check1 = domain.xlim == (0.0, 1.0)
    check2 = domain.ylim == (0.0, 2.0)
    check3 = domain.nelements == (3, 2)
    check4 = domain.npolynomial == 16
    check5 = domain.periodic == (false, false)

    return check1 && check2 && check3 && check4 && check5
end

function instantiate_periodic_plane(FT)
    domain = PeriodicPlane(
        FT,
        xlim = (0.0, 1.0),
        ylim = (0.0, 2.0),
        nelements = (3, 2),
        npolynomial = 16,
    )
    check1 = domain.xlim == (0.0, 1.0)
    check2 = domain.ylim == (0.0, 2.0)
    check3 = domain.nelements == (3, 2)
    check4 = domain.npolynomial == 16
    check5 = domain.periodic == (true, true)

    return check1 && check2 && check3 && check4 && check5
end

function instantiate_hybrid_plane(FT)
    domain = HybridPlane(
        FT,
        xlim = (0.0, 1.0),
        zlim = (0.0, 2.0),
        nelements = (3, 2),
        npolynomial = 16,
    )
    check1 = domain.xlim == (0.0, 1.0)
    check2 = domain.zlim == (0.0, 2.0)
    check3 = domain.nelements == (3, 2)
    check4 = domain.npolynomial == 16

    return check1 && check2 && check3 && check4
end

function instantiate_sphere(FT)
    domain = Sphere(FT, radius = 1.0, nelements = 4, npolynomial = 4)
    check1 = domain.radius == 1.0
    check2 = domain.nelements == 4
    check3 = domain.npolynomial == 4

    return check1 && check2 && check3
end

@testset "Domains" begin
    @info "Testing ClimaAtmos.Domains..."

    @testset "Domains" begin
        for FT in float_types
            @test instantiate_column(FT)
            @test instantiate_plane(FT)
            @test instantiate_periodic_plane(FT)
            @test instantiate_hybrid_plane(FT)
            @test instantiate_sphere(FT)

            I = Column(FT, zlim = (1.0, 2.0), nelements = 2)
            ✈ = Plane(
                FT,
                xlim = (0.0, 2.0),
                ylim = (0.0, 3.0),
                nelements = (3, 2),
                npolynomial = 16,
                periodic = (false, false),
            )
            HV = HybridPlane(
                FT,
                xlim = (0.0, 1.0),
                zlim = (0.0, 2.0),
                nelements = (3, 2),
                npolynomial = 16,
            )
            QIU = Sphere(FT, radius = 1.0, nelements = 4, npolynomial = 4)

            # Test ndims
            @test ndims(I) == 1
            @test ndims(✈) == 2
            @test ndims(HV) == 2
            @test ndims(QIU) == 2

            # Test length
            @test length(I) == 1.0

            # Test size
            @test size(I) == 1.0
            @test size(✈) == (2.0, 3.0)
            @test size(HV) == (1.0, 2.0)

            # Test show functions
            show(I)
            println()
            @test I isa Column{FT}
            show(✈)
            println()
            @test ✈ isa Plane{FT}
            show(HV)
            println()
            @test HV isa HybridPlane{FT}
            show(QIU)
            println()
            @test QIU isa Sphere{FT}

        end
    end
end
