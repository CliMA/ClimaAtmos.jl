include("test_cases/run_ekman_column_1d.jl")
include("test_cases/run_dry_rising_bubble_2d.jl")
include("test_cases/run_dry_rising_bubble_3d.jl")
include("test_cases/run_solid_body_rotation.jl")
include("test_cases/run_balanced_flow.jl")
include("test_cases/run_dry_shallow_baroclinic_wave.jl")

function test_cases(mode)
    # Column tests (1)
    @testset "Ekman column 1D" begin
        for FT in float_types
            run_ekman_column_1d(FT; mode)
        end
    end

    # @testset "Radiative-convective equilibirum" begin
    #     for FT in float_types
    #         run_radiative_convective_equilibrium(FT; mode)
    #     end
    # end

    # Box tests (2)
    @testset "Dry rising bubble 2D" begin
        for FT in float_types
            run_dry_rising_bubble_2d(FT; mode)
        end
    end

    # @testset "Mountain check" begin
    #     for FT in float_types
    #         run_mountain_check(FT; mode)
    #     end
    # end

    @testset "Dry rising bubble 3D" begin
        for FT in float_types
            run_dry_rising_bubble_3d(FT; mode)
        end
    end

    # @testset "Moist rising bubble 3D" begin
    #     for FT in float_types
    #         run_moist_rising_bubble_3d(FT; mode)
    #     end
    # end

    # Sphere tests (4)
    @testset "Solid-body rotation" begin
        for FT in float_types
            run_solid_body_rotation(FT; mode)
        end
    end

    @testset "Balanced flow" begin
        for FT in float_types
            run_balanced_flow(FT; mode)
        end
    end

    # @testset "Boundary conditions test" begin
    #     for FT in float_types
    #         run_incoming_energy_fluxes(FT; mode)
    #     end
    # end

    @testset "Dry shallow baroclinic wave" begin
        for FT in float_types
            run_dry_shallow_baroclinic_wave(FT; mode)
        end
    end

    # @testset "Moist shallow baroclinic wave" begin
    #     for FT in float_types
    #         run_moist_shallow_baroclinic_wave(FT; mode)
    #     end
    # end
end
