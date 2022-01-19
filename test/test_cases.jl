include("test_cases/run_1d_ekman_column.jl")
include("test_cases/run_2d_rising_bubble.jl")
include("test_cases/run_3d_rising_bubble.jl")
include("test_cases/run_3d_solid_body_rotation.jl")
include("test_cases/run_3d_balanced_flow.jl")
include("test_cases/run_3d_baroclinic_wave.jl")

function test_cases(test_names, test_mode)
    # Column tests
    if :test_1d_ekman_column ∈ test_names
        @testset "1D Ekman column" begin
            for FT in float_types
                run_1d_ekman_column(FT, test_mode = test_mode)
            end
        end
    end

    # @testset "1D radiative-convective equilibirum" begin
    # end

    # Box tests
    if :test_2d_rising_bubble ∈ test_names
        @testset "2D dry rising bubble" begin
            for FT in float_types
                run_2d_rising_bubble(FT, test_mode = test_mode)
            end
        end
    end

    # @testset "2D mountain" begin
    # end

    if :test_3d_rising_bubble ∈ test_names
        @testset "3D rising bubble" begin
            for FT in float_types
                run_3d_rising_bubble(FT, test_mode = test_mode)
            end
        end
    end

    # Sphere tests
    if :test_3d_solid_body_rotation ∈ test_names
        @testset "3D solid-body rotation" begin
            for FT in float_types
                run_3d_solid_body_rotation(FT, test_mode = test_mode)
            end
        end
    end

    if :test_3d_balanced_flow ∈ test_names
        @testset "3D balanced flow" begin
            for FT in float_types
                run_3d_balanced_flow(FT, test_mode = test_mode)
            end
        end
    end

    # @testset "3D boundary conditions" begin
    # end

    if :test_3d_baroclinic_wave ∈ test_names
        @testset "3D baroclinic wave" begin
            for FT in float_types
                run_3d_baroclinic_wave(FT, test_mode = test_mode)
            end
        end
    end
end
