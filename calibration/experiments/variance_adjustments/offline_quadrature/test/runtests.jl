using Test
using NCDatasets
using ClimaCore
import ClimaAtmos as CA
import Thermodynamics as TD
import ClimaAtmos.Parameters as CAP

# Include the scripts we want to test
# (Assuming we are running from the offline_quadrature directory)
include("../scripts/offline_quadrature.jl")
include("../scripts/run_highres_truth.jl")

@testset "Offline Quadrature Pipeline" begin
    
    # 1. Test NetCDF Variable Standard
    @testset "NetCDF Variable Standard" begin
        # Create a synthetic profile
        z = collect(0.0:10.0:1000.0)
        N = length(z)
        # Construct face coordinates (N+1 faces for N centers)
        z_faces = collect(-5.0:10.0:1005.0)
        prof = Dict(
            "z" => z,
            "z_faces" => z_faces,
            "T" => fill(280.0, N),
            "qv" => fill(0.01, N),
            "p" => fill(100000.0, N),
            "rho" => fill(1.2, N),
            "q_liq" => fill(0.0, N),
            "q_ice" => fill(0.0, N),
            "theta_li" => fill(280.0, N),
            "q_var_sgs" => fill(1e-6, N),
            "T_var_sgs" => fill(0.1, N),
            "corr_Tq" => fill(0.0, N)
        )
        
        tmp_nc = "test_profile.nc"
        write_highres_truth_profile_netcdf!(prof, tmp_nc)
        
        @test isfile(tmp_nc)
        NCDataset(tmp_nc) do ds
            @test haskey(ds, "theta_li")
            @test haskey(ds, "q_var_sgs")
            @test haskey(ds, "T_var_sgs")
            @test haskey(ds, "z_faces")
            @test length(ds["z_faces"]) == N + 1
        end
        
        # Verify round-trip via reader
        prof_read = read_highres_truth_profile(tmp_nc)
        @test haskey(prof_read, "z_faces")
        @test prof_read["z_faces"] ≈ z_faces
        rm(tmp_nc)
    end

    @testset "Entrypoint Include" begin
        include(joinpath(@__DIR__, "..", "scripts", "run_end_to_end.jl"))
        @test isdefined(Main, :run_offline_quadrature_end_to_end!)
    end

    # 2. Test Regridding Parity (Block Average)
    @testset "Regridding Parity" begin
        z_src = collect(0.0:1.0:100.0) # 1m resolution
        fields = Dict(
            "T" => fill(280.0, length(z_src)),
            "qv" => 0.01 .+ 0.001 .* sin.(z_src ./ 10.0),
            "p" => fill(100000.0, length(z_src)),
            "rho" => fill(1.2, length(z_src)),
            "theta_li" => fill(280.0, length(z_src)),
            "q_liq" => fill(0.0, length(z_src)),
            "q_ice" => fill(0.0, length(z_src)),
            "q_var_sgs" => fill(1e-6, length(z_src)),
            "T_var_sgs" => fill(0.1, length(z_src)),
            "q_var_total" => fill(1e-5, length(z_src)),
            "T_var_total" => fill(0.2, length(z_src)),
            "corr_Tq" => fill(0.5, length(z_src))
        )
        
        z_target_faces = collect(0.0:20.0:100.0) # 20m resolution
        z_target_centers = 10.0:20.0:90.0
        
        prof_r, qv_c, Tv_c, corr_c = regrid_truth_profile_block_average(z_src, fields, collect(z_target_centers), z_target_faces)
        
        @test length(qv_c) == 5
        # Verify that total variance in coarse cell is >= SGS variance
        @test all(prof_r["q_var_total"] .>= prof_r["q_var_sgs"])
        @test all(prof_r["T_var_total"] .>= prof_r["T_var_sgs"])
    end

    @testset "Linear Regridding" begin
        z_src = collect(0.0:1.0:100.0)
        fields = Dict(
            "T" => fill(300.0, 101),
            "qv" => fill(0.01, 101),
            "p" => fill(1e5, 101),
            "rho" => fill(1.2, 101),
            "theta_li" => fill(300.0, 101),
            "q_liq" => fill(0.0, 101),
            "q_ice" => fill(0.0, 101),
            "q_var_sgs" => fill(1e-6, 101),
            "T_var_sgs" => fill(1e-2, 101),
            "q_var_total" => fill(1e-6, 101),
            "T_var_total" => fill(1e-2, 101),
            "corr_Tq" => fill(0.5, 101)
        )
        z_target_centers = collect(10.0:20.0:90.0)
        
        prof_r, qv_c, Tv_c, corr_c = regrid_truth_profile_linear(z_src, fields, z_target_centers)
        
        @test length(qv_c) == 5
        @test all(qv_c .== 1e-6)
        @test all(Tv_c .== 1e-2)
        @test all(corr_c .== 0.5)
    end

    # 3. Test Integration Consistency
    @testset "Integration Parity" begin
        # This test ensures that the offline ᶜgradᵥ operator produces consistent results
        # for a simple linear profile in both supported float precisions.

        for FT in (Float32, Float64)
            z_faces = FT.(collect(0.0:100.0:1000.0))
            center_space = build_column_center_space(z_faces)

            # Linear q_tot profile: q = a*z + b
            # Gradient should be 'a'
            a = FT(1e-5)
            b = FT(0.01)

            f = ClimaCore.Fields.Field(FT, center_space)
            # Correct centers for 100m cells: 50, 150, 250, ...
            centers = FT.([50.0 + 100.0 * (i - 1) for i in 1:10])
            parent(f) .= a .* centers .+ b

            ᶠgradᵥ = ClimaCore.Operators.GradientC2F(
                bottom = ClimaCore.Operators.SetGradient(ClimaCore.Geometry.Covariant3Vector(FT(0.0))),
                top = ClimaCore.Operators.SetGradient(ClimaCore.Geometry.Covariant3Vector(FT(0.0))),
            )
            ᶜleft_bias = ClimaCore.Operators.LeftBiasedF2C()

            # Test gradient in the middle of the domain
            grad_f = @. ᶜleft_bias(ᶠgradᵥ(f))

            # Interior gradients should be exactly 'a'
            mid_grad = ClimaCore.Fields.level(grad_f, 5)[]
            lg = ClimaCore.Fields.local_geometry_field(center_space)
            lg_mid = ClimaCore.Fields.level(lg, 5)[]

            @test isapprox(ClimaCore.Geometry.WVector(mid_grad, lg_mid)[1], a, atol=FT(1e-10))
        end
    end

    @testset "Type Consistency: Float64 Profile + Float32 Thermo_params" begin
        # This test reproduces the worker failure mode: the saved NetCDF profile is
        # Float64, but the sidecar thermodynamics parameters are Float32.
        
        FT = Float64
        z = FT.(collect(0.0:10.0:100.0))
        N = length(z)
        z_faces = FT.(collect(-5.0:10.0:105.0))
        
        prof = Dict(
            "z" => z,
            "z_faces" => z_faces,
            "T" => fill(FT(280.0), N),
            "qv" => fill(FT(0.01), N),
            "p" => fill(FT(100000.0), N),
            "rho" => fill(FT(1.2), N),
            "q_liq" => fill(FT(0.0), N),
            "q_ice" => fill(FT(0.0), N),
            "theta_li" => fill(FT(280.0), N),
            "q_var_sgs" => fill(FT(1e-6), N),
            "T_var_sgs" => fill(FT(0.1), N),
            "corr_Tq" => fill(FT(0.0), N)
        )
        
        tmp_nc = "test_profile_f32.nc"
        write_highres_truth_profile_netcdf!(prof, tmp_nc)
        
        # Create Float32 thermo_params by extracting from ClimaAtmosParameters{Float32}
        params_f32 = CA.ClimaAtmosParameters(Float32)
        thp_f32 = CAP.thermodynamics_params(params_f32)
        
        # Define target grids for quadrature sweep
        z_target_centers = FT.(collect(50.0:20.0:90.0))
        z_target_faces = FT.(collect(40.0:20.0:100.0))
        target_grids = [
            ("linear_20m", z_target_centers, z_target_faces)
        ]
        
        # Create temporary output directory
        tmp_outdir = "test_quadrature_f32"
        mkpath(tmp_outdir)
        
        # This call should NOT fail with type mismatch.
        # Before the fix, it would fail with:
        # "expected AbstractThermodynamicsParameters{Float64}, got Float32"
        @test_nowarn run_offline_quadrature_from_netcdf!(
            tmp_nc;
            outdir=tmp_outdir,
            target_grids=target_grids,
            quadrature_orders=[1, 2],
            sgs_distributions=["lognormal"],
            skip_existing=false,
            regrid_method=:linear,
            thermo_params=thp_f32,
        )
        
        # Verify output was created
        @test isfile(joinpath(tmp_outdir, "linear", "linear_20m", "N_1", "lognormal.jld2"))
        
        # Cleanup
        rm(tmp_nc)
        rm(tmp_outdir, recursive=true)
    end
end
