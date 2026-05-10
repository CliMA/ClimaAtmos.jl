using Test

# Include the helper scripts from the project
include(joinpath(@__DIR__, "..", "scripts", "run_highres_truth.jl"))
include(joinpath(@__DIR__, "..", "scripts", "offline_quadrature.jl"))

@testset "High-res profile NetCDF IO" begin
    tmp = tempname() * ".nc"
    z = Float64[0.0, 50.0, 100.0]
    z_faces = Float64[0.0, 25.0, 75.0, 100.0]
    N = length(z)
    profile = Dict(
        "z" => z,
        "z_faces" => z_faces,
        "T" => Float64[280.0, 275.0, 270.0],
        "qv" => Float64[0.01, 0.008, 0.005],
        "p" => Float64[101325.0, 90000.0, 80000.0],
        "rho" => Float64[1.2, 1.1, 1.0],
        "q_liq" => Float64[0.0, 0.0, 0.0],
        "q_ice" => Float64[0.0, 0.0, 0.0],
        "theta_li" => Float64[300.0, 299.0, 298.0],
        "q_var_sgs" => Float64[1e-6, 1e-6, 1e-6],
        "T_var_sgs" => Float64[1e-4, 1e-4, 1e-4],
        "corr_Tq" => Float64[0.0, 0.0, 0.0],
    )

    write_highres_truth_profile_netcdf!(profile, tmp)
    prof2 = read_highres_truth_profile(tmp)

    @test haskey(prof2, "z_faces")
    @test length(prof2["z"]) == N
    @test length(prof2["z_faces"]) == N + 1

    rm(tmp; force=true)
end
