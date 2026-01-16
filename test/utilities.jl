using Test
using ClimaComms
ClimaComms.@import_required_backends
import Dates
using Random
Random.seed!(1234)
import ClimaAtmos as CA
using NCDatasets
import LinearAlgebra: norm_sqr

include("test_helpers.jl")

#####
##### Time and Date utilities
#####

@testset "time_to_seconds" begin
    # Seconds
    @test CA.time_to_seconds("10s") == 10
    @test CA.time_to_seconds("10secs") == 10

    # Minutes
    @test CA.time_to_seconds("2m") == 120
    @test CA.time_to_seconds("2mins") == 120

    # Hours
    @test CA.time_to_seconds("1h") == 3600
    @test CA.time_to_seconds("1hours") == 3600

    # Days
    @test CA.time_to_seconds("3d") == 3 * 86400
    @test CA.time_to_seconds("3days") == 3 * 86400

    # Weeks
    @test CA.time_to_seconds("50weeks") == 50 * 7 * 86400

    # Float input
    @test CA.time_to_seconds("1.5h") == 1.5 * 3600

    # Special case
    @test CA.time_to_seconds("Inf") == Inf

    # Bad formats
    @test_throws ErrorException CA.time_to_seconds("10")
    @test_throws ErrorException CA.time_to_seconds("10lightyears")
    @test_throws ErrorException CA.time_to_seconds("mins10")
end

@testset "isdivisible" begin
    @test CA.isdivisible(Dates.Month(1), Dates.Day(1))
    @test !CA.isdivisible(Dates.Month(1), Dates.Day(25))
    @test CA.isdivisible(Dates.Week(1), Dates.Day(1))
    @test CA.isdivisible(Dates.Day(1), Dates.Hour(1))
    @test CA.isdivisible(Dates.Hour(1), Dates.Second(1))
    @test CA.isdivisible(Dates.Minute(1), Dates.Second(30))
    @test !CA.isdivisible(Dates.Minute(1), Dates.Second(13))
    @test !CA.isdivisible(Dates.Day(1), Dates.Second(1e6))
    @test CA.isdivisible(Dates.Month(1), Dates.Hour(1))
end

@testset "promote_period" begin
    @test CA.promote_period(Dates.Hour(24)) == Dates.Day(1)
    @test CA.promote_period(Dates.Day(14)) == Dates.Week(2)
    @test CA.promote_period(Dates.Millisecond(1)) == Dates.Millisecond(1)
    @test CA.promote_period(Dates.Minute(120)) == Dates.Hour(2)
    @test CA.promote_period(Dates.Second(3600)) == Dates.Hour(1)
end

@testset "compound_period" begin
    @test CA.compound_period(3600, Dates.Second) == Dates.Hour(1)
    @test CA.compound_period(86400, Dates.Second) == Dates.Day(1)
    @test CA.compound_period(1.5, Dates.Hour) == Dates.CompoundPeriod(Dates.Hour(1), Dates.Minute(30))
end

@testset "time_and_units_str" begin
    @test occursin("1 hour", CA.time_and_units_str(3600))
    @test occursin("1 day", CA.time_and_units_str(86400))
end

@testset "parse_date" begin
    @test CA.parse_date("20000506") == Dates.DateTime(2000, 5, 6)
    @test CA.parse_date("20000506-0000") == Dates.DateTime(2000, 5, 6, 0, 0)
    @test_throws ErrorException CA.parse_date("20000506-00000")
    @test_throws ErrorException CA.parse_date("")
end

#####
##### File utilities
#####

@testset "sort_files_by_time" begin
    day_sec(t) = (floor(Int, t / 86400), floor(Int, t % 86400))
    filenames(d, s) = "day$d.$s.hdf5"
    filenames(t) = filenames(day_sec(t)...)
    t = map(i -> rand(1:(10^6)), 1:100)
    t_sorted = sort(t)
    fns = filenames.(t)
    sort!(fns)
    @test CA.sort_files_by_time(fns) == filenames.(t_sorted)
end

@testset "time_from_filename" begin
    @test CA.time_from_filename("day0.0.hdf5") == 0.0
    @test CA.time_from_filename("day1.0.hdf5") == 86400.0
    @test CA.time_from_filename("day1.3600.hdf5") == 86400.0 + 3600.0
    @test CA.time_from_filename("day10.43200.hdf5") == 10 * 86400.0 + 43200.0
end

#####
##### Math utilities
#####

@testset "fast_pow" begin
    # Test that fast_pow matches regular power for typical values
    @test CA.fast_pow(2.0, 3.0) ≈ 8.0
    @test CA.fast_pow(10.0, 2.0) ≈ 100.0
    @test CA.fast_pow(0.99999, 1000.0) ≈ 0.99999^1000 rtol = 1e-10
end

#####
##### Variable classification predicates
#####

@testset "is_*_var predicates" begin
    @test CA.is_energy_var(:ρe_tot) == true
    @test CA.is_energy_var(:ρq_tot) == false
    @test CA.is_momentum_var(:uₕ) == true
    @test CA.is_momentum_var(:u₃) == true
    @test CA.is_momentum_var(:ρ) == false
    @test CA.is_sgs_var(:sgsʲs) == true
    @test CA.is_tracer_var(:ρq_tot) == true
    @test CA.is_tracer_var(:ρ) == false
end

#####
##### Geometry and mesh utilities
#####

@testset "Get mesh metrics" begin
    # This just tests getter functions
    # Correctness is checked in ClimaCore.jl
    (; cent_space, face_space) = get_cartesian_spaces()
    lg_gⁱʲ = cent_space.grid.center_local_geometry.gⁱʲ
    lg_g³³ = lg_gⁱʲ.components.data.:9
    (; x) = Fields.coordinate_field(cent_space)
    @test Fields.field_values(CA.g³³_field(axes(x))) == lg_g³³
    @test maximum(abs.(lg_g³³ .- CA.g³³.(lg_gⁱʲ).components.data.:1)) == 0
    @test maximum(abs.(CA.g³ʰ.(lg_gⁱʲ).components.data.:1)) == 0
    @test maximum(abs.(CA.g³ʰ.(lg_gⁱʲ).components.data.:2)) == 0
end

@testset "horizontal_integral_at_boundary" begin
    # Test horizontal_integral_at_boundary which computes ∫∫ f dA at a horizontal level
    # Two method signatures: (field, level) and (level_field)
    #
    # For Taylor-Green vortex: u = sin(x)cos(y), v = -cos(x)sin(y)
    # These are antisymmetric, so their integrals over a periodic domain should be ≈ 0
    
    (; cent_space, face_space) = get_cartesian_spaces()
    _, fcoords = get_coords(cent_space, face_space)
    ᶠu, ᶠv, ᶠw = taylor_green_ic(fcoords)
    FT = eltype(ᶠu)
    halflevel = ClimaCore.Utilities.half
    
    # Method 1: horizontal_integral_at_boundary(field, level)
    # Extracts the level internally and integrates
    @test CA.horizontal_integral_at_boundary(ᶠu, halflevel) <= sqrt(eps(FT))
    @test CA.horizontal_integral_at_boundary(ᶠv, halflevel) <= sqrt(eps(FT))
    @test isfinite(CA.horizontal_integral_at_boundary(ᶠw, halflevel))  # w=0, just check it runs
    
    # Method 2: horizontal_integral_at_boundary(level_field)
    # Takes a pre-extracted 2D horizontal slice
    ᶠuₛ = Fields.level(ᶠu, halflevel)
    ᶠvₛ = Fields.level(ᶠv, halflevel)
    ᶠwₛ = Fields.level(ᶠw, halflevel)
    @test CA.horizontal_integral_at_boundary(ᶠuₛ) <= sqrt(eps(FT))
    @test CA.horizontal_integral_at_boundary(ᶠvₛ) <= sqrt(eps(FT))
    @test isfinite(CA.horizontal_integral_at_boundary(ᶠwₛ))
end

#####
##### Kinetic energy and strain rate
#####

@testset "kinetic_energy (c.f. analytical function)" begin
    # Test compute_kinetic against analytical solution for Taylor-Green vortex
    # 
    # Taylor-Green vortex velocities:
    #   u = sin(x) cos(y) cos(z)
    #   v = -cos(x) sin(y) cos(z)  
    #   w = 0
    #
    # Kinetic energy κ = (1/2)(u² + v² + w²):
    #   κ = (1/2) cos²(z) [sin²(x)cos²(y) + cos²(x)sin²(y)]
    
    (; cent_space, face_space) = get_cartesian_spaces()
    ccoords, fcoords = get_coords(cent_space, face_space)
    uₕ, uᵥ = get_cartesian_test_velocities(cent_space, face_space)
    (; x, y, z) = ccoords
    
    # Compute kinetic energy using the function under test
    κ = zeros(cent_space)
    κ .= CA.compute_kinetic(uₕ, uᵥ)
    
    # Analytical solution (derived above)
    ᶜκ_exact = @. (1 // 2) * cos(z)^2 * (sin(x)^2 * cos(y)^2 + cos(x)^2 * sin(y)^2)
    
    @test ᶜκ_exact ≈ κ
end

@testset "kinetic_energy (spherical geometry with topography)" begin
    # Test compute_kinetic on warped spherical geometry (with mountain)
    #
    # We define a physical velocity field:
    #   u_phys = u_zonal = U₀ cos(lat)
    #   v_phys = 0
    #   w_phys = W₀ sin(π * z_phys / z_max)
    #
    # We project this onto a grid with topography.
    # The kinetic energy should be exactly 0.5 * (u_phys^2 + w_phys^2).
    
    (; cent_space, face_space, z_max, FT) = get_spherical_extruded_spaces_with_topography()
    
    # Define Physical Velocity and Project
    U₀, W₀ = FT(10), FT(1)
    uₕ, uᵥ, _ = get_spherical_test_velocities(cent_space, face_space, z_max; U₀, W₀)
   
    # 3. Compute and Verify
    κ = CA.compute_kinetic(uₕ, uᵥ)
    
    # helper for exact calc
    c_coords = Fields.coordinate_field(cent_space)
    
    # Exact Kinetic Energy (Scalar)
    ᶜκ_exact = Fields.Field(FT, cent_space)
    @. ᶜκ_exact = 0.5 * (
        (U₀ * cosd(c_coords.lat))^2 +
        (W₀ * sin(FT(π) * c_coords.z / z_max))^2
    )
    
    @test maximum(abs.(κ .- ᶜκ_exact)) < FT(0.01) * maximum(abs.(ᶜκ_exact))
end

@testset "compute_strain_rate (Cartesian, analytical)" begin
    # Test compute_strain_rate_*_vertical against analytical solution for Taylor-Green vortex
    #
    # For Taylor-Green vortex: u = sin(x)cos(y)cos(z), v = -cos(x)sin(y)cos(z), w = 0
    # The strain rate tensor ε_ij = (1/2)(∂u_i/∂x_j + ∂u_j/∂x_i)
    #
    # Analytical off-diagonal components:
    #   ε₁₃ = (1/2) ∂u/∂z = -(1/2) sin(x) cos(y) sin(z)
    #   ε₂₃ = (1/2) ∂v/∂z = (1/2) cos(x) sin(y) sin(z)
    
    (; helem, cent_space, face_space) = get_cartesian_spaces()
    ccoords, fcoords = get_coords(cent_space, face_space)
    FT = eltype(ccoords.x)
    
    # Allocate tensor fields for strain rate output
    UVW = Geometry.UVWVector
    u₀ = UVW(FT(0), FT(0), FT(0))
    ᶜε = Fields.Field(typeof(u₀ * u₀'), cent_space)
    ᶠε = Fields.Field(typeof(u₀ * u₀'), face_space)
    
    # Get velocity from Taylor-Green vortex
    u, v, w = taylor_green_ic(ccoords)
    ᶠu, ᶠv, ᶠw = taylor_green_ic(fcoords)
    ᶜu = @. UVW(Geometry.UVector(u)) + UVW(Geometry.VVector(v)) + UVW(Geometry.WVector(w))
    ᶠu = @. UVW(Geometry.UVector(ᶠu)) + UVW(Geometry.VVector(ᶠv)) + UVW(Geometry.WVector(ᶠw))
    
    # Compute strain rates
    ᶜε .= CA.compute_strain_rate_center_vertical(Geometry.Covariant123Vector.(ᶠu))
    ᶠε .= CA.compute_strain_rate_face_vertical(Geometry.Covariant123Vector.(ᶜu))
    
    # Symmetry checks: ε_ij = ε_ji (tensor is symmetric)
    @test ᶜε.components.data.:2 == ᶜε.components.data.:4  # ε₁₂ = ε₂₁
    @test ᶜε.components.data.:3 == ᶜε.components.data.:7  # ε₁₃ = ε₃₁
    @test ᶜε.components.data.:6 == ᶜε.components.data.:8  # ε₂₃ = ε₃₂
    
    # Check off-diagonal components against analytical solution at centers
    (; x, y, z) = ccoords
    ᶜε₁₃_computed = ᶜε.components.data.:3
    ᶜε₂₃_computed = ᶜε.components.data.:6
    ᶜε₁₃_exact = @. -(1 // 2) * sin(x) * cos(y) * sin(z)
    ᶜε₂₃_exact = @. (1 // 2) * cos(x) * sin(y) * sin(z)
    
    # Relative error should be small (< 0.5%)
    @test maximum(abs.(ᶜε₁₃_computed .- ᶜε₁₃_exact) ./ (abs.(ᶜε₁₃_exact) .+ eps(FT))) < FT(0.005)
    @test maximum(abs.(ᶜε₂₃_computed .- ᶜε₂₃_exact) ./ (abs.(ᶜε₂₃_exact) .+ eps(FT))) < FT(0.005)
    
    # Check boundary conditions at faces (top and bottom should match analytical)
    (; x, y, z) = fcoords
    ᶠε₁₃_exact = @. -(1 // 2) * sin(x) * cos(y) * sin(z)
    for elem_id in 1:helem
        @test maximum(abs.(
            Fields.field_values(Fields.slab(ᶠε.components.data.:3, 1, elem_id)) .-
            Fields.field_values(Fields.slab(ᶠε₁₃_exact, 1, elem_id))
        )) < eps(FT)  # bottom face
        @test maximum(abs.(
            Fields.field_values(Fields.slab(ᶠε.components.data.:3, 11, elem_id)) .-
            Fields.field_values(Fields.slab(ᶠε₁₃_exact, 11, elem_id))
        )) < eps(FT)  # top face
    end
end

@testset "compute_full_strain_rate (Cartesian, consistency)" begin
    # Test compute_strain_rate_*_full! by comparing to explicit reference calculation
    # using gradient operators and symmetrization: ε = (1/2)(∇u + (∇u)ᵀ)
    
    (; helem, cent_space, face_space) = get_cartesian_spaces()
    ccoords, fcoords = get_coords(cent_space, face_space)
    FT = eltype(ccoords.x)
    
    # Type aliases
    UVW = Geometry.UVWVector
    UVec, VVec, WVec = Geometry.UVector, Geometry.VVector, Geometry.WVector
    
    # Allocate tensor fields
    u₀ = UVW(FT(0), FT(0), FT(0))
    ᶜε = Fields.Field(typeof(u₀ * u₀'), cent_space)
    ᶠε = Fields.Field(typeof(u₀ * u₀'), face_space)
    ᶜε_ref = Fields.Field(typeof(u₀ * u₀'), cent_space)
    ᶠε_ref = Fields.Field(typeof(u₀ * u₀'), face_space)
    
    # Get velocity from Taylor-Green vortex
    u, v, w = taylor_green_ic(ccoords)
    ᶠu, ᶠv, ᶠw = taylor_green_ic(fcoords)
    ᶜu = @. UVW(UVec(u)) + UVW(VVec(v)) + UVW(WVec(w))
    ᶠu_vec = @. UVW(UVec(ᶠu)) + UVW(VVec(ᶠv)) + UVW(WVec(ᶠw))
    
    # Compute using functions under test
    CA.compute_strain_rate_center_full!(ᶜε, ᶜu, ᶠu_vec)
    CA.compute_strain_rate_face_full!(ᶠε, ᶜu, ᶠu_vec)
    
    # Build reference: ε = (1/2)(∇u + (∇u)ᵀ)
    axis_uvw = (Geometry.UVWAxis(),)
    ᶜgradᵥ = Operators.GradientF2C()
    gradₕ = Operators.Gradient()
    @. ᶜε_ref = Geometry.project(axis_uvw, ᶜgradᵥ(ᶠu_vec))
    @. ᶜε_ref += Geometry.project(axis_uvw, gradₕ(ᶜu))
    @. ᶜε_ref = (ᶜε_ref + adjoint(ᶜε_ref)) / 2
    
    # Face reference with zero-gradient BCs
    ∇bc = Operators.SetGradient(Geometry.outer(WVec(0), UVW(0, 0, 0)))
    ᶠgradᵥ = Operators.GradientC2F(bottom = ∇bc, top = ∇bc)
    @. ᶠε_ref = Geometry.project(axis_uvw, ᶠgradᵥ(ᶜu))
    @. ᶠε_ref += Geometry.project(axis_uvw, gradₕ(ᶠu_vec))
    @. ᶠε_ref = (ᶠε_ref + adjoint(ᶠε_ref)) / 2
    
    # Symmetry checks
    @test ᶜε.components.data.:2 == ᶜε.components.data.:4
    @test ᶜε.components.data.:3 == ᶜε.components.data.:7
    @test ᶜε.components.data.:6 == ᶜε.components.data.:8
    @test ᶠε.components.data.:2 == ᶠε.components.data.:4
    @test ᶠε.components.data.:3 == ᶠε.components.data.:7
    @test ᶠε.components.data.:6 == ᶠε.components.data.:8
    
    # Consistency with explicit reference (all 9 tensor components)
    tol = sqrt(eps(FT))
    @test maximum(abs, ᶜε.components.data.:1 .- ᶜε_ref.components.data.:1) < tol
    @test maximum(abs, ᶜε.components.data.:2 .- ᶜε_ref.components.data.:2) < tol
    @test maximum(abs, ᶜε.components.data.:3 .- ᶜε_ref.components.data.:3) < tol
    @test maximum(abs, ᶜε.components.data.:4 .- ᶜε_ref.components.data.:4) < tol
    @test maximum(abs, ᶜε.components.data.:5 .- ᶜε_ref.components.data.:5) < tol
    @test maximum(abs, ᶜε.components.data.:6 .- ᶜε_ref.components.data.:6) < tol
    @test maximum(abs, ᶜε.components.data.:7 .- ᶜε_ref.components.data.:7) < tol
    @test maximum(abs, ᶜε.components.data.:8 .- ᶜε_ref.components.data.:8) < tol
    @test maximum(abs, ᶜε.components.data.:9 .- ᶜε_ref.components.data.:9) < tol
    
    @test maximum(abs, ᶠε.components.data.:1 .- ᶠε_ref.components.data.:1) < tol
    @test maximum(abs, ᶠε.components.data.:2 .- ᶠε_ref.components.data.:2) < tol
    @test maximum(abs, ᶠε.components.data.:3 .- ᶠε_ref.components.data.:3) < tol
    @test maximum(abs, ᶠε.components.data.:4 .- ᶠε_ref.components.data.:4) < tol
    @test maximum(abs, ᶠε.components.data.:5 .- ᶠε_ref.components.data.:5) < tol
    @test maximum(abs, ᶠε.components.data.:6 .- ᶠε_ref.components.data.:6) < tol
    @test maximum(abs, ᶠε.components.data.:7 .- ᶠε_ref.components.data.:7) < tol
    @test maximum(abs, ᶠε.components.data.:8 .- ᶠε_ref.components.data.:8) < tol
    @test maximum(abs, ᶠε.components.data.:9 .- ᶠε_ref.components.data.:9) < tol
end

@testset "compute_strain_rate (spherical geometry)" begin
    # Test strain rate computation on spherical geometry
    # Uses get_spherical_test_velocities which provides ᶠu_C123 (Covariant123Vector on faces)
    
    (; cent_space, face_space, z_max, FT) = get_spherical_extruded_spaces()
    
    # Get velocity fields from helper (includes Covariant123Vector for strain rate)
    U₀, W₀ = FT(10), FT(1)
    _, _, ᶠu = get_spherical_test_velocities(cent_space, face_space, z_max; U₀ = U₀, W₀ = W₀)
    
    # Compute strain rate - returns a lazy field
    ᶜstrain_rate = CA.compute_strain_rate_center_vertical(ᶠu)
    
    # Materialize by computing Frobenius norm squared (as done in actual code)
    ᶜstrain_rate_norm = @. norm_sqr(ᶜstrain_rate)
    
    # Basic sanity checks
    # 1. Should be finite everywhere
    @test all(isfinite, parent(ᶜstrain_rate_norm))
    
    # 2. Should have nonzero values (vertical shear exists from w = W₀ sin(πz/H))
    # ∂w/∂z = W₀ π/H cos(πz/H), which is nonzero at most z values
    @test maximum(parent(ᶜstrain_rate_norm)) > FT(0)
    
    # 3. Compare against analytical solution
    # For w = W₀ sin(πz/H), the strain rate tensor has ε₃₃ = ∂w/∂z = W₀ π/H cos(πz/H)
    # So norm_sqr(ε) = ε₃₃² = (W₀ π/H)² cos²(πz/H)
    ccoords = Fields.coordinate_field(cent_space)
    ᶜz = ccoords.z
    ᶜε₃₃_exact = @. W₀ * FT(π) / z_max * cos(FT(π) * ᶜz / z_max)
    ᶜstrain_rate_norm_exact = @. ᶜε₃₃_exact^2
    
    # The computed norm should match the analytical solution within numerical tolerance
    # (allowing for some discretization error from the finite difference gradient)
    rel_error = @. abs(ᶜstrain_rate_norm - ᶜstrain_rate_norm_exact) / (abs(ᶜstrain_rate_norm_exact) + eps(FT))
    @test maximum(parent(rel_error)) < FT(0.01)  # 1% relative tolerance for FD discretization
end

@testset "compute_strain_rate (spherical geometry with topography)" begin
    # Test strain rate computation on spherical geometry with terrain-following coordinates
    # Uses the same velocity profile as the flat test, but on a warped grid
    
    (; cent_space, face_space, z_max, FT) = get_spherical_extruded_spaces_with_topography()
    
    # Get velocity fields from helper
    U₀, W₀ = FT(10), FT(1)
    _, _, ᶠu = get_spherical_test_velocities(cent_space, face_space, z_max; U₀, W₀)
    
    # Compute strain rate
    ᶜstrain_rate = CA.compute_strain_rate_center_vertical(ᶠu)
    ᶜstrain_rate_norm = @. norm_sqr(ᶜstrain_rate)
    
    # 1. Should be finite everywhere (even on warped grid)
    @test all(isfinite, parent(ᶜstrain_rate_norm))
    
    # 2. Should have nonzero values
    @test maximum(parent(ᶜstrain_rate_norm)) > FT(0)
    
    # 3. Compare against analytical solution (same as flat case)
    # On a terrain-following grid, z is the physical height, so the formula is the same
    ccoords = Fields.coordinate_field(cent_space)
    ᶜz = ccoords.z
    ᶜε₃₃_exact = @. W₀ * FT(π) / z_max * cos(FT(π) * ᶜz / z_max)
    ᶜstrain_rate_norm_exact = @. ᶜε₃₃_exact^2
    
    # Allow slightly higher tolerance for warped grid discretization
    rel_error = @. abs(ᶜstrain_rate_norm - ᶜstrain_rate_norm_exact) / (abs(ᶜstrain_rate_norm_exact) + eps(FT))
    @test maximum(parent(rel_error)) < FT(0.01)  # 1% tolerance
end
