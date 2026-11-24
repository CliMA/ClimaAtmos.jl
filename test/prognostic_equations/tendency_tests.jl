#=
Core tendency computation unit tests for ClimaAtmos.jl

TODO: Implement tests for:
- Tendency accumulation correctness
- Individual tendency term verification
- Tendency budget closure (sum of parts = total)
- NaN/Inf detection in tendencies
- Performance regression tests

See src/prognostic_equations/remaining_tendency.jl for the implementation.
=#

using Test
using ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: Domains, Fields, Geometry, Meshes, Spaces
import Thermodynamics as TD

@testset "Large-scale advection energy tendency" begin
    # The setups prescribe a potential-temperature tendency dθ/dt, supplied as
    # dT/dt = Π dθ/dt. The energy source must change θ by dθ/dt δt when added
    # at constant density, which holds for the isobaric heat capacity.
    FT = Float64
    params = CA.ClimaAtmosParameters(FT)
    thermo_params = CA.Parameters.thermodynamics_params(params)
    cp_d = TD.Parameters.cp_d(thermo_params)
    cv_d = TD.Parameters.cv_d(thermo_params)
    R_d = TD.Parameters.R_d(thermo_params)

    domain = Domains.IntervalDomain(
        Geometry.ZPoint(FT(0)),
        Geometry.ZPoint(FT(3000));
        boundary_names = (:bottom, :top),
    )
    mesh = Meshes.IntervalMesh(domain; nelems = 10)
    face_space =
        Spaces.FaceFiniteDifferenceSpace(ClimaComms.device(), mesh)
    ᶜz = Fields.coordinate_field(Spaces.CenterFiniteDifferenceSpace(face_space)).z

    # Dry column at 290 K with pressure decreasing in height.
    ᶜT = @. FT(290) + 0 * ᶜz
    ᶜp = @. FT(1e5) * exp(ᶜz / FT(-8000))
    ᶜρ = @. ᶜp / (R_d * ᶜT)
    q_zero = @. 0 * ᶜz

    dθdt = FT(-2 / 86400)
    ls_adv = CA.LargeScaleAdvection(
        (tps, p, t, z) -> dθdt * TD.exner_given_pressure(tps, p),
        Returns(zero(FT)),
    )
    ᶜρe_tend = Base.materialize(
        CA.large_scale_advection_tendency_ρe_tot(
            ᶜρ, thermo_params, ᶜT, ᶜp, q_zero, q_zero, q_zero, FT(0), ls_adv,
        ),
    )
    ᶜΠ = @. TD.exner_given_pressure(thermo_params, ᶜp)
    @test parent(ᶜρe_tend) ≈ parent(@. ᶜρ * cp_d * ᶜΠ * dθdt)

    # Add one hour of the source at constant density and compare the change in
    # θ with the prescribed change.
    δt = FT(3600)
    ᶜT_new = @. ᶜT + ᶜρe_tend / ᶜρ * δt / cv_d
    ᶜδθ = @. TD.potential_temperature(thermo_params, ᶜT_new, ᶜρ) -
       TD.potential_temperature(thermo_params, ᶜT, ᶜρ)
    @test all(isapprox.(parent(ᶜδθ), dθdt * δt; rtol = 1e-3))
end

@testset "Tendency computations" begin
    @testset "Placeholder - tendency accumulation" begin
        @test_skip "Tendency accumulation tests not yet implemented"
    end

    @testset "Placeholder - tendency budget closure" begin
        @test_skip "Tendency budget closure tests not yet implemented"
    end

    @testset "Placeholder - NaN detection" begin
        @test_skip "NaN detection tests not yet implemented"
    end
end
