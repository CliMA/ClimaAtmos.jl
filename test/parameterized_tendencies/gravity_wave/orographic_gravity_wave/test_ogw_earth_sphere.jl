#=
Orographic gravity-wave drag on a coarse sphere with Earth topography: one compute step,
then 15 model steps.

Builds a small simulation (`h_elem = 4`, 10 levels) with `topography: "Earth"` and
`orographic_gravity_wave: "raw_topo"`, which loads the shipped `ogw_computed_drag_h4` drag
file (checking its build record) and builds the TOFD coefficient from the Earth elevation
data. It sets a uniform 10 m/s westerly and 2 m/s southerly wind (the model starts at
rest, which gives no drag) and calls the OGW compute step once, checking the wiring end to
end: the drag is finite, nonzero, within the limiter, and on balance opposes the wind. It
then runs the model for 15 steps with the drag recomputed at every step, and checks that
the state and the drag stay finite, the drag stays within the limiter, and the imposed
westerly is not reversed. The single-column physics is checked against GFDL `topo_drag` in
test_ogw.jl.

Takes about 4 minutes (8 threads) and about 12 GB of memory, mostly the simulation build,
which reads the full Earth elevation data for TOFD. So it runs as its own Buildkite step,
not in `test/runtests.jl`.
=#

using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaTimeSteppers as CTS
import ClimaCore: Fields, Geometry

@testset "OGW step on an Earth-topography sphere" begin
    output_dir = mktempdir()
    config = CA.AtmosConfig(
        Dict(
            "h_elem" => 4,
            "z_elem" => 10,
            "z_max" => 30000.0,
            "z_stretch" => false,
            "dt" => "400secs",
            "t_end" => "6000secs",
            "dt_ogw" => "400secs",
            "topography" => "Earth",
            "topo_smoothing" => true,
            "orographic_gravity_wave" => "raw_topo",
            "output_default_diagnostics" => false,
            "output_dir" => output_dir,
        );
        job_id = "ogw_earth_sphere",
    )
    simulation = CA.get_simulation(config)
    Y = simulation.integrator.u
    p = simulation.integrator.p
    FT = eltype(Y.c.ρ)

    u_wind, v_wind = FT(10), FT(2)
    ᶜlg = Fields.local_geometry_field(Y.c)
    Y.c.uₕ .= Geometry.Covariant12Vector.(Geometry.UVVector.(u_wind, v_wind), ᶜlg)
    CA.orographic_gravity_wave_compute_tendency!(
        Y,
        p,
        p.atmos.orographic_gravity_wave,
    )

    uf = Array(parent(p.orographic_gravity_wave.ᶜuforcing))
    vf = Array(parent(p.orographic_gravity_wave.ᶜvforcing))
    @test all(isfinite, uf) && all(isfinite, vf)
    @test count(!iszero, uf) > 0
    # The limiter caps the drag vector at 3e-3·√2 m/s².
    @test maximum(sqrt.(uf .^ 2 .+ vf .^ 2)) <= FT(3e-3) * sqrt(FT(2)) * (1 + sqrt(eps(FT)))
    # Summed over the globe, the drag decelerates the imposed wind.
    @test sum(uf .* u_wind .+ vf .* v_wind) < 0
    # TOFD uses the Earth elevation data, so its coefficient is nonzero somewhere.
    @test maximum(parent(p.orographic_gravity_wave.ᶜtofd_Cd)) > 0

    # Run 15 steps; dt_ogw = dt recomputes the drag at every step.
    integrator = simulation.integrator
    for _ in 1:15
        CTS.step!(integrator)
    end
    Y = integrator.u
    uf = Array(parent(p.orographic_gravity_wave.ᶜuforcing))
    vf = Array(parent(p.orographic_gravity_wave.ᶜvforcing))
    @test all(isfinite, parent(Y.c)) && all(isfinite, parent(Y.f))
    @test all(isfinite, uf) && all(isfinite, vf)
    @test maximum(sqrt.(uf .^ 2 .+ vf .^ 2)) <= FT(3e-3) * sqrt(FT(2)) * (1 + sqrt(eps(FT)))
    # The imposed westerly is not reversed: its mean over all cells stays positive.
    u_now = Array(parent(Geometry.UVVector.(Y.c.uₕ).components.data.:1))
    @test sum(u_now) / length(u_now) > 0
end
