using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaTimeSteppers as CTS

# Unit tests for the acoustic-substepping timestepper: config resolution, the
# sub-cycled kinetic-energy gradient, the exact tendency split, and the
# divergence-damping forms.

const CAP = CA.Parameters

function box_config(;
    damping_form = "3d",
    damping = "auto",
    vertical = "implicit",
    order = 2,
    substeps = "3",
    dt = "0.5secs",
)
    return Dict{String, Any}(
        "initial_condition" => "DryDensityCurrentProfile",
        "config" => "box",
        "FLOAT_TYPE" => "Float64",
        "hyperdiff" => nothing,
        "smagorinsky_lilly" => nothing,
        "x_max" => 6400.0,
        "y_max" => 6400.0,
        "z_max" => 6400.0,
        "x_elem" => 2,
        "y_elem" => 2,
        "z_elem" => 8,
        "z_stretch" => false,
        "dt" => dt,
        "t_end" => "60secs",
        "disable_surface_flux_tendency" => true,
        "output_default_diagnostics" => false,
        "dt_save_state_to_disk" => "Inf",
        "log_progress" => false,
        "acoustic_substeps" => substeps,
        "acoustic_substep_vertical" => vertical,
        "acoustic_substep_order" => order,
        "acoustic_substep_damping_form" => damping_form,
        "acoustic_substep_damping" => damping,
        "output_dir" => mktempdir(),
    )
end

resolved_alg(; kwargs...) = CA.ode_configuration(
    Float64,
    CA.AtmosConfig(box_config(; kwargs...); job_id = "resolve_probe").parsed_args,
)

box_integrator(; kwargs...) =
    CA.get_simulation(CA.AtmosConfig(box_config(; kwargs...); job_id = "unit")).integrator

@testset "config resolution" begin
    a = resolved_alg()  # defaults: 3d / auto
    @test a isa CA.AcousticMultirate
    @test a.damping_form isa CA.FullDivergenceDamping
    @test a.β_d == 0.4  # auto

    b = resolved_alg(damping_form = "horizontal", damping = 0.8)
    @test b.damping_form isa CA.HorizontalDivergenceDamping
    @test b.β_d == 0.8                                 # Real
    @test resolved_alg(damping = "0.8").β_d == 0.8     # numeric String

    # acoustic_substeps = 0 leaves the plain ODE algorithm (no substepping wrapper).
    @test !(resolved_alg(substeps = "0") isa CA.AcousticMultirate)
end

@testset "kinetic-energy gradient discrete form" begin
    integ = box_integrator()
    Y, p, t, f = integ.u, integ.p, integ.t, integ.cache.f
    f.cache!(Y, p, t)

    Yₜ = zero(Y)
    CA.kinetic_energy_gradient_tendency!(Yₜ, Y, p, t)

    # Momentum-only: mass and energy tendencies stay zero.
    @test all(iszero, parent(Yₜ.c.ρ))
    @test all(iszero, parent(Yₜ.c.ρe_tot))

    ᶜK = p.precomputed.ᶜK
    ᶜp = p.precomputed.ᶜp
    ᶜΦ = p.core.ᶜΦ
    thermo_params = CAP.thermodynamics_params(p.params)

    u₃_ref = zero(Y.f.u₃)
    @. u₃_ref = -CA.ᶠgradᵥ(ᶜK)
    @test parent(Yₜ.f.u₃) == parent(u₃_ref)

    uₕ_ref = zero(Y.c.uₕ)
    @. uₕ_ref = -CA.C12(CA.gradₕ(ᶜK + ᶜΦ - CA.phi_r(thermo_params, ᶜp)))
    @test parent(Yₜ.c.uₕ) == parent(uₕ_ref)
end

@testset "exact-split identity" begin
    # At the freeze state, the horizontal acoustic tendency plus the sub-cycled
    # momentum advection (kinetic-energy gradient and rotational momentum flux)
    # plus the frozen slow forcing reconstructs the full explicit tendency, for
    # both vertical treatments.
    for vertical in ("implicit", "explicit")
        integ = box_integrator(; vertical, damping = 0.0)
        Y, p, t, f = integ.u, integ.p, integ.t, integ.cache.f

        f.cache!(Y, p, t)
        T_exp, T_lim = zero(Y), zero(Y)
        f.T_exp_T_lim!(T_exp, T_lim, Y, p, t)

        G, G_lim, A_buf = zero(Y), zero(Y), zero(Y)
        CA.acoustic_slow_forcing!(G, G_lim, A_buf, f, Y, p, t)
        Ah, B, C = zero(Y), zero(Y), zero(Y)
        CA.horizontal_acoustic_tendency!(Ah, Y, p, t)
        CA.kinetic_energy_gradient_tendency!(B, Y, p, t)
        CA.rotational_momentum_flux_tendency!(C, Y, p, t)

        recon = zero(Y)
        @. recon = Ah + B + C + G
        for name in (:ρ, :ρe_tot, :uₕ)
            @test parent(getproperty(recon.c, name)) ≈
                  parent(getproperty(T_exp.c, name))
        end
        @test parent(recon.f.u₃) ≈ parent(T_exp.f.u₃)
    end
end

@testset "divergence-damping forms" begin
    integ = box_integrator()
    Y, p, t, f = integ.u, integ.p, integ.t, integ.cache.f
    f.cache!(Y, p, t)
    ν_d = 1.234

    Yh = zero(Y)
    CA.divergence_damping_tendency!(Yh, Y, p, CA.HorizontalDivergenceDamping(), ν_d)
    Y3 = zero(Y)
    CA.divergence_damping_tendency!(Y3, Y, p, CA.FullDivergenceDamping(), ν_d)

    # On the flat box the horizontal divergence of the full velocity equals that
    # of the horizontal velocity, so `3d` = `horizontal` + the vertical term.
    ᶜu = p.precomputed.ᶜu
    ᶠu³ = p.precomputed.ᶠu³
    δu, δuₕ = zero(Y.c.ρ), zero(Y.c.ρ)
    @. δu = CA.divₕ(ᶜu)
    @. δuₕ = CA.divₕ(Y.c.uₕ)
    @test parent(δu) ≈ parent(δuₕ)

    ᶜdivᵥu³ = zero(Y.c.ρ)
    @. ᶜdivᵥu³ = CA.ᶜdivᵥ(ᶠu³)
    vert = zero(Y.c.uₕ)
    @. vert = ν_d * CA.wgradₕ(ᶜdivᵥu³)
    diff = zero(Y.c.uₕ)
    @. diff = Y3.c.uₕ - Yh.c.uₕ
    @test parent(diff) ≈ parent(vert)
end

function run_steps!(integ, n_steps)
    for _ in 1:n_steps
        CTS.step!(integ)
    end
    return integ
end

@testset "state update" begin
    integ = box_integrator()
    u_before = deepcopy(integ.u)
    CTS.step!(integ)
    Δρ = maximum(abs, parent(integ.u.c.ρ) .- parent(u_before.c.ρ))
    Δu₃ = maximum(abs, parent(integ.u.f.u₃) .- parent(u_before.f.u₃))
    @test Δρ > 0
    @test Δu₃ > 0
    @test all(isfinite, parent(integ.u.c.ρ))
end

@testset "mass and energy conservation" begin
    integ = box_integrator(damping = 0.0)
    mass_0 = sum(integ.u.c.ρ)
    energy_0 = sum(integ.u.c.ρe_tot)
    run_steps!(integ, 20)
    mass_error = abs(sum(integ.u.c.ρ) - mass_0) / abs(mass_0)
    energy_error = abs(sum(integ.u.c.ρe_tot) - energy_0) / abs(energy_0)
    @test mass_error < 1e-8
    @test energy_error < 1e-4
end

@testset "small-timestep agreement with the plain scheme" begin
    n_steps = 5
    plain = box_integrator(substeps = "0", dt = "0.2secs", order = 1)
    substepped = box_integrator(substeps = "3", dt = "0.2secs", order = 1)
    run_steps!(plain, n_steps)
    run_steps!(substepped, n_steps)
    for name in (:ρ, :ρe_tot)
        reference = parent(getproperty(plain.u.c, name))
        candidate = parent(getproperty(substepped.u.c, name))
        relative_difference =
            maximum(abs, reference .- candidate) / maximum(abs, reference)
        @test relative_difference < 5e-2
    end
end

@testset "timestep sweep" begin
    for dt in ("0.5secs", "1secs", "2secs")
        integ = box_integrator(substeps = "auto", dt = dt, order = 1)
        t_start = integ.t
        run_steps!(integ, 5)
        @test integ.t > t_start
        @test all(isfinite, parent(integ.u.c.ρ))
        @test all(isfinite, parent(integ.u.f.u₃))
    end
end

@testset "sub-step count is rounded up to divide dt exactly" begin
    # See CliMA/ClimaTimeSteppers.jl#442: an ITime dt that the count does not
    # divide exactly throws in `dt / n_sub` rather than truncating, so
    # exact_n_sub rounds the resolved count up to a divisor.
    dt = CA.ITime(0.5)                       # 5e8 ns, not divisible by 3
    @test_throws ErrorException dt / 3
    @test CA.exact_n_sub(dt, 3) == 4         # next count that divides 5e8 ns
    @test CA.exact_n_sub(dt, 5) == 5         # already a divisor, left unchanged
    @test dt / CA.exact_n_sub(dt, 3) isa CA.ITime
    # The CFL-derived auto count is snapped through the same helper.
    n_auto = CA.exact_n_sub(dt, CA.auto_n_sub(dt, 130.0, 340.0))
    @test dt / n_auto isa CA.ITime
end
