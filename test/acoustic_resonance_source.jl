using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaTimeSteppers as CTS

# Tests for the acoustic-substepping resonance-source removal (the kinetic-energy
# gradient evaluated in the sub-cycle) and the generalized divergence damping.

const CAP = CA.Parameters

function box_config(;
    kinetic_energy = "fast",
    damping_form = "3d",
    damping = "auto",
    implicit_split = false,
    vertical = "implicit",
    order = 2,
    substeps = "3",
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
        "dt" => "0.5secs",
        "t_end" => "60secs",
        "disable_surface_flux_tendency" => true,
        "output_default_diagnostics" => false,
        "dt_save_state_to_disk" => "Inf",
        "log_progress" => false,
        "acoustic_substeps" => substeps,
        "acoustic_substep_vertical" => vertical,
        "acoustic_substep_order" => order,
        "acoustic_substep_kinetic_energy" => kinetic_energy,
        "acoustic_substep_damping_form" => damping_form,
        "acoustic_substep_damping" => damping,
        "acoustic_substep_implicit_split" => implicit_split,
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
    a = resolved_alg()  # defaults: fast / 3d / auto
    @test a isa CA.AcousticMultirate
    @test a.kinetic_energy isa CA.FastKineticEnergy
    @test a.damping_form isa CA.FullDivergenceDamping
    @test a.β_d == 0.4  # auto resolves to the fast value

    # Recoverability of the PR2-validated configuration.
    b = resolved_alg(
        kinetic_energy = "slow",
        damping_form = "horizontal",
        damping = 1.5,
    )
    @test b.kinetic_energy isa CA.FrozenKineticEnergy
    @test b.damping_form isa CA.HorizontalDivergenceDamping
    @test b.β_d == 1.5

    @test resolved_alg(kinetic_energy = "slow", damping = "auto").β_d == 1.5
    @test resolved_alg(damping = 0.8).β_d == 0.8      # Real
    @test resolved_alg(damping = "0.8").β_d == 0.8    # numeric String
    @test resolved_alg(damping_form = "3d_perturbation").damping_form isa
          CA.PerturbationDivergenceDamping

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
    # kinetic-energy gradient plus the frozen slow forcing reconstructs the full
    # explicit tendency, for both vertical treatments.
    for vertical in ("implicit", "explicit")
        integ = box_integrator(; vertical, damping = 0.0)
        Y, p, t, f = integ.u, integ.p, integ.t, integ.cache.f

        f.cache!(Y, p, t)
        T_exp, T_lim = zero(Y), zero(Y)
        f.T_exp_T_lim!(T_exp, T_lim, Y, p, t)

        G, G_lim, A_buf = zero(Y), zero(Y), zero(Y)
        CA.acoustic_slow_forcing!(
            G,
            G_lim,
            A_buf,
            f,
            Y,
            p,
            t,
            CA.FastKineticEnergy(),
        )
        Ah, B = zero(Y), zero(Y)
        CA.horizontal_acoustic_tendency!(Ah, Y, p, t)
        CA.kinetic_energy_gradient_tendency!(B, Y, p, t)

        recon = zero(Y)
        @. recon = Ah + B + G
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
    CA.divergence_damping_tendency!(
        Yh,
        Y,
        p,
        CA.HorizontalDivergenceDamping(),
        ν_d,
        nothing,
    )
    Y3 = zero(Y)
    CA.divergence_damping_tendency!(
        Y3,
        Y,
        p,
        CA.FullDivergenceDamping(),
        ν_d,
        nothing,
    )

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

    # With a divergence-free reference the perturbation form equals the full form.
    ᶜδ₀ = zero(Y.c.ρ)
    Yp = zero(Y)
    CA.divergence_damping_tendency!(
        Yp,
        Y,
        p,
        CA.PerturbationDivergenceDamping(),
        ν_d,
        ᶜδ₀,
    )
    @test parent(Yp.c.uₕ) == parent(Y3.c.uₕ)
end
