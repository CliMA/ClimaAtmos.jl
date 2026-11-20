#=
Tests of the deferred 1M tendencies: the `cache!` and `cache_imp!` hooks of the
time stepper (`set_precomputed_quantities_deferred!`,
`set_implicit_precomputed_quantities_deferred!`) leave the tendencies that only
the implicit solve reads to `ensure_1m_tendencies!`, which evaluates them at the
state of their reader. The cache that the readers see must be the cache of the
undeferred functions.
=#
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaTimeSteppers as CTS

include("../test_helpers.jl")

deferred_1m_test_config() = CA.AtmosConfig(
    Dict(
        "config" => "column",
        "initial_condition" => "DYCOMS_RF02",
        "FLOAT_TYPE" => "Float64",
        "turbconv" => "prognostic_edmfx",
        "implicit_diffusion" => true,
        "approximate_linear_solve_iters" => 2,
        "edmfx_entr_model" => "Generalized",
        "edmfx_detr_model" => "Generalized",
        "edmfx_sgs_mass_flux" => true,
        "edmfx_sgs_diffusive_flux" => true,
        "edmfx_nh_pressure" => true,
        "edmfx_vertical_diffusion" => true,
        "edmfx_filter" => true,
        "prognostic_tke" => true,
        "microphysics_model" => "1M",
        "z_max" => 1500.0,
        "z_elem" => 30,
        "z_stretch" => false,
        "dt" => "60secs",
        "t_end" => "30mins",
        "ode_algo" => "ARS222",
        "toml" => [joinpath(pkgdir(CA), "toml", "default_parameters.toml")],
        "output_default_diagnostics" => false,
    );
    job_id = "deferred_1m_tendency_test",
)

@testset "Deferred 1M tendencies equal the undeferred cache at their readers" begin
    config = deferred_1m_test_config()
    (; Y, p, simulation) = generate_test_simulation(config)
    (; integrator) = simulation
    for _ in 1:10
        CTS.step!(integrator)
    end
    t = integrator.t
    @test CA.has_implicit_1m_microphysics(p)

    names = (
        :ᶜmp_tendency⁰, :ᶜmp_tendencyʲs,
        :ᶜmp_tendency_implicit⁰, :ᶜmp_tendency_implicitʲs, :ᶜsgs_moments,
    )
    snapshot() = map(n -> copy(parent(getproperty(p.precomputed, n))), names)
    poison!(ns) = foreach(ns) do n
        fill!(parent(getproperty(p.precomputed, n)), NaN)
    end
    (; sgs_moments_are_stale, implicit_1m_tendency_is_stale) = p.scratch
    # which of the SGS moments and the implicit tendency are out of date
    stale() = (sgs_moments_are_stale[], implicit_1m_tendency_is_stale[])

    # The `cache!` hook defers the implicit tendency and evaluates the updraft
    # tendency, which the Newton solve reads.
    CA.set_precomputed_quantities!(Y, p, t)
    @test stale() == (false, false)
    ref = snapshot()
    @test any(!iszero, ref[2]) && any(!iszero, ref[3])
    poison!((:ᶜmp_tendencyʲs, :ᶜmp_tendency_implicit⁰, :ᶜmp_tendency_implicitʲs))
    CA.set_precomputed_quantities_deferred!(Y, p, t)
    @test stale() == (false, true)
    @test parent(p.precomputed.ᶜmp_tendencyʲs) == ref[2]
    CA.ensure_1m_tendencies!(Y, p)
    @test stale() == (false, false)
    @test snapshot() == ref

    # The `cache_imp!` hook defers the SGS moments and the implicit tendency to
    # the Newton residual and the Jacobian.
    CA.set_implicit_precomputed_quantities!(Y, p, t)
    @test stale() == (false, false)
    ref = snapshot()
    Yₜ_ref = similar(Y)
    CA.implicit_tendency!(Yₜ_ref, Y, p, t)
    for read! in (
        () -> CA.implicit_tendency!(similar(Y), Y, p, t),
        () -> CA.update_jacobian!(integrator.cache.newtons_method_cache.j, Y, p, p.dtγ, t),
    )
        poison!((:ᶜmp_tendency_implicit⁰, :ᶜmp_tendency_implicitʲs, :ᶜsgs_moments))
        CA.set_implicit_precomputed_quantities_deferred!(Y, p, t)
        @test stale() == (true, true)
        read!()
        @test stale() == (false, false)
        @test snapshot() == ref
    end
    Yₜ = similar(Y)
    CA.set_implicit_precomputed_quantities_deferred!(Y, p, t)
    CA.implicit_tendency!(Yₜ, Y, p, t)
    @test parent(Yₜ.c) == parent(Yₜ_ref.c)
    @test parent(Yₜ.f) == parent(Yₜ_ref.f)

    # The explicit tendency reads the SGS moments; the implicit tendency stays
    # deferred.
    poison!((:ᶜsgs_moments,))
    CA.set_implicit_precomputed_quantities_deferred!(Y, p, t)
    CA.ensure_sgs_moments!(Y, p)
    @test stale() == (false, true)
    @test parent(p.precomputed.ᶜsgs_moments) == ref[5]
    CA.ensure_1m_tendencies!(Y, p)
    @test snapshot() == ref
end
