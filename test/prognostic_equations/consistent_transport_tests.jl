#=
Consistency of the implicit vertical transport of energy, total water, and
condensates in non-equilibrium microphysics configurations. With one
reconstruction for all three, a scalar that is an affine function of the
others stays affine under transport, which keeps the temperature
diagnosed from `(ρe_tot, ρq_tot, ρq_cond)` that of the mixed air at any stage
Courant number. Also checks the Jacobian blocks of that transport against
finite differences and the temperature of a column with a water step
transported at a stage Courant number above one.
=#
using Test
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: Fields, Geometry, Operators, Spaces
import ClimaCore: MatrixFields
import ClimaTimeSteppers as CTS
import Thermodynamics as TD

include("../test_helpers.jl")

function _measure_allocs(f!, args...)
    f!(args...)
    return @allocated f!(args...)
end

column_config(; dt = "10secs", kwargs...) = CA.AtmosConfig(
    Dict{String, Any}(
        "config" => "column",
        "FLOAT_TYPE" => "Float64",
        "dt" => dt,
        "z_max" => 6000.0,
        "z_elem" => 60,
        "z_stretch" => false,
        "initial_condition" => "PrecipitatingColumn",
        "microphysics_model" => "1M",
        "disable_surface_flux_tendency" => true,
        "output_default_diagnostics" => false,
        (string(key) => value for (key, value) in kwargs)...,
    );
    job_id = "consistent_transport_test",
)

@testset "Affine consistency of energy and water transport" begin
    (; Y, p, simulation) = generate_test_simulation(column_config())
    t = simulation.integrator.t
    FT = eltype(Y)
    (; dtγ) = p
    (; ᶠu³, ᶜp) = p.precomputed
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    ᶠρ = @. CA.ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ
    z_c = Fields.coordinate_field(Y.c).z
    z_max = Spaces.z_max(axes(Y.f))
    @test CA.grid_mean_base_upwinding(Y) == Val(:first_order)
    @test p.atmos.numerics.energy_q_tot_upwinding == Val(:vanleer_limiter)

    # Only the grid-mean material transport is under test: sedimentation
    # velocities are zeroed, the other condensates removed, and the cache
    # pressure set to zero so that the pressure work vanishes.
    ᶜp .= FT(0)
    for w_name in (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ)
        fill!(parent(getproperty(p.precomputed, w_name)), FT(0))
    end
    for ρχ_name in (:ρq_icl, :ρq_rai, :ρq_sno)
        fill!(parent(getproperty(Y.c, ρχ_name)), FT(0))
    end
    A, B, C = FT(2.5e5), FT(2.4e6), FT(-3.3e5)
    u3_arr = parent(ᶠu³.components.data.:1)
    Yₜ = similar(Y)
    Yₜ_corr = similar(Y)
    Yₜ_wired = similar(Y)

    function test_affine(Yₜ_k, label)
        ρe = parent(Yₜ_k.c.ρe_tot)
        affine =
            A .* parent(Yₜ_k.c.ρ) .+ B .* parent(Yₜ_k.c.ρq_tot) .+
            C .* parent(Yₜ_k.c.ρq_lcl)
        scale = maximum(abs, affine)
        @test scale > 0 || label == "correction"
        @test maximum(abs, ρe .- affine) <= FT(1e-10) * max(scale, FT(1))
    end

    # The linear reconstructions commute with affine combinations of
    # independent profiles; the van Leer limiter is nonlinear and commutes
    # only when the profiles share one shape `ψ(z)` with coefficients of one
    # sign, as across a single cloud-top jump.
    ψ_step = @. ifelse(z_c < z_max / 2, FT(1), FT(0))
    ψ_smooth = @. FT(0.5) * (1 + tanh((z_max / 2 - z_c) / FT(300)))
    cases = (
        (Val(:first_order), ψ_step, ψ_smooth),
        (Val(:third_order), ψ_step, ψ_smooth),
        (Val(:vanleer_limiter), ψ_step, ψ_step),
        (Val(:vanleer_limiter), ψ_smooth, ψ_smooth),
    )
    for (upwinding, ψ_q, ψ_l) in cases, C_val in (FT(0.5), FT(2.5))
        @. Y.c.ρq_tot = Y.c.ρ * (FT(2e-3) + FT(8e-3) * ψ_q)
        @. Y.c.ρq_lcl = Y.c.ρ * (FT(1e-4) + FT(9e-4) * ψ_l)
        # An affine function of the transported specific contents stands in
        # for the thermodynamic energy.
        @. Y.c.ρe_tot = Y.c.ρ * A + B * Y.c.ρq_tot + C * Y.c.ρq_lcl
        u3_arr .= C_val / FT(dtγ)
        ᶜe_tot = @. CA.specific(Y.c.ρe_tot, Y.c.ρ)
        ᶜq_tot = @. CA.specific(Y.c.ρq_tot, Y.c.ρ)
        ᶜq_lcl = @. CA.specific(Y.c.ρq_lcl, Y.c.ρ)

        Yₜ .= zero(eltype(Yₜ))
        CA.implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
        test_affine(Yₜ, "residual")

        # The post-Newton antidiffusive correction of each transported scalar
        # with the reconstruction under test; `ρ` receives none.
        Yₜ_corr .= zero(eltype(Yₜ_corr))
        for (ᶜρχₜ, ᶜχ) in (
            (Yₜ_corr.c.ρe_tot, ᶜe_tot),
            (Yₜ_corr.c.ρq_tot, ᶜq_tot),
            (Yₜ_corr.c.ρq_lcl, ᶜq_lcl),
        )
            CA.add_antidiffusive_upwind1_transport!(
                ᶜρχₜ, ᶠρ, ᶠu³, ᶜχ, dtγ, upwinding,
            )
        end
        test_affine(Yₜ_corr, "correction")

        # The default `energy_q_tot_upwinding` wires the same correction into
        # `add_vertical_advection_corrections!` for all three scalars.
        if upwinding == Val(:vanleer_limiter)
            Yₜ_wired .= zero(eltype(Yₜ_wired))
            CA.add_vertical_advection_corrections!(Yₜ_wired, Y, p, dtγ)
            @test parent(Yₜ_wired.c) == parent(Yₜ_corr.c)
        end
    end
end

@testset "Jacobian of the upwind energy and water transport" begin
    (; Y, p, simulation) = generate_test_simulation(column_config())
    t = simulation.integrator.t
    FT = eltype(Y)
    dtγ = FT(60)
    (; ᶠu³) = p.precomputed
    z_c = Fields.coordinate_field(Y.c).z
    z_f = Fields.coordinate_field(Y.f).z
    z_max = Spaces.z_max(axes(Y.f))
    matrix = simulation.integrator.cache.newtons_method_cache.j.cache.matrix.matrix

    # The condensates stay above the finite-difference perturbation
    # everywhere, so the `max(0, q_liq)` of the thermodynamic state is inactive.
    @. Y.c.ρq_lcl =
        Y.c.ρ * (FT(1e-5) + FT(1e-3) * exp(-(z_c - FT(2000))^2 / FT(1e6)))
    @. Y.c.ρq_rai =
        Y.c.ρ * (FT(1e-5) + FT(2e-3) * exp(-(z_c - FT(1500))^2 / FT(1e6)))
    @. Y.c.ρq_tot =
        Y.c.ρ * FT(8e-3) * (1 + FT(0.3) * sin(FT(2π) * z_c / z_max)) +
        Y.c.ρq_lcl + Y.c.ρq_rai
    @. Y.f.u₃ = Geometry.Covariant3Vector(
        Geometry.WVector(FT(2.5) * sin(FT(π) * z_f / z_max)),
    )
    CA.set_precomputed_quantities!(Y, p, t)
    # The sedimentation velocities are zeroed so the blocks under test hold
    # the grid-mean advection alone.
    for w_name in (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ)
        fill!(parent(getproperty(p.precomputed, w_name)), FT(0))
    end
    CA.update_sedimentation_jacobian!(matrix, Y, p, dtγ)
    @test _measure_allocs(CA.update_sedimentation_jacobian!, matrix, Y, p, dtγ) ==
          0

    arr(f) = Array(parent(f))
    ᶜv = @. sin(z_c / FT(700)) + FT(0.5)
    Yₜ = similar(Y)
    # Central differences of `implicit_vertical_advection_tendency!`, with the
    # cache (and `ᶜp` in particular) refreshed at each perturbed state.
    function fd_derivative(row_sym, col_sym)
        field = getproperty(Y.c, col_sym)
        field_orig = copy(field)
        δ = FT(1e-3) * maximum(abs, arr(field_orig))
        tendencies = map((+1, -1)) do sign
            @. field = field_orig + sign * δ * ᶜv
            CA.set_implicit_precomputed_quantities!(Y, p, t)
            for w_name in (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ)
                fill!(parent(getproperty(p.precomputed, w_name)), FT(0))
            end
            fill!(Yₜ, 0)
            CA.implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
            copy(arr(getproperty(Yₜ.c, row_sym)))
        end
        @. field = field_orig
        CA.set_implicit_precomputed_quantities!(Y, p, t)
        for w_name in (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ)
            fill!(parent(getproperty(p.precomputed, w_name)), FT(0))
        end
        return dtγ .* (tendencies[1] .- tendencies[2]) ./ (2 * δ)
    end
    for (row_sym, col_sym) in (
        (:ρq_tot, :ρq_tot),
        (:ρe_tot, :ρe_tot),
        (:ρe_tot, :ρq_tot),
        (:ρe_tot, :ρ),
        (:ρq_tot, :ρ),
        (:ρe_tot, :ρq_lcl),
    )
        block = matrix[
            MatrixFields.FieldName(:c, row_sym),
            MatrixFields.FieldName(:c, col_sym),
        ]
        Jv = arr(@. block * ᶜv)
        row_sym == col_sym && (Jv .+= arr(ᶜv))
        fd = fd_derivative(row_sym, col_sym)
        @test norm(Jv .- fd) / norm(fd) < FT(5e-4)
    end
end

@testset "Temperature of a moist column transported at C_γ > 1" begin
    # A uniform temperature with a sharp step in total water and cloud liquid
    # under a uniform updraft. One forward-Euler application of the implicit
    # transport residual plus its post-Newton correction, with the Courant
    # limiter evaluated at `C_γ = 2.5`, applies the blend of first-order upwind
    # and van Leer fluxes that an implicit stage would. Every interior level
    # then receives air from the same distance below, so the temperature
    # diagnosed from the transported mixture stays uniform up to the mixing
    # across the step, which is where inconsistent reconstructions would leave
    # a spurious extremum.
    (; Y, p, simulation) = generate_test_simulation(column_config())
    t = simulation.integrator.t
    FT = eltype(Y)
    thp = CA.CAP.thermodynamics_params(p.params)
    z_c = Fields.coordinate_field(Y.c).z
    z_f = Fields.coordinate_field(Y.f).z
    n_levels = Spaces.nlevels(axes(Y.c))
    Δz = Spaces.z_max(axes(Y.f)) / n_levels
    dtγ = FT(p.dtγ)
    C_γ, C_euler = FT(2.5), FT(0.2)
    w = C_γ * Δz / dtγ
    dt_euler = C_euler * Δz / w
    T_0 = FT(285)
    (; ᶜΦ) = p.core
    ψ = @. ifelse(z_c < FT(1500), FT(1), FT(0))
    ᶜq_tot = @. FT(1e-3) + FT(6e-3) * ψ
    ᶜq_lcl = @. FT(5e-4) * ψ
    # The pressure of the initial column is kept, so it stays continuous
    # across the water step and the density jumps instead.
    CA.set_precomputed_quantities!(Y, p, t)
    @. Y.c.ρ =
        p.precomputed.ᶜp /
        (TD.gas_constant_air(thp, ᶜq_tot, ᶜq_lcl, FT(0)) * T_0)
    @. Y.c.ρq_tot = Y.c.ρ * ᶜq_tot
    @. Y.c.ρq_lcl = Y.c.ρ * ᶜq_lcl
    for ρχ_name in (:ρq_icl, :ρq_rai, :ρq_sno)
        fill!(parent(getproperty(Y.c, ρχ_name)), FT(0))
    end
    @. Y.f.u₃ = Geometry.Covariant3Vector(Geometry.WVector(w + 0 * z_f))
    ᶜK = CA.compute_kinetic(Y.c.uₕ, Y.f.u₃)
    @. Y.c.ρe_tot =
        Y.c.ρ * (TD.internal_energy(thp, T_0, ᶜq_tot, ᶜq_lcl, FT(0)) + ᶜK + ᶜΦ)
    CA.set_precomputed_quantities!(Y, p, t)
    for w_name in (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ)
        fill!(parent(getproperty(p.precomputed, w_name)), FT(0))
    end
    # The rigid lids compress or dilate the end cells, and the van Leer
    # stencil carries that two cells further in.
    interior = 4:(n_levels - 3)
    spread(f) = (x = vec(parent(f))[interior]; maximum(x) - minimum(x))
    spread_before = spread(p.precomputed.ᶜT)

    Yₜ = similar(Y)
    Yₜ .= zero(eltype(Yₜ))
    CA.implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    CA.add_vertical_advection_corrections!(Yₜ, Y, p, dtγ)
    for name in (:ρ, :ρe_tot, :ρq_tot, :ρq_lcl)
        ᶜχ = getproperty(Y.c, name)
        @. ᶜχ += dt_euler * $(getproperty(Yₜ.c, name))
    end
    CA.set_implicit_precomputed_quantities!(Y, p, t)
    spread_after = spread(p.precomputed.ᶜT)
    q_vap = @. CA.specific(Y.c.ρq_tot - Y.c.ρq_lcl, Y.c.ρ)
    # One first-order error of the upwind `e_tot` flux remains at the step:
    # `e_tot` is the water step on top of the geopotential ramp `g z`, and
    # the van Leer limiter zeroes the slope in the two cells at the step, so
    # each exchanges the ramp with a half-cell offset `g Δz / 2` through the
    # van Leer share `1 / C_γ` of the flux. The pressure work uses the
    # central face pressure, which is continuous across the density jump.
    g = CA.CAP.grav(p.params)
    cv_d = CA.CAP.cv_d(p.params)
    limiter_bound = 2 * (C_euler / C_γ) * g * Δz / (2 * cv_d)
    @test spread_before < 1e-6
    @test spread_after < spread_before + limiter_bound + FT(0.01)
    @test minimum(vec(parent(q_vap))[interior]) >= 0
end
