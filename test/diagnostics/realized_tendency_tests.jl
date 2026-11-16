#=
Tests the realized per-operator tendencies recorded by `RealizedIncrements`
and exposed by the `tn_<field>_<operator>` diagnostics of
`tendency_diagnostics.jl`.
=#
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaTimeSteppers as CTS

const FT = Float64

function realized_test_simulation()
    ctx = ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded())
    grid = CA.ColumnGrid(
        FT;
        context = ctx,
        z_elem = 20,
        z_max = 3000.0,
        z_stretch = false,
    )
    params = CA.ClimaAtmosParameters(FT)
    vdp = CA.Parameters.vert_diff_params(params)
    model = CA.AtmosModel(
        grid;
        params,
        setup = CA.Setups.Bomex(FT),
        microphysics_model = CA.NonEquilibriumMicrophysics1M(),
        vertical_diffusion = CA.VerticalDiffusion{FT}(;
            disable_momentum_vertical_diffusion = false,
            C_E = vdp.C_E,
        ),
    )
    return CA.AtmosSimulation(
        model;
        dt = 60,
        t_end = 600,
        output_dir = mktempdir(),
        default_callbacks = false,
    )
end

@testset "Realized operator tendencies" begin
    simulation = realized_test_simulation()
    integrator = simulation.integrator
    Y = integrator.u
    p = integrator.p

    # No recorder until a realized diagnostic asks for one.
    @test isnothing(CA.realized_increments(p))
    CTS.step!(integrator)
    @test isnothing(CA.realized_increments(p))

    # Computing a realized diagnostic allocates the recorder; its tendencies
    # are zero until the next step.
    diag = CA.Diagnostics.get_diagnostic_variable("tn_rhoe_turb")
    field = diag.compute(Y, p, integrator.t)
    recorder = CA.realized_increments(p)
    @test recorder isa CA.RealizedIncrements
    @test all(iszero, parent(field))
    @test CA.enable_realized_increments!(p, Y) === recorder

    for _ in 1:3
        Y_before = copy(Y)
        CTS.step!(integrator)
        Δt = FT(integrator.dt)

        # The realized tendencies of all operators close the step budget.
        total = similar(Y)
        total .= zero(eltype(total))
        foreach(t -> (total .+= t), recorder.tendencies)
        expected = (parent(Y) .- parent(Y_before)) ./ Δt
        scale = maximum(abs, parent(Y)) / Δt
        @test maximum(abs, parent(total) .- expected) <= 1e-10 * scale
    end

    # Surface fluxes and diffusion change the energy within the column step,
    # and the ARK step changes it through the wave core.
    tendencies = recorder.tendencies
    @test maximum(abs, parent(tendencies.turb.c.ρe_tot)) > 0
    @test maximum(abs, parent(tendencies.ark.c.ρe_tot)) > 0
    for name in CA.REALIZED_OPERATORS
        @test all(isfinite, parent(getproperty(tendencies, name)))
    end

    # The diagnostics return the recorded tendencies of the last step.
    for (short, name, path) in (
        ("tn_rhoe_turb", :turb, x -> x.c.ρe_tot),
        ("tn_rhoqt_micro", :micro, x -> x.c.ρq_tot),
        ("tn_rhoqlcl_wrepair", :wrepair, x -> x.c.ρq_lcl),
        ("tn_rho_ark", :ark, x -> x.c.ρ),
        ("tn_rhoqr_fixer", :fixer, x -> x.c.ρq_rai),
        ("tn_rhoqt_vtrans", :vtrans, x -> x.c.ρq_tot),
    )
        diag = CA.Diagnostics.get_diagnostic_variable(short)
        result = diag.compute(Y, p, integrator.t)
        @test parent(result) == parent(path(getproperty(tendencies, name)))
    end

    # Instantaneous ARK tendencies still evaluate at the current state.
    for short in ("tn_rhoe_iva", "tn_rhoqt_additional", "tn_rhoe_expvadv")
        diag = CA.Diagnostics.get_diagnostic_variable(short)
        result = diag.compute(Y, p, integrator.t)
        @test all(isfinite, parent(Base.materialize(result)))
    end
end

@testset "Tendency diagnostic manifest" begin
    names = CA.Diagnostics.tendency_debug_short_names()
    @test allunique(names)
    for name in names
        @test haskey(CA.Diagnostics.ALL_DIAGNOSTICS, name)
    end
    for entry in CA.Diagnostics.tendency_debug_field_manifest()
        for op in CA.REALIZED_OPERATORS
            @test op in entry.processes
        end
    end
end
