#=
Simulation assembly and time integration — port of the integration section
of baroclinic_wave_fddg_fluxform.jl. Both steppers run through
ClimaTimeSteppers (the OrdinaryDiffEqSSPRK dependency of the examples is
dropped): explicit SSP-RK3 via CTS.SSP33ShuOsher, HEVI via ARS343 + Newton
with the analytic column Jacobian.

Mirrors the shape of ClimaAtmos's AtmosSimulation / solve_atmos! (build
everything from the problem, return a compact result object).
=#

struct DGSimulation{M <: DGModel, Y}
    model::M
    Y₀::Y
end

"""
    DGSimulation(prob)

Build the model (grid, operators, constants) and the initial state.
"""
function DGSimulation(prob::DGProblem)
    m = DGModel(prob)
    Y₀ = initial_state(m)
    return DGSimulation(m, Y₀)
end

# Per-core tendency/Jacobian bundle
core_fns(::DGModel{FT, <:BaroclinicWaveFDDG}) where {FT} = (;
    rhs! = rhs_fddg!,
    rem! = remaining_tendency_fddg!,
    imp! = implicit_tendency_fddg!,
    jac = FDDGImplicitEquationJacobian,
    wfact! = fddg_implicit_equation_jacobian!,
)
core_fns(::DGModel{FT, <:VIProblem}) where {FT} = (;
    rhs! = rhs_vi!,
    rem! = remaining_tendency_vi!,
    imp! = implicit_tendency_vi!,
    jac = VIImplicitEquationJacobian,
    wfact! = vi_implicit_equation_jacobian!,
)

"""
    DGRunResult

Returned by [`run!`](@ref): `result.sol` (solution snapshots) and
`result.model` (the DGModel). Displays as one line — the underlying
solution types are enormous (especially on GPU).
"""
struct DGRunResult{S, M}
    sol::S
    model::M
end

function Base.show(io::IO, r::DGRunResult)
    t_end = try
        string(getfield(r, :sol).t[end])
    catch
        "?"
    end
    n = try
        length(getfield(r, :sol).u)
    catch
        "?"
    end
    print(
        io,
        "DGRunResult(t_end = $t_end s, $n snapshots; access via .sol, .model)",
    )
end
Base.show(io::IO, ::MIME"text/plain", r::DGRunResult) = show(io, r)

# Step monitor (physical-units diagnostics; a run must never be silent
# between startup and completion — a mid-run crash otherwise leaves no
# trace of when it happened).
function diag_str(Y, m::DGModel, t)
    uE, uN, w_c = dg_velocities(Y, m)
    p = dg_pressure(Y, m)
    Printf.@sprintf(
        "t=%8.0f  max|w|=%.4e  max|v|=%.4e  min p=%.4e  min ρ=%.4e",
        t,
        maximum(abs, parent(w_c)),
        maximum(abs, parent(uN)),
        minimum(parent(p)),
        minimum(parent(Y.c.ρ)),
    )
end

"""
    run!(sim::DGSimulation) -> DGRunResult

Integrate to `prob.t_end` with a step monitor every `prob.ndiag` steps and
a conservation report at the end.
"""
function run!(sim::DGSimulation)
    m = sim.model
    prob_cfg = m.prob
    FT = float_type(m)
    Y = copy(sim.Y₀)
    Δt = m.Δt
    t_end = prob_cfg.t_end

    mass_0 = sum(Y.c.ρ)
    energy_0 = sum(Y.c.ρe)

    fns = core_fns(m)
    # Smoke-check the RHS before committing to the run
    dY = similar(Y)
    fns.rhs!(dY, Y, m, FT(0))
    @info "Initial RHS" max_dρ = maximum(abs, parent(dY.c.ρ)) max_dρe =
        maximum(abs, parent(dY.c.ρe)) max_dc = maximum(abs, parent(dY.c)) max_df =
        maximum(abs, parent(dY.f))

    # min against t_end: a dt_save > t_end would otherwise collapse saveat
    # to [0] and silently discard the whole solution
    saveat_grid = collect(FT(0):min(t_end, prob_cfg.dt_save):t_end)
    # CTS-native callback (ClimaTimeSteppers ≥ 0.9 has its own callback
    # machinery; SciMLBase.DiscreteCallback is not accepted)
    monitor = CTS.Callbacks.EveryXSimulationSteps(
        # flush: julia fully buffers stdout when redirected to a file, so
        # without it the monitor lines appear only at process exit — AFTER
        # any crash's stderr, making a late crash look like an early one
        integrator -> (println(diag_str(integrator.u, m, integrator.t));
        flush(stdout)),
        prob_cfg.ndiag,
    )

    ode_prob, ode_algo = if prob_cfg.stepper == :hevi
        # Split-consistency check: rhs == implicit + remaining (exact by
        # construction; catches porting drift between the two paths)
        let dY1 = similar(Y), dY2 = similar(Y), dY3 = similar(Y)
            fns.rhs!(dY1, Y, m, FT(0))
            fns.imp!(dY2, Y, m, FT(0))
            fns.rem!(dY3, Y, m, FT(0))
            r(f) = maximum(
                abs,
                parent(getproperty(dY1, f)) .- parent(getproperty(dY2, f)) .-
                parent(getproperty(dY3, f)),
            )
            @info "HEVI split check" split_c = r(:c) split_f = r(:f)
        end
        jacobian = fns.jac(Y, m)
        ode_prob = CTS.ODEProblem(
            CTS.ClimaODEFunction(;
                T_imp! = CTS.ODEFunction(
                    (dY, Y, p, t) -> fns.imp!(dY, Y, p, t);
                    jac_prototype = jacobian,
                    Wfact = fns.wfact!,
                ),
                T_exp! = (dY, Y, p, t) -> fns.rem!(dY, Y, p, t),
            ),
            Y,
            (FT(0), t_end),
            m,
        )
        ode_prob,
        CTS.IMEXAlgorithm(CTS.ARS343(), CTS.NewtonsMethod(; max_iters = 2))
    else # :explicit — SSP-RK3 through CTS
        ode_prob = CTS.ODEProblem(
            CTS.ClimaODEFunction(;
                T_exp! = (dY, Y, p, t) -> fns.rhs!(dY, Y, p, t),
            ),
            Y,
            (FT(0), t_end),
            m,
        )
        ode_prob, CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher())
    end

    integrator = CTS.init(
        ode_prob,
        ode_algo;
        dt = Δt,
        saveat = saveat_grid,
        callback = monitor,
    )
    # ClimaAtmos-compatible NetCDF diagnostics (ua, va, wa, ta, pfull,
    # rhoa, ke, rv) — wired exactly like AtmosSimulations does
    writer = nothing
    if prob_cfg.output_dir !== nothing
        scheduled, writer = dg_diagnostics(
            m,
            Y;
            output_dir = prob_cfg.output_dir,
            period = prob_cfg.diag_period,
        )
        integrator =
            ClimaDiagnostics.IntegratorWithDiagnostics(integrator, scheduled)
    end
    CTS.solve!(integrator)
    writer === nothing || close(writer)
    sol = integrator.sol

    @info "Conservation" mass_rel = (sum(sol.u[end].c.ρ) - mass_0) / mass_0 energy_rel =
        (sum(sol.u[end].c.ρe) - energy_0) / energy_0
    if !prob_cfg.perturb
        Yend = sol.u[end]
        _, uN_end, w_end = dg_velocities(Yend, m)
        @info "Balanced-flow drift" max_v = maximum(abs, parent(uN_end)) max_w =
            maximum(abs, parent(w_end))
    end
    return DGRunResult(sol, m)
end
