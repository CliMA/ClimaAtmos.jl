#=
Rest-state well-balance probe.

Builds the model + rest initial state from a YAML config, evaluates the RHS
ONCE, and reports the per-component tendency residual. A well-balanced scheme
gives machine-zero everywhere at rest; a non-zero residual in the tangential
Cartesian momentum (ρu1/ρu2/ρu3, minus the radial part) localizes the missing
−∂_ξ3(Ja³) metric cross-term over terrain.

    julia --project=experiments/dg_dycore experiments/dg_dycore/rest_residual_probe.jl \
        experiments/dg_dycore/configs/held_suarez_curvilinear_roe_wb_topo.yml

Cheap (single tendency eval); the only cost is model build + precompile.
=#

include(joinpath(@__DIR__, "driver.jl"))  # includes DGDycore + `using`; problem_from_yaml
import ClimaCore: Fields, Geometry

isempty(ARGS) && error("usage: rest_residual_probe.jl <config.yml>")
prob = problem_from_yaml(ARGS[1])
prob.ic_source == :rest ||
    @warn "ic_source is $(prob.ic_source); probe expects :rest for a true balance test"

sim = DGSimulation(prob)
m = sim.model
Y = copy(sim.Y₀)
FT = DGDycore.float_type(m)

dY = similar(Y)
DGDycore.rhs_fddg!(dY, Y, m, FT(0))

# Radial unit vector to split the momentum residual into radial (handled by the
# vertical ρw pair) vs tangential (the along-surface term the cross-term fixes).
(; eR1, eR2, eR3) = m.fields
dr = @. dY.c.ρu1 * eR1 + dY.c.ρu2 * eR2 + dY.c.ρu3 * eR3
dt1 = @. dY.c.ρu1 - dr * eR1
dt2 = @. dY.c.ρu2 - dr * eR2
dt3 = @. dY.c.ρu3 - dr * eR3
dtmag = @. sqrt(dt1^2 + dt2^2 + dt3^2)

mx(f) = maximum(abs, parent(f))
@info "Rest residual (per component, |dY| at rest)" dρ = mx(dY.c.ρ) dρe =
    mx(dY.c.ρe) dρu1 = mx(dY.c.ρu1) dρu2 = mx(dY.c.ρu2) dρu3 = mx(dY.c.ρu3) dρw =
    mx(dY.f.ρw)
@info "Momentum residual split" radial = mx(dr) tangential = mx(dtmag)
