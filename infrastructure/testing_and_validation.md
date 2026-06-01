# Testing and Validation Guide

This guide covers testing patterns for CliMA code: type-stability verification, allocation regression testing, and test group organization.

## Type-stability checks

The canonical home for type-stability tooling (`@inferred`, `JET.@report_opt`, `@code_warntype`, the Float32/Float64 test template) is [type_stability.md](../performance/type_stability.md). Use `@inferred` as a CI regression gate for any new physics function:

```julia
@test @inferred(my_physics_func(FT(1.0), FT(2.0))) isa FT
```

## Allocation regression tests

After implementing or modifying hot-path code, verify zero allocations with the warm-up + `@allocated == 0` regression pattern documented in [allocation_debugging.md §1](../performance/allocation_debugging.md). Allocation benchmarks in `perf/` are not run in CI; allocation regressions must be caught at review time.

## Aqua.jl quality checks

All CliMA packages run `Aqua.jl` tests in CI. These checks catch common package quality issues:

- `test_stale_deps`: fails if a package in `[deps]` is not used in source code. This is the most common failure — usually caused by adding a dev tool to `[deps]` instead of `[extras]` (see [Dependency Management](../architecture/dependency_management.md)).
- `test_deps_compat`: fails if `[compat]` entries are missing for dependencies.
- `test_undefined_exports`: fails if an exported symbol is not defined.
- `test_unbound_args`: detects methods with unbound type parameters (can cause ambiguities).
- `test_ambiguities`: detects method ambiguities that could cause dispatch errors.
- `test_piracies`: detects type piracy (defining methods on types you don't own).

Standard pattern across CliMA repos:

```julia
# test/aqua.jl
using Aqua
using MyPackage

@testset "Aqua tests (performance)" begin
    ua = Aqua.detect_unbound_args_recursively(MyPackage)
    @test isempty(ua)

    ambs = Aqua.detect_ambiguities(MyPackage; recursive = true)
    # Filter to ambiguities involving our package only
    pkg_match(pkgname, pkdir::Nothing) = false
    pkg_match(pkgname, pkdir::AbstractString) = occursin(pkgname, pkdir)
    filter!(x -> pkg_match("MyPackage", pkgdir(last(x).module)), ambs)
    @test isempty(ambs)
end

@testset "Aqua tests (additional)" begin
    Aqua.test_all(MyPackage; ambiguities = false, unbound_args = false)
end
```

Note: `test_ambiguities` is separated from `test_all` because most repos filter ambiguities to their own module, excluding upstream dependency ambiguities that are not their responsibility.

## AD compatibility tests

Many CliMA packages include dedicated AD test files (for example, `test/ad_tests.jl` or `test/test_ad_compatibility.jl`). The standard pattern validates ForwardDiff gradients against finite differences:

```julia
using ForwardDiff

function check_derivative(f, x; rtol = 5e-2, atol = 1e-8)
    ad = ForwardDiff.derivative(f, x)
    ε = sqrt(eps(typeof(x)))
    fd = (f(x + ε) - f(x - ε)) / (2ε)
    @test isapprox(ad, fd; rtol, atol)
end
```

When adding new physics functions, add corresponding AD tests. See [AD Compatibility](../performance/ad_compatibility.md).

## GPU test files

Some CliMA packages maintain a separate `test/runtests_gpu.jl` entry point for GPU-specific tests (Thermodynamics.jl, SurfaceFluxes.jl); others dispatch within `runtests.jl` using `ARGS` or via Buildkite CI jobs (ClimaAtmos, ClimaCore, ClimaTimeSteppers). For the standard `ArrayType` selection pattern and `CUDA.allowscalar(false)` setup, see [clima_comms.md §4](clima_comms.md).

## Test isolation: SafeTestsets

Prefer `@safetestset` over `@testset` + nested `include` so variables and imports do not leak between test files:

```julia
using SafeTestsets

@safetestset "Test module A" begin
    @time include("test_module_A.jl")
end
@safetestset "Test module B" begin
    @time include("test_module_B.jl")
end
```

Model repos with many independent test files (ClimaAtmos, ClimaLand, ClimaCoupler, ClimaTimeSteppers) use `@safetestset`. ClimaCore uses a custom `UnitTest` driver (`test/tabulated_tests.jl`) that achieves the same isolation. Physics-library repos (Thermodynamics, CloudMicrophysics, SurfaceFluxes) use plain `@testset`s; if you add a new isolation-sensitive test file there, prefer `@safetestset`. (See also [SDP 22](../code-quality/software_design_patterns.md).)

## Scientifically meaningful tests

Beyond type-stability and allocation checks, CliMA tests should verify that code is **physically and mathematically correct**. When adding or modifying a physics function, include tests from as many of the following categories as applicable.

### 1. Physical and mathematical limits (edge cases)

Test behavior at extreme, degenerate, or boundary inputs. These catch numerical robustness issues that standard-range inputs miss.

```julia
# Zero input returns zero (CloudMicrophysics: terminal velocity)
@test terminal_velocity(rain, vel_type, ρ, FT(0)) ≈ 0 atol = eps(FT)

# Saturation vapor pressure vanishes as T → 0 (Thermodynamics)
for T in (FT(1e-5), eps(FT), FT(0))
    @test saturation_vapor_pressure(param_set, T, Liquid()) ≈ FT(0)
end

# NaN check for near-zero edge cases
@test !isnan(terminal_velocity(snow, vel_type, FT(0.24), FT(3.0f-45)))

# No snow production above freezing
@test conv_q_icl_to_q_sno(..., T_freeze + FT(30)) == FT(0)

# No snow melt below freezing
@test snow_melt(..., T_freeze - FT(2)) ≈ FT(0)
```

### 2. Round-trip (inverse) tests

When a function has a mathematical inverse, verify that composing them recovers the original input. This catches subtle sign errors, off-by-one factors, and incorrect formula transcriptions.

```julia
# Thermodynamics: internal_energy → air_temperature round-trip
e_int = TD.internal_energy(param_set, T0, q_tot, q_liq, q_ice)
T_rec = TD.air_temperature(param_set, TD.ρe(), e_int, q_tot, q_liq, q_ice)
@test T_rec ≈ T0

# Thermodynamics: q_vap_from_RH / relative_humidity round-trip
q_vap = TD.q_vap_from_RH(param_set, p, T, RH, TD.Liquid())
RH_recovered = TD.relative_humidity(param_set, T, p, q_vap)
@test isapprox(RH_recovered, RH; rtol = FT(1e-5))

# SurfaceFluxes: flux solver → profile recovery round-trip
# Solve for fluxes at height z, then recover the profile value at z
U_recovered = SF.compute_profile_value(output, z, SF.Momentum())
@test isapprox(U_recovered, U_input; rtol = FT(0.02))
```

### 3. Tests against analytical results

Where an analytical solution exists (even for simplified cases), test that the numerical implementation reproduces it. This is the strongest form of correctness test.

```julia
# Thermodynamics: ideal gas law (p = R_m ρ T)
p = TD.air_pressure(param_set, T, ρ, q_tot, q_liq, q_ice)
R_m = TD.gas_constant_air(param_set, q_tot, q_liq, q_ice)
@test p ≈ R_m * ρ * T

# Thermodynamics: Clausius-Clapeyron equation (d ln eₛ / dT = L / (Rᵥ T²))
dlog_es_dT_fd = (log(e_sat(T + δ)) - log(e_sat(T - δ))) / (2δ)
dlog_es_dT_cc = L / (R_v * T^2)
@test isapprox(dlog_es_dT_fd, dlog_es_dT_cc; rtol = FT(1e-3))

# ClimaTimeSteppers: forward Euler converges to exp(-t) for du/dt = -u
@test u[1] ≈ exp(-1.0) atol = 0.001

# ClimaTimeSteppers: convergence order matches tableau order
@test convergence_order(prob, sol, LSRK54CarpenterKennedy(), dts) ≈ 4 atol = 0.1

# CloudMicrophysics: accretion rate matches empirical formula (Grabowski 1996)
@test accretion(liquid, rain, vel, ce, q_liq, q_rai, ρ) ≈
      accretion_empir(q_rai, q_liq, q_tot) atol = 0.1 * accretion_empir(...)
```

### 4. Physical consistency tests

Verify thermodynamic identities, conservation laws, and monotonicity constraints that must hold regardless of the specific input values.

```julia
# Thermodynamic identities (Thermodynamics.jl)
@test TD.cp_m(ps, q_tot, q_liq, q_ice) -
      TD.cv_m(ps, q_tot, q_liq, q_ice) ≈ R_m           # cp - cv = R
@test h ≈ e_int + R_m * T                               # h = e + RT
@test TD.latent_heat_sublim(ps, T) ≈
      TD.latent_heat_vapor(ps, T) +
      TD.latent_heat_fusion(ps, T)                       # Lₛ = Lᵥ + L_f

# Mixture specific heats are mass-fraction weighted sums
cp_expected = (1-q_tot)*cp_d + q_vap*cp_v + q_liq*cp_l + q_ice*cp_i
@test TD.cp_m(param_set, q_tot, q_liq, q_ice) ≈ cp_expected

# Monotonicity (CloudMicrophysics: more content → higher velocity)
@test terminal_velocity(rain, vel, ρ, q_rai * 2) > terminal_velocity(rain, vel, ρ, q_rai)
# λ⁻¹ increases with specific content (larger mean particle size)
@test λ_inv_large > λ_inv_small

# Sign constraints (evaporation is negative, melting is positive)
@test evaporation_rate < 0  # subsaturated → evaporation removes mass
@test melt_rate > 0         # T > T_freeze → melting adds liquid

# Saturation partitioning: condensate is nonneg, vapor ≈ saturation
@test q_liq ≥ 0
@test q_ice ≥ 0
@test isapprox(q_vap, q_vap_sat; rtol = FT(1e-6))

# Derivative consistency: analytical vs. finite-difference
Δq = FT(1e-8)
fd_deriv = (rate(q + Δq) - rate(q - Δq)) / (2Δq)
@test sign(analytical_deriv) == sign(fd_deriv)
@test isapprox(analytical_deriv, fd_deriv; rtol = FT(0.2))
```

### General guidelines

- **Dual-precision**: run scientific tests for both `Float32` and `Float64` to catch type-promotion bugs.
- **Tolerances**: use `rtol` for scale-invariant comparisons and `atol` only when a value can be exactly zero. Use `eps(FT)` as a baseline for zero-comparison tolerances.
- **Name the law**: annotate each consistency test with the physical identity it verifies (e.g., "Clausius-Clapeyron", "ideal gas law", "Lₛ = Lᵥ + L_f") so that failures map directly to physics.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
