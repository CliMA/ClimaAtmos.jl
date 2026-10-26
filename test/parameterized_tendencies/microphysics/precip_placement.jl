#=
Unit tests for the opt-in log-linear rain and snow placement of the 1M
conditional-moment rule (`sgs_precip_placement: loglinear`,
`LogLinearNodePlacement`) and the updraft rule of the variance fidelity
(`sgs_variance_fidelity_model: updraft`, `updraft_variance_fidelity`):

  - the log-linear placement conserves the cell-mean rain and snow exactly
    over the points of the rule, for random cells (Float32, Float64),
  - κ = 0, f = 1 and ΔT_m = ∞ reduce it to the uniform placement, and so
    does a cell without rain and snow,
  - the default options give the tendencies of main's evaluator,
  - the warm-side snow depletion ψ is neutral in cold cells and with
    ΔT_m = ∞, and moves the snow off the warm points otherwise,
  - the updraft rule: values, clamps, the fallback, the cloudy-updraft mean,
    and the area blend that leaves inactive (minimum-area) updrafts out,
  - the config keys, the parameter defaults, and no allocations,
  - the model wiring in a Bomex EDMF column (1M, environment branch): both
    options run without allocating; the cache equals the driver with the
    height-dependent rain tilt; `α` follows the updraft rule in the active
    cloudy updrafts and not in minimum-area ones that carry condensate,
  - the grid-mean branch on a Float32 sphere with topography (no EDMF): the
    cache equals the driver with the tilt of the height above the surface,
    without allocating.
=#

using Test
using Random
import ClimaComms
ClimaComms.@import_required_backends
using ClimaAtmos
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import Thermodynamics as TD
import ClimaParams as CP
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
import ClimaCore: Fields

include("../../test_helpers.jl")

# ----------------------------------------------------------------------------
# Main's (c73e290) node evaluator and rule sum, for the default-path check
# ----------------------------------------------------------------------------
struct MainEvaluator{MP, TPS, FT}
    mp::MP
    tps::TPS
    ρ::FT
    w::FT
    q_rai::FT
    q_sno::FT
    λ::FT
    ξ_liq::FT
    ξ_ice::FT
    q_lcl::FT
    q_icl::FT
    λ_lagrange::FT
    λ_eq::FT
    mu_S::FT
    α::FT
    β_precip::FT
    shaft_frac::FT
    dt::FT
    nsubs::Int
end
function (e::MainEvaluator)(T_hat, q_tot_hat, flag_p)
    FT = typeof(e.ρ)
    q_tot_hat = max(FT(0), q_tot_hat)
    q_sat_hat = TD.q_vap_saturation(e.tps, T_hat, e.ρ)
    S′_hat = q_tot_hat - q_sat_hat - e.mu_S
    αS′_hat = e.α * S′_hat
    shifted_excess = max(FT(0), e.λ_lagrange + αS′_hat)
    q_lcl_hat, q_icl_hat = CA.sgs_local_condensate(
        e.λ, shifted_excess, e.ξ_liq, e.ξ_ice, e.q_lcl, e.q_icl,
    )
    q_c_eq_hat = max(FT(0), e.λ_eq + S′_hat)
    q_precip = e.q_rai + e.q_sno
    S_raw_cloud = q_tot_hat - q_precip - q_sat_hat
    S_eff_cloud =
        ifelse(q_c_eq_hat > zero(FT), q_c_eq_hat, min(zero(FT), S_raw_cloud))
    q_tot_eff = max(FT(0), q_sat_hat + S_eff_cloud + q_precip)
    q_rai_node, q_sno_node, q_tot_node =
        if e.β_precip > zero(FT)
            φ_p = CA.sgs_placement_factor(e.β_precip, flag_p, e.shaft_frac)
            shift = (φ_p - one(FT)) * q_precip
            (e.q_rai * φ_p, e.q_sno * φ_p, max(FT(0), q_tot_eff + shift))
        else
            (e.q_rai, e.q_sno, q_tot_eff)
        end
    return BMT.bulk_microphysics_tendencies(
        BMT.LinearizedAverage(), BMT.Microphysics1Moment(), e.mp, e.tps,
        e.ρ, T_hat, e.w, q_tot_node, q_lcl_hat, q_icl_hat, q_rai_node,
        q_sno_node, e.dt, e.nsubs,
    )
end
function main_sum_1m(
    quad, cmp, thp, ρ, T, w, q_lcl, q_icl, q_rai, q_sno, T′T′, q′q′, corr,
    λ_lagrange, α, ξ_liq, ξ_ice, dt, nsubs, λ, mu_S, β_precip, σ_S, a_p,
    floor, λ_eq,
)
    FT = typeof(ρ)
    S_star = CA.sgs_precip_shaft_threshold(max(a_p, floor), σ_S)
    q_c = q_lcl + q_icl
    α_safe = max(α, CA.ϵ_numerics(FT))
    t_p = ifelse(β_precip > zero(FT), S_star, FT(Inf))
    β_T = CA.sgs_temperature_regression(thp, T, ρ, σ_S, T′T′, q′q′, corr)
    rule = CA.conditional_rule(
        thp, ρ, T, mu_S, σ_S, β_T, -λ_lagrange / α_safe, -λ_eq, t_p,
        quad.T_min, quad.q_max,
    )
    g = CA.conditional_rule_mean(CA.NodeCondensate(λ_lagrange, α_safe), rule)
    s = ifelse(
        q_c > zero(FT),
        ifelse(g > zero(FT), q_c / max(g, floatmin(FT)), one(FT)),
        zero(FT),
    )
    shaft_frac = ifelse(
        β_precip > zero(FT), CA.conditional_rule_shaft_fraction(rule), one(FT),
    )
    ev = MainEvaluator(
        cmp, thp, ρ, w, q_rai, q_sno, λ, ξ_liq, ξ_ice, q_lcl, q_icl,
        s * λ_lagrange, λ_eq, mu_S, s * α_safe, β_precip, shaft_frac, dt, nsubs,
    )
    spread = CA.TemperatureSpread(
        thp, ρ, mu_S, CA.sgs_temperature_spread_at_fixed_excess(T′T′, β_T, σ_S),
        FT(TD.Parameters.T_freeze(thp)), quad.T_min, quad.q_max,
    )
    return CA.sum_over_conditional_nodes(ev, rule, spread)
end

# ----------------------------------------------------------------------------
# Cells
# ----------------------------------------------------------------------------
"""
A cell with mean excess `z σ_S`, cloud fraction of the excess `c_frac`.
"""
function make_cell(
    ::Type{FT}, thp, quad; T, ρ = FT(1), T′T′ = FT(0.5), rel_σq = FT(0.08),
    corr = FT(0.4), z = FT(0), c_frac = FT(0.5), q_rai = FT(5e-5),
    q_sno = FT(3e-5), α = FT(0.8),
) where {FT}
    q_sat = TD.q_vap_saturation(thp, T, ρ)
    q′q′ = (rel_σq * q_sat)^2
    σ_S = CA._sgs_linear_sigma_S(thp, ρ, T, T′T′, q′q′, corr)
    mu_S = z * σ_S + q_rai + q_sno
    q_tot = q_sat + mu_S
    q_c = c_frac * CA.truncated_gaussian_condensate(z * σ_S, σ_S)
    λ = TD.liquid_fraction_ramp(thp, T)
    q_lcl = λ * q_c
    q_icl = (1 - λ) * q_c
    mom = CA._compute_sgs_moments(
        thp, ρ, T, q_tot, q_lcl + q_icl, quad, T′T′, q′q′, corr, α,
        q_tot - q_rai - q_sno,
    )
    return (;
        ρ, T, T′T′, q′q′, corr, q_tot, q_lcl, q_icl, q_rai, q_sno, α, mom,
        mu_S, λ = TD.liquid_fraction(thp, T, q_lcl, q_icl),
    )
end

function random_cell(rng, ::Type{FT}, thp, quad) where {FT}
    # half of the cells within a few K of freezing, where the rule splits
    T = rand(rng) < 0.5 ? 270 + 7 * rand(rng) : 250 + 50 * rand(rng)
    return make_cell(
        FT, thp, quad;
        T = FT(T),
        ρ = FT(0.6 + 0.6 * rand(rng)),
        T′T′ = FT(0.05 + 3 * rand(rng)),
        rel_σq = FT(0.01 + 0.2 * rand(rng)),
        corr = FT(-0.6 + 1.5 * rand(rng)) |> c -> clamp(c, FT(-1), FT(1)),
        z = FT(clamp(3 * randn(rng) / 2, -4, 4)),
        c_frac = FT(rand(rng)),
        q_rai = FT(1e-7 + 3e-4 * rand(rng)),
        q_sno = FT(1e-7 + 3e-4 * rand(rng)),
        α = FT(0.2 + 1.3 * rand(rng)),
    )
end

ll_options(::Type{FT}; κ_r = 0.5, κ_s = 0.3, c = 0.3, a_min = 0.1, ΔT_m = 1) where {FT} =
    CA.LogLinearPrecipOptions(;
        rain_tilt = FT(κ_r), rain_tilt_height_slope = FT(0),
        rain_tilt_height_offset = FT(0), snow_tilt = FT(κ_s),
        intermittency_coeff = FT(c), intermittency_min = FT(a_min),
        melt_depletion_width = FT(ΔT_m),
    )

sgs_opts(::Type{FT}; ξ_liq = 0, ξ_ice = 0.5, β_p = 0, floor = 0.5) where {FT} =
    CA.SGSMicrophysicsOptions(;
        ξ_liq = FT(ξ_liq), ξ_ice = FT(ξ_ice),
        precip_incloud_fraction = FT(β_p), precip_frac_floor = FT(floor),
    )

# the packed log-linear driver and the packed default driver of a cell
loglinear_1m(c, quad, cmp, thp, a_p, ll; dt = 180, nsubs = 2, w = 0.1) =
    CA.microphysics_tendencies_1m(
        BMT.Microphysics1Moment(), quad, cmp, thp, c.ρ, c.T, oftype(c.ρ, w),
        c.q_tot, c.q_lcl, c.q_icl, c.q_rai, c.q_sno, c.T′T′, c.q′q′, c.corr,
        c.mom, c.α, oftype(c.ρ, dt), nsubs, c.λ, c.mu_S, a_p, ll.rain_tilt,
        sgs_opts(typeof(c.ρ)), ll,
    )
default_1m(c, quad, cmp, thp, a_p, opts; dt = 180, nsubs = 2, w = 0.1) =
    CA.microphysics_tendencies_1m(
        BMT.Microphysics1Moment(), quad, cmp, thp, c.ρ, c.T, oftype(c.ρ, w),
        c.q_tot, c.q_lcl, c.q_icl, c.q_rai, c.q_sno, c.T′T′, c.q′q′, c.corr,
        c.mom, c.α, oftype(c.ρ, dt), nsubs, c.λ, c.mu_S, a_p, opts,
    )

# the evaluator, rule and spread the log-linear driver sums over
function loglinear_rule(c, quad, cmp, thp, a_p, ll; dt = 180, nsubs = 2)
    FT = typeof(c.ρ)
    σ_S = c.mom.sigma_S
    (pl, a_pb) = CA.loglinear_node_placement(
        thp, c.T, c.q_tot, c.mu_S, σ_S, a_p, ll.rain_tilt, ll,
    )
    return CA._sgs_1m_rule_and_evaluator(
        BMT.Microphysics1Moment(), quad, cmp, thp, c.ρ, c.T, FT(0.1),
        c.q_lcl, c.q_icl, c.q_rai, c.q_sno, c.T′T′, c.q′q′, c.corr,
        c.mom.λ_lagrange, c.α, FT(0), FT(0.5), FT(dt), nsubs, c.λ, c.mu_S,
        FT(0), σ_S, CA.sgs_precip_shaft_threshold(a_pb, σ_S), c.mom.λ_eq, (),
        pl,
    )
end

# rule mean of the placed rain and snow (`f` times the precipitating sub-point)
function placed_means(ev, rule, spread, thp)
    pl = ev.placement
    return CA.sum_over_conditional_nodes(
        (T, q, flag) -> begin
            S′ = max(zero(q), q) - TD.q_vap_saturation(thp, T, ev.ρ) - ev.mu_S
            (q_r, q_s) = CA.loglinear_node_precip(pl, ev.q_rai, ev.q_sno, S′, T)
            (; r = pl.f * q_r, s = pl.f * q_s,
                warm = pl.f * q_s * max(T - pl.T_freeze, zero(T)))
        end,
        rule, spread,
    )
end

function allocs_loglinear(quad, cmp, thp, c, a_p, κ_r, w, dt, opts, ll)
    call() = CA.microphysics_tendencies_1m(
        BMT.Microphysics1Moment(), quad, cmp, thp, c.ρ, c.T, w,
        c.q_tot, c.q_lcl, c.q_icl, c.q_rai, c.q_sno, c.T′T′, c.q′q′, c.corr,
        c.mom, c.α, dt, 2, c.λ, c.mu_S, a_p, κ_r, opts, ll,
    )
    call()
    return @allocated call()
end
function allocs_updraft(α, sgsʲs, ρʲs, a_min, a_full, σ_S, c_u, α_min, q_cloud)
    call() = CA.updraft_variance_fidelity(
        α, sgsʲs, ρʲs, a_min, a_full, σ_S, c_u, α_min, q_cloud,
    )
    call()
    return @allocated call()
end

function allocs_mp_cache(Y, p)
    (; microphysics_model, turbconv_model) = p.atmos
    CA.set_microphysics_tendency_cache!(Y, p, microphysics_model, turbconv_model)
    return @allocated CA.set_microphysics_tendency_cache!(
        Y, p, microphysics_model, turbconv_model,
    )
end

# Bomex EDMF column with 1M microphysics and the given options, with rain and
# snow below 2500 m (above the height offset of the rain tilt), a cloudy
# updraft between 600 and 1500 m, and a minimum-area updraft that carries
# condensate elsewhere (as inactive updrafts do)
function wiring_simulation(placement, fidelity, job_id)
    config = CA.AtmosConfig(
        Dict{String, Any}(
            "initial_condition" => "Bomex",
            "FLOAT_TYPE" => "Float64",
            "config" => "column",
            "turbconv" => "prognostic_edmfx",
            "prognostic_tke" => true,
            "microphysics_model" => "1M",
            "z_max" => 3000.0,
            "z_elem" => 30,
            "z_stretch" => false,
            "dt" => "1secs",
            "t_end" => "10secs",
            "toml" => [joinpath(pkgdir(CA), "toml", "prognostic_edmfx.toml")],
            "output_default_diagnostics" => false,
            "sgs_precip_placement" => placement,
            "sgs_variance_fidelity_model" => fidelity,
        );
        job_id,
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    FT = eltype(Y)
    ᶜz = Fields.coordinate_field(Y.c).z
    @. Y.c.ρtke = FT(0.5) * Y.c.ρ
    @. Y.c.ρq_rai = ifelse(ᶜz < 2500, FT(4e-5), FT(0)) * Y.c.ρ
    @. Y.c.ρq_sno = ifelse(ᶜz < 2500, FT(1e-5), FT(0)) * Y.c.ρ
    sgs = Y.c.sgsʲs.:(1)
    a_min = CAP.min_area(CAP.turbconv_params(p.params))
    band = similar(ᶜz)
    @. band = ifelse((ᶜz > 600) & (ᶜz < 1500), FT(1), FT(0))
    @. sgs.ρa = ifelse(band > 0, FT(0.1), FT(1.5 * a_min)) * Y.c.ρ
    @. sgs.q_lcl = ifelse(band > 0, FT(5e-4), FT(1e-4))
    @. sgs.q_icl = FT(0)
    CA.set_precomputed_quantities!(Y, p, simulation.integrator.t)
    return (;
        Y, p, band = vec(Array(parent(band))) .> 0,
        z = vec(Array(parent(ᶜz))),
    )
end

# the linear σ_S of `materialized_variance_fidelity!` at the environment state
function env_sigma_S(thp, T, p, q_tot, q_liq, q_ice, T′T′, q′q′, corr)
    ρ = TD.air_density(thp, T, p, q_tot, q_liq, q_ice)
    γ = TD.∂q_vap_sat_∂T(thp, T, ρ)
    return sqrt(
        max(q′q′ - 2 * γ * corr * sqrt(max(T′T′ * q′q′, 0)) + γ^2 * T′T′, 0),
    )
end

# the 1M tendencies of the cache's log-linear call with the rain tilt `κ_r`
# (a field or a scalar), on the grid-mean or (with `env`) the environment inputs
# that `set_microphysics_tendency_cache!` leaves in the cache and the scratch
function loglinear_cache_reference(Y, p, κ_r; env)
    pc = p.precomputed
    thp = CAP.thermodynamics_params(p.params)
    cmp = CAP.microphysics_1m_params(p.params)
    quad = p.atmos.sgs_quadrature
    nsubs = p.atmos.microphysics_model.n_substeps_quad
    opts = CA.sgs_microphysics_options(p.params)
    ll = CA.loglinear_precip_options(p.params)
    out = similar(env ? pc.ᶜmp_tendency⁰ : pc.ᶜmp_tendency)
    if env
        s = p.scratch
        @. out = CA.microphysics_tendencies_1m(
            BMT.Microphysics1Moment(), quad, cmp, thp, s.ᶜtemp_scalar, pc.ᶜT⁰,
            CA.w_component(CA.WVec(pc.ᶜu⁰)), pc.ᶜq_tot_nonneg⁰,
            s.ᶜtemp_scalar_2, s.ᶜtemp_scalar_3, s.ᶜtemp_scalar_4,
            s.ᶜtemp_scalar_5, pc.ᶜT′T′, pc.ᶜq′q′, pc.ᶜcorr_Tq, pc.ᶜsgs_moments,
            pc.ᶜα, p.dt, nsubs, s.ᶜtemp_scalar_6, s.ᶜtemp_scalar_7,
            pc.ᶜprecip_frac, κ_r, opts, ll,
        )
    else
        ρ = Y.c.ρ
        q_lcl = @. CA.specific(Y.c.ρq_lcl, ρ)
        q_icl = @. CA.specific(Y.c.ρq_icl, ρ)
        @. out = CA.microphysics_tendencies_1m(
            BMT.Microphysics1Moment(), quad, cmp, thp, ρ, pc.ᶜT,
            CA.w_component(CA.WVec(pc.ᶜu)), pc.ᶜq_tot_nonneg, q_lcl, q_icl,
            CA.specific(Y.c.ρq_rai, ρ), CA.specific(Y.c.ρq_sno, ρ),
            pc.ᶜT′T′, pc.ᶜq′q′, pc.ᶜcorr_Tq, pc.ᶜsgs_moments, pc.ᶜα, p.dt,
            nsubs, TD.liquid_fraction(thp, pc.ᶜT, max(0, q_lcl), max(0, q_icl)),
            pc.ᶜq_tot_nonneg - TD.q_vap_saturation(thp, pc.ᶜT, ρ),
            pc.ᶜprecip_frac, κ_r, opts, ll,
        )
    end
    return Array(parent(out))
end

function close_tendencies(a, b, ::Type{FT}, q_tot; k = 256) where {FT}
    # rounding of the total water enters the vapour excess, which the
    # relaxation (timescales ≥ 10 s) turns into rates
    atol = k * eps(FT) * q_tot / FT(10)
    return all(
        abs(x - y) <= k * eps(FT) * max(abs(x), abs(y)) + atol for
        (x, y) in zip(values(a), values(b))
    )
end

@testset "Log-linear precipitation placement and updraft variance fidelity" begin

    @testset "config and parameters" begin
        @test CA.get_sgs_precip_placement(Dict()) == CA.ShaftPrecipPlacement()
        @test CA.get_sgs_precip_placement(
            Dict("sgs_precip_placement" => "loglinear"),
        ) == CA.LogLinearPrecipPlacement()
        @test_throws ErrorException CA.get_sgs_precip_placement(
            Dict("sgs_precip_placement" => "lognormal"),
        )
        @test CA.get_sgs_variance_fidelity_model(Dict()) ==
              CA.EquilibriumVarianceFidelity()
        @test CA.get_sgs_variance_fidelity_model(
            Dict("sgs_variance_fidelity_model" => "updraft"),
        ) == CA.UpdraftVarianceFidelity()
        defaults = CA.default_config_dict()
        @test CA.get_sgs_precip_placement(defaults) == CA.ShaftPrecipPlacement()
        @test CA.get_sgs_variance_fidelity_model(defaults) ==
              CA.EquilibriumVarianceFidelity()
        water = CA.AtmosWater()
        @test water.sgs_precip_placement == CA.ShaftPrecipPlacement()
        @test water.sgs_variance_fidelity_model == CA.EquilibriumVarianceFidelity()
        for FT in (Float32, Float64)
            params = CA.ClimaAtmosParameters(FT)
            @test CAP.sgs_precip_rain_tilt(params) == FT(0.5)
            @test CAP.sgs_precip_rain_tilt_height_slope(params) == FT(5e-4)
            @test CAP.sgs_precip_rain_tilt_height_offset(params) == FT(1500)
            @test CAP.sgs_precip_snow_tilt(params) == FT(0.3)
            @test CAP.sgs_precip_intermittency_coeff(params) == FT(0.3)
            @test CAP.sgs_precip_intermittency_min(params) == FT(0.1)
            @test CAP.sgs_snow_melt_depletion_width(params) == FT(1)
            @test CAP.sgs_fidelity_updraft_coeff(params) == FT(0.6)
            @test CAP.sgs_fidelity_updraft_min(params) == FT(0.05)
            @test CAP.sgs_fidelity_updraft_full_area(params) == FT(1e-3)
            ll = CA.loglinear_precip_options(params)
            @test ll isa CA.LogLinearPrecipOptions{FT}
            # κ_r = κ_r0 below the offset, rising linearly above it
            @test CA.loglinear_rain_tilt(ll, FT(300)) == FT(0.5)
            @test CA.loglinear_rain_tilt(ll, FT(1500)) == FT(0.5)
            @test CA.loglinear_rain_tilt(ll, FT(5200)) ≈ FT(2.35)
            # the width may be infinite; c, a_min and ΔT_m must be positive
            sq_inf = CA.SGSQuadratureParameters(
                FT, (; sgs_snow_melt_depletion_width = FT(Inf)),
            )
            @test CA.loglinear_precip_options(sq_inf).melt_depletion_width == Inf
            for bad in (
                (; sgs_snow_melt_depletion_width = FT(0)),
                (; sgs_precip_intermittency_coeff = FT(0)),
                (; sgs_precip_intermittency_min = FT(0)),
            )
                @test_throws ErrorException CA.loglinear_precip_options(
                    CA.SGSQuadratureParameters(FT, bad),
                )
            end
        end
    end

    @testset "exact conservation over the rule (random cells)" begin
        rng = MersenneTwister(4925)
        for FT in (Float32, Float64)
            toml_dict = CP.create_toml_dict(FT)
            thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
            cmp = CMP.Microphysics1MParams(toml_dict)
            quad = CA.SGSQuadrature(FT; distribution = CA.GaussianSGS(), q_max = FT(0.1))
            tol = 64 * eps(FT)
            n_split = 0
            for _ in 1:200
                c = random_cell(rng, FT, thp, quad)
                a_p = FT(rand(rng))
                ll = ll_options(
                    FT;
                    κ_r = 3 * rand(rng), κ_s = rand(rng),
                    c = rand(rng, (0.1, 0.3, 1, 20)),
                    ΔT_m = rand(rng, (0.5, 1, 2, Inf)),
                )
                (ev, rule, spread) = loglinear_rule(c, quad, cmp, thp, a_p, ll)
                pl = ev.placement
                @test pl isa CA.LogLinearNodePlacement{FT}
                @test 0 < pl.f <= 1
                @test pl.f == min(1, ll.intermittency_coeff * max(a_p, FT(0.1)))
                m = placed_means(ev, rule, spread, thp)
                @test m.r ≈ c.q_rai rtol = tol
                @test m.s ≈ c.q_sno rtol = tol
                # the rule's freezing split is exercised
                n_split += any(
                    n ->
                        n.w > 0 &&
                        CA._freezing_split(
                            n.T, spread.σ⊥, spread.T_freeze,
                        )[3] > 0,
                    rule.nodes,
                )
                out = loglinear_1m(c, quad, cmp, thp, a_p, ll)
                @test all(isfinite, values(out))
            end
            @test n_split > 10
        end
    end

    @testset "uniform limit: κ = 0, f = 1, ΔT_m = ∞" begin
        for FT in (Float32, Float64)
            toml_dict = CP.create_toml_dict(FT)
            thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
            cmp = CMP.Microphysics1MParams(toml_dict)
            quad = CA.SGSQuadrature(FT; distribution = CA.GaussianSGS(), q_max = FT(0.1))
            ll = ll_options(FT; κ_r = 0, κ_s = 0, c = 1e6, ΔT_m = Inf)
            for T in FT.((290, 273.6, 262)), z in FT.((-2, -0.3, 0.8))
                c = make_cell(FT, thp, quad; T, z, T′T′ = FT(1.5))
                for a_p in FT.((0, 0.4))
                    ref = default_1m(c, quad, cmp, thp, a_p, sgs_opts(FT))
                    out = loglinear_1m(c, quad, cmp, thp, a_p, ll)
                    @test close_tendencies(out, ref, FT, c.q_tot)
                    # κ = 0 adds no edge: the rule of the uniform placement
                    (ev, rule, _) = loglinear_rule(c, quad, cmp, thp, a_p, ll)
                    @test ev.placement.f == 1
                    @test CA.precip_placement_edge(ev.placement, FT(0), FT(1)) == Inf
                end
            end
        end
    end

    @testset "cells without rain and snow: the uniform rule" begin
        rng = MersenneTwister(4926)
        for FT in (Float32, Float64)
            toml_dict = CP.create_toml_dict(FT)
            thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
            cmp = CMP.Microphysics1MParams(toml_dict)
            quad = CA.SGSQuadrature(FT; distribution = CA.GaussianSGS(), q_max = FT(0.1))
            for _ in 1:20
                c = random_cell(rng, FT, thp, quad)
                c = merge(c, (; q_rai = FT(0), q_sno = FT(0)))
                a_p = FT(rand(rng))
                ref = default_1m(c, quad, cmp, thp, a_p, sgs_opts(FT))
                out = loglinear_1m(c, quad, cmp, thp, a_p, ll_options(FT; κ_r = 1.2))
                @test out == ref
            end
        end
    end

    @testset "intermittency concentrates the rain: less evaporation" begin
        for FT in (Float32, Float64)
            toml_dict = CP.create_toml_dict(FT)
            thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
            cmp = CMP.Microphysics1MParams(toml_dict)
            quad = CA.SGSQuadrature(FT; distribution = CA.GaussianSGS(), q_max = FT(0.1))
            # subsaturated, cloud-free, warm: the rain only evaporates
            c = make_cell(
                FT, thp, quad; T = FT(290), z = FT(-2), c_frac = FT(0),
                q_rai = FT(1e-4), q_sno = FT(0),
            )
            rain(κ_r, c_f) = loglinear_1m(
                c, quad, cmp, thp, FT(0.5),
                ll_options(FT; κ_r, κ_s = 0, c = c_f, ΔT_m = Inf); dt = 20,
            ).dq_rai_dt
            full = rain(0, 1e6)
            @test full < rain(0, 0.3) < 0
            # tilting the rain toward the moist side also evaporates less
            @test full < rain(1, 1e6) < 0
        end
    end

    @testset "default options: main's tendencies" begin
        for FT in (Float32, Float64)
            toml_dict = CP.create_toml_dict(FT)
            thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
            cmp = CMP.Microphysics1MParams(toml_dict)
            quad = CA.SGSQuadrature(FT; distribution = CA.GaussianSGS(), q_max = FT(0.1))
            for T in FT.((291, 273.4, 258)), z in FT.((-1.5, 0.4)),
                (β_p, floor) in ((0, 0.5), (1, 0.1), (0.5, 0.5))

                c = make_cell(FT, thp, quad; T, z, T′T′ = FT(1.2))
                a_p = FT(0.3)
                ξ_l, ξ_i = FT(0), FT(0.5)
                ref = main_sum_1m(
                    quad, cmp, thp, c.ρ, c.T, FT(0.1), c.q_lcl, c.q_icl,
                    c.q_rai, c.q_sno, c.T′T′, c.q′q′, c.corr, c.mom.λ_lagrange,
                    c.α, ξ_l, ξ_i, FT(180), 2, c.λ, c.mu_S, FT(β_p),
                    c.mom.sigma_S, a_p, FT(floor), c.mom.λ_eq,
                )
                out = default_1m(
                    c, quad, cmp, thp, a_p,
                    sgs_opts(FT; ξ_liq = ξ_l, ξ_ice = ξ_i, β_p, floor),
                )
                @test out === ref
            end
        end
    end

    @testset "warm-side snow depletion ψ" begin
        for FT in (Float32, Float64)
            toml_dict = CP.create_toml_dict(FT)
            thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
            cmp = CMP.Microphysics1MParams(toml_dict)
            quad = CA.SGSQuadrature(FT; distribution = CA.GaussianSGS(), q_max = FT(0.1))
            T_f = FT(TD.Parameters.T_freeze(thp))
            # pointwise: ψ is relative to the cell mean, neutral below freezing
            # and with ΔT_m = ∞, and falls by e per ΔT_m of warmth
            pl(inv_ΔT, T_mean) = CA.LogLinearNodePlacement(
                FT(0.5), FT(0.3), FT(inv_ΔT), T_f, max(T_mean, T_f), FT(1e-4),
                FT(1), FT(1), FT(1),
            )
            S′ = FT(5e-5)
            g_cold = CA.loglinear_placement_weights(pl(1, T_f - 3), S′, T_f - 1)
            @test g_cold === CA.loglinear_placement_weights(pl(0, T_f - 3), S′, T_f - 1)
            @test g_cold[2] == exp(FT(0.3) * (S′ / FT(1e-4)))
            g1 = CA.loglinear_placement_weights(pl(1, T_f - 3), S′, T_f + 1)
            g2 = CA.loglinear_placement_weights(pl(1, T_f - 3), S′, T_f + 2)
            @test g1[1] == g2[1] == g_cold[1]
            @test g2[2] / g1[2] ≈ exp(-FT(1)) rtol = 10 * eps(FT)
            @test CA.loglinear_placement_weights(pl(0, T_f + 2), S′, T_f + 5) ===
                  CA.loglinear_placement_weights(pl(0, T_f + 2), S′, T_f - 5)

            # cold cell, no point above freezing: ΔT_m changes nothing
            c = make_cell(FT, thp, quad; T = T_f - 12, z = FT(0.3), T′T′ = FT(0.3))
            out_1 = loglinear_1m(c, quad, cmp, thp, FT(0.4), ll_options(FT; ΔT_m = 1))
            out_∞ = loglinear_1m(c, quad, cmp, thp, FT(0.4), ll_options(FT; ΔT_m = Inf))
            @test out_1 === out_∞

            # melting cell: the snow leaves the warm points, its mean is kept
            c = make_cell(FT, thp, quad; T = T_f + FT(0.8), z = FT(0.2), T′T′ = FT(4))
            ll_1 = ll_options(FT; ΔT_m = 1)
            ll_∞ = ll_options(FT; ΔT_m = Inf)
            (ev1, rule1, spread1) = loglinear_rule(c, quad, cmp, thp, FT(0.4), ll_1)
            (ev∞, rule∞, spread∞) = loglinear_rule(c, quad, cmp, thp, FT(0.4), ll_∞)
            m1 = placed_means(ev1, rule1, spread1, thp)
            m∞ = placed_means(ev∞, rule∞, spread∞, thp)
            @test m1.s ≈ c.q_sno rtol = 64 * eps(FT)
            @test m∞.s ≈ c.q_sno rtol = 64 * eps(FT)
            @test m1.r == m∞.r
            @test m1.warm < FT(0.9) * m∞.warm
            out_1 = loglinear_1m(c, quad, cmp, thp, FT(0.4), ll_1)
            out_∞ = loglinear_1m(c, quad, cmp, thp, FT(0.4), ll_∞)
            @test out_1.dq_sno_dt != out_∞.dq_sno_dt
        end
    end

    @testset "updraft rule of the variance fidelity" begin
        for FT in (Float32, Float64)
            c_u, α_min, a_min, q_cloud = FT(0.6), FT(0.05), FT(1e-5), FT(1e-5)
            a_full = FT(1e-3)
            α_eq = FT(1.3)
            up(ρa, q_lcl, q_icl) = (; ρa = FT(ρa), q_lcl = FT(q_lcl), q_icl = FT(q_icl))
            ρʲ = FT(1.1)
            rule(sgsʲs, ρʲs, σ_S; a_full = a_full) = CA.updraft_variance_fidelity(
                α_eq, sgsʲs, ρʲs, a_min, a_full, FT(σ_S), c_u, α_min, q_cloud,
            )
            # one updraft: α σ_S = c_u q_c,up
            @test rule((up(0.1ρʲ, 1e-4, 2e-5),), (ρʲ,), 3e-4) ≈ FT(0.24)
            @test rule((up(0.1ρʲ, 1e-4, 2e-5),), (ρʲ,), 3e-4) isa FT
            # clamps: α_min for little condensate, 2 for a narrow PDF
            @test rule((up(0.1ρʲ, 2e-5, 0),), (ρʲ,), 3e-4) == α_min
            @test rule((up(0.1ρʲ, 1e-3, 0),), (ρʲ,), 1e-4) == FT(2)
            @test rule((up(0.1ρʲ, 1e-3, 0),), (ρʲ,), 0) == FT(2)
            # fallback to the equilibrium rule: no updraft above the minimum
            # area, or no cloud (q_c ≤ 1e-5) in it
            @test rule((up(0.5 * a_min * ρʲ, 1e-3, 0),), (ρʲ,), 3e-4) == α_eq
            @test rule((up(0.1ρʲ, 0, 0),), (ρʲ,), 3e-4) == α_eq
            @test rule((up(0.1ρʲ, -1e-6, 0),), (ρʲ,), 3e-4) == α_eq
            @test rule((up(0.1ρʲ, 5e-6, 4e-6),), (ρʲ,), 3e-4) == α_eq
            # two updrafts: area-weighted, small ones excluded
            ρs = (ρʲ, FT(0.9))
            two = (up(0.1ρʲ, 1e-4, 0), up(0.3 * FT(0.9), 2e-4, 0))
            @test rule(two, ρs, 3e-4) ≈ c_u * FT(1.75e-4) / FT(3e-4)
            two_small = (up(0.1ρʲ, 1e-4, 0), up(0.5 * a_min * FT(0.9), 2e-4, 0))
            @test rule(two_small, ρs, 3e-4) ≈ c_u * FT(1e-4) / FT(3e-4)
            # only the updrafts that carry cloud enter the mean
            dry = (up(0.05ρʲ, 1e-3, 0), up(0.05 * FT(0.9), 0, 0))
            @test rule(dry, ρs, 5e-4) ≈ FT(1.2)
            # an updraft held near the minimum area by the area limiters
            # barely moves α, whatever it carries; the weight of the rule
            # grows linearly to 1 at `a_full`
            α_up = c_u * FT(3e-4) / FT(2e-4)
            ghost = rule((up(1.5 * a_min * ρʲ, 3e-4, 0),), (ρʲ,), 2e-4)
            W = (FT(1.5) * a_min - a_min) / (a_full - a_min)
            @test ghost ≈ (1 - W) * α_eq + W * α_up rtol = 100 * eps(FT)
            @test abs(ghost - α_eq) < FT(0.01) * abs(α_up - α_eq)
            a_mid = (a_min + a_full) / 2
            @test rule((up(a_mid * ρʲ, 3e-4, 0),), (ρʲ,), 2e-4) ≈
                  (α_eq + α_up) / 2 rtol = 100 * eps(FT)
            @test rule((up(a_full * ρʲ, 3e-4, 0),), (ρʲ,), 2e-4) ≈ α_up
            # `a_full = a_min`: the rule wherever an updraft exceeds a_min
            @test rule((up(1.5 * a_min * ρʲ, 3e-4, 0),), (ρʲ,), 2e-4; a_full = a_min) ≈
                  α_up rtol = 10 * eps(FT)
        end
    end

    @testset "model wiring (Bomex EDMF column, 1M)" begin
        def = wiring_simulation("shaft", "equilibrium", "precip_placement_default")
        opt = wiring_simulation("loglinear", "updraft", "precip_placement_opt")
        @test def.p.atmos.sgs_precip_placement isa CA.ShaftPrecipPlacement
        @test opt.p.atmos.sgs_precip_placement isa CA.LogLinearPrecipPlacement
        @test opt.p.atmos.sgs_variance_fidelity_model isa CA.UpdraftVarianceFidelity
        @test allocs_mp_cache(opt.Y, opt.p) <= allocs_mp_cache(def.Y, def.p)
        mp_def = Array(parent(def.p.precomputed.ᶜmp_tendency⁰))
        mp_opt = Array(parent(opt.p.precomputed.ᶜmp_tendency⁰))
        @test all(isfinite, mp_opt)
        @test mp_opt != mp_def
        # the environment call is the driver with the rain tilt of the height
        # above the (flat) surface, which rises above the offset of 1500 m
        (; microphysics_model, turbconv_model) = opt.p.atmos
        CA.set_microphysics_tendency_cache!(
            opt.Y, opt.p, microphysics_model, turbconv_model,
        )
        ll = CA.loglinear_precip_options(opt.p.params)
        ᶜz = Fields.coordinate_field(opt.Y.c).z
        ᶜκ_r = @. ll.rain_tilt +
           ll.rain_tilt_height_slope * max(0, ᶜz - ll.rain_tilt_height_offset)
        @test maximum(Array(parent(ᶜκ_r))) > ll.rain_tilt + 0.4
        ref = loglinear_cache_reference(opt.Y, opt.p, ᶜκ_r; env = true)
        @test ref ≈ mp_opt rtol = 1e-12
        # (sensitivity: without the height term the rates differ)
        ref_κ0 = loglinear_cache_reference(opt.Y, opt.p, ll.rain_tilt; env = true)
        @test !isapprox(ref_κ0, mp_opt; rtol = 1e-6)
        # α: the updraft rule with the updraft liquid (5e-4) in the
        # cloudy-updraft band; the equilibrium rule away from it (next to it
        # the band's cloud fraction feeds back on the variances), where the
        # minimum-area updraft carries 1e-4 that would otherwise set α
        α_def = vec(Array(parent(def.p.precomputed.ᶜα)))
        α_opt = vec(Array(parent(opt.p.precomputed.ᶜα)))
        band = opt.band
        @test count(band) >= 5
        pc = opt.p.precomputed
        thp = CAP.thermodynamics_params(opt.p.params)
        σ_S = vec(
            Array(
                parent(
                    env_sigma_S.(
                        thp, pc.ᶜT⁰, pc.ᶜp, pc.ᶜq_tot_nonneg⁰, pc.ᶜq_liq⁰, pc.ᶜq_ice⁰,
                        pc.ᶜT′T′, pc.ᶜq′q′, pc.ᶜcorr_Tq,
                    ),
                ),
            ),
        )
        c_u = CAP.sgs_fidelity_updraft_coeff(opt.p.params)
        α_up = clamp.(c_u * 5e-4 ./ max.(σ_S, CA.ϵ_numerics(Float64)), 0.05, 2)
        @test α_opt[band] ≈ α_up[band] rtol = 1e-10
        @test α_opt[band] != α_def[band]
        far = (opt.z .< 400) .| (opt.z .> 1800)
        @test count(far) >= 10
        @test maximum(abs.(α_opt[far] .- α_def[far])) < 2e-2
        α_ghost = clamp.(c_u * 1e-4 ./ max.(σ_S, CA.ϵ_numerics(Float64)), 0.05, 2)
        @test maximum(abs.(α_ghost[far] .- α_def[far])) > 0.1
    end

    @testset "grid-mean branch: Float32 sphere with topography" begin
        config = CA.AtmosConfig(
            Dict{String, Any}(
                "config" => "sphere",
                "FLOAT_TYPE" => "Float32",
                "h_elem" => 2,
                "z_elem" => 10,
                "z_max" => 30000.0,
                "topography" => "DCMIP200",
                "initial_condition" => "MoistBaroclinicWave",
                "microphysics_model" => "1M",
                "dt" => "100secs",
                "t_end" => "200secs",
                "output_default_diagnostics" => false,
                "sgs_precip_placement" => "loglinear",
                "sgs_variance_fidelity_model" => "updraft",
            );
            job_id = "precip_placement_sphere",
        )
        (; Y, p, simulation) = generate_test_simulation(config)
        FT = eltype(Y)
        @test FT == Float32
        @test !(p.atmos.turbconv_model isa CA.PrognosticEDMFX)
        ᶜz = Fields.coordinate_field(Y.c).z
        z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
        @test maximum(Array(parent(z_sfc))) > 1000
        # rain and snow through the lowest 8 km above the surface
        @. Y.c.ρq_rai = ifelse(ᶜz - z_sfc < 8000, FT(5e-5), FT(0)) * Y.c.ρ
        @. Y.c.ρq_sno = ifelse(ᶜz - z_sfc < 8000, FT(2e-5), FT(0)) * Y.c.ρ
        CA.set_precomputed_quantities!(Y, p, simulation.integrator.t)
        @test all(isfinite, Array(parent(p.precomputed.ᶜα)))
        allocs = allocs_mp_cache(Y, p)
        @info "sphere log-linear microphysics cache allocations" allocs
        @test allocs == 0
        mp = Array(parent(p.precomputed.ᶜmp_tendency))
        @test all(isfinite, mp)
        ll = CA.loglinear_precip_options(p.params)
        κ_r(z_agl) =
            ll.rain_tilt +
            ll.rain_tilt_height_slope * max(0, z_agl - ll.rain_tilt_height_offset)
        ᶜκ_r = @. κ_r(ᶜz - z_sfc)
        ref = loglinear_cache_reference(Y, p, ᶜκ_r; env = false)
        @test ref ≈ mp rtol = 1e-6
        # (sensitivity: the tilt of the height above sea level differs)
        ref_msl = loglinear_cache_reference(Y, p, (@. κ_r(ᶜz)); env = false)
        @test !isapprox(ref_msl, mp; rtol = 1e-4)
    end

    @testset "no allocations" begin
        FT = Float64
        toml_dict = CP.create_toml_dict(FT)
        thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
        cmp = CMP.Microphysics1MParams(toml_dict)
        quad = CA.SGSQuadrature(FT; distribution = CA.GaussianSGS(), q_max = FT(0.1))
        c = make_cell(FT, thp, quad; T = FT(273.9), z = FT(-0.4), T′T′ = FT(2))
        @test allocs_loglinear(
            quad, cmp, thp, c, FT(0.4), FT(0.7), FT(0.1), FT(180),
            sgs_opts(FT), ll_options(FT),
        ) == 0
        sgsʲs = (
            (; ρa = FT(0.1), q_lcl = FT(1e-4), q_icl = FT(0)),
            (; ρa = FT(0.2), q_lcl = FT(2e-4), q_icl = FT(1e-5)),
        )
        @test allocs_updraft(
            FT(1), sgsʲs, (FT(1), FT(1)), FT(1e-5), FT(1e-3), FT(3e-4), FT(0.6),
            FT(0.05), FT(1e-5),
        ) == 0
    end
end
