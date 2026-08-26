"""
The prior for the SOCRATES microphysics timescales.

Widths are given in the **unconstrained** space: the physical mean is mapped through the constraint and
a `Normal(uncons_μ, unconstrained_σ)` placed there. For a timescale bounded over eight decades the
unconstrained coordinate is logarithmic, so a σ of 3–4 spans orders of magnitude.

A width in physical units cannot express this: `constrained_gaussian` has no distribution with a 2 s
spread about a 1000 s mean inside eight-decade bounds, and returns one whose constrained mean is ~4e-12 s.
"""

using EnsembleKalmanProcesses: EnsembleKalmanProcesses as EKP

"""
The calibrated parameters: `name => (mean, lower, upper, unconstrained_σ)`, with `mean` and the bounds
in seconds.

These are the non-equilibrium 1-moment relaxation timescales. Condensation/evaporation and
sublimation/deposition set how fast vapour converts to cloud liquid and cloud ice; the two
autoconversion timescales set how fast cloud converts to rain and snow. Together they control the
mixed-phase partitioning the Atlas LES comparison is sensitive to.
"""
const DEFAULT_PRIOR_SPEC = (
    condensation_evaporation_timescale = (1.0e2, 1e0, 1.0e8, 3.0),
    sublimation_deposition_timescale = (1.0e3, 1e1, 1.0e8, 3.0),
    rain_autoconversion_timescale = (5.0e4, 1e1, 1.0e8, 3.0),
    snow_autoconversion_timescale = (1.0e3, 1e1, 1.0e8, 3.0),

)

"""
    default_prior(spec = DEFAULT_PRIOR_SPEC)

The combined `ParameterDistribution` over the calibrated parameters, which is what
`EKP.construct_initial_ensemble` consumes.
"""
function default_prior(spec = DEFAULT_PRIOR_SPEC)
    isempty(spec) && error("The prior needs at least one parameter")
    distributions = map(collect(pairs(spec))) do (name, (mean, lower, upper, σ))
        lower <= mean <= upper || error(
            "Prior mean $mean for `$name` is outside its bounds ($lower, $upper).",
        )
        σ > 0 || error("Prior `unconstrained_σ` for `$name` must be positive, got $σ")
        constraint = EKP.ParameterDistributions.bounded(lower, upper)
        return EKP.ParameterDistributions.ParameterDistribution(
            EKP.ParameterDistributions.Parameterized(
                EKP.ParameterDistributions.Normal(
                    constraint.constrained_to_unconstrained(mean),
                    σ,
                ),
            ),
            constraint,
            String(name),
        )
    end
    return EKP.ParameterDistributions.combine_distributions(distributions)
end

"""Names of the calibrated parameters, in prior order."""
prior_names(spec = DEFAULT_PRIOR_SPEC) = collect(String.(keys(spec)))
