module Benchmarks

abstract type AbstractBenchmark end

struct Benchmark{FT,N,T,NA,P,D,M,B,I,C} <: AbstractBenchmark
    params::P
    domain::D
    boundary_conditions::B
    initial_conditions::I
    model::M
    callbacks::C
    metrics::NTuple{N,T}
end

function Benchmark3DRisingBubble(
    FT,
    base::AbstractModelStyle,
    thermodynamics::AbstractModelStyle,
    moisture::AbstractModelStyle;
    nelements,
    npolynomial,
    stepper,
    dt,
    callbacks,
    kwargs...
)
    params =
    domain = HybridBox(
        FT,
        xlim = (-5e2, 5e2),
        ylim = (-5e2, 5e2),
        zlim = (0.0, 1e3),
        nelements = nelements,
        npolynomial = npolynomial,
    )
    bcs = nothing
    ics = init_3d_rising_bubble(
        FT,
        params,
        thermo_style = thermodynamics,
        moist_style = moisture,
    )
    model = Nonhydrostatic3DModel(
        domain = domain,
        boundary_conditions = nothing,
        parameters = params,
        base = base,
        thermodynamics = thermodynamics,
        moisture = moisture,
        hyperdiffusivity = FT(100);
        kwargs...
    )
    ics = init_3d_rising_bubble(
        FT,
        params,
        thermo_style = thermodynamics,
        moist_style = moisture,
    )



end
