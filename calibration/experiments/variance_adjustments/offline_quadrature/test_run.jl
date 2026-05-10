using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "scripts", "run_end_to_end.jl"))

experiment_dir = joinpath(@__DIR__, "..")
outdir = joinpath(@__DIR__, "outputs", "test_verification")

# Run just one simple case (trmm)
outputs = run_offline_quadrature_end_to_end!(;
    outdir = outdir,
    plot_include_cases = ["trmm_column_varquad_diagnostic_edmfx"],
    sgs_distributions = ["lognormal", "lognormal_vertical_profile_full_cubature"],
    regrid_methods = [:linear],
    quadrature_orders = [1],
    plot_after = false,
    parallel = :sequential
)

println("Test OK: ", keys(outputs))
