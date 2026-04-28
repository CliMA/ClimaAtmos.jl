# Single place to resolve `analysis/plotting` → experiment root (avoids repeated `const` path blocks in every file).
"""
    va_variance_adjustments_plot_paths() -> (; plotting, analysis, experiment, figures)

`plotting` is the directory containing this file (`.../analysis/plotting`). Call when
you need paths (cheap; no global state).
"""
function va_variance_adjustments_plot_paths()
    plotting = string(@__DIR__)
    analysis = abspath(joinpath(plotting, ".."))
    experiment = abspath(joinpath(analysis, ".."))
    figures = joinpath(analysis, "figures")
    return (; plotting, analysis, experiment, figures)
end
