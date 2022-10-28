# From: https://timholy.github.io/SnoopCompile.jl/stable/snoopr/
using SnoopCompileCore
invalidations = @snoopr begin
    include(joinpath(dirname(@__DIR__), "examples", "hybrid", "driver.jl"))
    nothing
end;

import ReportMetrics
ReportMetrics.report_invalidations(;
    job_name = "invalidations",
    invalidations,
    process_filename = x -> last(split(x, "packages/")),
)
