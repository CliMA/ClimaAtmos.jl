import ClimaAtmos as CA
redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))

# From: https://timholy.github.io/SnoopCompile.jl/stable/snoopr/
using SnoopCompileCore
invalidations = @snoopr begin
    config = CA.AtmosConfig(Dict("output_dir" => "test_invalidations"))
    include(joinpath(dirname(@__DIR__), "examples", "hybrid", "driver.jl"))
    nothing
end;

import SnoopCompile
import PrettyTables # load report_invalidations
SnoopCompile.report_invalidations(;
    invalidations,
    process_filename = x -> last(split(x, "packages/")),
)
