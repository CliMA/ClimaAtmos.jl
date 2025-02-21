redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))

# From: https://timholy.github.io/SnoopCompile.jl/dev/tutorials/invalidations/#Tutorial-on-@snoop_invalidations
using SnoopCompileCore
invalidations = @snoop_invalidations begin
    include(joinpath(dirname(@__DIR__), ".buildkite", "ci_deps.jl"))
    import ClimaComms
    ClimaComms.@import_required_backends
    import ClimaAtmos as CA
    (; config_file, job_id) = CA.commandline_kwargs()
    config = CA.AtmosConfig(config_file; job_id)
    include(joinpath(dirname(@__DIR__), ".buildkite", "ci_driver.jl"))
    nothing
end;

import SnoopCompile
import PrettyTables # load report_invalidations
SnoopCompile.report_invalidations(;
    invalidations,
    process_filename = x -> last(split(x, "packages/")),
)
