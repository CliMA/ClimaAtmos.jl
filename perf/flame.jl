ca_dir = joinpath(dirname(@__DIR__));
include(joinpath(ca_dir, "examples", "hybrid", "cli_options.jl"));

ENV["CI_PERF_SKIP_RUN"] = true # we only need haskey(ENV, "CI_PERF_SKIP_RUN") == true

filename = joinpath(ca_dir, "examples", "hybrid", "driver.jl")
dict = parsed_args_per_job_id(; trigger = "benchmark.jl")
(s, parsed_args_prescribed) = parse_commandline()
parsed_args_target = dict["perf_target_unthreaded"];
parsed_args = merge(parsed_args_target, parsed_args_prescribed);


try # capture integrator
    include(filename)
catch err
    if err.error !== :exit_profile
        rethrow(err.error)
    end
end

function do_work!(integrator)
    for _ in 1:20
        OrdinaryDiffEq.step!(integrator)
    end
end

do_work!(integrator) # compile first
import Profile
Profile.clear_malloc_data()
Profile.clear()
prof = Profile.@profile begin
    do_work!(integrator)
end

if !isempty(get(ENV, "CI_PERF_CPUPROFILE", ""))

    (; output_dir, job_id) = simulation
    import ChromeProfileFormat
    output_path = output_dir
    cpufile = job_id * ".cpuprofile"
    ChromeProfileFormat.save_cpuprofile(joinpath(output_path, cpufile))

    if !isempty(get(ENV, "BUILDKITE", ""))
        import URIs

        print_link_url(url) = print("\033]1339;url='$(url)'\a\n")

        profiler_url(uri) = URIs.URI(
            "https://profiler.firefox.com/from-url/$(URIs.escapeuri(uri))",
        )

        # copy the file to the clima-ci bucket
        buildkite_pipeline = ENV["BUILDKITE_PIPELINE_SLUG"]
        buildkite_buildnum = ENV["BUILDKITE_BUILD_NUMBER"]
        buildkite_step = ENV["BUILDKITE_STEP_KEY"]

        profile_uri = "$buildkite_pipeline/build/$buildkite_buildnum/$buildkite_step/$cpufile"
        gs_profile_uri = "gs://clima-ci/$profile_uri"
        dl_profile_uri = "https://storage.googleapis.com/clima-ci/$profile_uri"

        # sync to bucket
        run(`gsutil cp $(joinpath(output_path, cpufile)) $gs_profile_uri`)

        # print link
        println("+++ Profiler link for '$profile_uri': ")
        print_link_url(profiler_url(dl_profile_uri))
    end
else
    import PProf
    PProf.pprof()
    # http://localhost:57599/ui/flamegraph?tf
end
