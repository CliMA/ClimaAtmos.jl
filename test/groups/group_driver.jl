import ClimaAtmos as CA

for (config_file, job_id) in jobs
    config = CA.AtmosConfig(config_file; job_id)
    try
        include(joinpath(pkgdir(CA), "examples", "hybrid", "driver.jl"))
    catch e
        @error e
    end
end
