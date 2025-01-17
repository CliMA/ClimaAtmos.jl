jobs = (
    "test/parameterized_tendencies/gravity_wave/non_orographic_gravity_wave/nogw_test_3d.jl",
    "test/parameterized_tendencies/gravity_wave/non_orographic_gravity_wave/nogw_test_mima.jl",
    "test/parameterized_tendencies/gravity_wave/non_orographic_gravity_wave/nogw_test_single_column.jl",
    "test/parameterized_tendencies/gravity_wave/orographic_gravity_wave/ogwd_baseflux.jl",
    "test/parameterized_tendencies/gravity_wave/orographic_gravity_wave/ogwd_3d.jl",
)

for job_file in jobs
    try
        include(job_file)
    catch e
        @error e
    end
end

config_file = "config/model_configs/single_column_nonorographic_gravity_wave.yml"
job_id =  "single_column_nonorographic_gravity_wave"

config = CA.AtmosConfig(config_file; job_id)
try
    include(joinpath(pkgdir(CA), "examples", "hybrid", "driver.jl"))
catch e
    @error e
end
