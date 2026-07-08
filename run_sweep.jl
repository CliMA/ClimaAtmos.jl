import CUDA
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import Random
using YAML
using NCDatasets
using Statistics
using Printf

Random.seed!(1234)

# Create TOML files for A values with type = "float"
A_values = [0.0, 0.2, 0.4, 0.6, 0.8]
for A in A_values
    toml_path = joinpath("toml", "prognostic_edmfx_A$(A).toml")
    open(toml_path, "w") do io
        println(io, "[EDMF_interface_entr_efficiency]")
        println(io, "value = $(A)")
        println(io, "type = \"float\"")
    end
end

cases = ["DYCOMS_RF01", "Rico"]
dz_values = [50, 100, 200]

for case in cases
    for A in A_values
        for dz in dz_values
            job_id = "$(case)_dz$(dz)_A$(A)"
            @info "=== Running $(job_id) ==="

            if case == "DYCOMS_RF01"
                base_config = YAML.load_file(
                    "config/model_configs/prognostic_edmfx_dycoms_rf01_column.yml",
                )
                base_config["device"] = "CUDADevice"
                base_config["z_max"] = 1600
                z_elem = div(1600, dz)
                base_config["z_elem"] = z_elem
                base_config["netcdf_interpolation_num_points"] = [2, 2, z_elem]
                base_config["toml"] =
                    ["toml/prognostic_edmfx.toml", "toml/prognostic_edmfx_A$(A).toml"]
            else # Rico
                base_config =
                    YAML.load_file("config/model_configs/prognostic_edmfx_rico_column.yml")
                base_config["device"] = "CUDADevice"
                base_config["z_max"] = 4000
                base_config["t_end"] = "86300secs" # < 86400 to get hourly averages
                z_elem = div(4000, dz)
                base_config["z_elem"] = z_elem
                base_config["netcdf_interpolation_num_points"] = [8, 8, z_elem]
                base_config["toml"] =
                    ["toml/prognostic_edmfx_1M.toml", "toml/prognostic_edmfx_A$(A).toml"]
            end

            try
                config = CA.AtmosConfig(base_config; job_id = job_id)
                simulation = CA.get_simulation(config)
                sol_res = CA.solve_atmos!(simulation)
                @info "=== Finished $(job_id) successfully ==="
            catch e
                @error "=== Failed $(job_id) ===" exception = (e, catch_backtrace())
            end
        end
    end
end

@info "=== All simulations completed. Analyzing results ==="

for case in cases
    println("\n====================================================================")
    println("SUMMARY FOR CASE: $(case)")
    println("====================================================================")
    for A in A_values
        println("\n--- Entrainment Parameter A = $(A) ---")
        @printf(
            "%-10s %-20s %-20s %-30s\n",
            "dz (m)",
            "Mean LWP (g/m^2)",
            "Mean Cloud Frac (%)",
            "LWP Time Series (g/m^2)"
        )
        for dz in dz_values
            job_id = "$(case)_dz$(dz)_A$(A)"
            lwp_file = joinpath("output", job_id, "lwp_1h.nc")
            clt_file = joinpath("output", job_id, "clt_1h.nc")

            lwp_mean = NaN
            clt_mean = NaN
            lwp_ts_str = ""

            if isfile(lwp_file)
                NCDataset(lwp_file, "r") do ds
                    lwp_data = ds["lwp"][:] .* 1000.0 # convert kg/m^2 to g/m^2
                    lwp_ts = [mean(lwp_data[:, :, t]) for t in 1:size(lwp_data, 3)]
                    lwp_mean = mean(lwp_ts)
                    lwp_ts_str = join([@sprintf("%.1f", x) for x in lwp_ts], ", ")
                end
            end

            if isfile(clt_file)
                NCDataset(clt_file, "r") do ds
                    clt_data = ds["clt"][:] .* 100.0
                    clt_mean = mean(clt_data)
                end
            end

            @printf("%-10d %-20.2f %-20.2f %-30s\n", dz, lwp_mean, clt_mean, lwp_ts_str)
        end
    end
end
