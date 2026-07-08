using NCDatasets
using Statistics
using Printf

cases = ["DYCOMS_RF01", "Rico"]
A_values = [0.0, 0.2, 0.4, 0.6, 0.8]
dz_values = [50, 100, 200]

function get_latest_output_dir(job_id)
    base_dir = joinpath("output", job_id)
    if !ispath(base_dir)
        return nothing
    end
    subdirs = filter(x -> startswith(x, "output_"), readdir(base_dir))
    if isempty(subdirs)
        return nothing
    end
    sort!(subdirs)
    return joinpath(base_dir, last(subdirs))
end

function safe_mean(arr)
    if arr isa Number
        return (!ismissing(arr) && !isnan(arr)) ? Float64(arr) : NaN
    end
    clean = filter(x -> !ismissing(x) && !isnan(x), arr)
    return isempty(clean) ? NaN : mean(clean)
end

for case in cases
    println(
        "\n========================================================================================================================",
    )
    println("DETAILED ANALYSIS FOR CASE: $(case)")
    println(
        "========================================================================================================================",
    )

    for A in A_values
        println(
            "\n------------------------------------------------------------------------------------------------------------------------",
        )
        println("Entrainment Parameter A = $(A)")
        println(
            "------------------------------------------------------------------------------------------------------------------------",
        )
        @printf(
            "%-8s %-18s %-18s %-25s %-25s %-25s\n",
            "dz (m)",
            "Mean LWP (g/m^2)",
            "Mean Cloud Frac (%)",
            "Cloud Base Height (m)",
            "Cloud Top Height (m)",
            "Max q_liq (g/kg)"
        )

        for dz in dz_values
            job_id = "$(case)_dz$(dz)_A$(A)"
            out_dir = get_latest_output_dir(job_id)

            lwp_mean = NaN
            clt_mean = NaN
            cb_height_str = "N/A"
            ct_height_str = "N/A"
            max_qliq = NaN

            if !isnothing(out_dir)
                lwp_file = joinpath(out_dir, "lwp_1h_average.nc")
                clt_file = joinpath(out_dir, "clt_1h_average.nc")
                clw_file = joinpath(out_dir, "clw_1h_average.nc")

                if isfile(lwp_file)
                    NCDataset(lwp_file, "r") do ds
                        lwp_data = ds["lwp"][:] .* 1000.0 # convert kg/m^2 to g/m^2
                        lwp_mean = safe_mean(lwp_data)
                    end
                end

                if isfile(clt_file)
                    NCDataset(clt_file, "r") do ds
                        clt_data = ds["clt"][:] .* 100.0
                        clt_mean = safe_mean(clt_data)
                    end
                end

                if isfile(clw_file)
                    NCDataset(clw_file, "r") do ds
                        clw_data = ds["clw"][:] .* 1000.0 # convert kg/kg to g/kg
                        z_data = ds["z"][:]
                        n_z = length(z_data)

                        if ndims(clw_data) == 4
                            clw_final = [safe_mean(clw_data[:, :, k, end]) for k in 1:n_z]
                        elseif ndims(clw_data) == 2
                            clw_final = [safe_mean(clw_data[k, end]) for k in 1:n_z]
                        elseif ndims(clw_data) == 1
                            n_time = length(clw_data) ÷ n_z
                            clw_reshaped = reshape(clw_data, (n_z, n_time))
                            clw_final = [safe_mean(clw_reshaped[k, end]) for k in 1:n_z]
                        else
                            clw_final = zeros(n_z)
                        end

                        max_qliq = maximum(filter(!isnan, clw_final))
                        cloud_levels = findall(x -> x > 0.01, clw_final) # > 0.01 g/kg
                        if !isempty(cloud_levels)
                            cb_height_str = @sprintf("%.1f", z_data[first(cloud_levels)])
                            ct_height_str = @sprintf("%.1f", z_data[last(cloud_levels)])
                        else
                            cb_height_str = "No Cloud"
                            ct_height_str = "No Cloud"
                        end
                    end
                end
            end

            @printf(
                "%-8d %-18.2f %-18.2f %-25s %-25s %-25.3f\n",
                dz,
                lwp_mean,
                clt_mean,
                cb_height_str,
                ct_height_str,
                max_qliq
            )
        end
    end
end
