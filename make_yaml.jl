"""
    A script to automate the creation of YAML files with the specified initial
    conditions.
"""

using YAML

default_data = YAML.load_file("./prognostic_edmfx_dycoms_rf02_column.yml")

output_dir = "LWP_N_config"


function modify_yaml(data::Dict, q_tot_0, theta_0, theta_i, z_i, toml)

    data["q_tot_0_dycoms_rf02"] = q_tot_0
    data["theta_0_dycoms_rf02"] = theta_0
    data["theta_i_dycoms_rf02"] = theta_i
    data["z_i_dycoms_rf02"] = z_i
    data["toml"] = [toml]

    return data
end

for qtot0 in [6.5, 8.5, 10.5]
    for theta0 in [284.0, 287.0, 290.0, 294.0]
        for theta_jump in [6.0, 8.0, 10.0]
            for zi in [500, 800, 1000, 1300]
                for N in [30e6, 100e6, 250e6, 500e6]
                    thetai = theta0 + theta_jump
                    toml = "LWP_N_toml/prognostic_edmfx_1M_prescribed_Nd_$N.toml"

                    modified_data = modify_yaml(default_data, qtot0, theta0, thetai, zi, toml)

                    filename = "prognostic_edmfx_dycoms_rf02_column_qtot0_$(qtot0)_theta0_$(theta0)_thetai_$(thetai)_zi_$(zi)_prescribedN_$(N).yml"
                    outpath = joinpath(output_dir, filename)

                    open(outpath, "w") do io
                        YAML.write(io, modified_data)
                    end
                end
            end
        end
    end
end
