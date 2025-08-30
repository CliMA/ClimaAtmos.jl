# Dependencies.
import YAML

"""
    modify_yaml(data, q_tot_0, theta_0, theta_i, z_i, toml)

Given a yaml file, modify some of the defined keywords with the given arguments.
"""
function modify_yaml(data::Dict, q_tot_0, theta_0, theta_i, z_i, toml)
    data["q_tot_0_dycoms_rf02"] = q_tot_0
    data["theta_0_dycoms_rf02"] = theta_0
    data["theta_i_dycoms_rf02"] = theta_i
    data["z_i_dycoms_rf02"] = z_i
    data["toml"] = [toml]

    return data
end

"""
    make_yamls(default_data, output_dir)

Given a default yaml file, create copies with perturbed initial conditions and 
store them in output_dir.
"""
function make_yamls(
    default_data_path::String,
    output_dir::String;
    is_prog::Bool = true,
)
    if is_prog
        prefix = "prognostic"
    else
        prefix = "diagnostic"
    end

    for qtot0 in [6.5, 8.5, 10.5] # 6.5 to 10.5 g/kg.
        for theta0 in [284.0, 287.0, 291.0, 294.0] # 284 to 294 K.
            for theta_jump in [6.0, 8.0, 10.0] # +6 to +10 K.
                for zi in [500, 800, 1000, 1300] # 500 to 1300 m.
                    for N in [30e6, 100e6, 250e6, 500e6] # 30 to 500 N/cm^3.
                        thetai = theta0 + theta_jump
                        toml = "LWP_N/toml/$(prefix)_edmfx_1M_prescribed_Nd_$N.toml"

                        default_data = YAML.load_file(default_data_path)
                        modified_data = modify_yaml(
                            default_data,
                            qtot0,
                            theta0,
                            thetai,
                            zi,
                            toml,
                        )

                        filename = "$(prefix)_edmfx_dycoms_rf02_column_qtot0_$(qtot0)_theta0_$(theta0)_thetai_$(thetai)_zi_$(zi)_prescribedN_$(N).yml"
                        outpath = joinpath(output_dir, filename)

                        open(outpath, "w") do io
                            YAML.write(io, modified_data)
                        end
                    end
                end
            end
        end
    end
end
