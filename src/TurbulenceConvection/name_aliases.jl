"""
    name_aliases()

Returns a `Dict` containing:
 - a key (`String`), which we consider to be our core variable name
 - values (`Tuple` of `String`s), which we consider to be aliases of our core variable name
"""
function name_aliases()
    dict = Dict(
        "zc" => ("z_half",),
        "zf" => ("z",),
        "p_c" => ("p_half",),
        "p_f" => ("p",),
        "ρ_c" => ("rho0_half",),
        "ρ_f" => ("rho0",),
        "updraft_area" => ("updraft_fraction",),
        "updraft_thetal" => ("updraft_thetali",),
        "thetal_mean" => ("thetali_mean", "theta_mean"),
        "total_flux_h" => ("resolved_z_flux_thetali", "resolved_z_flux_theta"),
        "total_flux_qt" => ("resolved_z_flux_qt", "qt_flux_z"),
        "u_mean" => ("u_translational_mean",),
        "v_mean" => ("v_translational_mean",),
        "tke_mean" => ("tke_nd_mean",),
        "total_flux_s" => ("s_flux_z",),
        "lwp_mean" => ("lwp",),
        "iwp_mean" => ("iwp",),
        "rwp_mean" => ("rwp",),
        "swp_mean" => ("swp",),
        "cloud_base_mean" => ("cloud_base",),
        "cloud_top_mean" => ("cloud_top",),
    )
    return dict
end

"""
    get_nc_data(ds::NCDatasets.Dataset, var::String)

Returns the data for variable `var`, trying first its aliases
defined in `name_aliases`, in the `ds::NCDatasets.Dataset`.
"""
function get_nc_data(ds, var::String)
    dict = name_aliases()
    key_options = haskey(dict, var) ? (dict[var]..., var) : (var,)

    for key in key_options
        if haskey(ds, key)
            return ds[key]
        else
            for group_option in ["profiles", "reference", "timeseries"]
                haskey(ds.group, group_option) || continue
                if haskey(ds.group[group_option], key)
                    return ds.group[group_option][key]
                end
            end
        end
    end
    return nothing
end
