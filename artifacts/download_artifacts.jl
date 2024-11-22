include(joinpath(@__DIR__, "artifact_funcs.jl"))

# Trigger download if data doesn't exist locally
function trigger_download(lazy_download = true)
    @info "Era global dataset path:`$(era_global_dataset_path())`"
    @info "Era single column dataset path:`$(era_single_column_dataset_path())`"
    @info "topo dataset path:`$(topo_res_path())`"
    @info "MiMA convective gravity wave path:`$(mima_gwf_path())`"
    @info "GFDL OGWD test data:`$(gfdl_ogw_data_path())`"
    return nothing
end
trigger_download()
