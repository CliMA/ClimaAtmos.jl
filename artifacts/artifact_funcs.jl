import ArtifactWrappers as AW

# ERA5 data 1973 Jan: downloaded from Caltech box
function era_global_dataset_path()
    era_dataset = AW.ArtifactWrapper(
        @__DIR__,
        "era-global",
        AW.ArtifactFile[AW.ArtifactFile(
            url = "https://caltech.box.com/shared/static/2489yvlwhnxsrl05jvqnch7fcrd0k1vw.nc",
            filename = "box-era5-monthly.nc",
        ),],
    )
    return AW.get_data_folder(era_dataset)
end

# download ERA5 nc data: wind, temperature, geopotential height
function era_single_column_dataset_path()
    era_dataset = AW.ArtifactWrapper(
        @__DIR__,
        "era-single-column",
        AW.ArtifactFile[AW.ArtifactFile(
            url = "https://caltech.box.com/shared/static/of5wi39o643a333yy9vbx5pnf0za503g.nc",
            filename = "box-single_column_test.nc",
        ),],
    )
    return AW.get_data_folder(era_dataset)
end

function topo_res_path()
    topo_data = AW.ArtifactWrapper(
        @__DIR__,
        "topo-info",
        AW.ArtifactFile[AW.ArtifactFile(
            url = "https://caltech.box.com/shared/static/isa7l4ow4xvv9vs09bivdwttbnnw5tte.nc",
            filename = "topo_drag.res.nc",
        ),],
    )
    return AW.get_data_folder(topo_data)
end

function mima_gwf_path()
    mima_data = AW.ArtifactWrapper(
        @__DIR__,
        "mima-gwf",
        AW.ArtifactFile[AW.ArtifactFile(
            url = "https://caltech.box.com/shared/static/6r566rv4631ibfbr5p5vtv4ls7vl5fge.nc",
            filename = "mima_gwf.nc",
        ),],
    )
    return AW.get_data_folder(mima_data)
end

function topo_elev_dataset_path()
    etopo1_elev_data = AW.ArtifactWrapper(
        @__DIR__,
        "topo-elev-info",
        AW.ArtifactFile[AW.ArtifactFile(
            url = "https://caltech.box.com/shared/static/gvilybsu5avxso1wubxjthpip9skc6mf.nc",
            filename = "ETOPO1_coarse.nc",
        ),],
    )
    return AW.get_data_folder(etopo1_elev_data)
end


function gfdl_ogw_data_path()
    gfdl_data = AW.ArtifactWrapper(
        @__DIR__,
        "gfdl-ogw-data",
        AW.ArtifactFile[AW.ArtifactFile(
            url = "https://caltech.box.com/shared/static/zubipz298q5ar5rpfais8c0ymtfyz2oc.nc",
            filename = "gfdl_ogw.nc",
        ),],
    )
    return AW.get_data_folder(gfdl_data)
end

function rrtmgp_artifact_path(path)
    artifact_name = "RRTMGPReferenceData"
    artifacts_file = joinpath(path, "test", "Artifacts.toml")
    return joinpath(
        Pkg.Artifacts.ensure_artifact_installed(artifact_name, artifacts_file),
        artifact_name,
    )
end
