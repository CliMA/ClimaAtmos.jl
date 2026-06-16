using NCDatasets

function unpack(dir, vars)
    tarfile = joinpath(dir, "nc_files.tar")
    files = [v * "_10m_inst.nc" for v in vars]
    missing_files = filter(f -> !isfile(joinpath(dir, f)), files)
    if !isempty(missing_files) && isfile(tarfile)
        run(Cmd(`tar xf nc_files.tar $(missing_files)`; dir))
    end
end

function report(dir)
    vars = ["clw", "waup", "arup", "iwp", "swp", "cli"]
    unpack(dir, vars)
    clw = NCDataset(joinpath(dir, "clw_10m_inst.nc")) do ds
        Array(ds["clw"][:, :])
    end
    z = NCDataset(joinpath(dir, "clw_10m_inst.nc")) do ds
        Array(ds["z"][:])
    end
    wa = NCDataset(joinpath(dir, "waup_10m_inst.nc")) do ds
        Array(ds["waup"][:, :])
    end
    ar = NCDataset(joinpath(dir, "arup_10m_inst.nc")) do ds
        Array(ds["arup"][:, :])
    end
    iwp = NCDataset(joinpath(dir, "iwp_10m_inst.nc")) do ds
        Array(ds["iwp"][:])
    end
    swp = NCDataset(joinpath(dir, "swp_10m_inst.nc")) do ds
        Array(ds["swp"][:])
    end

    nt = size(clw, 1)
    println(rpad("t(h)", 6), rpad("zmax_clw", 10), rpad("max_w", 10), rpad("max_area", 10), rpad("iwp", 10), "swp")
    for t in 1:4:nt
        zs = z[clw[t, :] .> 1e-6]
        zmax = isempty(zs) ? 0.0 : maximum(zs)
        println(
            rpad(round((t - 1) / 6, digits = 1), 6),
            rpad(round(zmax, digits = 0), 10),
            rpad(round(maximum(wa[t, :]), sigdigits = 3), 10),
            rpad(round(maximum(ar[t, :]), sigdigits = 3), 10),
            rpad(round(iwp[t], sigdigits = 3), 10),
            round(swp[t], sigdigits = 3),
        )
    end
end

report(ARGS[1])
