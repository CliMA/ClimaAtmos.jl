using NCDatasets

function first_nonfinite_along_time(file, varname)
    ds = NCDataset(file, "r")
    try
        @assert haskey(ds, varname) "missing variable $(varname)"
        v = ds[varname]
        dims = Tuple(String.(dimnames(v)))
        tpos = findfirst(==("time"), dims)
        if isnothing(tpos)
            return (dims = dims, nt = 0, bad = 0)
        end
        A = ndims(v) == 1 ? v[:] : Array(v.var[:])
        nt = size(A, tpos)
        bad = 0
        for t in 1:nt
            slab = selectdim(A, tpos, t)
            if any(x -> !isfinite(x), slab)
                bad = t
                break
            end
        end
        return (dims = dims, nt = nt, bad = bad)
    finally
        close(ds)
    end
end

files_and_vars = [
    (
        "output/simple_socrates/iteration_000/member_003/config_1/output_active/thetaa_10m_inst.nc",
        ["thetaa"],
    ),
    (
        "output/simple_socrates/iteration_000/member_003/config_1/output_active/hus_10m_inst.nc",
        ["hus"],
    ),
    (
        "output/simple_socrates/iteration_000/member_003/config_1/output_active/clw_10m_inst.nc",
        ["clw"],
    ),
    (
        "output/simple_socrates/iteration_000/member_003/config_1/debug_inputs/forcing_inputs_on_model_z.nc",
        ["ta", "hus", "ua", "va", "wa", "tntha", "tnhusha"],
    ),
]

for (f, vars) in files_and_vars
    println("FILE: ", f)
    for vn in vars
        try
            r = first_nonfinite_along_time(f, vn)
            println("  ", vn, " dims=", r.dims, " nt=", r.nt, " first_bad_t=", r.bad)
        catch err
            println("  ", vn, " ERROR: ", sprint(showerror, err))
        end
    end
end
