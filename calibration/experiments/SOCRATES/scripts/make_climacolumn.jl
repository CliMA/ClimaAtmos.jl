# Write the sampled SSCF forcing to ClimaColumn NetCDF files.
#
# Nothing in the run path needs this — runs build the forcing in memory. It exists so the forcing can
# be inspected with ordinary NetCDF tools, handed to another model, or driven back through ClimaAtmos's
# stock file-backed `Setups.ForcingFromFile`.

using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "model", "SocratesModel.jl"))

OUTPUT = joinpath(@__DIR__, "..", "runs", "climacolumn")

for case in SocratesModel.socrates_cases()
    name = SocratesModel.case_name(case)
    try
        path = SocratesModel.write_climacolumn(
            case,
            joinpath(OUTPUT, "$(name).nc");
            overwrite = true,
        )
        @info "wrote" name path z_levels = length(SocratesModel.socrates_z(Float64, case))
    catch e
        @error "failed" name exception = e
    end
end

# On a coarsened or custom grid, pass the levels the file should carry:
#   z = SM.socrates_z(Float64, case; dz_min = 50)
#   SM.write_climacolumn(case, path; z, overwrite = true)