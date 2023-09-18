import ClimaAtmos as CA

# Read all the diagnostics we know how to compute, and print them into a
# markdown table that is later compiled into the docs

# basename(pwd()) if the code is run from inside the docs folder. If we don't
# have that, we will assume that we are running from the project root. If this
# code is run from anywhere but these two places, mkdocs will fail to find
# availbale_diagnostics.md
prefix = basename(pwd()) == "docs" ? "" : "docs/"

out_path = "$(prefix)src/available_diagnostics.md"

open(out_path, "w") do file

    write(file, "# Available diagnostic variables\n\n")

    write(
        file,
        "| Short name | Long name | Standard name | Units | Comments |\n",
    )
    write(file, "|---|---|---|---|---|\n")

    for d in values(CA.Diagnostics.ALL_DIAGNOSTICS)
        write(file, "| `$(d.short_name)` ")
        write(file, "| $(d.long_name) ")
        write(file, "| `$(d.standard_name)` ")
        write(file, "| $(d.units) ")
        write(file, "| $(d.comments)|\n")
    end
end
@info "Written $out_path"
