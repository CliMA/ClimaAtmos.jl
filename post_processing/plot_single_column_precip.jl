import ClimaAtmos as CA
import ClimaCore as CO
import CairoMakie as MK

fig = MK.Figure(resolution = (1200, 600))
ax1 = MK.Axis(fig[1, 1], ylabel = "z [km]", xlabel = "q_tot [g/kg]")
ax2 = MK.Axis(fig[1, 2],                    xlabel = "q_liq [g/kg]")
ax3 = MK.Axis(fig[1, 3],                    xlabel = "q_ice [g/kg]")
ax4 = MK.Axis(fig[2, 1], ylabel = "z [km]", xlabel = "T [K]")
ax5 = MK.Axis(fig[2, 2],                    xlabel = "q_rai [g/kg]")
ax6 = MK.Axis(fig[2, 3],                    xlabel = "q_sno [g/kg]")

path = joinpath(pkgdir(CA), "output", "single_column_precipitation_test")

col = Dict("0" => :navy, "500" => :blue2, "1000" => :royalblue, "1500" => :skyblue1)

for time in ["0", "500", "1000", "1500"]

    fqₜ = CO.InputOutput.HDF5Reader(joinpath(path, "hus_inst_" * time * ".0.h5"))
    fqₗ = CO.InputOutput.HDF5Reader(joinpath(path, "clw_inst_" * time * ".0.h5"))
    fqᵢ = CO.InputOutput.HDF5Reader(joinpath(path, "cli_inst_" * time * ".0.h5"))
    fqᵣ = CO.InputOutput.HDF5Reader(joinpath(path, "husra_inst_" * time * ".0.h5"))
    fqₛ = CO.InputOutput.HDF5Reader(joinpath(path, "hussn_inst_" * time * ".0.h5"))
    fTₐ = CO.InputOutput.HDF5Reader(joinpath(path, "ta_inst_" * time * ".0.h5"))
    fwₐ = CO.InputOutput.HDF5Reader(joinpath(path, "wa_inst_" * time * ".0.h5"))

    qₜ = CO.InputOutput.read_field(fqₜ, "hus_inst")
    qₗ = CO.InputOutput.read_field(fqₗ, "clw_inst")
    qᵢ = CO.InputOutput.read_field(fqᵢ, "cli_inst")
    qᵣ = CO.InputOutput.read_field(fqᵣ, "husra_inst")
    qₛ = CO.InputOutput.read_field(fqₛ, "hussn_inst")
    Tₐ = CO.InputOutput.read_field(fTₐ, "ta_inst")
    wₐ = CO.InputOutput.read_field(fwₐ, "wa_inst")

    qₜ_col = CO.Fields.column(qₜ,1,1,1)
    qₗ_col = CO.Fields.column(qₗ,1,1,1)
    qᵢ_col = CO.Fields.column(qᵢ,1,1,1)
    qᵣ_col = CO.Fields.column(qᵣ,1,1,1)
    qₛ_col = CO.Fields.column(qₛ,1,1,1)
    Tₐ_col = CO.Fields.column(Tₐ,1,1,1)
    wₐ_col = CO.Fields.column(wₐ,1,1,1)
    z = CO.Fields.coordinate_field(qₜ_col).z

    MK.lines!(ax1, vec(parent(qₜ_col)) .* 1e3, vec(parent(z)) ./ 1e3, color=col[time])
    MK.lines!(ax2, vec(parent(qₗ_col)) .* 1e3, vec(parent(z)) ./ 1e3, color=col[time])
    MK.lines!(ax3, vec(parent(qᵢ_col)) .* 1e3, vec(parent(z)) ./ 1e3, color=col[time])
    MK.lines!(ax4, vec(parent(Tₐ_col)),        vec(parent(z)) ./ 1e3, color=col[time])
    MK.lines!(ax5, vec(parent(qᵣ_col)) .* 1e3, vec(parent(z)) ./ 1e3, color=col[time])
    MK.lines!(ax6, vec(parent(qₛ_col)) .* 1e3, vec(parent(z)) ./ 1e3, color=col[time])

    for fid in [fqₜ, fqₗ, fqᵢ, fqᵣ, fqₛ, fTₐ, fwₐ]
        close(fid)
    end
end

MK.save("todo.pdf", fig)
