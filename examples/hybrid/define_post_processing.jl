
space_string(::Spaces.FaceExtrudedFiniteDifferenceSpace) = "(Face field)"
space_string(::Spaces.CenterExtrudedFiniteDifferenceSpace) = "(Center field)"

# TODO: Make this a RecipesBase.@recipe
function profile_animation(sol, output_dir)
    # Column animations
    Y0 = first(sol.u)
    for prop_chain in Fields.property_chains(Y0)
        var_name = join(prop_chain, "_")
        var_space = axes(Fields.single_field(Y0, prop_chain))
        Ni, Nj, _, _, Nh = size(ClimaCore.Spaces.local_geometry_data(var_space))
        n_columns = Nh * Nj * Ni # TODO: is this correct?
        @info "Creating animation with n_columns = $n_columns, for $var_name"
        anim = Plots.@animate for Y in sol.u
            var = Fields.single_field(Y, prop_chain)
            temporary = ClimaCore.column(var, 1, 1, 1)
            ϕ_col_ave = deepcopy(vec(temporary))
            ϕ_col_std = deepcopy(vec(temporary))
            ϕ_col_ave .= 0
            ϕ_col_std .= 0
            local_geom = Fields.local_geometry_field(axes(var))
            z_f = ClimaCore.column(local_geom, 1, 1, 1)
            z_f = z_f.coordinates.z
            z = vec(z_f)
            for h in 1:Nh, j in 1:Nj, i in 1:Ni
                ϕ_col = ClimaCore.column(var, i, j, h)
                ϕ_col_ave .+= vec(ϕ_col) ./ n_columns
            end
            for h in 1:Nh, j in 1:Nj, i in 1:Ni
                ϕ_col = ClimaCore.column(var, i, j, h)
                ϕ_col_std .+= sqrt.((vec(ϕ_col) .- ϕ_col_ave) .^ 2 ./ n_columns)
            end

            # TODO: use xribbon when supported: https://github.com/JuliaPlots/Plots.jl/issues/2702
            # Plots.plot(ϕ_col_ave, z ./ 1000; label = "Mean & Variance", xerror=ϕ_col_std)
            # Plots.plot!(; ylabel = "z [km]", xlabel = "$var_name", markershape = :circle)

            Plots.plot(
                z ./ 1000,
                ϕ_col_ave;
                label = "Mean & Std",
                grid = false,
                ribbon = ϕ_col_std,
                fillalpha = 0.5,
            )
            Plots.plot!(;
                ylabel = "$var_name",
                xlabel = "z [km]",
                markershape = :circle,
            )
            Plots.title!("$(space_string(var_space))")
        end
        Plots.mp4(anim, joinpath(output_dir, "$var_name.mp4"), fps = 5)
    end
end
