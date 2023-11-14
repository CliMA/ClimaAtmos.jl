include(joinpath(@__DIR__, "remap", "remap_helpers.jl"))

space_string(::Spaces.FaceExtrudedFiniteDifferenceSpace) = "(Face field)"
space_string(::Spaces.CenterExtrudedFiniteDifferenceSpace) = "(Center field)"

import ClimaCoreTempestRemap: def_space_coord
import ClimaCoreSpectra: power_spectrum_1d, power_spectrum_2d
import ClimaCore: Geometry, Fields, Spaces
using LinearAlgebra: norm
import ClimaAtmos: DryModel, EquilMoistModel, NonEquilMoistModel

function process_name(s::AbstractString)
    # "c_ρ", "c_ρe", "c_uₕ_1", "c_uₕ_2", "f_w_1"
    s = replace(s, "components_data_" => "")
    s = replace(s, "ₕ" => "_h")
    s = replace(s, "ρ" => "rho")
    return s
end
processed_varname(pc::Tuple) = process_name(join(pc, "_"))

# TODO: Make this a RecipesBase.@recipe
function profile_animation(sol, output_dir, fps)
    # Column animations
    Y0 = first(sol.u)
    for prop_chain in Fields.property_chains(Y0)
        var_name = processed_varname(prop_chain)
        var_space = axes(Fields.single_field(Y0, prop_chain))
        Ni, Nj, _, _, Nh = size(ClimaCore.Spaces.local_geometry_data(var_space))
        n_columns = Nh * Nj * Ni # TODO: is this correct?
        @info(
            "Creating profile animation",
            n_columns,
            var_name,
            n_timesteps = length(sol.u)
        )
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
                ϕ_col_std .+=
                    sqrt.((vec(ϕ_col) .- ϕ_col_ave) .^ 2 ./ n_columns)
            end

            # TODO: use xribbon when supported: https://github.com/JuliaPlots/Plots.jl/issues/2702
            Plots.plot(
                ϕ_col_ave,
                z ./ 1000;
                label = "Mean & Std",
                grid = false,
                xerror = ϕ_col_std,
                fillalpha = 0.5,
            )
            Plots.plot!(;
                xlabel = "$var_name",
                ylabel = "z [km]",
                markershape = :circle,
            )
            Plots.title!("$(space_string(var_space))")
        end
        Plots.mp4(
            anim,
            joinpath(output_dir, "profile_$var_name.mp4"),
            fps = fps,
        )
    end
end

function contour_animations(sol, output_dir, fps)
    for prop_chain in Fields.property_chains(sol.u[end])
        var_name = processed_varname(prop_chain)
        @info "Creating contour animation" var_name n_timesteps = length(sol.u)
        anim = Plots.@animate for Y in sol.u
            var = Fields.single_field(Y, prop_chain)
            level = 3
            # TODO: do not use ClimaCore internals
            if axes(var) isa Spaces.FaceExtrudedFiniteDifferenceSpace
                level = ClimaCore.Utilities.PlusHalf(level)
            end
            clim = (minimum(var), maximum(var))
            Plots.plot(var, level = level, clim = clim)
        end
        Plots.mp4(
            anim,
            joinpath(output_dir, "contour_$var_name.mp4"),
            fps = fps,
        )
    end
end

function postprocessing_box(sol, output_dir)
    for prop_chain in Fields.property_chains(sol.u[1])
        var_name = processed_varname(prop_chain)
        t_start = sol.t[1]
        var_start = Fields.single_field(sol.u[1], prop_chain)
        t_end = sol.t[end]
        var_end = Fields.single_field(sol.u[end], prop_chain)
        @info(
            "L₂ norm",
            var_name,
            t_start,
            norm(var_start),
            t_end,
            norm(var_end)
        )
    end

    Y = sol.u[end]
    FT = Spaces.undertype(axes(Y.c))
    y_max = maximum(Fields.coordinate_field(axes(Y.f)).y)
    ᶠw = Geometry.WVector.(Y.f.u₃).components.data.:1
    p = Plots.plot(ᶠw, slice = (:, FT(y_max / 2), :), clim = (-0.1, 0.1))
    Plots.png(p, joinpath(output_dir, "w.png"))

    if :ρq_tot in propertynames(Y.c)
        qt = Y.c.ρq_tot ./ Y.c.ρ
        pq =
            Plots.plot(qt, slice = (:, FT(y_max / 2), :), clim = (-0.02, 0.020))
        Plots.png(pq, joinpath(output_dir, "qt.png"))
    end
end

function postprocessing(sol, output_dir, fps)
    for prop_chain in Fields.property_chains(sol.u[1])
        var_name = processed_varname(prop_chain)
        t_start = sol.t[1]
        var_start = Fields.single_field(sol.u[1], prop_chain)
        t_end = sol.t[end]
        var_end = Fields.single_field(sol.u[end], prop_chain)
        @info(
            "L₂ norm",
            var_name,
            t_start,
            norm(var_start),
            t_end,
            norm(var_end)
        )
    end

    ᶠw_max = maximum(
        map(u -> maximum(parent(ClimaCore.Geometry.WVector.(u.f.w))), sol.u),
    )
    ᶠw_min = minimum(
        map(u -> minimum(parent(ClimaCore.Geometry.WVector.(u.f.w))), sol.u),
    )
    @info "maximum vertical velocity" ᶠw_max
    @info "maximum vertical velocity" ᶠw_min

    # contour_animations(sol, output_dir, fps) # For generic contours:

    anim = Plots.@animate for Y in sol.u
        ᶜv = Geometry.UVVector.(Y.c.uₕ).components.data.:2
        Plots.plot(ᶜv, level = 3, clim = (-6, 6))
    end
    Plots.mp4(anim, joinpath(output_dir, "v.mp4"), fps = fps)

    anim = Plots.@animate for Y in sol.u
        ᶠw = Geometry.WVector.(Y.f.u₃).components.data.:1
        Plots.plot(
            ᶠw,
            level = ClimaCore.Utilities.PlusHalf(3),
            clim = (-0.02, 0.02),
        )
    end
    Plots.mp4(anim, joinpath(output_dir, "w.mp4"), fps = fps)

    prop_chains = Fields.property_chains(sol.u[1])
    FT = Spaces.undertype(axes(Y.c))
    if any(pc -> pc == (:c, :ρq_tot), prop_chains)
        anim = Plots.@animate for Y in sol.u
            ᶜq_tot = Y.c.ρq_tot ./ Y.c.ρ
            Plots.plot(ᶜq_tot .* FT(1e3), level = 3, clim = (0, 1))
        end
        Plots.mp4(anim, joinpath(output_dir, "contour_q_tot.mp4"), fps = fps)
    else
        @info "Moisture not found" prop_chains
    end

    profile_animation(sol, output_dir, fps)
end

function safe_index(ius, t)
    iu = if isempty(ius)
        @warn "Could not find desired time for plotting, falling back on last day."
        length(t)
    else
        first(ius)
    end
end

function custom_postprocessing(sol, output_dir, p)
    thermo_params = CAP.thermodynamics_params(params)
    get_var(i, var) = Fields.single_field(sol.u[i], var)
    n = length(sol.u)
    #! format: off
    get_row(var) = [
        "Y.$(join(var, '.'))";;
        "$(norm(get_var(1, var), 2)) → $(norm(get_var(n, var), 2))";;
        "$(mean(get_var(1, var))) → $(mean(get_var(n, var)))";;
        "$(maximum(abs, get_var(1, var))) → $(maximum(abs, get_var(n, var)))";;
        "$(minimum(abs, get_var(1, var))) → $(minimum(abs, get_var(n, var)))";;
    ]
    #! format: on
    pretty_table(
        vcat(map(get_row, Fields.property_chains(sol.u[1]))...);
        title = "Change in Y from t = $(sol.t[1]) to t = $(sol.t[n]):",
        header = ["var", "‖var‖₂", "mean(var)", "max(∣var∣)", "min(∣var∣)"],
        alignment = :c,
    )

    anim = @animate for (Y, t) in zip(sol.u, sol.t)
        CA.set_precomputed_quantities!(Y, p, t) # sets ᶜts
        Plots.plot(
            vec(TD.air_temperature.(thermo_params, p.precomputed.ᶜts)),
            vec(Fields.coordinate_field(Y.c).z ./ 1000);
            xlabel = "T [K]",
            ylabel = "z [km]",
            xlims = (190, 310),
            legend = false,
        )
    end
    Plots.mp4(anim, joinpath(output_dir, "T.mp4"), fps = 10)

    anim = @animate for Y in sol.u
        w = Geometry.WVector.(Y.f.u₃).components.data.:1
        Plots.plot(
            vec(w),
            vec(Fields.coordinate_field(Y.f).z ./ 1000);
            xlabel = "w [m/s]",
            ylabel = "z [km]",
            xlims = (-0.02, 0.02),
            legend = false,
        )
    end
    Plots.mp4(anim, joinpath(output_dir, "w.mp4"), fps = 10)
end

function postprocessing_plane(sol, output_dir, p)
    thermo_params = CAP.thermodynamics_params(p.params)
    get_var(i, var) = Fields.single_field(sol.u[i], var)
    n = length(sol.u)
    #! format: off
    get_row(var) = [
        "Y.$(join(var, '.'))";;
        "$(norm(get_var(1, var), 2)) → $(norm(get_var(n, var), 2))";;
        "$(mean(get_var(1, var))) → $(mean(get_var(n, var)))";;
        "$(maximum(abs, get_var(1, var))) → $(maximum(abs, get_var(n, var)))";;
        "$(minimum(abs, get_var(1, var))) → $(minimum(abs, get_var(n, var)))";;
    ]
    #! format: on
    pretty_table(
        vcat(map(get_row, Fields.property_chains(sol.u[1]))...);
        title = "Change in Y from t = $(sol.t[1]) to t = $(sol.t[n]):",
        header = ["var", "‖var‖₂", "mean(var)", "max(∣var∣)", "min(∣var∣)"],
        alignment = :c,
    )

    Y = sol.u[end]

    ## Plots for last timestep
    function gen_plot_plane(
        variable::Fields.Field,
        filename::String,
        title::String,
        xlabel::String,
        ylabel::String;
        output_dir = output_dir,
    )
        # Set up Figure and Axes
        f = CairoMakie.Figure(; font = "CMU Serif")
        gaa = f[1, 1] = GridLayout()
        Axis(gaa[1, 1], aspect = 2, title = title)
        paa = fieldcontourf!(variable)
        Colorbar(gaa[1, 2], paa, label = xlabel)
        fig_png = joinpath(output_dir, filename)
        CairoMakie.save(fig_png, f)
    end

    gen_plot_plane(
        Geometry.UVector.(p.precomputed.ᶜu),
        "horz_velocity.png",
        "Horizontal Velocity",
        "u[m/s]",
        "z[m]",
    )

    gen_plot_plane(
        Geometry.WVector.(p.precomputed.ᶜu),
        "vert_velocity.png",
        "Vertical Velocity",
        "w[m/s]",
        "z[m]",
    )

    gen_plot_plane(
        TD.virtual_pottemp.(thermo_params, p.precomputed.ᶜts),
        "virtual_pottemp.png",
        "Virtual Pottemp",
        "Theta[K]",
        "z[m]",
    )
end
