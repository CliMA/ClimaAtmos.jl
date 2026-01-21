using CairoMakie

# Plot Utilites
function generate_empty_figure(;
    size::Tuple = (2000, 2000),
    bgcolor::Tuple = (0.98, 0.98, 0.98),
    fontsize = 40,
)
    fig = Figure(;
        backgroundcolor = RGBf(bgcolor[1], bgcolor[2], bgcolor[3]),
        size,
        fontsize,
    )
    return fig
end

function create_plot!(
    fig::Figure;
    X = lon,
    Y = lat,
    Z = nothing,
    p_loc::Tuple = (1, 1),
    title = "",
    xlabel = "",
    ylabel = "",
    xscale = identity,
    yscale = identity,
    level::Int = 1,
    timeind::Int = 1,
    yreversed = true,
    colormap = :vik100,
    extendhigh = :magenta,
    extendlow = :cyan,
    colorrange = nothing,
    label = ("",),
    levels = 25,
    linewidth = 2,
    orientation = :vertical,
)
    if Z == nothing
        generic_axis = fig[p_loc[1], p_loc[2]] = GridLayout()
        axis = Axis(generic_axis[1, 1]; title, xlabel, ylabel, xscale, yscale)
        CairoMakie.lines!(X, Y; linewidth, label = label[1], linestyle = :solid)
    else
        generic_axis = fig[p_loc[1], p_loc[2]] = GridLayout() # Generic Axis Layout
        Axis(generic_axis[1, 1]; title, xlabel, ylabel, yscale, yreversed) # Plot specific attributes
        # custom_levels is a workaround for plotting constant fields with CairoMakie
        # If colorrange is specified, use it to create levels, otherwise use provided levels
        if colorrange !== nothing
            custom_levels = range(
                colorrange[1],
                colorrange[2];
                length = (levels isa Integer ? levels : 25),
            )
        elseif minimum(Z) ≈ maximum(Z)
            custom_levels = minimum(Z):0.1:(minimum(Z) + 0.2)
        else
            custom_levels = levels
        end

        generic_plot = CairoMakie.contourf!(
            X,
            Y,
            Z;
            levels = custom_levels,
            colormap = colormap,
            extendhigh = extendhigh,
            extendlow = extendlow,
        ) # Add plot contents
        Colorbar(generic_axis[1, 2], generic_plot)
    end
end
