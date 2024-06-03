import ClimaAnalysis
import CairoMakie
using Statistics
import ClimaAnalysis.Visualize as viz
import ClimaAnalysis.Visualize: line_plot1D!

more_kwargs = Dict(:plot => Dict(), 
                          :axis => Dict(:dim_on_y => true))

function generate_empty_figure(;
    resolution::Tuple = (8000, 8000),
    bgcolor::Tuple = (0.98, 0.98, 0.98),
    fontsize = 40,
)
    fig = Figure(;
        backgroundcolor = RGBf(bgcolor[1], bgcolor[2], bgcolor[3]),
        resolution,
        fontsize,
    )
    return fig
end

function horizontal_mean(var::ClimaAnalysis.OutputVar)
    ClimaAnalysis.average_y(ClimaAnalysis.average_x(var));
end

output_path = dir;

function plot_my_vars(output_path)
    for sn in collect(keys(simdir.vars))
        try 
            var = get(simdir; short_name = sn, reduction="inst");
            reduced_var = horizontal_mean(var)
            fig = CairoMakie.Figure(size = (1000, 1000));
            viz.plot!(fig, reduced_var, time=Inf)
            CairoMakie.save(joinpath(output_path, "$(sn).pdf"), fig)
        catch
            @warn "Either the reduction or dims don't match for x-y averages"
        end
    end
end

function get_perturbation_vars(var::ClimaAnalysis.OutputVar; n = 1)
    # Get raw data
    var_data =  var.data;
    # Get variable horizontal average at current timestep;
    spacemean_var = ClimaAnalysis.average_y(ClimaAnalysis.average_x(var));
    spacemean_var_data =  spacemean_var.data;
    new_data = similar(var_data);
    size_var_data = size(var_data);
    size_spacemean_var_data = size(spacemean_var_data); 
    @assert size_var_data[1] == size_spacemean_var_data[1];
    @assert size_var_data[end] == size_spacemean_var_data[end];
    for tdim in 1:size_var_data[1]
        for idim = 1:size_var_data[2]
            for jdim = 1:size_var_data[3]
                for ldim in 1:size_var_data[end]
                    new_data[tdim, idim, jdim, ldim] = 
                    (var_data[tdim, idim, jdim, ldim] .- spacemean_var_data[tdim, ldim]) ^ n;
                end
            end
        end
    end
   return ClimaAnalysis.OutputVar(var.attributes, 
                                  var.dims, 
                                  var.dim_attributes,
                                  new_data)
end

function get_perturbation_vars(var1::ClimaAnalysis.OutputVar, var2::ClimaAnalysis.OutputVar)
    # Get raw data
    var1_data =  var1.data;
    var2_data =  var2.data;
    # Get variable horizontal average at current timestep;
    spacemean_var1 = ClimaAnalysis.average_y(ClimaAnalysis.average_x(var1));
    spacemean_var2 = ClimaAnalysis.average_y(ClimaAnalysis.average_x(var2));
    spacemean_var1_data =  spacemean_var1.data;
    spacemean_var2_data =  spacemean_var2.data;
    new_data = similar(var1_data);
    size_var1_data = size(var1_data);
    size_var2_data = size(var2_data);
    size_spacemean_var1_data = size(spacemean_var1_data); 
    for tdim in 1:size_var1_data[1]
        for idim = 1:size_var1_data[2]
            for jdim = 1:size_var1_data[3]
                for ldim in 1:size_var1_data[end]
                    new_data[tdim, idim, jdim, ldim] = 
                    (var1_data[tdim, idim, jdim, ldim] .- spacemean_var1_data[tdim, ldim]) * 
                    (var2_data[tdim, idim, jdim, ldim] .- spacemean_var2_data[tdim, ldim]) 
                end
            end
        end
    end
    # TODO: Fix Attributes
    new_name = "⟨"*var1.attributes["short_name"]*"′"*var2.attributes["short_name"]*"′"*"⟩"
    return ClimaAnalysis.OutputVar(Dict("short_name" => new_name, "units"=>"", "long_name"=>""), 
                                   var1.dims, 
                                   var1.dim_attributes,
                                   new_data)
end

function get_perturbation_vars(var1::ClimaAnalysis.OutputVar, var2::ClimaAnalysis.OutputVar, var3::ClimaAnalysis.OutputVar)
    # Get raw data
    var1_data =  var1.data;
    var2_data =  var2.data;
    var3_data =  var3.data;
    # Get variable horizontal average at current timestep;
    spacemean_var1 = ClimaAnalysis.average_y(ClimaAnalysis.average_x(var1));
    spacemean_var2 = ClimaAnalysis.average_y(ClimaAnalysis.average_x(var2));
    spacemean_var3 = ClimaAnalysis.average_y(ClimaAnalysis.average_x(var3));
    spacemean_var1_data =  spacemean_var1.data;
    spacemean_var2_data =  spacemean_var2.data;
    spacemean_var3_data =  spacemean_var3.data;
    new_data = similar(var1_data);
    size_var1_data = size(var1_data);
    size_var2_data = size(var2_data);
    size_var3_data = size(var2_data);
    size_spacemean_var1_data = size(spacemean_var1_data); 
    for tdim in 1:size_var1_data[1]
        for idim = 1:size_var1_data[2]
            for jdim = 1:size_var1_data[3]
                for ldim in 1:size_var1_data[end]
                    new_data[tdim, idim, jdim, ldim] = 
                    (var1_data[tdim, idim, jdim, ldim] .- spacemean_var1_data[tdim, ldim]) * 
                    (var2_data[tdim, idim, jdim, ldim] .- spacemean_var2_data[tdim, ldim]) *
                    (var3_data[tdim, idim, jdim, ldim] .- spacemean_var3_data[tdim, ldim]) 
                end
            end
        end
    end
    # TODO: Fix Attributes
    new_name = "⟨"*var1.attributes["short_name"]*"′"*var2.attributes["short_name"]*"′"*var3.attributes["short_name"]*"′"*"⟩"
    return ClimaAnalysis.OutputVar(Dict("short_name" => new_name, "units"=>"", "long_name"=>""), 
                                   var1.dims, 
                                   var1.dim_attributes,
                                   new_data)
end

function plot_var(fig, var1;t_sim=Inf,p_loc = (1,1), more_kwargs...)
    line_plot1D!(
        fig,
        var1;p_loc, more_kwargs...);
    return fig
end

############
############
############

function make_generic_plots(; reduction = "inst", period="10m", dir=nothing)
    simdir = ClimaAnalysis.SimDir(dir)
    reduction = "inst"
    period = "10m"
    t_sim = Inf;

    ua = get(simdir; short_name = "ua", reduction, period);
    ha = get(simdir; short_name = "ha", reduction, period);
    ta = get(simdir; short_name= "ta", reduction, period);
    va = get(simdir; short_name = "va", reduction, period);
    wa = get(simdir; short_name = "wa", reduction, period);
    clw = get(simdir; short_name ="clw", reduction, period);
    hus = get(simdir; short_name ="hus", reduction, period);
    hur = get(simdir; short_name ="hur", reduction, period);
    cl = get(simdir; short_name ="cl", reduction, period);
    pr = get(simdir; short_name = "pr", reduction, period);
    thetaa = get(simdir; short_name ="thetaa", reduction, period);
    evspsbl = get(simdir; short_name ="evspsbl", reduction, period);
    hfes = get(simdir; short_name ="hfes", reduction, period);

    function reduce_var(var)
        ClimaAnalysis.average_time(ClimaAnalysis.window(ClimaAnalysis.average_y(ClimaAnalysis.average_x(var)),"time"; left=3600*5, right=Inf));
    end

    ta_mean = reduce_var(ta)
    thetaa_mean = reduce_var(thetaa)
    clw_mean = reduce_var(clw)
    cl_mean = reduce_var(cl)
    hus_mean = reduce_var(hus)
    hur_mean = reduce_var(hur)
    ha_mean = reduce_var(ha)
    ua_mean = reduce_var(ua)
    va_mean = reduce_var(va)
    wa_mean = reduce_var(wa)
    clw_mean = reduce_var(clw)
    pr_mean = reduce_var(pr)
    w′ = reduce_var(get_perturbation_vars(wa))
    u′u′ = reduce_var(get_perturbation_vars(ua,ua))
    v′v′ = reduce_var(get_perturbation_vars(va,va))
    w′w′ = reduce_var(get_perturbation_vars(wa,wa))
    w′w′w′ = reduce_var(get_perturbation_vars(wa,wa,wa))
    u′w′ = reduce_var(get_perturbation_vars(ua,wa))
    v′w′ = reduce_var(get_perturbation_vars(va,wa))
    u′v′ = reduce_var(get_perturbation_vars(ua,va))
    w′θ′ = reduce_var(get_perturbation_vars(wa,thetaa))
    w′ha′ = reduce_var(get_perturbation_vars(wa, ha))
    w′qt′ = reduce_var(get_perturbation_vars(wa,hus))
    w′ql′ = reduce_var(get_perturbation_vars(wa,clw))
    tke_red = 1/2 * (u′u′.data .+ w′w′.data + v′v′.data )

    tke_mean = ClimaAnalysis.OutputVar(Dict("short_name" => "tke", "units"=>"", "long_name"=>"turbulent kinetic energy"), 
                                        w′ql′.dims, 
                                        w′ql′.dim_attributes,
                                        tke_red)



    more_kwargs = Dict(:plot => Dict(), 
                    :axis => Dict(:dim_on_y => true, :ylabel => "z"),)
            

    MAX_NUM_ROWS = 5
    is_comparison = false
    function makefig()
        fig = CairoMakie.Figure(; size = (2400, 400 * MAX_NUM_ROWS), fontsize=12)
        if is_comparison
            for (col, path) in enumerate(output_path)
                # CairoMakie seems to use this Label to determine the width of the figure.
                # Here we normalize the length so that all the columns have the same width.
                LABEL_LENGTH = 240
                normalized_path =
                    lpad(path, LABEL_LENGTH + 1, " ")[(end - LABEL_LENGTH):end]

                CairoMakie.Label(fig[0, col], path)
            end
        end
        return fig
    end
    MAX_PLOTS_PER_PAGE = 10
    MAX_NUM_COLS = 4
    gridlayout() =
    map(1:MAX_PLOTS_PER_PAGE) do i
        row = mod(div(i - 1, MAX_NUM_COLS), MAX_NUM_ROWS) + 1
        col = mod(i - 1, MAX_NUM_COLS) + 1
        return fig[row, col] = CairoMakie.GridLayout()
    end

    # Plots
    fig = makefig()
    gridlayout()
    function genplots!(fig)
        # Unitful unit combinations????
        more_kwargs = Dict(:plot => Dict(), 
                            :axis => Dict(:title=>"", :dim_on_y => true, :ylabel=>"z [m]", :xlabel=>"u′u′ [m²s⁻²]"))
        plot_var(fig, u′u′; p_loc = (1,1), more_kwargs)

        more_kwargs = Dict(:plot => Dict(), 
                            :axis => Dict(:title=>"",:dim_on_y => true, :ylabel=>"z [m]", :xlabel=>"v′v′ [m²s⁻²]"))
        plot_var(fig, v′v′; p_loc = (1,2), more_kwargs)

        more_kwargs = Dict(:plot => Dict(), 
                            :axis => Dict(:title=>"",:dim_on_y => true, :ylabel=>"z [m]", :xlabel=>"w′w′ [m²s⁻²]"))
        plot_var(fig, w′w′; p_loc = (1,3), more_kwargs)

        more_kwargs = Dict(:plot => Dict(), 
                            :axis => Dict(:title=>"",:dim_on_y => true, :ylabel=>"z [m]", :xlabel=>"tke [m²s⁻²]"))
        plot_var(fig,tke_mean; p_loc = (1,4), more_kwargs)

        more_kwargs = Dict(:plot => Dict(), 
                            :axis => Dict(:title=>"",:dim_on_y => true, :ylabel=>"z [m]", :xlabel=>"w′qₜ′ [ms⁻¹]"))
        plot_var(fig,w′qt′; p_loc = (2,1), more_kwargs)

        more_kwargs = Dict(:plot => Dict(), 
                            :axis => Dict(:title=>"",:dim_on_y => true, :ylabel=>"z [m]", :xlabel=>"w′qₗ′ [ms⁻¹]"))
        plot_var(fig,w′ql′; p_loc = (2,2), more_kwargs)
        
        more_kwargs = Dict(:plot => Dict(), 
                            :axis => Dict(:title=>"",:dim_on_y => true, :ylabel=>"z [m]", :xlabel=>"w′ha′ [m³s⁻³]"))
        plot_var(fig,w′ha′; p_loc = (2,3), more_kwargs)
        
        more_kwargs = Dict(:plot => Dict(), 
                            :axis => Dict(:title=>"",:dim_on_y => true, :ylabel=>"z [m]", :xlabel=>"w′θ′ [Kms⁻¹]"))
        plot_var(fig, hur_mean; p_loc = (2,4), more_kwargs)
        
        more_kwargs = Dict(:plot => Dict(), 
                            :axis => Dict(:title=>"",:dim_on_y => true, :ylabel=>"z [m]", :xlabel=>"u [ms⁻¹]"))
        plot_var(fig, ua_mean; p_loc = (3,1), more_kwargs)
        
        more_kwargs = Dict(:plot => Dict(), 
                            :axis => Dict(:title=>"",:dim_on_y => true, :ylabel=>"z [m]", :xlabel=>"v [ms⁻¹]"))
        plot_var(fig, va_mean; p_loc = (3,2), more_kwargs)
        
        more_kwargs = Dict(:plot => Dict(), 
                            :axis => Dict(:title=>"",:dim_on_y => true, :ylabel=>"z [m]", :xlabel=>"w [ms⁻¹]"))
        plot_var(fig, wa_mean; p_loc = (3,3), more_kwargs)
        
        more_kwargs = Dict(:plot => Dict(), 
                            :axis => Dict(:title=>"",:dim_on_y => true, :ylabel=>"z [m]", :xlabel=>"cl [%]"))
        plot_var(fig, cl_mean; p_loc = (3,4), more_kwargs)
        
        more_kwargs = Dict(:plot => Dict(), 
                            :axis => Dict(:title=>"",:dim_on_y => true, :ylabel=>"z [m]", :xlabel=>"qₜ [kg/kg]"))
        plot_var(fig, hus_mean; p_loc = (4,1), more_kwargs)
        
        more_kwargs = Dict(:plot => Dict(), 
                            :axis => Dict(:title=>"",:dim_on_y => true, :ylabel=>"z [m]", :xlabel=>"θ [K]"))
        plot_var(fig, thetaa_mean; p_loc = (4,2), more_kwargs)

        more_kwargs = Dict(:plot => Dict(), 
        :axis => Dict(:title=>"",:dim_on_y => true, :ylabel=>"z [m]", :xlabel=>"qₗ [kg/kg]"))
        plot_var(fig, clw_mean; p_loc = (4,3), more_kwargs)
        
        more_kwargs = Dict(:plot => Dict(), 
                            :axis => Dict(:title=>"",:dim_on_y => true, :ylabel=>"z [m]", :xlabel=>"w′w′w′ [m³s⁻³]"))
        plot_var(fig, w′w′w′; p_loc = (4,4), more_kwargs)
        fig[0, :] = CairoMakie.Label(fig, "$(trunc(ua.dims["time"][end] / 3600.,digits=2)) hours")
    end
    genplots!(fig)
    display(fig)
end
