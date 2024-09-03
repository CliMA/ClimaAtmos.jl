import ClimaAnalysis
import CairoMakie
using Statistics
import ClimaComms
import ClimaAnalysis.Visualize as viz
import ClimaCore.InputOutput
import Plots, ClimaCorePlots
import ClimaCore.Fields

figres_x = 1000;
figres_y = 400;
days = 86400;

"""
    get_extrema_from_ncvars(var)
Assuming you have access to a ClimaAnalysis.OutputVar type object 
called `var`, gets the Cartesian indices of its extrema in a tuple
form. 
"""
function get_extrema_from_ncvars(var)
    _, imin = findmin(var.data)
    _, imax = findmax(var.data)
    dim_offset = 1 
    latmin = var.dims["lat"][imin.I[2+dim_offset]]
    latmax = var.dims["lat"][imax.I[2+dim_offset]]
    lonmin = var.dims["lon"][imin.I[1+dim_offset]]
    lonmax = var.dims["lon"][imax.I[1+dim_offset]]
    level = var.dims["z_reference"][imax.I[3+dim_offset]]
    # returns a (mincoords, maxcoords) tuple containing
    # the coordinates of interest (pass these as the target 
    # lon, lat coords to the cartesian index search function)
    return ((lonmin, latmin),(lonmax, latmax))
end

"""
    nearest_coord(field, target_lon, target_lat)
Given some `field` in lon-lat-z space, collects the
DataLayout indices corresponding to a specific long-lat
coordinate pair, and returns the index tuple (there may 
by multiple such matches at element edges). "Nearest" 
search depends on grid resolution keywords 
`res_lat` and `res_lon`. 
"""
function nearest_coord(field::XA, 
                       target_lon::FT, target_lat::FT; res_lat = FT(2), res_lon = FT(2)) where {XA, FT}
    lat_field = Fields.coordinate_field(field).lat
    lon_field = Fields.coordinate_field(field).long
    plat = parent(lat_field)
    plon = parent(lon_field)
    # We know coordinates occupy one field index, and are identical on all vertical levels
    # The accuracy criteria should depend on grid resolution (e.g. 2° grid is shown here)
    ind = findall(abs.(plat[1,:,:,:,:] .- target_lat) .< res_lat .&& abs.(plon[1,:,:,:,:] .- target_lon) .< res_lon)
    if length(ind) == 0
        @warn "No nearest indices were found :("
    end
    @info "$(length(ind)) nearest neighbours found"
    ind_tuple = [];
    for ix in ind 
        push!(ind_tuple, (1,ix.I...))
    end
    return ind_tuple
end

function get_ijh_indices(ind_tuple::XA) where {XA}
    sol = ind_tuple
    for (ii,x) in enumerate(ind_tuple)
        sol[ii] = x[2], x[3], last(x) 
    end
    return sol
end

function get_specific_column(field::F; target_lat = FT(-67), target_lon = FT(125)) where {F, FT}
    if target_lat == nothing || target_lon == nothing
        ituple = get_extrema_from_ncvars(hfes)
        # Assume max for argument's sake
        target_lat = ituple[2][2]
        target_lon = ituple[2][1]
    end
    ind_tuple = nearest_coord(field, target_lon, target_lat)
    colind = get_ijh_indices(ind_tuple)
    return Fields.column(field, colind[1]...) # Whats the right field iterator for this?
end

function get_location(ind_tuple::Tuple, X::XB) where {XB}
    return (parent(Fields.coordinate_field(X).long)[ind_tuple...],  parent(Fields.coordinate_field(X).lat)[ind_tuple...])
end

##########
##########
##########
##########

function getlevel(x, i)
    return Fields.level(x,i)
end
function get_debug_field(field)
    debug_field = zero(field)
    index_max = findmax(parent(field))[2].I
    index_min = findmin(parent(field))[2].I
    parent(debug_field)[index_max...] = FT(1000) # Assign arbitrary  -> Inf
    parent(debug_field)[index_min...] = FT(-1000) # Assign arbitrary  -> -Inf
    return index_max, index_min, debug_field
end

maxind, minind, debug_field = get_debug_field(field)

#Plots.plot(getlevel(get_debug_field(Y.c.ρq_tot ./ Y.c.ρ), index_max[1]))
#Plots.plot(getlevel(get_debug_field(Y.c.ρq_tot ./ Y.c.ρ), index_min[1]))

function gen_all_time_plots(var; lon_avg=true, yaxis_min= 0, yaxis_max = 20000)
    lon_avg == 1 ? var = ClimaAnalysis.average_lon(var) : nothing
    more_kwargs = Dict( 
                    :cb => Dict(:levels => [0:0.1:1]), 
                    :axis => Dict(:dim_on_y => true, :limits => (nothing, (yaxis_min,yaxis_max)))
                  )
    for (ii,time) in enumerate(var.dims["time"])
        fig = CairoMakie.Figure(size = (figres_x, figres_y));viz.plot!(
                                       fig,
                                       var;
                                       more_kwargs,
                                       time=time); 
        Colorbar(fig; highclip = 1, lowclip=0)
                                       display(fig)
        end
end

function gen_all_layer_time_plots(var; lon_avg=false, zlevel = 0)
    lon_avg == 1 ? var = ClimaAnalysis.average_lon(var) : nothing
    more_kwargs = Dict( 
                    :cb => Dict(:colorrange => (0,5)), 
                    :axis => Dict(:dim_on_y => true, :limits => (nothing, nothing))
                  )
    for (ii,time) in enumerate(var.dims["time"])
    # ClimaAnalysis Plots
        fig = CairoMakie.Figure(size = (figres_x, figres_y));viz.plot!(
                                       fig,
                                       var;
                                       more_kwargs,
                                       z_reference = zlevel,
                                       time=time); display(fig)
        end
end

function gen_all_surface_time_plots(var; ts=0, te=360, ti=30)
    more_kwargs = Dict( 
                    :cb => Dict(:colorrange => (0,5)), 
                    :axis => Dict(:dim_on_y => true, :limits => (nothing, nothing))
                  )
    for ii in var.dims["time"]
    # ClimaAnalysis Plots
        fig = CairoMakie.Figure(size = (figres_x, figres_y));viz.plot!(
                                       fig,
                                       var;
                                       more_kwargs,
                                       time=ii); display(fig)
        end
end


function gen_snapshot_plots(var; lon_avg=true, kwargs = more_kwargs)
    # ClimaAnalysis Plots
    lon_avg == 1 ? var = ClimaAnalysis.average_lon(var) : nothing
    for ii in var.dims["time"]
    fig = CairoMakie.Figure(size = (figres_x, figres_y));viz.plot!(
        fig,
        var,
        time=ii, kwargs...); display(fig)
    end
end

function record_all_time_plots(var; lon_avg=true, ts=0, te=360)
    # ClimaAnalysis Plots
    fig = CairoMakie.Figure(size = (figres_x, figres_y));
    CairoMakie.record(fig, "test.gif", ii) do ii
        gen_snapshot_plots(var; ii, lon_avg)
    end
end

function gen_all_levels(var; nlevels=63)
    # ClimaCore Plots
    for ii = 1:nlevels
        fig = Plots.plot(getlevel(Y.c.ρq_tot ./ Y.c.ρ,ii))
        display(fig)
    end
end


### GET vars
function get_test_vars(simdir)
    ua  = get(simdir; short_name = "ua", reduction, period);
    va = get(simdir; short_name = "va", reduction, period);
    ta  = get(simdir; short_name= "ta", reduction, period);
    wa = get(simdir; short_name = "wa", reduction, period);
    hus = get(simdir; short_name ="hus", reduction, period);
    hur = get(simdir; short_name = "hur", reduction, period);
    clw = get(simdir; short_name ="clw", reduction, period);
    cli = get(simdir; short_name ="cli", reduction, period);
    cl = get(simdir; short_name = "cl", reduction, period); 
    hfes = get(simdir; short_name ="hfes", reduction, period);
    evspsbl = get(simdir; short_name ="evspsbl", reduction, period);
    thetaa = get(simdir; short_name ="thetaa", reduction, period);
    rhoa = get(simdir; short_name = "rhoa", reduction, period);
    pr = get(simdir; short_name = "pr", reduction, period);
    #hussn = get(simdir; short_name = "hussn");
    #husra = get(simdir; short_name = "husra");
    #rlu = get(simdir; short_name ="rlu", reduction, period);
    #rlut = get(simdir; short_name ="rlut", reduction, period);
    #rsut = get(simdir; short_name ="rsut", reduction, period);
end

more_kwargs = Dict( 
                  :cb => Dict(:colorrange => (0,5)), 
                  :axis => Dict(:title=>"", :dim_on_y => true, :limits => (nothing, (0,1e4)))
              )

more_kwargs = Dict( 
    :cb => Dict(:colorrange => (0,5)), 
    :axis => Dict(:dim_on_y => true, :limits => (nothing, nothing))
)