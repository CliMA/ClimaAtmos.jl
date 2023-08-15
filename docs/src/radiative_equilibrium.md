# Example
A suite of concrete examples are provided here as a guidance for running single column Radiative Equilibrium.  


# Radiative Equilibrium
Radiative Equilibrium can be a useful tool for analyzing atmospheric temperature profiles with differing conditions. To run a radiative equilibrium simulation open run the following commands in your terminal (after instantiating the necessary packages):

```
julia --color=yes --project=examples examples/hybrid/driver.jl --rad clearsky --idealized_h2o true --hyperdiff false --config column --initial_condition IsothermalProfile --z_max 70e3 --z_elem 70 --dz_bottom 100 --dz_top 10000 --t_end 654days --dt 3hours --dt_save_to_sol 30hours --dt_save_to_disk 100days --prognostic_surface true --job_id single_column_radiative_equilibrium_clearsky_prognostic_surface_temp
```

The command line inputs above can be changed to adjust the kind of simulation desired. Options such as t_end can just be increased to the desired length, other arguments such as radiation type can be changed based on the options in `cli_options.jl`. 

After a simulation is ran the output files should be available in the output folder, where hdf5 files store all the simulation's data, and two mp4s should be produced with a vertical temperature profile and a vertical wind profile which change over time. If you want to quickly access data from the final day of the simulation wants to be accessed quick simply add a -i when running the initial simulation, like so:

```
julia -i --color=yes --proj...
```

This will bring you into a julia environment, where you can run propertynames(Y) and propertynames(p) to see accesible values, examples:


![](assets/CO2_sample_code.png)



Here is a typical vertical temperature profile on the final day of a simulation (the dot is surface temperature):
** add some sort of png here

Note: The surface temperature must also converge before the simulation reaches radiative equilibrium, here is a typical convergence of surface temperature plot:

![](assets/example_temp_profile.png)



# Greenhouse Gas Concentrations
In `radiation.jl` one can change the concentrations of greenhouse gases in order to test the affects on the atmospheric temperature profile.

To change the ozone concentration multiply the following variable the desired coefficent:

![](assets/Ozone_sample_code.png)

Then comparing the results between ozone simulations could look something like this:

![](assets/O%E2%82%83_exp_temp_profile.png)
![](assets/O%E2%82%83_exp_temp_difference.png)

To change the CO2 concentration also multiply the volume mixing ratio by the desired coefficient.

![](assets/CO2_sample_code.png)

Results from increasing CO2 concentration could look something like this:

![](assets/CO%E2%82%82_exp_temp_profile.png)
![](assets/CO%E2%82%82_exp_temp_difference.png)


A similar process to the ones above can work for increasing the concentration of any other gases in the atmosphere



# Data extraction in Julia (Is this necessary?)
To access the data from hdf5 files one can use the following code to create a var `diagnostics` which stores values from a hdf5 file path:

```
import ClimaAtmos: time_from_filename
import ClimaCore: Geometry, Spaces, Fields, InputOutput
import ClimaComms
import CairoMakie: Makie
import Statistics: mean

function read_hdf5_file(file_path)
    reader =
        InputOutput.HDF5Reader(file_path, ClimaComms.SingletonCommsContext())
    diagnostics = InputOutput.read_field(reader, "diagnostics")
    close(reader)
    return time_from_filename(file_path), diagnostics
end

time, diagnostics = read_hdf5_file("sample/file/path")
```

`diagnostics` will contain the values from the state of the simulation on a given day. propertynames(diagnostics) can be useful for finding the variables names. Example: `diagnostics.sfc_temperature` will result in the surface temperature at that day in the simulation.

--could explain more code
add comment called hide (look in documenter.jl)
lines!(x,y) # hide
display(fig)  # hide

Note: the data structures used to store the data can be quite tricky to use 
