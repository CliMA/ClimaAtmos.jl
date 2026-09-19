import PrecompileTools
import Logging
import Dates

# Grids, spaces, and a dry cache: the parts of a simulation that every
# configuration goes through. Case-specific models, the integrator, and the
# diagnostic writers are precompiled in `.buildkite/PrecompileCI`, which builds
# on top of this and does not repeat it.
#
# Set `precompile_workload = false` in LocalPreferences.toml to skip this.
PrecompileTools.@compile_workload begin
    Logging.with_logger(Logging.NullLogger()) do
        FT = Float32
        # TODO: compile CUDA methods as well
        context = ClimaComms.context(ClimaComms.CPUSingleThreaded())
        params = ClimaAtmosParameters(FT)

        sphere_grid = SphereGrid(FT; context)
        column_grid = ColumnGrid(FT; context)
        box_grid = BoxGrid(FT; context)
        plane_grid = PlaneGrid(FT; context)
        foreach(get_spaces, (sphere_grid, column_grid, box_grid, plane_grid))

        column_model = AtmosModel(
            column_grid;
            params,
            setup = Setups.IsothermalProfile(; temperature = FT(300)),
        )
        sphere_model = AtmosModel(sphere_grid; params)
        for model in (column_model, sphere_model)
            Y = initial_state(model)
            build_cache(
                Y, model, params, FT(60), Dates.DateTime(2010, 1, 1), nothing,
            )
        end
    end
end
