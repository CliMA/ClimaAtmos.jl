module PrecompileCI

using PrecompileTools, Logging
import Dates: DateTime
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaAtmos.Diagnostics as CAD
import ClimaComms
import ClimaParams

function _build_state_and_cache(
    FT,
    model,
    spaces;
    params,
    setup = CA.Setups.DecayingProfile(; params),
    aerosol_names = String[],
)
    dt = FT(1)
    start_date = DateTime(2010, 1, 1)
    Y = CA.Setups.initial_state(
        setup,
        params,
        model,
        spaces.center_space,
        spaces.face_space,
    )
    p = CA.build_cache(
        Y,
        model,
        params,
        dt,
        start_date,
        aerosol_names,
        (),       # time_varying_trace_gas_names
        nothing,  # steady_state_velocity
        nothing,  # vwb_species
    )
    return Y, p
end

# Compute and materialize every default diagnostic for `model`, exercising the
# same lazy-broadcast path that JITs on the first diagnostics callback. The
# reduction/schedule choices (driven by `duration`) do not affect the compute
# kernel types, so any duration is fine. Each compute is guarded so that a
# fixture that cannot be built in a given environment (e.g. radiation without
# artifacts) is skipped rather than failing precompilation.
function _precompile_default_diagnostics(
    FT,
    model,
    spaces;
    params,
    aerosol_names = String[],
)
    start_date = DateTime(2010, 1, 1)
    t_start = FT(0)
    duration = FT(86400)
    Y, p = _build_state_and_cache(FT, model, spaces; params, aerosol_names)
    diagnostics = CAD.default_diagnostics(
        model,
        duration,
        start_date,
        t_start;
        output_writer = CAD.DictWriter(),
        topography = false,
    )
    seen = Set{String}()
    for sd in diagnostics
        var = sd.variable
        # Dedup: the same variable appears once per reduction, but the compute
        # kernel is identical, so compiling it once is enough.
        var.short_name in seen && continue
        push!(seen, var.short_name)
        result =
            isnothing(var.compute) ? var.compute!(nothing, Y, p, t_start) :
            var.compute(Y, p, t_start)
        Base.materialize(result)
    end
    return nothing
end

@compile_workload begin
    with_logger(NullLogger()) do
        FT = Float32 # Float64?
        h_elem = 6 # 16, 30?
        z_elem = 10 # 30, 31, 63?
        x_elem = y_elem = 2
        x_max = y_max = 1e8
        z_max = FT(30000.0)
        dz_bottom = FT(500)
        z_stretch = true
        bubble = true
        nh_poly = 3 # GLL{4} = nh_poly + 1
        # TODO: compile CUDA methods as well
        context = ClimaComms.context(ClimaComms.CPUSingleThreaded())
        topography = CA.NoTopography()
        params = CA.ClimaAtmosParameters(FT)
        radius = CA.Parameters.planet_radius(params)

        sphere_grid = CA.SphereGrid(
            FT;
            context,
            radius, h_elem, nh_poly,
            z_elem, z_max, z_stretch, dz_bottom,
            bubble, topography,
        )
        box_grid = CA.BoxGrid(
            FT;
            context,
            x_elem, x_max, y_elem, y_max, nh_poly, periodic_x = true, periodic_y = true,
            z_elem, z_max, z_stretch, dz_bottom,
            bubble, topography,
        )
        plane_grid = CA.PlaneGrid(
            FT;
            context,
            x_elem, x_max, nh_poly, periodic_x = true,
            z_elem, z_max, z_stretch, dz_bottom,
            topography,
        )
        column_grid = CA.ColumnGrid(
            FT; context, z_elem, z_max, z_stretch, dz_bottom,
        )
        all_grids = (sphere_grid, box_grid, plane_grid, column_grid)
        foreach(CA.get_spaces, all_grids)

        # Precompile the default diagnostics for the model configurations
        # exercised in CI, on both the sphere (aquaplanet/baroclinic) and the
        # column (SCM/EDMFX) spaces.
        sphere_spaces = CA.get_spaces(sphere_grid)
        column_spaces = CA.get_spaces(column_grid)

        # Shared EDMF configuration -- minimal (no entr/detr, no SGS fluxes),
        # mirroring test/diagnostics/unit_diagnostics.jl.
        tcp = CAP.turbconv_params(params)
        edmfx_model = CA.EDMFXModel(;
            entr_model = CA.NoEntrainment(),
            detr_model = CA.NoDetrainment(),
            scale_blending_method = CA.SmoothMinimumBlending(),
        )
        pedmfx = CA.PrognosticEDMFX(; area_fraction = tcp.min_area)

        # (grid_spaces, model, aerosol_names) fixtures. Each is guarded so a
        # configuration that cannot be built in the current environment does not
        # abort precompilation of the others.
        sphere_fixtures = (
            # Dry + slab ocean (core diagnostics on the sphere)
            (
                CA.AtmosModel(;
                    microphysics_model = CA.DryModel(),
                    temperature = CA.SurfaceConditions.SlabOceanTemperature{FT}(),
                ), String[]),
            # Equilibrium 0-moment (core + moist diagnostics on the sphere)
            (CA.AtmosModel(;
                    microphysics_model = CA.EquilibriumMicrophysics0M(),
                ), String[]),
            # All-sky radiation with clear-sky diagnostics + aerosol
            (
                CA.AtmosModel(;
                    microphysics_model = CA.EquilibriumMicrophysics0M(),
                    radiation_mode = CA.RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics(;
                        aerosol_radiation = true,
                    ),
                ), ["DST01"]),
        )
        column_fixtures = (
            # Microphysics variants (precipitation diagnostics)
            (
                CA.AtmosModel(; microphysics_model = CA.NonEquilibriumMicrophysics1M()),
                String[],
            ),
            (
                CA.AtmosModel(; microphysics_model = CA.NonEquilibriumMicrophysics2M()),
                String[],
            ),
            # Prognostic EDMFX (draft/environment diagnostics) with 0M and 1M
            (
                CA.AtmosModel(;
                    microphysics_model = CA.EquilibriumMicrophysics0M(),
                    turbconv_model = pedmfx, edmfx_model,
                ), String[]),
            (
                CA.AtmosModel(;
                    microphysics_model = CA.NonEquilibriumMicrophysics1M(),
                    turbconv_model = pedmfx, edmfx_model,
                ), String[]),
        )

        for (spaces, fixtures) in
            ((sphere_spaces, sphere_fixtures), (column_spaces, column_fixtures))
            for (model, aerosol_names) in fixtures
                try
                    _precompile_default_diagnostics(
                        FT, model, spaces; params, aerosol_names,
                    )
                catch
                    # Best-effort precompilation: skip fixtures that cannot be
                    # built in this environment (e.g. missing radiation artifacts).
                end
            end
        end
    end
end

end # module PrecompileCI
