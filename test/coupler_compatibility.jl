using Test
import ClimaComms
ClimaComms.@import_required_backends
import Thermodynamics as TD
import SurfaceFluxes as SF
import StaticArrays as SA
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaAtmos.SurfaceConditions: projected_vector_data, CT1, CT2
import ClimaCore: Spaces, Fields
import ClimaCore.Utilities: half

# Surface parameters for tests
const Z0M = 1e-3
const Z0B = 1e-5
const GUSTINESS = 1
const BETA = 1
const T_SFC_1 = 300.0
const T_SFC_2 = 290.0

# These tests verify that the coupler can modify surface temperature in ClimaAtmos.
# The "Hacky" version overwrites internal data structures.
# The "Proper" version uses the public API (set_surface_conditions!).
#
# In the first test, the ClimaAtmos "cache" p is overwritten so that it contains
# a surface field specified by the coupler, and then the internal function
# set_precomputed_quantities! is called to verify that this surface field is
# used correctly.
#
# In the second test, the cache is not overwritten. Instead, the coupler defines
# its own surface fields, and then it copies the data from those fields into the
# cache by calling set_surface_conditions!.

# Since overwriting the internal data structures of ClimaAtmos makes it harder
# to develop a modular API, the pattern demonstrated here should
# only be used to quickly write tests for the coupler.

@testset "Coupler Compatibility (Hacky Version)" begin
    config = CA.AtmosConfig(
        Dict(
            "initial_condition" => "DryBaroclinicWave",
            "output_dir_style" => "RemovePreexisting",
        );
        job_id = "coupler_compatibility1",
    )
    simulation = CA.AtmosSimulation(config)
    (; integrator) = simulation
    (; p, t) = integrator
    Y = integrator.u
    FT = eltype(Y)
    thermo_params = CAP.thermodynamics_params(p.params)

    # Create a surface state field and overwrite p.sfc_setup
    surface_state = CA.SurfaceConditions.SurfaceState(;
        parameterization = CA.SurfaceConditions.MoninObukhov(;
            z0m = FT(Z0M),
            z0b = FT(Z0B),
        ),
        T = FT(NaN),
        gustiness = FT(GUSTINESS),
        beta = FT(BETA),
    )
    sfc_setup = similar(Spaces.level(Y.f, half), typeof(surface_state))
    @. sfc_setup = (surface_state,)
    
    p_overwritten = CA.AtmosCache(
        p.dt,
        p.atmos,
        p.numerics,
        p.params,
        p.core,
        sfc_setup,
        p.ghost_buffer,
        p.precomputed,
        p.scratch,
        p.hyperdiff,
        p.external_forcing,
        p.non_orographic_gravity_wave,
        p.orographic_gravity_wave,
        p.radiation,
        p.tracers,
        p.net_energy_flux_toa,
        p.net_energy_flux_sfc,
        p.steady_state_velocity,
        p.conservation_check,
    )

    # Verify set_precomputed_quantities! updates surface temperature correctly
    @. sfc_setup.T = FT(T_SFC_1)
    CA.set_precomputed_quantities!(Y, p_overwritten, t)
    sfc_T = @. TD.air_temperature(thermo_params, p.precomputed.sfc_conditions.ts)
    @test all(isequal(T_SFC_1), parent(sfc_T))
    
    @. sfc_setup.T = FT(T_SFC_2)
    CA.set_precomputed_quantities!(Y, p_overwritten, t)
    sfc_T = @. TD.air_temperature(thermo_params, p.precomputed.sfc_conditions.ts)
    @test all(isequal(T_SFC_2), parent(sfc_T))
end

@testset "Coupler Compatibility (Proper Version)" begin
    config = CA.AtmosConfig(
        Dict(
            "initial_condition" => "DryBaroclinicWave",
            "surface_setup" => "PrescribedSurface",
        );
        job_id = "coupler_compatibility2",
    )
    simulation = CA.AtmosSimulation(config)

    # Check: ρ_flux_uₕ is initialized to zero
    @test all(
        iszero,
        parent(simulation.integrator.p.precomputed.sfc_conditions.ρ_flux_uₕ),
    )

    (; integrator) = simulation
    (; p, t) = integrator
    Y = integrator.u
    FT = eltype(Y)
    thermo_params = CAP.thermodynamics_params(p.params)

    # Allocate surface fields
    sfc_ts = similar(Spaces.level(Y.f, half), TD.PhaseDry{FT})
    sfc_conditions = similar(sfc_ts, SF.SurfaceFluxConditions{FT})

    # Helper: compute surface thermodynamic state assuming adiabatic profile
    function surface_ts(surface_T, interior_ts, thermo_params)
        cv = TD.cv_m(thermo_params, interior_ts)
        R = TD.gas_constant_air(thermo_params, interior_ts)
        interior_ρ = TD.air_density(thermo_params, interior_ts)
        interior_T = TD.air_temperature(thermo_params, interior_ts)
        surface_ρ = interior_ρ * (surface_T / interior_T)^(cv / R)
        return TD.PhaseDry_ρT(thermo_params, surface_ρ, surface_T)
    end

    # Helper: compute surface flux conditions
    function surface_conditions(
        surface_ts,
        surface_z,
        interior_ts,
        interior_u,
        interior_v,
        interior_z,
        surface_params,
    )
        FT = eltype(surface_ts)
        surface_values = SF.StateValues(surface_z, SA.SVector(FT(0), FT(0)), surface_ts)
        interior_values = SF.StateValues(
            interior_z,
            SA.SVector(interior_u, interior_v),
            interior_ts,
        )
        surface_inputs = SF.ValuesOnly(
            interior_values,
            surface_values,
            FT(Z0M),
            FT(Z0B);
            gustiness = FT(GUSTINESS),
            beta = FT(BETA),
        )
        return SF.surface_conditions(surface_params, surface_inputs)
    end

    # Update surface fields given temperature
    function update_surface_fields!(sfc_ts, sfc_conditions, surface_T, p)
        (; params) = p
        (; ᶜts, ᶜu) = p.precomputed
        thermo_params = CAP.thermodynamics_params(params)
        surface_params = CAP.surface_fluxes_params(params)

        sfc_z_values = Fields.field_values(Fields.coordinate_field(sfc_ts).z)
        sfc_ts_values = Fields.field_values(sfc_ts)
        sfc_conditions_values = Fields.field_values(sfc_conditions)

        int_local_geometry = Fields.local_geometry_field(Fields.level(ᶜts, 1))
        int_local_geometry_values = Fields.field_values(int_local_geometry)
        int_z_values = Fields.field_values(int_local_geometry.coordinates.z)
        int_ts_values = Fields.field_values(Fields.level(ᶜts, 1))
        int_u_values = Fields.field_values(Fields.level(ᶜu, 1))

        @. sfc_ts_values = surface_ts(surface_T, int_ts_values, thermo_params)
        @. sfc_conditions_values = surface_conditions(
            sfc_ts_values,
            sfc_z_values,
            int_ts_values,
            projected_vector_data(CT1, int_u_values, int_local_geometry_values),
            projected_vector_data(CT2, int_u_values, int_local_geometry_values),
            int_z_values,
            surface_params,
        )
    end

    # Verify set_surface_conditions! updates surface temperature correctly
    update_surface_fields!(sfc_ts, sfc_conditions, FT(T_SFC_1), p)
    CA.SurfaceConditions.set_surface_conditions!(p, sfc_conditions, sfc_ts)
    sfc_T = @. TD.air_temperature(thermo_params, p.precomputed.sfc_conditions.ts)
    @test all(isequal(T_SFC_1), parent(sfc_T))
    
    update_surface_fields!(sfc_ts, sfc_conditions, FT(T_SFC_2), p)
    CA.SurfaceConditions.set_surface_conditions!(p, sfc_conditions, sfc_ts)
    sfc_T = @. TD.air_temperature(thermo_params, p.precomputed.sfc_conditions.ts)
    @test all(isequal(T_SFC_2), parent(sfc_T))
end

@testset "Coupler Initialization" begin
    # Verify PrescribedSurface works with RRTMGP and diagnostic EDMF.
    # Also verify non-zero t_start works.
    config = CA.AtmosConfig(
        Dict(
            "surface_setup" => "PrescribedSurface",
            "moist" => "equil",
            "rad" => "clearsky",
            "co2_model" => "fixed",
            "turbconv" => "diagnostic_edmfx",
            "output_default_diagnostics" => false,
            "t_start" => "1secs",
        );
        job_id = "coupler_compatibility3",
    )
    simulation = CA.AtmosSimulation(config)
end
