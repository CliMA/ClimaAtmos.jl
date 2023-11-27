using Test
import Thermodynamics as TD
import SurfaceFluxes as SF
import StaticArrays as SA
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaAtmos.SurfaceConditions: projected_vector_data, CT1, CT2
import ClimaCore: Spaces, Fields
import ClimaCore.Utilities: half

const z0m = 1e-3
const z0b = 1e-5
const gustiness = 1
const beta = 1
const T1 = 300
const T2 = 290

# This file contains two tests that illustrate how the coupler can modify the
# surface temperature in ClimaAtmos.

# In the first test, the ClimaAtmos "cache" p is overwritten so that it contains
# a surface field specified by the coupler, and then the internal function
# set_precomputed_quantities! is called to verify that this surface field is
# used correctly.

# In the second test, the cache is not overwritten. Instead, the coupler defines
# its own surface fields, and then it copies the data from those fields into the
# cache by calling set_surface_conditions!.

# Since overwriting the internal data structures of ClimaAtmos makes it harder
# to develop a modular API, the pattern demonstrated in the first test should
# only be used to quickly write tests for the coupler. For more long-term code,
# the pattern demonstrated in the second test is preferable.

@testset "Coupler Compatibility (Hacky Version)" begin
    # Initialize a model. The value of surface_setup is irrelevant, since it
    # will get overwritten.
    config = CA.AtmosConfig(Dict("initial_condition" => "DryBaroclinicWave"))
    integrator = CA.get_integrator(config)
    (; p, t) = integrator
    Y = integrator.u
    FT = eltype(Y)
    thermo_params = CAP.thermodynamics_params(p.params)

    # Override p.sfc_setup with a Field of SurfaceStates. The value of T is
    # irrelevant, since it will get updated.
    surface_state = CA.SurfaceConditions.SurfaceState(;
        parameterization = CA.SurfaceConditions.MoninObukhov(;
            z0m = FT(z0m),
            z0b = FT(z0b),
        ),
        T = FT(NaN),
        gustiness = FT(gustiness),
        beta = FT(beta),
    )
    sfc_setup = similar(Spaces.level(Y.f, half), typeof(surface_state))
    @. sfc_setup = (surface_state,)
    p_overwritten = CA.AtmosCache(
        p.simulation.dt,
        p.simulation,
        p.atmos,
        p.numerics,
        p.params,
        p.core,
        sfc_setup,
        p.ghost_buffer,
        p.env_thermo_quad,
        p.precomputed,
        p.scratch,
        p.hyperdiff,
        p.do_dss,
        p.rayleigh_sponge,
        p.viscous_sponge,
        p.precipitation,
        p.subsidence,
        p.large_scale_advection,
        p.edmf_coriolis,
        p.forcing,
        p.non_orographic_gravity_wave,
        p.orographic_gravity_wave,
        p.radiation,
        p.net_energy_flux_toa,
        p.net_energy_flux_sfc,
    )

    # Test that set_precomputed_quantities! can be used to update the surface
    # temperature to T1 and then to T2.
    @. sfc_setup.T = FT(T1)
    CA.set_precomputed_quantities!(Y, p_overwritten, t)
    sfc_T =
        @. TD.air_temperature(thermo_params, p.precomputed.sfc_conditions.ts)
    @test all(isequal(T1), parent(sfc_T))
    @. sfc_setup.T = FT(T2)
    CA.set_precomputed_quantities!(Y, p_overwritten, t)
    sfc_T =
        @. TD.air_temperature(thermo_params, p.precomputed.sfc_conditions.ts)
    @test all(isequal(T2), parent(sfc_T))
end

@testset "Coupler Compatibility (Proper Version)" begin
    # Initialize a model. Set surface_setup to PrescribedSurface to prevent
    # ClimaAtmos from modifying the surface conditions.
    config = CA.AtmosConfig(
        Dict(
            "initial_condition" => "DryBaroclinicWave",
            "surface_setup" => "PrescribedSurface",
        ),
    )
    integrator = CA.get_integrator(config)
    (; p, t) = integrator
    Y = integrator.u
    FT = eltype(Y)
    thermo_params = CAP.thermodynamics_params(p.params)

    # Allocate fields for storing the thermodynamic state and fluxes at the
    # surface.
    sfc_ts = similar(Spaces.level(Y.f, half), TD.PhaseDry{FT})
    sfc_conditions = similar(sfc_ts, SF.SurfaceFluxConditions{FT})

    # Define a function for updating these fields, given the temperature at the
    # surface and the current cache p.
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
    function surface_ts(surface_T, interior_ts, thermo_params)
        # Assume an adiabatic profile with constant cv and R to get surface_ρ.
        cv = TD.cv_m(thermo_params, interior_ts)
        R = TD.gas_constant_air(thermo_params, interior_ts)
        interior_ρ = TD.air_density(thermo_params, interior_ts)
        interior_T = TD.air_temperature(thermo_params, interior_ts)
        surface_ρ = interior_ρ * (surface_T / interior_T)^(cv / R)
        return TD.PhaseDry_ρT(thermo_params, surface_ρ, surface_T)
    end
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
        surface_values =
            SF.StateValues(surface_z, SA.SVector(FT(0), FT(0)), surface_ts)
        interior_values = SF.StateValues(
            interior_z,
            SA.SVector(interior_u, interior_v),
            interior_ts,
        )
        surface_inputs = SF.ValuesOnly(
            interior_values,
            surface_values,
            FT(z0m),
            FT(z0b),
            FT(gustiness),
            FT(beta),
        )
        return SF.surface_conditions(surface_params, surface_inputs)
    end

    # Test that set_surface_conditions! can be used to update the surface
    # temperature to T1 and then to T2.
    update_surface_fields!(sfc_ts, sfc_conditions, FT(T1), p)
    CA.SurfaceConditions.set_surface_conditions!(p, sfc_conditions, sfc_ts)
    sfc_T =
        @. TD.air_temperature(thermo_params, p.precomputed.sfc_conditions.ts)
    @test all(isequal(T1), parent(sfc_T))
    update_surface_fields!(sfc_ts, sfc_conditions, FT(T2), p)
    CA.SurfaceConditions.set_surface_conditions!(p, sfc_conditions, sfc_ts)
    sfc_T =
        @. TD.air_temperature(thermo_params, p.precomputed.sfc_conditions.ts)
    @test all(isequal(T2), parent(sfc_T))
end

@testset "Coupler Initialization" begin
    # Verify that using PrescribedSurface does not break the initialization of
    # RRTMGP or diagnostic EDMF. We currently need a moisture model in order to
    # use diagnostic EDMF.
    config = CA.AtmosConfig(
        Dict(
            "surface_setup" => "PrescribedSurface",
            "moist" => "equil",
            "rad" => "clearsky",
            "turbconv" => "diagnostic_edmfx",
        ),
    )
    integrator = CA.get_integrator(config)
end
