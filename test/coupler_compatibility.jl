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

const z0m = 1e-3
const z0b = 1e-5
const gustiness = 1
const beta = 1
const T1 = 300
const T2 = 290

# TODO: Add surface flux exchange tests after new SurfaceFluxes.jl is integrated

#In this test, the ClimaAtmos "cache" p is overwritten so that it contains
# a surface field specified by the coupler, and then the internal function
# set_precomputed_quantities! is called to verify that this surface field is
# used correctly.

# Since overwriting the internal data structures of ClimaAtmos makes it harder
# to develop a modular API, the pattern demonstrated here should
# only be used to quickly write tests for the coupler.

@testset "Coupler Compatibility (Hacky Version)" begin
    # Initialize a model. The value of surface_setup is irrelevant, since it
    # will get overwritten.
    config = CA.AtmosConfig(
        Dict(
            "initial_condition" => "DYCOMS_RF02",
            "microphysics_model" => "0M",
            "config" => "column",
            "output_default_diagnostics" => false,
        );
        job_id = "coupler_compatibility1",
    )
    simulation = CA.get_simulation(config)
    (; integrator) = simulation
    (; p, t) = integrator
    Y = integrator.u
    FT = eltype(Y)
    thermo_params = CAP.thermodynamics_params(p.params)

    # Coupler pattern: build an AtmosSurface with a CoupledTemperature whose
    # field the driver writes into between steps, plus per-cell boundary
    # overrides for gustiness/beta. Re-build the atmos with this surface and
    # overwrite p.atmos / p.sfc_setup.
    sfc_space = Spaces.level(Y.f, half)
    T_field = similar(sfc_space, FT)
    @. T_field = FT(NaN)
    overrides = CA.SurfaceConditions.SurfaceBoundaryOverrides(;
        gustiness = FT(gustiness), beta = FT(beta),
    )
    overrides_field = similar(sfc_space, typeof(overrides))
    @. overrides_field = (overrides,)

    new_surface = CA.AtmosSurface(
        CA.SurfaceConditions.MoninObukhov(; z0m = FT(z0m), z0b = FT(z0b)),
        CA.SurfaceConditions.CoupledTemperature(T_field),
        overrides_field,
        p.atmos.surface.surface_albedo,
    )
    # AtmosModel is immutable, so swapping in `new_surface` requires rebuilding
    # the whole struct positionally — the kwarg form would reset every other
    # field (microphysics, radiation, ...) to its default and lose the config.
    a = p.atmos
    new_atmos = CA.AtmosModel{
        typeof(a.water), typeof(a.scm_setup), typeof(a.radiation),
        typeof(a.turbconv), typeof(a.prescribed_flow), typeof(a.gravity_wave),
        typeof(a.vertical_diffusion), typeof(a.sponge), typeof(new_surface),
        typeof(a.numerics), typeof(a.chemistry),
    }(
        a.water, a.scm_setup, a.radiation, a.turbconv, a.prescribed_flow,
        a.gravity_wave, a.vertical_diffusion, a.sponge, new_surface, a.numerics,
        a.chemistry, a.disable_surface_flux_tendency,
    )
    p_overwritten = CA.AtmosCache(
        p.dt,
        new_atmos,
        p.numerics,
        p.params,
        p.core,
        overrides_field,
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

    @. T_field = FT(T1)
    CA.set_precomputed_quantities!(Y, p_overwritten, t)
    sfc_T = p_overwritten.precomputed.sfc_conditions.T_sfc
    @test all(isequal(T1), parent(sfc_T))
    @. T_field = FT(T2)
    CA.set_precomputed_quantities!(Y, p_overwritten, t)
    sfc_T = p_overwritten.precomputed.sfc_conditions.T_sfc
    @test all(isequal(T2), parent(sfc_T))
end

@testset "Coupler Initialization" begin
    # Verify that using PrescribedSurface does not break the initialization of
    # RRTMGP or diagnostic EDMF. We currently need a moisture model in order to
    # use diagnostic EDMF.
    #
    # Also verify we can start with a different t_start than 0
    config = CA.AtmosConfig(
        Dict(
            "surface_setup" => "PrescribedSurface",
            "microphysics_model" => "0M",
            "rad" => "clearsky",
            "config" => "column",
            "turbconv" => "prognostic_edmfx",
            # NOTE: We do not output diagnostics because it leads to problems with Ubuntu on
            # GitHub actions taking too long to run (for unknown reasons). If you need this,
            # remove the following line and check that the test runs in less than a few
            # minutes on GitHub
            "output_default_diagnostics" => false,
            "t_start" => "1secs",
        );
        job_id = "coupler_compatibility3",
    )
    simulation = CA.get_simulation(config)
end
