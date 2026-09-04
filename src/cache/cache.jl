"""
    AtmosCache

Cache of preallocated fields and model settings passed as `p` to every tendency
function, callback, and diagnostic.

The cache is built once by `build_cache` and mutated in place thereafter: no
`Field` may be allocated inside a tendency or cache setter (use `p.scratch` or
lazy broadcasting instead). Entries of `p.precomputed` are refreshed from the
current state `Y` by the `set_*_precomputed_quantities!` family of functions
(see `set_precomputed_quantities!`); entries of `p.scratch` hold no persistent
state and may be overwritten by any function.

# Fields

  - `dt`: Simulation timestep, also used by callbacks and tendencies [s].
  - `atmos`: The `AtmosModel` configuration.
  - `numerics`: Limiters (quasi-monotone, tracer-nonnegativity, vertical water borrowing).
  - `params`: The `ClimaAtmosParameters` used by the model.
  - `core`: Generally used quantities, such as the geopotential `ᶜΦ`, its gradients,
    the Coriolis fields `ᶜf³`/`ᶠf¹²`, and the surface unit basis vector.
  - `sfc_setup`: Surface boundary overrides, used by `update_surface_conditions!`
    and the coupler.
  - `ghost_buffer`: Center and face ghost buffers used by DSS.
  - `precomputed`: Quantities updated by `set_precomputed_quantities!`.
  - `scratch`: Preallocated temporary fields with no persistent state.
  - `hyperdiff`: Hyperdiffusion quantities for grid-scale and subgrid-scale
    variables, potentially with ghost buffers for DSS.
  - `external_forcing`: Parameters used by external-forcing tendencies.
  - `non_orographic_gravity_wave`: Parameters used by the non-orographic gravity
    wave tendency.
  - `orographic_gravity_wave`: Parameters used by the orographic gravity wave tendency.
  - `radiation`: Radiation model cache (e.g. the RRTMGP solver).
  - `tracers`: Prescribed aerosol and trace-gas inputs.
  - `net_energy_flux_toa`: Net radiative energy that has entered through the top of
    the atmosphere, accumulated over the domain area and over time by the
    `flux_accumulation!` callback. A one-element vector, so that it can be mutated
    in place [J].
  - `net_energy_flux_sfc`: The same accumulated energy at the surface; not
    accumulated when the surface temperature is a `SlabOceanTemperature` [J].
  - `steady_state_velocity`: Predicted steady-state velocity, if `check_steady_state`
    is `true`; otherwise `nothing`.
  - `conservation_check`: Column-integrated precipitation energy tendency, used for
    the conservation check with a prognostic surface temperature.
"""
struct AtmosCache{
    FT,
    AM,
    NUM,
    CAP,
    COR,
    SFC,
    GHOST,
    PREC,
    SCRA,
    HYPE,
    EXTFORCING,
    NONGW,
    ORGW,
    RAD,
    TRAC,
    NETFLUXTOA,
    NETFLUXSFC,
    SSV,
    CONSCHECK,
}
    # Timestep of the simulation (in seconds); also used by callbacks and tendencies
    dt::FT

    # AtmosModel
    atmos::AM

    # Limiter
    numerics::NUM

    # ClimaAtmosParameters that have to be used
    params::CAP

    # Variables that are used generally, such as ᶜΦ
    core::COR

    # Used by update_surface_conditions! in set_precomputed_quantities! and coupler
    sfc_setup::SFC

    # Center and face ghost buffers used by DSS
    ghost_buffer::GHOST

    # Quantities that are updated with set_precomputed_quantities!
    precomputed::PREC

    # Pre-allocated areas of memory to store temporary values
    scratch::SCRA

    # Hyperdiffision quantities for grid and subgrid scale quantities, potentially with
    # ghost buffers for DSS
    hyperdiff::HYPE

    # Additional parameters used by the various tendencies
    external_forcing::EXTFORCING
    non_orographic_gravity_wave::NONGW
    orographic_gravity_wave::ORGW
    radiation::RAD
    tracers::TRAC

    # Net energy flux coming through top of atmosphere and surface
    net_energy_flux_toa::NETFLUXTOA
    net_energy_flux_sfc::NETFLUXSFC

    # Predicted steady-state velocity, if `check_steady_state` is `true`
    steady_state_velocity::SSV

    # Conservation check for prognostic surface temperature
    conservation_check::CONSCHECK
end

# Allow cache to be moved on the CPU. Used by ClimaCoupler to save checkpoints
Adapt.@adapt_structure AtmosCache

# Functions on which the model depends:
# CAP.R_d(params)         # dry specific gas constant
# CAP.kappa_d(params)     # dry adiabatic exponent
# CAP.T_triple(params)    # triple point temperature of water
# CAP.MSLP(params)        # reference pressure
# CAP.grav(params)        # gravitational acceleration
# CAP.Omega(params)       # rotation rate (only used if space is spherical)
# CAP.cv_d(params)        # dry isochoric specific heat capacity
# The value of cv_d is implied by the values of R_d and kappa_d

# The model also depends on f_plane_coriolis_frequency(params)
# This is a constant Coriolis frequency that is only used if space is flat

"""
    build_cache(Y, atmos, params, dt, start_date, steady_state_velocity)

Allocate and initialize the `AtmosCache` `p` for the initial state `Y` and model
configuration `atmos`.

All fields needed during time stepping are allocated here; subsequent code only
mutates them in place. After allocating the precomputed quantities, this calls
`set_precomputed_quantities!(Y, ..., 0)` once so that the cache is consistent
with the initial state, and then builds the component caches (hyperdiffusion,
gravity waves, radiation, tracers).

# Arguments

  - `Y`: Initial prognostic state, used for its spaces and element type.
  - `atmos`: The `AtmosModel` configuration.
  - `params`: The `ClimaAtmosParameters`.
  - `dt`: Simulation timestep [s].
  - `start_date`: Simulation start date, used for time-varying inputs and radiation.
  - `steady_state_velocity`: Predicted steady-state velocity for the
    `check_steady_state` diagnostic, or `nothing`.

# Returns

A fully initialized `AtmosCache`.
"""
function build_cache(
    Y,
    atmos,
    params,
    dt,
    start_date,
    steady_state_velocity,
)
    FT = eltype(params)
    dt = FT(dt)

    aerosol_names = atmos.radiation.aerosol_names
    time_varying_trace_gas_names = atmos.radiation.time_varying_trace_gases

    ᶜcoord = Fields.local_geometry_field(Y.c).coordinates
    ᶠcoord = Fields.local_geometry_field(Y.f).coordinates
    grav = FT(CAP.grav(params))
    ᶜΦ = geopotential.(grav, ᶜcoord.z)
    ᶠΦ = geopotential.(grav, ᶠcoord.z)

    (; ᶜf³, ᶠf¹²) = compute_coriolis(ᶜcoord, ᶠcoord, params)

    ghost_buffer =
        !do_dss(axes(Y.c)) ? (;) :
        (; c = Spaces.create_dss_buffer(Y.c), f = Spaces.create_dss_buffer(Y.f))

    net_energy_flux_toa = [Geometry.WVector(FT(0))]
    net_energy_flux_sfc = [Geometry.WVector(FT(0))]

    conservation_check =
        !(atmos.microphysics_model isa DryModel) ?
        (;
            col_integrated_precip_energy_tendency = zeros(
                axes(Fields.level(Geometry.WVector.(Y.f.u₃), half)),
            )
        ) : (; col_integrated_precip_energy_tendency = (;))

    sem_quasimonotone_limiter = if isnothing(atmos.numerics.limiter)
        nothing
    elseif atmos.numerics.limiter isa QuasiMonotoneLimiter
        Limiters.QuasiMonotoneLimiter(similar(Y.c, FT))
    end

    nonneg_lim = atmos.water.tracer_nonnegativity_method
    tracer_nonnegativity_limiter = if nonneg_lim isa TracerNonnegativityElementConstraint
        Limiters.QuasiMonotoneLimiter(similar(Y.c.ρq_tot, FT))
    else
        nothing
    end

    vertical_water_borrowing_limiter = nothing
    vertical_water_borrowing_species =
        atmos.numerics.vertical_water_borrowing_species

    if atmos.water.tracer_nonnegativity_method isa TracerNonnegativityVerticalWaterBorrowing
        vertical_water_borrowing_limiter = Limiters.VerticalMassBorrowingLimiter((FT(0.0),))
    end

    numerics = (;
        sem_quasimonotone_limiter,
        tracer_nonnegativity_limiter,
        vertical_water_borrowing_limiter,
        vertical_water_borrowing_species,
    )

    sfc_local_geometry =
        Fields.level(Fields.local_geometry_field(Y.f), Fields.half)

    core = (
        ᶜΦ,
        ᶠgradᵥ_ᶜΦ = ᶠgradᵥ.(ᶜΦ),
        ᶜgradᵥ_ᶠΦ = ᶜgradᵥ.(ᶠΦ),
        ᶜf³,
        ᶠf¹²,
        # Used by diagnostics such as hfres, evspblw
        surface_ct3_unit = CT3.(
            unit_basis_vector_data.(CT3, sfc_local_geometry),
        ),
    )
    external_forcing = external_forcing_cache(Y, atmos, params, start_date)
    sfc_setup = atmos.surface.boundary_overrides
    scratch = temporary_quantities(Y, atmos)

    precomputed = precomputed_quantities(Y, atmos)
    precomputing_arguments = (;
        atmos,
        core,
        params,
        sfc_setup,
        precomputed,
        scratch,
        dt,
        conservation_check,
        external_forcing,
    )

    # When flux_scheme is nothing, the surface conditions are entirely
    # supplied by an external driver, so we pre-fill safe defaults
    isnothing(atmos.surface.flux_scheme) &&
        SurfaceConditions.init_sfc_conditions_zero!(precomputing_arguments)

    set_precomputed_quantities!(Y, precomputing_arguments, FT(0))

    radiation_args =
        atmos.radiation_mode isa RRTMGPI.AbstractRRTMGPMode ?
        (
            start_date,
            params,
            aerosol_names,
            time_varying_trace_gas_names,
            atmos.insolation,
        ) : ()
    hyperdiff = hyperdiffusion_cache(Y, atmos)
    non_orographic_gravity_wave = non_orographic_gravity_wave_cache(Y, atmos)
    orographic_gravity_wave = orographic_gravity_wave_cache(Y, atmos)
    radiation = radiation_model_cache(Y, atmos, radiation_args...)
    tracers = tracer_cache(Y, aerosol_names, time_varying_trace_gas_names, start_date)

    args = (
        dt,
        atmos,
        numerics,
        params,
        core,
        sfc_setup,
        ghost_buffer,
        precomputed,
        scratch,
        hyperdiff,
        external_forcing,
        non_orographic_gravity_wave,
        orographic_gravity_wave,
        radiation,
        tracers,
        net_energy_flux_toa,
        net_energy_flux_sfc,
        steady_state_velocity,
        conservation_check,
    )

    return AtmosCache{map(typeof, args)...}(args...)
end


"""
    compute_coriolis(ᶜcoord, ᶠcoord, params) -> (; ᶜf³, ᶠf¹²)

Compute the Coriolis parameter fields for the given coordinate fields.

On a sphere, `ᶜf³` is the vertical Coriolis component `2Ω sin(lat)`; with deep
spherical geometry, the horizontal component `ᶠf¹²` is also nonzero. On a plane,
`ᶜf³` is the constant `f_plane_coriolis_frequency(params)` and `ᶠf¹²` is
`nothing`. Called from `build_cache`.
"""
function compute_coriolis(ᶜcoord, ᶠcoord, params)
    if eltype(ᶜcoord) <: Geometry.LatLongZPoint
        Ω = CAP.Omega(params)
        global_geom = Spaces.global_geometry(axes(ᶜcoord))
        if global_geom isa Geometry.DeepSphericalGlobalGeometry
            coriolis_deep(coord::Geometry.LatLongZPoint) = Geometry.LocalVector(
                Geometry.Cartesian123Vector(zero(Ω), zero(Ω), 2 * Ω),
                global_geom,
                coord,
            )
            ᶜf³ = @. CT3(CT123(coriolis_deep(ᶜcoord)))
            ᶠf¹² = @. CT12(CT123(coriolis_deep(ᶠcoord)))
        else
            coriolis_shallow(coord::Geometry.LatLongZPoint) =
                Geometry.WVector(2 * Ω * sind(coord.lat))
            ᶜf³ = @. CT3(coriolis_shallow(ᶜcoord))
            ᶠf¹² = nothing
        end
    else
        f = CAP.f_plane_coriolis_frequency(params)
        coriolis_f_plane(coord) = Geometry.WVector(f)
        ᶜf³ = @. CT3(coriolis_f_plane(ᶜcoord))
        ᶠf¹² = nothing
    end
    return (; ᶜf³, ᶠf¹²)
end
