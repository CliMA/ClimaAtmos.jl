struct AtmosCache{
    TT,
    AM,
    NUM,
    CAP,
    COR,
    SFC,
    GHOST,
    PREC,
    SCRA,
    HYPE,
    PR,
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
    """Timestep of the simulation (in seconds). This is also used by callbacks and tendencies"""
    dt::TT

    """AtmosModel"""
    atmos::AM

    """Limiter"""
    numerics::NUM

    """ClimaAtmosParameters that have to be used"""
    params::CAP

    """Variables that are used generally, such as ᶜΦ"""
    core::COR

    """Used by update_surface_conditions! in set_precomputed_quantities! and coupler"""
    sfc_setup::SFC

    """Center and face ghost buffers used by DSS"""
    ghost_buffer::GHOST

    """Quantities that are updated with set_precomputed_quantities!"""
    precomputed::PREC

    """Pre-allocated areas of memory to store temporary values"""
    scratch::SCRA

    """Hyperdiffision quantities for grid and subgrid scale quantities, potentially with
       ghost buffers for DSS"""
    hyperdiff::HYPE

    """Additional parameters used by the various tendencies"""
    precipitation::PR
    external_forcing::EXTFORCING
    non_orographic_gravity_wave::NONGW
    orographic_gravity_wave::ORGW
    radiation::RAD
    tracers::TRAC

    """Net energy flux coming through top of atmosphere and surface"""
    net_energy_flux_toa::NETFLUXTOA
    net_energy_flux_sfc::NETFLUXSFC

    """Predicted steady-state velocity, if `check_steady_state` is `true`"""
    steady_state_velocity::SSV

    """Conservation check for prognostic surface temperature"""
    conservation_check::CONSCHECK
end

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
function build_cache(
    Y,
    atmos,
    params,
    surface_setup,
    sim_info,
    aerosol_names,
    steady_state_velocity,
)
    (; dt, start_date, output_dir) = sim_info
    FT = eltype(params)

    ᶜcoord = Fields.local_geometry_field(Y.c).coordinates
    ᶠcoord = Fields.local_geometry_field(Y.f).coordinates
    grav = FT(CAP.grav(params))
    ᶜΦ = grav .* ᶜcoord.z
    ᶠΦ = grav .* ᶠcoord.z

    (; ᶜf³, ᶠf¹²) = compute_coriolis(ᶜcoord, ᶠcoord, params)

    ghost_buffer =
        !do_dss(axes(Y.c)) ? (;) :
        (; c = Spaces.create_dss_buffer(Y.c), f = Spaces.create_dss_buffer(Y.f))

    net_energy_flux_toa = [Geometry.WVector(FT(0))]
    net_energy_flux_sfc = [Geometry.WVector(FT(0))]

    conservation_check =
        !(atmos.precip_model isa NoPrecipitation) ?
        (;
            col_integrated_precip_energy_tendency = zeros(
                axes(Fields.level(Geometry.WVector.(Y.f.u₃), half)),
            )
        ) : (; col_integrated_precip_energy_tendency = (;))

    limiter = if isnothing(atmos.numerics.limiter)
        nothing
    elseif atmos.numerics.limiter isa QuasiMonotoneLimiter
        Limiters.QuasiMonotoneLimiter(similar(Y.c, FT))
    end

    numerics = (; limiter)

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
            unit_basis_vector_data.(CT3, sfc_local_geometry)
        ),
    )

    sfc_setup = surface_setup(params)
    scratch = temporary_quantities(Y, atmos)

    precomputed = precomputed_quantities(Y, atmos)
    precomputing_arguments =
        (; atmos, core, params, sfc_setup, precomputed, scratch, dt)

    # Coupler compatibility
    isnothing(precomputing_arguments.sfc_setup) &&
        SurfaceConditions.set_dummy_surface_conditions!(precomputing_arguments)

    set_precomputed_quantities!(Y, precomputing_arguments, FT(0))

    radiation_args =
        atmos.radiation_mode isa RRTMGPI.AbstractRRTMGPMode ?
        (
            start_date,
            params,
            atmos.ozone,
            atmos.co2,
            aerosol_names,
            atmos.insolation,
        ) : ()

    hyperdiff = hyperdiffusion_cache(Y, atmos)
    precipitation = precipitation_cache(Y, atmos)
    external_forcing = external_forcing_cache(Y, atmos, params)
    non_orographic_gravity_wave = non_orographic_gravity_wave_cache(Y, atmos)
    orographic_gravity_wave = orographic_gravity_wave_cache(Y, atmos)
    radiation = radiation_model_cache(Y, atmos, radiation_args...)
    tracers = tracer_cache(Y, atmos, aerosol_names, start_date)

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
        precipitation,
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


function compute_coriolis(ᶜcoord, ᶠcoord, params)
    if eltype(ᶜcoord) <: Geometry.LatLongZPoint
        Ω = CAP.Omega(params)
        global_geom = Spaces.global_geometry(axes(ᶜcoord))
        if global_geom isa Geometry.DeepSphericalGlobalGeometry
            @info "using deep atmosphere"
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
