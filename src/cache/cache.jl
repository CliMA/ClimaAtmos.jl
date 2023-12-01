struct AtmosCache{
    FT <: AbstractFloat,
    SD,
    AM,
    NUM,
    CAP,
    COR,
    SFC,
    GHOST,
    ENV,
    PREC,
    SCRA,
    HYPE,
    DSS,
    RS,
    VS,
    PR,
    SUB,
    LSAD,
    EDMFCOR,
    FOR,
    NONGW,
    ORGW,
    RAD,
    NETFLUXTOA,
    NETFLUXSFC,
}
    """Timestep of the simulation (in seconds). This is also used by callbacks and tendencies"""
    dt::FT

    """Start date (used for insolation)."""
    start_date::SD

    """AtmosModel"""
    atmos::AM

    """Limiter"""
    numerics::NUM

    """ClimaAtmosParameters that have to be used"""
    params::CAP

    """Variables that are used generally, such as ᶜρ_ref, ᶜΦ"""
    core::COR

    """Used by update_surface_conditions! in set_precomputed_quantities! and coupler"""
    sfc_setup::SFC

    """Center and face ghost buffers used by DSS"""
    ghost_buffer::GHOST

    env_thermo_quad::ENV

    """Quantities that are updated with set_precomputed_quantities!"""
    precomputed::PREC

    """Pre-allocated areas of memory to store temporary values"""
    scratch::SCRA

    """Hyperdiffision quantities for grid and subgrid scale quantities, potentially with
       ghost buffers for DSS"""
    hyperdiff::HYPE

    do_dss::DSS

    """Additional parameters used by the various tendencies"""
    rayleigh_sponge::RS
    viscous_sponge::VS
    precipitation::PR
    subsidence::SUB
    large_scale_advection::LSAD
    edmf_coriolis::EDMFCOR
    forcing::FOR
    non_orographic_gravity_wave::NONGW
    orographic_gravity_wave::ORGW
    radiation::RAD

    """Net energy flux coming through top of atmosphere and surface"""
    net_energy_flux_toa::NETFLUXTOA
    net_energy_flux_sfc::NETFLUXSFC
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
function build_cache(Y, atmos, params, surface_setup, dt, start_date)
    FT = eltype(params)

    ᶜcoord = Fields.local_geometry_field(Y.c).coordinates
    grav = FT(CAP.grav(params))
    ᶜΦ = grav .* ᶜcoord.z

    if atmos.numerics.use_reference_state
        R_d = FT(CAP.R_d(params))
        MSLP = FT(CAP.MSLP(params))
        T_ref = FT(255)
        ᶜρ_ref = @. MSLP * exp(-grav * ᶜcoord.z / (R_d * T_ref)) / (R_d * T_ref)
        ᶜp_ref = @. ᶜρ_ref * R_d * T_ref
    else
        ᶜρ_ref = zero(ᶜΦ)
        ᶜp_ref = zero(ᶜΦ)
    end

    if eltype(ᶜcoord) <: Geometry.LatLongZPoint
        Ω = CAP.Omega(params)
        ᶜf = @. 2 * Ω * sind(ᶜcoord.lat)
    else
        f = CAP.f_plane_coriolis_frequency(params)
        ᶜf = map(_ -> f, ᶜcoord)
    end
    ᶜf = @. CT3(Geometry.WVector(ᶜf))

    quadrature_style = Spaces.horizontal_space(axes(Y.c)).quadrature_style
    do_dss = quadrature_style isa Spaces.Quadratures.GLL
    ghost_buffer =
        !do_dss ? (;) :
        (; c = Spaces.create_dss_buffer(Y.c), f = Spaces.create_dss_buffer(Y.f))

    net_energy_flux_toa = [Geometry.WVector(FT(0))]
    net_energy_flux_sfc = [Geometry.WVector(FT(0))]

    limiter =
        isnothing(atmos.numerics.limiter) ? nothing :
        atmos.numerics.limiter(similar(Y.c, FT))

    numerics = (; limiter)

    sfc_local_geometry =
        Fields.level(Fields.local_geometry_field(Y.f), Fields.half)

    core = (
        ᶜΦ,
        ᶠgradᵥ_ᶜΦ = ᶠgradᵥ.(ᶜΦ),
        ᶜρ_ref,
        ᶜp_ref,
        ᶜT = similar(Y.c, FT),
        ᶜf,
        ∂ᶜK_∂ᶠu₃ = similar(Y.c, BidiagonalMatrixRow{Adjoint{FT, CT3{FT}}}),
        # Used by diagnostics such as hfres, evspblw
        surface_ct3_unit = CT3.(
            unit_basis_vector_data.(CT3, sfc_local_geometry)
        ),
    )

    sfc_setup = surface_setup(params)
    scratch = temporary_quantities(Y, atmos)
    env_thermo_quad = SGSQuadrature(FT)

    precomputed = precomputed_quantities(Y, atmos)
    precomputing_arguments =
        (; atmos, core, params, sfc_setup, precomputed, scratch, dt)

    # Coupler compatibility
    isnothing(precomputing_arguments.sfc_setup) &&
        SurfaceConditions.set_dummy_surface_conditions!(precomputing_arguments)

    set_precomputed_quantities!(Y, precomputing_arguments, FT(0))

    radiation_args =
        atmos.radiation_mode isa RRTMGPI.AbstractRRTMGPMode ?
        (params, precomputed.ᶜp) : ()

    hyperdiff = hyperdiffusion_cache(Y, atmos)
    rayleigh_sponge = rayleigh_sponge_cache(Y, atmos)
    viscous_sponge = viscous_sponge_cache(Y, atmos)
    precipitation = precipitation_cache(Y, atmos)
    subsidence = subsidence_cache(Y, atmos)
    large_scale_advection = large_scale_advection_cache(Y, atmos)
    edmf_coriolis = edmf_coriolis_cache(Y, atmos)
    forcing = forcing_cache(Y, atmos)
    non_orographic_gravity_wave = non_orographic_gravity_wave_cache(Y, atmos)
    orographic_gravity_wave = orographic_gravity_wave_cache(Y, atmos)
    radiation = radiation_model_cache(Y, atmos, radiation_args...)

    args = (
        dt,
        start_date,
        atmos,
        numerics,
        params,
        core,
        sfc_setup,
        ghost_buffer,
        env_thermo_quad,
        precomputed,
        scratch,
        hyperdiff,
        do_dss,
        rayleigh_sponge,
        viscous_sponge,
        precipitation,
        subsidence,
        large_scale_advection,
        edmf_coriolis,
        forcing,
        non_orographic_gravity_wave,
        orographic_gravity_wave,
        radiation,
        net_energy_flux_toa,
        net_energy_flux_sfc,
    )

    return AtmosCache{map(typeof, args)...}(args...)
end
