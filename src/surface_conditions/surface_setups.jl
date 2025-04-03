abstract type SurfaceSetup end

"""
    PrescribedSurface()

Used to indicate that there is no surface parameterization and
that the surface conditions will be explicitly prescribed (e.g. by the coupler).
"""
struct PrescribedSurface <: SurfaceSetup end
(surface_setup::PrescribedSurface)(params) = nothing

"""
    DefaultMoninObukhov()

Monin-Obukhov surface, see the link below for more information
https://clima.github.io/SurfaceFluxes.jl/dev/SurfaceFluxes/#Monin-Obukhov-Similarity-Theory-(MOST)
"""
struct DefaultMoninObukhov <: SurfaceSetup end
function (::DefaultMoninObukhov)(params)
    FT = eltype(params)
    z0 = FT(1e-5)
    return SurfaceState(; parameterization = MoninObukhov(; z0))
end

"""
    DefaultExchangeCoefficients()

Bulk surface, parameterized only by a default exchange coefficient.
"""
struct DefaultExchangeCoefficients <: SurfaceSetup end
function (::DefaultExchangeCoefficients)(params)
    FT = eltype(params)
    C = params.C_H
    return SurfaceState(; parameterization = ExchangeCoefficients(C))
end

# All of the following are specific to TC cases

struct GABLS end
function (::GABLS)(params)
    FT = eltype(params)
    p = FT(1e5)
    q_vap = FT(0)
    z0 = FT(0.1)
    parameterization = MoninObukhov(; z0)
    function surface_state(surface_coordinates, interior_z, t)
        _FT = eltype(surface_coordinates) # do not capture FT
        SurfaceState(;
            parameterization,
            T = 265 - _FT(0.25) * _FT(t) / 3600,
            p,
            q_vap,
        )
    end
    return surface_state
end

struct Soares end
function (::Soares)(params)
    FT = eltype(params)
    T = FT(300)
    p = FT(1e5)
    q_vap = FT(5e-3)
    z0 = FT(0.16) # 0.16 is taken from the Nieuwstadt paper.
    θ_flux = FT(0.06)
    q_flux = FT(2.5e-5)
    ustar::FT = 0.28 # just to initilize grid mean covariances
    parameterization =
        MoninObukhov(; z0, fluxes = θAndQFluxes(; θ_flux, q_flux), ustar)
    return SurfaceState(; parameterization, T, p, q_vap)
end

struct Bomex end
function (::Bomex)(params)
    FT = eltype(params)
    T = FT(300.4)
    p = FT(101500)
    q_vap = FT(0.02245)
    θ_flux = FT(8e-3)
    q_flux = FT(5.2e-5)
    z0 = FT(1e-4)
    ustar = FT(0.28)
    fluxes = θAndQFluxes(; θ_flux, q_flux)
    parameterization = MoninObukhov(; z0, fluxes, ustar)
    return SurfaceState(; parameterization, T, p, q_vap)
end

struct DYCOMS_RF01 end
function (::DYCOMS_RF01)(params)
    FT = eltype(params)
    T = FT(292.5)
    p = FT(101780)
    q_vap = FT(0.01384) # 0.01384 is taken from Pycles. TODO: Compute qstar(T).
    z0 = FT(1e-4)
    shf = FT(15)
    lhf = FT(115)
    ustar = FT(0.25) # not specified in the literature, taken from DYCOMS_RF02.
    parameterization = MoninObukhov(; z0, fluxes = HeatFluxes(; shf, lhf), ustar)
    return SurfaceState(; parameterization, T, p, q_vap)
end

struct DYCOMS_RF02 end
function (::DYCOMS_RF02)(params)
    FT = eltype(params)
    T = FT(292.5)
    p = FT(101780)
    q_vap = FT(0.01384) # 0.01384 is taken from Pycles. TODO: Compute qstar(T).
    shf = FT(16)
    lhf = FT(93)
    z0 = FT(1e-4)
    ustar = FT(0.25)
    parameterization =
        MoninObukhov(; z0, fluxes = HeatFluxes(; shf, lhf), ustar)
    return SurfaceState(; parameterization, T, p, q_vap)
end

struct Rico end
function (::Rico)(params)
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    T_surface = FT(299.8)
    p_surface = FT(101540)
    z0 = FT(1.5e-4)
    Cd = FT(0.001229)
    Ch = FT(0.001094)
    Cq = FT(0.001133) # TODO: Add support for Cq to SF.Coefficients.
    # Saturated surface conditions for a given surface temperature and pressure
    p_sat_surface =
        TD.saturation_vapor_pressure(thermo_params, T_surface, TD.Liquid())
    ϵ_v = TD.Parameters.R_d(thermo_params) / TD.Parameters.R_v(thermo_params)
    q_surface = ϵ_v * p_sat_surface / (p_surface - p_sat_surface * (1 - ϵ_v))

    function surface_state(surface_coordinates, interior_z, t)
        # Adjust the coefficients from 20 m to the actual value of z.
        adjustment = (log(20 / z0) / log(interior_z / z0))^2
        parameterization =
            ExchangeCoefficients(Cd = Cd * adjustment, Ch = Ch * adjustment)
        SurfaceState(;
            parameterization,
            T = T_surface,
            p = p_surface,
            q_vap = q_surface,
        )
    end
    return surface_state
end

struct TRMM_LBA end
function (::TRMM_LBA)(params)
    FT = eltype(params)
    T = FT(296.85)  # 0C + 23.7
    p = FT(99130)
    q_vap = FT(0.02245)
    z0 = FT(1e-4)
    ustar = FT(0.28) # 0.28 is taken from Bomex. TODO: Approximate from LES TKE.
    function surface_state(surface_coordinates, interior_z, t)
        _FT = eltype(surface_coordinates) # do not capture FT
        value = cos(_FT(π) / 2 * (1 - _FT(t) / (_FT(5.25) * 3600)))
        shf = 270 * max(0, value)^_FT(1.5)
        lhf = 554 * max(0, value)^_FT(1.3)
        fluxes = HeatFluxes(; shf, lhf)
        parameterization = MoninObukhov(; z0, fluxes, ustar)
        return SurfaceState(; parameterization, T, p, q_vap)
    end
    return surface_state
end

struct ShipwayHill2012 end
function (::ShipwayHill2012)(params)
    function surface_state(surface_coordinates, interior_z, t)
        FT = eltype(surface_coordinates)
        T = FT(297.9)  # surface temperature (K)
        p = FT(100700)  # surface pressure (Pa)
        rv₀ = FT(0.015)  # water vapour mixing ratio at surface (kg/kg)
        q_vap = rv₀ / (1 + rv₀)  # specific humidity at surface, assuming unsaturated surface air (kg/kg)
        Cd = FT(0.0)
        Ch = FT(0.0)
        parameterization = ExchangeCoefficients(; Cd, Ch)
        return SurfaceState(; parameterization, T, p, q_vap)
    end
    return surface_state
end

struct SimplePlume end
function (::SimplePlume)(params)
    FT = eltype(params)
    T = FT(310)
    p = FT(101500)
    q_vap = FT(0.02245)
    θ_flux = FT(8e-2)
    q_flux = FT(0)
    z0 = FT(1e-4)
    ustar = FT(0.28)
    fluxes = θAndQFluxes(; θ_flux, q_flux)
    parameterization = MoninObukhov(; z0, fluxes, ustar)
    return SurfaceState(; parameterization, T, p, q_vap)
end

struct GCMDriven
    external_forcing_file::String
    cfsite_number::String
end
function (surface_setup::GCMDriven)(params)
    FT = eltype(params)
    (; external_forcing_file, cfsite_number) = surface_setup
    T = FT.(gcm_surface_conditions(external_forcing_file, cfsite_number))
    # z0 = FT(1e-4)  # zrough
    # parameterization = MoninObukhov(; z0)
    parameterization = MoninObukhov(; z0m = FT(1e-4), z0b = FT(3e-6))
    return SurfaceState(; parameterization, T, gustiness = FT(1.0))
end

function gcm_surface_conditions(external_forcing_file, cfsite_number)
    NC.NCDataset(external_forcing_file) do ds
        mean(gcm_driven_timeseries(ds.group[cfsite_number], "ts"))
    end
end

struct ReanalysisTimeVarying end


function (surface_setup::ReanalysisTimeVarying)(params)
    FT = eltype(params)
    z0 = FT(1e-4)  # zrough
    parameterization = MoninObukhov(; z0)
    return SurfaceState(; parameterization)
end


struct ISDAC end
function (::ISDAC)(params)
    FT = eltype(params)
    T = FT(267)  # K
    p = FT(102000)  # Pa
    z0 = FT(4e-4)  # m  surface roughness length
    parameterization = MoninObukhov(; z0)
    return SurfaceState(; parameterization, T, p)
end
