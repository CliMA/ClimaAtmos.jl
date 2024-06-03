abstract type SurfaceSetup end

"""
    PrescribedSurface()

Used to indicate that there is no surface parameterization and
that the surface conditions will be explicitly prescribed (e.g. by the coupler).
The surface conditions must be set by calling `set_surface_conditions!`.
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
struct Nieuwstadt end
function (::Nieuwstadt)(params)
    FT = eltype(params)
    T = FT(300)
    p = FT(1e5)
    q_vap = FT(0)
    z0 = FT(0.16)
    θ_flux = FT(0.06)
    parameterization = MoninObukhov(; z0, fluxes = θAndQFluxes(; θ_flux))
    return SurfaceState(; parameterization, T, p, q_vap)
end

struct GABLS end
function (::GABLS)(params)
    FT = eltype(params)
    p = FT(1e5)
    q_vap = FT(0)
    z0 = FT(0.1)
    parameterization = MoninObukhov(; z0)
    surface_state(surface_coordinates, interior_z, t) = SurfaceState(;
        parameterization,
        T = 265 - FT(0.25) * FT(t) / 3600,
        p,
        q_vap,
    )
    return surface_state
end

struct BuoyancyDrivenBubble end
function (::BuoyancyDrivenBubble)(params)
    FT = eltype(params)
    p = FT(1e5)
    q_vap = FT(0.020)
    z0 = FT(0.1)
    θ_flux = FT(5e-3)
    q_flux = FT(1e-4)
    fluxes = θAndQFluxes(; θ_flux, q_flux)
    parameterization = MoninObukhov(; z0, fluxes)
    function surface_state(surface_coordinates, interior_z, t) 
        SurfaceState(;
            parameterization,
            T = 300 + 2 * (sin(π*surface_coordinates.x/1000))^2,
            p,
            q_vap,
        )
    end
    return surface_state
end

# TODO: The paper only specifies that T = 299.88. Where did all of these values
# come from?
struct GateIII end
function (::GateIII)(params)
    FT = eltype(params)
    T = FT(299.184)
    p = FT(101300)
    q_vap = FT(0.0165)
    Cd = FT(0.0012)
    Ch = FT(0.0034337)
    Cq = FT(0.0034337) # TODO: Add support for Cq to SF.Coefficients.
    parameterization = ExchangeCoefficients(Cd, Ch)
    return SurfaceState(; parameterization, T, p, q_vap)
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
    parameterization =
        MoninObukhov(; z0, fluxes = θAndQFluxes(; θ_flux, q_flux))
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

struct LifeCycleTan2018 end
function (::LifeCycleTan2018)(params)
    FT = eltype(params)
    T = FT(300.4)
    p = FT(101500)
    q_vap = FT(0.02245)
    θ_flux0 = FT(8e-3)
    q_flux0 = FT(5.2e-5)
    z0 = FT(1e-4)
    ustar = FT(0.28)
    function surface_state(surface_coordinates, interior_z, t)
        weight = FT(0.01) + FT(0.99) * (cos(2 * FT(π) * t / 3600) + 1) / 2
        fluxes =
            θAndQFluxes(; θ_flux = θ_flux0 * weight, q_flux = q_flux0 * weight)
        parameterization = MoninObukhov(; z0, fluxes, ustar)
        return SurfaceState(; parameterization, T, p, q_vap)
    end
    return surface_state
end

struct ARM_SGP end
function (::ARM_SGP)(params)
    FT = eltype(params)
    θ = FT(299)
    p = FT(97000)
    q_vap = FT(0.0152)
    t_data = FT[0, 4, 6.5, 7.5, 10, 12.5, 14.5] .* 3600
    shf_data = FT[-30, 90, 140, 140, 100, -10, -10]
    lhf_data = FT[5, 250, 450, 500, 420, 180, 0]
    z0 = FT(1e-4)
    ustar = FT(0.28) # 0.28 is taken from Bomex. TODO: Approximate from LES TKE.
    thermo_params = CAP.thermodynamics_params(params)
    ts = TD.PhaseNonEquil_pθq(thermo_params, p, θ, TD.PhasePartition(q_vap))
    T = TD.air_temperature(thermo_params, ts)
    shf = Intp.extrapolate(
        Intp.interpolate((t_data,), shf_data, Intp.Gridded(Intp.Linear())),
        Intp.Flat(),
    )
    lhf = Intp.extrapolate(
        Intp.interpolate(t_data, lhf_data, Intp.Gridded(Intp.Linear())),
        Intp.Flat(),
    )
    function surface_state(surface_coordinates, interior_z, t)
        fluxes = HeatFluxes(; shf = shf(t), lhf = lhf(t))
        parameterization = MoninObukhov(; z0, fluxes, ustar)
        return SurfaceState(; parameterization, T, p, q_vap)
    end
    return surface_state
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
    parameterization = MoninObukhov(; z0, fluxes = HeatFluxes(; shf, lhf))
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
        value = cos(FT(π) / 2 * (1 - FT(t) / (FT(5.25) * 3600)))
        shf = 270 * max(0, value)^FT(1.5)
        lhf = 554 * max(0, value)^FT(1.3)
        fluxes = HeatFluxes(; shf, lhf)
        parameterization = MoninObukhov(; z0, fluxes, ustar)
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
    θ_flux = FT(8)
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
    z0 = FT(1e-4)  # zrough
    parameterization = MoninObukhov(; z0)
    return SurfaceState(; parameterization, T)
end

function gcm_surface_conditions(external_forcing_file, cfsite_number)
    NC.NCDataset(external_forcing_file) do ds
        mean(gcm_driven_timeseries(ds.group[cfsite_number], "ts"))
    end
end
