#####
##### Precomputed quantities for precipitation processes
#####

import CloudMicrophysics.MicrophysicsNonEq as CMNe
import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics.Microphysics2M as CM2
import CloudMicrophysics.P3Scheme as CMP3

import Thermodynamics as TD
import ClimaCore.Operators as Operators
import ClimaCore.Fields as Fields

const Iₗ = TD.internal_energy_liquid
const Iᵢ = TD.internal_energy_ice

"""
   Kin(ᶜw_precip, ᶜu_air)

    - ᶜw_precip - teminal velocity of cloud consensate or precipitation
    - ᶜu_air - air velocity

Helper function to compute the kinetic energy of cloud condensate and
precipitation.
"""
function Kin(ᶜw_precip, ᶜu_air)
    return @. lazy(
        norm_sqr(
            Geometry.UVWVector(0, 0, -(ᶜw_precip)) + Geometry.UVWVector(ᶜu_air),
        ) / 2,
    )
end

"""
    Smallest mass value that is different than zero for the purpose of mass_weigthed
    averaging of terminal velocities.
"""
ϵ_numerics(FT) = sqrt(floatmin(FT))

"""
    set_precipitation_velocities!(Y, p, moisture_model, microphysics_model, turbconv_model)

Updates the precipitation terminal velocity, cloud sedimentation velocity,
and their contribution to total water and energy advection.
"""
function set_precipitation_velocities!(Y, p, _, _, _)
    (; ᶜwₜqₜ, ᶜwₕhₜ) = p.precomputed
    @. ᶜwₜqₜ = Geometry.WVector(0)
    @. ᶜwₕhₜ = Geometry.WVector(0)
    return nothing
end
function set_precipitation_velocities!(Y, p, ::NonEquilMoistModel, ::Microphysics1Moment, _)
    (; ᶜwₗ, ᶜwᵢ, ᶜwᵣ, ᶜwₛ, ᶜwₜqₜ, ᶜwₕhₜ, ᶜts, ᶜu) = p.precomputed
    (; ᶜΦ) = p.core
    cmc = CAP.microphysics_cloud_params(p.params)
    cmp = CAP.microphysics_1m_params(p.params)
    thp = CAP.thermodynamics_params(p.params)

    # compute the precipitation terminal velocity [m/s]
    @. ᶜwᵣ = CM1.terminal_velocity(
        cmp.pr,
        cmp.tv.rain,
        Y.c.ρ,
        max(zero(Y.c.ρ), Y.c.ρq_rai / Y.c.ρ),
    )
    @. ᶜwₛ = CM1.terminal_velocity(
        cmp.ps,
        cmp.tv.snow,
        Y.c.ρ,
        max(zero(Y.c.ρ), Y.c.ρq_sno / Y.c.ρ),
    )
    # compute sedimentation velocity for cloud condensate [m/s]
    @. ᶜwₗ = CMNe.terminal_velocity(
        cmc.liquid,
        cmc.Ch2022.rain,
        Y.c.ρ,
        max(zero(Y.c.ρ), Y.c.ρq_liq / Y.c.ρ),
    )
    @. ᶜwᵢ = CMNe.terminal_velocity(
        cmc.ice,
        cmc.Ch2022.small_ice,
        Y.c.ρ,
        max(zero(Y.c.ρ), Y.c.ρq_ice / Y.c.ρ),
    )

    # compute their contributions to energy and total water advection
    @. ᶜwₜqₜ =
        Geometry.WVector(
            ᶜwₗ * Y.c.ρq_liq +
            ᶜwᵢ * Y.c.ρq_ice +
            ᶜwᵣ * Y.c.ρq_rai +
            ᶜwₛ * Y.c.ρq_sno,
        ) / Y.c.ρ
    @. ᶜwₕhₜ =
        Geometry.WVector(
            ᶜwₗ * Y.c.ρq_liq * (Iₗ(thp, ᶜts) + ᶜΦ + $(Kin(ᶜwₗ, ᶜu))) +
            ᶜwᵢ * Y.c.ρq_ice * (Iᵢ(thp, ᶜts) + ᶜΦ + $(Kin(ᶜwᵢ, ᶜu))) +
            ᶜwᵣ * Y.c.ρq_rai * (Iₗ(thp, ᶜts) + ᶜΦ + $(Kin(ᶜwᵣ, ᶜu))) +
            ᶜwₛ * Y.c.ρq_sno * (Iᵢ(thp, ᶜts) + ᶜΦ + $(Kin(ᶜwₛ, ᶜu))),
        ) / Y.c.ρ
    return nothing
end
function set_precipitation_velocities!(Y, p,
    ::NonEquilMoistModel, ::Microphysics1Moment, turbconv_model::PrognosticEDMFX,
)
    (; ᶜwₗ, ᶜwᵢ, ᶜwᵣ, ᶜwₛ, ᶜwₜqₜ, ᶜwₕhₜ) = p.precomputed
    (; ᶜΦ) = p.core
    (; ᶜwₗʲs, ᶜwᵢʲs, ᶜwᵣʲs, ᶜwₛʲs) = p.precomputed
    (; ᶜts⁰, ᶜtsʲs) = p.precomputed
    cmc = CAP.microphysics_cloud_params(p.params)
    cmp = CAP.microphysics_1m_params(p.params)
    thp = CAP.thermodynamics_params(p.params)
    FT = eltype(p.params)

    ᶜρ⁰ = @. lazy(TD.air_density(thp, ᶜts⁰))
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
    n = n_mass_flux_subdomains(turbconv_model)

    # scratch to compute env mass flux
    ᶜimplied_env_mass_flux = p.scratch.ᶜtemp_scalar
    # scratch to add positive masses of subdomains
    # TODO use Y.c.ρq instead of ᶜρχ for computing gs velocities by averaging velocities 
    # over subdomains once negative subdomain mass issues are resolved
    # We use positive masses for mass-weighted averaging gs terminal velocity. This ensures
    # that the value remains between sgs terminal velocity values (important for stability).
    ᶜρχ = p.scratch.ᶜtemp_scalar_2
    # scratch for adding energy fluxes over subdomains
    ᶜρwₕhₜ = p.scratch.ᶜtemp_scalar_3

    # Compute gs sedimentation velocity based on subdomain velocities (assuming gs flux 
    # equals sum of sgs fluxes)

    ᶜq_liq⁰ = ᶜspecific_env_value(@name(q_liq), Y, p)
    ᶜq_ice⁰ = ᶜspecific_env_value(@name(q_ice), Y, p)
    ᶜq_rai⁰ = ᶜspecific_env_value(@name(q_rai), Y, p)
    ᶜq_sno⁰ = ᶜspecific_env_value(@name(q_sno), Y, p)

    # Cloud liquid
    ᶜρa⁰χ⁰ = @. lazy(max(zero(Y.c.ρ), ᶜρa⁰) * max(zero(Y.c.ρ), ᶜq_liq⁰))
    @. ᶜρχ = ᶜρa⁰χ⁰
    @. ᶜwₗ = ᶜρa⁰χ⁰ * CMNe.terminal_velocity(
        cmc.liquid,
        cmc.Ch2022.rain,
        ᶜρ⁰,
        ᶜq_liq⁰,
    )
    @. ᶜimplied_env_mass_flux = 0
    # add updraft contributions
    for j in 1:n
        ᶜρaʲχʲ = @. lazy(
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).ρa) *
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_liq),
        )
        @. ᶜρχ += ᶜρaʲχʲ
        @. ᶜwₗ += ᶜρaʲχʲ * ᶜwₗʲs.:($$j)
        @. ᶜimplied_env_mass_flux -=
            Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_liq * ᶜwₗʲs.:($$j)
    end
    # average
    @. ᶜwₗ = ifelse(ᶜρχ > ϵ_numerics(FT), ᶜwₗ / ᶜρχ, FT(0))
    @. ᶜimplied_env_mass_flux += Y.c.ρq_liq * ᶜwₗ
    # contribution of env q_liq sedimentation to htot
    @. ᶜρwₕhₜ = ᶜimplied_env_mass_flux * (Iₗ(thp, ᶜts⁰) + ᶜΦ)

    # Cloud ice
    ᶜρa⁰χ⁰ = @. lazy(max(zero(Y.c.ρ), ᶜρa⁰) * max(zero(Y.c.ρ), ᶜq_ice⁰))
    @. ᶜρχ = ᶜρa⁰χ⁰
    @. ᶜwᵢ = ᶜρa⁰χ⁰ * CMNe.terminal_velocity(
        cmc.ice,
        cmc.Ch2022.small_ice,
        ᶜρ⁰,
        ᶜq_ice⁰,
    )
    @. ᶜimplied_env_mass_flux = 0
    # add updraft contributions
    for j in 1:n
        ᶜρaʲχʲ = @. lazy(
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).ρa) *
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_ice),
        )
        @. ᶜρχ += ᶜρaʲχʲ
        @. ᶜwᵢ += ᶜρaʲχʲ * ᶜwᵢʲs.:($$j)
        @. ᶜimplied_env_mass_flux -=
            Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_ice * ᶜwᵢʲs.:($$j)
    end
    # average
    @. ᶜwᵢ = ifelse(ᶜρχ > ϵ_numerics(FT), ᶜwᵢ / ᶜρχ, FT(0))
    @. ᶜimplied_env_mass_flux += Y.c.ρq_ice * ᶜwᵢ
    # contribution of env q_liq sedimentation to htot
    @. ᶜρwₕhₜ += ᶜimplied_env_mass_flux * (Iᵢ(thp, ᶜts⁰) + ᶜΦ)

    # Rain
    ᶜρa⁰χ⁰ = @. lazy(max(zero(Y.c.ρ), ᶜρa⁰) * max(zero(Y.c.ρ), ᶜq_rai⁰))
    @. ᶜρχ = ᶜρa⁰χ⁰
    @. ᶜwᵣ = ᶜρa⁰χ⁰ * CM1.terminal_velocity(
        cmp.pr,
        cmp.tv.rain,
        ᶜρ⁰,
        ᶜq_rai⁰,
    )
    @. ᶜimplied_env_mass_flux = 0
    # add updraft contributions
    for j in 1:n
        ᶜρaʲχʲ = @. lazy(
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).ρa) *
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_rai),
        )
        @. ᶜρχ += ᶜρaʲχʲ
        @. ᶜwᵣ += ᶜρaʲχʲ * ᶜwᵣʲs.:($$j)
        @. ᶜimplied_env_mass_flux -=
            Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_rai * ᶜwᵣʲs.:($$j)
    end
    # average
    @. ᶜwᵣ = ifelse(ᶜρχ > ϵ_numerics(FT), ᶜwᵣ / ᶜρχ, FT(0))
    @. ᶜimplied_env_mass_flux += Y.c.ρq_rai * ᶜwᵣ
    # contribution of env q_liq sedimentation to htot
    @. ᶜρwₕhₜ += ᶜimplied_env_mass_flux * (Iₗ(thp, ᶜts⁰) + ᶜΦ)

    # Snow
    ᶜρa⁰χ⁰ = @. lazy(max(zero(Y.c.ρ), ᶜρa⁰) * max(zero(Y.c.ρ), ᶜq_sno⁰))
    @. ᶜρχ = ᶜρa⁰χ⁰
    @. ᶜwₛ = ᶜρa⁰χ⁰ * CM1.terminal_velocity(
        cmp.ps,
        cmp.tv.snow,
        ᶜρ⁰,
        ᶜq_sno⁰,
    )
    @. ᶜimplied_env_mass_flux = 0
    # add updraft contributions
    for j in 1:n
        ᶜρaʲχʲ = @. lazy(
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).ρa) *
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_sno),
        )
        @. ᶜρχ += ᶜρaʲχʲ
        @. ᶜwₛ += ᶜρaʲχʲ * ᶜwₛʲs.:($$j)
        @. ᶜimplied_env_mass_flux -=
            Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_sno * ᶜwₛʲs.:($$j)
    end
    # average
    @. ᶜwₛ = ifelse(ᶜρχ > ϵ_numerics(FT), ᶜwₛ / ᶜρχ, FT(0))
    @. ᶜimplied_env_mass_flux += Y.c.ρq_sno * ᶜwₛ
    # contribution of env q_liq sedimentation to htot
    @. ᶜρwₕhₜ += ᶜimplied_env_mass_flux * (Iᵢ(thp, ᶜts⁰) + ᶜΦ)

    # Add contributions to energy and total water advection
    # TODO do we need to add kinetic energy of subdomains?
    for j in 1:n
        @. ᶜρwₕhₜ +=
            Y.c.sgsʲs.:($$j).ρa *
            (
                ᶜwₗʲs.:($$j) * Y.c.sgsʲs.:($$j).q_liq * (Iₗ(thp, ᶜtsʲs.:($$j)) + ᶜΦ) +
                ᶜwᵢʲs.:($$j) * Y.c.sgsʲs.:($$j).q_ice * (Iᵢ(thp, ᶜtsʲs.:($$j)) + ᶜΦ) +
                ᶜwᵣʲs.:($$j) * Y.c.sgsʲs.:($$j).q_rai * (Iₗ(thp, ᶜtsʲs.:($$j)) + ᶜΦ) +
                ᶜwₛʲs.:($$j) * Y.c.sgsʲs.:($$j).q_sno * (Iᵢ(thp, ᶜtsʲs.:($$j)) + ᶜΦ)
            )
    end
    @. ᶜwₕhₜ = Geometry.WVector(ᶜρwₕhₜ) / Y.c.ρ

    @. ᶜwₜqₜ =
        Geometry.WVector(
            ᶜwₗ * Y.c.ρq_liq +
            ᶜwᵢ * Y.c.ρq_ice +
            ᶜwᵣ * Y.c.ρq_rai +
            ᶜwₛ * Y.c.ρq_sno,
        ) / Y.c.ρ

    return nothing
end
function set_precipitation_velocities!(
    Y,
    p,
    moisture_model::NonEquilMoistModel,
    microphysics_model::Microphysics2Moment,
    _,
)
    (; ᶜwₗ, ᶜwᵢ, ᶜwᵣ, ᶜwₛ, ᶜwₙₗ, ᶜwₙᵣ, ᶜwₜqₜ, ᶜwₕhₜ, ᶜts, ᶜu) = p.precomputed
    (; ᶜΦ) = p.core

    cmc = CAP.microphysics_cloud_params(p.params)
    cm1p = CAP.microphysics_1m_params(p.params)
    cm2p = CAP.microphysics_2m_params(p.params)
    thp = CAP.thermodynamics_params(p.params)

    # compute the precipitation terminal velocity [m/s]
    # TODO sedimentation of snow is based on the 1M scheme
    @. ᶜwₙᵣ = getindex(
        CM2.rain_terminal_velocity(
            cm2p.sb,
            cm2p.rtv,
            max(zero(Y.c.ρ), specific(Y.c.ρq_rai, Y.c.ρ)),
            Y.c.ρ,
            max(zero(Y.c.ρ), Y.c.ρn_rai),
        ),
        1,
    )
    @. ᶜwᵣ = getindex(
        CM2.rain_terminal_velocity(
            cm2p.sb,
            cm2p.rtv,
            max(zero(Y.c.ρ), specific(Y.c.ρq_rai, Y.c.ρ)),
            Y.c.ρ,
            max(zero(Y.c.ρ), Y.c.ρn_rai),
        ),
        2,
    )
    @. ᶜwₛ = CM1.terminal_velocity(
        cm1p.ps,
        cm1p.tv.snow,
        Y.c.ρ,
        max(zero(Y.c.ρ), specific(Y.c.ρq_sno, Y.c.ρ)),
    )
    # compute sedimentation velocity for cloud condensate [m/s]
    # TODO sedimentation of ice is based on the 1M scheme
    @. ᶜwₙₗ = getindex(
        CM2.cloud_terminal_velocity(
            cm2p.sb.pdf_c,
            cm2p.ctv,
            max(zero(Y.c.ρ), specific(Y.c.ρq_liq, Y.c.ρ)),
            Y.c.ρ,
            max(zero(Y.c.ρ), Y.c.ρn_liq),
        ),
        1,
    )
    @. ᶜwₗ = getindex(
        CM2.cloud_terminal_velocity(
            cm2p.sb.pdf_c,
            cm2p.ctv,
            max(zero(Y.c.ρ), specific(Y.c.ρq_liq, Y.c.ρ)),
            Y.c.ρ,
            max(zero(Y.c.ρ), Y.c.ρn_liq),
        ),
        2,
    )
    @. ᶜwᵢ = CMNe.terminal_velocity(
        cmc.ice,
        cmc.Ch2022.small_ice,
        Y.c.ρ,
        max(zero(Y.c.ρ), specific(Y.c.ρq_ice, Y.c.ρ)),
    )

    # compute their contributions to energy and total water advection
    @. ᶜwₜqₜ =
        Geometry.WVector(
            ᶜwₗ * Y.c.ρq_liq +
            ᶜwᵢ * Y.c.ρq_ice +
            ᶜwᵣ * Y.c.ρq_rai +
            ᶜwₛ * Y.c.ρq_sno,
        ) / Y.c.ρ
    @. ᶜwₕhₜ =
        Geometry.WVector(
            ᶜwₗ * Y.c.ρq_liq * (Iₗ(thp, ᶜts) + ᶜΦ + $(Kin(ᶜwₗ, ᶜu))) +
            ᶜwᵢ * Y.c.ρq_ice * (Iᵢ(thp, ᶜts) + ᶜΦ + $(Kin(ᶜwᵢ, ᶜu))) +
            ᶜwᵣ * Y.c.ρq_rai * (Iₗ(thp, ᶜts) + ᶜΦ + $(Kin(ᶜwᵣ, ᶜu))) +
            ᶜwₛ * Y.c.ρq_sno * (Iᵢ(thp, ᶜts) + ᶜΦ + $(Kin(ᶜwₛ, ᶜu))),
        ) / Y.c.ρ
    return nothing
end
function set_precipitation_velocities!(
    Y,
    p,
    moisture_model::NonEquilMoistModel,
    microphysics_model::Microphysics2Moment,
    turbconv_model::PrognosticEDMFX,
)
    (; ᶜwₗ, ᶜwᵢ, ᶜwᵣ, ᶜwₛ, ᶜwₙₗ, ᶜwₙᵣ, ᶜwₜqₜ, ᶜwₕhₜ) = p.precomputed
    (; ᶜΦ) = p.core
    (; ᶜwₗʲs, ᶜwᵢʲs, ᶜwᵣʲs, ᶜwₛʲs, ᶜwₙₗʲs, ᶜwₙᵣʲs) = p.precomputed
    (; ᶜts⁰, ᶜtsʲs) = p.precomputed
    cmc = CAP.microphysics_cloud_params(p.params)
    cm1p = CAP.microphysics_1m_params(p.params)
    cm2p = CAP.microphysics_2m_params(p.params)
    thp = CAP.thermodynamics_params(p.params)
    FT = eltype(p.params)

    ᶜρ⁰ = @. lazy(TD.air_density(thp, ᶜts⁰))
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
    n = n_mass_flux_subdomains(turbconv_model)

    # scratch to compute env velocities
    ᶜw⁰ = p.scratch.ᶜtemp_scalar
    # scratch to add positive masses of subdomains
    # TODO use Y.c.ρq instead of ᶜρχ for computing gs velocities by averaging velocities 
    # over subdomains once negative subdomain mass issues are resolved 
    ᶜρχ = p.scratch.ᶜtemp_scalar_2
    # scratch for adding energy fluxes over subdomains
    ᶜρwₕhₜ = p.scratch.ᶜtemp_scalar_3

    # Compute gs sedimentation velocity based on subdomain velocities (assuming gs flux 
    # equals sum of sgs fluxes)

    ᶜq_liq⁰ = ᶜspecific_env_value(@name(q_liq), Y, p)
    ᶜq_ice⁰ = ᶜspecific_env_value(@name(q_ice), Y, p)
    ᶜq_rai⁰ = ᶜspecific_env_value(@name(q_rai), Y, p)
    ᶜq_sno⁰ = ᶜspecific_env_value(@name(q_sno), Y, p)
    ᶜn_liq⁰ = ᶜspecific_env_value(@name(n_liq), Y, p)
    ᶜn_rai⁰ = ᶜspecific_env_value(@name(n_rai), Y, p)

    # Cloud liquid (number)
    @. ᶜw⁰ = getindex(
        CM2.rain_terminal_velocity(
            cm2p.sb,
            cm2p.rtv,
            max(zero(Y.c.ρ), ᶜq_liq⁰),
            ᶜρ⁰,
            max(zero(Y.c.ρ), ᶜn_liq⁰),
        ),
        1,
    )
    @. ᶜwₙₗ = ᶜρa⁰ * ᶜn_liq⁰ * ᶜw⁰
    @. ᶜρχ = max(zero(Y.c.ρ), ᶜρa⁰ * ᶜn_liq⁰)
    for j in 1:n
        @. ᶜwₙₗ += Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).n_liq * ᶜwₙₗʲs.:($$j)
        @. ᶜρχ += max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).n_liq)
    end
    @. ᶜwₙₗ = ifelse(ᶜρχ > FT(0), ᶜwₙₗ / ᶜρχ, FT(0))

    # Cloud liquid (mass)
    @. ᶜw⁰ = getindex(
        CM2.rain_terminal_velocity(
            cm2p.sb,
            cm2p.rtv,
            max(zero(Y.c.ρ), ᶜq_liq⁰),
            ᶜρ⁰,
            max(zero(Y.c.ρ), ᶜn_liq⁰),
        ),
        2,
    )
    @. ᶜwₗ = ᶜρa⁰ * ᶜq_liq⁰ * ᶜw⁰
    @. ᶜρχ = max(zero(Y.c.ρ), ᶜρa⁰ * ᶜq_liq⁰)
    for j in 1:n
        @. ᶜwₗ += Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_liq * ᶜwₗʲs.:($$j)
        @. ᶜρχ += max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_liq)
    end
    @. ᶜwₗ = ifelse(ᶜρχ > FT(0), ᶜwₗ / ᶜρχ, FT(0))

    # contribution of env cloud liquid advection to htot advection
    @. ᶜρwₕhₜ = ᶜρa⁰ * ᶜq_liq⁰ * ᶜw⁰ * (Iₗ(thp, ᶜts⁰) + ᶜΦ)

    # Cloud ice
    # TODO sedimentation of ice is based on the 1M scheme
    @. ᶜw⁰ = CMNe.terminal_velocity(
        cmc.ice,
        cmc.Ch2022.small_ice,
        ᶜρ⁰,
        max(zero(Y.c.ρ), ᶜq_ice⁰),
    )
    @. ᶜwᵢ = ᶜρa⁰ * ᶜq_ice⁰ * ᶜw⁰
    @. ᶜρχ = max(zero(Y.c.ρ), ᶜρa⁰ * ᶜq_ice⁰)
    for j in 1:n
        @. ᶜwᵢ += Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_ice * ᶜwᵢʲs.:($$j)
        @. ᶜρχ += max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_ice)
    end
    @. ᶜwᵢ = ifelse(ᶜρχ > FT(0), ᶜwᵢ / ᶜρχ, FT(0))

    # contribution of env cloud ice advection to htot advection
    @. ᶜρwₕhₜ += ᶜρa⁰ * ᶜq_ice⁰ * ᶜw⁰ * (Iᵢ(thp, ᶜts⁰) + ᶜΦ)

    # Rain (number)
    @. ᶜw⁰ = getindex(
        CM2.rain_terminal_velocity(
            cm2p.sb,
            cm2p.rtv,
            max(zero(Y.c.ρ), ᶜq_rai⁰),
            ᶜρ⁰,
            max(zero(Y.c.ρ), ᶜn_rai⁰),
        ),
        1,
    )
    @. ᶜwₙᵣ = ᶜρa⁰ * ᶜn_rai⁰ * ᶜw⁰
    @. ᶜρχ = max(zero(Y.c.ρ), ᶜρa⁰ * ᶜn_rai⁰)
    for j in 1:n
        @. ᶜwₙᵣ += Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).n_rai * ᶜwₙᵣʲs.:($$j)
        @. ᶜρχ += max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).n_rai)
    end
    @. ᶜwₙᵣ = ifelse(ᶜρχ > FT(0), ᶜwₙᵣ / ᶜρχ, FT(0))

    # Rain (mass)
    @. ᶜw⁰ = getindex(
        CM2.rain_terminal_velocity(
            cm2p.sb,
            cm2p.rtv,
            max(zero(Y.c.ρ), ᶜq_rai⁰),
            ᶜρ⁰,
            max(zero(Y.c.ρ), ᶜn_rai⁰),
        ),
        2,
    )
    @. ᶜwᵣ = ᶜρa⁰ * ᶜq_rai⁰ * ᶜw⁰
    @. ᶜρχ = max(zero(Y.c.ρ), ᶜρa⁰ * ᶜq_rai⁰)
    for j in 1:n
        @. ᶜwᵣ += Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_rai * ᶜwᵣʲs.:($$j)
        @. ᶜρχ += max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_rai)
    end
    @. ᶜwᵣ = ifelse(ᶜρχ > FT(0), ᶜwᵣ / ᶜρχ, FT(0))

    # contribution of env rain advection to qtot advection
    @. ᶜρwₕhₜ += ᶜρa⁰ * ᶜq_rai⁰ * ᶜw⁰ * (Iₗ(thp, ᶜts⁰) + ᶜΦ)

    # Snow
    # TODO sedimentation of snow is based on the 1M scheme
    @. ᶜw⁰ = CM1.terminal_velocity(
        cm1p.ps,
        cm1p.tv.snow,
        ᶜρ⁰,
        max(zero(Y.c.ρ), ᶜq_sno⁰),
    )
    @. ᶜwₛ = ᶜρa⁰ * ᶜq_sno⁰ * ᶜw⁰
    @. ᶜρχ = max(zero(Y.c.ρ), ᶜρa⁰ * ᶜq_sno⁰)
    for j in 1:n
        @. ᶜwₛ += Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_sno * ᶜwₛʲs.:($$j)
        @. ᶜρχ += max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_sno)
    end
    @. ᶜwₛ = ifelse(ᶜρχ > FT(0), ᶜwₛ / ᶜρχ, FT(0))

    # contribution of env snow advection to htot advection
    @. ᶜρwₕhₜ += ᶜρa⁰ * ᶜq_sno⁰ * ᶜw⁰ * (Iᵢ(thp, ᶜts⁰) + ᶜΦ)

    # Add contributions to energy and total water advection
    # TODO do we need to add kinetic energy of subdomains?
    for j in 1:n
        @. ᶜρwₕhₜ +=
            Y.c.sgsʲs.:($$j).ρa *
            (
                ᶜwₗʲs.:($$j) * Y.c.sgsʲs.:($$j).q_liq * (Iₗ(thp, ᶜtsʲs.:($$j)) + ᶜΦ) +
                ᶜwᵢʲs.:($$j) * Y.c.sgsʲs.:($$j).q_ice * (Iᵢ(thp, ᶜtsʲs.:($$j)) + ᶜΦ) +
                ᶜwᵣʲs.:($$j) * Y.c.sgsʲs.:($$j).q_rai * (Iₗ(thp, ᶜtsʲs.:($$j)) + ᶜΦ) +
                ᶜwₛʲs.:($$j) * Y.c.sgsʲs.:($$j).q_sno * (Iᵢ(thp, ᶜtsʲs.:($$j)) + ᶜΦ)
            )
    end
    @. ᶜwₕhₜ = Geometry.WVector(ᶜρwₕhₜ) / Y.c.ρ

    @. ᶜwₜqₜ =
        Geometry.WVector(
            ᶜwₗ * Y.c.ρq_liq +
            ᶜwᵢ * Y.c.ρq_ice +
            ᶜwᵣ * Y.c.ρq_rai +
            ᶜwₛ * Y.c.ρq_sno,
        ) / Y.c.ρ

    return nothing
end

function set_precipitation_velocities!(
    Y, p, ::NonEquilMoistModel, ::Microphysics2MomentP3,
)
    ## liquid quantities (2M warm rain)
    (; ᶜwₗ, ᶜwᵣ, ᶜwnₗ, ᶜwnᵣ, ᶜwₜqₜ, ᶜwₕhₜ, ᶜts, ᶜu) = p.precomputed
    (; ᶜΦ) = p.core

    (; ρ, ρq_liq, ρn_liq, ρq_rai, ρn_rai) = Y.c
    (; sb, rtv, ctv) = p.params.microphysics_2mp3_params.warm
    thp = CAP.thermodynamics_params(p.params)

    # Number- and mass weighted rain terminal velocity [m/s]
    ᶜrai_w_terms = @. lazy(
        CM2.rain_terminal_velocity(
            sb, rtv,
            max(zero(ρ), specific(ρq_rai, ρ)),
            ρ, max(zero(ρ), ρn_rai),
        ),
    )
    @. ᶜwnᵣ = getindex(ᶜrai_w_terms, 1)
    @. ᶜwᵣ = getindex(ᶜrai_w_terms, 2)
    # Number- and mass weighted cloud liquid terminal velocity [m/s]
    ᶜliq_w_terms = @. lazy(
        CM2.cloud_terminal_velocity(
            sb.pdf_c, ctv,
            max(zero(ρ), specific(ρq_liq, ρ)),
            ρ, max(zero(ρ), ρn_liq),
        ),
    )
    @. ᶜwnₗ = getindex(ᶜliq_w_terms, 1)
    @. ᶜwₗ = getindex(ᶜliq_w_terms, 2)

    ## Ice quantities
    (; ρq_ice, ρn_ice, ρq_rim, ρb_rim) = Y.c
    (; ᶜwᵢ) = p.precomputed
    (; cold) = CAP.microphysics_2mp3_params(p.params)

    # Number- and mass weighted ice terminal velocity [m/s]
    # Calculate terminal velocities
    (; ᶜlogλ, ᶜwnᵢ) = p.precomputed
    use_aspect_ratio = true  # TODO: Make a config option
    ᶜF_rim = @. lazy(ρq_rim / ρq_ice)
    ᶜρ_rim = @. lazy(ρq_rim / ρb_rim)
    ᶜstate_p3 = @. lazy(CMP3.P3State(cold.params,
        max(0, ρq_ice), max(0, ρn_ice), ᶜF_rim, ᶜρ_rim,
    ))
    @. ᶜlogλ = CMP3.get_distribution_logλ(ᶜstate_p3)
    args = (cold.velocity_params, ρ, ᶜstate_p3, ᶜlogλ)
    @. ᶜwnᵢ = CMP3.ice_terminal_velocity_number_weighted(args...; use_aspect_ratio)
    @. ᶜwᵢ = CMP3.ice_terminal_velocity_mass_weighted(args...; use_aspect_ratio)

    # compute their contributions to energy and total water advection
    @. ᶜwₜqₜ = Geometry.WVector(ᶜwₗ * ρq_liq + ᶜwᵢ * ρq_ice + ᶜwᵣ * ρq_rai) / ρ
    @. ᶜwₕhₜ =
        Geometry.WVector(
            ᶜwₗ * ρq_liq * (Iₗ(thp, ᶜts) + ᶜΦ + $(Kin(ᶜwₗ, ᶜu))) +
            ᶜwᵢ * ρq_ice * (Iᵢ(thp, ᶜts) + ᶜΦ + $(Kin(ᶜwᵢ, ᶜu))) +
            ᶜwᵣ * ρq_rai * (Iₗ(thp, ᶜts) + ᶜΦ + $(Kin(ᶜwᵣ, ᶜu))),
        ) / ρ
    return nothing
end

"""
    set_precipitation_cache!(Y, p, microphysics_model, turbconv_model)

Computes the cache needed for precipitation tendencies. When run without edmf
model this involves computing precipitation sources based on the grid mean
properties. When running with edmf model this means summing the precipitation
sources from the sub-domains.
"""
set_precipitation_cache!(Y, p, _, _) = nothing
function set_precipitation_cache!(Y, p, ::Microphysics0Moment, _)
    (; params, dt) = p
    (; ᶜts) = p.precomputed
    (; ᶜS_ρq_tot, ᶜS_ρe_tot) = p.precomputed
    (; ᶜΦ) = p.core
    cm_params = CAP.microphysics_0m_params(params)
    thermo_params = CAP.thermodynamics_params(params)
    @. ᶜS_ρq_tot =
        Y.c.ρ * q_tot_0M_precipitation_sources(
            thermo_params,
            cm_params,
            dt,
            Y.c.ρq_tot / Y.c.ρ,
            ᶜts,
        )
    @. ᶜS_ρe_tot =
        ᶜS_ρq_tot *
        e_tot_0M_precipitation_sources_helper(thermo_params, ᶜts, ᶜΦ)
    return nothing
end
function set_precipitation_cache!(
    Y,
    p,
    ::Microphysics0Moment,
    ::DiagnosticEDMFX,
)
    # For environment we multiply by grid mean ρ and not byᶜρa⁰
    # assuming a⁰=1
    (; ᶜΦ) = p.core
    (; ᶜSqₜᵖ⁰, ᶜSqₜᵖʲs, ᶜρaʲs) = p.precomputed
    (; ᶜS_ρq_tot, ᶜS_ρe_tot) = p.precomputed
    (; ᶜts, ᶜtsʲs) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)

    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    ρ = Y.c.ρ

    @. ᶜS_ρq_tot = ᶜSqₜᵖ⁰ * ρ
    @. ᶜS_ρe_tot =
        ᶜSqₜᵖ⁰ *
        ρ *
        e_tot_0M_precipitation_sources_helper(thermo_params, ᶜts, ᶜΦ)
    for j in 1:n
        @. ᶜS_ρq_tot += ᶜSqₜᵖʲs.:($$j) * ᶜρaʲs.:($$j)
        @. ᶜS_ρe_tot +=
            ᶜSqₜᵖʲs.:($$j) *
            ᶜρaʲs.:($$j) *
            e_tot_0M_precipitation_sources_helper(
                thermo_params,
                ᶜtsʲs.:($$j),
                ᶜΦ,
            )
    end
    return nothing
end
function set_precipitation_cache!(
    Y,
    p,
    ::Microphysics0Moment,
    ::PrognosticEDMFX,
)
    (; ᶜΦ) = p.core
    (; ᶜSqₜᵖ⁰, ᶜSqₜᵖʲs) = p.precomputed
    (; ᶜS_ρq_tot, ᶜS_ρe_tot) = p.precomputed
    (; ᶜts⁰, ᶜtsʲs) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, p.atmos.turbconv_model))

    n = n_mass_flux_subdomains(p.atmos.turbconv_model)

    @. ᶜS_ρq_tot = ᶜSqₜᵖ⁰ * ᶜρa⁰
    @. ᶜS_ρe_tot =
        ᶜSqₜᵖ⁰ *
        ᶜρa⁰ *
        e_tot_0M_precipitation_sources_helper(thermo_params, ᶜts⁰, ᶜΦ)
    for j in 1:n
        @. ᶜS_ρq_tot += ᶜSqₜᵖʲs.:($$j) * Y.c.sgsʲs.:($$j).ρa
        @. ᶜS_ρe_tot +=
            ᶜSqₜᵖʲs.:($$j) *
            Y.c.sgsʲs.:($$j).ρa *
            e_tot_0M_precipitation_sources_helper(
                thermo_params,
                ᶜtsʲs.:($$j),
                ᶜΦ,
            )
    end
    return nothing
end
function set_precipitation_cache!(Y, p, ::Microphysics1Moment, _)
    (; dt) = p
    (; ᶜts, ᶜwᵣ, ᶜwₛ, ᶜu) = p.precomputed
    (; ᶜSqₗᵖ, ᶜSqᵢᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ) = p.precomputed

    ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
    ᶜq_rai = @. lazy(specific(Y.c.ρq_rai, Y.c.ρ))
    ᶜq_sno = @. lazy(specific(Y.c.ρq_sno, Y.c.ρ))
    ᶜq_liq = @. lazy(specific(Y.c.ρq_liq, Y.c.ρ))
    ᶜq_ice = @. lazy(specific(Y.c.ρq_ice, Y.c.ρ))

    ᶜSᵖ = p.scratch.ᶜtemp_scalar
    ᶜSᵖ_snow = p.scratch.ᶜtemp_scalar_2
    ᶜ∇T = p.scratch.ᶜtemp_CT123

    # get thermodynamics and 1-moment microphysics params
    (; params) = p
    cmp = CAP.microphysics_1m_params(params)
    thp = CAP.thermodynamics_params(params)

    # compute precipitation source terms on the grid mean
    compute_precipitation_sources!(
        ᶜSᵖ,
        ᶜSᵖ_snow,
        ᶜSqₗᵖ,
        ᶜSqᵢᵖ,
        ᶜSqᵣᵖ,
        ᶜSqₛᵖ,
        Y.c.ρ,
        ᶜq_tot,
        ᶜq_liq,
        ᶜq_ice,
        ᶜq_rai,
        ᶜq_sno,
        ᶜts,
        dt,
        cmp,
        thp,
    )

    # compute precipitation sinks on the grid mean
    compute_precipitation_sinks!(
        ᶜSᵖ,
        ᶜSqᵣᵖ,
        ᶜSqₛᵖ,
        Y.c.ρ,
        ᶜq_tot,
        ᶜq_liq,
        ᶜq_ice,
        ᶜq_rai,
        ᶜq_sno,
        ᶜts,
        dt,
        cmp,
        thp,
    )
    return nothing
end
function set_precipitation_cache!(
    Y,
    p,
    ::Microphysics1Moment,
    ::DiagnosticEDMFX,
)
    # Nothing needs to be done on the grid mean. The Sources are computed
    # in edmf sub-domains.
    return nothing
end
function set_precipitation_cache!(
    Y,
    p,
    ::Microphysics1Moment,
    ::PrognosticEDMFX,
)
    # Nothing needs to be done on the grid mean. The Sources are computed
    # in edmf sub-domains.
    return nothing
end
function set_precipitation_cache!(Y, p, ::Microphysics2Moment, _)
    (; dt) = p
    (; ᶜts) = p.precomputed
    (; ᶜSqₗᵖ, ᶜSqᵢᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ) = p.precomputed
    (; ᶜSnₗᵖ, ᶜSnᵣᵖ) = p.precomputed

    ᶜSᵖ = p.scratch.ᶜtemp_scalar
    ᶜS₂ᵖ = p.scratch.ᶜtemp_scalar_2

    # get thermodynamics and microphysics params
    (; params) = p
    cmp = CAP.microphysics_2m_params(params)
    thp = CAP.thermodynamics_params(params)

    # compute warm precipitation sources on the grid mean (based on SB2006 2M scheme)
    compute_warm_precipitation_sources_2M!(
        ᶜSᵖ,
        ᶜS₂ᵖ,
        ᶜSnₗᵖ,
        ᶜSnᵣᵖ,
        ᶜSqₗᵖ,
        ᶜSqᵣᵖ,
        Y.c.ρ,
        lazy.(specific.(Y.c.ρn_liq, Y.c.ρ)),
        lazy.(specific.(Y.c.ρn_rai, Y.c.ρ)),
        lazy.(specific.(Y.c.ρq_tot, Y.c.ρ)),
        lazy.(specific.(Y.c.ρq_liq, Y.c.ρ)),
        lazy.(specific.(Y.c.ρq_ice, Y.c.ρ)),
        lazy.(specific.(Y.c.ρq_rai, Y.c.ρ)),
        lazy.(specific.(Y.c.ρq_sno, Y.c.ρ)),
        ᶜts,
        dt,
        cmp,
        thp,
    )

    #TODO - implement 2M cold processes!
    @. ᶜSqᵢᵖ = 0
    @. ᶜSqₛᵖ = 0

    return nothing
end
function set_precipitation_cache!(
    Y,
    p,
    ::Microphysics2Moment,
    ::DiagnosticEDMFX,
)
    error("Not implemented yet")
    return nothing
end
function set_precipitation_cache!(
    Y,
    p,
    ::Microphysics2Moment,
    ::PrognosticEDMFX,
)
    # Nothing needs to be done on the grid mean. The Sources are computed
    # in edmf sub-domains.
    return nothing
end

function set_precipitation_cache!(Y, p, ::Microphysics2MomentP3, ::Nothing)
    ### Rainy processes (2M)
    (; turbconv_model) = p.atmos
    set_precipitation_cache!(Y, p, Microphysics2Moment(), turbconv_model)
    # NOTE: the above function sets `ᶜSqᵢᵖ` to `0`. For P3, need to update `ᶜSqᵢᵖ` below!!

    ### Icy processes (P3)
    (; ᶜScoll, ᶜts, ᶜlogλ) = p.precomputed

    # get thermodynamics and microphysics params
    (; params) = p
    params_2mp3 = CAP.microphysics_2mp3_params(params)
    thermo_params = CAP.thermodynamics_params(params)

    ᶜY_reduced = (;
        Y.c.ρ,
        # condensate
        Y.c.ρq_liq, Y.c.ρn_liq, Y.c.ρq_rai, Y.c.ρn_rai,
        # ice
        Y.c.ρq_ice, Y.c.ρn_ice, Y.c.ρq_rim, Y.c.ρb_rim,
    )

    # compute warm precipitation sources on the grid mean (based on SB2006 2M scheme)
    compute_cold_precipitation_sources_P3!(
        ᶜScoll, params_2mp3, thermo_params, ᶜY_reduced, ᶜts, ᶜlogλ,
    )

    return nothing

end

"""
    set_precipitation_surface_fluxes!(Y, p, precipitation model)

Computes the flux of rain and snow at the surface. For the 0-moment microphysics
it is an integral of the source terms in the column. For 1-moment microphysics
it is the flux through the bottom cell face.
"""
set_precipitation_surface_fluxes!(Y, p, _) = nothing
function set_precipitation_surface_fluxes!(
    Y,
    p,
    microphysics_model::Microphysics0Moment,
)
    ᶜT = p.scratch.ᶜtemp_scalar
    (; ᶜts) = p.precomputed  # assume ᶜts has been updated
    (; ᶜS_ρq_tot, ᶜS_ρe_tot) = p.precomputed
    (; surface_rain_flux, surface_snow_flux) = p.precomputed
    (; col_integrated_precip_energy_tendency) = p.conservation_check

    # update total column energy source for surface energy balance
    Operators.column_integral_definite!(
        col_integrated_precip_energy_tendency,
        ᶜS_ρe_tot,
    )
    # update surface precipitation fluxes in cache for coupler's use
    thermo_params = CAP.thermodynamics_params(p.params)
    T_freeze = TD.Parameters.T_freeze(thermo_params)
    FT = eltype(p.params)
    @. ᶜT = TD.air_temperature(thermo_params, ᶜts)
    ᶜ3d_rain = @. lazy(ifelse(ᶜT >= T_freeze, ᶜS_ρq_tot, FT(0)))
    ᶜ3d_snow = @. lazy(ifelse(ᶜT < T_freeze, ᶜS_ρq_tot, FT(0)))
    Operators.column_integral_definite!(surface_rain_flux, ᶜ3d_rain)
    Operators.column_integral_definite!(surface_snow_flux, ᶜ3d_snow)
    return nothing
end
function set_precipitation_surface_fluxes!(
    Y,
    p,
    microphysics_model::Union{Microphysics1Moment, Microphysics2Moment},
)
    (; surface_rain_flux, surface_snow_flux) = p.precomputed
    (; col_integrated_precip_energy_tendency) = p.conservation_check
    (; ᶜwᵣ, ᶜwₛ, ᶜwₗ, ᶜwᵢ, ᶜwₕhₜ) = p.precomputed
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    sfc_J = Fields.level(ᶠJ, Fields.half)
    sfc_space = axes(sfc_J)

    # Jacobian-weighted extrapolation from interior to surface, consistent with
    # the reconstruction of density on cell faces, ᶠρ = ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ
    sfc_lev(x) =
        Fields.Field(Fields.field_values(Fields.level(x, 1)), sfc_space)
    int_J = sfc_lev(ᶜJ)
    int_ρ = sfc_lev(Y.c.ρ)
    sfc_ρ = @. lazy(int_ρ * int_J / sfc_J)

    # Constant extrapolation to surface, consistent with simple downwinding
    # Temporary scratch variables are used here until CC.field_values supports <lazy> fields
    ᶜq_rai = p.scratch.ᶜtemp_scalar
    ᶜq_sno = p.scratch.ᶜtemp_scalar_2
    ᶜq_liq = p.scratch.ᶜtemp_scalar_3
    ᶜq_ice = p.scratch.ᶜtemp_scalar_4
    @. ᶜq_rai = specific(Y.c.ρq_rai, Y.c.ρ)
    @. ᶜq_sno = specific(Y.c.ρq_sno, Y.c.ρ)
    @. ᶜq_liq = specific(Y.c.ρq_liq, Y.c.ρ)
    @. ᶜq_ice = specific(Y.c.ρq_ice, Y.c.ρ)
    sfc_qᵣ =
        Fields.Field(Fields.field_values(Fields.level(ᶜq_rai, 1)), sfc_space)
    sfc_qₛ =
        Fields.Field(Fields.field_values(Fields.level(ᶜq_sno, 1)), sfc_space)
    sfc_qₗ =
        Fields.Field(Fields.field_values(Fields.level(ᶜq_liq, 1)), sfc_space)
    sfc_qᵢ =
        Fields.Field(Fields.field_values(Fields.level(ᶜq_ice, 1)), sfc_space)
    sfc_wᵣ = Fields.Field(Fields.field_values(Fields.level(ᶜwᵣ, 1)), sfc_space)
    sfc_wₛ = Fields.Field(Fields.field_values(Fields.level(ᶜwₛ, 1)), sfc_space)
    sfc_wₗ = Fields.Field(Fields.field_values(Fields.level(ᶜwₗ, 1)), sfc_space)
    sfc_wᵢ = Fields.Field(Fields.field_values(Fields.level(ᶜwᵢ, 1)), sfc_space)
    sfc_wₕhₜ = Fields.Field(
        Fields.field_values(Fields.level(ᶜwₕhₜ.components.data.:1, 1)),
        sfc_space,
    )

    @. surface_rain_flux = sfc_ρ * (sfc_qᵣ * (-sfc_wᵣ) + sfc_qₗ * (-sfc_wₗ))
    @. surface_snow_flux = sfc_ρ * (sfc_qₛ * (-sfc_wₛ) + sfc_qᵢ * (-sfc_wᵢ))
    @. col_integrated_precip_energy_tendency = sfc_ρ * (-sfc_wₕhₜ)

    return nothing
end

function set_precipitation_surface_fluxes!(Y, p, ::Microphysics2MomentP3)
    set_precipitation_surface_fluxes!(Y, p, Microphysics2Moment())
    # TODO: Figure out what to do for ρn_ice, ρq_rim, ρb_rim
end
