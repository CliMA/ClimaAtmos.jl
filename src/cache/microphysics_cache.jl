#####
##### Precomputed quantities for precipitation processes
#####

import CloudMicrophysics.MicrophysicsNonEq as CMNe
import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics.Microphysics2M as CM2
import CloudMicrophysics.P3Scheme as CMP3
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT

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

# TODO - move to Utilities. Make sure it is used consistently
"""
    ϵ_numerics(FT)

Generic numerical-zero threshold.  Used for variance floors, σ guards,
and density-weighted mass checks — anywhere the exact value does not
matter as long as it is small but safely above underflow.
"""
ϵ_numerics(FT) = cbrt(floatmin(FT))

"""
    set_precipitation_velocities!(Y, p, microphysics_model, turbconv_model)

Updates the grid mean precipitation terminal velocity, cloud sedimentation velocity,
and their contribution to total water and energy advection.

For prognostic EDMF it also computes the sedimentation velocities in sub-domains
and ensures that the grid-scale flux is equal to the sum of sub-grid-scale fluxes.
"""
function set_precipitation_velocities!(Y, p, _, _)
    (; ᶜwₜqₜ, ᶜwₕhₜ) = p.precomputed
    @. ᶜwₜqₜ = Geometry.WVector(0)
    @. ᶜwₕhₜ = Geometry.WVector(0)
    return nothing
end
# TODO - a lot of code repetition between microphysic categories within functions
# and between different microphysics options. Refactor!
function set_precipitation_velocities!(
    Y,
    p,
    microphysics_model::NonEquilibriumMicrophysics1M,
    _,
)
    (; ᶜwₗ, ᶜwᵢ, ᶜwᵣ, ᶜwₛ, ᶜwₜqₜ, ᶜwₕhₜ, ᶜT, ᶜu) = p.precomputed
    (; ᶜΦ) = p.core
    cmc = CAP.microphysics_cloud_params(p.params)
    cmp = CAP.microphysics_1m_params(p.params)
    thp = CAP.thermodynamics_params(p.params)

    # compute the precipitation terminal velocity [m/s]
    @. ᶜwᵣ = CM1.terminal_velocity(
        cmp.precip.rain,
        cmp.terminal_velocity.rain,
        Y.c.ρ,
        max(zero(Y.c.ρ), Y.c.ρq_rai / Y.c.ρ),
    )
    @. ᶜwₛ = CM1.terminal_velocity(
        cmp.precip.snow,
        cmp.terminal_velocity.snow,
        Y.c.ρ,
        max(zero(Y.c.ρ), Y.c.ρq_sno / Y.c.ρ),
    )
    # compute sedimentation velocity for cloud condensate [m/s]
    @. ᶜwₗ = CMNe.terminal_velocity(
        cmc.liquid,
        cmc.stokes,
        Y.c.ρ,
        max(zero(Y.c.ρ), Y.c.ρq_lcl / Y.c.ρ),
    )
    @. ᶜwᵢ = CMNe.terminal_velocity(
        cmc.ice,
        cmc.Ch2022.small_ice,
        Y.c.ρ,
        max(zero(Y.c.ρ), Y.c.ρq_icl / Y.c.ρ),
    )

    # compute their contributions to energy and total water advection
    @. ᶜwₜqₜ =
        Geometry.WVector(
            ᶜwₗ * Y.c.ρq_lcl +
            ᶜwᵢ * Y.c.ρq_icl +
            ᶜwᵣ * Y.c.ρq_rai +
            ᶜwₛ * Y.c.ρq_sno,
        ) / Y.c.ρ
    @. ᶜwₕhₜ =
        Geometry.WVector(
            ᶜwₗ * Y.c.ρq_lcl * (Iₗ(thp, ᶜT) + ᶜΦ + $(Kin(ᶜwₗ, ᶜu))) +
            ᶜwᵢ * Y.c.ρq_icl * (Iᵢ(thp, ᶜT) + ᶜΦ + $(Kin(ᶜwᵢ, ᶜu))) +
            ᶜwᵣ * Y.c.ρq_rai * (Iₗ(thp, ᶜT) + ᶜΦ + $(Kin(ᶜwᵣ, ᶜu))) +
            ᶜwₛ * Y.c.ρq_sno * (Iᵢ(thp, ᶜT) + ᶜΦ + $(Kin(ᶜwₛ, ᶜu))),
        ) / Y.c.ρ
    return nothing
end
function set_precipitation_velocities!(
    Y,
    p,
    microphysics_model::NonEquilibriumMicrophysics1M,
    turbconv_model::PrognosticEDMFX,
)
    (; ᶜΦ) = p.core
    (; ᶜp) = p.precomputed
    (; ᶜwₗ, ᶜwᵢ, ᶜwᵣ, ᶜwₛ, ᶜwₜqₜ, ᶜwₕhₜ) = p.precomputed
    (; ᶜwₗʲs, ᶜwᵢʲs, ᶜwᵣʲs, ᶜwₛʲs, ᶜTʲs, ᶜρʲs) = p.precomputed
    (; ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed

    FT = eltype(p.params)
    cmc = CAP.microphysics_cloud_params(p.params)
    cmp = CAP.microphysics_1m_params(p.params)
    thp = CAP.thermodynamics_params(p.params)

    ᶜρ⁰ = @. lazy(
        TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰),
    )
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

    # Compute gs sedimentation velocity based on subdomain velocities
    # assuming grid-scale flux equals to the sum of sub-grid-scale fluxes.
    # TODO - below code is very repetetive between liquid, ice, rain and snow,
    # but also not exactly the same. Also repeated in 2m microphysics.
    # Think of a way to fix that.

    ###
    ### Cloud liquid
    ###
    ᶜq_lcl⁰ = ᶜspecific_env_value(@name(q_lcl), Y, p)
    ᶜρa⁰χ⁰ = @. lazy(max(zero(Y.c.ρ), ᶜρa⁰) * max(zero(Y.c.ρ), ᶜq_lcl⁰))
    @. ᶜρχ = ᶜρa⁰χ⁰
    @. ᶜwₗ = ᶜρa⁰χ⁰ * CMNe.terminal_velocity(
        cmc.liquid,
        cmc.stokes,
        ᶜρ⁰,
        ᶜq_lcl⁰,
    )
    @. ᶜimplied_env_mass_flux = 0
    # add updraft contributions
    for j in 1:n
        ᶜρaʲχʲ = @. lazy(
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).ρa) *
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_lcl),
        )
        @. ᶜρχ += ᶜρaʲχʲ
        @. ᶜwₗʲs.:($$j) = CMNe.terminal_velocity(
            cmc.liquid,
            cmc.stokes,
            max(zero(Y.c.ρ), ᶜρʲs.:($$j)),
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_lcl),
        )
        @. ᶜwₗ += ᶜρaʲχʲ * ᶜwₗʲs.:($$j)
        @. ᶜimplied_env_mass_flux -=
            Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_lcl * ᶜwₗʲs.:($$j)
    end
    # average (clamp to prevent spurious negatives from numerical errors at low mass)
    @. ᶜwₗ = ifelse(ᶜρχ > ϵ_numerics(FT), max(ᶜwₗ / ᶜρχ, zero(ᶜwₗ / ᶜρχ)), zero(ᶜwₗ))
    @. ᶜimplied_env_mass_flux += Y.c.ρq_lcl * ᶜwₗ
    # contribution of env q_liq sedimentation to htot
    @. ᶜρwₕhₜ = ᶜimplied_env_mass_flux * (Iₗ(thp, ᶜT⁰) + ᶜΦ)

    ###
    ### Cloud ice
    ###
    ᶜq_icl⁰ = ᶜspecific_env_value(@name(q_icl), Y, p)
    ᶜρa⁰χ⁰ = @. lazy(max(zero(Y.c.ρ), ᶜρa⁰) * max(zero(Y.c.ρ), ᶜq_icl⁰))
    @. ᶜρχ = ᶜρa⁰χ⁰
    @. ᶜwᵢ = ᶜρa⁰χ⁰ * CMNe.terminal_velocity(
        cmc.ice,
        cmc.Ch2022.small_ice,
        ᶜρ⁰,
        ᶜq_icl⁰,
    )
    @. ᶜimplied_env_mass_flux = 0
    # add updraft contributions
    for j in 1:n
        ᶜρaʲχʲ = @. lazy(
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).ρa) *
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_icl),
        )
        @. ᶜρχ += ᶜρaʲχʲ
        @. ᶜwᵢʲs.:($$j) = CMNe.terminal_velocity(
            cmc.ice,
            cmc.Ch2022.small_ice,
            max(zero(Y.c.ρ), ᶜρʲs.:($$j)),
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_icl),
        )
        @. ᶜwᵢ += ᶜρaʲχʲ * ᶜwᵢʲs.:($$j)
        @. ᶜimplied_env_mass_flux -=
            Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_icl * ᶜwᵢʲs.:($$j)
    end
    # average (clamp to prevent spurious negatives from numerical errors at low mass)
    @. ᶜwᵢ = ifelse(ᶜρχ > ϵ_numerics(FT), max(ᶜwᵢ / ᶜρχ, zero(ᶜwᵢ / ᶜρχ)), zero(ᶜwᵢ))
    @. ᶜimplied_env_mass_flux += Y.c.ρq_icl * ᶜwᵢ
    # contribution of env q_liq sedimentation to htot
    @. ᶜρwₕhₜ += ᶜimplied_env_mass_flux * (Iᵢ(thp, ᶜT⁰) + ᶜΦ)

    ###
    ### Rain
    ###
    ᶜq_rai⁰ = ᶜspecific_env_value(@name(q_rai), Y, p)
    ᶜρa⁰χ⁰ = @. lazy(max(zero(Y.c.ρ), ᶜρa⁰) * max(zero(Y.c.ρ), ᶜq_rai⁰))
    @. ᶜρχ = ᶜρa⁰χ⁰
    @. ᶜwᵣ =
        ᶜρa⁰χ⁰ * CM1.terminal_velocity(
            cmp.precip.rain,
            cmp.terminal_velocity.rain,
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
        @. ᶜwᵣʲs.:($$j) = CM1.terminal_velocity(
            cmp.precip.rain,
            cmp.terminal_velocity.rain,
            max(zero(Y.c.ρ), ᶜρʲs.:($$j)),
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_rai),
        )
        @. ᶜwᵣ += ᶜρaʲχʲ * ᶜwᵣʲs.:($$j)
        @. ᶜimplied_env_mass_flux -=
            Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_rai * ᶜwᵣʲs.:($$j)
    end
    # average (clamp to prevent spurious negatives from numerical errors at low mass)
    @. ᶜwᵣ = ifelse(ᶜρχ > ϵ_numerics(FT), max(ᶜwᵣ / ᶜρχ, zero(ᶜwᵣ / ᶜρχ)), zero(ᶜwᵣ))
    @. ᶜimplied_env_mass_flux += Y.c.ρq_rai * ᶜwᵣ
    # contribution of env q_liq sedimentation to htot
    @. ᶜρwₕhₜ += ᶜimplied_env_mass_flux * (Iₗ(thp, ᶜT⁰) + ᶜΦ)

    ###
    ### Snow
    ###
    ᶜq_sno⁰ = ᶜspecific_env_value(@name(q_sno), Y, p)
    ᶜρa⁰χ⁰ = @. lazy(max(zero(Y.c.ρ), ᶜρa⁰) * max(zero(Y.c.ρ), ᶜq_sno⁰))
    @. ᶜρχ = ᶜρa⁰χ⁰
    @. ᶜwₛ =
        ᶜρa⁰χ⁰ * CM1.terminal_velocity(
            cmp.precip.snow,
            cmp.terminal_velocity.snow,
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
        # compute terminal velocity for precipitation
        @. ᶜwₛʲs.:($$j) = CM1.terminal_velocity(
            cmp.precip.snow,
            cmp.terminal_velocity.snow,
            max(zero(Y.c.ρ), ᶜρʲs.:($$j)),
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_sno),
        )
        @. ᶜwₛ += ᶜρaʲχʲ * ᶜwₛʲs.:($$j)
        @. ᶜimplied_env_mass_flux -=
            Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_sno * ᶜwₛʲs.:($$j)
    end
    # average (clamp to prevent spurious negatives from numerical errors at low mass)
    @. ᶜwₛ = ifelse(ᶜρχ > ϵ_numerics(FT), max(ᶜwₛ / ᶜρχ, zero(ᶜwₛ / ᶜρχ)), zero(ᶜwₛ))
    @. ᶜimplied_env_mass_flux += Y.c.ρq_sno * ᶜwₛ
    # contribution of env q_liq sedimentation to htot
    @. ᶜρwₕhₜ += ᶜimplied_env_mass_flux * (Iᵢ(thp, ᶜT⁰) + ᶜΦ)

    # Add contributions to energy and total water advection
    # TODO do we need to add kinetic energy of subdomains?
    for j in 1:n
        @. ᶜρwₕhₜ +=
            Y.c.sgsʲs.:($$j).ρa *
            (
                ᶜwₗʲs.:($$j) * Y.c.sgsʲs.:($$j).q_lcl * (Iₗ(thp, ᶜTʲs.:($$j)) + ᶜΦ) +
                ᶜwᵢʲs.:($$j) * Y.c.sgsʲs.:($$j).q_icl * (Iᵢ(thp, ᶜTʲs.:($$j)) + ᶜΦ) +
                ᶜwᵣʲs.:($$j) * Y.c.sgsʲs.:($$j).q_rai * (Iₗ(thp, ᶜTʲs.:($$j)) + ᶜΦ) +
                ᶜwₛʲs.:($$j) * Y.c.sgsʲs.:($$j).q_sno * (Iᵢ(thp, ᶜTʲs.:($$j)) + ᶜΦ)
            )
    end
    @. ᶜwₕhₜ = Geometry.WVector(ᶜρwₕhₜ) / Y.c.ρ

    @. ᶜwₜqₜ =
        Geometry.WVector(
            ᶜwₗ * Y.c.ρq_lcl +
            ᶜwᵢ * Y.c.ρq_icl +
            ᶜwᵣ * Y.c.ρq_rai +
            ᶜwₛ * Y.c.ρq_sno,
        ) / Y.c.ρ

    return nothing
end

function set_precipitation_velocities!(Y, p,
    ::NonEquilibriumMicrophysics2M, turbconv_model::PrognosticEDMFX,
)
    (; ᶜwₗ, ᶜwᵢ, ᶜwᵣ, ᶜwₛ, ᶜwₙₗ, ᶜwₙᵣ, ᶜwₜqₜ, ᶜwₕhₜ) = p.precomputed
    (; ᶜΦ) = p.core
    (; ᶜwₗʲs, ᶜwᵢʲs, ᶜwᵣʲs, ᶜwₛʲs, ᶜwₙₗʲs, ᶜwₙᵣʲs) = p.precomputed
    (; ᶜp, ᶜTʲs, ᶜρʲs, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed
    cmc = CAP.microphysics_cloud_params(p.params)
    cm1p = CAP.microphysics_1m_params(p.params)
    cm2p = CAP.microphysics_2m_params(p.params)
    thp = CAP.thermodynamics_params(p.params)
    FT = eltype(p.params)

    ᶜρ⁰ = @. lazy(
        TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰),
    )
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

    # Compute grid-scale sedimentation velocity based on subdomain velocities
    # assuming grid-scale flux is equal to the  sum of sub-grid-scale fluxes.

    ᶜq_lcl⁰ = ᶜspecific_env_value(@name(q_lcl), Y, p)
    ᶜn_lcl⁰ = ᶜspecific_env_value(@name(n_lcl), Y, p)
    ###
    ### Cloud liquid (number)
    ###
    @. ᶜw⁰ = getindex(
        CM2.cloud_terminal_velocity(
            cm2p.warm_rain.seifert_beheng.pdf_c,
            cmc.stokes,
            max(zero(Y.c.ρ), ᶜq_lcl⁰),
            ᶜρ⁰,
            max(zero(Y.c.ρ), ᶜn_lcl⁰),
        ),
        1,
    )
    @. ᶜwₙₗ = ᶜρa⁰ * ᶜn_lcl⁰ * ᶜw⁰
    @. ᶜρχ = max(zero(Y.c.ρ), ᶜρa⁰ * ᶜn_lcl⁰)
    for j in 1:n
        @. ᶜwₙₗʲs.:($$j) = getindex(
            CM2.cloud_terminal_velocity(
                cm2p.warm_rain.seifert_beheng.pdf_c,
                cmc.stokes,
                max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_lcl),
                ᶜρʲs.:($$j),
                max(zero(Y.c.ρ), ᶜρʲs.:($$j) * Y.c.sgsʲs.:($$j).n_lcl),
            ),
            1,
        )
        @. ᶜwₙₗ += Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).n_lcl * ᶜwₙₗʲs.:($$j)
        @. ᶜρχ += max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).n_lcl)
    end
    @. ᶜwₙₗ = ifelse(ᶜρχ > FT(0), ᶜwₙₗ / ᶜρχ, FT(0))

    ###
    ### Cloud liquid (mass)
    ###
    @. ᶜw⁰ = getindex(
        CM2.cloud_terminal_velocity(
            cm2p.warm_rain.seifert_beheng.pdf_c,
            cmc.stokes,
            max(zero(Y.c.ρ), ᶜq_lcl⁰),
            ᶜρ⁰,
            max(zero(Y.c.ρ), ᶜn_lcl⁰),
        ),
        2,
    )
    @. ᶜwₗ = ᶜρa⁰ * ᶜq_lcl⁰ * ᶜw⁰
    @. ᶜρχ = max(zero(Y.c.ρ), ᶜρa⁰ * ᶜq_lcl⁰)
    for j in 1:n
        @. ᶜwₗʲs.:($$j) = getindex(
            CM2.cloud_terminal_velocity(
                cm2p.warm_rain.seifert_beheng.pdf_c,
                cmc.stokes,
                max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_lcl),
                ᶜρʲs.:($$j),
                max(zero(Y.c.ρ), ᶜρʲs.:($$j) * Y.c.sgsʲs.:($$j).n_lcl),
            ),
            2,
        )
        @. ᶜwₗ += Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_lcl * ᶜwₗʲs.:($$j)
        @. ᶜρχ += max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_lcl)
    end
    @. ᶜwₗ = ifelse(ᶜρχ > FT(0), ᶜwₗ / ᶜρχ, FT(0))
    # contribution of env cloud liquid advection to htot advection
    @. ᶜρwₕhₜ = ᶜρa⁰ * ᶜq_lcl⁰ * ᶜw⁰ * (Iₗ(thp, ᶜT⁰) + ᶜΦ)

    ###
    ### Cloud ice
    ###
    ᶜq_icl⁰ = ᶜspecific_env_value(@name(q_icl), Y, p)
    # TODO sedimentation of ice is based on the 1M scheme
    @. ᶜw⁰ = CMNe.terminal_velocity(
        cmc.ice,
        cmc.Ch2022.small_ice,
        ᶜρ⁰,
        max(zero(Y.c.ρ), ᶜq_icl⁰),
    )
    @. ᶜwᵢ = ᶜρa⁰ * ᶜq_icl⁰ * ᶜw⁰
    @. ᶜρχ = max(zero(Y.c.ρ), ᶜρa⁰ * ᶜq_icl⁰)
    for j in 1:n
        @. ᶜwᵢʲs.:($$j) = CMNe.terminal_velocity(
            cmc.ice,
            cmc.Ch2022.small_ice,
            ᶜρʲs.:($$j),
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_icl),
        )
        @. ᶜwᵢ += Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_icl * ᶜwᵢʲs.:($$j)
        @. ᶜρχ += max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_icl)
    end
    @. ᶜwᵢ = ifelse(ᶜρχ > FT(0), ᶜwᵢ / ᶜρχ, FT(0))
    # contribution of env cloud ice advection to htot advection
    @. ᶜρwₕhₜ += ᶜρa⁰ * ᶜq_icl⁰ * ᶜw⁰ * (Iᵢ(thp, ᶜT⁰) + ᶜΦ)

    ###
    ### Rain (number)
    ###
    (; seifert_beheng) = cm2p.warm_rain
    rtv = cmc.Ch2022.rain
    ᶜq_rai⁰ = ᶜspecific_env_value(@name(q_rai), Y, p)
    ᶜn_rai⁰ = ᶜspecific_env_value(@name(n_rai), Y, p)
    @. ᶜw⁰ = getindex(
        CM2.rain_terminal_velocity(
            seifert_beheng, rtv,
            max(zero(Y.c.ρ), ᶜq_rai⁰),
            ᶜρ⁰,
            # I think this should be ρn_rai⁰
            max(zero(Y.c.ρ), ᶜn_rai⁰),
        ),
        1,
    )
    @. ᶜwₙᵣ = ᶜρa⁰ * ᶜn_rai⁰ * ᶜw⁰
    @. ᶜρχ = max(zero(Y.c.ρ), ᶜρa⁰ * ᶜn_rai⁰)
    for j in 1:n
        @. ᶜwₙᵣʲs.:($$j) = getindex(
            CM2.rain_terminal_velocity(
                cm2p.warm_rain.seifert_beheng,
                cmc.Ch2022.rain,
                max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_rai),
                ᶜρʲs.:($$j),
                max(zero(Y.c.ρ), ᶜρʲs.:($$j) * Y.c.sgsʲs.:($$j).n_rai),
            ),
            1,
        )
        @. ᶜwₙᵣ += Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).n_rai * ᶜwₙᵣʲs.:($$j)
        @. ᶜρχ += max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).n_rai)
    end
    @. ᶜwₙᵣ = ifelse(ᶜρχ > FT(0), ᶜwₙᵣ / ᶜρχ, FT(0))

    ###
    ### Rain (mass)
    ###
    @. ᶜw⁰ = getindex(
        CM2.rain_terminal_velocity(
            cm2p.warm_rain.seifert_beheng,
            cmc.Ch2022.rain,
            max(zero(Y.c.ρ), ᶜq_rai⁰),
            ᶜρ⁰,
            max(zero(Y.c.ρ), ᶜn_rai⁰),
        ),
        2,
    )
    @. ᶜwᵣ = ᶜρa⁰ * ᶜq_rai⁰ * ᶜw⁰
    @. ᶜρχ = max(zero(Y.c.ρ), ᶜρa⁰ * ᶜq_rai⁰)
    for j in 1:n
        @. ᶜwᵣʲs.:($$j) = getindex(
            CM2.rain_terminal_velocity(
                cm2p.warm_rain.seifert_beheng,
                cmc.Ch2022.rain,
                max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_rai),
                ᶜρʲs.:($$j),
                max(zero(Y.c.ρ), ᶜρʲs.:($$j) * Y.c.sgsʲs.:($$j).n_rai),
            ),
            2,
        )
        @. ᶜwᵣ += Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_rai * ᶜwᵣʲs.:($$j)
        @. ᶜρχ += max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_rai)
    end
    @. ᶜwᵣ = ifelse(ᶜρχ > FT(0), ᶜwᵣ / ᶜρχ, FT(0))
    # contribution of env rain advection to qtot advection
    @. ᶜρwₕhₜ += ᶜρa⁰ * ᶜq_rai⁰ * ᶜw⁰ * (Iₗ(thp, ᶜT⁰) + ᶜΦ)

    # Add contributions to energy and total water advection
    # TODO do we need to add kinetic energy of subdomains?
    for j in 1:n
        @. ᶜρwₕhₜ +=
            Y.c.sgsʲs.:($$j).ρa *
            (
                ᶜwₗʲs.:($$j) * Y.c.sgsʲs.:($$j).q_liq * (Iₗ(thp, ᶜTʲs.:($$j)) + ᶜΦ) +
                ᶜwᵢʲs.:($$j) * Y.c.sgsʲs.:($$j).q_ice * (Iᵢ(thp, ᶜTʲs.:($$j)) + ᶜΦ) +
                ᶜwᵣʲs.:($$j) * Y.c.sgsʲs.:($$j).q_rai * (Iₗ(thp, ᶜTʲs.:($$j)) + ᶜΦ)
            )
    end
    @. ᶜwₕhₜ = Geometry.WVector(ᶜρwₕhₜ) / Y.c.ρ

    @. ᶜwₜqₜ =
        Geometry.WVector(
            ᶜwₗ * Y.c.ρq_liq +
            ᶜwᵢ * Y.c.ρq_ice +
            ᶜwᵣ * Y.c.ρq_rai,
        ) / Y.c.ρ

    return nothing
end


function set_precipitation_velocities!(Y, p, ::NonEquilibriumMicrophysics2M, _)
    params_2m = CAP.microphysics_2m_params(p.params)
    # 1. liquid quantities (2M warm rain)
    (; ᶜwₗ, ᶜwᵣ, ᶜwₙₗ, ᶜwₙᵣ) = p.precomputed

    (; ρ, ρq_lcl, ρn_lcl, ρq_rai, ρn_rai) = Y.c
    cmc = CAP.microphysics_cloud_params(p.params)

    # Access 2M warm rain params from unified container
    (; seifert_beheng) = params_2m.warm_rain
    ctv = cmc.stokes  # Cloud terminal velocity
    rtv = cmc.Ch2022.rain  # Rain terminal velocity from cloud_params

    # Number- and mass weighted rain terminal velocity [m/s]
    ᶜrai_w_terms = @. lazy(
        CM2.rain_terminal_velocity(
            seifert_beheng, rtv,
            max(zero(ρ), specific(ρq_rai, ρ)), ρ, max(zero(ρ), ρn_rai),
        ),
    )
    @. ᶜwₙᵣ = getindex(ᶜrai_w_terms, 1)
    @. ᶜwᵣ = getindex(ᶜrai_w_terms, 2)
    # Number- and mass weighted cloud liquid terminal velocity [m/s]
    ᶜlcl_w_terms = @. lazy(
        CM2.cloud_terminal_velocity(
            seifert_beheng.pdf_c, ctv,
            max(zero(ρ), specific(ρq_lcl, ρ)), ρ, max(zero(ρ), ρn_lcl),
        ),
    )
    @. ᶜwₙₗ = getindex(ᶜlcl_w_terms, 1)
    @. ᶜwₗ = getindex(ᶜlcl_w_terms, 2)

    # 2. Ice quantities
    (; ρq_ice, ρn_ice, ρq_rim, ρb_rim) = Y.c
    (; ᶜwᵢ) = p.precomputed

    # P3 ice params from unified container
    (; terminal_velocity, scheme) = params_2m.ice
    # p3_ice = params_2m.ice

    # Number- and mass weighted ice terminal velocity [m/s]
    # Calculate terminal velocities
    (; ᶜlogλ, ᶜwₙᵢ) = p.precomputed
    use_aspect_ratio = true  # TODO: Make a config option
    ᶜρq_ice_safe = @. lazy(max(zero(ρq_ice), ρq_ice))
    ᶜρn_ice_safe = @. lazy(max(zero(ρn_ice), ρn_ice))
    ᶜF_rim = @. lazy(ifelse(iszero(ρq_ice), zero(ρq_ice), ρq_rim / ρq_ice))
    ᶜρ_rim = @. lazy(ifelse(iszero(ρq_ice), zero(ρq_ice), ρq_rim / ρb_rim))
    ᶜstate_p3 = @. lazy(CMP3.P3State(scheme, ᶜρq_ice_safe, ᶜρn_ice_safe, ᶜF_rim, ᶜρ_rim))

    ## Idea: moving operations into pointwise functions
    # THIS WORKS!!! We'll have it in a pointwise function yay
    @. ᶜlogλ = CMP3.get_distribution_logλ_from_prognostic(scheme, ρq_ice, ρn_ice, ρq_rim, ρb_rim)

    # Try this: THIS DOES NOT WORK
    # ᶜρq_ice_safe = @. max(zero(ρq_ice), ρq_ice)
    # ᶜρn_ice_safe = @. max(zero(ρn_ice), ρn_ice)
    # ᶜF_rim = @. ifelse(iszero(ρq_ice), zero(ρq_ice), ρq_rim / ρq_ice)
    # ᶜρ_rim = @. ifelse(iszero(ρq_ice), zero(ρq_ice), ρq_rim / ρb_rim)
    # ᶜstate_p3 = @. lazy(CMP3.P3State(scheme, ᶜρq_ice_safe, ᶜρn_ice_safe, ᶜF_rim, ᶜρ_rim))
    # @. ᶜlogλ = CMP3.get_distribution_logλ(ᶜstate_p3)

    # Try to allocate the ᶜstate_p3 field, and pass that to get_distribution_logλ. THIS DOES NOT WORK
    # ᶜstate_p3 = @. lazy(CMP3.P3State(scheme, ᶜρq_ice_safe, ᶜρn_ice_safe, ᶜF_rim, ᶜρ_rim))
    # @. ᶜlogλ = CMP3.get_distribution_logλ(ᶜstate_p3)
    
    ### Option 1: CHECK. THIS SEEMS TO WORK
    # ᶜstate_p3 = @. CMP3.P3State(scheme, ᶜρq_ice_safe, ᶜρn_ice_safe, ᶜF_rim, ᶜρ_rim)
    # @. ᶜlogλ = CMP3.get_distribution_logλ(ᶜstate_p3)

    ### Option 2: CHECK. THIS SEEMS TO WORK <-- works in fresh session
    # THIS DID NOT WORK IN AN INFILTRATE SESSION
    # @. ᶜlogλ = CMP3.get_distribution_logλ(
    #     scheme,
    #     ᶜρq_ice_safe,
    #     ᶜρn_ice_safe,
    #     ᶜF_rim,
    #     ᶜρ_rim,
    # )

    ### Option 2a: CHECK THIS
    # THIS DID NOT WORK IN AN INFILTRATE SESSION
    # @. ᶜlogλ = CMP3.get_distribution_logλ(
    #     scheme,
    #     max(zero(ρq_ice), ρq_ice),
    #     max(zero(ρn_ice), ρn_ice),
    #     ifelse(iszero(ρq_ice), zero(ρq_ice), ρq_rim / ρq_ice),
    #     ifelse(iszero(ρq_ice), zero(ρq_ice), ρq_rim / ρb_rim),
    # )

    ### Option 3
    # @. ᶜlogλ = CMP3.get_distribution_logλ(
    #     CMP3.P3State(scheme, ᶜρq_ice_safe, ᶜρn_ice_safe, ᶜF_rim, ᶜρ_rim)
    # )

    @. ᶜwₙᵢ = CMP3.ice_terminal_velocity_number_weighted(terminal_velocity, ρ, ᶜstate_p3, ᶜlogλ; use_aspect_ratio)
    @. ᶜwᵢ = CMP3.ice_terminal_velocity_mass_weighted(terminal_velocity, ρ, ᶜstate_p3, ᶜlogλ; use_aspect_ratio)
    # @. ᶜwₙᵢ = CMP3.ice_terminal_velocity_number_weighted(p3_ice.terminal_velocity, ρ, CMP3.P3State(p3_ice.scheme, ᶜρq_ice_safe, ᶜρn_ice_safe, ᶜF_rim, ᶜρ_rim), ᶜlogλ; use_aspect_ratio)
    # @. ᶜwᵢ = CMP3.ice_terminal_velocity_mass_weighted(p3_ice.terminal_velocity, ρ, CMP3.P3State(p3_ice.scheme, ᶜρq_ice_safe, ᶜρn_ice_safe, ᶜF_rim, ᶜρ_rim), ᶜlogλ; use_aspect_ratio)

    # 3. Compute their contributions to energy and total water advection
    thp = CAP.thermodynamics_params(p.params)
    (; ᶜwₜqₜ, ᶜwₕhₜ, ᶜT, ᶜu) = p.precomputed
    (; ᶜΦ) = p.core
    @. ᶜwₜqₜ = Geometry.WVector(ᶜwₗ * ρq_lcl + ᶜwᵢ * ρq_ice + ᶜwᵣ * ρq_rai) / ρ
    @. ᶜwₕhₜ =
        Geometry.WVector(
            ᶜwₗ * ρq_lcl * (Iₗ(thp, ᶜT) + ᶜΦ + $(Kin(ᶜwₗ, ᶜu))) +
            ᶜwᵢ * ρq_ice * (Iᵢ(thp, ᶜT) + ᶜΦ + $(Kin(ᶜwᵢ, ᶜu))) +
            ᶜwᵣ * ρq_rai * (Iₗ(thp, ᶜT) + ᶜΦ + $(Kin(ᶜwᵣ, ᶜu))),
        ) / ρ
    return nothing
end

"""
    update_implicit_microphysics_cache!(Y, p, microphysics_model, turbconv_model)

Refresh microphysics precomputed quantities that depend on the current Newton
iterate `Y`.  Called from `set_implicit_precomputed_quantities!` at every
Newton iteration of the implicit solve.

All schemes freeze the specific microphysics tendencies computed in the
explicit stage; only density-weighted source terms and surface fluxes are
refreshed here.

- **0M**: recomputes `ρ × mp_tendency.dq_tot_dt` from the frozen
  `ᶜmp_tendency` (ρ × tendency).  For EDMF variants, the per-subdomain
  specific tendencies are re-aggregated with the current ρ / ρa.
- **1M/2M**: refreshes only `set_precipitation_surface_fluxes!`.  The
  specific tendencies (mp_tendency) are frozen; density weighting is
  applied at tendency-evaluation time in `tendency.jl`.
- **default**: no-op (microphysics not active or not implicit).
"""
update_implicit_microphysics_cache!(Y, p, _, _) = nothing

function update_implicit_microphysics_cache!(
    Y, p, mm::EquilibriumMicrophysics0M, _,
)
    (; ᶜmp_tendency, ᶜρ_dq_tot_dt, ᶜρ_de_tot_dt) = p.precomputed
    @. ᶜρ_dq_tot_dt = Y.c.ρ * ᶜmp_tendency.dq_tot_dt
    @. ᶜρ_de_tot_dt = ᶜρ_dq_tot_dt * ᶜmp_tendency.e_tot_hlpr

    set_precipitation_surface_fluxes!(Y, p, mm)
    return nothing
end

function update_implicit_microphysics_cache!(
    Y, p, mm::EquilibriumMicrophysics0M, tm::DiagnosticEDMFX,
)
    (; ᶜmp_tendency, ᶜmp_tendencyʲs, ᶜρaʲs) = p.precomputed
    (; ᶜρ_dq_tot_dt, ᶜρ_de_tot_dt) = p.precomputed
    n = n_mass_flux_subdomains(tm)

    @. ᶜρ_dq_tot_dt = ᶜmp_tendency.dq_tot_dt * ρa⁰(Y.c.ρ, ᶜρaʲs, tm)
    @. ᶜρ_de_tot_dt = ᶜρ_dq_tot_dt * ᶜmp_tendency.e_tot_hlpr
    for j in 1:n
        @. ᶜρ_dq_tot_dt += ᶜρaʲs.:($$j) * ᶜmp_tendencyʲs.:($$j).dq_tot_dt
        @. ᶜρ_de_tot_dt +=
            ᶜρaʲs.:($$j) * ᶜmp_tendencyʲs.:($$j).dq_tot_dt *
            ᶜmp_tendencyʲs.:($$j).e_tot_hlpr
    end
    set_precipitation_surface_fluxes!(Y, p, mm)
    return nothing
end

function update_implicit_microphysics_cache!(
    Y, p, mm::EquilibriumMicrophysics0M, tm::PrognosticEDMFX,
)
    (; ᶜmp_tendencyʲs, ᶜmp_tendency⁰) = p.precomputed
    (; ᶜρ_dq_tot_dt, ᶜρ_de_tot_dt) = p.precomputed
    n = n_mass_flux_subdomains(tm)

    @. ᶜρ_dq_tot_dt = ᶜmp_tendency⁰.dq_tot_dt * ρa⁰(Y.c.ρ, Y.c.sgsʲs, tm)
    @. ᶜρ_de_tot_dt = ᶜρ_dq_tot_dt * ᶜmp_tendency⁰.e_tot_hlpr
    for j in 1:n
        ρdq_tot_dtʲ = @. lazy(Y.c.sgsʲs.:($$j).ρa * ᶜmp_tendencyʲs.:($$j).dq_tot_dt)
        @. ᶜρ_dq_tot_dt += ρdq_tot_dtʲ
        @. ᶜρ_de_tot_dt += ρdq_tot_dtʲ * ᶜmp_tendencyʲs.:($$j).e_tot_hlpr
    end
    set_precipitation_surface_fluxes!(Y, p, mm)
    return nothing
end

function update_implicit_microphysics_cache!(
    Y, p, mm::NonEquilibriumMicrophysics, _,
)
    set_precipitation_surface_fluxes!(Y, p, mm)
    return nothing
end

"""
    set_microphysics_tendency_cache!(Y, p, microphysics_model, turbconv_model)

Compute and cache the microphysics source terms (`ᶜmp_tendency`, Jacobian
coefficients, etc.) for the current state `Y`.

**Dispatch table** (microphysics_model × turbconv_model):

| Model    | Nothing / default      | DiagnosticEDMFX | PrognosticEDMFX |
|----------|------------------------|-----------------|-----------------|
| DryModel | no-op                  | no-op (fallback)| no-op (fallback)|
| 0M       | grid-mean (± SGS quad) | EDMF-weighted   | EDMF-weighted   |
| 1M       | grid-mean (± SGS quad) | EDMF-weighted   | EDMF-weighted   |
| 2M       | grid-mean              | not implemented | EDMF-weighted   |

**Non-EDMF path** computes microphysics on the grid-mean state, with an optional sum over
SGS quadrature points (controlled by `p.atmos.sgs_quadrature`) to sample subgrid variability.

**EDMF path** computes tendencies separately for updrafts and the environment:

*Updrafts* use direct BMT evaluation (no SGS quadrature) because:
1. Updrafts are coherent turbulent structures with more homogeneous thermodynamic properties
2. Updraft area fraction is usually small (~1-10%), so SGS variance within updrafts has limited
   impact on the grid-mean tendency.

*Environment* uses SGS quadrature integration (when `sgs_quadrature` is configured) because
the environment dominates the grid-mean variance. The quadrature captures subgrid-scale
fluctuations in temperature and moisture, which is important for threshold processes like
condensation/evaporation at cloud edges.

The grid-mean source is then the area-weighted sum:
  `ᶜS_ρq_tot = ᶜSqₜᵐ⁰ * ᶜρa⁰ + Σⱼ ᶜSqₜᵐʲ * ᶜρaʲ`
"""
set_microphysics_tendency_cache!(Y, p, _, _) = nothing

###
### 0 Moment Microphysics
###

function set_microphysics_tendency_cache!(Y, p, ::EquilibriumMicrophysics0M, _)
    (; dt) = p
    (; ᶜΦ) = p.core
    (; ᶜT, ᶜq_tot_nonneg, ᶜmp_tendency) = p.precomputed

    cm0 = CAP.microphysics_0m_params(p.params)
    thp = CAP.thermodynamics_params(p.params)

    ### Grid-mean microphysics tendency with/without quadrature sampling.
    sgs_quad = p.atmos.sgs_quadrature
    if not_quadrature(sgs_quad)
        # Evaluate on the grid-mean.
        (; ᶜq_liq, ᶜq_ice) = p.precomputed
        @. ᶜmp_tendency = microphysics_tendencies_0m(
            cm0, thp, Y.c.ρ, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice, ᶜΦ, dt,
        )
    else
        # Evaluate over quadrature points. Both dq_tot_dt and e_tot_hlpr
        # are SGS-averaged so that the energy helper is consistent with
        # the nonlinear dependence on condensate at each quadrature point.
        (; ᶜT′T′, ᶜq′q′) = p.precomputed
        corr_Tq = correlation_Tq(p.params)
        @. ᶜmp_tendency = microphysics_tendencies_0m(
            $(sgs_quad), cm0, thp, Y.c.ρ, ᶜT, ᶜq_tot_nonneg,
            ᶜT′T′, ᶜq′q′, corr_Tq, ᶜΦ, dt,
        )
    end

    # TODO - duplicated with tendency and implicit cache update
    (; ᶜρ_dq_tot_dt, ᶜρ_de_tot_dt) = p.precomputed

    @. ᶜρ_dq_tot_dt = Y.c.ρ * ᶜmp_tendency.dq_tot_dt
    @. ᶜρ_de_tot_dt = ᶜρ_dq_tot_dt * ᶜmp_tendency.e_tot_hlpr
    return nothing
end

function set_microphysics_tendency_cache!(
    Y, p, ::EquilibriumMicrophysics0M, tm::DiagnosticEDMFX,
)
    (; dt) = p
    (; ᶜΦ) = p.core
    (; ᶜmp_tendency) = p.precomputed
    (; ᶜT, ᶜq_tot_nonneg) = p.precomputed

    thp = CAP.thermodynamics_params(p.params)
    cm0 = CAP.microphysics_0m_params(p.params)

    ### Updraft contribution is computed in diagnostic EDMF integral loop

    ### Environment contribution on the grid mean or quadrature points
    sgs_quad = p.atmos.sgs_quadrature
    if not_quadrature(sgs_quad)
        (; ᶜq_liq, ᶜq_ice) = p.precomputed
        @. ᶜmp_tendency = microphysics_tendencies_0m(
            cm0, thp, Y.c.ρ, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice, ᶜΦ, dt,
        )
    else
        (; ᶜT′T′, ᶜq′q′) = p.precomputed
        corr_Tq = correlation_Tq(p.params)
        @. ᶜmp_tendency = microphysics_tendencies_0m(
            $(sgs_quad), cm0, thp, Y.c.ρ, ᶜT, ᶜq_tot_nonneg,
            ᶜT′T′, ᶜq′q′, corr_Tq, ᶜΦ, dt,
        )
    end

    # TODO - duplicated with tendency and implicit cache update
    (; ᶜmp_tendencyʲs, ᶜρaʲs) = p.precomputed
    (; ᶜρ_dq_tot_dt, ᶜρ_de_tot_dt) = p.precomputed
    n = n_mass_flux_subdomains(tm)
    @. ᶜρ_dq_tot_dt = ᶜmp_tendency.dq_tot_dt * ρa⁰(Y.c.ρ, ᶜρaʲs, tm)
    @. ᶜρ_de_tot_dt = ᶜρ_dq_tot_dt * ᶜmp_tendency.e_tot_hlpr
    for j in 1:n
        @. ᶜρ_dq_tot_dt += ᶜρaʲs.:($$j) * ᶜmp_tendencyʲs.:($$j).dq_tot_dt
        @. ᶜρ_de_tot_dt +=
            ᶜρaʲs.:($$j) * ᶜmp_tendencyʲs.:($$j).dq_tot_dt *
            ᶜmp_tendencyʲs.:($$j).e_tot_hlpr
    end

    return nothing
end

function set_microphysics_tendency_cache!(
    Y, p, ::EquilibriumMicrophysics0M, tm::PrognosticEDMFX,
)
    (; ᶜΦ) = p.core
    (; dt) = p
    (; ᶜp) = p.precomputed

    (; ᶜmp_tendencyʲs, ᶜmp_tendency⁰) = p.precomputed
    (; ᶜρʲs, ᶜTʲs, ᶜq_tot_nonnegʲs, ᶜq_liqʲs, ᶜq_iceʲs) = p.precomputed
    (; ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed

    thp = CAP.thermodynamics_params(p.params)
    cm0 = CAP.microphysics_0m_params(p.params)

    n = n_mass_flux_subdomains(tm)

    for j in 1:n
        # Point-wise evaluation of microphysics tendencies in the updraft
        @. ᶜmp_tendencyʲs.:($$j) = microphysics_tendencies_0m(
            cm0, thp, ᶜρʲs.:($$j), ᶜTʲs.:($$j), ᶜq_tot_nonnegʲs.:($$j),
            ᶜq_liqʲs.:($$j), ᶜq_iceʲs.:($$j), ᶜΦ, dt,
        )
    end

    ### Environment contribution with/without quadrature sampling.
    ᶜρ⁰ = @. lazy(
        TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰),
    )
    sgs_quad = p.atmos.sgs_quadrature
    if not_quadrature(sgs_quad)
        # Evaluate on the grid-mean.
        @. ᶜmp_tendency⁰ = microphysics_tendencies_0m(
            cm0, thp, ᶜρ⁰, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜΦ, dt,
        )
    else
        # Evaluate over quadrature points.
        (; ᶜT′T′, ᶜq′q′) = p.precomputed
        corr_Tq = correlation_Tq(p.params)
        @. ᶜmp_tendency⁰ = microphysics_tendencies_0m(
            $(sgs_quad), cm0, thp, ᶜρ⁰, ᶜT⁰, ᶜq_tot_nonneg⁰,
            ᶜT′T′, ᶜq′q′, corr_Tq, ᶜΦ, dt,
        )
    end

    # TODO - duplicated with tendency and implicit cache update
    (; ᶜρ_dq_tot_dt, ᶜρ_de_tot_dt) = p.precomputed

    @. ᶜρ_dq_tot_dt = ᶜmp_tendency⁰.dq_tot_dt * ρa⁰(Y.c.ρ, Y.c.sgsʲs, tm)
    @. ᶜρ_de_tot_dt = ᶜρ_dq_tot_dt * ᶜmp_tendency⁰.e_tot_hlpr
    for j in 1:n
        @. ᶜρ_dq_tot_dt += ᶜmp_tendencyʲs.:($$j).dq_tot_dt * Y.c.sgsʲs.:($$j).ρa
        @. ᶜρ_de_tot_dt +=
            ᶜmp_tendencyʲs.:($$j).dq_tot_dt * Y.c.sgsʲs.:($$j).ρa *
            ᶜmp_tendencyʲs.:($$j).e_tot_hlpr
    end

    return nothing
end

###
### 1 Moment Microphysics
###

function set_microphysics_tendency_cache!(
    Y, p, mp1m::NonEquilibriumMicrophysics1M, _,
)
    (; dt) = p
    (; ᶜT, ᶜq_tot_nonneg, ᶜmp_tendency) = p.precomputed

    thp = CAP.thermodynamics_params(p.params)
    cmp = CAP.microphysics_1m_params(p.params)

    # Get specific humidities
    ᶜq_lcl = @. lazy(specific(Y.c.ρq_lcl, Y.c.ρ))
    ᶜq_icl = @. lazy(specific(Y.c.ρq_icl, Y.c.ρ))
    ᶜq_rai = @. lazy(specific(Y.c.ρq_rai, Y.c.ρ))
    ᶜq_sno = @. lazy(specific(Y.c.ρq_sno, Y.c.ρ))

    # Grid mean or quadrature sum over the SGS fluctuations
    # (writes into pre-allocated ᶜmp_tendency to avoid NamedTuple allocation)
    sgs_quad = p.atmos.sgs_quadrature
    nsubs = mp1m.n_substeps
    if not_quadrature(sgs_quad)
        @. ᶜmp_tendency = microphysics_tendencies_1m(
            Y.c.ρ, ᶜq_tot_nonneg, ᶜq_lcl, ᶜq_icl, ᶜq_rai, ᶜq_sno,
            ᶜT, cmp, thp, dt, nsubs,
        )
    else
        (; ᶜT′T′, ᶜq′q′) = p.precomputed # T-based variances from cache
        corr_Tq = correlation_Tq(p.params)
        @. ᶜmp_tendency = microphysics_tendencies_1m(
            BMT.Microphysics1Moment(), sgs_quad, cmp, thp, Y.c.ρ, ᶜT,
            ᶜq_tot_nonneg, ᶜq_lcl, ᶜq_icl, ᶜq_rai, ᶜq_sno,
            ᶜT′T′, ᶜq′q′, corr_Tq, dt,
        )
    end

    return nothing
end

function set_microphysics_tendency_cache!(
    Y, p, mp1m::NonEquilibriumMicrophysics1M, ::DiagnosticEDMFX,
)
    (; dt) = p
    (; ᶜT, ᶜq_tot_nonneg, ᶜmp_tendency) = p.precomputed

    thp = CAP.thermodynamics_params(p.params)
    cm1 = CAP.microphysics_1m_params(p.params)

    ### Updraft contribution is computed in the diagnostic EDMF integral loop

    ### Environment contribution
    ᶜq_lcl = @. lazy(specific(Y.c.ρq_lcl, Y.c.ρ))
    ᶜq_icl = @. lazy(specific(Y.c.ρq_icl, Y.c.ρ))
    ᶜq_rai = @. lazy(specific(Y.c.ρq_rai, Y.c.ρ))
    ᶜq_sno = @. lazy(specific(Y.c.ρq_sno, Y.c.ρ))

    # Grid mean or quadrature sum over the SGS fluctuations
    # (writes into pre-allocated ᶜmp_tendency to avoid NamedTuple allocation)
    sgs_quad = p.atmos.sgs_quadrature
    nsubs = mp1m.n_substeps
    if not_quadrature(sgs_quad)
        @. ᶜmp_tendency = microphysics_tendencies_1m(
            Y.c.ρ, ᶜq_tot_nonneg, ᶜq_lcl, ᶜq_icl, ᶜq_rai, ᶜq_sno, ᶜT, cm1, thp,
            dt, nsubs,
        )
    else
        (; ᶜT′T′, ᶜq′q′) = p.precomputed # T-based variances from cache
        corr_Tq = correlation_Tq(p.params)
        @. ᶜmp_tendency = microphysics_tendencies_1m(
            BMT.Microphysics1Moment(), sgs_quad, cm1, thp, Y.c.ρ, ᶜT,
            ᶜq_tot_nonneg, ᶜq_lcl, ᶜq_icl, ᶜq_rai, ᶜq_sno,
            ᶜT′T′, ᶜq′q′, corr_Tq, dt,
        )
    end

    return nothing
end

function set_microphysics_tendency_cache!(
    Y, p, mp1m::NonEquilibriumMicrophysics1M, tm::PrognosticEDMFX,
)
    (; dt) = p
    (; ᶜρʲs, ᶜTʲs, ᶜq_tot_nonnegʲs) = p.precomputed
    (; ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed
    (; ᶜmp_tendency⁰, ᶜmp_tendencyʲs) = p.precomputed

    thp = CAP.thermodynamics_params(p.params)
    cmp = CAP.microphysics_1m_params(p.params)

    n = n_mass_flux_subdomains(tm)
    nsubs = mp1m.n_substeps

    ### Updraft contribution
    for j in 1:n
        @. ᶜmp_tendencyʲs.:($$j) = microphysics_tendencies_1m(
            ᶜρʲs.:($$j), ᶜq_tot_nonnegʲs.:($$j),
            Y.c.sgsʲs.:($$j).q_lcl, Y.c.sgsʲs.:($$j).q_icl,
            Y.c.sgsʲs.:($$j).q_rai, Y.c.sgsʲs.:($$j).q_sno,
            ᶜTʲs.:($$j), cmp, thp, dt, nsubs,
        )
    end

    ### Environment contribution
    ᶜq_lcl⁰ = ᶜspecific_env_value(@name(q_lcl), Y, p)
    ᶜq_icl⁰ = ᶜspecific_env_value(@name(q_icl), Y, p)
    ᶜq_rai⁰ = ᶜspecific_env_value(@name(q_rai), Y, p)
    ᶜq_sno⁰ = ᶜspecific_env_value(@name(q_sno), Y, p)
    ᶜρ⁰ = @. lazy(
        TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰),
    )
    sgs_quad = p.atmos.sgs_quadrature
    if not_quadrature(sgs_quad)
        @. ᶜmp_tendency⁰ = microphysics_tendencies_1m(
            ᶜρ⁰, ᶜq_tot_nonneg⁰, ᶜq_lcl⁰, ᶜq_icl⁰, ᶜq_rai⁰, ᶜq_sno⁰,
            ᶜT⁰, cmp, thp, dt, nsubs,
        )
    else
        (; ᶜT′T′, ᶜq′q′) = p.precomputed # T-based variances from cache
        corr_Tq = correlation_Tq(p.params)
        @. ᶜmp_tendency⁰ = microphysics_tendencies_1m(
            BMT.Microphysics1Moment(), sgs_quad, cmp, thp, ᶜρ⁰, ᶜT⁰,
            ᶜq_tot_nonneg⁰, ᶜq_lcl⁰, ᶜq_icl⁰, ᶜq_rai⁰, ᶜq_sno⁰,
            ᶜT′T′, ᶜq′q′, corr_Tq, dt,
        )
    end

    return nothing
end

###
### 2-moment + P3 microphysics
###

function set_microphysics_tendency_cache!(Y, p, ::NonEquilibriumMicrophysics2M, _)
    (; dt) = p
    (; ᶜT, ᶜp, ᶜu, ᶜq_tot_nonneg, ᶜmp_tendency, ᶜlogλ) = p.precomputed
    @unpack_sup (;
        ᶜρ, ᶜρq_lcl, ᶜρn_lcl, ᶜρq_rai, ᶜρn_rai, ᶜρq_ice, ᶜρn_ice, ᶜρq_rim, ᶜρb_rim) = Y
    (; microphysics_tendency_timestepping) = p.atmos

    # get thermodynamics and microphysics parameters
    cmp = CAP.microphysics_2m_params(p.params)
    thp = CAP.thermodynamics_params(p.params)

    # Grid mean or quadrature sum over the SGS fluctuations
    # (writes into pre-allocated ᶜmp_tendency to avoid NamedTuple allocation)
    # TODO - looks like only grid-mean version is implemented now
    sgs_quad = something(p.atmos.sgs_quadrature, GridMeanSGS())
    # Cell-centered vertical velocity for w/p-dependent activation schemes
    # (Twomey, FixedARG). DiagnosticNc / NoActivation ignore them.
    ᶜw = @. lazy(w_component(Geometry.WVector(ᶜu)))
    @. ᶜmp_tendency = microphysics_tendencies_quadrature_and_limits_2m(
        # bulk tendency args
        sgs_quad, cmp, thp, ᶜρ, ᶜT, ᶜq_tot_nonneg,
        specific(ᶜρq_lcl, ᶜρ), specific(ᶜρn_lcl, ᶜρ),
        specific(ᶜρq_rai, ᶜρ), specific(ᶜρn_rai, ᶜρ),
        specific(ᶜρq_ice, ᶜρ), specific(ᶜρn_ice, ᶜρ),
        specific(ᶜρq_rim, ᶜρ), specific(ᶜρb_rim, ᶜρ),
        ᶜlogλ,
        # inpc_log_shift, w, p — w/p ride through to the activation scheme
        zero(ᶜρ), ᶜw, ᶜp,
        # tendency-limiter dispatch (Implicit / Explicit / Nothing)
        microphysics_tendency_timestepping, dt,
    )

    # Aerosol activation based on ARG 2000. Requires prescribed aerosols.
    # TODO - should be part of BMT
    # TODO - also only acting on grid mean
    if hasproperty(p, :tracers) && hasproperty(p.tracers, :prescribed_aerosols_field)
        pap = p.params.prescribed_aerosol_params
        acp = CAP.microphysics_cloud_params(p.params).activation

        # Get prescribed aerosol concentrations
        seasalt_num = p.scratch.ᶜtemp_scalar
        seasalt_mean_radius = p.scratch.ᶜtemp_scalar_2
        sulfate_num = p.scratch.ᶜtemp_scalar_3
        compute_prescribed_aerosol_properties!(
            seasalt_num, seasalt_mean_radius, sulfate_num,
            p.tracers.prescribed_aerosols_field, pap,
        )
        # Compute aerosol activation
        @. ᶜmp_tendency.dn_lcl_dt +=
            aerosol_activation_sources(
                acp, seasalt_num, seasalt_mean_radius, sulfate_num,
                specific(Y.c.ρq_tot, Y.c.ρ),
                specific(Y.c.ρq_lcl + Y.c.ρq_rai, Y.c.ρ),
                specific(Y.c.ρq_ice, Y.c.ρ),
                specific(Y.c.ρn_lcl + Y.c.ρn_rai, Y.c.ρ),
                Y.c.ρ, ᶜw, cmp, thp, ᶜT, ᶜp, dt, (pap,),
            )
    end
    return nothing
end

function set_microphysics_tendency_cache!(
    Y, p, ::NonEquilibriumMicrophysics2M, ::DiagnosticEDMFX,
)
    error("Not implemented yet")
    return nothing
end

function set_microphysics_tendency_cache!(
    Y, p, ::NonEquilibriumMicrophysics2M, tm::PrognosticEDMFX,
)
    (; dt) = p
    thp = CAP.thermodynamics_params(p.params)
    cm1p = CAP.microphysics_1m_params(p.params)
    cm2p = CAP.microphysics_2m_params(p.params)
    acp = CAP.microphysics_cloud_params(p.params).activation
    pap = p.params.prescribed_aerosol_params

    n = n_mass_flux_subdomains(tm)

    (; ᶜρʲs, ᶜTʲs, ᶜuʲs, ᶜq_tot_nonnegʲs) = p.precomputed
    (; ᶜu⁰, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed
    (; ᶜwₗʲs, ᶜwᵢʲs, ᶜwᵣʲs, ᶜwₛʲs, ᶜwₙₗʲs, ᶜwₙᵣʲs) = p.precomputed
    (; ᶜmp_tendency⁰, ᶜmp_tendencyʲs) = p.precomputed

    # Get prescribed aerosol concentrations
    seasalt_num = p.scratch.ᶜtemp_scalar_3
    seasalt_mean_radius = p.scratch.ᶜtemp_scalar_4
    sulfate_num = p.scratch.ᶜtemp_scalar_5
    if hasproperty(p, :tracers) &&
       hasproperty(p.tracers, :prescribed_aerosols_field)
        compute_prescribed_aerosol_properties!(
            seasalt_num, seasalt_mean_radius, sulfate_num,
            p.tracers.prescribed_aerosols_field, pap,
        )
    else
        @. seasalt_num = 0
        @. seasalt_mean_radius = 0
        @. sulfate_num = 0
    end

    ### Updraft contribution
    sgs_quad_j = something(p.atmos.sgs_quadrature, GridMeanSGS())
    for j in 1:n
        # Microphysics. `ᶜwʲ_pos` is the per-updraft vertical velocity
        # used by w/p-dependent activation schemes (Twomey, FixedARG);
        # DiagnosticNc / NoActivation ignore it. Updraft microphysics is
        # warm-only here (no P3 ice fields), so q_ice…logλ are zero-padded.
        ᶜwʲ = @. lazy(max(0, w_component(Geometry.WVector(ᶜuʲs.:($$j)))))
        ᶜρʲ = ᶜρʲs.:($j)
        @. ᶜmp_tendencyʲs.:($$j) = microphysics_tendencies_quadrature_and_limits_2m(
            sgs_quad_j, cm2p, thp, ᶜρʲ, ᶜTʲs.:($$j), ᶜq_tot_nonnegʲs.:($$j),
            Y.c.sgsʲs.:($$j).q_lcl, Y.c.sgsʲs.:($$j).n_lcl,
            Y.c.sgsʲs.:($$j).q_rai, Y.c.sgsʲs.:($$j).n_rai,
            zero(ᶜρʲ), zero(ᶜρʲ), zero(ᶜρʲ), zero(ᶜρʲ), zero(ᶜρʲ),  # q_ice…logλ
            zero(ᶜρʲ),                                              # inpc_log_shift
            ᶜwʲ, ᶜp,
            p.atmos.microphysics_tendency_timestepping, dt,
        )
        #ᶜmp_tendencyʲs.:($j).dq_ice_dt = 0
        #ᶜmp_tendencyʲs.:($j).dq_rim_dt = 0
        #ᶜmp_tendencyʲs.:($j).db_rim_dt = 0
        # Aerosol activation
        @. ᶜmp_tendencyʲs.:($$j).dn_lcl_dt += aerosol_activation_sources(
            acp, seasalt_num, seasalt_mean_radius, sulfate_num,
            ᶜq_tot_nonnegʲs.:($$j),
            Y.c.sgsʲs.:($$j).q_lcl + Y.c.sgsʲs.:($$j).q_rai,
            Y.c.sgsʲs.:($$j).q_icl,
            Y.c.sgsʲs.:($$j).n_lcl + Y.c.sgsʲs.:($$j).n_rai,
            ᶜρʲs.:($$j), ᶜwʲ, cm2p, thp, ᶜTʲs.:($$j), ᶜp, dt, (pap,),
        )
    end

    ### Environment contribution
    ᶜn_lcl⁰ = ᶜspecific_env_value(@name(n_lcl), Y, p)
    ᶜn_rai⁰ = ᶜspecific_env_value(@name(n_rai), Y, p)
    ᶜq_lcl⁰ = ᶜspecific_env_value(@name(q_lcl), Y, p)
    ᶜq_rai⁰ = ᶜspecific_env_value(@name(q_rai), Y, p)
    ᶜq_icl⁰ = ᶜspecific_env_value(@name(q_icl), Y, p)
    ᶜq_sno⁰ = ᶜspecific_env_value(@name(q_sno), Y, p)
    ᶜρ⁰ = @. lazy(
        TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰),
    )

    # Environment mean or quadrature sum over the SGS fluctuations
    # TODO - looks like only mean version is implemented now
    SG_quad = something(p.atmos.sgs_quadrature, GridMeanSGS())
    # Cell-centered environment vertical velocity for w/p-dependent
    # activation schemes (Twomey, FixedARG); also reused for the aerosol
    # activation block below.
    ᶜw⁰ = @. lazy(w_component(Geometry.WVector(ᶜu⁰)))
    # Single broadcast: bulk tendency + tendency limiter via
    # `microphysics_tendencies_quadrature_and_limits_2m`. Warm-only here, so
    # q_ice…logλ are zero-padded.
    @. ᶜmp_tendency⁰ = microphysics_tendencies_quadrature_and_limits_2m(
        SG_quad, cm2p, thp, ᶜρ⁰, ᶜT⁰, ᶜq_tot_nonneg⁰,
        ᶜq_lcl⁰, ᶜn_lcl⁰, ᶜq_rai⁰, ᶜn_rai⁰,
        zero(ᶜρ⁰), zero(ᶜρ⁰), zero(ᶜρ⁰), zero(ᶜρ⁰), zero(ᶜρ⁰),  # q_ice…logλ
        zero(ᶜρ⁰),                                              # inpc_log_shift
        ᶜw⁰, ᶜp,
        p.atmos.microphysics_tendency_timestepping, dt,
    )
    #@. ᶜmp_tendency⁰.dq_ice_dt = 0
    #@. ᶜmp_tendency⁰.dq_sno_dt = 0
    # Aerosol activation
    # TODO - make it part of BMT
    # TODO - should be included in limiting
    @. ᶜmp_tendency⁰.dn_lcl_dt += aerosol_activation_sources(
        acp, seasalt_num, seasalt_mean_radius, sulfate_num, ᶜq_tot_nonneg⁰,
        ᶜq_lcl⁰ + ᶜq_rai⁰, ᶜq_icl⁰ + ᶜq_sno⁰, ᶜn_lcl⁰ + ᶜn_rai⁰,
        ᶜρ⁰, ᶜw⁰, cm2p, thp, ᶜT⁰, ᶜp, dt, (pap,),
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
    microphysics_model::EquilibriumMicrophysics0M,
)
    (; ᶜT) = p.precomputed
    (; ᶜρ_dq_tot_dt, ᶜρ_de_tot_dt) = p.precomputed
    (; surface_rain_flux, surface_snow_flux) = p.precomputed
    (; col_integrated_precip_energy_tendency) = p.precomputed

    # update total column energy source for surface energy balance
    Operators.column_integral_definite!(
        col_integrated_precip_energy_tendency,
        ᶜρ_de_tot_dt,
    )
    # update surface precipitation fluxes in cache for coupler's use
    thermo_params = CAP.thermodynamics_params(p.params)
    T_freeze = TD.Parameters.T_freeze(thermo_params)
    FT = eltype(p.params)
    ᶜ3d_rain = @. lazy(ifelse(ᶜT >= T_freeze, ᶜρ_dq_tot_dt, FT(0)))
    ᶜ3d_snow = @. lazy(ifelse(ᶜT < T_freeze, ᶜρ_dq_tot_dt, FT(0)))
    Operators.column_integral_definite!(surface_rain_flux, ᶜ3d_rain)
    Operators.column_integral_definite!(surface_snow_flux, ᶜ3d_snow)
    return nothing
end

function set_precipitation_surface_fluxes!(Y, p,
    microphysics_model::Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M},
)
    (; surface_rain_flux, surface_snow_flux) = p.precomputed
    (; col_integrated_precip_energy_tendency) = p.precomputed
    (; ᶜwᵣ, ᶜwₗ, ᶜwᵢ, ᶜwₕhₜ) = p.precomputed
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    sfc_J = Fields.level(ᶠJ, Fields.half)
    sfc_space = axes(sfc_J)

    # Jacobian-weighted extrapolation from interior to surface, consistent with
    # the reconstruction of density on cell faces, ᶠρ = ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ
    sfc_lev(x) = Fields.Field(Fields.field_values(Fields.level(x, 1)), sfc_space)
    int_J = sfc_lev(ᶜJ)
    int_ρ = sfc_lev(Y.c.ρ)
    sfc_ρ = @. lazy(int_ρ * int_J / sfc_J)

    # Constant extrapolation to surface, consistent with simple downwinding
    # Temporary scratch variables are used here until CC.field_values supports <lazy> fields
    # Note: the 1M state uses `ρq_icl` for cloud ice while the 2M+P3 state
    # uses `ρq_ice`. Branch on the microphysics model to pick the right field.
    ᶜq_rai = @. p.scratch.ᶜtemp_scalar = specific(Y.c.ρq_rai, Y.c.ρ)
    ᶜq_lcl = @. p.scratch.ᶜtemp_scalar_2 = specific(Y.c.ρq_lcl, Y.c.ρ)
    if microphysics_model isa NonEquilibriumMicrophysics1M
        ᶜq_ice = @. p.scratch.ᶜtemp_scalar_3 = specific(Y.c.ρq_icl, Y.c.ρ)
    else
        ᶜq_ice = @. p.scratch.ᶜtemp_scalar_3 = specific(Y.c.ρq_ice, Y.c.ρ)
    end

    sfc_qᵣ = Fields.Field(Fields.field_values(Fields.level(ᶜq_rai, 1)), sfc_space)
    sfc_qₗ = Fields.Field(Fields.field_values(Fields.level(ᶜq_lcl, 1)), sfc_space)
    sfc_qᵢ = Fields.Field(Fields.field_values(Fields.level(ᶜq_ice, 1)), sfc_space)
    sfc_wᵣ = Fields.Field(Fields.field_values(Fields.level(ᶜwᵣ, 1)), sfc_space)
    sfc_wₗ = Fields.Field(Fields.field_values(Fields.level(ᶜwₗ, 1)), sfc_space)
    sfc_wᵢ = Fields.Field(Fields.field_values(Fields.level(ᶜwᵢ, 1)), sfc_space)
    sfc_wₕhₜ = Fields.Field(
        Fields.field_values(Fields.level(ᶜwₕhₜ.components.data.:1, 1)), sfc_space,
    )

    @. surface_rain_flux = sfc_ρ * (sfc_qᵣ * (-sfc_wᵣ) + sfc_qₗ * (-sfc_wₗ))
    @. col_integrated_precip_energy_tendency = sfc_ρ * (-sfc_wₕhₜ)
    if microphysics_model isa NonEquilibriumMicrophysics1M
        (; ᶜwₛ) = p.precomputed
        ᶜq_sno = @. p.scratch.ᶜtemp_scalar_4 = specific(Y.c.ρq_sno, Y.c.ρ)
        sfc_qₛ = Fields.Field(Fields.field_values(Fields.level(ᶜq_sno, 1)), sfc_space)
        sfc_wₛ = Fields.Field(Fields.field_values(Fields.level(ᶜwₛ, 1)), sfc_space)
        @. surface_snow_flux = sfc_ρ * (sfc_qₛ * (-sfc_wₛ) + sfc_qᵢ * (-sfc_wᵢ))
    else
        # 2M+P3 only prognose ice, no snow
        @. surface_snow_flux = sfc_ρ * (sfc_qᵢ * (-sfc_wᵢ))
        # TODO: Figure out what to do for ρq_rim, ρb_rim
    end

    return nothing
end

