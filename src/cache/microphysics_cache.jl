#####
##### Precomputed quantities for precipitation processes
#####

import CloudMicrophysics.MicrophysicsNonEq as CMNe
import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics.Microphysics2M as CM2
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT

import Thermodynamics as TD
import ClimaCore.Operators as Operators
import ClimaCore.Fields as Fields

const Iₗ = TD.internal_energy_liquid
const Iᵢ = TD.internal_energy_ice

"""
    internal_energy_func(name::MatrixFields.FieldName)

Return the internal-energy function appropriate to the phase of the tracer
`name`: `TD.internal_energy_liquid` for cloud liquid and rain, and
`TD.internal_energy_ice` for cloud ice and snow. Accepts the specific,
density-weighted, and `Y.c`-qualified spellings of each tracer name. Called from
`set_precipitation_velocities!`.
"""
internal_energy_func(
    ::Union{
        MatrixFields.FieldName{(:q_lcl,)},
        MatrixFields.FieldName{(:q_rai,)},
        MatrixFields.FieldName{(:ρq_lcl,)},
        MatrixFields.FieldName{(:ρq_rai,)},
        MatrixFields.FieldName{(:c, :ρq_lcl)},
        MatrixFields.FieldName{(:c, :ρq_rai)},
    },
) = TD.internal_energy_liquid
internal_energy_func(
    ::Union{
        MatrixFields.FieldName{(:q_icl,)},
        MatrixFields.FieldName{(:q_sno,)},
        MatrixFields.FieldName{(:ρq_icl,)},
        MatrixFields.FieldName{(:ρq_sno,)},
        MatrixFields.FieldName{(:c, :ρq_icl)},
        MatrixFields.FieldName{(:c, :ρq_sno)},
    },
) = TD.internal_energy_ice

"""
    Kin(ᶜw_precip, ᶜu_air)

Return a lazy broadcast of the specific kinetic energy of falling cloud
condensate or precipitation, combining the (downward) terminal velocity with
the air velocity.

# Arguments

  - `ᶜw_precip`: Terminal velocity of cloud condensate or precipitation [m/s].
  - `ᶜu_air`: Air velocity [m/s].
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

Return a generic numerical-zero threshold. Used for variance floors, σ guards,
and density-weighted mass checks, anywhere the exact value does not matter as
long as it is small but safely above underflow.
"""
ϵ_numerics(FT) = cbrt(floatmin(FT))

"""
    set_precipitation_velocities!(Y, p, microphysics_model, turbconv_model)

Update the grid-mean precipitation terminal velocities and cloud sedimentation
velocities, and their contributions to total water and energy advection.

Dispatches on `microphysics_model` (`NonEquilibriumMicrophysics1M`,
`NonEquilibriumMicrophysics2M`, `NonEquilibriumMicrophysics2MP3`) and, for the
1M and 2M schemes, on `turbconv_model::PrognosticEDMFX`; the default method
zeroes the advection contributions. Returns `nothing`.

Mutates in `p.precomputed`: the velocity fields available for the scheme (`ᶜwₗ`,
`ᶜwᵢ`, `ᶜwᵣ`, `ᶜwₛ`, plus number-weighted `ᶜwₙₗ`/`ᶜwₙᵣ` for 2M and `ᶜwnᵢ`/`ᶜlogλ`
for P3 ice) and the advection contributions `ᶜwₜqₜ` and `ᶜwₕhₜ`.

For `PrognosticEDMFX`, the per-subdomain velocities (`ᶜwₗʲs`, `ᶜwᵢʲs`, `ᶜwᵣʲs`,
`ᶜwₛʲs`, ...) are also computed, and the grid-mean velocities are mass-weighted
averages over the subdomains, so that the grid-scale flux equals the sum of the
sub-grid-scale fluxes.
"""
function set_precipitation_velocities!(Y, p, _, _)
    (; ᶜwₜqₜ, ᶜwₕhₜ) = p.precomputed
    @. ᶜwₜqₜ = Geometry.WVector(0)
    @. ᶜwₕhₜ = Geometry.WVector(0)
    return nothing
end
function set_precipitation_velocities!(
    Y,
    p,
    microphysics_model::NonEquilibriumMicrophysics1M,
    _,
)
    (; ᶜwₗ, ᶜwᵢ, ᶜwᵣ, ᶜwₛ, ᶜwₜqₜ, ᶜwₕhₜ, ᶜT, ᶜu) = p.precomputed
    (; ᶜΦ) = p.core
    atmos = p.atmos
    params = p.params
    cmc = CAP.microphysics_cloud_params(params)
    cmp = CAP.microphysics_1m_params(params)
    thp = CAP.thermodynamics_params(params)

    # scratch for adding energy fluxes over subdomains
    ᶜρwₕhₜ = p.scratch.ᶜtemp_scalar
    @. ᶜρwₕhₜ = 0

    terminal_velocity_function(name, ρ, q) = terminal_velocity(
        microphysics_model,
        velocity_mode(atmos, name),
        name,
        params,
        cmc,
        cmp,
        ρ,
        q,
    )

    microphysics_tracers = (
        (@name(q_lcl), @name(ᶜwₗ)),
        (@name(q_icl), @name(ᶜwᵢ)),
        (@name(q_rai), @name(ᶜwᵣ)),
        (@name(q_sno), @name(ᶜwₛ)),
    )
    MatrixFields.unrolled_foreach(microphysics_tracers) do (χ_name, w_name)
        MatrixFields.has_field(Y.c, get_ρχ_name(χ_name)) || return

        e_int_func = internal_energy_func(χ_name)

        ᶜρχ = MatrixFields.get_field(Y.c, get_ρχ_name(χ_name))
        ᶜw = MatrixFields.get_field(p.precomputed, w_name)
        @. ᶜw = terminal_velocity_function(χ_name, Y.c.ρ, max(zero(Y.c.ρ), ᶜρχ / Y.c.ρ))

        @. ᶜρwₕhₜ += ᶜw * ᶜρχ * (e_int_func(thp, ᶜT) + ᶜΦ + $(Kin(ᶜw, ᶜu)))
    end

    # compute their contributions to energy and total water advection
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
    atmos = p.atmos
    params = p.params

    cmc = CAP.microphysics_cloud_params(params)
    cmp = CAP.microphysics_1m_params(params)
    thp = CAP.thermodynamics_params(params)

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
    @. ᶜρwₕhₜ = 0

    # Compute gs sedimentation velocity based on subdomain velocities
    # assuming grid-scale flux equals to the sum of sub-grid-scale fluxes.

    terminal_velocity_function(name, ρ, q) = terminal_velocity(
        microphysics_model,
        velocity_mode(atmos, name),
        name,
        params,
        cmc,
        cmp,
        ρ,
        q,
    )

    microphysics_tracers = (
        (@name(q_lcl), @name(ᶜwₗʲs.:(1)), @name(ᶜwₗ)),
        (@name(q_icl), @name(ᶜwᵢʲs.:(1)), @name(ᶜwᵢ)),
        (@name(q_rai), @name(ᶜwᵣʲs.:(1)), @name(ᶜwᵣ)),
        (@name(q_sno), @name(ᶜwₛʲs.:(1)), @name(ᶜwₛ)),
    )
    MatrixFields.unrolled_foreach(microphysics_tracers) do (χ_name, wʲ_name, w_name)
        MatrixFields.has_field(Y.c.sgsʲs.:(1), χ_name) || return

        e_int_func = internal_energy_func(χ_name)
        ᶜw = MatrixFields.get_field(p.precomputed, w_name)

        ᶜχ⁰ = ᶜspecific_env_value(χ_name, Y, p)
        ᶜρa⁰χ⁰ = @. lazy(max(zero(Y.c.ρ), ᶜρa⁰) * max(zero(Y.c.ρ), ᶜχ⁰))
        @. ᶜρχ = ᶜρa⁰χ⁰
        @. ᶜw = ᶜρa⁰χ⁰ * terminal_velocity_function(χ_name, ᶜρ⁰, ᶜχ⁰)
        @. ᶜimplied_env_mass_flux = 0
        # add updraft contributions
        for j in 1:n
            ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:(1), χ_name)
            ᶜwʲ = MatrixFields.get_field(p.precomputed, wʲ_name)

            ᶜρaʲχʲ = @. lazy(
                max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).ρa) *
                max(zero(Y.c.ρ), ᶜχʲ),
            )
            @. ᶜρχ += ᶜρaʲχʲ
            @. ᶜwʲ = terminal_velocity_function(χ_name, ᶜρʲs.:($$j), max(zero(Y.c.ρ), ᶜχʲ))
            @. ᶜw += ᶜρaʲχʲ * ᶜwʲ
            @. ᶜimplied_env_mass_flux -=
                Y.c.sgsʲs.:($$j).ρa * ᶜχʲ * ᶜwʲ

            # Add contributions to energy and total water advection
            @. ᶜρwₕhₜ +=
                Y.c.sgsʲs.:($$j).ρa * ᶜwʲ * ᶜχʲ * (e_int_func(thp, ᶜTʲs.:($$j)) + ᶜΦ)
        end
        # average
        @. ᶜw = gs_terminal_velocity(
            microphysics_model,
            velocity_mode(atmos, χ_name),
            χ_name,
            params,
            ᶜw,
            ᶜρχ,
        )
        @. ᶜimplied_env_mass_flux += MatrixFields.get_field(Y.c, get_ρχ_name(χ_name)) * ᶜw
        # contribution of env sedimentation to htot
        @. ᶜρwₕhₜ += ᶜimplied_env_mass_flux * (e_int_func(thp, ᶜT⁰) + ᶜΦ)
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
function set_precipitation_velocities!(
    Y,
    p,
    microphysics_model::NonEquilibriumMicrophysics2M,
    _,
)
    (; ᶜwₗ, ᶜwᵢ, ᶜwᵣ, ᶜwₛ, ᶜwₙₗ, ᶜwₙᵣ, ᶜwₜqₜ, ᶜwₕhₜ, ᶜT, ᶜu) = p.precomputed
    (; ᶜΦ) = p.core

    cmc = CAP.microphysics_cloud_params(p.params)
    cm1p = CAP.microphysics_1m_params(p.params)
    cm2p = CAP.microphysics_2m_params(p.params)
    thp = CAP.thermodynamics_params(p.params)

    # compute the precipitation terminal velocity [m/s]
    # TODO sedimentation of snow is based on the 1M scheme
    @. ᶜwₙᵣ = getindex(
        CM2.rain_terminal_velocity(
            cm2p.warm_rain.seifert_beheng,
            cmc.Ch2022.rain,
            max(zero(Y.c.ρ), specific(Y.c.ρq_rai, Y.c.ρ)),
            Y.c.ρ,
            max(zero(Y.c.ρ), Y.c.ρn_rai),
        ),
        1,
    )
    @. ᶜwᵣ = getindex(
        CM2.rain_terminal_velocity(
            cm2p.warm_rain.seifert_beheng,
            cmc.Ch2022.rain,
            max(zero(Y.c.ρ), specific(Y.c.ρq_rai, Y.c.ρ)),
            Y.c.ρ,
            max(zero(Y.c.ρ), Y.c.ρn_rai),
        ),
        2,
    )
    @. ᶜwₛ = CM1.terminal_velocity(
        cm1p.precip.snow,
        cm1p.terminal_velocity.snow,
        Y.c.ρ,
        max(zero(Y.c.ρ), specific(Y.c.ρq_sno, Y.c.ρ)),
    )
    # compute sedimentation velocity for cloud condensate [m/s]
    # TODO sedimentation of ice is based on the 1M scheme
    @. ᶜwₙₗ = getindex(
        CM2.cloud_terminal_velocity(
            cm2p.warm_rain.seifert_beheng.pdf_c,
            cmc.stokes,
            max(zero(Y.c.ρ), specific(Y.c.ρq_lcl, Y.c.ρ)),
            Y.c.ρ,
            max(zero(Y.c.ρ), Y.c.ρn_lcl),
        ),
        1,
    )
    @. ᶜwₗ = getindex(
        CM2.cloud_terminal_velocity(
            cm2p.warm_rain.seifert_beheng.pdf_c,
            cmc.stokes,
            max(zero(Y.c.ρ), specific(Y.c.ρq_lcl, Y.c.ρ)),
            Y.c.ρ,
            max(zero(Y.c.ρ), Y.c.ρn_lcl),
        ),
        2,
    )
    @. ᶜwᵢ = CMNe.terminal_velocity(
        cmc.ice,
        cmc.Ch2022.small_ice,
        Y.c.ρ,
        max(zero(Y.c.ρ), specific(Y.c.ρq_icl, Y.c.ρ)),
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
    microphysics_model::NonEquilibriumMicrophysics2M,
    turbconv_model::PrognosticEDMFX,
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
    ᶜq_rai⁰ = ᶜspecific_env_value(@name(q_rai), Y, p)
    ᶜn_rai⁰ = ᶜspecific_env_value(@name(n_rai), Y, p)
    @. ᶜw⁰ = getindex(
        CM2.rain_terminal_velocity(
            cm2p.warm_rain.seifert_beheng,
            cmc.Ch2022.rain,
            max(zero(Y.c.ρ), ᶜq_rai⁰),
            ᶜρ⁰,
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

    ###
    ### Snow
    ####
    ᶜq_sno⁰ = ᶜspecific_env_value(@name(q_sno), Y, p)
    # TODO sedimentation of snow is based on the 1M scheme
    @. ᶜw⁰ = CM1.terminal_velocity(
        cm1p.precip.snow,
        cm1p.terminal_velocity.snow,
        ᶜρ⁰,
        max(zero(Y.c.ρ), ᶜq_sno⁰),
    )
    @. ᶜwₛ = ᶜρa⁰ * ᶜq_sno⁰ * ᶜw⁰
    @. ᶜρχ = max(zero(Y.c.ρ), ᶜρa⁰ * ᶜq_sno⁰)
    for j in 1:n
        @. ᶜwₛʲs.:($$j) = CM1.terminal_velocity(
            cm1p.precip.snow,
            cm1p.terminal_velocity.snow,
            ᶜρʲs.:($$j),
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_sno),
        )
        @. ᶜwₛ += Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_sno * ᶜwₛʲs.:($$j)
        @. ᶜρχ += max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_sno)
    end
    @. ᶜwₛ = ifelse(ᶜρχ > FT(0), ᶜwₛ / ᶜρχ, FT(0))
    # contribution of env snow advection to htot advection
    @. ᶜρwₕhₜ += ᶜρa⁰ * ᶜq_sno⁰ * ᶜw⁰ * (Iᵢ(thp, ᶜT⁰) + ᶜΦ)

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
function set_precipitation_velocities!(
    Y, p, ::NonEquilibriumMicrophysics2MP3, _,
)
    ## liquid quantities (2M warm rain)
    (; ᶜwₗ, ᶜwᵣ, ᶜwnₗ, ᶜwnᵣ, ᶜwₜqₜ, ᶜwₕhₜ, ᶜT, ᶜu) = p.precomputed
    (; ᶜΦ) = p.core

    (; ρ, ρq_lcl, ρn_lcl, ρq_rai, ρn_rai) = Y.c
    params_2mp3 = CAP.microphysics_2mp3_params(p.params)
    thp = CAP.thermodynamics_params(p.params)
    cmc = CAP.microphysics_cloud_params(p.params)

    # Access 2M warm rain params from unified container
    sb = params_2mp3.warm_rain.seifert_beheng
    rtv = cmc.Ch2022.rain  # Rain terminal velocity from cloud_params

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
            sb.pdf_c, cmc.stokes,
            max(zero(ρ), specific(ρq_lcl, ρ)),
            ρ, max(zero(ρ), ρn_lcl),
        ),
    )
    @. ᶜwnₗ = getindex(ᶜliq_w_terms, 1)
    @. ᶜwₗ = getindex(ᶜliq_w_terms, 2)

    ## Ice quantities
    (; ρq_icl, ρn_ice, ρq_rim, ρb_rim) = Y.c
    (; ᶜwᵢ) = p.precomputed

    # P3 ice params from unified container
    p3_ice = params_2mp3.ice

    # Number- and mass weighted ice terminal velocity [m/s]
    # Calculate terminal velocities
    (; ᶜlogλ, ᶜwnᵢ) = p.precomputed
    use_aspect_ratio = true  # TODO: Make a config option
    ᶜF_rim = @. lazy(ρq_rim / ρq_icl)
    ᶜρ_rim = @. lazy(ρq_rim / ρb_rim)
    ᶜstate_p3 = @. lazy(CMP3.P3State(p3_ice.scheme,
        max(0, ρq_icl), max(0, ρn_ice), ᶜF_rim, ᶜρ_rim,
    ))
    @. ᶜlogλ = CMP3.get_distribution_logλ(ᶜstate_p3)
    args = (p3_ice.terminal_velocity, ρ, ᶜstate_p3, ᶜlogλ)
    @. ᶜwnᵢ = CMP3.ice_terminal_velocity_number_weighted(args...; use_aspect_ratio)
    @. ᶜwᵢ = CMP3.ice_terminal_velocity_mass_weighted(args...; use_aspect_ratio)

    # compute their contributions to energy and total water advection
    @. ᶜwₜqₜ = Geometry.WVector(ᶜwₗ * ρq_lcl + ᶜwᵢ * ρq_icl + ᶜwᵣ * ρq_rai) / ρ
    @. ᶜwₕhₜ =
        Geometry.WVector(
            ᶜwₗ * ρq_lcl * (Iₗ(thp, ᶜT) + ᶜΦ + $(Kin(ᶜwₗ, ᶜu))) +
            ᶜwᵢ * ρq_icl * (Iᵢ(thp, ᶜT) + ᶜΦ + $(Kin(ᶜwᵢ, ᶜu))) +
            ᶜwᵣ * ρq_rai * (Iₗ(thp, ᶜT) + ᶜΦ + $(Kin(ᶜwᵣ, ᶜu))),
        ) / ρ
    return nothing
end

"""
    update_implicit_microphysics_cache!(Y, p, microphysics_model, turbconv_model)

Refresh the microphysics precomputed quantities that depend on the current Newton
iterate `Y`. Called from `set_implicit_precomputed_quantities!` at every Newton
iteration of the implicit solve, and only when
`p.atmos.microphysics_tendency_timestepping == Implicit()`. Returns `nothing`.

All schemes freeze the specific microphysics tendencies (`ᶜmp_tendency` and its
subdomain counterparts) computed in the explicit stage; only density-weighted
source terms and surface fluxes are refreshed here. The refreshed fields have
Dual-typed copies in `implicit_precomputed_quantities`, so autodiff can write
into them safely.

Dispatch on `microphysics_model` and `turbconv_model`:

  - `EquilibriumMicrophysics0M`: rewrites `ᶜρ_dq_tot_dt = ρ * ᶜmp_tendency.dq_tot_dt`
    and `ᶜρ_de_tot_dt = ᶜρ_dq_tot_dt * ᶜmp_tendency.e_tot_hlpr`, then updates the
    surface fluxes.
  - `EquilibriumMicrophysics0M` with `PrognosticEDMFX`: re-aggregates the frozen
    per-subdomain specific tendencies `ᶜmp_tendency⁰` and `ᶜmp_tendencyʲs` with the
    current `ρa⁰` and `Y.c.sgsʲs.:(j).ρa`, then updates the surface fluxes.
  - `NonEquilibriumMicrophysics` (1M, 2M, 2M+P3): calls
    `set_precipitation_surface_fluxes!` only. Density weighting of the specific
    tendencies is applied at tendency-evaluation time in `tendency.jl`.
  - Default: no-op (microphysics not active).
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
    Y,
    p,
    mm::NonEquilibriumMicrophysics1M,
    tm,
)
    # Recompute the sedimentation velocities from the current implicit-solver state.
    # This is expecially important for PrognosticEDMFX `ᶜwᵣʲs`.
    # Leaving updraft velocities at their explicit-stage values treats the
    # nonlinear part of the sedimentation flux explicitly.
    # No number of Newton iterations can correct a field that nothing updates.
    if p.atmos.terminal_velocity_liquid isa DiagnosticTerminalVelocity ||
       p.atmos.terminal_velocity_ice isa DiagnosticTerminalVelocity ||
       p.atmos.terminal_velocity_rain isa DiagnosticTerminalVelocity ||
       p.atmos.terminal_velocity_snow isa DiagnosticTerminalVelocity
        set_precipitation_velocities!(Y, p, mm, tm)
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

Compute and cache the specific microphysics tendencies for the current state `Y`.
Returns `nothing`.

Writes `p.precomputed.ᶜmp_tendency` on the non-EDMF path, and
`p.precomputed.ᶜmp_tendency⁰` together with `p.precomputed.ᶜmp_tendencyʲs` on the
`PrognosticEDMFX` path. For `EquilibriumMicrophysics0M`, the density-weighted
sources `ᶜρ_dq_tot_dt` and `ᶜρ_de_tot_dt` are also written here; for the other
schemes the density weighting is applied at tendency-evaluation time in
`tendency.jl`. The 2M+P3 method additionally writes `ᶜScoll`.

**Dispatch table** (`microphysics_model` × `turbconv_model`):

| Model    | Nothing / default      | PrognosticEDMFX  |
|:-------- |:---------------------- |:---------------- |
| DryModel | no-op                  | no-op (fallback) |
| 0M       | grid-mean (± SGS quad) | EDMF-weighted    |
| 1M       | grid-mean (± SGS quad) | EDMF-weighted    |
| 2M       | grid-mean              | EDMF-weighted    |
| 2MP3     | grid-mean (no EDMF)    | —                |

**Non-EDMF path** computes microphysics on the grid-mean state, with an optional sum over
SGS quadrature points (controlled by `p.atmos.sgs_quadrature`) to sample subgrid variability.

**EDMF path** computes tendencies separately for updrafts and the environment:

*Updrafts* use direct BMT evaluation (no SGS quadrature) because:

 1. Updrafts are coherent turbulent structures with more homogeneous thermodynamic properties.
 2. Updraft area fraction is usually small (~1-10%), so SGS variance within updrafts has limited
    impact on the grid-mean tendency.

*Environment* uses SGS quadrature integration (when `sgs_quadrature` is configured) because
the environment dominates the grid-mean variance. The quadrature captures subgrid-scale
fluctuations in temperature and moisture, which is important for threshold processes like
condensation/evaporation at cloud edges.

For `EquilibriumMicrophysics0M` under EDMF, the grid-mean source is the
area-weighted sum
`ᶜρ_dq_tot_dt = ᶜmp_tendency⁰.dq_tot_dt * ρa⁰ + Σⱼ ᶜmp_tendencyʲs.:(j).dq_tot_dt * ρaʲ`.

Reads the subdomain thermodynamic state (`ᶜρʲs`, `ᶜTʲs`, `ᶜq_tot_nonnegʲs`,
`ᶜT⁰`, ...) and, when quadrature is active, the SGS (co)variances `ᶜT′T′`,
`ᶜq′q′`, and `ᶜsgs_moments`. Uses `p.scratch` fields for the environment inputs.

See also `set_precipitation_velocities!` and `set_precipitation_surface_fluxes!`.
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

# Register cap for the 1-moment microphysics broadcasts (GPU only; no-op on
# CPU). The per-point function inlines ~16 `pow` and ~12 `exp`/`log` bodies per
# substep and compiles to 220-255 registers per thread, which limits the kernel
# to 8 warps per SM; Nsight Compute shows 58% of issue cycles with no eligible
# warp. Capping the allocator at 128 registers trades a little spilling for
# occupancy. Measured on an A100 (he16ze63, Float32): updraft kernel 2.1 -> 1.4
# ms (bitwise-identical results); SGS-quadrature environment kernel 22.7 -> 12.6
# ms together with the single-loop `sum_over_quadrature_points` (results differ
# only by summation order, ~1e-13). Caps of 96 and 64 were slower again. The
# cap only pays when the kernel has no other local-memory traffic: a dynamic
# index into a broadcast argument (see `static_select`) puts the whole argument
# tuple in local memory and makes the cap counterproductive. Re-measure before
# changing.
const MICROPHYSICS_1M_KERNEL_MAXREGS = 128

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
    nsubs_quad = mp1m.n_substeps_quad
    if not_quadrature(sgs_quad)
        ClimaCore.DataLayouts.with_kernel_maxregs(MICROPHYSICS_1M_KERNEL_MAXREGS) do
            @. ᶜmp_tendency = microphysics_tendencies_1m(
                Y.c.ρ, ᶜq_tot_nonneg, ᶜq_lcl, ᶜq_icl, ᶜq_rai, ᶜq_sno,
                ᶜT, cmp, thp, dt, nsubs,
            )
        end
    else
        (; ᶜT′T′, ᶜq′q′, ᶜsgs_moments) = p.precomputed
        corr_Tq = correlation_Tq(p.params)
        α = sgs_variance_fidelity(CAP.cloud_fraction_steepness_scale(p.params))
        ClimaCore.DataLayouts.with_kernel_maxregs(MICROPHYSICS_1M_KERNEL_MAXREGS) do
            @. ᶜmp_tendency = microphysics_tendencies_1m(
                BMT.Microphysics1Moment(), sgs_quad, cmp, thp, Y.c.ρ, ᶜT,
                ᶜq_tot_nonneg, ᶜq_lcl, ᶜq_icl, ᶜq_rai, ᶜq_sno,
                ᶜT′T′, ᶜq′q′, corr_Tq, ᶜsgs_moments.λ_lagrange, α,
                dt, nsubs_quad,
            )
        end
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
    nsubs_quad = mp1m.n_substeps_quad

    ### Updraft contribution
    ClimaCore.DataLayouts.with_kernel_maxregs(MICROPHYSICS_1M_KERNEL_MAXREGS) do
        for j in 1:n
            @. ᶜmp_tendencyʲs.:($$j) = microphysics_tendencies_1m(
                ᶜρʲs.:($$j), ᶜq_tot_nonnegʲs.:($$j),
                Y.c.sgsʲs.:($$j).q_lcl, Y.c.sgsʲs.:($$j).q_icl,
                Y.c.sgsʲs.:($$j).q_rai, Y.c.sgsʲs.:($$j).q_sno,
                ᶜTʲs.:($$j), cmp, thp, dt, nsubs,
            )
        end
    end

    ### Environment contribution
    # Materialize the environment inputs into scratch fields so the SGS-quadrature
    # microphysics kernel below reads them instead of recomputing them inline at
    # every grid point. `air_density` and the `specific_env_value` reconstructions
    # (grid mean minus updrafts) are otherwise fused into that broadcast and inflate
    # its fixed, per-point cost; hoisting them into cheap separate kernels is
    # numerically identical and materially faster for this register-bound kernel.
    ᶜρ⁰ = p.scratch.ᶜtemp_scalar
    ᶜq_lcl⁰ = p.scratch.ᶜtemp_scalar_2
    ᶜq_icl⁰ = p.scratch.ᶜtemp_scalar_3
    ᶜq_rai⁰ = p.scratch.ᶜtemp_scalar_4
    ᶜq_sno⁰ = p.scratch.ᶜtemp_scalar_5
    @. ᶜρ⁰ = TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰)
    ᶜq_lcl⁰ .= ᶜspecific_env_value(@name(q_lcl), Y, p)
    ᶜq_icl⁰ .= ᶜspecific_env_value(@name(q_icl), Y, p)
    ᶜq_rai⁰ .= ᶜspecific_env_value(@name(q_rai), Y, p)
    ᶜq_sno⁰ .= ᶜspecific_env_value(@name(q_sno), Y, p)
    sgs_quad = p.atmos.sgs_quadrature
    if not_quadrature(sgs_quad)
        ClimaCore.DataLayouts.with_kernel_maxregs(MICROPHYSICS_1M_KERNEL_MAXREGS) do
            @. ᶜmp_tendency⁰ = microphysics_tendencies_1m(
                ᶜρ⁰, ᶜq_tot_nonneg⁰, ᶜq_lcl⁰, ᶜq_icl⁰, ᶜq_rai⁰, ᶜq_sno⁰,
                ᶜT⁰, cmp, thp, dt, nsubs,
            )
        end
    else
        (; ᶜT′T′, ᶜq′q′, ᶜsgs_moments) = p.precomputed
        corr_Tq = correlation_Tq(p.params)
        α = sgs_variance_fidelity(CAP.cloud_fraction_steepness_scale(p.params))
        # The liquid fraction `λ` and the linearized SGS saturation-excess mean
        # `mu_S` are held fixed across the quadrature (they depend only on the mean
        # state), so compute them once here and pass them in, instead of recomputing
        # them inside every quadrature evaluation.
        ᶜλ⁰ = p.scratch.ᶜtemp_scalar_6
        ᶜmu_S⁰ = p.scratch.ᶜtemp_scalar_7
        @. ᶜλ⁰ = TD.liquid_fraction(thp, ᶜT⁰, max(0, ᶜq_lcl⁰), max(0, ᶜq_icl⁰))
        @. ᶜmu_S⁰ = ᶜq_tot_nonneg⁰ - TD.q_vap_saturation(thp, ᶜT⁰, ᶜρ⁰)
        ClimaCore.DataLayouts.with_kernel_maxregs(MICROPHYSICS_1M_KERNEL_MAXREGS) do
            @. ᶜmp_tendency⁰ = microphysics_tendencies_1m(
                BMT.Microphysics1Moment(), sgs_quad, cmp, thp, ᶜρ⁰, ᶜT⁰,
                ᶜq_tot_nonneg⁰, ᶜq_lcl⁰, ᶜq_icl⁰, ᶜq_rai⁰, ᶜq_sno⁰,
                ᶜT′T′, ᶜq′q′, corr_Tq, ᶜsgs_moments.λ_lagrange, α,
                dt, nsubs_quad, ᶜλ⁰, ᶜmu_S⁰,
            )
        end
    end

    return nothing
end

###
### 2-moment + P3 microphysics
###

function set_microphysics_tendency_cache!(
    Y, p, ::NonEquilibriumMicrophysics2M, _,
)
    (; dt) = p
    (; ᶜT, ᶜp, ᶜu, ᶜq_tot_nonneg, ᶜmp_tendency) = p.precomputed

    # get thermodynamics and microphysics params
    cmp = CAP.microphysics_2m_params(p.params)
    thp = CAP.thermodynamics_params(p.params)

    # Get specific quantities
    ᶜq_lcl = @. lazy(specific(Y.c.ρq_lcl, Y.c.ρ))
    ᶜq_rai = @. lazy(specific(Y.c.ρq_rai, Y.c.ρ))
    ᶜn_lcl = @. lazy(specific(Y.c.ρn_lcl, Y.c.ρ))
    ᶜn_rai = @. lazy(specific(Y.c.ρn_rai, Y.c.ρ))

    # Grid mean or quadrature sum over the SGS fluctuations
    # (writes into pre-allocated ᶜmp_tendency to avoid NamedTuple allocation)
    # TODO - looks like only grid-mean version is implemented now
    sgs_quad = something(p.atmos.sgs_quadrature, GridMeanSGS())
    @. ᶜmp_tendency = microphysics_tendencies_quadrature_2m(
        sgs_quad, cmp, thp, Y.c.ρ, ᶜT,
        ᶜq_tot_nonneg, ᶜq_lcl, ᶜn_lcl, ᶜq_rai, ᶜn_rai,
    )
    # Apply the limiter
    apply_2m_tendency_limits!(
        ᶜmp_tendency, p.atmos.microphysics_tendency_timestepping,
        ᶜq_lcl, ᶜn_lcl, ᶜq_rai, ᶜn_rai, dt,
    )
    #TODO - implement cold processes via P3
    @. ᶜmp_tendency.dq_ice_dt = 0
    @. ᶜmp_tendency.dq_rim_dt = 0
    @. ᶜmp_tendency.db_rim_dt = 0

    # Aerosol activation based on ARG 2000. Requires prescribed aerosols.
    # TODO - should be part of BMT
    # TODO - also only acting on grid mean
    if hasproperty(p, :tracers) &&
       hasproperty(p.tracers, :prescribed_aerosols_field)

        # Get aerosol parameters and vertical velocity
        pap = p.params.prescribed_aerosol_params
        acp = CAP.microphysics_cloud_params(p.params).activation
        ᶜw = @. lazy(w_component(Geometry.WVector(ᶜu)))

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
                specific(Y.c.ρq_icl + Y.c.ρq_sno, Y.c.ρ),
                specific(Y.c.ρn_lcl + Y.c.ρn_rai, Y.c.ρ),
                Y.c.ρ, ᶜw, cmp, thp, ᶜT, ᶜp, dt, (pap,),
            )
    end
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
    for j in 1:n
        # Microphysics
        compute_2m_precipitation_tendencies!(
            ᶜmp_tendencyʲs.:($j), ᶜρʲs.:($j), ᶜq_tot_nonnegʲs.:($j),
            Y.c.sgsʲs.:($j).q_lcl, Y.c.sgsʲs.:($j).n_lcl,
            Y.c.sgsʲs.:($j).q_rai, Y.c.sgsʲs.:($j).n_rai,
            ᶜTʲs.:($j), dt, cm2p, thp,
            p.atmos.microphysics_tendency_timestepping,
        )
        #ᶜmp_tendencyʲs.:($j).dq_ice_dt = 0
        #ᶜmp_tendencyʲs.:($j).dq_rim_dt = 0
        #ᶜmp_tendencyʲs.:($j).db_rim_dt = 0
        # Aerosol activation
        ᶜwʲ = @. lazy(max(0, w_component(Geometry.WVector(ᶜuʲs.:($$j)))))
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
    @. ᶜmp_tendency⁰ = microphysics_tendencies_quadrature_2m(
        SG_quad, cm2p, thp, ᶜρ⁰, ᶜT⁰, ᶜq_tot_nonneg⁰,
        ᶜq_lcl⁰, ᶜn_lcl⁰, ᶜq_rai⁰, ᶜn_rai⁰,
    )
    # Apply the limiter
    apply_2m_tendency_limits!(
        ᶜmp_tendency⁰, p.atmos.microphysics_tendency_timestepping,
        ᶜq_lcl⁰, ᶜn_lcl⁰, ᶜq_rai⁰, ᶜn_rai⁰, dt,
    )
    #@. ᶜmp_tendency⁰.dq_ice_dt = 0
    #@. ᶜmp_tendency⁰.dq_sno_dt = 0
    # Aerosol activation
    # TODO - make it part of BMT
    # TODO - should be included in limiting
    ᶜw⁰ = @. lazy(w_component(Geometry.WVector(ᶜu⁰)))
    @. ᶜmp_tendency⁰.dn_lcl_dt += aerosol_activation_sources(
        acp, seasalt_num, seasalt_mean_radius, sulfate_num, ᶜq_tot_nonneg⁰,
        ᶜq_lcl⁰ + ᶜq_rai⁰, ᶜq_icl⁰ + ᶜq_sno⁰, ᶜn_lcl⁰ + ᶜn_rai⁰,
        ᶜρ⁰, ᶜw⁰, cm2p, thp, ᶜT⁰, ᶜp, dt, (pap,),
    )
    return nothing
end
function set_microphysics_tendency_cache!(
    Y, p, ::NonEquilibriumMicrophysics2MP3, _,
)
    (; dt) = p
    (; ᶜT, ᶜmp_tendency, ᶜScoll, ᶜlogλ) = p.precomputed

    # get thermodynamics and microphysics params
    params_2mp3 = CAP.microphysics_2mp3_params(p.params)
    thp = CAP.thermodynamics_params(p.params)

    # Get specific quantities (warm rain)
    ᶜq_lcl = @. lazy(specific(Y.c.ρq_lcl, Y.c.ρ))
    ᶜq_rai = @. lazy(specific(Y.c.ρq_rai, Y.c.ρ))
    ᶜn_lcl = @. lazy(specific(Y.c.ρn_lcl, Y.c.ρ))
    ᶜn_rai = @. lazy(specific(Y.c.ρn_rai, Y.c.ρ))
    # Get specific quantities (P3 ice)
    ᶜq_icl = @. lazy(specific(Y.c.ρq_icl, Y.c.ρ))
    ᶜn_ice = @. lazy(specific(Y.c.ρn_ice, Y.c.ρ))
    ᶜq_rim = @. lazy(specific(Y.c.ρq_rim, Y.c.ρ))
    ᶜb_rim = @. lazy(specific(Y.c.ρb_rim, Y.c.ρ))

    # Compute microphysics tendency
    # TODO - looks like aerosol activation is missing
    @. ᶜmp_tendency = BMT.bulk_microphysics_tendencies(
        BMT.Microphysics2Moment(), params_2mp3, thp, Y.c.ρ, ᶜT,
        ᶜq_lcl, ᶜn_lcl, ᶜq_rai, ᶜn_rai, ᶜq_icl, ᶜn_ice, ᶜq_rim, ᶜb_rim, ᶜlogλ,
    )
    # Apply coupled limiting directly
    ᶜf_liq = @. lazy(
        coupled_sink_limit_factor(
            ᶜmp_tendency.dq_lcl_dt, ᶜmp_tendency.dn_lcl_dt, ᶜq_lcl, ᶜn_lcl, dt,
        ),
    )
    ᶜf_rai = @. lazy(
        coupled_sink_limit_factor(
            ᶜmp_tendency.dq_rai_dt, ᶜmp_tendency.dn_rai_dt, ᶜq_rai, ᶜn_rai, dt,
        ),
    )
    @. ᶜmp_tendency.dq_lcl_dt *= ᶜf_liq
    @. ᶜmp_tendency.dn_lcl_dt *= ᶜf_liq
    @. ᶜmp_tendency.dq_rai_dt *= ᶜf_rai
    @. ᶜmp_tendency.dn_rai_dt *= ᶜf_rai
    # TODO - unify the P3 logic with mp_tendency
    @. ᶜScoll.dq_rim_dt = ᶜmp_tendency.dq_rim_dt
    @. ᶜScoll.db_rim_dt = ᶜmp_tendency.db_rim_dt
    # TODO - snow not used in P3 (ice encompasses all frozen hydrometeors)
    # Fix the structure of the named tuple
    @. ᶜmp_tendency.dq_sno_dt = 0
    return nothing
end

"""
    set_precipitation_surface_fluxes!(Y, p, microphysics_model)

Compute the surface fluxes of rain and snow and the column-integrated
precipitation energy tendency. Returns `nothing`.

Mutates `p.precomputed.surface_rain_flux` and `p.precomputed.surface_snow_flux`
[kg/m²/s], and `p.precomputed.col_integrated_precip_energy_tendency` [W/m²]. The
sign convention is upward-positive, so all three are negative when precipitation
reaches the surface.

Dispatch on `microphysics_model`:

  - `EquilibriumMicrophysics0M`: the fluxes are column integrals of the
    density-weighted `q_tot` sink `ᶜρ_dq_tot_dt`, partitioned into rain and snow by
    the freezing temperature.
  - `NonEquilibriumMicrophysics1M` and `NonEquilibriumMicrophysics2M`: the fluxes
    are evaluated at the bottom cell face from the level-1 specific humidities and
    terminal velocities (`ᶜwᵣ`, `ᶜwₛ`, `ᶜwₗ`, `ᶜwᵢ`), with a Jacobian-weighted
    extrapolation of density to the surface. Uses `p.scratch` scalars.
  - `NonEquilibriumMicrophysics2MP3`: delegates to the 2M method; `ρn_ice`,
    `ρq_rim`, and `ρb_rim` are not yet accounted for.
  - Default: no-op.

Must be called after `set_microphysics_tendency_cache!` (0M needs the column
`q_tot` sink) and after `set_precipitation_velocities!` (1M/2M need the terminal
velocities).
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
function set_precipitation_surface_fluxes!(
    Y,
    p,
    microphysics_model::Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M},
)
    (; surface_rain_flux, surface_snow_flux) = p.precomputed
    (; col_integrated_precip_energy_tendency) = p.precomputed
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
    ᶜq_lcl = p.scratch.ᶜtemp_scalar_3
    ᶜq_icl = p.scratch.ᶜtemp_scalar_4
    @. ᶜq_rai = specific(Y.c.ρq_rai, Y.c.ρ)
    @. ᶜq_sno = specific(Y.c.ρq_sno, Y.c.ρ)
    @. ᶜq_lcl = specific(Y.c.ρq_lcl, Y.c.ρ)
    @. ᶜq_icl = specific(Y.c.ρq_icl, Y.c.ρ)
    sfc_qᵣ =
        Fields.Field(Fields.field_values(Fields.level(ᶜq_rai, 1)), sfc_space)
    sfc_qₛ =
        Fields.Field(Fields.field_values(Fields.level(ᶜq_sno, 1)), sfc_space)
    sfc_qₗ =
        Fields.Field(Fields.field_values(Fields.level(ᶜq_lcl, 1)), sfc_space)
    sfc_qᵢ =
        Fields.Field(Fields.field_values(Fields.level(ᶜq_icl, 1)), sfc_space)
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

function set_precipitation_surface_fluxes!(
    Y,
    p,
    ::NonEquilibriumMicrophysics2MP3,
)
    set_precipitation_surface_fluxes!(Y, p, NonEquilibriumMicrophysics2M())
    # TODO: Figure out what to do for ρn_ice, ρq_rim, ρb_rim
end
