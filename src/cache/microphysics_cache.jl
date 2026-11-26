#####
##### Precomputed quantities for precipitation processes
#####

import CloudMicrophysics.MicrophysicsNonEq as CMNe
import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics.Microphysics2M as CM2
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
import CloudMicrophysics.P3Scheme as CMP3

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
@inline function Kin(ᶜw_precip, ᶜu_air)
    return @inline @. lazy(
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
    incloud_specific_humidity(q, cf, cf_floor)

In-cloud specific humidity `max(q, 0) / clamp(cf, cf_floor, 1)` [kg/kg] of a
condensate species with mean specific humidity `q` that is confined to the
cloud fraction `cf`. The floor `cf_floor ∈ (0, 1]` bounds the concentration
factor by `1 / cf_floor`; `cf_floor = 1` returns the mean `max(q, 0)`.
"""
@inline incloud_specific_humidity(q, cf, cf_floor) =
    max(zero(q), q) / clamp(cf, cf_floor, one(cf_floor))

"""
    is_cloud_condensate(χ_name)

Whether the tracer `χ_name` is cloud liquid or cloud ice, the species that the
cloud fraction confines.
"""
is_cloud_condensate(::MatrixFields.FieldName) = false
is_cloud_condensate(
    ::Union{MatrixFields.FieldName{(:q_lcl,)}, MatrixFields.FieldName{(:q_icl,)}},
) = true

"""
    sedimentation_reads_cloud_fraction(atmos, params)

Whether `set_precipitation_velocities!` reads the cloud fraction: the 1M scheme
with `sgs_cloud_sedimentation_cf_floor < 1`, which evaluates the cloud liquid
and cloud ice velocities at the in-cloud specific humidity.
`set_explicit_precomputed_quantities!` then sets the velocities after it has
updated the cloud fraction.
"""
sedimentation_reads_cloud_fraction(atmos, params) =
    atmos.microphysics_model isa NonEquilibriumMicrophysics1M &&
    CAP.sgs_cloud_sedimentation_cf_floor(params) < 1

"""
    set_precipitation_velocities!(Y, p, microphysics_model, turbconv_model)

Update the grid-mean precipitation terminal velocities and cloud sedimentation
velocities, and the precipitation energy flux they carry.

Dispatches on `microphysics_model` (`NonEquilibriumMicrophysics1M`,
`NonEquilibriumMicrophysics2M`, `NonEquilibriumMicrophysics2MP3`) and, for the
1M and 2M schemes, on `turbconv_model::PrognosticEDMFX`; the default method
zeroes `ᶜwₕhₜ`. Returns `nothing`.

Mutates in `p.precomputed`: the velocity fields available for the scheme (`ᶜwₗ`,
`ᶜwᵢ`, `ᶜwᵣ`, `ᶜwₛ`, plus number-weighted `ᶜwₙₗ`/`ᶜwₙᵣ` for 2M and `ᶜwnᵢ`
for P3 ice) and the precipitation energy flux `ᶜwₕhₜ`.

For `PrognosticEDMFX`, the per-subdomain velocities (`ᶜwₗʲs`, `ᶜwᵢʲs`, `ᶜwᵣʲs`,
`ᶜwₛʲs`, ...) are also computed, and the grid-mean velocities are mass-weighted
averages over the subdomains, so that the grid-scale flux equals the sum of the
sub-grid-scale fluxes.

With `sgs_cloud_sedimentation_cf_floor < 1`, the 1M cloud liquid and cloud ice
terminal velocities are evaluated at the in-cloud specific humidity
(`incloud_specific_humidity`): of the grid mean with the cloud fraction
`ᶜcloud_fraction`, or, for `PrognosticEDMFX`, of the environment with its cloud
fraction `ᶜcloud_fraction⁰`. Both are those of the last
`set_covariance_cache_and_cloud_fraction!`. The velocity multiplies the mean
condensate, so the sedimentation flux is the in-cloud flux weighted by the
cloud fraction.

For the 1M scheme under SGS quadrature, the grid-mean and environment rain and
snow velocities are the mass-weighted means over the precipitation shaft of the
microphysics and the rest of the cell (`shaft_weighted_terminal_velocity`,
`precip_shaft_options`), with the precipitation fraction `ᶜprecip_frac` of the
last `set_covariance_cache_and_cloud_fraction!`; the updraft velocities use the
updraft means.
"""
function set_precipitation_velocities!(Y, p, _, _)
    (; ᶜwₕhₜ) = p.precomputed
    @. ᶜwₕhₜ = WVec(0)
    return nothing
end
function set_precipitation_velocities!(
    Y,
    p,
    microphysics_model::NonEquilibriumMicrophysics1M,
    _,
)
    (; ᶜwₗ, ᶜwᵢ, ᶜwᵣ, ᶜwₛ, ᶜwₕhₜ, ᶜT, ᶜu, ᶜcloud_fraction) = p.precomputed
    (; ᶜΦ) = p.core
    ᶜρ = Y.c.ρ
    atmos = p.atmos
    params = p.params
    cf_floor = CAP.sgs_cloud_sedimentation_cf_floor(params)
    use_incloud = sedimentation_reads_cloud_fraction(atmos, params)
    cmc = CAP.microphysics_cloud_params(params)
    cmp = CAP.microphysics_1m_params(params)
    thp = CAP.thermodynamics_params(params)
    eps_FT = eps(eltype(thp))
    # Carried into the terminal-velocity kernels in the type, so the Chen 2022
    # coefficients fold to literals instead of being read from a struct per point.
    # Only the velocity members: `aml` and `activation` are not needed here and
    # a type parameter has to be isbits.
    vcmc, vcmp = let dev = ClimaComms.device(Y.c)
        tv = (; cmc.liquid, cmc.ice, cmc.stokes, cmc.Ch2022)
        _fold(dev, tv), _fold(dev, cmp)
    end

    # scratch for adding energy fluxes over subdomains
    ᶜρwₕhₜ = p.scratch.ᶜtemp_scalar
    @. ᶜρwₕhₜ = 0

    terminal_velocity_function(name, ρ, q) = terminal_velocity(
        microphysics_model,
        velocity_mode(atmos, name),
        name,
        params,
        vcmc,
        vcmp,
        ρ,
        q,
    )
    (; β_p, a_floor, shaft_placement) = precip_shaft_options(atmos, params)
    shaft_terminal_velocity_function(name, ρ, q, a_p) =
        shaft_weighted_terminal_velocity(
            terminal_velocity_function, name, ρ, q, β_p, a_p, a_floor,
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
        if use_incloud && is_cloud_condensate(χ_name)
            @. ᶜw = terminal_velocity_function(
                χ_name,
                ᶜρ,
                incloud_specific_humidity(ᶜρχ / ᶜρ, ᶜcloud_fraction, cf_floor),
            )
        elseif shaft_placement && on_precip_shaft(χ_name)
            (; ᶜprecip_frac) = p.precomputed
            @. ᶜw = shaft_terminal_velocity_function(
                χ_name, ᶜρ, ᶜρχ / ᶜρ, ᶜprecip_frac,
            )
        else
            @. ᶜw = terminal_velocity_function(χ_name, ᶜρ, max(zero(ᶜρ), ᶜρχ / ᶜρ))
        end

        # The energy flux carries only the non-negative condensate mass.
        @. ᶜρwₕhₜ +=
            ᶜw *
            max(zero(ᶜρ), ᶜρχ) *
            (e_int_func(thp, ᶜT) + ᶜΦ + $(Kin(ᶜw, ᶜu)))
    end

    # compute their contribution to the precipitation energy flux
    @. ᶜwₕhₜ = WVec(ᶜρwₕhₜ) / ᶜρ
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
    (; ᶜwₗ, ᶜwᵢ, ᶜwᵣ, ᶜwₛ, ᶜwₕhₜ) = p.precomputed
    (; ᶜwₗʲs, ᶜwᵢʲs, ᶜwᵣʲs, ᶜwₛʲs, ᶜTʲs, ᶜρʲs, ᶜuʲs) = p.precomputed
    (; ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜu⁰, ᶜcloud_fraction⁰) = p.precomputed
    # `velocity_mode` reads the terminal velocity modes from the water model, so
    # the loop does not capture the whole `p.atmos`, which is not a bits type.
    water = p.atmos.water
    params = p.params
    cf_floor = CAP.sgs_cloud_sedimentation_cf_floor(params)
    use_incloud = sedimentation_reads_cloud_fraction(p.atmos, params)

    cmc = CAP.microphysics_cloud_params(params)
    cmp = CAP.microphysics_1m_params(params)
    thp = CAP.thermodynamics_params(params)
    eps_FT = eps(eltype(thp))
    # Carried into the terminal-velocity kernels in the type, so the Chen 2022
    # coefficients fold to literals instead of being read from a struct per point.
    # Only the velocity members: `aml` and `activation` are not needed here and
    # a type parameter has to be isbits.
    vcmc, vcmp = let dev = ClimaComms.device(Y.c)
        tv = (; cmc.liquid, cmc.ice, cmc.stokes, cmc.Ch2022)
        _fold(dev, tv), _fold(dev, cmp)
    end
    draft_indices = StaticOneTo(n_mass_flux_subdomains(turbconv_model))

    terminal_velocity_function(name, ρ, q) = terminal_velocity(
        microphysics_model,
        velocity_mode(water, name),
        name,
        params,
        vcmc,
        vcmp,
        ρ,
        q,
    )
    (; β_p, a_floor, shaft_placement) =
        precip_shaft_options(p.atmos, params)
    shaft_terminal_velocity_function(name, ρ, q, a_p) =
        shaft_weighted_terminal_velocity(
            terminal_velocity_function, name, ρ, q, β_p, a_p, a_floor,
        )
    # The column loop needs a field in the slot of `ᶜprecip_frac` even when
    # the shaft placement is off; `ᶜp` has the same type and is unread then.
    ᶜa_p = shaft_placement ? p.precomputed.ᶜprecip_frac : ᶜp

    # Fuse the column loop across all 4 species and the final WVec outputs so
    # `ᶜρ⁰`, `ᶜρa⁰`, and `ᶜρwₕhₜ` stay in registers across species.
    # TODO: use Y.c.ρq instead of ᶜρχ for computing gs velocities by averaging
    # velocities over subdomains once negative subdomain mass issues are
    # resolved
    species_names = (
        (@name(q_lcl), @name(ρq_lcl)),
        (@name(q_icl), @name(ρq_icl)),
        (@name(q_rai), @name(ρq_rai)),
        (@name(q_sno), @name(ρq_sno)),
    )
    DataLayouts.foreach_column(
        ((ᶜwₗ, ᶜwₗʲs), (ᶜwᵢ, ᶜwᵢʲs), (ᶜwᵣ, ᶜwᵣʲs), (ᶜwₛ, ᶜwₛʲs)), ᶜwₕhₜ,
        Y.c, ᶜΦ, ᶜTʲs, ᶜρʲs, ᶜuʲs, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜu⁰,
        ᶜcloud_fraction⁰, ᶜa_p,
    ) do ᶜw_pairs, ᶜwₕhₜ,
    Yc, ᶜΦ, ᶜTʲs, ᶜρʲs, ᶜuʲs, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜu⁰,
    ᶜcloud_fraction⁰, ᶜa_p
        ᶜρ⁰ = @. max(
            eps_FT,
            TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰),
        )
        ᶜρa⁰ = @. ρa⁰(Yc.ρ, Yc.sgsʲs, turbconv_model)
        ᶜρa⁰_pos = @. max(zero(Yc.ρ), ᶜρa⁰)
        ᶜρwₕhₜ = @. zero(Yc.ρ)
        # Velocities in UVW components, converted once for all species
        ᶜuvw⁰ = @. Geometry.UVWVector(ᶜu⁰)
        ᶜuvwʲs = unrolled_map(
            @inline(j -> (@. Geometry.UVWVector(ᶜuʲs.:($$j)))),
            draft_indices,
        )

        MatrixFields.unrolled_foreach(
            map(tuple, species_names, ᶜw_pairs),
        ) do ((χ_name, ρχ_name), (ᶜw, ᶜwʲs))
            @inline
            e_int_func = internal_energy_func(χ_name)
            ᶜρχ_gs = MatrixFields.get_field(Yc, ρχ_name)
            ᶜχ⁰ = @. specific_env_value(
                χ_name, ᶜρχ_gs, Yc.ρ, Yc.sgsʲs, ᶜρa⁰, turbconv_model,
            )

            ᶜρa⁰χ⁰ = @. ᶜρa⁰_pos * max(zero(Yc.ρ), ᶜχ⁰)
            ᶜρχ = @. ᶜρa⁰χ⁰
            if use_incloud && is_cloud_condensate(χ_name)
                @. ᶜw =
                    ᶜρa⁰χ⁰ * terminal_velocity_function(
                        χ_name,
                        ᶜρ⁰,
                        incloud_specific_humidity(ᶜχ⁰, ᶜcloud_fraction⁰, cf_floor),
                    )
            elseif shaft_placement && on_precip_shaft(χ_name)
                @. ᶜw =
                    ᶜρa⁰χ⁰ *
                    shaft_terminal_velocity_function(χ_name, ᶜρ⁰, ᶜχ⁰, ᶜa_p)
            else
                @. ᶜw =
                    ᶜρa⁰χ⁰ *
                    terminal_velocity_function(χ_name, ᶜρ⁰, max(zero(Yc.ρ), ᶜχ⁰))
            end
            ᶜimplied_env_mass_flux = @. zero(Yc.ρ)

            for j in draft_indices
                ᶜsgsʲ = Yc.sgsʲs.:($j)
                ᶜχʲ = MatrixFields.get_field(ᶜsgsʲ, χ_name)
                ᶜwʲ = ᶜwʲs.:($j)
                ᶜρʲ = ᶜρʲs.:($j)
                ᶜTʲ = ᶜTʲs.:($j)
                ᶜuʲ = ᶜuvwʲs[j]

                ᶜρaʲχʲ = @. max(zero(Yc.ρ), ᶜsgsʲ.ρa) * max(zero(Yc.ρ), ᶜχʲ)
                @. ᶜρχ += ᶜρaʲχʲ
                @. ᶜwʲ = terminal_velocity_function(χ_name, ᶜρʲ, max(zero(Yc.ρ), ᶜχʲ))
                @. ᶜw += ᶜρaʲχʲ * ᶜwʲ
                @. ᶜimplied_env_mass_flux -= ᶜρaʲχʲ * ᶜwʲ
                @. ᶜρwₕhₜ +=
                    ᶜρaʲχʲ * ᶜwʲ *
                    (e_int_func(thp, ᶜTʲ) + ᶜΦ + $(Kin(ᶜwʲ, ᶜuʲ)))
            end

            @. ᶜw = gs_terminal_velocity(
                microphysics_model,
                velocity_mode(water, χ_name),
                χ_name,
                params,
                ᶜw,
                ᶜρχ,
            )
            # Clamping `ᶜimplied_env_mass_flux ≥ 0` departs from
            # `env + Σ updrafts = total` where the updraft sedimentation flux
            # exceeds the grid-mean flux, keeping the environment sedimentation
            # flux downward.
            @. ᶜimplied_env_mass_flux =
                max(zero(Yc.ρ), ᶜimplied_env_mass_flux + max(zero(Yc.ρ), ᶜρχ_gs) * ᶜw)
            @. ᶜρwₕhₜ +=
                ᶜimplied_env_mass_flux *
                (e_int_func(thp, ᶜT⁰) + ᶜΦ + $(Kin(ᶜw, ᶜuvw⁰)))
            return nothing
        end

        @. ᶜwₕhₜ = WVec(ᶜρwₕhₜ) / Yc.ρ
    end
    return nothing
end
function set_precipitation_velocities!(
    Y,
    p,
    microphysics_model::NonEquilibriumMicrophysics2M,
    _,
)
    (; ᶜwₗ, ᶜwᵢ, ᶜwᵣ, ᶜwₛ, ᶜwₙₗ, ᶜwₙᵣ, ᶜwₕhₜ, ᶜT, ᶜu) = p.precomputed
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

    # The energy flux carries only the non-negative condensate mass.
    @. ᶜwₕhₜ =
        WVec(
            ᶜwₗ * max(zero(Y.c.ρ), Y.c.ρq_lcl) * (Iₗ(thp, ᶜT) + ᶜΦ + $(Kin(ᶜwₗ, ᶜu))) +
            ᶜwᵢ * max(zero(Y.c.ρ), Y.c.ρq_icl) * (Iᵢ(thp, ᶜT) + ᶜΦ + $(Kin(ᶜwᵢ, ᶜu))) +
            ᶜwᵣ * max(zero(Y.c.ρ), Y.c.ρq_rai) * (Iₗ(thp, ᶜT) + ᶜΦ + $(Kin(ᶜwᵣ, ᶜu))) +
            ᶜwₛ * max(zero(Y.c.ρ), Y.c.ρq_sno) * (Iᵢ(thp, ᶜT) + ᶜΦ + $(Kin(ᶜwₛ, ᶜu))),
        ) / Y.c.ρ
    return nothing
end
function set_precipitation_velocities!(
    Y,
    p,
    microphysics_model::NonEquilibriumMicrophysics2M,
    turbconv_model::PrognosticEDMFX,
)
    (; ᶜwₗ, ᶜwᵢ, ᶜwᵣ, ᶜwₛ, ᶜwₙₗ, ᶜwₙᵣ, ᶜwₕhₜ) = p.precomputed
    (; ᶜΦ) = p.core
    (; ᶜwₗʲs, ᶜwᵢʲs, ᶜwᵣʲs, ᶜwₛʲs, ᶜwₙₗʲs, ᶜwₙᵣʲs) = p.precomputed
    (; ᶜp, ᶜTʲs, ᶜρʲs, ᶜuʲs, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜu⁰) =
        p.precomputed
    cmc = CAP.microphysics_cloud_params(p.params)
    cm1p = CAP.microphysics_1m_params(p.params)
    cm2p = CAP.microphysics_2m_params(p.params)
    thp = CAP.thermodynamics_params(p.params)
    FT = eltype(p.params)
    zero_FT = FT(0)

    eps_FT = eps(FT)
    ᶜρ⁰ = @. lazy(
        max(
            eps_FT,
            TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰),
        ),
    )
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))

    # scratch for adding energy fluxes over subdomains
    ᶜρwₕhₜ = p.scratch.ᶜtemp_scalar_3

    # Compute grid-scale sedimentation velocity based on subdomain velocities
    # assuming grid-scale flux is equal to the  sum of sub-grid-scale fluxes,
    # with the positive masses of the subdomains summed in `ᶜρχ`.
    # TODO use Y.c.ρq instead of ᶜρχ for computing gs velocities by averaging velocities
    # over subdomains once negative subdomain mass issues are resolved

    ᶜq_lcl⁰ = ᶜspecific_env_value(@name(q_lcl), Y, p)
    ᶜn_lcl⁰ = ᶜspecific_env_value(@name(n_lcl), Y, p)
    ###
    ### Cloud liquid (number)
    ###
    DataLayouts.foreach_column(
        Y.c,
        ᶜq_lcl⁰,
        ᶜρ⁰,
        ᶜn_lcl⁰,
        ᶜwₙₗ,
        ᶜρa⁰,
        ᶜwₙₗʲs,
        ᶜρʲs,
    ) do Yc, ᶜq_lcl⁰, ᶜρ⁰, ᶜn_lcl⁰, ᶜwₙₗ, ᶜρa⁰, ᶜwₙₗʲs, ᶜρʲs
        ᶜw⁰ = @. getindex(
            CM2.cloud_terminal_velocity(
                cm2p.warm_rain.seifert_beheng.pdf_c,
                cmc.stokes,
                max(zero(Yc.ρ), ᶜq_lcl⁰),
                ᶜρ⁰,
                max(zero(Yc.ρ), ᶜρ⁰ * ᶜn_lcl⁰),
            ),
            1,
        )
        @. ᶜwₙₗ = ᶜρa⁰ * ᶜn_lcl⁰ * ᶜw⁰
        ᶜρχ = @. max(zero(Yc.ρ), ᶜρa⁰ * ᶜn_lcl⁰)
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            @. ᶜwₙₗʲs.:($$j) = getindex(
                CM2.cloud_terminal_velocity(
                    cm2p.warm_rain.seifert_beheng.pdf_c,
                    cmc.stokes,
                    max(zero(Yc.ρ), Yc.sgsʲs.:($$j).q_lcl),
                    ᶜρʲs.:($$j),
                    max(zero(Yc.ρ), ᶜρʲs.:($$j) * Yc.sgsʲs.:($$j).n_lcl),
                ),
                1,
            )
            @. ᶜwₙₗ += Yc.sgsʲs.:($$j).ρa * Yc.sgsʲs.:($$j).n_lcl * ᶜwₙₗʲs.:($$j)
            @. ᶜρχ += max(zero(Yc.ρ), Yc.sgsʲs.:($$j).ρa * Yc.sgsʲs.:($$j).n_lcl)
        end
        @. ᶜwₙₗ = ifelse(ᶜρχ > zero_FT, ᶜwₙₗ / ᶜρχ, zero_FT)
    end

    ###
    ### Cloud liquid (mass)
    ###
    DataLayouts.foreach_column(
        Y.c,
        ᶜq_lcl⁰,
        ᶜρ⁰,
        ᶜn_lcl⁰,
        ᶜwₗ,
        ᶜρa⁰,
        ᶜwₗʲs,
        ᶜρʲs,
        ᶜρwₕhₜ,
        ᶜT⁰,
        ᶜΦ,
        ᶜu⁰,
    ) do Yc, ᶜq_lcl⁰, ᶜρ⁰, ᶜn_lcl⁰, ᶜwₗ, ᶜρa⁰, ᶜwₗʲs, ᶜρʲs, ᶜρwₕhₜ, ᶜT⁰, ᶜΦ, ᶜu⁰
        ᶜw⁰ = @. getindex(
            CM2.cloud_terminal_velocity(
                cm2p.warm_rain.seifert_beheng.pdf_c,
                cmc.stokes,
                max(zero(Yc.ρ), ᶜq_lcl⁰),
                ᶜρ⁰,
                max(zero(Yc.ρ), ᶜρ⁰ * ᶜn_lcl⁰),
            ),
            2,
        )
        @. ᶜwₗ = ᶜρa⁰ * ᶜq_lcl⁰ * ᶜw⁰
        ᶜρχ = @. max(zero(Yc.ρ), ᶜρa⁰ * ᶜq_lcl⁰)
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            @. ᶜwₗʲs.:($$j) = getindex(
                CM2.cloud_terminal_velocity(
                    cm2p.warm_rain.seifert_beheng.pdf_c,
                    cmc.stokes,
                    max(zero(Yc.ρ), Yc.sgsʲs.:($$j).q_lcl),
                    ᶜρʲs.:($$j),
                    max(zero(Yc.ρ), ᶜρʲs.:($$j) * Yc.sgsʲs.:($$j).n_lcl),
                ),
                2,
            )
            @. ᶜwₗ += Yc.sgsʲs.:($$j).ρa * Yc.sgsʲs.:($$j).q_lcl * ᶜwₗʲs.:($$j)
            @. ᶜρχ += max(zero(Yc.ρ), Yc.sgsʲs.:($$j).ρa * Yc.sgsʲs.:($$j).q_lcl)
        end
        @. ᶜwₗ = ifelse(ᶜρχ > zero_FT, ᶜwₗ / ᶜρχ, zero_FT)
        # contribution of env cloud liquid advection to htot advection
        @. ᶜρwₕhₜ =
            (
                Yc.ρq_lcl * ᶜwₗ -
                Yc.sgsʲs.:(1).ρa * Yc.sgsʲs.:(1).q_lcl * ᶜwₗʲs.:(1)
            ) * (Iₗ(thp, ᶜT⁰) + ᶜΦ + $(Kin(ᶜwₗ, ᶜu⁰)))
    end

    ###
    ### Cloud ice
    ###
    ᶜq_icl⁰ = ᶜspecific_env_value(@name(q_icl), Y, p)
    # TODO sedimentation of ice is based on the 1M scheme
    DataLayouts.foreach_column(
        Y.c,
        ᶜρ⁰,
        ᶜq_icl⁰,
        ᶜwᵢ,
        ᶜρa⁰,
        ᶜwᵢʲs,
        ᶜρʲs,
        ᶜρwₕhₜ,
        ᶜT⁰,
        ᶜΦ,
        ᶜu⁰,
    ) do Yc, ᶜρ⁰, ᶜq_icl⁰, ᶜwᵢ, ᶜρa⁰, ᶜwᵢʲs, ᶜρʲs, ᶜρwₕhₜ, ᶜT⁰, ᶜΦ, ᶜu⁰
        ᶜw⁰ = @. CMNe.terminal_velocity(
            cmc.ice,
            cmc.Ch2022.small_ice,
            ᶜρ⁰,
            max(zero(Yc.ρ), ᶜq_icl⁰),
        )
        @. ᶜwᵢ = ᶜρa⁰ * ᶜq_icl⁰ * ᶜw⁰
        ᶜρχ = @. max(zero(Yc.ρ), ᶜρa⁰ * ᶜq_icl⁰)
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            @. ᶜwᵢʲs.:($$j) = CMNe.terminal_velocity(
                cmc.ice,
                cmc.Ch2022.small_ice,
                ᶜρʲs.:($$j),
                max(zero(Yc.ρ), Yc.sgsʲs.:($$j).q_icl),
            )
            @. ᶜwᵢ += Yc.sgsʲs.:($$j).ρa * Yc.sgsʲs.:($$j).q_icl * ᶜwᵢʲs.:($$j)
            @. ᶜρχ += max(zero(Yc.ρ), Yc.sgsʲs.:($$j).ρa * Yc.sgsʲs.:($$j).q_icl)
        end
        @. ᶜwᵢ = ifelse(ᶜρχ > zero_FT, ᶜwᵢ / ᶜρχ, zero_FT)
        # contribution of env cloud ice advection to htot advection
        @. ᶜρwₕhₜ +=
            (
                Yc.ρq_icl * ᶜwᵢ -
                Yc.sgsʲs.:(1).ρa * Yc.sgsʲs.:(1).q_icl * ᶜwᵢʲs.:(1)
            ) * (Iᵢ(thp, ᶜT⁰) + ᶜΦ + $(Kin(ᶜwᵢ, ᶜu⁰)))
    end

    ###
    ### Rain (number)
    ###
    ᶜq_rai⁰ = ᶜspecific_env_value(@name(q_rai), Y, p)
    ᶜn_rai⁰ = ᶜspecific_env_value(@name(n_rai), Y, p)
    DataLayouts.foreach_column(
        Y.c,
        ᶜq_rai⁰,
        ᶜρ⁰,
        ᶜn_rai⁰,
        ᶜwₙᵣ,
        ᶜρa⁰,
        ᶜwₙᵣʲs,
        ᶜρʲs,
    ) do Yc, ᶜq_rai⁰, ᶜρ⁰, ᶜn_rai⁰, ᶜwₙᵣ, ᶜρa⁰, ᶜwₙᵣʲs, ᶜρʲs
        ᶜw⁰ = @. getindex(
            CM2.rain_terminal_velocity(
                cm2p.warm_rain.seifert_beheng,
                cmc.Ch2022.rain,
                max(zero(Yc.ρ), ᶜq_rai⁰),
                ᶜρ⁰,
                max(zero(Yc.ρ), ᶜρ⁰ * ᶜn_rai⁰),
            ),
            1,
        )
        @. ᶜwₙᵣ = ᶜρa⁰ * ᶜn_rai⁰ * ᶜw⁰
        ᶜρχ = @. max(zero(Yc.ρ), ᶜρa⁰ * ᶜn_rai⁰)
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            @. ᶜwₙᵣʲs.:($$j) = getindex(
                CM2.rain_terminal_velocity(
                    cm2p.warm_rain.seifert_beheng,
                    cmc.Ch2022.rain,
                    max(zero(Yc.ρ), Yc.sgsʲs.:($$j).q_rai),
                    ᶜρʲs.:($$j),
                    max(zero(Yc.ρ), ᶜρʲs.:($$j) * Yc.sgsʲs.:($$j).n_rai),
                ),
                1,
            )
            @. ᶜwₙᵣ += Yc.sgsʲs.:($$j).ρa * Yc.sgsʲs.:($$j).n_rai * ᶜwₙᵣʲs.:($$j)
            @. ᶜρχ += max(zero(Yc.ρ), Yc.sgsʲs.:($$j).ρa * Yc.sgsʲs.:($$j).n_rai)
        end
        @. ᶜwₙᵣ = ifelse(ᶜρχ > zero_FT, ᶜwₙᵣ / ᶜρχ, zero_FT)
    end

    ###
    ### Rain (mass)
    ###
    DataLayouts.foreach_column(
        Y.c,
        ᶜq_rai⁰,
        ᶜρ⁰,
        ᶜn_rai⁰,
        ᶜwᵣ,
        ᶜρa⁰,
        ᶜwᵣʲs,
        ᶜρʲs,
        ᶜρwₕhₜ,
        ᶜT⁰,
        ᶜΦ,
        ᶜu⁰,
    ) do Yc, ᶜq_rai⁰, ᶜρ⁰, ᶜn_rai⁰, ᶜwᵣ, ᶜρa⁰, ᶜwᵣʲs, ᶜρʲs, ᶜρwₕhₜ, ᶜT⁰, ᶜΦ, ᶜu⁰
        ᶜw⁰ = @. getindex(
            CM2.rain_terminal_velocity(
                cm2p.warm_rain.seifert_beheng,
                cmc.Ch2022.rain,
                max(zero(Yc.ρ), ᶜq_rai⁰),
                ᶜρ⁰,
                max(zero(Yc.ρ), ᶜρ⁰ * ᶜn_rai⁰),
            ),
            2,
        )
        @. ᶜwᵣ = ᶜρa⁰ * ᶜq_rai⁰ * ᶜw⁰
        ᶜρχ = @. max(zero(Yc.ρ), ᶜρa⁰ * ᶜq_rai⁰)
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            @. ᶜwᵣʲs.:($$j) = getindex(
                CM2.rain_terminal_velocity(
                    cm2p.warm_rain.seifert_beheng,
                    cmc.Ch2022.rain,
                    max(zero(Yc.ρ), Yc.sgsʲs.:($$j).q_rai),
                    ᶜρʲs.:($$j),
                    max(zero(Yc.ρ), ᶜρʲs.:($$j) * Yc.sgsʲs.:($$j).n_rai),
                ),
                2,
            )
            @. ᶜwᵣ += Yc.sgsʲs.:($$j).ρa * Yc.sgsʲs.:($$j).q_rai * ᶜwᵣʲs.:($$j)
            @. ᶜρχ += max(zero(Yc.ρ), Yc.sgsʲs.:($$j).ρa * Yc.sgsʲs.:($$j).q_rai)
        end
        @. ᶜwᵣ = ifelse(ᶜρχ > zero_FT, ᶜwᵣ / ᶜρχ, zero_FT)
        # contribution of env rain advection to qtot advection
        @. ᶜρwₕhₜ +=
            (
                Yc.ρq_rai * ᶜwᵣ -
                Yc.sgsʲs.:(1).ρa * Yc.sgsʲs.:(1).q_rai * ᶜwᵣʲs.:(1)
            ) * (Iₗ(thp, ᶜT⁰) + ᶜΦ + $(Kin(ᶜwᵣ, ᶜu⁰)))
    end

    ###
    ### Snow
    ####
    ᶜq_sno⁰ = ᶜspecific_env_value(@name(q_sno), Y, p)
    # TODO sedimentation of snow is based on the 1M scheme
    DataLayouts.foreach_column(
        Y.c,
        ᶜρ⁰,
        ᶜq_sno⁰,
        ᶜwₛ,
        ᶜρa⁰,
        ᶜwₛʲs,
        ᶜρʲs,
        ᶜρwₕhₜ,
        ᶜT⁰,
        ᶜΦ,
        ᶜu⁰,
    ) do Yc, ᶜρ⁰, ᶜq_sno⁰, ᶜwₛ, ᶜρa⁰, ᶜwₛʲs, ᶜρʲs, ᶜρwₕhₜ, ᶜT⁰, ᶜΦ, ᶜu⁰
        ᶜw⁰ = @. CM1.terminal_velocity(
            cm1p.precip.snow,
            cm1p.terminal_velocity.snow,
            ᶜρ⁰,
            max(zero(Yc.ρ), ᶜq_sno⁰),
        )
        @. ᶜwₛ = ᶜρa⁰ * ᶜq_sno⁰ * ᶜw⁰
        ᶜρχ = @. max(zero(Yc.ρ), ᶜρa⁰ * ᶜq_sno⁰)
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            @. ᶜwₛʲs.:($$j) = CM1.terminal_velocity(
                cm1p.precip.snow,
                cm1p.terminal_velocity.snow,
                ᶜρʲs.:($$j),
                max(zero(Yc.ρ), Yc.sgsʲs.:($$j).q_sno),
            )
            @. ᶜwₛ += Yc.sgsʲs.:($$j).ρa * Yc.sgsʲs.:($$j).q_sno * ᶜwₛʲs.:($$j)
            @. ᶜρχ += max(zero(Yc.ρ), Yc.sgsʲs.:($$j).ρa * Yc.sgsʲs.:($$j).q_sno)
        end
        @. ᶜwₛ = ifelse(ᶜρχ > zero_FT, ᶜwₛ / ᶜρχ, zero_FT)
        # contribution of env snow advection to htot advection
        @. ᶜρwₕhₜ +=
            (
                Yc.ρq_sno * ᶜwₛ -
                Yc.sgsʲs.:(1).ρa * Yc.sgsʲs.:(1).q_sno * ᶜwₛʲs.:(1)
            ) * (Iᵢ(thp, ᶜT⁰) + ᶜΦ + $(Kin(ᶜwₛ, ᶜu⁰)))
    end

    # Add contributions to energy and total water advection
    DataLayouts.foreach_point(
        Y.c,
        ᶜρwₕhₜ,
        ᶜwₗʲs,
        ᶜTʲs,
        ᶜΦ,
        ᶜuʲs,
        ᶜwᵢʲs,
        ᶜwᵣʲs,
        ᶜwₛʲs,
    ) do Yc, ᶜρwₕhₜ, ᶜwₗʲs, ᶜTʲs, ᶜΦ, ᶜuʲs, ᶜwᵢʲs, ᶜwᵣʲs, ᶜwₛʲs
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            @. ᶜρwₕhₜ +=
                Yc.sgsʲs.:($$j).ρa * (
                    ᶜwₗʲs.:($$j) *
                    Yc.sgsʲs.:($$j).q_lcl *
                    (Iₗ(thp, ᶜTʲs.:($$j)) + ᶜΦ + $(Kin(ᶜwₗʲs.:($j), ᶜuʲs.:($j)))) +
                    ᶜwᵢʲs.:($$j) *
                    Yc.sgsʲs.:($$j).q_icl *
                    (Iᵢ(thp, ᶜTʲs.:($$j)) + ᶜΦ + $(Kin(ᶜwᵢʲs.:($j), ᶜuʲs.:($j)))) +
                    ᶜwᵣʲs.:($$j) *
                    Yc.sgsʲs.:($$j).q_rai *
                    (Iₗ(thp, ᶜTʲs.:($$j)) + ᶜΦ + $(Kin(ᶜwᵣʲs.:($j), ᶜuʲs.:($j)))) +
                    ᶜwₛʲs.:($$j) *
                    Yc.sgsʲs.:($$j).q_sno *
                    (Iᵢ(thp, ᶜTʲs.:($$j)) + ᶜΦ + $(Kin(ᶜwₛʲs.:($j), ᶜuʲs.:($j))))
                )
        end
    end
    @. ᶜwₕhₜ = WVec(ᶜρwₕhₜ) / Y.c.ρ

    return nothing
end
function set_precipitation_velocities!(
    Y, p, ::NonEquilibriumMicrophysics2MP3, _,
)
    ## liquid quantities (2M warm rain)
    (; ᶜwₗ, ᶜwᵣ, ᶜwₙₗ, ᶜwₙᵣ, ᶜwₕhₜ, ᶜT, ᶜu) = p.precomputed
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
    @. ᶜwₙᵣ = getindex(ᶜrai_w_terms, 1)
    @. ᶜwᵣ = getindex(ᶜrai_w_terms, 2)
    # Number- and mass weighted cloud liquid terminal velocity [m/s]
    ᶜliq_w_terms = @. lazy(
        CM2.cloud_terminal_velocity(
            sb.pdf_c, cmc.stokes,
            max(zero(ρ), specific(ρq_lcl, ρ)),
            ρ, max(zero(ρ), ρn_lcl),
        ),
    )
    @. ᶜwₙₗ = getindex(ᶜliq_w_terms, 1)
    @. ᶜwₗ = getindex(ᶜliq_w_terms, 2)

    ## Ice quantities
    (; ρq_icl, ρn_ice, ρq_rim, ρb_rim) = Y.c
    (; ᶜwᵢ) = p.precomputed

    # P3 ice params from unified container
    p3_ice = params_2mp3.ice

    # Number- and mass weighted ice terminal velocity [m/s]
    # Calculate terminal velocities
    (; ᶜwnᵢ) = p.precomputed
    ᶜstate_p3 = @. lazy(
        CMP3.P3State(p3_ice.scheme, p3_ice.integration,
            ρq_icl, ρn_ice, ρq_rim, ρb_rim,
        ),
    )
    ᶜw_ice = @. lazy(
        CMP3.ice_terminal_velocities(p3_ice.terminal_velocity, ᶜstate_p3, ρ),
    )
    @. ᶜwnᵢ = getindex(ᶜw_ice, 1)
    @. ᶜwᵢ = getindex(ᶜw_ice, 2)

    # compute their contribution to the precipitation energy flux
    @. ᶜwₕhₜ =
        WVec(
            ᶜwₗ * ρq_lcl * (Iₗ(thp, ᶜT) + ᶜΦ + $(Kin(ᶜwₗ, ᶜu))) +
            ᶜwᵢ * ρq_icl * (Iᵢ(thp, ᶜT) + ᶜΦ + $(Kin(ᶜwᵢ, ᶜu))) +
            ᶜwᵣ * ρq_rai * (Iₗ(thp, ᶜT) + ᶜΦ + $(Kin(ᶜwᵣ, ᶜu))),
        ) / ρ
    return nothing
end

"""
    env_detrained_condensate_fraction(q_lcl, q_icl, sigma_S, λ_eq)

Fraction `f_neq ∈ [0, 1]` of the environment cloud condensate
`q_c = q_lcl + q_icl` in excess of its equilibrium value,

    f_neq = clamp((q_c − q_c_eq) / q_c, 0, 1),

and `0` where `q_c = 0`. The equilibrium condensate
`q_c_eq = E[max(0, λ_eq + S′)]`, `S′ ~ N(0, σ_S²)`, is that of the SGS moments
`sigma_S` and `λ_eq` (`_compute_sgs_moments`); it vanishes for `λ_eq = −Inf`.

The excess `q_c − q_c_eq` is the vapor deficit of the environment: for a narrow
distribution with a saturated mean, it equals `q_sat − q_vap`. Cloud liquid
removes the deficit within its condensation timescale, so an excess persists
where detrainment from updrafts supplies condensate, and `f_neq` marks
environment condensate of convective origin. Cloud ice, which relaxes more
slowly, enters `f_neq` through the total `q_c`: the excess is independent of
the phase partition, and where it is positive, the vapor is subsaturated with
respect to liquid, so that cloud liquid there evaporates within its
condensation timescale unless detrainment supplies it.
"""
@inline function env_detrained_condensate_fraction(q_lcl, q_icl, sigma_S, λ_eq)
    q_c = max(zero(q_lcl), q_lcl) + max(zero(q_icl), q_icl)
    # `ifelse` discards the NaN of `truncated_gaussian_condensate` at
    # `λ_eq = −Inf`.
    q_c_eq = ifelse(
        isfinite(λ_eq),
        truncated_gaussian_condensate(λ_eq, sigma_S),
        zero(λ_eq),
    )
    # The floor on the denominator gives `f_neq = 0` at `q_c = 0` without a
    # comparison of `q_c` to zero, which `ForwardDiff` decides on the partials
    # where the values tie.
    q_c_floor = floatmin(typeof(q_c))
    return clamp((q_c - q_c_eq) / max(q_c, q_c_floor), zero(q_c), one(q_c))
end

"""
    draft_mean_updraft_velocity(sgsʲs, ρʲs, uʲs, lg, a_min)

Area-weighted mean upward velocity of the drafts [m/s],

    w_up = Σⱼ aʲ max(wʲ, 0) / max(Σⱼ aʲ, a_min),

with the draft areas `aʲ = max(0, draft_area(ρaʲ, ρʲ))` and the vertical
velocities `wʲ` of the draft velocities `uʲs` at the local geometry `lg`. The
floor `a_min` on the total draft area takes `w_up` to zero as the drafts
vanish.
"""
@inline function draft_mean_updraft_velocity(sgsʲs, ρʲs, uʲs, lg, a_min)
    aʲs = unrolled_map(sgsʲs, ρʲs) do sgsʲ, ρʲ
        max(zero(ρʲ), draft_area(sgsʲ.ρa, ρʲ))
    end
    w⁺ʲs = unrolled_map(uʲs) do uʲ
        wʲ = get_physical_w(uʲ, lg)
        max(zero(wʲ), wʲ)
    end
    return unrolled_dotproduct(aʲs, w⁺ʲs) / max(unrolled_sum(aʲs), a_min)
end

"""
    env_autoconversion_velocity(w⁰, w_up, f_neq, c_w, w_d)

Vertical velocity [m/s] at which the `Kessler1M` rain autoconversion of the
`PrognosticEDMFX` environment is evaluated,

    w_mp⁰ = max(w⁰, f_neq (c_w w_up + w_d)).

`Kessler1M` blends its threshold and timescale between quiescent (stratiform)
and convective values with `max(w, 0)`. The environment descends where
updrafts rise, so at its own velocity `w⁰`, condensate that updrafts detrained
into it converts at the quiescent values. That condensate, the fraction `f_neq`
of the environment condensate (`env_detrained_condensate_fraction`), takes the
velocity of the drafts that formed it: the mean updraft velocity `w_up` of the
cell (`draft_mean_updraft_velocity`) with the weight `c_w`
(`sgs_env_autoconversion_updraft_velocity_weight`), plus the velocity `w_d`
(`sgs_env_autoconversion_detrained_velocity`) for condensate that has outlived
its drafts. The velocity scales with `f_neq`: a stratocumulus deck in
equilibrium (`f_neq ≈ 0`) converts at the quiescent values, and condensate
detrained into a subsaturated cumulus layer (`f_neq ≈ 1`) at the convective
ones.

With the velocity scale `w_0` of `Kessler1M`
(`rain_autoconversion_velocity_scale`), the blending factor in a descending
environment is `x² / (1 + x²)`, `x = f_neq (c_w w_up + w_d) / w_0`: quadratic
in `f_neq` for `x ≪ 1`, and one half at `f_neq = w_0 / (c_w w_up + w_d)`. The
parameters `c_w` and `w_d` hence act relative to `w_0`; values that take
`c_w w_up + w_d` far above `w_0` turn the blending into a switch at small
`f_neq`.
"""
@inline env_autoconversion_velocity(w⁰, w_up, f_neq, c_w, w_d) =
    max(w⁰, f_neq * (c_w * w_up + w_d))

"""
    uses_env_autoconversion_velocity(cmp, params)

Whether the environment rain autoconversion of the 1M parameters `cmp` is
evaluated at `env_autoconversion_velocity`: the autoconversion variant is
`Kessler1M`, the one that reads the vertical velocity, and one of
`sgs_env_autoconversion_updraft_velocity_weight` and
`sgs_env_autoconversion_detrained_velocity` is positive.
"""
uses_env_autoconversion_velocity(cmp, params) =
    cmp.processes.rain_autoconversion isa CMP.Kessler1M && (
        CAP.sgs_env_autoconversion_updraft_velocity_weight(params) > 0 ||
        CAP.sgs_env_autoconversion_detrained_velocity(params) > 0
    )

"""
    ᶜenv_autoconversion_velocity(Y, p, ᶜq_lcl⁰, ᶜq_icl⁰)

Lazy field of `env_autoconversion_velocity` for the environment cloud liquid
`ᶜq_lcl⁰` and cloud ice `ᶜq_icl⁰`, with the SGS moments `ᶜsgs_moments` and the
draft states of `Y` and `p.precomputed`.
"""
function ᶜenv_autoconversion_velocity(Y, p, ᶜq_lcl⁰, ᶜq_icl⁰)
    (; ᶜu⁰, ᶜρʲs, ᶜuʲs, ᶜsgs_moments) = p.precomputed
    FT = eltype(p.params)
    c_w = FT(CAP.sgs_env_autoconversion_updraft_velocity_weight(p.params))
    w_d = FT(CAP.sgs_env_autoconversion_detrained_velocity(p.params))
    a_min = FT(CAP.min_area(CAP.turbconv_params(p.params)))
    ᶜsgsʲs = Y.c.sgsʲs
    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶜσ_S = ᶜsgs_moments.sigma_S
    ᶜλ_eq = ᶜsgs_moments.λ_eq
    return @. lazy(
        env_autoconversion_velocity(
            w_component(WVec(ᶜu⁰)),
            draft_mean_updraft_velocity(ᶜsgsʲs, ᶜρʲs, ᶜuʲs, ᶜlg, a_min),
            env_detrained_condensate_fraction(ᶜq_lcl⁰, ᶜq_icl⁰, ᶜσ_S, ᶜλ_eq),
            c_w,
            w_d,
        ),
    )
end
