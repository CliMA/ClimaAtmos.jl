#####
##### Tendencies applied to the grid-mean atmospheric state due to subgrid-scale (SGS)
##### fluxes computed by the EDMFX scheme
#####

"""
    edmfx_sgs_mass_flux_tendency!(Yₜ, Y, p, t, turbconv_model)

Computes and applies tendencies to the grid-mean prognostic variables due to the
divergence of subgrid-scale (SGS) mass fluxes from EDMFX updrafts and the environment.

This involves terms of the form `- ∂(ρₖ aₖ w′ₖ ϕ′ₖ)/∂z`, where `k` denotes
an SGS component (updraft `j` or environment `0`), `aₖ` is the area fraction,
`w′ₖ` is the vertical velocity deviation from the grid mean, and `ϕ′ₖ` is the
deviation of a conserved variable `ϕ` (such as total enthalpy or specific humidity)
from its grid-mean value. These terms represent the redistribution of energy and tracers
by the resolved SGS circulations relative to the grid mean flow.

The specific implementation depends on the `turbconv_model` (e.g., `PrognosticEDMFX`
or `DiagnosticEDMFX`). A generic fallback doing nothing is also provided.
The function modifies `Yₜ.c` (grid-mean tendencies) in place.

Arguments:
- `Yₜ`: The tendency state vector for grid-mean variables.
- `Y`: The current state vector (used for grid-mean and SGS properties).
- `p`: Cache containing parameters, precomputed fields, atmospheric model settings,
       and scratch space.
- `t`: Current simulation time.
- `turbconv_model`: The turbulence convection model instance.
"""
edmfx_sgs_mass_flux_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing

function edmfx_sgs_mass_flux_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
)

    n = n_mass_flux_subdomains(turbconv_model)
    (; edmfx_sgsflux_upwinding) = p.atmos.numerics
    (; ᶠu³) = p.precomputed
    (; ᶜΦ,) = p.core
    (; ᶠu³ʲs, ᶜKʲs, ᶜρʲs) = p.precomputed
    (; ᶠu³⁰, ᶜK⁰, ᶜts⁰, ᶜts, ᶜu⁰) = p.precomputed
    (; ᶜwₜqₜ, ᶜwₕhₜ, ᶜwₜʲs, ᶜwₕʲs) = p.precomputed # TODO - cleanup the names here

    thermo_params = CAP.thermodynamics_params(p.params)
    cmp = CAP.microphysics_1m_params(p.params)
    cmc = CAP.microphysics_cloud_params(p.params)
    ᶜρ⁰ = @. lazy(TD.air_density(thermo_params, ᶜts⁰))
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
    (; dt) = p
    ᶜJ = Fields.local_geometry_field(Y.c).J

    if p.atmos.edmfx_model.sgs_mass_flux isa Val{true}
        # Enthalpy fluxes. First sum up the draft fluxes
        # TODO: Isolate assembly of flux term pattern to a function and
        # reuse (both in prognostic and diagnostic EDMFX)
        # [best after removal of precomputed quantities]
        ᶠu³_diff = p.scratch.ᶠtemp_CT3
        ᶜa_scalar = p.scratch.ᶜtemp_scalar
        ᶜh_tot = @. lazy(
            TD.total_specific_enthalpy(
                thermo_params,
                ᶜts,
                specific(Y.c.ρe_tot, Y.c.ρ),
            ),
        )
        if p.atmos.moisture_model isa NonEquilMoistModel &&
            p.atmos.microphysics_model isa Microphysics1Moment

            ᶜq_liq⁰ = ᶜspecific_env_value(Val(:q_liq), Y, p)
            ᶜq_ice⁰ = ᶜspecific_env_value(Val(:q_ice), Y, p)
            ᶜq_rai⁰ = ᶜspecific_env_value(Val(:q_rai), Y, p)
            ᶜq_sno⁰ = ᶜspecific_env_value(Val(:q_sno), Y, p)

            ᶜwₗ⁰ = @. lazy(CMNe.terminal_velocity(
                cmc.liquid,
                cmc.Ch2022.rain,
                ᶜρ⁰,
                max(zero(Y.c.ρ), ᶜq_liq⁰)))
            ᶜwᵢ⁰ = @. lazy(CMNe.terminal_velocity(
                cmc.ice,
                cmc.Ch2022.small_ice,
                ᶜρ⁰,
                max(zero(Y.c.ρ), ᶜq_ice⁰)))
            ᶜwᵣ⁰ = @. lazy(CM1.terminal_velocity(
                cmp.pr,
                cmp.tv.rain,
                ᶜρ⁰,
                max(zero(Y.c.ρ), ᶜq_rai⁰)))
            ᶜwₛ⁰ = @. lazy(CM1.terminal_velocity(
                cmp.ps,
                cmp.tv.snow,
                ᶜρ⁰,
                max(zero(Y.c.ρ), ᶜq_sno⁰)))
        end

        for j in 1:n
            if p.atmos.moisture_model isa NonEquilMoistModel &&
                p.atmos.microphysics_model isa Microphysics1Moment
                @. ᶠu³_diff = (
                    (ᶠu³ʲs.:($$j) + CT3(ᶠinterp(Geometry.WVector(-1 * ᶜwₕʲs.:($$j))))) -
                    (ᶠu³ + CT3(ᶠinterp(Geometry.WVector(-1 * ᶜwₕhₜ))))
                )
            else
                @. ᶠu³_diff = ᶠu³ʲs.:($$j) - (ᶠu³ + CT3(ᶠinterp(Geometry.WVector(-1 * ᶜwₕhₜ))))
            end
            @. ᶜa_scalar =
                (Y.c.sgsʲs.:($$j).mse + ᶜKʲs.:($$j) - ᶜh_tot) *
                draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))
            vtt = vertical_transport(
                ᶜρʲs.:($j),
                ᶠu³_diff,
                ᶜa_scalar,
                dt,
                edmfx_sgsflux_upwinding,
            )
            @. Yₜ.c.ρe_tot += vtt
        end
        # Add the environment fluxes
        if p.atmos.moisture_model isa NonEquilMoistModel &&
            p.atmos.microphysics_model isa Microphysics1Moment

            #TODO - some weird matrix multiplication bug here
            #=
            ᶠwₕ³⁰ = @. lazy(
                ᶠinterp(ᶜq_liq⁰ * (Iₗ(thermo_params, ᶜts⁰) + ᶜΦ + $(Kin(ᶜwₗ⁰, ᶜu⁰))) * CT3(Geometry.WVector(-1 * ᶜwₗ⁰))) +
                ᶠinterp(ᶜq_ice⁰ * (Iᵢ(thermo_params, ᶜts⁰) + ᶜΦ + $(Kin(ᶜwᵢ⁰, ᶜu⁰))) * CT3(Geometry.WVector(-1 * ᶜwᵢ⁰))) +
                ᶠinterp(ᶜq_rai⁰ * (Iₗ(thermo_params, ᶜts⁰) + ᶜΦ + $(Kin(ᶜwᵣ⁰, ᶜu⁰))) * CT3(Geometry.WVector(-1 * ᶜwᵣ⁰))) +
                ᶠinterp(ᶜq_sno⁰ * (Iᵢ(thermo_params, ᶜts⁰) + ᶜΦ + $(Kin(ᶜwₛ⁰, ᶜu⁰))) * CT3(Geometry.WVector(-1 * ᶜwₛ⁰)))
            )
            =#
            ᶠwₕ³⁰ = @. lazy(
                ᶠinterp(ᶜq_liq⁰ * (Iₗ(thermo_params, ᶜts⁰) + ᶜΦ) * CT3(Geometry.WVector(-1 * ᶜwₗ⁰))) +
                ᶠinterp(ᶜq_ice⁰ * (Iᵢ(thermo_params, ᶜts⁰) + ᶜΦ) * CT3(Geometry.WVector(-1 * ᶜwᵢ⁰))) +
                ᶠinterp(ᶜq_rai⁰ * (Iₗ(thermo_params, ᶜts⁰) + ᶜΦ) * CT3(Geometry.WVector(-1 * ᶜwᵣ⁰))) +
                ᶠinterp(ᶜq_sno⁰ * (Iᵢ(thermo_params, ᶜts⁰) + ᶜΦ) * CT3(Geometry.WVector(-1 * ᶜwₛ⁰)))
            )
            @. ᶠu³_diff = (
                (ᶠu³⁰ + ᶠwₕ³⁰) -
                (ᶠu³ + CT3(ᶠinterp(Geometry.WVector(-1 * ᶜwₕhₜ ))))
            )
        else
            @. ᶠu³_diff = (ᶠu³⁰ - (ᶠu³ + CT3(ᶠinterp(Geometry.WVector(-1 * ᶜwₕhₜ)))))
        end
        ᶜmse⁰ = ᶜspecific_env_mse(Y, p)
        @. ᶜa_scalar = (ᶜmse⁰ + ᶜK⁰ - ᶜh_tot) * draft_area(ᶜρa⁰, ᶜρ⁰)
        vtt = vertical_transport(
            ᶜρ⁰,
            ᶠu³_diff,
            ᶜa_scalar,
            dt,
            edmfx_sgsflux_upwinding,
        )
        @. Yₜ.c.ρe_tot += vtt

        if !(p.atmos.moisture_model isa DryModel)
            # Specific humidity fluxes: First sum up the draft fluxes
            for j in 1:n
                if p.atmos.moisture_model isa NonEquilMoistModel &&
                    p.atmos.microphysics_model isa Microphysics1Moment
                    @. ᶠu³_diff = (
                        (ᶠu³ʲs.:($$j) + CT3(ᶠinterp(Geometry.WVector(-1 * ᶜwₜʲs.:($$j))))) -
                        (ᶠu³ + CT3(ᶠinterp(Geometry.WVector(-1 * ᶜwₜqₜ))))
                    )
                else
                    @. ᶠu³_diff = (ᶠu³ʲs.:($$j) - (ᶠu³ + CT3(ᶠinterp(Geometry.WVector(-1 * ᶜwₜqₜ)))))
                end
                @. ᶜa_scalar =
                    (Y.c.sgsʲs.:($$j).q_tot - specific(Y.c.ρq_tot, Y.c.ρ)) *
                    draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))
                vtt = vertical_transport(
                    ᶜρʲs.:($j),
                    ᶠu³_diff,
                    ᶜa_scalar,
                    dt,
                    edmfx_sgsflux_upwinding,
                )
                @. Yₜ.c.ρq_tot += vtt
            end
            # Add the environment fluxes
            ᶜq_tot⁰ = ᶜspecific_env_value(Val(:q_tot), Y, p)
            if p.atmos.moisture_model isa NonEquilMoistModel &&
                p.atmos.microphysics_model isa Microphysics1Moment

                ᶠwₜ³⁰ = @. lazy(
                    ᶠinterp(ᶜq_liq⁰ * CT3(Geometry.WVector(-1 * ᶜwₗ⁰))) +
                    ᶠinterp(ᶜq_ice⁰ * CT3(Geometry.WVector(-1 * ᶜwᵢ⁰))) +
                    ᶠinterp(ᶜq_rai⁰ * CT3(Geometry.WVector(-1 * ᶜwᵣ⁰))) +
                    ᶠinterp(ᶜq_sno⁰ * CT3(Geometry.WVector(-1 * ᶜwₛ⁰)))
                )

                @. ᶠu³_diff = (
                    (ᶠu³⁰ + ᶠwₜ³⁰) -
                    (ᶠu³ + CT3(ᶠinterp(Geometry.WVector(-1 * ᶜwₜqₜ))))
                )
            else
                @. ᶠu³_diff = (ᶠu³⁰ - (ᶠu³ + CT3(ᶠinterp(Geometry.WVector(-1 * ᶜwₜqₜ)))))
            end

            @. ᶜa_scalar =
                (ᶜq_tot⁰ - specific(Y.c.ρq_tot, Y.c.ρ)) * draft_area(ᶜρa⁰, ᶜρ⁰)
            vtt = vertical_transport(
                ᶜρ⁰,
                ᶠu³_diff,
                ᶜa_scalar,
                dt,
                edmfx_sgsflux_upwinding,
            )
            @. Yₜ.c.ρq_tot += vtt
        end

        # TODO - Add 2M tracer velocities when sedimentation in the updrafts is added
        microphysics_1M_tracers = (
            (@name(c.sgsʲs.:(1).q_liq), @name(c.ρq_liq), :q_liq, @name(ᶜwₗʲs.:(1)), @name(ᶜwₗ)),
            (@name(c.sgsʲs.:(1).q_ice), @name(c.ρq_ice), :q_ice, @name(ᶜwᵢʲs.:(1)), @name(ᶜwᵢ)),
            (@name(c.sgsʲs.:(1).q_rai), @name(c.ρq_rai), :q_rai, @name(ᶜwᵣʲs.:(1)), @name(ᶜwᵣ)),
            (@name(c.sgsʲs.:(1).q_sno), @name(c.ρq_sno), :q_sno, @name(ᶜwₛʲs.:(1)), @name(ᶜwₛ)),
        )
        microphysics_2M_tracers = (
            (@name(c.sgsʲs.:(1).q_liq), @name(c.ρq_liq), :q_liq, @name(ᶜwₗ)),
            (@name(c.sgsʲs.:(1).q_ice), @name(c.ρq_ice), :q_ice, @name(ᶜwᵢ)),
            (@name(c.sgsʲs.:(1).q_rai), @name(c.ρq_rai), :q_rai, @name(ᶜwᵣ)),
            (@name(c.sgsʲs.:(1).q_sno), @name(c.ρq_sno), :q_sno, @name(ᶜwₛ)),
            (@name(c.sgsʲs.:(1).n_liq), @name(c.ρn_liq), :n_liq, @name(ᶜwnₗ)),
            (@name(c.sgsʲs.:(1).n_rai), @name(c.ρn_rai), :n_rai, @name(ᶜwnᵣ)),
        )

        # Liquid, ice, rain and snow specific humidity fluxes
        if p.atmos.moisture_model isa NonEquilMoistModel &&
            p.atmos.microphysics_model isa Microphysics1Moment

            for j in 1:n
                MatrixFields.unrolled_foreach(
                    microphysics_1M_tracers,
                ) do (χʲ_name, ρχ_name, _, wʲ_name, w_name)
                    ᶜχʲ = MatrixFields.get_field(Y, χʲ_name)
                    ᶜρχ = MatrixFields.get_field(Y, ρχ_name)
                    ᶜχ = (@. lazy(specific(ᶜρχ, Y.c.ρ)))
                    ᶜwʲ = MatrixFields.get_field(p.precomputed, wʲ_name)
                    ᶜw = MatrixFields.get_field(p.precomputed, w_name)

                    ᶠw³ʲ = (@. lazy(CT3(ᶠinterp(Geometry.WVector(-1 * ᶜwʲ)))))
                    ᶠw³ = (@. lazy(CT3(ᶠinterp(Geometry.WVector(-1 * ᶜw)))))

                    @. ᶠu³_diff = (ᶠu³ʲs.:($$j) + ᶠw³ʲ) - (ᶠu³ + ᶠw³)

                    @. ᶜa_scalar =
                        (ᶜχʲ - ᶜχ) * draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))
                    vtt = vertical_transport(
                        ᶜρʲs.:($j),
                        ᶠu³_diff,
                        ᶜa_scalar,
                        dt,
                        edmfx_sgsflux_upwinding,
                    )
                    ᶜρχₜ = MatrixFields.get_field(Yₜ, ρχ_name)
                    @. ᶜρχₜ += vtt
                end
            end
            MatrixFields.unrolled_foreach(
                microphysics_1M_tracers,
            ) do (_, ρχ_name, χ_name, _, w_name)
                ᶜχ⁰ = ᶜspecific_env_value(Val(χ_name), Y, p)
                ᶜρχ = MatrixFields.get_field(Y, ρχ_name)
                ᶜχ = (@. lazy(specific(ᶜρχ, Y.c.ρ)))
                ᶜw = MatrixFields.get_field(p.precomputed, w_name)

                ᶠw³ = (@. lazy(CT3(ᶠinterp(Geometry.WVector(-1 * ᶜw)))))

                named_tuple_of_env_velocities = (; q_liq = ᶜwₗ⁰, q_ice = ᶜwᵢ⁰, q_rai = ᶜwᵣ⁰, q_sno = ᶜwₛ⁰)
                ᶜw⁰_tmp = p.scratch.ᶜtemp_scalar_3
                # TODO
                #ᶜw⁰_tmp .= named_tuple_of_env_velocities.(Val(χ_name))
                if χ_name == :q_liq
                    ᶜw⁰_tmp = ᶜwₗ⁰
                elseif χ_name == :q_ice
                    ᶜw⁰_tmp = ᶜwᵢ⁰
                elseif χ_name == :q_rai
                    ᶜw⁰_tmp = ᶜwᵣ⁰
                elseif χ_name == :q_sno
                    ᶜw⁰_tmp = ᶜwₛ⁰
                else
                    error("Wrong microphysics tracer")
                end
                ᶠw³⁰ = (@. lazy(CT3(ᶠinterp(Geometry.WVector(-1 * ᶜw⁰_tmp)))))

                @. ᶠu³_diff = (ᶠu³⁰ + ᶠw³⁰) - (ᶠu³ + ᶠw³)

                @. ᶜa_scalar = (ᶜχ⁰ - ᶜχ) * draft_area(ᶜρa⁰, ᶜρ⁰)
                vtt = vertical_transport(
                    ᶜρ⁰,
                    ᶠu³_diff,
                    ᶜa_scalar,
                    dt,
                    edmfx_sgsflux_upwinding,
                )
                ᶜρχₜ = MatrixFields.get_field(Yₜ, ρχ_name)
                @. ᶜρχₜ += vtt
            end
        end
        if p.atmos.moisture_model isa NonEquilMoistModel &&
            p.atmos.microphysics_model isa Microphysics2Moment

            for j in 1:n
                MatrixFields.unrolled_foreach(
                    microphysics_2M_tracers,
                ) do (χʲ_name, ρχ_name, _)
                    ᶜχʲ = MatrixFields.get_field(Y, χʲ_name)
                    ᶜρχ = MatrixFields.get_field(Y, ρχ_name)
                    ᶜχ = (@. lazy(specific(ᶜρχ, Y.c.ρ)))

                    ᶠw³ = (@. lazy(CT3(ᶠinterp(Geometry.WVector(-1 * MatrixFields.get_field(p.precomputed, w_name))))))

                    @. ᶠu³_diff = ᶠu³ʲs.:($$j) - (ᶠu³ + ᶠw³)

                    @. ᶜa_scalar =
                        (ᶜχʲ - ᶜχ) * draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))
                    vtt = vertical_transport(
                        ᶜρʲs.:($j),
                        ᶠu³_diff,
                        ᶜa_scalar,
                        dt,
                        edmfx_sgsflux_upwinding,
                    )
                    ᶜρχₜ = MatrixFields.get_field(Yₜ, ρχ_name)
                    @. ᶜρχₜ += vtt
                end
            end
            MatrixFields.unrolled_foreach(
                microphysics_1M_tracers,
            ) do (_, ρχ_name, χ_name, _, w_name)
                ᶜχ⁰ = ᶜspecific_env_value(Val(χ_name), Y, p)
                ᶜρχ = MatrixFields.get_field(Y, ρχ_name)
                ᶜχ = (@. lazy(specific(ᶜρχ, Y.c.ρ)))

                ᶠw³ = (@. lazy(CT3(ᶠinterp(Geometry.WVector(-1 * MatrixFields.get_field(p.precomputed, w_name))))))

                @. ᶠu³_diff = ᶠu³⁰ - (ᶠu³ + ᶠw³)

                @. ᶜa_scalar = (ᶜχ⁰ - ᶜχ) * draft_area(ᶜρa⁰, ᶜρ⁰)
                vtt = vertical_transport(
                    ᶜρ⁰,
                    ᶠu³_diff,
                    ᶜa_scalar,
                    dt,
                    edmfx_sgsflux_upwinding,
                )
                ᶜρχₜ = MatrixFields.get_field(Yₜ, ρχ_name)
                @. ᶜρχₜ += vtt
            end
        end
    end
    # TODO - add vertical momentum fluxes
    return nothing
end

function edmfx_sgs_mass_flux_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::DiagnosticEDMFX,
)

    turbconv_params = CAP.turbconv_params(p.params)
    a_max = CAP.max_area(turbconv_params)
    n = n_mass_flux_subdomains(turbconv_model)
    (; edmfx_sgsflux_upwinding) = p.atmos.numerics
    (; ᶠu³) = p.precomputed
    (; ᶜρaʲs, ᶜρʲs, ᶠu³ʲs, ᶜKʲs, ᶜmseʲs, ᶜq_totʲs, ᶜts) = p.precomputed
    (; dt) = p
    ᶜJ = Fields.local_geometry_field(Y.c).J
    FT = eltype(Y)

    if p.atmos.edmfx_model.sgs_mass_flux isa Val{true}
        thermo_params = CAP.thermodynamics_params(p.params)
        # energy
        ᶜh_tot = @. lazy(
            TD.total_specific_enthalpy(
                thermo_params,
                ᶜts,
                specific(Y.c.ρe_tot, Y.c.ρ),
            ),
        )
        ᶠu³_diff = p.scratch.ᶠtemp_CT3
        ᶜa_scalar = p.scratch.ᶜtemp_scalar
        for j in 1:n
            @. ᶠu³_diff = ᶠu³ʲs.:($$j) - ᶠu³
            @. ᶜa_scalar =
                (ᶜmseʲs.:($$j) + ᶜKʲs.:($$j) - ᶜh_tot) * min(
                    min(draft_area(ᶜρaʲs.:($$j), ᶜρʲs.:($$j)), a_max),
                    FT(0.02) / max(
                        Geometry.WVector(ᶜinterp(ᶠu³_diff)).components.data.:1,
                        eps(FT),
                    ),
                )
            vtt = vertical_transport(
                ᶜρʲs.:($j),
                ᶠu³_diff,
                ᶜa_scalar,
                dt,
                edmfx_sgsflux_upwinding,
            )
            @. Yₜ.c.ρe_tot += vtt
        end
        # TODO: add environment flux?

        if !(p.atmos.moisture_model isa DryModel)
            # Specific humidity fluxes
            for j in 1:n
                @. ᶠu³_diff = ᶠu³ʲs.:($$j) - ᶠu³
                # @. ᶜa_scalar =
                #     (ᶜq_totʲs.:($$j) - specific(Y.c.ρq_tot, Y.c.ρ) *
                #     draft_area(ᶜρaʲs.:($$j), ᶜρʲs.:($$j))
                # TODO: remove this filter when mass flux is treated implicitly
                @. ᶜa_scalar =
                    (ᶜq_totʲs.:($$j) - specific(Y.c.ρq_tot, Y.c.ρ)) * min(
                        min(draft_area(ᶜρaʲs.:($$j), ᶜρʲs.:($$j)), a_max),
                        FT(0.02) / max(
                            Geometry.WVector(
                                ᶜinterp(ᶠu³_diff),
                            ).components.data.:1,
                            eps(FT),
                        ),
                    )
                vtt = vertical_transport(
                    ᶜρʲs.:($j),
                    ᶠu³_diff,
                    ᶜa_scalar,
                    dt,
                    edmfx_sgsflux_upwinding,
                )
                @. Yₜ.c.ρq_tot += vtt
            end
            # TODO: add environment flux?
        end
        # TODO: the following adds the environment flux to the tendency
        # Make active and test later
        # @. ᶠu³_diff = p.precomputed.ᶠu³⁰ - ᶠu³
        # ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model)
        # ᶜρ⁰ = p.scratch.ᶜtemp_scalar_2
        # @. ᶜρ⁰ = TD.air_density(
        #     CAP.thermodynamics_params(p.params),
        #     p.precomputed.ᶜts⁰,
        # )
        # ᶜmse⁰ = @.lazy(ᶜspecific_env_mse(Y, p))
        # @. ᶜa_scalar =
        #     (ᶜmse⁰ + p.precomputed.ᶜK⁰ - ᶜh_tot) * draft_area(ᶜρa⁰, ᶜρ⁰)
        # vtt = vertical_transport(
        #     ᶜρ⁰,
        #     ᶠu³_diff,
        #     ᶜa_scalar,
        #     dt,
        #     edmfx_sgsflux_upwinding,
        # )
        # @. Yₜ.c.ρe_tot += vtt
        # if !(p.atmos.moisture_model isa DryModel)
        #     ᶜq_tot⁰ = @specific_env_value(:q_tot, Y.c, turbconv_model))
        #     @. ᶜa_scalar =
        #         (ᶜq_tot⁰ - specific(Y.c.ρq_tot, Y.c.ρ)) *
        #         draft_area(ᶜρa⁰, ᶜρ⁰)
        #     vtt = vertical_transport(
        #         ᶜρ⁰,
        #         ᶠu³_diff,
        #         ᶜa_scalar,
        #         dt,
        #         edmfx_sgsflux_upwinding,
        #     )
        #     @. Yₜ.c.ρq_tot += vtt
        # end
    end

end

"""
    edmfx_sgs_diffusive_flux_tendency!(Yₜ, Y, p, t, turbconv_model)

Computes and applies the tendency to the grid-mean state `Y` due to SGS
diffusive fluxes from the EDMFX environment. This involves calculating the
divergence of turbulent fluxes, which are parameterized using eddy diffusivity
and viscosity closures.

This function parameterizes these fluxes using an eddy-diffusivity/viscosity
approach (K-theory) for the environment (sgs⁰). Tendencies are calculated for
total energy, moisture species, momentum, and optionally TKE.
The form is typically `- ∂/∂z(-D ∂ϕ/∂z)`, where `D` is an effective SGS eddy
diffusivity for the quantity `ϕ`.

The specific implementation depends on the `turbconv_model`. A generic fallback
doing nothing is also provided. The function modifies `Yₜ.c` (grid-mean tendencies)
in place.

Arguments:
- `Yₜ`: The tendency state vector for grid-mean variables.
- `Y`: The current state vector (used for grid-mean and SGS properties).
- `p`: Cache containing parameters, precomputed fields, atmospheric model settings,
       and scratch space.
- `t`: Current simulation time.
- `turbconv_model`: The turbulence convection model instance.
"""
edmfx_sgs_diffusive_flux_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing

function edmfx_sgs_diffusive_flux_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
)
    FT = Spaces.undertype(axes(Y.c))
    (; dt, params) = p
    turbconv_params = CAP.turbconv_params(params)
    c_d = CAP.tke_diss_coeff(turbconv_params)
    (; ᶜu⁰, ᶜK⁰, ᶜlinear_buoygrad, ᶜstrain_rate_norm) = p.precomputed
    (; ρatke_flux) = p.precomputed
    ᶠgradᵥ = Operators.GradientC2F()
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
    ᶜtke⁰ = @. lazy(specific_tke(Y.c.ρ, Y.c.sgs⁰.ρatke, ᶜρa⁰, turbconv_model))

    if p.atmos.edmfx_model.sgs_diffusive_flux isa Val{true}

        (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = p.precomputed
        # scratch to prevent GPU Kernel parameter memory error
        ᶜmixing_length_field = p.scratch.ᶜtemp_scalar_2
        ᶜmixing_length_field .= ᶜmixing_length(Y, p)
        ᶜK_u = @. lazy(
            eddy_viscosity(turbconv_params, ᶜtke⁰, ᶜmixing_length_field),
        )
        ᶜprandtl_nvec = @. lazy(
            turbulent_prandtl_number(
                params,
                ᶜlinear_buoygrad,
                ᶜstrain_rate_norm,
            ),
        )
        ᶜK_h = @. lazy(eddy_diffusivity(ᶜK_u, ᶜprandtl_nvec))
        ᶠρaK_h = p.scratch.ᶠtemp_scalar
        @. ᶠρaK_h = ᶠinterp(ᶜρa⁰) * ᶠinterp(ᶜK_h)
        ᶠρaK_u = p.scratch.ᶠtemp_scalar
        @. ᶠρaK_u = ᶠinterp(ᶜρa⁰) * ᶠinterp(ᶜK_u)

        # Total enthalpy diffusion
        ᶜdivᵥ_ρe_tot = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(C3(FT(0))),
        )

        ᶜmse⁰ = ᶜspecific_env_mse(Y, p)
        @. Yₜ.c.ρe_tot -= ᶜdivᵥ_ρe_tot(-(ᶠρaK_h * ᶠgradᵥ(ᶜmse⁰ + ᶜK⁰)))
        if use_prognostic_tke(turbconv_model)
            # Turbulent TKE transport (diffusion)
            ᶜdivᵥ_ρatke = Operators.DivergenceF2C(
                top = Operators.SetValue(C3(FT(0))),
                bottom = Operators.SetValue(ρatke_flux),
            )
            # Add flux divergence and dissipation term, relaxing TKE to zero
            # in one time step if tke < 0
            @. Yₜ.c.sgs⁰.ρatke -=
                ᶜdivᵥ_ρatke(-(ᶠρaK_u * ᶠgradᵥ(ᶜtke⁰))) + ifelse(
                    ᶜtke⁰ >= FT(0),
                    tke_dissipation(
                        turbconv_params,
                        Y.c.sgs⁰.ρatke,
                        ᶜtke⁰,
                        ᶜmixing_length_field,
                    ),
                    Y.c.sgs⁰.ρatke / float(dt),
                )
        end
        if !(p.atmos.moisture_model isa DryModel)
            # Specific humidity diffusion
            ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar
            ᶜdivᵥ_ρq_tot = Operators.DivergenceF2C(
                top = Operators.SetValue(C3(FT(0))),
                bottom = Operators.SetValue(C3(FT(0))),
            )
            ᶜq_tot⁰ = ᶜspecific_env_value(Val(:q_tot), Y, p)
            @. ᶜρχₜ_diffusion = ᶜdivᵥ_ρq_tot(-(ᶠρaK_h * ᶠgradᵥ(ᶜq_tot⁰)))
            @. Yₜ.c.ρq_tot -= ᶜρχₜ_diffusion
            @. Yₜ.c.ρ -= ᶜρχₜ_diffusion  # Effect of moisture diffusion on (moist) air mass
        end

        cloud_tracers = (
            (@name(c.ρq_liq), :q_liq),
            (@name(c.ρq_ice), :q_ice),
            (@name(c.ρn_liq), :n_liq),
        )
        precip_tracers = (
            (@name(c.ρq_rai), :q_rai),
            (@name(c.ρq_sno), :q_sno),
            (@name(c.ρn_rai), :n_rai),
        )

        α_vert_diff_tracer = CAP.α_vert_diff_tracer(params)
        ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar
        ᶜdivᵥ_ρq = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(C3(FT(0))),
        )
        MatrixFields.unrolled_foreach(cloud_tracers) do (ρχ_name, χ_name)
            MatrixFields.has_field(Y, ρχ_name) || return
            ᶜχ⁰ = ᶜspecific_env_value(Val(χ_name), Y, p)
            @. ᶜρχₜ_diffusion = ᶜdivᵥ_ρq(-(ᶠρaK_h * ᶠgradᵥ(ᶜχ⁰)))
            ᶜρχₜ = MatrixFields.get_field(Yₜ, ρχ_name)
            @. ᶜρχₜ -= ᶜρχₜ_diffusion
        end
        # TODO - do I need to change anything in the implicit solver
        # to include the α_vert_diff_tracer?
        MatrixFields.unrolled_foreach(precip_tracers) do (ρχ_name, χ_name)
            MatrixFields.has_field(Y, ρχ_name) || return
            ᶜχ⁰ = ᶜspecific_env_value(Val(χ_name), Y, p)
            @. ᶜρχₜ_diffusion =
                ᶜdivᵥ_ρq(-(ᶠρaK_h * α_vert_diff_tracer * ᶠgradᵥ(ᶜχ⁰)))
            ᶜρχₜ = MatrixFields.get_field(Yₜ, ρχ_name)
            @. ᶜρχₜ -= ᶜρχₜ_diffusion
        end

        # Momentum diffusion
        ᶠstrain_rate = p.scratch.ᶠtemp_UVWxUVW
        ᶠstrain_rate .= compute_strain_rate_face(ᶜu⁰)
        @. Yₜ.c.uₕ -= C12(ᶜdivᵥ(-(2 * ᶠρaK_u * ᶠstrain_rate)) / Y.c.ρ)
    end
    return nothing
end

function edmfx_sgs_diffusive_flux_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::Union{EDOnlyEDMFX, DiagnosticEDMFX},
)

    # Assumes envinronmental area fraction is 1 (so draft area fraction is negligible)
    # TODO: Relax this assumption and construct diagnostic EDMF fluxes in parallel to
    # prognostic fluxes
    FT = Spaces.undertype(axes(Y.c))
    (; dt, params) = p
    turbconv_params = CAP.turbconv_params(params)
    thermo_params = CAP.thermodynamics_params(params)
    c_d = CAP.tke_diss_coeff(turbconv_params)
    (; ᶜu, ᶜts) = p.precomputed
    (; ρatke_flux) = p.precomputed
    ᶠgradᵥ = Operators.GradientC2F()
    ᶜtke⁰ = @. lazy(specific_tke(Y.c.ρ, Y.c.sgs⁰.ρatke, Y.c.ρ, turbconv_model))

    if p.atmos.edmfx_model.sgs_diffusive_flux isa Val{true}

        (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = p.precomputed
        # scratch to prevent GPU Kernel parameter memory error
        ᶜmixing_length_field = p.scratch.ᶜtemp_scalar_2
        ᶜmixing_length_field .= ᶜmixing_length(Y, p)
        ᶜK_u = @. lazy(
            eddy_viscosity(turbconv_params, ᶜtke⁰, ᶜmixing_length_field),
        )
        ᶜprandtl_nvec = @. lazy(
            turbulent_prandtl_number(
                params,
                ᶜlinear_buoygrad,
                ᶜstrain_rate_norm,
            ),
        )
        ᶜK_h = @. lazy(eddy_diffusivity(ᶜK_u, ᶜprandtl_nvec))

        ᶠρaK_h = p.scratch.ᶠtemp_scalar
        @. ᶠρaK_h = ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜK_h)
        ᶠρaK_u = p.scratch.ᶠtemp_scalar
        @. ᶠρaK_u = ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜK_u)

        # Total enthalpy diffusion
        ᶜdivᵥ_ρe_tot = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(C3(FT(0))),
        )
        ᶜh_tot = @. lazy(
            TD.total_specific_enthalpy(
                thermo_params,
                ᶜts,
                specific(Y.c.ρe_tot, Y.c.ρ),
            ),
        )
        @. Yₜ.c.ρe_tot -= ᶜdivᵥ_ρe_tot(-(ᶠρaK_h * ᶠgradᵥ(ᶜh_tot)))

        if use_prognostic_tke(turbconv_model)
            # Turbulent TKE transport (diffusion)
            ᶜdivᵥ_ρatke = Operators.DivergenceF2C(
                top = Operators.SetValue(C3(FT(0))),
                bottom = Operators.SetValue(ρatke_flux),
            )
            # Add flux divergence and dissipation term, relaxing TKE to zero
            # in one time step if tke < 0
            @. Yₜ.c.sgs⁰.ρatke -=
                ᶜdivᵥ_ρatke(-(ᶠρaK_u * ᶠgradᵥ(ᶜtke⁰))) + ifelse(
                    ᶜtke⁰ >= FT(0),
                    tke_dissipation(
                        turbconv_params,
                        Y.c.sgs⁰.ρatke,
                        ᶜtke⁰,
                        ᶜmixing_length_field,
                    ),
                    Y.c.sgs⁰.ρatke / float(dt),
                )
        end

        if !(p.atmos.moisture_model isa DryModel)
            # Specific humidity diffusion
            ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar
            ᶜdivᵥ_ρq_tot = Operators.DivergenceF2C(
                top = Operators.SetValue(C3(FT(0))),
                bottom = Operators.SetValue(C3(FT(0))),
            )
            @. ᶜρχₜ_diffusion =
                ᶜdivᵥ_ρq_tot(-(ᶠρaK_h * ᶠgradᵥ(specific(Y.c.ρq_tot, Y.c.ρ))))
            @. Yₜ.c.ρq_tot -= ᶜρχₜ_diffusion
            @. Yₜ.c.ρ -= ᶜρχₜ_diffusion
        end

        cloud_tracers = (@name(c.ρq_liq), @name(c.ρq_ice), @name(c.ρn_liq))
        precip_tracers = (@name(c.ρq_rai), @name(c.ρq_sno), @name(c.ρn_rai))

        α = CAP.α_vert_diff_tracer(params)
        ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar
        ᶜdivᵥ_ρq = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(C3(FT(0))),
        )
        MatrixFields.unrolled_foreach(cloud_tracers) do ρχ_name
            MatrixFields.has_field(Y, ρχ_name) || return
            ᶜρχ = MatrixFields.get_field(Y, ρχ_name)
            ᶜχ = (@. lazy(specific(ᶜρχ, Y.c.ρ)))
            @. ᶜρχₜ_diffusion = ᶜdivᵥ_ρq(-(ᶠρaK_h * ᶠgradᵥ(ᶜχ)))
            ᶜρχₜ = MatrixFields.get_field(Yₜ, ρχ_name)
            @. ᶜρχₜ -= ᶜρχₜ_diffusion
        end
        MatrixFields.unrolled_foreach(precip_tracers) do ρχ_name
            MatrixFields.has_field(Y, ρχ_name) || return
            ᶜρχ = MatrixFields.get_field(Y, ρχ_name)
            ᶜχ = (@. lazy(specific(ᶜρχ, Y.c.ρ)))
            @. ᶜρχₜ_diffusion = ᶜdivᵥ_ρq(-(ᶠρaK_h * α * ᶠgradᵥ(ᶜχ)))
            ᶜρχₜ = MatrixFields.get_field(Yₜ, ρχ_name)
            @. ᶜρχₜ -= ᶜρχₜ_diffusion
        end

        # Momentum diffusion
        ᶠstrain_rate = p.scratch.ᶠtemp_UVWxUVW
        ᶠstrain_rate .= compute_strain_rate_face(ᶜu)
        @. Yₜ.c.uₕ -= C12(ᶜdivᵥ(-(2 * ᶠρaK_u * ᶠstrain_rate)) / Y.c.ρ)
    end

    return nothing
end
