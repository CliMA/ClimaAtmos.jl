#####
##### Subsidence forcing
#####

import Thermodynamics as TD
import ClimaCore.Fields as Fields

#####
##### Subsidence
#####

subsidence_tracer(::Val{:h_tot}, thermo_params, ts, ρ, ρe_tot) =
    TD.total_specific_enthalpy(thermo_params, ts, ρe_tot / ρ)
subsidence_tracer(::Val{:q_tot}, thermo_params, ts, ρ, ρe_tot) =
    TD.total_specific_humidity(thermo_params, ts)
subsidence_tracer(::Val{:q_liq}, thermo_params, ts, ρ, ρe_tot) =
    TD.liquid_specific_humidity(thermo_params, ts)
subsidence_tracer(::Val{:q_ice}, thermo_params, ts, ρ, ρe_tot) =
    TD.ice_specific_humidity(thermo_params, ts)

function subsidence_tendency(χname, subsidence, thermo_params, ᶜts, ᶜρ, ᶜρe_tot)
    subsidence isa Nothing && return NullBroadcasted()
    ᶠspace = Spaces.face_space(axes(ᶜρ))
    ᶠu³ = if subsidence isa Subsidence
        ᶠsubsidence³(ᶠspace, subsidence)
    else
        subsidence
    end
    ᶜχ = @. lazy(subsidence_tracer(χname, thermo_params, ᶜts, ᶜρ, ᶜρe_tot))
    return subsidence_tendency(ᶜρ, ᶜχ, ᶠu³, Val(:first_order))
end

function ᶠsubsidence³(ᶠspace, subsidence::Subsidence)
    ᶠz = Fields.coordinate_field(ᶠspace).z
    ᶠlg = Fields.local_geometry_field(ᶠspace)
    (; prof) = subsidence
    return @. lazy(prof(ᶠz) * CT3(unit_basis_vector_data(CT3, ᶠlg)))
end
subsidence_tendency(ᶜρ, ᶜχ, ᶠu³, ::Val{:none}) =
    @. lazy(- ᶜρ * (ᶜsubdivᵥ(ᶠu³ * ᶠinterp(ᶜχ)) - ᶜχ * ᶜsubdivᵥ(ᶠu³)))
subsidence_tendency(ᶜρ, ᶜχ, ᶠu³, ::Val{:first_order}) =
    @. lazy(- ᶜρ * (ᶜsubdivᵥ(ᶠupwind1(ᶠu³, ᶜχ)) - ᶜχ * ᶜsubdivᵥ(ᶠu³)))
subsidence_tendency(ᶜρ, ᶜχ, ᶠu³, ::Val{:third_order}) =
    @. lazy(- ᶜρ * (ᶜsubdivᵥ(ᶠupwind3(ᶠu³, ᶜχ)) - ᶜχ * ᶜsubdivᵥ(ᶠu³)))
