import Thermodynamics as TD
import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields
import ClimaCore.Spaces as Spaces
import .InitialConditions as ICs

"""
    set_discrete_hydrostatic_balanced_state!(Y, p)
Modify the energy variable in state `Y` given Y and the cache `p` so that
`Y` is in discrete hydrostatic balance.
"""
function set_discrete_hydrostatic_balanced_state!(Y, p)
    FT = Spaces.undertype(axes(Y.c))
    ᶠgradᵥ_ᶜp = similar(Y.f.u₃)
    thermo_params = CAP.thermodynamics_params(p.params)
    ᶜp = @. Base.materialize(TD.air_pressure(thermo_params, p.precomputed.ᶜts))
    set_discrete_hydrostatic_balanced_pressure!(
        ᶜp,
        ᶠgradᵥ_ᶜp,
        Y.c.ρ,
        p.core.ᶠgradᵥ_ᶜΦ,
        FT(CAP.MSLP(p.params)),
    )
    if p.atmos.moisture_model isa DryModel
        @. p.precomputed.ᶜts =
            TD.PhaseDry_ρp(thermo_params, Y.c.ρ, ᶜp)
    elseif p.atmos.moisture_model isa EquilMoistModel
        @. p.precomputed.ᶜts = TD.PhaseEquil_ρpq(
            thermo_params,
            Y.c.ρ,
            ᶜp,
            Y.c.ρq_tot / Y.c.ρ,
        )
    else
        error("Unsupported moisture model")
    end
    ᶜlocal_geometry = Fields.local_geometry_field(Y.c)
    ls(params, thermo_state, geometry, velocity) =
        ICs.LocalState(; params, thermo_state, geometry, velocity)
    @. Y.c = merge(
        Y.c,
        ICs.energy_variables(
            ls(
                p.params,
                p.precomputed.ᶜts,
                ᶜlocal_geometry,
                Geometry.UVWVector(Y.c.uₕ) +
                Geometry.UVWVector(ᶜinterp(Y.f.u₃)),
            ),
        ),
    ) # broadcasting doesn't seem to work with kwargs, so define interim ls()
end

u₃_component(u::Geometry.AxisTensor) = u.u₃

"""
    set_discrete_hydrostatic_balanced_pressure!(ᶜp, ᶠgradᵥ_ᶜp, ᶜρ, ᶠgradᵥ_ᶜΦ, p1)
Construct discrete hydrostatic balanced pressure `ᶜp` from density `ᶜρ`,
potential energy gradient `ᶠgradᵥ_ᶜΦ`, and surface pressure `p1`.

Yₜ.f.u₃ = 0 ==>
-(ᶠgradᵥ_ᶜp / ᶠinterp(ᶜρ) + ᶠgradᵥ_ᶜΦ) = 0 ==>
ᶠgradᵥ_ᶜp = -(ᶠgradᵥ_ᶜΦ * ᶠinterp(ᶜρ))

ᶠgradᵥ(ᶜp)[i] = ᶠgradᵥ_ᶜp[i] ∀ i ∈ PlusHalf(0):PlusHalf(N) ==>
ᶠgradᵥ(ᶜp)[i] = ᶠgradᵥ_ᶜp[i] ∀ i ∈ PlusHalf(1):PlusHalf(N-1) ==>
ᶠgradᵥ(ᶜp)[PlusHalf(i-1)] = ᶠgradᵥ_ᶜp[PlusHalf(i-1)] ∀ i ∈ 2:N ==>
ᶜp[i] - ᶜp[i-1] = ᶠgradᵥ_ᶜp[PlusHalf(i-1)] ∀ i ∈ 2:N ==>
ᶜp[i] = ᶜp[i-1] + ᶠgradᵥ_ᶜp[PlusHalf(i-1)] ∀ i ∈ 2:N ==>
ᶜp_data[i] = ᶜp_data[i-1] + ᶠgradᵥ_ᶜp_data[i] ∀ i ∈ 2:N
"""
function set_discrete_hydrostatic_balanced_pressure!(
    ᶜp,
    ᶠgradᵥ_ᶜp,
    ᶜρ,
    ᶠgradᵥ_ᶜΦ,
    p1,
)
    @. ᶠgradᵥ_ᶜp = -(ᶠgradᵥ_ᶜΦ * ᶠinterp(ᶜρ))
    ᶜp_data = Fields.field_values(ᶜp)
    ᶠgradᵥ_ᶜp_data = Fields.field_values(ᶠgradᵥ_ᶜp)
    ᶜp_data_lev₋₁ = Spaces.level(ᶜp_data, 1)
    @. ᶜp_data_lev₋₁ = p1
    @inbounds for i in 2:Spaces.nlevels(axes(ᶜp))
        ᶜp_data_lev = Spaces.level(ᶜp_data, i)
        ᶜp_data_lev₋₁ = Spaces.level(ᶜp_data, i - 1)
        u₃_data_lev = Spaces.level(ᶠgradᵥ_ᶜp_data, i)
        @. ᶜp_data_lev = ᶜp_data_lev₋₁ + u₃_component(u₃_data_lev)
    end
end
