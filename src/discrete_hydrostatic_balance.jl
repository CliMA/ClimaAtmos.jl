import LinearAlgebra: norm_sqr
import Thermodynamics as TD
import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields
import ClimaCore.Spaces as Spaces
import ClimaCore.Operators as Operators
import ClimaAtmos.InitialConditions as ICs

"""
    set_discrete_hydrostatic_balanced_state!(Y, p)
Modify the energy variable in state `Y` given Y and the cache `p` so that 
`Y` is in discrete hydrostatic balance.
"""
function set_discrete_hydrostatic_balanced_state!(Y, p)
    ᶜinterp = Operators.InterpolateF2C()
    FT = Spaces.undertype(axes(Y.c))
    Fields.bycolumn(axes(Y.c.ρ)) do colidx
        set_discrete_hydrostatic_balanced_pressure!(
            p.ᶜp,
            similar(Y.f.w),
            Y.c.ρ,
            p.ᶠgradᵥ_ᶜΦ,
            FT(CAP.MSLP(p.params)),
            colidx,
        )
    end
    thermo_params = CAP.thermodynamics_params(p.params)
    C123 = Geometry.Covariant123Vector
    @. p.ᶜK = norm_sqr(C123(Y.c.uₕ) + C123(ᶜinterp(Y.f.w))) / 2
    if p.atmos.moisture_model isa DryModel
        @. p.ᶜts = TD.PhaseDry_ρp(thermo_params, Y.c.ρ, p.ᶜp)
    elseif p.atmos.moisture_model isa EquilMoistModel
        @. p.ᶜts =
            TD.PhaseEquil_ρpq(thermo_params, Y.c.ρ, p.ᶜp, Y.c.ρq_tot / Y.c.ρ)
    else
        error("Unsupported moisture model")
    end
    # assume ᶜΦ has been updated
    ᶜ𝔼_kwarg = @. ICs.energy_vars(thermo_params, p.ᶜts, p.ᶜK, p.ᶜΦ, p.atmos)
    @. Y.c = merge(Y.c, ᶜ𝔼_kwarg)
end

"""
    set_discrete_hydrostatic_balanced_pressure!(ᶜp, ᶠgradᵥ_ᶜp, ᶜρ, ᶠgradᵥ_ᶜΦ, p1, colidx)
Construct discrete hydrostatic balanced pressure `ᶜp` from density `ᶜρ`, 
potential energy gradient `ᶠgradᵥ_ᶜΦ`, and surface pressure `p1`.

Yₜ.f.w = 0 ==>
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
    colidx,
)
    ᶠinterp = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    @. ᶠgradᵥ_ᶜp[colidx] = -(ᶠgradᵥ_ᶜΦ[colidx] * ᶠinterp(ᶜρ[colidx]))
    ᶜp_data = Fields.field_values(ᶜp[colidx])
    ᶠgradᵥ_ᶜp_data = Fields.field_values(ᶠgradᵥ_ᶜp[colidx])
    ᶜp_data[1] = p1
    for i in 2:Spaces.nlevels(axes(ᶜp))
        ᶜp_data[i] = ᶜp_data[i - 1] + ᶠgradᵥ_ᶜp_data[i].u₃
    end
end
