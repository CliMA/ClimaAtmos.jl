"""
    SurfaceConditions

Surface flux computation for ClimaAtmos, organized in two layers:

**Model layer** (`AtmosSurface` in `types.jl`): Controls the *type* of surface
temperature evolution via `sfc_temperature` (`SSTFormula` subtype — which
formula) and `surface_model` (`SurfaceTemperatureModel` subtype — prescribed
vs prognostic slab ocean).

**State layer** (`SurfaceState` in `surface_state.jl`): Controls the surface
flux parameterization and boundary values (roughness lengths, exchange
coefficients, prescribed T/q/u/v). Configured via the `surface_setup` kwarg on
`AtmosSimulation` or a setup's `surface_condition()` method, stored in
`p.sfc_setup`.

In **coupler mode** (`CouplerManagedSurface`), ClimaAtmos does not compute
surface fluxes — the coupler writes directly to `p.precomputed.sfc_conditions`.
"""
module SurfaceConditions

import ..Parameters as CAP
import ..DryModel
import ..ZonallySymmetricSST
import ..RCEMIPIISST
import ..ExternalColumnInputSST
import ..SlabOceanSST
import ..PrescribedSST
import ..CT1, ..CT2, ..C12, ..CT12, ..C3
import ..unit_basis_vector_data, ..projected_vector_data
import ..geopotential

import ClimaCore: DataLayouts, Geometry, Fields
import ClimaCore.Geometry: ⊗
import ClimaCore.Utilities: half
import SurfaceFluxes as SF
import SurfaceFluxes.Parameters as SFP
import SurfaceFluxes.UniversalFunctions as UF
import Thermodynamics as TD
import ClimaUtilities.TimeVaryingInputs: evaluate!

import Interpolations
import StaticArrays as SA

include("surface_state.jl")
include("surface_setups.jl")
include("surface_conditions.jl")

end
