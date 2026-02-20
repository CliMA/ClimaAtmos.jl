#=
julia --project=.buildkite
using Revise; include("test/parameterized_tendencies/forcing/held_suarez.jl")
=#
using ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos
import ClimaAtmos as CA
using ClimaCore.CommonSpaces
import ClimaAtmos.Thermodynamics as TD
using ClimaCore: Spaces, Fields, Geometry, ClimaCore
using Test
using Base.Broadcast: materialize

pkgversion(ClimaCore) < v"0.14.20" && exit() # CommonSpaces
using ClimaCore.CommonSpaces

### Common Objects ###
@testset "Held Suarez functions" begin
    FT = Float64
    ᶜspace = ExtrudedCubedSphereSpace(
        FT;
        z_elem = 10,
        z_min = 0,
        z_max = 1,
        radius = 10,
        h_elem = 10,
        n_quad_points = 4,
        staggering = CellCenter(),
    )
    ᶠspace = Spaces.face_space(ᶜspace)
    ᶠz = Fields.coordinate_field(ᶠspace).z
    ᶜz = Fields.coordinate_field(ᶜspace).z
    zmax = maximum(ᶠz)
    ᶜuₕ = map(z -> zero(Geometry.Covariant12Vector{eltype(z)}), ᶜz)
    ᶜuₕₜ = map(z -> zero(Geometry.Covariant12Vector{eltype(z)}), ᶜz)
    @. ᶜuₕ.components.data.:1 = 1
    @. ᶜuₕ.components.data.:2 = 1
    ᶜρ = Fields.Field(FT, ᶜspace)
    ᶜp = Fields.Field(FT, ᶜspace)
    ρe_totₜ = Fields.Field(FT, ᶜspace)
    ts_surf = Fields.Field(TD.PhaseDry{FT}, ᶜspace)
    microphysics_model = CA.DryModel()
    forcing = CA.HeldSuarezForcing()
    params = CA.ClimaAtmosParameters(FT)
    ### Component test begins here
    bc = CA.held_suarez_forcing_tendency_ρe_tot(
        ᶜρ,
        ᶜuₕ,
        ᶜp,
        params,
        ts_surf,
        microphysics_model,
        forcing,
    )
    @. ρe_totₜ = bc
    bc = CA.held_suarez_forcing_tendency_uₕ(
        ᶜuₕ,
        ᶜp,
        params,
        ts_surf,
        microphysics_model,
        forcing,
    )
    @. ᶜuₕₜ = bc
    # TODO: test profiles
end
