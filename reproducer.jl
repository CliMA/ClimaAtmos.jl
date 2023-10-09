#=
julia --project=examples
using Revise; include("reproducer.jl")

For buildkite:
```
  - group: "GPU reproducer"
    steps:

      - label: "CUDA reproducer"
        key: "cuda_repro"
        command: >
          julia --project -e 'using CUDA; CUDA.versioninfo()'
          julia --color=yes --project=examples reproducer.jl
        agents:
          slurm_gpus: 1
```
=#

using Revise
import Random
Random.seed!(1234)

import ClimaCore.Domains as Domains
import ClimaCore.Topologies as Topologies
import ClimaCore.Spaces as Spaces
import ClimaCore.Meshes as Meshes
import ClimaCore.Geometry as Geometry
import ClimaComms

FT = Float32
context = ClimaComms.SingletonCommsContext(ClimaComms.device())
zelem = 10
helem = 4;
Nq = 4;
radius = FT(128);
zlim = (0, 1);
vertdomain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(zlim[1]),
    Geometry.ZPoint{FT}(zlim[2]);
    boundary_tags = (:bottom, :top),
);
vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem);
vtopology = Topologies.IntervalTopology(context, vertmesh);
vspace = Spaces.CenterFiniteDifferenceSpace(vtopology);

hdomain = Domains.SphereDomain(radius);
hmesh = Meshes.EquiangularCubedSphere(hdomain, helem);
htopology = Topologies.Topology2D(context, hmesh);
quad = Spaces.Quadratures.GLL{Nq}();
hspace = Spaces.SpectralElementSpace2D(htopology, quad);
cspace = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace);
fspace = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellFace}(cspace);

import Thermodynamics as TD
t = FT(0)
thermo_params = TD.Thermodynamics.Parameters.ThermodynamicsParameters{FT}(
    273.16,
    101325.0,
    100000.0,
    1859.0,
    4181.0,
    2100.0,
    2.5008f6,
    2.8344f6,
    611.657,
    273.16,
    273.15,
    150.0,
    1000.0,
    298.15,
    6864.8,
    10513.6,
    0.2857143,
    8.31446,
    0.02897,
    0.01801528,
    290.0,
    220.0,
    9.81,
    233.0,
    1.0,
);

nt = (;
    ρ = FT(1),
    ᶜΦ = FT(1),
    ᶜK = FT(1),
    ᶜspecific = (; e_tot = FT(1), q_tot = FT(0.001)),
    ᶜts = zero(TD.PhaseEquil{FT}),
);
fields = fill(nt, cspace);
(; ρ, ᶜspecific, ᶜK, ᶜts, ᶜΦ) = fields;

function thermo_state(thermo_params, ρ, e_int, q_tot)
    get_ts(ρ::Real, e_int::Real, q_tot::Real) = TD.PhaseEquil_ρeq(
        thermo_params,
        ρ,
        e_int,
        q_tot,
        3,
        eltype(thermo_params)(0.003),
    )
    return get_ts(ρ, e_int, q_tot)
end

ts_gs(specific, K, Φ, ρ) =
    thermo_state(thermo_params, ρ, specific.e_tot - K - Φ, specific.q_tot)
@. ᶜts = ts_gs(ᶜspecific, ᶜK, ᶜΦ, ρ)
