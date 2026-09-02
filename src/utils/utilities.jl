#####
##### Utility functions
#####
import ClimaComms
import ClimaCore: Spaces, Topologies, Fields, Geometry, Quadratures, Grids
import ClimaUtilities.TimeManager: ITime
import LinearAlgebra: norm_sqr
using Dates: DateTime, @dateformat_str
import StaticArrays: SVector, SMatrix
import Thermodynamics.Parameters as TDP

"""
    enforce_mass_energy_consistency!(Y, p, ᶜΔρq_tot)

Restore mass and energy consistency after the limiter has changed `ρq_tot` by
`ᶜΔρq_tot`, mutating `Y`.

Water added to (or removed from) the total-water budget must also appear in the
air-mass and energy budgets. The added water is treated as vapor, so `Y.c.ρ` is
incremented by `ᶜΔρq_tot` and `Y.c.ρe_tot` by
`ᶜΔρq_tot (e_int_vapor(T) + Φ)`.

# Arguments

  - `Y`: State; `Y.c.ρ` and `Y.c.ρe_tot` are mutated.
  - `p`: Cache; supplies the parameters, the precomputed temperature `ᶜT`, and
    the geopotential `ᶜΦ` from `p.core`.
  - `ᶜΔρq_tot`: Change in total-water density applied by the limiter [kg/m³].

# Returns

`nothing`.
"""
function enforce_mass_energy_consistency!(Y, p, ᶜΔρq_tot)
    thp = CAP.thermodynamics_params(p.params)
    ᶜT = p.precomputed.ᶜT
    ᶜΦ = p.core.ᶜΦ
    @. Y.c.ρ += ᶜΔρq_tot
    @. Y.c.ρe_tot += ᶜΔρq_tot * (TD.internal_energy_vapor(thp, ᶜT) + ᶜΦ)
    return nothing
end

"""
    is_energy_var(symbol)
    is_momentum_var(symbol)
    is_sgs_var(symbol)
    is_tracer_var(symbol)

Classify a top-level field name of `Y.c` or `Y.f` by the role it plays in the
state.

`is_tracer_var` is the complement: every name that is not `ρ`, `ρtke`, an energy,
a momentum, or an SGS variable.
"""
is_energy_var(symbol) = symbol in (:ρe_tot,)
is_momentum_var(symbol) = symbol in (:uₕ, :u₃)
is_sgs_var(symbol) = symbol in (:sgsʲs,)
is_tracer_var(symbol) = !(
    symbol == :ρ ||
    symbol == :ρtke ||
    is_energy_var(symbol) ||
    is_momentum_var(symbol) ||
    is_sgs_var(symbol)
)

# we may be hitting a slow path:
# https://stackoverflow.com/questions/14687665/very-slow-stdpow-for-bases-very-close-to-1
"""
    fast_pow(x, y)

Compute `x^y` as `exp(y log(x))`, trading a little accuracy for speed.

`^` takes a slow path for bases very close to 1. Requires `x > 0`. Called from
`held_suarez_forcing_tendency!`.
"""
fast_pow(x, y) = exp(y * log(x))

"""
    geopotential(grav, z)

Compute the geopotential `Φ = g z` [m²/s²] at height `z` [m] with gravitational
acceleration `grav` [m/s²].
"""
geopotential(grav, z) = grav * z

"""
    pressure_to_height(p, T, q, thermo_params)

Convert pressure levels to approximate geometric heights with the hypsometric
equation.

The inputs are sorted from the surface upward, the layer-mean virtual
temperature `Tv = T (1 + (R_v/R_d - 1) q)` is formed, and
`Δz = R_d Tv / g · ln(p_below / p_above)` is accumulated from the surface. The
computation is done in `Float64` regardless of the input element type, and the
result is returned in the order of the input `p`.

# Arguments

  - `p`: Pressure levels, in any order [Pa].
  - `T`: Temperature at those levels [K].
  - `q`: Specific humidity at those levels [kg/kg].
  - `thermo_params`: `Thermodynamics` parameter set, supplying `g`, `R_d`, `R_v`.

# Returns

`Vector{Float64}` of heights above the surface, with `z = 0` at the
highest-pressure level [m].
"""
function pressure_to_height(p, T, q, thermo_params)
    g = TDP.grav(thermo_params)
    R_d = TDP.R_d(thermo_params)
    R_v = TDP.R_v(thermo_params)

    p_pa = Float64.(p)
    q_kgkg = Float64.(q)
    T_K = Float64.(T)

    # Sort by pressure (descending = surface to TOA)
    sort_idx = sortperm(p_pa, rev = true)
    p_sorted = p_pa[sort_idx]
    T_sorted = T_K[sort_idx]
    q_sorted = q_kgkg[sort_idx]

    Tv = T_sorted .* (1.0 .+ (R_v / R_d - 1.0) .* q_sorted)

    # Integrate hypsometric equation from surface
    n = length(p_sorted)
    z = zeros(n)
    z[1] = 0.0  # Surface

    for i in 2:n
        Tv_mean = 0.5 * (Tv[i - 1] + Tv[i])
        dz = R_d * Tv_mean / g * log(p_sorted[i - 1] / p_sorted[i])
        z[i] = z[i - 1] + dz
    end

    # Return in original order
    inv_sort_idx = invperm(sort_idx)
    return z[inv_sort_idx]
end

"""
    time_from_filename(file)

Parse the simulation time in seconds from a `dayD.S.hdf5` output filename.

The basename is split on `.`: the first field is the day count, the second the
seconds within that day.

# Examples

```julia
julia> time_from_filename("day4.46906.hdf5")
392506.0
```
"""
function time_from_filename(file)
    arr = split(basename(file), ".")
    day = parse(Float64, replace(arr[1], "day" => ""))
    sec = parse(Float64, arr[2])
    return day * (60 * 60 * 24) + sec
end

"""
    sort_files_by_time(files)

Sort `files` in place by the time parsed from each filename with
`time_from_filename`, and return the permuted array.
"""
sort_files_by_time(files) =
    permute!(files, sortperm(time_from_filename.(files)))

"""
    compute_kinetic(uₕ, uᵥ)

Compute the specific kinetic energy at cell centers from the horizontal and
vertical velocity components,

    κ = 1/2 (uₕ⋅uʰ + 2 uʰ⋅ᶜI(uᵥ) + ᶜI(uᵥ⋅uᵛ)),

where `ᶜI` interpolates from faces to centers.

# Arguments

  - `uₕ`: `Covariant1Vector`- or `Covariant12Vector`-valued center `Field`.
  - `uᵥ`: `Covariant3Vector`-valued face `Field`.

# Returns

A lazy center `Field` with the specific kinetic energy [m²/s²]; materialize it
with `κ .= compute_kinetic(uₕ, uᵥ)`.
"""
function compute_kinetic(uₕ, uᵥ)
    @assert eltype(uₕ) <: Union{C1, C2, C12}
    @assert eltype(uᵥ) <: C3
    FT = Spaces.undertype(axes(uₕ))
    onehalf = FT(1 / 2)
    return @. lazy(
        onehalf * (
            dot(C123(uₕ), CT123(uₕ)) +
            ᶜinterp(dot(C123(uᵥ), CT123(uᵥ))) +
            2 * dot(CT123(uₕ), ᶜinterp(C123(uᵥ)))
        ),
    )
end

"""
    compute_kinetic(Y::Fields.FieldVector)

Compute the specific kinetic energy at cell centers from the model state `Y`,
i.e. from `Y.c.uₕ` and `Y.f.u₃`.
"""
compute_kinetic(Y::Fields.FieldVector) = compute_kinetic(Y.c.uₕ, Y.f.u₃)

"""
    compute_strain_rate_center_vertical(ᶠu)

Compute the strain rate at cell centers from the face velocity `ᶠu`, keeping
vertical gradients only.

# Returns

A lazy center `Field` with the symmetric `UVW × UVW` strain-rate tensor
`(∇ᵥu + (∇ᵥu)ᵀ) / 2` [1/s].
"""
function compute_strain_rate_center_vertical(ᶠu)
    axis_uvw = Geometry.UVWAxis()
    return @. lazy(
        (
            Geometry.project((axis_uvw,), ᶜgradᵥ(UVW(ᶠu))) +
            adjoint(Geometry.project((axis_uvw,), ᶜgradᵥ(UVW(ᶠu))))
        ) / 2,
    )
end

"""
    compute_strain_rate_face_vertical(ᶜu)

Compute the strain rate at cell faces from the center velocity `ᶜu`, keeping
vertical gradients only.

Zero vertical-gradient boundary conditions are imposed at the top and bottom
faces.

# Returns

A lazy face `Field` with the symmetric `UVW × UVW` strain-rate tensor
`(∇ᵥu + (∇ᵥu)ᵀ) / 2` [1/s].
"""
function compute_strain_rate_face_vertical(ᶜu)
    ∇ᵥuvw_boundary = Geometry.outer(Geometry.WVector(0), Geometry.UVWVector(0, 0, 0))
    ∇bc = Operators.SetGradient(∇ᵥuvw_boundary)
    ᶠgradᵥ = Operators.GradientC2F(bottom = ∇bc, top = ∇bc)
    axis_uvw = Geometry.UVWAxis()
    return @. lazy(
        (
            Geometry.project((axis_uvw,), ᶠgradᵥ(UVW(ᶜu))) +
            adjoint(Geometry.project((axis_uvw,), ᶠgradᵥ(UVW(ᶜu))))
        ) / 2,
    )
end

"""
    compute_strain_rate_center_horizontal(ᶜu)

Compute the strain rate at cell centers from velocity at cell centers, with horizontal gradients only.
"""
function compute_strain_rate_center_horizontal(ᶜu)
    axis_uvw = Geometry.UVWAxis()
    return @. lazy(
        (
            Geometry.project((axis_uvw,), gradₕ(UVW(ᶜu))) +
            adjoint(Geometry.project((axis_uvw,), gradₕ(UVW(ᶜu))))
        ) / 2,
    )
end

"""
    compute_strain_rate_center_full!(ᶜε, ᶜu, ᶠu)

Compute the full strain rate tensor at cell centers, mutating `ᶜε`.

The vertical part is the face-to-center gradient of `ᶠu`, the horizontal part the
spectral gradient of `ᶜu`; their sum is then symmetrized.

# Arguments

  - `ᶜε`: Preallocated `UVW × UVW` tensor center `Field`, overwritten with the
    strain rate [1/s].
  - `ᶜu`: Velocity at cell centers.
  - `ᶠu`: Velocity at cell faces. Both reconstructions are needed for the full
    tensor.

# Returns

`ᶜε`.

# Notes

  - Use the `ᶜu` and `ᶠu` produced by `set_velocity_quantities!` and
    `set_implicit_precomputed_quantities_part1!`.
  - Because both vertical and horizontal gradients are involved, the computation
    cannot (yet) be made lazy, hence the preallocated output field.

See also `compute_strain_rate_face_full!` for the face-centered version.
"""
function compute_strain_rate_center_full!(ᶜε, ᶜu, ᶠu)
    axis_uvw = (Geometry.UVWAxis(),)
    @. ᶜε = Geometry.project(axis_uvw, ᶜgradᵥ(UVW(ᶠu)))  # vertical component
    @. ᶜε += Geometry.project(axis_uvw, gradₕ(UVW(ᶜu)))  # horizontal component
    @. ᶜε = (ᶜε + adjoint(ᶜε)) / 2
    return ᶜε
end

"""
    compute_strain_rate_face_full!(ᶠε, ᶜu, ᶠu)

Compute the full strain rate tensor at cell faces, mutating `ᶠε`.

The vertical part is the center-to-face gradient of `ᶜu`, the horizontal part the
spectral gradient of `ᶠu`; their sum is then symmetrized.

# Arguments

  - `ᶠε`: Preallocated `UVW × UVW` tensor face `Field`, overwritten with the
    strain rate [1/s].
  - `ᶜu`: Velocity at cell centers.
  - `ᶠu`: Velocity at cell faces. Both reconstructions are needed for the full
    tensor.

# Returns

`ᶠε`.

# Notes

  - Use the `ᶜu` and `ᶠu` produced by `set_velocity_quantities!` and
    `set_implicit_precomputed_quantities_part1!`.
  - Because both vertical and horizontal gradients are involved, the computation
    cannot (yet) be made lazy, hence the preallocated output field.
  - Zero vertical-gradient boundary conditions are imposed at the top and bottom
    faces.

See also `compute_strain_rate_center_full!` for the center version.
"""
function compute_strain_rate_face_full!(ᶠε, ᶜu, ᶠu)
    ∇ᵥuvw_boundary = Geometry.outer(Geometry.WVector(0), UVW(0, 0, 0))
    ∇bc = Operators.SetGradient(∇ᵥuvw_boundary)
    ᶠgradᵥ = Operators.GradientC2F(bottom = ∇bc, top = ∇bc)
    axis_uvw = (Geometry.UVWAxis(),)
    @. ᶠε = Geometry.project(axis_uvw, ᶠgradᵥ(UVW(ᶜu)))  # vertical component
    @. ᶠε += Geometry.project(axis_uvw, gradₕ(UVW(ᶠu)))  # horizontal component
    @. ᶠε = (ᶠε + adjoint(ᶠε)) / 2
    return ᶠε
end

"""
    strain_rate_norm(S, axis = Geometry.UVWAxis())

Return a lazy representation of the strain rate norm `|S| = √(2 S:S)` [1/s].

The tensor `S` is first projected onto `axis`, so that

  - `axis = Geometry.UVAxis()` gives the horizontal strain rate norm, and
  - `axis = Geometry.WAxis()` gives the vertical strain rate norm.

# Arguments

  - `S`: Strain-rate tensor field [1/s].
  - `axis = Geometry.UVWAxis()`: Axis to project onto before taking the norm.
"""
function strain_rate_norm(S, axis = Geometry.UVWAxis())
    S_proj = @. lazy(Geometry.project((axis,), S, (axis,)))
    S_norm = @. lazy(sqrt(2 * norm_sqr(S_proj)))
    return S_norm
end

"""
    g³³_field(space)

Extract `g³³` from `space`, the `(3, 3)` component of the metric tensor `gⁱʲ`
that converts covariant to contravariant `AxisTensor`s.

The component is the last one of `gⁱʲ` in both 2D (4 components) and 3D (9
components) spaces.

# Returns

A `Field` over `space` with the scalar component `g³³`.
"""
function g³³_field(space)
    g_field = Fields.local_geometry_field(space).gⁱʲ.components.data
    end_index = fieldcount(eltype(g_field)) # This will be 4 in 2D and 9 in 3D.
    return g_field.:($end_index) # For both 2D and 3D spaces, g³³ = g[end].
end

"""
    horizontal_filter_scale(space::Spaces.ExtrudedFiniteDifferenceSpace)
    horizontal_filter_scale(space::Spaces.FiniteDifferenceSpace)

Return the horizontal filter length scale `Δx_h` of `space` [m].

For extruded 2D/3D spaces this is the per-node spectral-element length scale
`Spaces.node_horizontal_length_scale`. For single columns it is `Inf`: a column
has no horizontal discretization, and its filter scale is set by the forcing or
the ensemble it represents, not by a grid length.
"""
horizontal_filter_scale(space::Spaces.ExtrudedFiniteDifferenceSpace) =
    Spaces.undertype(space)(
        Spaces.node_horizontal_length_scale(Spaces.horizontal_space(space)),
    )
# Do not route single columns through node_horizontal_length_scale: its
# PointSpace method returns the placeholder 1 [m].
horizontal_filter_scale(space::Spaces.FiniteDifferenceSpace) =
    Spaces.undertype(space)(Inf)

"""
    resolvability_filter_scale(Δx_h, Δz)
    resolvability_filter_scale(space)

Compute the resolvability filter scale of the dynamical solution [m],

    Δ_f = max(Δx_h, Δz),

the smallest length scale resolvable in *every* direction of the grid: an
eddy can be handed over to the resolved dynamics only if the coarsest grid
direction resolves it, so `Δ_f` is the correct grid-scale cap for SGS mixing
lengths. In a GCM (`Δx_h` ≫ boundary-layer depth) and in a single column
(`Δx_h = Inf`, see `horizontal_filter_scale`) the cap is inert and
the mixing length is purely physical (convergent under vertical refinement);
in the gray zone the cap binds at `Δx_h`, shrinking the SGS eddies as the
horizontal resolution starts to resolve them; in the isotropic LES limit
`Δ_f = Δ` recovers a Deardorff-type grid-scale bound. (In the near-isotropic
regime, a volumetric scale `(Δx Δy Δz)^{1/3}` is a possible refinement —
change it here, in one place.)

The `space` method returns `Δ_f` as a lazy field over `space`, combining
`horizontal_filter_scale` with the local layer thickness `Fields.Δz_field`.
"""
resolvability_filter_scale(Δx_h, Δz) = max(Δx_h, Δz)
function resolvability_filter_scale(space::Spaces.AbstractSpace)
    Δx_h = horizontal_filter_scale(space)
    Δz = Fields.Δz_field(space)
    return @. lazy(resolvability_filter_scale(Δx_h, Δz))
end

"""
    g³³(gⁱʲ)

Extract the `g³³` sub-tensor of the metric tensor `gⁱʲ`, reshaped as a
`Contravariant3Axis × Contravariant3Axis` `AxisTensor`.
"""
g³³(gⁱʲ) = reshape(
    gⁱʲ,
    (Geometry.Contravariant3Axis(), Geometry.Contravariant3Axis()),
)


"""
    g³ʰ(gⁱʲ)

Extract the `g³ʰ` sub-tensor of the metric tensor `gⁱʲ`, the coupling between
the vertical and horizontal contravariant directions that is non-zero over
sloped terrain.

The result is always a `Contravariant3Axis × Contravariant12Axis` `AxisTensor`;
in 2D spaces the missing horizontal component is filled with zero. Throws if
`gⁱʲ` has no vertical or no horizontal sub-axis.
"""
function g³ʰ(gⁱʲ)
    full_CT_axis = axes(gⁱʲ)[1]
    N = length(full_CT_axis)
    gⁱʲ_components = Geometry.components(gⁱʲ)
    FT = eltype(gⁱʲ_components)
    g³ʰ_components = if full_CT_axis == Geometry.Contravariant123Axis()
        @inbounds SMatrix{1, 2, FT, 2}(
            gⁱʲ_components[N, 1],
            gⁱʲ_components[N, 2],
        )
    elseif full_CT_axis == Geometry.Contravariant13Axis()
        @inbounds val = gⁱʲ_components[N, 1]
        SMatrix{1, 2, FT, 2}(val, zero(FT))
    elseif full_CT_axis == Geometry.Contravariant23Axis()
        @inbounds val = gⁱʲ_components[N, 1]
        SMatrix{1, 2, FT, 2}(zero(FT), val)
    else
        error("$full_CT_axis is missing either vertical or horizontal sub-axes")
    end
    axes_tuple = (Geometry.Contravariant3Axis(), Geometry.Contravariant12Axis())
    return Geometry.AxisTensor(axes_tuple, g³ʰ_components)
end

"""
    has_topography(space)

Return `true` if `space` has a non-flat hypsography, i.e. if the model has
terrain. Single columns never do.
"""
has_topography(space::Spaces.FiniteDifferenceSpace) = false
has_topography(space) = Spaces.grid(space).hypsography != Grids.Flat()

"""
    unit_basis_vector_data(type, local_geometry)

Return the component of a vector of type `V` whose physical length is 1.

`V` must be a single-component vector type, i.e. a basis-vector type such as
`CT3`. Multiplying by this factor converts a physical magnitude into the
component of `V`.
"""
function unit_basis_vector_data(::Type{V}, local_geometry) where {V}
    FT = Geometry.undertype(typeof(local_geometry))
    return FT(1) / Geometry._norm(V(FT(1)), local_geometry)
end

"""
    projected_vector_data(::Type{V}, vector, local_geometry)

Project `vector` onto the axis of `V`, then return its single component rescaled
to physical units.

`V` must be a single-component vector type, i.e. a basis-vector type such as
`CT1`. Inverse of the scaling applied by `unit_basis_vector_data`.
"""
projected_vector_data(::Type{V}, vector, local_geometry) where {V} =
    V(vector, local_geometry)[1] / unit_basis_vector_data(V, local_geometry)

"""
    get_physical_w(u, local_geometry)

Return the physical vertical velocity [m/s], the projection of the full velocity
vector `u` onto the local vertical axis.
"""
get_physical_w(u, local_geometry) = Geometry.WVector(u, local_geometry)[1]

"""
    time_to_seconds(t::Number)
    time_to_seconds(t::ITime)
    time_to_seconds(s::String)

Normalize a time specification to a number of seconds, as a `Float64`.

This is the single entry point for the three time representations that arrive
through the script and YAML interfaces:

  - `Number`: already in seconds, converted to `Float64` (`Inf` included).
  - `ITime`: converted to seconds.
  - `String`: a non-negative number immediately followed by one of the units `s`,
    `secs`, `m`, `mins`, `h`, `hours`, `d`, `days`, `weeks`, e.g. `"20mins"`. The
    literal `"Inf"` is also accepted. Anything else throws.
"""
time_to_seconds(t::Number) = Float64(t)
time_to_seconds(t::ITime) = float(t)
function time_to_seconds(s::String)
    s == "Inf" && return Inf
    # match a number followed by one of the supported units of time
    m = match(r"^(\d+(?:\.\d+)?)(s|secs|m|mins|h|hours|d|days|weeks)$", s)
    isnothing(m) &&
        error(
            "Bad format for flag $s. Examples: `10secs`, `20mins`, `30hours`, `40days`, `50weeks`",
        )
    value = parse(Float64, m.captures[1])
    unit = m.captures[2]
    factor_groups = Dict(
        ["s", "secs"] => 1,
        ["m", "mins"] => 60,
        ["h", "hours"] => 3600,
        ["d", "days"] => 86400,
        ["weeks"] => 604800,
    )
    factors = Dict(unit => val for (units, val) in factor_groups for unit in units)
    return value * factors[unit]
end

"""
    error_if_crashed(ret_code)

Throw if `ret_code` is `:simulation_crashed`, and do nothing otherwise.
"""
function error_if_crashed(ret_code)
    ret_code == :simulation_crashed &&
        error("The ClimaAtmos simulation has crashed. See the stack trace for details.")
end

"""
    verify_callbacks(t)

Throw if the saved times `t` contain duplicates, which would mean the callbacks
saved the solution twice at the same timestep.
"""
function verify_callbacks(t)
    if length(t) ≠ length(unique(t))
        @show length(t)
        @show length(unique(t))
        error(
            string(
                "Saving duplicate solutions at the same time.",
                "Please change the callbacks to not save ",
                "duplicate solutions at the same timestep.",
            ),
        )
    end
end


"""
    do_dss(space)

Return whether the horizontal space of `space` requires direct stiffness
summation, i.e. whether its quadrature is Gauss-Lobatto-Legendre.

Single columns have no horizontal space and always return `false`.
"""
function do_dss(space::Spaces.AbstractSpace)
    h_space = Spaces.horizontal_space(space)
    # Discontinuous (DG) spaces are coupled by interface numerical fluxes in
    # the tendencies (Operators.tendency_completion), never by DSS.
    Grids.discretization(Spaces.grid(h_space)) isa Grids.DG && return false
    return Spaces.quadrature_style(h_space) isa Quadratures.GLL
end

function do_dss(::Spaces.FiniteDifferenceSpace)
    return false
end

using ClimaComms
"""
    is_distributed(context)

Return whether `context` is an MPI communications context, i.e. whether the run
spans more than one process.
"""
is_distributed(::ClimaComms.SingletonCommsContext) = false
is_distributed(::ClimaComms.MPICommsContext) = true

"""
    summary_string(x)

Return a string similar to the output of `dump(x)`, but without type parameters.

Recurses through the fields of `x`, indenting one level per nesting depth, and
falls back to `repr` for values without fields. Used to log model and algorithm
configurations whose types are too parameter-heavy to print directly.
"""
summary_string(x) = summary_string(x, 0)
summary_string(x, depth) =
    fieldcount(typeof(x)) == 0 ? repr(x) :
    (string(nameof(typeof(x))) * '(') *
    mapreduce(*, 1:fieldcount(typeof(x))) do i
        field =
            x isa Tuple ? ':' * string(i) : string(fieldname(typeof(x), i))
        ('\n' * "  "^(depth + 1) * field * " = ") *
        (summary_string(getfield(x, i), depth + 1) * ',')
    end *
    ('\n' * "  "^depth * ')')

# From BenchmarkTools
"""
    prettytime(t)

Format a duration `t` [ns] as a string with the largest unit that keeps the
value above 1, e.g. `"1.234 ms"`.
"""
function prettytime(t)
    if t < 1e3
        value, units = t, "ns"
    elseif t < 1e6
        value, units = t / 1e3, "μs"
    elseif t < 1e9
        value, units = t / 1e6, "ms"
    else
        value, units = t / 1e9, "s"
    end
    return "$(round(value, digits=3)) $units"
end

import Dates

"""
    time_and_units_str(x::Real)

Format a time `x` [s] as a human-readable string, truncated to its two largest
non-zero units, e.g. `"1 day, 3 hours"`.
"""
time_and_units_str(x::Real) =
    trunc_time(string(compound_period(x, Dates.Second)))

"""
    compound_period(x::Real, ::Type{T}) where {T <: Dates.Period}

Build a canonicalized `Dates.CompoundPeriod` from the real value `x`, whose
units are given by the period type `T`.

The value is converted to whole nanoseconds, rounding up, before
canonicalization.
"""
function compound_period(x::Real, ::Type{T}) where {T <: Dates.Period}
    nf = Dates.value(convert(Dates.Nanosecond, T(1)))
    ns = Dates.Nanosecond(ceil(x * nf))
    return Dates.canonicalize(Dates.CompoundPeriod(ns))
end

"""
    trunc_time(s::String)

Keep only the first two comma-separated components of a canonicalized period
string, e.g. `"1 day, 3 hours, 4 minutes"` becomes `"1 day, 3 hours"`.
"""
trunc_time(s::String) = count(',', s) > 1 ? join(split(s, ",")[1:2], ",") : s


"""
    prettymemory(b)

Format a memory size `b` [bytes] as a string with the largest binary unit that
keeps the value above 1, e.g. `"340.0 MiB"`.
"""
function prettymemory(b)
    if b < 1024
        return string(b, " bytes")
    elseif b < 1024^2
        value, units = b / 1024, "KiB"
    elseif b < 1024^3
        value, units = b / 1024^2, "MiB"
    else
        value, units = b / 1024^3, "GiB"
    end
    return "$(round(value, digits=2)) $units"
end

"""
    @timed_log verbose "message" expr

Evaluate `expr` and return its value. If `verbose` is true, also `@info` the
message with elapsed time and allocations appended, e.g.
`"Building cache (1.2 s, 340.0 MiB)"`. When `verbose` is false the expression is
evaluated without any logging or timing overhead.

`stats.value` is returned so the macro can wrap the right-hand side of a
destructuring assignment, e.g. `(Y, t_start, spaces) = @timed_log verbose "..." f()`.
"""
macro timed_log(verbose, message, ex)
    quote
        if $(esc(verbose))
            local stats = @timed $(esc(ex))
            @info string(
                $(esc(message)),
                " (",
                prettytime(stats.time * 1e9),
                ", ",
                prettymemory(stats.gcstats.allocd),
                ")",
            )
            stats.value
        else
            $(esc(ex))
        end
    end
end

"""
    AllNothing

Singleton whose every property is `nothing`.

The instance `all_nothing` stands in for a cache or configuration section that
is absent, so that a destructuring assignment such as
`(; ᶜuʲs) = n > 0 ? p.precomputed : all_nothing` stays type-stable and binds
`nothing` when the section does not exist.
"""
struct AllNothing end
const all_nothing = AllNothing()
Base.getproperty(::AllNothing, ::Symbol) = nothing

"""
    horizontal_integral_at_boundary(f::Fields.Field, lev)
    horizontal_integral_at_boundary(f::Fields.Field)

Compute the horizontal integral of `f` at the vertical level `lev`.

The two-argument form takes a face extruded field and first extracts level
`lev`; the one-argument form takes the resulting 2D spectral-element field. The
2D space carries no vertical metric, so the integral is reconstructed from the
level's `Δz` field.
"""
function horizontal_integral_at_boundary(f::Fields.Field, lev)
    @assert axes(f) isa Spaces.FaceExtrudedFiniteDifferenceSpace
    horizontal_integral_at_boundary(Spaces.level(f, lev))
end

function horizontal_integral_at_boundary(f::Fields.Field)
    @assert axes(f) isa Spaces.SpectralElementSpace2D
    sum(f ./ Fields.Δz_field(axes(f)) .* 2) # TODO: is there a way to ensure this is derived from face z? The 2d topology doesn't contain this info
end

"""
    isdivisible(dt_large::Dates.Period, dt_small::Dates.Period)

Check if two periods are evenly divisible, i.e., if the larger period can be
expressed as an integer multiple of the smaller period.

In this, take into account the case when periods do not have fixed size, e.g.,
one month is a variable number of days.

# Examples

```julia
julia> isdivisible(Dates.Year(1), Dates.Month(1))
true

julia> isdivisible(Dates.Month(1), Dates.Day(1))
true

julia> isdivisible(Dates.Month(1), Dates.Week(1))
false
```

# Notes

Not all combinations are implemented. The fallback method warns and returns
`false`; if a combination you need is missing, please add a method.
"""
function isdivisible(dt_large::Dates.Period, dt_small::Dates.Period)
    @warn "The combination $(typeof(dt_large)) and $(dt_small) was not covered. Please add a method to handle this case."
    return false
end

# For FixedPeriod and OtherPeriod, it is easy, we can directly divide the two
# (as long as they are both the same)
function isdivisible(dt_large::Dates.FixedPeriod, dt_small::Dates.FixedPeriod)
    return isinteger(dt_large / dt_small)
end

function isdivisible(dt_large::Dates.OtherPeriod, dt_small::Dates.OtherPeriod)
    return isinteger(dt_large / dt_small)
end

function isdivisible(
    dt_large::Union{Dates.Month, Dates.Year},
    dt_small::Dates.FixedPeriod,
)
    # The only case where periods are commensurate for Month/Year is when we
    # have a Day or an integer divisor of a day. (Note that 365 and 366 don't
    # have any common divisor)
    return isinteger(Dates.Day(1) / dt_small)
end

"""
    promote_period(period::Dates.Period)

Promote a period to the largest possible period type.

A period is re-expressed in the largest unit that divides it exactly: 24 hours
becomes 1 day, 14 days becomes 2 weeks. Millisecond always divides, so a fixed
period is always representable. Variable-length periods (`Month`, `Year`) are
returned unchanged.

# Examples

```julia
julia> promote_period(Hour(24))
1 day

julia> promote_period(Day(14))
2 weeks

julia> promote_period(Second(86401))
86401 seconds

julia> promote_period(Millisecond(1))
1 millisecond
```
"""
function promote_period(period::Dates.Period)
    ms = Int(Dates.toms(period))
    # Hard to do this with varying periods like Month/Year...
    PeriodTypes = [
        Dates.Week,
        Dates.Day,
        Dates.Hour,
        Dates.Minute,
        Dates.Second,
        Dates.Millisecond,
    ]
    for PeriodType in PeriodTypes
        period_ms = Int(Dates.toms(PeriodType(1)))
        if ms % period_ms == 0
            # Millisecond will always match, if nothing else matches
            return PeriodType(ms // period_ms)
        end
    end
end

function promote_period(period::Dates.OtherPeriod)
    # For varying periods, we just return them as they are
    return period
end

"""
    parse_date(date_str::AbstractString)
    parse_date(dt::DateTime)

Parse a date string into a `DateTime`, or pass a `DateTime` through unchanged.

Only two formats are supported, and anything else throws:

  - `yyyymmdd`, e.g. `"20100901"`.
  - `yyyymmdd-HHMM`, e.g. `"20100901-0600"`.
"""
function parse_date(date_str::AbstractString)
    # Define a mapping between allowed formats and corresponding date format
    date_format_mapping = Dict(
        r"^\d{8}$" => dateformat"yyyymmdd",
        r"^\d{8}-\d{4}$" => dateformat"yyyymmdd-HHMM",
    )
    for (pattern, format) in date_format_mapping
        !isnothing(match(pattern, date_str)) &&
            return DateTime(date_str, format)
    end
    error(
        "Date string $date_str does not match any of the allowed formats: yyyymmdd or yyyymmdd-HHMM",
    )
end
parse_date(dt::DateTime) = dt

"""
    iscolumn(space)

Return whether `space` is a single column, i.e. a `FiniteDifferenceSpace`.
"""
iscolumn(space::Spaces.FiniteDifferenceSpace) = true
iscolumn(space) = false

"""
    issphere(space)

Return whether the horizontal domain of `space` is a sphere.
"""
function issphere(space)
    return Meshes.domain(Spaces.topology(Spaces.horizontal_space(space))) isa
           Domains.SphereDomain
end

"""
    clima_to_era5_name_dict()

Return a `Dict` mapping ClimaAtmos (CMIP-style) variable names to the
corresponding ERA5 short names.

Two of the ERA5 names are misleading: `w` is a pressure velocity [Pa/s], and
`z` is a geopotential [m²/s²], not a height.
"""
function clima_to_era5_name_dict()
    Dict(
        "ua" => "u",
        "va" => "v",
        "wap" => "w", # era5 w is in Pa/s, this is confusing notation
        "hus" => "q",
        "ta" => "t",
        "zg" => "z", # era5 z is geopotential in m^2/s^2, this is confusing notation
        "clw" => "clwc",
        "cli" => "ciwc",
        "ts" => "skt",
        "hfls" => "slhf",
        "hfss" => "sshf",
    )
end

"""
    log_context(context)

Log the device and, for distributed runs, the number of processes of the
communications `context`.
"""
function log_context(context)
    device = nameof(typeof(ClimaComms.device(context)))
    if context isa ClimaComms.SingletonCommsContext
        @info "Single-process ClimaAtmos run on $device"
    else
        @info "Distributed ClimaAtmos run" device nprocs = ClimaComms.nprocs(context)
    end
end

"""
    normal_cdf_inv(p)

Compute the standard normal quantile function Φ⁻¹(p) [-] with the rational
approximation of Abramowitz & Stegun (Eq. 26.2.22).

Maximum absolute error ≈ 4.5 × 10⁻⁴ for p ∈ (0, 1). Uses only `log`,
`sqrt`, and polynomial arithmetic — no `SpecialFunctions` dependency. The input
is clamped away from 0 and 1 by `ϵ_numerics(FT)`, so the result is finite for
any `p`.

For 0 < p ≤ ½ the approximation is

    t = √(−2 ln p),
    Φ⁻¹(p) ≈ −(t − (a₀ + a₁t + a₂t²)/(1 + b₁t + b₂t² + b₃t³))

and for p > ½ it uses the antisymmetry Φ⁻¹(p) = −Φ⁻¹(1−p).

Compared with the tanh-based inverse `atanh(2p−1)/coeff_cdf`, this
correctly captures the Gaussian-tail scaling Φ⁻¹(p) ~ −√(−2 ln p)
for small p, avoiding the extreme underestimate that causes the Newton
step to diverge.
"""
@inline function normal_cdf_inv(p::FT) where {FT}
    p_safe = clamp(p, ϵ_numerics(FT), one(FT) - ϵ_numerics(FT))
    q = min(p_safe, one(FT) - p_safe)   # work in the lower tail
    t = sqrt(-FT(2) * log(q))
    num = FT(2.515517) + t * (FT(0.802853) + t * FT(0.010328))
    den = one(FT) + t * (FT(1.432788) + t * (FT(0.189269) + t * FT(0.001308)))
    z = t - num / den
    return ifelse(p_safe < FT(0.5), -z, z)
end

"""
    normal_cdf(z)

Compute the standard normal CDF Φ(z) [-] with the rational approximation of
Abramowitz & Stegun (Eq. 26.2.17).

Maximum absolute error ≈ 7.5 × 10⁻⁸ over the full real line.  Uses only
`exp` and basic arithmetic — no `SpecialFunctions` dependency.

The approximation for z ≥ 0 is

    Φ(z) ≈ 1 − φ(z) · p(t),   t = 1/(1 + 0.2316419 z),

where `p` is a degree-5 polynomial with the A&S coefficients, and
`Φ(z) = 1 − Φ(−z)` for z < 0.
"""
@inline function normal_cdf(z::FT) where {FT}
    inv_sqrt2pi = one(FT) / sqrt(FT(2) * FT(π))
    z_abs = abs(z)
    t = one(FT) / (one(FT) + FT(0.2316419) * z_abs)
    poly =
        t * (
            FT(0.319381530) +
            t * (
                FT(-0.356563782) +
                t * (
                    FT(1.781477937) +
                    t * (FT(-1.821255978) + t * FT(1.330274429))
                )
            )
        )
    q = inv_sqrt2pi * exp(-z_abs * z_abs / 2) * poly
    return ifelse(z >= zero(FT), one(FT) - q, q)
end
