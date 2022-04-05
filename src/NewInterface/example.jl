push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using Test
using ClimaCore: Fields

if !@isdefined Domains
    include("../Domains/Domains.jl") # don't define Domains twice
end
using ..Domains

include("vars.jl")
include("lazydots.jl")
include("formula_interface.jl")
include("boundary_conditions.jl")
include("tendency_interface.jl")
include("models.jl")
include("formula_functions.jl")
include("tendency_terms.jl")

const γ = 1.400 # heat capacity ratio of dry air
const R_d = 287.1 # specific gas constant of dry air (J/kg/K)
const p_0 = 1.000e5 # reference pressure (Pa)
const g = 9.806 # acceleration due to gravity (m/s^2)

function init_dry_rising_bubble_2d(coord, ::Var{(:c,)})
    x_c = 0.0
    z_c = 350.0
    r_c = 250.0
    θ_b = 300.0
    θ_c = 0.5
    cp_d = R_d * γ / (γ - 1)
    cv_d = R_d / (γ - 1)

    @unpack x, z = coord
    r = sqrt((x - x_c)^2 + (z - z_c)^2)

    # potential temperature perturbation
    θ_p = r < r_c ? 0.5 * θ_c * (1.0 + cospi(r / r_c)) : 0.0

    θ = θ_b + θ_p                  # potential temperature
    π_exn = 1.0 - g * z / cp_d / θ # exner function
    T = π_exn * θ                  # temperature
    p = p_0 * π_exn^(cp_d / R_d)   # pressure
    ρ = p / R_d / T                # density
    ρθ = ρ * θ                     # potential temperature density

    return (ρ = ρ, ρθ = ρθ)
end

function init_dry_rising_bubble_2d(coord, ::Var{(:f,)})
    return (ρw = Geometry.WVector(0.0),)
end

function main()
    zero_flux_bcs = VerticalBoundaryConditions(
        SetOperatorInputsAtBoundary(; total_face_flux = Geometry.WVector(0.0)),
        SetOperatorInputsAtBoundary(; total_face_flux = Geometry.WVector(0.0)),
    )
    zero_velocity_bcs = VerticalBoundaryConditions(
        SetTendencyAtBoundary(; tendency = Geometry.WVector(0.0)),
        SetTendencyAtBoundary(; tendency = Geometry.WVector(0.0)),
    )
    tendencies = (
        Tendency(Var(:c, :ρ), VerticalAdvection(Var(:c, :ρ))),
        Tendency(Var(:c, :ρθ), zero_flux_bcs, VerticalAdvection(Var(:c, :ρθ))),
        Tendency(
            Var(:f, :ρw),
            zero_velocity_bcs,
            VerticalAdvection(Var(:f, :ρw)),
            PressureGradient(Var(:f, :ρw)),
            Gravity(Var(:f, :ρw)),
        ),
    )
    diagnostics_formulas =
        (Formula(Var(:c, :ρe_tot), DefaultFluidFunction(Var(:c, :ρe_tot))),)
    model = Model(DefaultFluidFunction; tendencies, diagnostics_formulas)

    domain = HybridPlane(
        xlim = (-5e2, 5e2),
        zlim = (0.0, 1e3),
        nelements = (10, 50),
        npolynomial = 4,
    )
    center_coords, face_coords =
        Fields.coordinate_field.(make_function_space(domain))

    consts = (
        γ,
        R_d,
        p_0,
        c = (Φ = g .* center_coords.z,),
        f = (∇Φ = Ref(Geometry.WVector(g)),), # need Ref for Field broadcasts
    )

    Yc = map(coord -> init_dry_rising_bubble_2d(coord, Var(:c)), center_coords)
    Yf = map(coord -> init_dry_rising_bubble_2d(coord, Var(:f)), face_coords)
    Y = Fields.FieldVector{Float64}((c = Yc, f = Yf))
    # TODO: File an issue that FieldVector(c = Yc, f = Yf) is not type-stable.

    f = ode_function(instantiate(model, consts, Y, 0.0))

    return f, Y
end

@inferred main() # test for type stability

f, Y = main()
∂ₜY = similar(Y)
f(∂ₜY, Y, nothing, 0.0)

#=
Ideas for higher-level interface:

- Compressible vs. Incompressible (ρ ∈ Y vs. ρ ∈ consts)
- Conservative vs. Convective (w vs. ρw and uₕ vs. ρuₕ)
- Hydrostatic vs. Non-hydrostatic (w ∈ Y or ρw ∈ Y vs. w ∈ cache or ρw ∈ cache)
    - Do we actually want this functionality?
- Energy variable (ρθ vs. ρe_tot)
    - Do we want any other options?
- Number of horizontal dimensions (0 vs. 1 vs. 2)

default_dry_fluid_model(;
    is_compressible::Bool = true,
    is_conservative::Bool = true,
    is_hydrostatic::Bool = false,
    energy_var::Symbol = :ρe_tot,
    n_horizontal_dims::Int = 2,
) = Model(...)

default_moist_fluid_model(
    dry_fluid_model::Model;
    saturation_adjustment_scheme::Symbol = :default,
    kwargs...,
) = Model(...)

Such an interface is convenient for constructing default models, but it is
inconvenient for disabling/replacing/adding tendency terms, boundary conditions,
and formula functions. If we have such an interface, we should make it easy to
do the latter with functions like disable_tendency_term(var, tendency_term).
=#

# TODO: Use ∇⨉c for Operators.CurlC2F.

# TODO: Make finalizers (post-materialize in-place operations) for formulas and
# tendencies; e.g., Spaces.weighted_dss!.
# TODO: Merge cache variables with identical return spaces.
# TODO: Merge finalizers for merged cache variables and independent variables.

# TODO: Add conservation checks. If tendency_term is used for vars′ ⊂ vars, but
# there is some non-empty set vars′′ ⊂ vars∖vars′ for which
# cache_reqs(tendency_term, var, vars) is defined for all var ∈ vars′′, print
# the warning string(
#     "model includes $(typeof(tendency_term).name.wrapper) of ",
#     join(vars′, ", ", " and "), ", but not of ", join(vars′′, ", ", " and "),
# )

# # Join a collection of items using the serial (Oxford) comma.
# comma_join(items) = length(items) == 2 ? join(items, " and ") :
#     join(items, ", ", ", and ")
# comma_join(io::IO, items) = length(items) == 2 ? join(io, items, " and ") :
#     join(io, items, ", ", ", and ")
