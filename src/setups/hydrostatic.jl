# Taken from InitialConditions
import ClimaComms
import ClimaCore.Domains as Domains
import ClimaCore.Meshes as Meshes
import ClimaCore.Operators as Operators
import ClimaCore.Topologies as Topologies
import ClimaCore.Spaces as Spaces
import Interpolations as Intp
import SciMLBase

const FunctionOrSpline =
    Union{Function, APL.AbstractProfile, Intp.Extrapolation}

"""
    ColumnInterpolatableField(::Fields.ColumnField)

A column field object that can be interpolated
in the z-coordinate. For example:

!!! warn
    This function allocates and is not GPU-compatible
    so please avoid using this inside `step!` only use
    this for initialization.
"""
struct ColumnInterpolatableField{F, D}
    f::F
    data::D
    function ColumnInterpolatableField(f::Fields.ColumnField)
        zdata = vec(parent(Fields.Fields.coordinate_field(f).z))
        fdata = vec(parent(f))
        data = Intp.extrapolate(
            Intp.interpolate((zdata,), fdata, Intp.Gridded(Intp.Linear())),
            Intp.Flat(),
        )
        return new{typeof(f), typeof(data)}(f, data)
    end
end
(f::ColumnInterpolatableField)(z) = Spaces.undertype(axes(f.f))(f.data(z))

"""
    column_indefinite_integral(f, ϕ₀, zspan; nelems = 100)

The column integral, returned as an interpolate-able field.
"""
function column_indefinite_integral(
    f::Function,
    ϕ₀::FT,
    zspan::Tuple{FT, FT};
    nelems = 100, # sets resolution for integration
) where {FT <: Real}
    # --- Make a space for integration:
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(first(zspan)),
        Geometry.ZPoint(last(zspan));
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain; nelems)
    context = ClimaComms.SingletonCommsContext()
    z_topology = Topologies.IntervalTopology(context, z_mesh)
    cspace = Spaces.CenterFiniteDifferenceSpace(z_topology)
    fspace = Spaces.FaceFiniteDifferenceSpace(z_topology)
    # ---
    zc = Fields.coordinate_field(cspace)
    ᶠintegral = Fields.Field(FT, fspace)
    Operators.column_integral_indefinite!(f, ᶠintegral, ϕ₀)
    return ColumnInterpolatableField(ᶠintegral)
end

"""
    hydrostatic_pressure_profile(; thermo_params, p_0, [T, θ, q_tot, z_max])

Solves the initial value problem `p'(z) = -g * ρ(z)` for all `z ∈ [0, z_max]`,
given `p(0)`, either `T(z)` or `θ(z)`, and optionally also `q_tot(z)`. If
`q_tot(z)` is not given, it is assumed to be 0. If `z_max` is not given, it is
assumed to be 30 km. Note that `z_max` should be the maximum elevation to which
the specified profiles T(z), θ(z), and/or q_tot(z) are valid.
"""
function hydrostatic_pressure_profile(;
    thermo_params,
    p_0,
    T = nothing,
    θ = nothing,
    q_tot = nothing,
    z_max = 30000,
)
    FT = eltype(thermo_params)
    grav = TD.Parameters.grav(thermo_params)

    # Compute air density from (p, z) using either T(z) or θ(z), with optional q_tot(z)
    function ρ_from_profile(p, z, ::Nothing, ::Nothing, _)
        error("Either T or θ must be specified")
    end
    function ρ_from_profile(p, z, T::FunctionOrSpline, θ::FunctionOrSpline, _)
        error("Only one of T and θ can be specified")
    end
    function ρ_from_profile(p, z, T::FunctionOrSpline, ::Nothing, ::Nothing)
        TD.air_density(thermo_params, oftype(p, T(z)), p)
    end
    function ρ_from_profile(p, z, ::Nothing, θ::FunctionOrSpline, ::Nothing)
        T_val = TD.air_temperature(thermo_params, TD.pθ_li(), p, oftype(p, θ(z)))
        TD.air_density(thermo_params, T_val, p)
    end
    function ρ_from_profile(p, z, T::FunctionOrSpline, ::Nothing, q_tot::FunctionOrSpline)
        TD.air_density(thermo_params, oftype(p, T(z)), p, oftype(p, q_tot(z)), FT(0), FT(0))
    end
    function ρ_from_profile(p, z, ::Nothing, θ::FunctionOrSpline, q_tot::FunctionOrSpline)
        q = oftype(p, q_tot(z))
        T_val = TD.air_temperature(thermo_params, TD.pθ_li(), p, oftype(p, θ(z)), q)
        TD.air_density(thermo_params, T_val, p, q, FT(0), FT(0))
    end
    dp_dz(p, z) = -grav * ρ_from_profile(p, z, T, θ, q_tot)

    return column_indefinite_integral(dp_dz, p_0, (FT(0), FT(z_max)))
end
