if !haskey(ENV, "BUILDKITE")
    import Pkg
    Pkg.develop(Pkg.PackageSpec(; path = dirname(@__DIR__)))
end
using Test

using LinearAlgebra

using ClimaCore: Geometry, Spaces, Fields
using ClimaAtmos: Domains
using ClimaAtmos: Models
using ClimaAtmos.Models.Nonhydrostatic2DModels: Nonhydrostatic2DModel
using ClimaAtmos.Models.Nonhydrostatic3DModels: Nonhydrostatic3DModel
using ClimaAtmos.Models.SingleColumnModels: SingleColumnModel

float_types = (Float32, Float64)

@testset "Models: Styles" begin
    for FT in float_types
        # auxiliary data
        styles = (
            Models.AdvectiveForm(),
            Models.ConservativeForm(),
            Models.Dry(),
            Models.EquilibriumMoisture(),
            Models.NonEquilibriumMoisture(),
        )
        names = (
            (:ρ, :uh, :w),
            (:ρ, :ρuh, :ρw),
            nothing,
            (:ρq_tot,),
            (:ρq_tot, :ρq_liq, :ρq_ice),
            (:ρθ,),
            (:ρe_tot,),
        )
        types_nh2d = (
            (ρ = FT, uh = Geometry.UVector{FT}, w = Geometry.WVector{FT}),
            (ρ = FT, ρuh = Geometry.UVector{FT}, ρw = Geometry.WVector{FT}),
            Nothing,
            (ρq_tot = FT,),
            (ρq_tot = FT, ρq_liq = FT, ρq_ice = FT),
            (ρθ = FT,),
            (ρe_tot = FT,),
        )
        types_nh3d = (
            (
                ρ = FT,
                uh = Geometry.Covariant12Vector{FT},
                w = Geometry.Covariant3Vector{FT},
            ),
            (
                ρ = FT,
                ρuh = Geometry.Covariant12Vector{FT},
                ρw = Geometry.Covariant3Vector{FT},
            ),
            Nothing,
            (ρq_tot = FT,),
            (ρq_tot = FT, ρq_liq = FT, ρq_ice = FT),
            (ρθ = FT,),
            (ρe_tot = FT,),
        )
        spaces = (
            (
                ρ = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},
                uh = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},
                w = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellFace},
            ),
            (
                ρ = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},
                ρuh = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},
                ρw = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellFace},
            ),
            Nothing,
            (ρq_tot = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},),
            (
                ρq_tot =
                    Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},
                ρq_liq =
                    Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},
                ρq_ice =
                    Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},
            ),
            (ρθ = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},),
            (ρe_tot = Spaces.ExtrudedFiniteDifferenceSpace{Spaces.CellCenter},),
        )

        # test variable_names
        for (s, n) in zip(styles, names)
            @test Models.variable_names(s) == n
        end

        # test variable_types
        domain = Domains.HybridPlane(
            FT,
            xlim = (-5e2, 5e2),
            zlim = (0.0, 1e3),
            nelements = (10, 10),
            npolynomial = 3,
        )
        model = Nonhydrostatic2DModel(
            domain = domain,
            boundary_conditions = nothing,
            parameters = (),
        )
        for (s, t) in zip(styles, types_nh2d)
            @test Models.variable_types(s, model, FT) == t
        end

        # test variable_spaces
        for (s, sp) in zip(styles, spaces)
            @test Models.variable_spaces(s, model) == sp
        end
    end
end

@testset "Models: Nonhydrostatic2DModels" begin
    for FT in float_types
        # auxiliary structs
        domain = Domains.HybridPlane(
            FT,
            xlim = (-5e2, 5e2),
            zlim = (0.0, 1e3),
            nelements = (10, 10),
            npolynomial = 3,
        )
        # test constructors
        model = Nonhydrostatic2DModel(
            domain = domain,
            boundary_conditions = nothing,
            parameters = (),
        )
        @test model.domain isa Domains.AbstractHybridDomain
        @test model.moisture == Models.Dry()
        @test model.thermodynamics == Models.PotentialTemperature()

        model = Nonhydrostatic2DModel(
            domain = domain,
            base = Models.AdvectiveForm(),
            thermodynamics = Models.TotalEnergy(),
            moisture = Models.NonEquilibriumMoisture(),
            boundary_conditions = nothing,
            parameters = (),
        )
        @test model.domain isa Domains.AbstractHybridDomain
        @test model.base == Models.AdvectiveForm()
        @test model.moisture == Models.NonEquilibriumMoisture()
        @test model.thermodynamics == Models.TotalEnergy()

        # test components
        model = Nonhydrostatic2DModel(
            domain = domain,
            thermodynamics = Models.TotalEnergy(),
            moisture = Models.Dry(),
            boundary_conditions = nothing,
            parameters = (),
        )
        @test keys(Models.components(model)) ==
              (:base, :thermodynamics, :moisture)

        # test variable_names 
        model = Nonhydrostatic2DModel(
            domain = domain,
            thermodynamics = Models.TotalEnergy(),
            moisture = Models.NonEquilibriumMoisture(),
            boundary_conditions = nothing,
            parameters = (),
        )
        @test Models.variable_names(model).base ==
              Models.variable_names(model.base)
        @test Models.variable_names(model).thermodynamics ==
              Models.variable_names(model.thermodynamics)
        @test Models.variable_names(model).moisture ==
              Models.variable_names(model.moisture)

        # test default_initial_conditions
        model = Nonhydrostatic2DModel(
            domain = domain,
            boundary_conditions = nothing,
            parameters = (),
        )
        Y = Models.default_initial_conditions(model)
        @test Y isa Fields.FieldVector
        @test Y.base isa Fields.FieldVector
        @test Y.thermodynamics isa Fields.FieldVector
        @test Y.base.ρ isa Fields.Field
        @test Y.base.ρuh isa Fields.Field
        @test Y.base.ρw isa Fields.Field
        @test Y.thermodynamics.ρθ isa Fields.Field
        @test norm(Y.base.ρ) == 0
        @test norm(Y.base.ρuh) == 0
        @test norm(Y.base.ρw) == 0
        @test norm(Y.thermodynamics.ρθ) == 0
    end
end

@testset "Models: Nonhydrostatic3DModels" begin
    for FT in float_types
        # auxiliary structs
        domain = Domains.HybridBox(
            FT,
            xlim = (-5e2, 5e2),
            ylim = (-5e2, 5e2),
            zlim = (0.0, 1e3),
            nelements = (10, 10, 10),
            npolynomial = 3,
        )

        # test constructors
        model = Nonhydrostatic3DModel(
            domain = domain,
            boundary_conditions = nothing,
            parameters = (),
            hyperdiffusivity = FT(100),
        )
        @test model.domain isa Domains.AbstractHybridDomain
        @test model.moisture == Models.Dry()
        @test model.thermodynamics == Models.TotalEnergy()
        @test model.hyperdiffusivity == FT(100)

        model = Nonhydrostatic3DModel(
            domain = domain,
            base = Models.AdvectiveForm(),
            thermodynamics = Models.TotalEnergy(),
            moisture = Models.NonEquilibriumMoisture(),
            hyperdiffusivity = FT(100),
            boundary_conditions = nothing,
            parameters = (),
        )
        @test model.domain isa Domains.AbstractHybridDomain
        @test model.base == Models.AdvectiveForm()
        @test model.moisture == Models.NonEquilibriumMoisture()
        @test model.thermodynamics == Models.TotalEnergy()

        # test components
        model = Nonhydrostatic3DModel(
            domain = domain,
            thermodynamics = Models.TotalEnergy(),
            moisture = Models.Dry(),
            hyperdiffusivity = FT(100),
            boundary_conditions = nothing,
            parameters = (),
        )
        @test keys(Models.components(model)) ==
              (:base, :thermodynamics, :moisture)

        # test variable_names, variable_types, variable_spaces 
        model = Nonhydrostatic3DModel(
            domain = domain,
            thermodynamics = Models.TotalEnergy(),
            moisture = Models.NonEquilibriumMoisture(),
            hyperdiffusivity = FT(100),
            boundary_conditions = nothing,
            parameters = (),
        )
        @test Models.variable_names(model).base ==
              Models.variable_names(model.base)
        @test Models.variable_names(model).thermodynamics ==
              Models.variable_names(model.thermodynamics)
        @test Models.variable_names(model).moisture ==
              Models.variable_names(model.moisture)

        # test default_initial_conditions
        model = Nonhydrostatic3DModel(
            domain = domain,
            hyperdiffusivity = FT(100),
            boundary_conditions = nothing,
            parameters = (),
        )
        Y = Models.default_initial_conditions(model)
        @test Y isa Fields.FieldVector
        @test Y.base isa Fields.FieldVector
        @test Y.thermodynamics isa Fields.FieldVector
        @test Y.base.ρ isa Fields.Field
        @test Y.base.uh isa Fields.Field
        @test Y.base.w isa Fields.Field
        @test Y.thermodynamics.ρe_tot isa Fields.Field
        @test norm(Y.base.ρ) == 0
        @test norm(Y.base.uh) == 0
        @test norm(Y.base.w) == 0
        @test norm(Y.thermodynamics.ρe_tot) == 0
    end
end

@testset "Models: SingleColumnModels" begin
    for FT in float_types
        # auxiliary structs
        domain = Domains.Column(FT, zlim = (0.0, 3e2), nelements = 15)

        # test constructors
        model = SingleColumnModel(
            domain = domain,
            boundary_conditions = nothing,
            parameters = (),
        )
        @test model.domain isa Domains.AbstractVerticalDomain
        @test model.base == Models.AdvectiveForm()
        @test model.moisture == Models.Dry()
        @test model.thermodynamics == Models.TotalEnergy()
        @test model.turbconv isa Nothing

        model = SingleColumnModel(
            domain = domain,
            thermodynamics = Models.PotentialTemperature(),
            moisture = Models.EquilibriumMoisture(),
            turbconv = Models.ConstantViscosity(),
            boundary_conditions = nothing,
            parameters = (),
        )
        @test model.moisture == Models.EquilibriumMoisture()
        @test model.thermodynamics == Models.PotentialTemperature()
        @test model.turbconv == Models.ConstantViscosity()

        # test components
        model = SingleColumnModel(
            domain = domain,
            thermodynamics = Models.TotalEnergy(),
            moisture = Models.Dry(),
            turbconv = Models.ConstantViscosity(),
            boundary_conditions = nothing,
            parameters = (),
        )
        @test keys(Models.components(model)) ==
              (:base, :thermodynamics, :moisture, :turbconv)

        # test variable_names, variable_types, variable_spaces
        model = SingleColumnModel(
            domain = domain,
            thermodynamics = Models.TotalEnergy(),
            moisture = Models.Dry(),
            turbconv = Models.ConstantViscosity(),
            boundary_conditions = nothing,
            parameters = (),
        )
        @test Models.variable_names(model).base ==
              Models.variable_names(model.base)
        @test Models.variable_names(model).thermodynamics ==
              Models.variable_names(model.thermodynamics)
        @test Models.variable_names(model).moisture ==
              Models.variable_names(model.moisture)
        @test Models.variable_names(model).turbconv ==
              Models.variable_names(model.turbconv)

        # test default_initial_conditions
        model = SingleColumnModel(
            domain = domain,
            thermodynamics = Models.TotalEnergy(),
            moisture = Models.Dry(),
            turbconv = Models.ConstantViscosity(),
            boundary_conditions = nothing,
            parameters = (),
        )
        Y = Models.default_initial_conditions(model)
        @test Y isa Fields.FieldVector
        @test Y.base isa Fields.FieldVector
        @test Y.thermodynamics isa Fields.FieldVector
        @test Y.base.ρ isa Fields.Field
        @test Y.base.uh isa Fields.Field
        @test Y.base.w isa Fields.Field
        @test Y.thermodynamics.ρe_tot isa Fields.Field
        @test norm(Y.base.ρ) == 0
        @test norm(Y.base.uh) == 0
        @test norm(Y.base.w) == 0
        @test norm(Y.thermodynamics.ρe_tot) == 0
    end
end
