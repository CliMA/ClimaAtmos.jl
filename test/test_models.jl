using ClimaAtmos.Models
using ClimaAtmos.Models.Nonhydrostatic2DModels: Nonhydrostatic2DModel

@testset "Models" begin
    for FT in float_types
        # auxiliary structs
        domain = HybridPlane(
            FT,
            xlim = (-5e2, 5e2),
            zlim = (0.0, 1e3),
            nelements = (10, 10),
            npolynomial = 3,
        )

        # test constructors
        model = Nonhydrostatic2DModel(
            domain = domain,
            thermodynamics = PotentialTemperature(),
            moisture = Dry(),
            boundary_conditions = nothing,
            parameters = (),
        )
        @test model.domain isa AbstractHybridDomain
        @test model.moisture == Dry()
        @test model.thermodynamics == PotentialTemperature()

        # test state_variable_names
        model = Nonhydrostatic2DModel(
            domain = domain,
            thermodynamics = PotentialTemperature(),
            moisture = Dry(),
            boundary_conditions = nothing,
            parameters = (),
        )
        @test Models.state_variable_names(model).base == (:ρ, :ρuh, :ρw)
        @test Models.state_variable_names(model).thermodynamics == (:ρθ,)

        model = Nonhydrostatic2DModel(
            domain = domain,
            thermodynamics = TotalEnergy(),
            moisture = EquilibriumMoisture(),
            boundary_conditions = nothing,
            parameters = (),
        )
        @test Models.state_variable_names(model).base == (:ρ, :ρuh, :ρw)
        @test Models.state_variable_names(model).thermodynamics == (:ρe_tot,)
        @test Models.state_variable_names(model).moisture == (:ρq_tot,)

        model = Nonhydrostatic2DModel(
            domain = domain,
            thermodynamics = TotalEnergy(),
            moisture = NonEquilibriumMoisture(),
            boundary_conditions = nothing,
            parameters = (),
        )
        @test Models.state_variable_names(model).base == (:ρ, :ρuh, :ρw)
        @test Models.state_variable_names(model).thermodynamics == (:ρe_tot,)
        @test Models.state_variable_names(model).moisture ==
              (:ρq_tot, :ρq_liq, :ρq_ice)
    end
end
