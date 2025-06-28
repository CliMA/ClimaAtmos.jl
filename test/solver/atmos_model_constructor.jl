using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.RRTMGPInterface as RRTMGPI
using Dates

const FT = Float32

@testset "AtmosModel Constructor Tests" begin
    
    @testset "Flat Interface - Documentation Examples" begin
        
        @testset "Basic dry model example" begin
            # Test the first example from documentation
            model = CA.AtmosModel(;
                moisture_model = CA.DryModel(),
                surface_model = CA.PrescribedSurfaceTemperature(),
                precip_model = CA.NoPrecipitation()
            )
            
            @test model.moisture_model isa CA.DryModel
            @test model.surface_model isa CA.PrescribedSurfaceTemperature  
            @test model.precip_model isa CA.NoPrecipitation
            # Note: flat interface doesn't provide defaults, so these will be nothing
            @test model.sfc_temperature === nothing
            @test model.insolation === nothing
            @test model.disable_surface_flux_tendency == false  # this is a direct field
        end
        
        @testset "Moist model with radiation example" begin
            # Test the second example from documentation - fix RRTMGPI constructor
            model = CA.AtmosModel(;
                moisture_model = CA.EquilMoistModel(),
                precip_model = CA.Microphysics0Moment(),
                radiation_mode = RRTMGPI.AllSkyRadiation(
                    false, # idealized_h2o
                    false, # idealized_clouds  
                    CA.InteractiveCloudInRadiation(), # cloud
                    false, # add_isothermal_boundary_layer
                    false, # aerosol_radiation
                    false, # reset_rng_seed
                    false  # deep_atmosphere
                ),
                ozone = CA.IdealizedOzone(),
                co2 = CA.FixedCO2()
            )
            
            @test model.moisture_model isa CA.EquilMoistModel
            @test model.precip_model isa CA.Microphysics0Moment
            @test model.radiation_mode isa RRTMGPI.AllSkyRadiation
            @test model.ozone isa CA.IdealizedOzone
            @test model.co2 isa CA.FixedCO2
        end
        
        @testset "Model with hyperdiffusion and sponge layers example" begin
            # Test the third example from documentation
            model = CA.AtmosModel(;
                moisture_model = CA.NonEquilMoistModel(),
                precip_model = CA.Microphysics1Moment(),
                hyperdiff = CA.ClimaHyperdiffusion(; 
                    ν₄_vorticity_coeff = 1e15, 
                    ν₄_scalar_coeff = 1e15, 
                    divergence_damping_factor = 1.0
                ),
                rayleigh_sponge = CA.RayleighSponge(; zd = 12000.0, α_uₕ = 4.0, α_w = 2.0),
                disable_surface_flux_tendency = false
            )
            
            @test model.moisture_model isa CA.NonEquilMoistModel
            @test model.precip_model isa CA.Microphysics1Moment
            @test model.hyperdiff isa CA.ClimaHyperdiffusion
            @test model.hyperdiff.ν₄_vorticity_coeff == 1e15
            @test model.hyperdiff.ν₄_scalar_coeff == 1e15
            @test model.hyperdiff.divergence_damping_factor == 1.0
            @test model.rayleigh_sponge isa CA.RayleighSponge
            @test model.rayleigh_sponge.zd == 12000.0
            @test model.rayleigh_sponge.α_uₕ == 4.0
            @test model.rayleigh_sponge.α_w == 2.0
            @test model.disable_surface_flux_tendency == false
        end
    end
    
    @testset "Individual Parameter Tests" begin
        
        @testset "Moisture models" begin
            dry_model = CA.AtmosModel(; moisture_model = CA.DryModel())
            @test dry_model.moisture_model isa CA.DryModel
            
            equil_model = CA.AtmosModel(; moisture_model = CA.EquilMoistModel())
            @test equil_model.moisture_model isa CA.EquilMoistModel
            
            nonequil_model = CA.AtmosModel(; moisture_model = CA.NonEquilMoistModel())
            @test nonequil_model.moisture_model isa CA.NonEquilMoistModel
        end
        
        @testset "Precipitation models" begin
            no_precip = CA.AtmosModel(; precip_model = CA.NoPrecipitation())
            @test no_precip.precip_model isa CA.NoPrecipitation
            
            micro0 = CA.AtmosModel(; precip_model = CA.Microphysics0Moment())
            @test micro0.precip_model isa CA.Microphysics0Moment
            
            micro1 = CA.AtmosModel(; precip_model = CA.Microphysics1Moment())
            @test micro1.precip_model isa CA.Microphysics1Moment
            
            micro2 = CA.AtmosModel(; precip_model = CA.Microphysics2Moment())
            @test micro2.precip_model isa CA.Microphysics2Moment
        end
        
        @testset "Cloud models" begin
            grid_cloud = CA.AtmosModel(; cloud_model = CA.GridScaleCloud())
            @test grid_cloud.cloud_model isa CA.GridScaleCloud
            
            quad_cloud = CA.AtmosModel(; cloud_model = CA.QuadratureCloud(CA.SGSQuadrature(FT)))
            @test quad_cloud.cloud_model isa CA.QuadratureCloud
            
            sgs_cloud = CA.AtmosModel(; cloud_model = CA.SGSQuadratureCloud(CA.SGSQuadrature(FT)))
            @test sgs_cloud.cloud_model isa CA.SGSQuadratureCloud
        end
        
        @testset "Radiation modes" begin
            clear_sky = CA.AtmosModel(; radiation_mode = RRTMGPI.ClearSkyRadiation(false, false, false, false))
            @test clear_sky.radiation_mode isa RRTMGPI.ClearSkyRadiation
            
            all_sky = CA.AtmosModel(; radiation_mode = RRTMGPI.AllSkyRadiation(
                false, false, CA.InteractiveCloudInRadiation(), false, false, false, false
            ))
            @test all_sky.radiation_mode isa RRTMGPI.AllSkyRadiation
        end
        
        @testset "Surface models" begin
            prescribed_temp = CA.AtmosModel(; surface_model = CA.PrescribedSurfaceTemperature())
            @test prescribed_temp.surface_model isa CA.PrescribedSurfaceTemperature
            
            prognostic_temp = CA.AtmosModel(; surface_model = CA.PrognosticSurfaceTemperature{FT}())
            @test prognostic_temp.surface_model isa CA.PrognosticSurfaceTemperature
        end
    end
    
    @testset "Error Handling" begin
        
        @testset "Invalid parameter name" begin
            @test_throws ErrorException CA.AtmosModel(;
                moisture_model = CA.DryModel(),
                invalid_parameter = "invalid"
            )
        end
        
        @testset "Helpful error message" begin
            try
                CA.AtmosModel(; invalid_param = "test")
            catch e
                @test e isa ErrorException
                @test occursin("Unknown AtmosModel parameter: invalid_param", e.msg)
                @test occursin("Available parameters:", e.msg)
                @test occursin("moisture_model", e.msg)  # Check some known valid params are listed
                @test occursin("surface_model", e.msg)
            end
        end
    end
    
    @testset "Interface Compatibility" begin
        model = CA.AtmosModel(;
            moisture_model = CA.NonEquilMoistModel(),
            precip_model = CA.Microphysics1Moment(),
            forcing_type = CA.HeldSuarezForcing(),
            rayleigh_sponge = CA.RayleighSponge(; zd = 10000.0, α_uₕ = 3.0, α_w = 2.0)
        )
        
        # Test that property access works through the compatibility layer
        @test model.moisture_model isa CA.NonEquilMoistModel
        @test model.precip_model isa CA.Microphysics1Moment
        @test model.forcing_type isa CA.HeldSuarezForcing
        @test model.rayleigh_sponge isa CA.RayleighSponge
        
        # Test that grouped properties also work
        @test model.moisture isa CA.AtmosMoistureModel
        @test model.forcing isa CA.AtmosForcing
        @test model.sponge isa CA.AtmosSponge
        
        # Test that nested access works
        @test model.moisture.moisture_model isa CA.NonEquilMoistModel
        @test model.forcing.forcing_type isa CA.HeldSuarezForcing
        @test model.sponge.rayleigh_sponge isa CA.RayleighSponge
    end
    
    @testset "Default Values in Flat Interface" begin
        # Test that the flat interface sets unspecified params to nothing (no defaults)
        minimal_model = CA.AtmosModel(; moisture_model = CA.DryModel())
        
        @test minimal_model.moisture_model isa CA.DryModel
        @test minimal_model.precip_model === nothing  # flat interface default 
        @test minimal_model.cloud_model === nothing  # flat interface default
        @test minimal_model.surface_model === nothing  # flat interface default
        @test minimal_model.forcing_type === nothing  # flat interface default
        @test minimal_model.hyperdiff === nothing  # flat interface default
        @test minimal_model.numerics === nothing  # flat interface default
        @test minimal_model.disable_surface_flux_tendency == false  # explicit default
    end
    
    @testset "Broadcasting Compatibility" begin
        # Test that AtmosModel and its components are broadcastable
        model = CA.AtmosModel(; moisture_model = CA.DryModel())
        
        @test Base.broadcastable(model) == tuple(model)
        @test Base.broadcastable(model.moisture) == tuple(model.moisture)
        @test Base.broadcastable(model.moisture_model) == tuple(model.moisture_model)
    end
end 