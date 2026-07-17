using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Setups as Setups
import ClimaCore: CommonSpaces, Fields, Grids, Meshes, Spaces

center_column_space(FT, device) = CommonSpaces.ColumnSpace(
    FT;
    z_min = 0, z_max = 15000, z_elem = 32,
    device, staggering = Grids.CellCenter(),
)

center_box_space(FT, device) = CommonSpaces.Box3DSpace(
    FT;
    x_min = 0, x_max = 1000, x_elem = 2,
    y_min = 0, y_max = 1000, y_elem = 2,
    z_min = 0, z_max = 15000, z_elem = 16,
    periodic_x = true, periodic_y = true,
    n_quad_points = 3,
    stretch = Meshes.GeneralizedExponentialStretching(FT(300), FT(2000)),
    device, staggering = Grids.CellCenter(),
)

function monotone_decreasing_in_z(field)
    nz = Spaces.nlevels(axes(field))
    for level in 1:(nz - 1)
        below = Array(parent(Fields.level(field, level)))
        above = Array(parent(Fields.level(field, level + 1)))
        all(below .> above) || return false
    end
    return true
end

# Evaluate the hydrostatic pressure profile of each AtmosphericProfilesLibrary
# setup on the active device through `initial_condition_field`, on a uniform
# single column and on a vertically stretched multi-column box. See PR #4664.
# Reference pressure values are checked in test/larcform1.jl.
device = ClimaComms.device()
@testset "Hydrostatic pressure profiles on $(nameof(typeof(device)))" begin
    @testset "$FT" for FT in (Float64, Float32)
        params = CA.ClimaAtmosParameters(FT)
        thermo_params = CA.Parameters.thermodynamics_params(params)
        larcform1 = Setups.Larcform1(; prognostic_tke = true, thermo_params)
        shipwayhill = Setups.ShipwayHill2012(; thermo_params)
        trmm_lba = Setups.TRMM_LBA(; prognostic_tke = true, thermo_params)
        @test shipwayhill isa Setups.ShipwayHill2012

        spaces = (center_column_space(FT, device), center_box_space(FT, device))
        for space in spaces, setup in (larcform1, shipwayhill, trmm_lba)
            ic_p(lg) = Setups.center_initial_condition(setup, lg, params).p
            ᶜp = Setups.initial_condition_field(ic_p, space)
            @test ᶜp isa Fields.Field
            @test parent(ᶜp) isa ClimaComms.array_type(device)
            p_dev = Array(parent(ᶜp))
            @test all(isfinite, p_dev)
            @test all(>(0), p_dev)
            @test monotone_decreasing_in_z(ᶜp)
            # The device-resident field equals the host interpolant evaluated at
            # the field's own heights.
            z = Array(parent(Fields.coordinate_field(space).z))
            @test setup.profiles.p.(z) == p_dev
        end
    end
end
