using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: CommonSpaces, Fields, Grids, Spaces

column_space(FT, device) = CommonSpaces.ColumnSpace(FT;
    z_min = 0, z_max = 16400, z_elem = 82,
    device, staggering = Grids.CellCenter(),
)

# See #4712
device = ClimaComms.device()
@testset "TRMM_LBA radiation on $(nameof(typeof(device)))" begin
    @test isbits(CA.RadiationTRMM_LBA())

    @testset "$FT" for FT in (Float64, Float32)
        rad_profile = CA.Setups.APL.TRMM_LBA_radiation(FT)
        z_knots = CA.Setups.APL.TRMM_LBA_radiation_z(FT)
        @test eltype(z_knots) == FT
        space = column_space(FT, device)
        ᶜz = Fields.coordinate_field(space).z
        ᶜdTdt_rad = Fields.Field(FT, space)
        for t in (FT(0), FT(1000), FT(5000))
            CA.set_trmm_lba_dTdt_rad!(ᶜdTdt_rad, rad_profile, z_knots, t)
            @test parent(ᶜdTdt_rad) isa ClimaComms.array_type(device)
            dev_vals = Array(parent(ᶜdTdt_rad))
            @test all(isfinite, dev_vals)
            @test dev_vals ≈ rad_profile.(t, Array(parent(ᶜz)))
        end
    end
end
