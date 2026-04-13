"""
    surface_fractions_cache(Y, land_sea_mask_file)

Build a NamedTuple of static 2D surface fraction fields, allocated on the
bottom face (surface) space of `Y`.

Fields returned:
- `land_sea_mask`: land fraction at each surface point (0 = pure ocean,
  1 = pure land). Loaded from `land_sea_mask_file` if provided; defaults to
  all zeros (pure ocean) when the path is empty.

Additional surface fraction fields (e.g. `desert_fraction`) should be added
here as the relevant parameterizations are implemented.
"""
function surface_fractions_cache(Y, land_sea_mask_file)
    sfc_space = axes(Fields.level(Y.f, Fields.half))
    FT = Spaces.undertype(sfc_space)

    land_sea_mask = zeros(sfc_space)

    if !isempty(land_sea_mask_file)
        extrapolation_bc = (Intp.Periodic(), Intp.Flat())
        tvi = TimeVaryingInput(
            land_sea_mask_file,
            "land_fraction",
            sfc_space;
            regridder_type = :InterpolationsRegridder,
            regridder_kwargs = (; extrapolation_bc),
            method = LinearInterpolation(),
        )
        # Static field: evaluate once at t=0 and store.
        TimeVaryingInputs.evaluate!(land_sea_mask, tvi, FT(0))
    end

    return (; land_sea_mask)
end
