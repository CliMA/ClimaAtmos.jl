module COSPPrecipSubcolumns

import ClimaCore: Fields, Quadratures, Spaces

export prec_scops!

"""
    prec_scops!(prec_frac, large_scale_precipitation_flux, frac_out)

Generate precipitation subcolumn masks from large-scale precipitation fluxes and
cloud subcolumn masks. Convective precipitation flux is assumed to be zero.

Inputs

  - `large_scale_precipitation_flux`
  - `frac_out`: cloud subcolumn masks from `COSPSubcolumns.scops!`

Outputs

  - writes precipitation masks in-place into `prec_frac`
  - `0`: no precipitation
  - `1`: large-scale precipitation
"""
function prec_scops!(
    prec_frac::NTuple{N},
    large_scale_precipitation_flux::Fields.Field,
    frac_out::NTuple{N},
) where {N}
    N > 0 || throw(ArgumentError("prec_frac must contain at least one subcolumn"))

    _check_field_axes(prec_frac, large_scale_precipitation_flux, "prec_frac")
    _check_field_axes(frac_out, large_scale_precipitation_flux, "frac_out")

    _prec_scops_fields!(prec_frac, large_scale_precipitation_flux, frac_out)

    return nothing
end

function _prec_scops_fields!(
    prec_frac,
    large_scale_precipitation_flux,
    frac_out,
)
    nlev = Spaces.nlevels(axes(large_scale_precipitation_flux))
    nsubcolumns = length(prec_frac)

    for column_index in _column_indices(large_scale_precipitation_flux)
        prec_frac_columns =
            ntuple(i -> Fields.column(prec_frac[i], column_index...), nsubcolumns)
        frac_columns =
            ntuple(i -> Fields.column(frac_out[i], column_index...), nsubcolumns)
        precipitation_column =
            Fields.column(large_scale_precipitation_flux, column_index...)

        _prec_scops_field_column!(
            prec_frac_columns,
            precipitation_column,
            frac_columns,
            nlev,
        )
    end

    return nothing
end

function _prec_scops_field_column!(
    prec_frac,
    large_scale_precipitation_flux,
    frac_out,
    nlev,
)
    nsubcolumns = length(prec_frac)
    no_precip = zero(eltype(prec_frac[1]))
    large_scale_precip = one(eltype(prec_frac[1]))
    large_scale_cloud = one(eltype(frac_out[1]))

    @inbounds for isubcolumn in 1:nsubcolumns
        _fill_column!(prec_frac[isubcolumn], no_precip, nlev)
    end

    has_large_scale_cloud =
        ntuple(
            isubcolumn -> _has_large_scale_cloud(
                frac_out[isubcolumn],
                large_scale_cloud,
                nlev,
            ),
            nsubcolumns,
        )

    @inbounds for ilev in nlev:-1:1
        _level_value(large_scale_precipitation_flux, ilev) > 0 || continue

        found_precip_column = false

        for isubcolumn in 1:nsubcolumns
            current_cloud =
                _level_value(frac_out[isubcolumn], ilev) == large_scale_cloud
            precip_from_above =
                ilev < nlev &&
                _level_value(prec_frac[isubcolumn], ilev + 1) == large_scale_precip

            if current_cloud || precip_from_above
                _set_level_value!(
                    prec_frac[isubcolumn],
                    ilev,
                    large_scale_precip,
                )
                found_precip_column = true
            end
        end

        if !found_precip_column && ilev > 1
            for isubcolumn in 1:nsubcolumns
                if _level_value(frac_out[isubcolumn], ilev - 1) ==
                   large_scale_cloud
                    _set_level_value!(
                        prec_frac[isubcolumn],
                        ilev,
                        large_scale_precip,
                    )
                    found_precip_column = true
                end
            end
        end

        if !found_precip_column
            for isubcolumn in 1:nsubcolumns
                if has_large_scale_cloud[isubcolumn]
                    _set_level_value!(
                        prec_frac[isubcolumn],
                        ilev,
                        large_scale_precip,
                    )
                    found_precip_column = true
                end
            end
        end

        if !found_precip_column
            for isubcolumn in 1:nsubcolumns
                _set_level_value!(
                    prec_frac[isubcolumn],
                    ilev,
                    large_scale_precip,
                )
            end
        end
    end

    return nothing
end

function _has_large_scale_cloud(frac_out, large_scale_cloud, nlev)
    @inbounds for ilev in 1:nlev
        _level_value(frac_out, ilev) == large_scale_cloud && return true
    end
    return false
end

function _fill_column!(field, value, nlev)
    @inbounds for ilev in 1:nlev
        _set_level_value!(field, ilev, value)
    end
    return nothing
end

function _column_indices(field::Fields.Field)
    space = axes(field)
    space isa Spaces.FiniteDifferenceSpace && return ((1, 1, 1),)

    horizontal_space = Spaces.horizontal_space(space)
    quadrature_points =
        1:Quadratures.degrees_of_freedom(
            Spaces.quadrature_style(horizontal_space),
        )
    slab_indices = Spaces.eachslabindex(horizontal_space)

    return horizontal_space isa Spaces.SpectralElementSpace1D ?
           Iterators.product(quadrature_points, slab_indices) :
           Iterators.product(quadrature_points, quadrature_points, slab_indices)
end

@inline _level_value(field, ilev) = Fields.level(field, ilev)[]
@inline _set_level_value!(field, ilev, value) = (Fields.level(field, ilev)[] = value)

function _check_field_axes(fields, reference, name)
    for field in fields
        axes(field) == axes(reference) ||
            throw(
                DimensionMismatch(
                    "$name field must have the same axes as large_scale_precipitation_flux",
                ),
            )
    end
end

end
