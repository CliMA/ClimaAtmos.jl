using ForwardDiff: Dual
using ClimaCore: Spaces, Fields, Operators, MatrixFields

# TODO: Turn this into a unit test for the implicit solver once the code is more
# modular.

function autodiff_wfact_block(Y, p, dtγ, t, Yₜ_name, Y_name, colidx)
    Y_field = MatrixFields.get_field(Y, Y_name)
    bot_level = Operators.left_idx(axes(Y_field))
    top_level = Operators.right_idx(axes(Y_field))
    partials = ntuple(_ -> 0, top_level - bot_level + 1)

    Yᴰ = Dual.(Y, partials...)
    Yᴰ_field = MatrixFields.get_field(Yᴰ, Y_name)
    ith_ε(i) = Dual.(0, Base.setindex(partials, 1, i)...)
    set_level_εs!(level) =
        parent(Spaces.level(Yᴰ_field, level)) .+= ith_ε(level - bot_level + 1)
    foreach(set_level_εs!, bot_level:top_level)

    (; atmos) = p
    convert_to_duals(fields) =
        Fields._values(similar(Fields.FieldVector(; fields...), eltype(Yᴰ)))
    dry_atmos_name_pairs = map(propertynames(atmos)) do name
        name => name == :moisture_model ? DryModel() : atmos.:($name)
    end
    dry_atmos = AtmosModel(; dry_atmos_name_pairs...)
    pᴰ = (;
        p...,
        convert_to_duals(temporary_quantities(atmos, axes(Y.c), axes(Y.f)))...,
        convert_to_duals(precomputed_quantities(Y, dry_atmos))...,
        atmos = dry_atmos,
        sfc_setup = nothing,
    )

    Yₜᴰ = similar(Yᴰ)
    implicit_tendency!(Yₜᴰ, Yᴰ, pᴰ, t)
    Yₜᴰ_field = MatrixFields.get_field(Yₜᴰ, Yₜ_name)
    ∂Yₜ∂Y_array_block =
        vcat(map(d -> [d.partials.values...]', parent(Yₜᴰ_field[colidx]))...)
    return Yₜ_name == Y_name ? dtγ * ∂Yₜ∂Y_array_block - I :
           dtγ * ∂Yₜ∂Y_array_block
end

function verify_wfact(A, Y, p, dtγ, t, colidx)
    for (Yₜ_name, Y_name) in keys(A.matrix)
        computed_block = map(
            x -> x[1],
            MatrixFields.column_field2array(A.matrix[Yₜ_name, Y_name][colidx]),
        )
        Yₜ_name == Y_name && computed_block == -I && continue
        reference_block =
            autodiff_wfact_block(Y, p, dtγ, t, Yₜ_name, Y_name, colidx)
        max_error = maximum(abs.(computed_block - reference_block))
        @info t, Yₜ_name, Y_name, maximum(reference_block), max_error
        # display(computed_block)
        # display(reference_block)
    end
end
