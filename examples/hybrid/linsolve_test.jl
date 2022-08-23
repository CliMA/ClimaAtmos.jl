
function verify_matrix(x, A, b, update_matrix = false; kwargs...)
    (; dtγ_ref, S, S_column_arrays) = A
    (; ∂ᶜρₜ∂ᶠ𝕄, ∂ᶜ𝔼ₜ∂ᶠ𝕄, ∂ᶠ𝕄ₜ∂ᶜ𝔼, ∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶠ𝕄ₜ∂ᶠ𝕄, ∂ᶜ𝕋ₜ∂ᶠ𝕄_named_tuple) = A
    dtγ = dtγ_ref[]
    dtγ² = dtγ^2
    FT = eltype(eltype(S))
    @assert A.test && Operators.bandwidths(eltype(∂ᶜ𝔼ₜ∂ᶠ𝕄)) == (-half, half)
    Ni, Nj, _, Nv, Nh = size(Fields.field_values(x.c))
    Nᶜf = DataLayouts.typesize(FT, eltype(x.c))
    J_col = zeros(FT, Nv * Nᶜf + Nv + 1, Nv * Nᶜf + Nv + 1)
    for h in 1:Nh, j in 1:Nj, i in 1:Ni
        x_col = Fields.FieldVector(;
            c = Spaces.column(x.c, i, j, h),
            f = Spaces.column(x.f, i, j, h),
        )
        b_col = Fields.FieldVector(;
            c = Spaces.column(b.c, i, j, h),
            f = Spaces.column(b.f, i, j, h),
        )
        ᶜρ_position = findfirst(isequal(:ρ), propertynames(x.c))
        ᶜρ_offset = DataLayouts.fieldtypeoffset(FT, eltype(x.c), ᶜρ_position)
        ᶜρ_indices = (Nv * ᶜρ_offset + 1):(Nv * (ᶜρ_offset + 1))
        ᶜ𝔼_position = findfirst(is_energy_var, propertynames(x.c))
        ᶜ𝔼_offset = DataLayouts.fieldtypeoffset(FT, eltype(x.c), ᶜ𝔼_position)
        ᶜ𝔼_indices = (Nv * ᶜ𝔼_offset + 1):(Nv * (ᶜ𝔼_offset + 1))
        ᶠ𝕄_indices = (Nv * Nᶜf + 1):(Nv * (Nᶜf + 1) + 1)
        J_col[ᶜρ_indices, ᶠ𝕄_indices] .=
            matrix_column(∂ᶜρₜ∂ᶠ𝕄, axes(x.f), i, j, h)
        J_col[ᶜ𝔼_indices, ᶠ𝕄_indices] .=
            matrix_column(∂ᶜ𝔼ₜ∂ᶠ𝕄, axes(x.f), i, j, h)
        J_col[ᶠ𝕄_indices, ᶜρ_indices] .=
            matrix_column(∂ᶠ𝕄ₜ∂ᶜρ, axes(x.c), i, j, h)
        J_col[ᶠ𝕄_indices, ᶜ𝔼_indices] .=
            matrix_column(∂ᶠ𝕄ₜ∂ᶜ𝔼, axes(x.c), i, j, h)
        J_col[ᶠ𝕄_indices, ᶠ𝕄_indices] .=
            matrix_column(∂ᶠ𝕄ₜ∂ᶠ𝕄, axes(x.f), i, j, h)
        for ᶜ𝕋_position in findall(is_tracer_var, propertynames(x.c))
            ᶜ𝕋_offset =
                DataLayouts.fieldtypeoffset(FT, eltype(x.c), ᶜ𝕋_position)
            ᶜ𝕋_indices = (Nv * ᶜ𝕋_offset + 1):(Nv * (ᶜ𝕋_offset + 1))
            ᶜ𝕋_name = propertynames(x.c)[ᶜ𝕋_position]
            ∂ᶜ𝕋ₜ∂ᶠ𝕄 = getproperty(∂ᶜ𝕋ₜ∂ᶠ𝕄_named_tuple, ᶜ𝕋_name)
            J_col[ᶜ𝕋_indices, ᶠ𝕄_indices] .=
                matrix_column(∂ᶜ𝕋ₜ∂ᶠ𝕄, axes(x.f), i, j, h)
        end
        @assert (-LinearAlgebra.I + dtγ * J_col) * x_col ≈ b_col
    end
end
