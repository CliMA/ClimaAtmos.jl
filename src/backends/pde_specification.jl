# TODO: make this work
@scalars ρ ρe ϕ
@vectors ρu
γ = 1.4
g = 9.8
x, y, z = coordinates(grid)
ϕ .= g * (x^2 + y^2 + z^2)
p = (γ-1) * (ρe - (ρu' * ρu) / (2 * ρ) - ρ * ϕ)
u = ρu / ρ
@pde [
    ∂ᵗ(ρ)  = -∇⋅(ρu),
    ∂ᵗ(ρu) = -∇⋅(ρu ⊗ u + p * I),
    ∂ᵗ(ρe) = -∇⋅(u * (ρe + p)),
]
