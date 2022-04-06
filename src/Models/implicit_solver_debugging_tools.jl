get_var(obj, ::Tuple{}) = obj
get_var(obj, tup::Tuple) = get_var(getproperty(obj, tup[1]), Base.tail(tup))
function exact_column_jacobian_block(
    implicit_tendency!,
    Y,
    p,
    t,
    i,
    j,
    h,
    Yₜ_name,
    Y_name,
)
    T = eltype(Y)
    Y_var = get_var(Y, Y_name)
    Y_var_vert_space = Spaces.column(axes(Y_var), i, j, h)
    bot_level = Operators.left_idx(Y_var_vert_space)
    top_level = Operators.right_idx(Y_var_vert_space)
    partials = ntuple(_ -> zero(T), top_level - bot_level + 1)
    Yᴰ = Dual.(Y, partials...)
    Yᴰ_var = get_var(Yᴰ, Y_name)
    ith_ε(i) = Dual.(zero(T), Base.setindex(partials, one(T), i)...)
    set_level_εs!(level) =
        parent(Spaces.level(Yᴰ_var, level)) .+= ith_ε(level - bot_level + 1)
    foreach(set_level_εs!, bot_level:top_level)
    Yₜᴰ = similar(Yᴰ)
    implicit_tendency!(Yₜᴰ, Yᴰ, p, t)
    col = Spaces.column(get_var(Yₜᴰ, Yₜ_name), i, j, h)
    return vcat(map(dual -> [dual.partials.values...]', parent(col))...)
end

# Note: These only work for scalar stencils.
vector_column(arg, i, j, h) = parent(Spaces.column(arg, i, j, h))
function matrix_column(stencil, stencil_input_space, i, j, h)
    lbw, ubw = Operators.bandwidths(eltype(stencil))
    coefs_column = Spaces.column(stencil, i, j, h).coefs
    row_space = axes(coefs_column)
    lrow = Operators.left_idx(row_space)
    rrow = Operators.right_idx(row_space)
    num_rows = rrow - lrow + 1
    col_space = Spaces.column(stencil_input_space, i, j, h)
    lcol = Operators.left_idx(col_space)
    rcol = Operators.right_idx(col_space)
    num_cols = rcol - lcol + 1
    diag_key_value(diag) =
        (diag + lrow - lcol) => view(
            parent(getproperty(coefs_column, diag - lbw + 1)),
            (max(lrow, lcol - diag):min(rrow, rcol - diag)) .- (lrow - 1),
        )
    return spdiagm(num_rows, num_cols, map(diag_key_value, lbw:ubw)...)
end
