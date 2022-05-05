
function Base.show(io::IO, ::Type{T}) where {T <: ClimaCore.Fields.Field}
    values_type(::Type{T}) where {V, T <: ClimaCore.Fields.Field{V}} = V
    V = values_type(T)

    _apply!(f, ::T, match_list) where {T} = nothing # sometimes we need this...
    function _apply!(f, ::Type{T}, match_list) where {T}
        if f(T)
            push!(match_list, T)
        end
        for p in T.parameters
            _apply!(f, p, match_list)
        end
    end
    #=
        apply(::T) where {T <: Any}
    Recursively traverse type `T` and apply
    `f` to the types (and type parameters).
    Returns a list of matches where `f(T)` is true.
    =#
    apply(f, ::T) where {T} = apply(f, T)
    function apply(f, ::Type{T}) where {T}
        match_list = []
        _apply!(f, T, match_list)
        return match_list
    end

    nts = apply(x -> x <: NamedTuple, eltype(V))
    syms = unique(map(nt -> fieldnames(nt), nts))
    s = join(syms, ",")
    print(io, "Field{$s} (trunc disp)")
end
