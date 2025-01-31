# TODO: use https://github.com/CliMA/NullBroadcasts.jl when released

"""
    NullBroadcasted()

A `Base.AbstractBroadcasted` that represents arithmetic object.

An `NullBroadcasted()` can be added to, subtracted from, or multiplied by any
value in a broadcast expression without incurring a runtime performance
penalty.

For example, the following rules hold when broadcasting instances of
`NullBroadcasted`:

```
1 + NullBroadcasted() == 1
NullBroadcasted() + 1 == 1
1 - NullBroadcasted() == 1
1 * NullBroadcasted() == NullBroadcasted()
1 / NullBroadcasted() == NullBroadcasted()
 ```
"""
struct NullBroadcasted <: Base.AbstractBroadcasted end
Base.broadcastable(x::NullBroadcasted) = x

struct NullBroadcastedStyle <: Base.BroadcastStyle end
Base.BroadcastStyle(::Type{<:NullBroadcasted}) = NullBroadcasted()

# Specialize on AbstractArrayStyle to avoid ambiguities with AbstractBroadcasted.
Base.BroadcastStyle(::NullBroadcasted, ::Base.Broadcast.AbstractArrayStyle) =
    NullBroadcasted()
Base.BroadcastStyle(::Base.Broadcast.AbstractArrayStyle, ::NullBroadcasted) =
    NullBroadcasted()

# Add another method to avoid ambiguity between the previous two.
Base.BroadcastStyle(::NullBroadcasted, ::NullBroadcasted) = NullBroadcasted()

broadcasted_sum(args) =
    if isempty(args)
        NullBroadcasted()
    elseif length(args) == 1
        args[1]
    else
        Base.broadcasted(+, args...)
    end
Base.broadcasted(::NullBroadcasted, ::typeof(+), args...) =
    broadcasted_sum(filter(arg -> !(arg isa NullBroadcasted), args))

Base.broadcasted(op::typeof(-), ::NullBroadcasted, arg) =
    Base.broadcasted(op, arg)
Base.broadcasted(op::typeof(-), arg, ::NullBroadcasted) =
    Base.broadcasted(Base.identity, arg)
Base.broadcasted(op::typeof(-), a::NullBroadcasted) = NullBroadcasted()
Base.broadcasted(op::typeof(-), a::NullBroadcasted, ::NullBroadcasted) =
    Base.broadcasted(op, a)

Base.broadcasted(op::typeof(+), ::NullBroadcasted, args...) =
    Base.broadcasted(op, args...)
Base.broadcasted(op::typeof(+), arg, ::NullBroadcasted, args...) =
    Base.broadcasted(op, arg, args...)
Base.broadcasted(
    op::typeof(+),
    a::NullBroadcasted,
    ::NullBroadcasted,
    args...,
) = Base.broadcasted(op, a, args...)

Base.broadcasted(op::typeof(*), ::NullBroadcasted, args...) = NullBroadcasted()
Base.broadcasted(op::typeof(*), arg, ::NullBroadcasted) = NullBroadcasted()
Base.broadcasted(op::typeof(*), ::NullBroadcasted, ::NullBroadcasted) =
    NullBroadcasted()
Base.broadcasted(op::typeof(/), ::NullBroadcasted, args...) = NullBroadcasted()
Base.broadcasted(op::typeof(/), arg, ::NullBroadcasted) = NullBroadcasted()
Base.broadcasted(op::typeof(/), ::NullBroadcasted, ::NullBroadcasted) =
    NullBroadcasted()

Base.broadcasted(op::typeof(identity), a::NullBroadcasted) = a

function skip_materialize(dest, bc::Base.Broadcast.Broadcasted)
    if typeof(bc.f) <: typeof(+) || typeof(bc.f) <: typeof(-)
        if length(bc.args) == 2 &&
           bc.args[1] === dest &&
           bc.args[2] === Base.Broadcast.Broadcasted(NullBroadcasted, ())
            return true
        else
            return false
        end
    else
        return false
    end
end

Base.Broadcast.instantiate(
    bc::Base.Broadcast.Broadcasted{NullBroadcastedStyle},
) = x

Base.Broadcast.materialize!(dest, x::NullBroadcasted) =
    error("NullBroadcasted objects cannot be materialized.")
Base.Broadcast.materialize(dest, x::NullBroadcasted) =
    error("NullBroadcasted objects cannot be materialized.")
