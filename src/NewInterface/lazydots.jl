bcexpr(f, args...) = Expr(:call, :(Base.broadcasted), f, args...)

hasdot(x) = false
hasdot(x::Symbol) = Base.isoperator(x) && first(string(x)) == '.' && x !== :..

undot(x::Symbol) = Symbol(string(x)[2:end])

lazydots(x) = x
function lazydots(x::Expr)
    head = x.head
    lazyargs = Base.mapany(lazydots, x.args)
    if head === :. && lazyargs[2] isa Expr
        bcexpr(lazyargs[1], lazyargs[2].args...)
    elseif head === :call && hasdot(lazyargs[1])
        bcexpr(undot(lazyargs[1]), lazyargs[2:end]...)
    elseif head === Symbol(".&&") # TODO: temporary fix for outdated formatter
        bcexpr(:(Base.andand), lazyargs...)
    elseif head === Symbol(".||") # TODO: temporary fix for outdated formatter
        bcexpr(:(Base.oror), lazyargs...)
    elseif head === :comparison && any(hasdot, lazyargs[2:2:end])
        if !all(hasdot, lazyargs[2:2:end])
            s_comp = join(lazyargs[2:2:end], ", ", " and ")
            s = string(
                "cannot lazily broadcast `$x`, since a combination of dotted",
                "and ordinary comparisons ($s_comp) must call `materialize`",
            ) # TODO: temporary fix for outdated formatter
            throw(ErrorException(s))
        end
        ith_bcexpr =
            i -> bcexpr(undot(lazyargs[i]), lazyargs[i - 1], lazyargs[i + 1])
        bcexpr(:&, map(ith_bcexpr, 2:2:length(lazyargs))...)
    elseif first(string(head)) == '.' && last(string(head)) == '='
        s = "cannot lazily broadcast `$x`, since $head must call `materialize!`"
        throw(ErrorException(s))
    else
        Expr(head, lazyargs...)
    end
end

"""
    @lazydots code

Turns all dotted operations in `code` into calls to `Base.broadcasted`.

Ideally, evaluating `code` and evaluating `Base.materialize(@lazydots code)`
will result in the same computation. However, this transformed code is not
guaranteed to evaluate in exactly the same way as the original. For example, it
might perform some computations multiple times, whereas the original code would
materialize those computations and then reference the materialized output. The
transformed code might also throw an error because one of its function calls
expects a materialized argument rather than a `Broadcasted` object. (The adjoint
`'` is an example of such a function.) Although this macro catches some
potential errors before they are encountered when the transformed code is run,
not all mistakes can be detected at the syntactic level. So, special care must
be taken when using this macro to ensure that `Base.materialize(@lazydots code)`
is identical to `code`.

```@repl
@lazydots @. [1, 2, 3] * [3 2 1] + [1 2 3]
Base.materialize(ans)
@lazydots (A = [1, 2 + 3] .+ 1; B = [true, false] .| false; @. ifelse(B, A, 3A))
Base.materialize(ans)
@lazydots @. [1, 2] < ([2, 1] + [0 3]) > [0 0] # double computation of 2nd term
Base.materialize(ans)
@lazydots @. [1, 2] + ([1 2] + [2 3])' # error due to applying ' to Broadcasted
```
"""
macro lazydots(code)
    esc(lazydots(macroexpand(Main, code)))
end
