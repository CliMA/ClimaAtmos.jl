#####
##### Guard: every bare `[`X`](@ref)` in a docstring must point at a symbol that
##### some page actually renders.
#####
##### Documenter resolves `@ref` only against docstrings spliced into a page by a
##### `@docs` block, and `checkdocs = :exports` leaves most docstrings unrendered
##### (the package documents ~1100 bindings; the pages render a curated subset).
##### A `@ref` to an unrendered target is therefore silent until the *referring*
##### docstring is added to a page, at which point the build fails with a
##### `:cross_references` error in what looks like an unrelated pull request.
##### This check turns that latent failure into an immediate one.
#####
##### Resolution mirrors Documenter: each reference is resolved in the module the
##### docstring belongs to, not in `ClimaAtmos`, so a docstring inside a
##### submodule may refer to its own names unqualified.
#####
##### To satisfy the check, either add the target to a `@docs` block, or write it
##### as plain code (`` `X` ``) instead of a link.

import Documenter
import Documenter.DocSystem as DS

"""
    rendered_bindings(src_dir)

Return the set of bindings that `@docs` blocks under `src_dir` splice into pages,
resolved the way Documenter's `@docs` expander resolves them.
"""
function rendered_bindings(src_dir)
    bindings = Set{Any}()
    for file in readdir(src_dir; join = true)
        endswith(file, ".md") || continue
        for block in eachmatch(r"```@docs\n(.*?)```"s, read(file, String))
            for line in split(block.captures[1], "\n")
                entry = strip(line)
                (isempty(entry) || startswith(entry, "#")) && continue
                try
                    push!(bindings, DS.binding(Main, Meta.parse(entry)))
                catch
                    # Unparseable entries are Documenter's own error to report.
                end
            end
        end
    end
    return bindings
end

"""
    submodules(root)

Return `root` together with every module defined inside it, recursively.
"""
function submodules(root, found = Set{Module}())
    push!(found, root)
    for name in names(root; all = true)
        isdefined(root, name) || continue
        value = getfield(root, name)
        if isa(value, Module) &&
           value !== root &&
           parentmodule(value) === root &&
           !(value in found)
            submodules(value, found)
        end
    end
    return found
end

"""
    unresolvable_docstring_refs(root_module, src_dir)

Return a vector of `(file, line, target)` for every bare `@ref` in a docstring of
`root_module` (or one of its submodules) whose target no page renders.
"""
function unresolvable_docstring_refs(root_module, src_dir)
    rendered = rendered_bindings(src_dir)
    offenders = Tuple{String, Int, String}[]
    for mod in submodules(root_module)
        meta = try
            Base.Docs.meta(mod)
        catch
            continue
        end
        for (_, multidoc) in meta, (_, docstr) in multidoc.docs
            file = string(get(docstr.data, :path, "?"))
            line = get(docstr.data, :linenumber, 0)
            for ref in eachmatch(r"\[`([^`]+)`\]\(@ref\)", join(docstr.text))
                target = ref.captures[1]
                expr = try
                    Meta.parse(target)
                catch
                    continue
                end
                binding = try
                    DS.binding(mod, expr)
                catch
                    nothing
                end
                if binding === nothing || !(binding in rendered)
                    push!(offenders, (file, line, target))
                end
            end
        end
    end
    return sort(offenders)
end

"""
    check_docstring_refs(root_module, src_dir)

Throw if any docstring cross-reference is unresolvable; otherwise report the
number of references checked.
"""
function check_docstring_refs(root_module, src_dir)
    offenders = unresolvable_docstring_refs(root_module, src_dir)
    if !isempty(offenders)
        message = IOBuffer()
        println(
            message,
            "$(length(offenders)) docstring cross-reference(s) point at symbols " *
            "that no documentation page renders:",
        )
        for (file, line, target) in offenders
            println(message, "  ", file, ":", line, "  [`", target, "`](@ref)")
        end
        println(
            message,
            "\nAdd each target to a `@docs` block, or write it as `` `X` `` " *
            "instead of a link.\nSee docs/check_docstring_refs.jl for why this " *
            "is checked here.",
        )
        error(String(take!(message)))
    end
    return nothing
end
