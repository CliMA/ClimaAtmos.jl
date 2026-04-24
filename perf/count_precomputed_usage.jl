"""
    count_precomputed_usage.jl

Static analysis script to count how many times each precomputed quantity field
(held in `p.precomputed`) is accessed across the source code, and to estimate
the performance impact of recomputing each field on-the-fly instead of caching.

Usage:
    julia perf/count_precomputed_usage.jl [src_dir]

where `src_dir` defaults to `src/` relative to the repository root.

Output:
    A table of fields sorted by usage count, with:
    - field name
    - total number of access patterns in source files
    - number of distinct files that access the field
    - estimated computation cost category
    - estimated relative performance ratio (cost of recomputing / cost of caching)

Performance estimation methodology:
    Each field is categorized by how expensive it is to compute. When a field
    is cached (computed once), the cost is 1× that of computing it. If the field
    were instead recomputed every time it is used (N times), the cost would be
    N× that of computing it. The "performance ratio" reported is N (the usage
    count), which represents the factor by which computation would increase if
    the field were removed from the cache and recomputed on the fly.

    Fields that are expensive to compute AND used many times have the highest
    recomputation cost, making them the best candidates to keep in the cache.
    Fields that are cheap to compute OR used rarely may be candidates for
    on-the-fly computation.

Computation cost categories (for individual field computations):
    :trivial  – O(1) field access / copy
    :cheap    – simple arithmetic, vector/tensor operations
    :moderate – thermodynamic function calls (non-iterative)
    :expensive – iterative solver (e.g., saturation adjustment)
    :external – computed by external subsystem (surface conditions, radiation)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helper: walk a directory and return all *.jl files
# ─────────────────────────────────────────────────────────────────────────────
function find_julia_files(dir)
    jl_files = String[]
    for (root, _, files) in walkdir(dir)
        for file in files
            endswith(file, ".jl") && push!(jl_files, joinpath(root, file))
        end
    end
    return jl_files
end

# ─────────────────────────────────────────────────────────────────────────────
# Field name extraction
# ─────────────────────────────────────────────────────────────────────────────
"""
    extract_identifiers(s)

Extract all Julia identifiers from a string. Julia identifiers may contain
Unicode letters (including decorators like ᶜ, ᶠ, ʲ, ⁰, ∂, subscripts, etc.),
ASCII letters/digits, and underscores.
"""
function extract_identifiers(s::AbstractString)
    # Collect maximal sequences of characters that can appear in Julia identifiers:
    #   word characters (\w), Unicode modifier letters (ʲ, etc.), superscripts,
    #   subscripts, and ∂.
    # We build a character-level regex rather than a code-unit regex to handle
    # arbitrary multi-byte Unicode correctly.
    ids = String[]
    buf = Char[]
    for c in s
        if is_julia_ident_char(c)
            push!(buf, c)
        else
            if !isempty(buf)
                push!(ids, String(buf))
                empty!(buf)
            end
        end
    end
    isempty(buf) || push!(ids, String(buf))
    return ids
end

"""
    is_julia_ident_char(c)

Return true when `c` may appear inside a Julia identifier.
This includes ASCII word characters, Unicode letters, combining marks,
modifier letters, superscripts, subscripts, and a handful of math symbols
that Julia accepts in identifiers (notably ∂ and prime ′).
"""
function is_julia_ident_char(c::Char)
    c == '_' && return true
    isletter(c) && return true
    isdigit(c) && return true
    # Superscripts: U+00B2 (²), U+00B3 (³), U+00B9 (¹), U+2070–U+209F
    '\u00B2' <= c <= '\u00B3' && return true
    c == '\u00B9' && return true
    '\u2070' <= c <= '\u209F' && return true
    # Subscripts U+2080–U+209F already covered above
    # Modifier letters: U+02B0–U+02FF  (includes ʲ U+02B2)
    '\u02B0' <= c <= '\u02FF' && return true
    # Phonetic extensions: U+1D00–U+1DFF (includes ᶜ U+1D9C, ᶠ U+1DA0, ᵣ U+1D63)
    '\u1D00' <= c <= '\u1DFF' && return true
    # Mathematical operators: ∂ (U+2202)
    c == '∂' && return true
    # Prime characters: ′ (U+2032), ″ (U+2033), etc.
    '\u2032' <= c <= '\u2037' && return true
    # Greek letters and Coptic: U+0370–U+03FF
    '\u0370' <= c <= '\u03FF' && return true
    return false
end

# ─────────────────────────────────────────────────────────────────────────────
# Cost category definitions
# ─────────────────────────────────────────────────────────────────────────────

"""
    computation_cost(field_name)

Return a symbol describing the estimated computation cost to produce `field_name`,
together with a short human-readable explanation.

Cost symbols and their meaning:
  :trivial  – O(1) memory read (field stored verbatim, no arithmetic)
  :cheap    – simple element-wise arithmetic or interpolation (a few FLOPs/point)
  :moderate – single non-iterative thermodynamic function call (dozens FLOPs/point)
  :expensive – iterative solver or saturation adjustment (hundreds–thousands FLOPs/point)
  :external – computed by an external subsystem (surface scheme, radiation, etc.)
  :unknown  – cost not estimated for this field
"""
function computation_cost(field::AbstractString)
    # ── Velocity / kinematic quantities (cheap: interpolation + arithmetic) ───
    field in ("ᶜu", "ᶠu³", "ᶠu", "ᶜK",
              "ᶜu⁰", "ᶠu³⁰", "ᶠu₃⁰",
              "ᶜuʲs", "ᶠu³ʲs", "ᶜKʲs", "ᶠKᵥʲs",
              "ᶠρ_diffʲs", "ᶜK⁰") && return :cheap, "velocity/kinetic energy (interpolation + arithmetic)"

    # ── Moisture quantities, non-iterative (moderate) ────────────────────────
    field in ("ᶜq_tot_nonneg", "ᶜq_liq", "ᶜq_ice",
              "ᶜq_tot_nonneg⁰", "ᶜq_liq⁰", "ᶜq_ice⁰",
              "ᶜq_tot_nonnegʲs", "ᶜq_liqʲs", "ᶜq_iceʲs") &&
        return :moderate, "moisture fractions (clamped specific humidity)"

    # ── Temperature (expensive for EquilibriumMicrophysics0M because of ────
    #    saturation adjustment; moderate for dry / NonEquilibrium) ────────────
    field in ("ᶜT", "ᶜT⁰", "ᶜTʲs") &&
        return :expensive, "temperature (saturation adjustment or air_temperature call)"

    # ── Thermodynamic derived quantities (moderate) ──────────────────────────
    field in ("ᶜp", "ᶜh_tot") &&
        return :moderate, "thermodynamic derived quantity (air_pressure / total_enthalpy)"

    # ── Saturation adjustment cache (expensive – same as ᶜT) ─────────────────
    field == "ᶜsa_result" &&
        return :expensive, "saturation adjustment result (iterative solver)"

    # ── Density of SGS subdomains (moderate) ────────────────────────────────
    field in ("ᶜρʲs", "ᶜρaʲs") &&
        return :moderate, "SGS subdomain density (thermodynamic + mass fraction)"

    # ── SGS moisture-energy ──────────────────────────────────────────────────
    field in ("ᶜmseʲs", "ᶜq_totʲs", "q_totʲs") &&
        return :moderate, "SGS moist static energy / total water"

    # ── Entrainment / detrainment (expensive: closure integrals) ─────────────
    field in ("ᶜentrʲs", "ᶜdetrʲs", "ᶜturb_entrʲs") &&
        return :expensive, "entrainment/detrainment (closure computation)"

    # ── Turbulence / buoyancy diagnostics (moderate) ─────────────────────────
    field in ("ᶜlinear_buoygrad", "ᶜstrain_rate_norm") &&
        return :moderate, "turbulence diagnostics (strain rate / buoyancy gradient)"

    # ── SGS quadrature covariances (moderate) ────────────────────────────────
    field in ("ᶜT′T′", "ᶜq′q′") &&
        return :moderate, "SGS temperature/moisture variance"

    # ── Diagnostic precipitation SGS (moderate) ──────────────────────────────
    field in ("ᶜq_lclʲs", "ᶜq_iclʲs", "ᶜq_raiʲs", "ᶜq_snoʲs") &&
        return :moderate, "SGS diagnostic precipitation condensate fractions"

    # ── Cloud fraction (moderate) ─────────────────────────────────────────────
    field == "ᶜcloud_fraction" &&
        return :moderate, "cloud fraction (quadrature or simple threshold)"

    # ── Gradient fields (cheap: vertical difference operator) ─────────────────
    field in ("ᶜgradᵥ_q_tot", "ᶜgradᵥ_θ_liq_ice") &&
        return :cheap, "vertical gradient (finite difference)"

    # ── Precipitation velocities (moderate) ───────────────────────────────────
    field in ("ᶜwₗ", "ᶜwᵢ", "ᶜwᵣ", "ᶜwₛ",
              "ᶜwₙₗ", "ᶜwₙᵣ", "ᶜwnᵢ",
              "ᶜwₗʲs", "ᶜwᵢʲs", "ᶜwᵣʲs", "ᶜwₛʲs",
              "ᶜwₙₗʲs", "ᶜwₙᵣʲs",
              "ᶜwₜqₜ", "ᶜwₕhₜ") &&
        return :moderate, "precipitation/sedimentation terminal velocity"

    # ── Microphysics tendency / derivative caches (expensive) ─────────────────
    field in ("ᶜmp_tendency", "ᶜmp_tendencyʲs", "ᶜmp_tendency⁰",
              "ᶜmp_derivative", "ᶜmp_derivativeʲs",
              "ᶜρ_dq_tot_dt", "ᶜρ_de_tot_dt",
              "ᶜ∂tendency_∂q_tot", "ᶜ∂tendency_∂q_totʲs",
              "ᶜScoll", "ᶜlogλ") &&
        return :expensive, "microphysics tendency (cloud/precip formation rates)"

    # ── Surface conditions (external – computed by SurfaceConditions) ─────────
    field in ("sfc_conditions", "ustar", "obukhov_length",
              "buoyancy_flux", "ρ_flux_h_tot", "ρ_flux_q_tot") &&
        return :external, "surface conditions (Monin-Obukhov / bulk flux scheme)"

    # ── Surface precipitation fluxes (expensive: column integrals) ───────────
    field in ("surface_rain_flux", "surface_snow_flux",
              "col_integrated_precip_energy_tendency") &&
        return :expensive, "surface precipitation flux (column integral)"

    # ── TKE-related fluxes (moderate) ────────────────────────────────────────
    field in ("ρtke_flux",) &&
        return :moderate, "TKE flux"

    # ── Smagorinsky-Lilly / AMD tensors (expensive) ──────────────────────────
    field in ("ᶜS", "ᶠS", "ᶜS_norm_h", "ᶜS_norm_v",
              "ᶜL_h", "ᶜL_v", "ᶜνₜ_h", "ᶜνₜ_v", "ᶜD_h", "ᶜD_v",
              "ᶜτ_amd", "ᶠτ_amd", "ᶜD_amd", "ᶠD_amd") &&
        return :expensive, "Smagorinsky-Lilly / AMD SGS tensor"

    # ── Non-hydrostatic pressure (moderate) ───────────────────────────────────
    field in ("ᶠnh_pressure³_buoyʲs", "ᶠnh_pressure³_dragʲs") &&
        return :moderate, "SGS non-hydrostatic pressure"

    # ── 2M microphysics extras (expensive) ───────────────────────────────────
    field in ("ᶜwₙₗ", "ᶜwₙᵣ", "ᶜwnₗ", "ᶜwnᵣ", "ᶜwnᵢ") &&
        return :expensive, "2-moment microphysics number concentration velocity"

    return :unknown, "cost not estimated"
end

const COST_RANK = Dict(
    :trivial   => 1,
    :cheap     => 2,
    :moderate  => 3,
    :expensive => 4,
    :external  => 4,
    :unknown   => 0,
)

const COST_LABEL = Dict(
    :trivial   => "trivial  ",
    :cheap     => "cheap    ",
    :moderate  => "moderate ",
    :expensive => "expensive",
    :external  => "external ",
    :unknown   => "unknown  ",
)

# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────
function count_precomputed_usage(src_dir::AbstractString)
    jl_files = find_julia_files(src_dir)
    isempty(jl_files) && error("No Julia files found in $src_dir")

    # field => total occurrence count (across all files)
    usage_count = Dict{String,Int}()
    # field => number of distinct files
    file_count  = Dict{String,Int}()

    for filepath in jl_files
        content = read(filepath, String)

        # Remove triple-quoted strings (docstrings) and single-line comments to
        # avoid false positives from code examples in documentation.
        # We process them iteratively to handle multiple docstrings per file.
        # Step 1: Remove triple-quoted strings (non-greedy to handle multiple docstrings).
        clean = replace(content, r"\"\"\".*?\"\"\""s => "")
        # Step 2: Remove single-line comments (only outside string literals,
        # which is sufficient for our purposes since we already stripped docstrings).
        clean = replace(clean, r"#[^\n]*" => "")

        fields_in_file = Set{String}()

        # ── Pattern 1: direct field access: p.precomputed.FIELD or cache.precomputed.FIELD ──
        for m in eachmatch(r"(?:p|cache)\.precomputed\.([^\s,;(){}\[\].]+)", clean)
            field = String(m.captures[1])
            # strip any trailing punctuation that crept in
            field = rstrip(field, (')', ']', ',', ';', '.', '\n'))
            isempty(field) && continue
            usage_count[field] = get(usage_count, field, 0) + 1
            push!(fields_in_file, field)
        end

        # ── Pattern 2: destructuring: (; f1, f2, ...) = p.precomputed or = cache.precomputed ──
        # The semicolon group can span multiple lines.
        # We require that the RHS is exactly p.precomputed (or cache.precomputed),
        # not a sub-field like p.precomputed.sfc_conditions.
        # Note: Julia's top-level destructuring (; f1, f2) never uses nested tuples
        # for named-tuple fields, so the [^)]* pattern is sufficient.
        for m in eachmatch(r"\(\s*;([^)]*)\)\s*=\s*(?:p|cache)\.precomputed\b(?!\.)"s, clean)
            group = m.captures[1]
            # Remove nested comments that might survive the first pass
            group = replace(group, r"#[^\n]*" => "")
            # Split on common delimiters: comma, whitespace, newline
            for field in extract_identifiers(group)
                isempty(field) && continue
                # Reject pure ASCII digit tokens (not valid Julia identifiers);
                # use ASCII-only check to avoid rejecting names with subscript digits
                # like u₃ (where '₃' is U+2083, detected as a digit by isdigit).
                all(c -> isascii(c) && isdigit(c), field) && continue
                usage_count[field] = get(usage_count, field, 0) + 1
                push!(fields_in_file, field)
            end
        end

        for field in fields_in_file
            file_count[field] = get(file_count, field, 0) + 1
        end
    end

    return usage_count, file_count
end

# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

"""
    performance_ratio_label(n_uses, cost_sym)

Given the number of times a field is used (`n_uses`) and its computation cost
category, return a string describing the estimated performance ratio if the
field were recomputed on-the-fly instead of being cached.

The ratio is defined as:
    recompute_cost / cache_cost  =  n_uses  (for the same computation)

A ratio of 1 means no change; a ratio > 1 means recomputing is more expensive.
For expensive fields the label carries an additional warning.
"""
function performance_ratio_label(n_uses::Int, cost_sym::Symbol)
    ratio = n_uses   # recomputing N times vs. caching once ⟹ N× cost
    warning = cost_sym in (:expensive, :external) ? " ⚠" : ""
    return "$(ratio)×$(warning)"
end

function print_report(usage_count, file_count)
    # Determine the set of real field names (exclude obvious parser artifacts)
    # A "real" field must appear at least once in the actual precomputed NamedTuple
    # We keep any field that starts with a valid identifier character and has
    # length ≥ 2 (to exclude lone letters from multiline regex noise).
    fields = String[]
    for (f, _) in usage_count
        length(f) < 2 && continue
        # Discard tokens that look like partial identifiers from multiline splits
        # (they would contain no ᶜ/ᶠ prefix and be ≤ 3 chars with only ASCII)
        is_ascii_short = all(isascii, f) && length(f) <= 2
        is_ascii_short && continue
        # Discard tokens ending with a backtick (artifact from docstring code)
        endswith(f, '`') && continue
        push!(fields, f)
    end

    # Sort by: usage count (desc), then cost rank (desc), then name
    sort!(fields; by = f -> begin
        c, _ = computation_cost(f)
        (-usage_count[f], -get(COST_RANK, c, 0), f)
    end)

    # Column widths
    col_field  = max(45, maximum(length, fields; init = 45))
    col_uses   = 8
    col_files  = 8
    col_cost   = 11
    col_ratio  = 14

    hdr = lpad("Uses", col_uses) *
          lpad("Files", col_files) *
          "  " * rpad("Cost", col_cost) *
          rpad("Perf. ratio (recompute/cache)", col_ratio)
    rule = "─" ^ (col_field + length(hdr) + 2)

    println()
    println("Precomputed quantity usage analysis")
    println("=" ^ length(rule))
    println(
        "Each row shows a field of `p.precomputed`, how often it appears in source",
    )
    println(
        "code, and an estimate of how much more expensive it would be to recompute",
    )
    println("it on-the-fly instead of using the cached value.")
    println()
    println(
        "Performance ratio = (number of usages) × (relative compute cost).",
    )
    println(
        "A high ratio means the field is a strong candidate to keep cached.",
    )
    println()
    println(rpad("Field", col_field + 2) * hdr)
    println(rule)

    for f in fields
        n_uses  = usage_count[f]
        n_files = get(file_count, f, 0)
        cost_sym, _ = computation_cost(f)
        cost_lbl = get(COST_LABEL, cost_sym, "unknown  ")
        ratio_lbl = performance_ratio_label(n_uses, cost_sym)

        println(
            rpad(f, col_field + 2) *
            lpad(string(n_uses), col_uses) *
            lpad(string(n_files), col_files) *
            "  " * rpad(cost_lbl, col_cost) *
            ratio_lbl,
        )
    end

    println(rule)
    println()

    # ── Summary: top candidates to keep in cache ──────────────────────────────
    println("Summary: top fields by estimated recomputation cost")
    println("  (best candidates to keep in the precomputed cache)")
    println()
    println("  By usage count alone:")
    top_by_use = first(fields, min(5, length(fields)))
    for f in top_by_use
        c, desc = computation_cost(f)
        println("    • $f — $(usage_count[f]) usages, cost: $c ($desc)")
    end
    println()
    println("  Expensive/external fields used ≥ 5 times:")
    for f in fields
        c, desc = computation_cost(f)
        if c in (:expensive, :external) && usage_count[f] >= 5
            println("    • $f — $(usage_count[f]) usages, cost: $c ($desc)")
        end
    end
    println()
    println("  Cheap fields used ≤ 3 times (candidates to recompute on-the-fly):")
    cheap_rare = [f for f in fields if begin
        c, _ = computation_cost(f)
        c in (:trivial, :cheap) && usage_count[f] <= 3
    end]
    if isempty(cheap_rare)
        println("    (none found)")
    else
        for f in cheap_rare
            c, _ = computation_cost(f)
            println("    • $f — $(usage_count[f]) usages, cost: $c")
        end
    end

    println()
    println("Cost categories:")
    println("  trivial   – O(1) memory access / copy")
    println("  cheap     – simple arithmetic or interpolation")
    println("  moderate  – non-iterative thermodynamic function call")
    println("  expensive – iterative solver (e.g. saturation adjustment) or microphysics rates")
    println("  external  – computed by an external subsystem (surface scheme, etc.)")
    println()
    println("⚠  = high-cost field; recomputing N times would be particularly expensive.")
    println()
    println("Note: 'Uses' counts the number of `p.precomputed` (or `cache.precomputed`)")
    println("access patterns in source code (both destructuring and direct access).")
    println("It is a static upper bound on the number of times the field is read per")
    println("time step (some accesses may be conditional on the model configuration).")
end

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
let
    repo_root = joinpath(@__DIR__, "..")
    src_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(repo_root, "src")

    @info "Scanning Julia source files in: $src_dir"
    usage_count, file_count = count_precomputed_usage(src_dir)
    print_report(usage_count, file_count)
end
