"""
    GPUPerfModel

Instruction-level performance model of CUDA kernels on one streaming
multiprocessor (SM), fed with LLVM IR (device-free, e.g. from
[`device_ir`](@ref)) or SASS text. Package-free (only `device_ir` needs CUDA).

    using .GPUPerfModel
    K = parse_llvm(device_ir(kernel, Tuple{CuDeviceVector{Float32, 1}}))  # or parse_sass
    p = simulate(K; device = :sm_80, threads = 256, blocks = 4096, regs = 32,
                 hits = (l1 = 0.0, l2 = 0.5, dram = 0.5), trips = Dict("L25" => 63))
    p.time_us, p.binding, p.stalls            # `display(p)` prints the report

# Model
- All resident warps run the same dynamic block trace (uniform control flow),
  except that a branch of the thread index against run-time sizes outside loops
  (a thread-range split, e.g. vertex threads, then face threads) sends whole blocks
  down each side in proportion (`Kernel.splits`, 8 block traces), a lane test
  that splits the lanes of every warp runs both sides (SIMT divergence; they
  reconverge at the immediate post-dominator, `Kernel.merges`), and a lane test
  that sends the warps of a block different ways (e.g. lane 0 of each group of
  64 threads) gives each warp its own trace (`Kernel.wsplit`: its side, or both
  sides where its lanes split). On a divergent side with at most half the
  lanes active, only those lanes move data (`Kernel.lanes`).
- A call of a device function that LLVM did not inline (a separate `define`; in
  SASS a `CALL` of a subroutine or of another function in the text) is stepped
  into: the call site runs the callee's ops, memory traffic and loops, with its
  parameters bound to the call's arguments (`Kernel.calls`; the copies share
  their code in the instruction caches, `Kernel.alias`; recursive functions and
  nesting deeper than `CALL_DEPTH` stay one branch op, noted).
  A loop's header runs `trips` times (by header label, else inferred from IR
  induction variables, else `default_trip`, which `loops = :few | :short | :long`
  sets to 4 | 16 | 64 source iterations (an unrolled loop's header runs that over its
  unroll factor, a remainder loop below the factor); `loops` also caps an inferred
  bound that the loop can leave early on computed data); a loop tested at its top
  runs its body one time fewer); a loop-carried value is ready
  when the previous iteration produced it, so a serial recurrence (and a load
  issued in order behind it) costs its latency per trip; other branches follow `conds`
  (P(true) per branch predicate), `probs`, `!prof` weights, 0 for edges into
  `unreachable` paths, entry and extent guards outside the loops enter the side
  that does work (or leads into a loop), else uniform, by error diffusion.
- Each scheduler (SM partition) issues ≤ 1 instruction per cycle, in order
  per warp (greedy-then-oldest across warps), once the operands are ready and
  a pipe is free; the pipe stays busy `32 × schedulers / rate` cycles (`rate`
  = results/clk/SM) and the result is ready after the dependent latency.
- SM-wide units: `:lsu` (L1/shared data path, busy bytes/bandwidth per warp
  access, at least one cycle per L1 bank pass (`Kernel.waves`: the 32 banks of
  shared memory; words of different lines in one bank serialize, and a pass
  serves ≤ 128 B: 8-/16-byte accesses run per 16/8 lanes), plus one cycle
  per 128-byte line (`Kernel.lines`) that misses L1 or is stored; shuffles too)
  and `:bar`, the address divergence unit (ADU: barriers, and MATCH.ANY and REDUX
  at a pass per distinct operand value, see `:ADU`). A shuffle with a run-time member
  mask gets the convergence check that ptxas emits on sm_70+ (MATCH.ANY, REDUX.OR,
  VOTEU.ANY, LOP3, BRA.DIV; one per mask and block, `Kernel.syncs`); the passes are
  the mask's distinct values over a warp's lanes (lane pairs: 16), so 64-thread
  column scans that shuffle with pair masks are ADU-bound. A global access moves the 32-byte sectors
  its warp touches (`Kernel.sectors`, evaluated per lane from the IR address
  and averaged over consecutive warps: lanes a long run-time extent apart (one
  thread per column of a column-major array) touch 32 sectors per 4-byte
  access, 8× the coalesced 128 bytes; a warp over 252 contiguous bytes at any
  4-byte offset (e.g. a 63-level Float32 column) 4.875; a loaded index is its
  1-based position in an identity map, so lanes that read one entry share it;
  with `inner_extent = 1` a phi that merges forward paths takes each lane's
  incoming value; unknown: coalesced; through a loaded index with `gather =
  :scattered`: a sector per lane); a partially written sector is first read
  from DRAM. Loads of a module constant table (`Kernel.tables`) hit L1; a load
  whose address advances by less than a line per loop iteration (a lane that
  walks an array, e.g. a column's levels) misses once per 128-byte line, and
  one that reads the same address in every iteration
  once per loop entry (`Kernel.walks`, if the reused lines fit in the L1). An
  address that advances by a constant per loop iteration is a pointer the loop
  increments (loop strength reduction).
  L2 and DRAM bandwidth are FIFO queues with
  a per-SM share whose delay adds to load latency; the DRAM latency beyond L2
  grows as 1/(1 - u) with the DRAM utilization u, up to the measured loaded
  value (`loaded`); each global load is served
  by one level (L1, L2 or DRAM) drawn per warp from the `hits` fractions by
  error diffusion, so a batch of loads waits for its slowest member
  (`hit_levels = false`: the mean latency instead); `cache = :warm` serves the
  DRAM fraction from L2 and keeps stores there. With `inner_extent = 1` (a
  degenerate innermost dimension) the run-time sizes that the kernel divides its
  thread index by are 1, and loads whose lanes' addresses are known hit L1 for
  the sectors that the block's earlier loads touched (`Kernel.extent1`) instead
  of `hits.l1`.
- A warp tracks variable-latency results (loads, MUFU, shuffles, S2R) with
  `scoreboards` dependency barriers (6 on sm_70+); a 7th outstanding result
  joins the group that clears last, and consumers of that group wait for all
  of it (as ptxas shares barriers between batched loads).
- `bar.sync` releases a block `BAR` latency after its last warp arrives, so
  barriers end overlap windows and wait for the slowest warp.
- A warp fetches its next instruction when the previous one issues, through
  a per-scheduler L0 and a per-SM L1 instruction cache (LRU over lines); a
  miss delays the issue by the line-fill cost (`:no_instruction`); a SASS
  operand from a constant bank fills its 64-byte line into the same per-SM
  cache on first use. `stagger` offsets the resident blocks' start times so
  that they run different code.
- `spills = true` allocates `regs` registers per thread by the furthest-
  next-use rule and inserts the `STL`/`LDL` traffic (see [`spill`](@ref);
  `remat = true` recomputes cheap values instead), which hits L1 while the
  resident threads' spill area (or the SASS stack frame) fits beside the
  shared memory, else L2 (`local_hits` overrides). `expansion` > 1 charges
  extra issue slots per op.
- A prefetch (`llvm.prefetch`, or inline PTX `prefetch.global.L1|L2`; class
  `PREF`) takes an LSU slot and moves one line per warp from DRAM into L2
  (with probability `hits.dram`) without a result; which later loads hit
  because of it is an input: `load_hits` overrides `hits` per load (by its
  static op index), with the hits-weighted mean latency (no level drawing).
- Finished blocks are replaced; long grids extrapolate the steady state (and a
  simulation beyond `SIM_OPS` ops runs 2 and 4 trips of the kernel's outermost
  loop and extrapolates linearly), and the result is never below any unit's
  contention-free busy time. `time_us` is the cycles at `clock_ghz` plus
  `launch_us` per kernel.
Not modeled: divergence beyond lane tests (data-dependent branches follow
`conds`), cache behavior (hit rates are inputs), shared-memory bank conflicts
(`shared_ways`), instruction
prefetch across lines, dual issue, outstanding-load (MSHR) limits, SASS stall
counts (control words), uneven block dispatch across SMs.
"""
module GPUPerfModel

export DEVICES, device_ir, parse_llvm, parse_sass, simulate, op_counts, occupancy, spill

# ─── Device table ────────────────────────────────────────────────────────────

# class => (dependent latency [cycles], results/clk/SM, pipes...) for sm_60,
# sm_70, sm_80. `:lsu`/`:bar` are SM-wide; other pipes exist per scheduler;
# two pipes = either executes it. Sources (each checked 2026-10-26 against
# the cited table):
#  PG   CUDA C Programming Guide 12.5 §5.4.1 table + footnotes, §5.4.3
#       (__syncthreads 32/16/16 per clk), §16.5-16.7 (schedulers, SFUs)
#  BP   CUDA C Best Practices Guide 13.x Table 5 (8.0: add.s32 128, IADD3 64)
#  J18  Jia et al., arXiv:1804.06826, Tables 3.1, 3.4, 4.1 (P100, V100)
#  J19  Jia et al., arXiv:1903.07486, Table 3.2 (L1 load throughput)
#  A19  Arafa et al., arXiv:1905.08778, Table IV, Fig. 6a (P100)
#  A22  Abdelkhalik et al., arXiv:2208.11174, Tables II, IV, V (A100)
#  L25  Luo et al., arXiv:2501.12084, Tables 3-5 (A100 PCIe 40GB)
#  Z20  Zhang et al., arXiv:2004.05371, Table II (bar.sync, shfl)
#  WP   NVIDIA A100 whitepaper Tables 4-5; nvidia.com/a100 (80GB SXM: 2039 GB/s)
#  NC   Nsight Compute 2025 on an A100-SXM4-80GB (ClimaCore kernels): the ADU pipe's peak,
#       sm__inst_executed_pipe_adu.peak_sustained, is 0.5 inst/clk/SM; MATCH.ANY counts one
#       inst per distinct operand value among the lanes there (one in smsp__ counters)
#  ASM  assumption, no measurement found: check against Nsight Compute
const OPS = Dict(
    # FADD FMUL FFMA; J18, A22 T2, PG
    :FP32 => ((6, 64, :fma), (4, 64, :fma), (4, 64, :fma)),
    # IADD3; J18, A22 T2, PG, BP
    :IADD => ((6, 64, :fma), (4, 64, :alu), (4, 128, :alu, :fma)),
    # LOP3 SHF SEL MOV; J18, PG; sm_80 lat ASM
    :ALU => ((6, 64, :fma), (4, 64, :alu), (4, 64, :alu)),
    # ISETP FSETP; J18, PG; sm_80 lat ASM
    :SETP => ((12, 32, :fma), (4, 64, :alu), (4, 64, :alu)),
    # 6.0: 3 XMAD (A19, PG); J18; A22 (3→4)
    :IMAD => ((13, 64 / 3, :fma), (5, 64, :fma), (4, 64, :fma)),
    # IMAD.WIDE: half rate (A22 T5); 6.0/7.0 ASM
    :IMADW => ((13, 16, :fma), (5, 32, :fma), (4, 32, :fma)),
    # DADD DMUL DFMA; J18, A22 T2, PG
    :FP64 => ((8, 32, :fp64), (8, 32, :fp64), (5, 32, :fp64)),
    # MUFU POPC; J18, PG; sm_80 lat ASM
    :XU => ((14, 16, :xu), (14, 16, :xu), (14, 16, :xu)),
    # I2F F2I F2F; J18 (6.0), PG; lat ASM
    :CONV => ((14, 16, :xu), (14, 16, :xu), (14, 16, :xu)),
    # Z20, PG; sm_80 lat ASM
    :SHFL => ((31, 32, :lsu), (22, 32, :lsu), (22, 32, :lsu)),
    # Z20 (after last arrival), PG §5.4.3
    :BAR => ((218, 32, :bar), (22, 16, :bar), (22, 16, :bar)),
    # MATCH.ANY (a pass per distinct value), REDUX: the address divergence unit, which
    # also runs BAR (`:bar`; NC, PG §5.4.3); lat ASM (= SHFL, plus the extra passes)
    :ADU => ((31, 16, :bar), (22, 16, :bar), (22, 16, :bar)),
    # ASM: issue-limited only
    :BRA => ((1, 64, :cbu), (1, 128, :cbu), (1, 128, :cbu)),
    # ASM: latency = shared memory
    :S2R => ((24, 64, :cbu), (19, 128, :cbu), (29, 128, :cbu)),
    # ASM: uniform datapath (U* ops)
    :UNI => ((6, 64, :uni), (4, 128, :uni), (4, 128, :uni)),
)

struct Device
    name::String
    sm::Int
    n_sm::Int
    clock_ghz::Float64
    n_sched::Int     # warp schedulers per SM, one issue per clock each (PG §16)
    max_warps::Int   # occupancy limits: PG technical specifications, cuda_occupancy.h
    max_blocks::Int
    regs::Int
    smem::Int
    smem_block::Int
    smem_unit::Int
    smem_reserved::Int
    ops::Dict{Symbol, Tuple}

    # cycles; dram_loaded: see `loaded`
    lat::NamedTuple{(:l1, :l2, :dram, :shared, :con, :dram_loaded), NTuple{6, Float64}}

    # B/clk (l1, shared per SM)
    bw::NamedTuple{(:l1, :shared, :l2, :dram), NTuple{4, Float64}}

    # L1 data cache (shared memory carveout 0)
    l1_bytes::Int
    icache::NamedTuple{
        (:l0, :l1, :line0, :line1, :miss0, :miss1, :inst),
        NTuple{7, Float64},
    }  # B, B, B, B, clk, clk, B
end

# Instruction supply (J18 §3.3, Fig. 3.4-3.5; J19 §3.3, Table 3.1): a per-scheduler L0
# (Volta 12 KiB, 3-way, 256 B lines; Turing 16 KiB), a per-SM L1 (Volta 128 KiB, 8-way, 512
# B lines; Turing 46 KiB), then the unified L2. One warp of independent FFMAs streams at ~2
# cycles/instruction from L0, ~4 from L1 and ~8 from L2 (J18 Fig. 3.4-3.5 axes), i.e.
# filling a 256 B L0 line from L1 costs ~32 cycles and a 512 B L1 line from L2 ~192 (≈ L2
# latency: no prefetch across lines). GA100 has the same L0-per-partition, L1-per-SM
# hierarchy (WP, GA100 SM figure) with unpublished sizes: Volta values (ASM). P100 (J18
# Table 3.1): 8 KiB L1 per SM (modeled per scheduler, ASM), 128 KiB L1.5, 8 B instructions
# plus one control word per three (10.67 B); fill costs ASM. Instructions are 16 B on
# sm_70+.
const ICACHE_70 = (
    l0 = 12288.0,
    l1 = 131072.0,
    line0 = 256.0,
    line1 = 512.0,
    miss0 = 32.0,
    miss1 = 192.0,
    inst = 16.0,
)

device(
    i,
    name,
    sm,
    n_sm,
    ghz,
    n_sched,
    smem,
    smem_block,
    unit,
    reserved,
    lat,
    bw,
    l1_bytes,
    icache,
) = Device(name, sm,
    n_sm, ghz, n_sched, 64, 32, 65536, smem, smem_block, unit, reserved,
    Dict{Symbol, Tuple}(c => r[i] for (c, r) in OPS),
    lat, bw, l1_bytes, icache)

const GiB = 1.073741824

const DEVICES = Dict(
    # P100-PCIE: lat J18 T3.1 (L1 82, L2 234, shared 24, const 24), A19 Fig. 6a (DRAM 482);
    # bw J19 T3.2 (L1 31.3), J18 T3.1 (shared 7763 GiB/s, DRAM 510 GiB/s), J18 T3.4 (L2 1624
    # GB/s)
    :sm_60 =>
        device(1, "Tesla P100-PCIE-16GB", 60, 56, 1.328, 2, 65536, 49152, 256, 0,
            (
                l1 = 82.0,
                l2 = 234.0,
                dram = 482.0,
                shared = 24.0,
                con = 24.0,
                dram_loaded = 482.0,
            ),  # loaded ASM (= unloaded)
            (
                l1 = 31.3,
                shared = 7763GiB / 56 / 1.328,
                l2 = 1624 / 1.328,
                dram = 510GiB / 1.328,
            ), 24576,  # J18 T3.1
            (; ICACHE_70..., l0 = 8192.0, miss0 = 48.0, miss1 = 234.0, inst = 32 / 3)),
    # V100-PCIE: lat J18 (L1 28, L2 193, DRAM 375 = L2 miss + TLB hit, shared 19, const 27);
    # bw J18 (L1 108.3, shared 12080 GiB/s, L2 2155 GB/s, DRAM 750 GiB/s)
    :sm_70 =>
        device(2, "Tesla V100-PCIE-16GB", 70, 80, 1.380, 4, 98304, 98304, 256, 0,
            (
                l1 = 28.0,
                l2 = 193.0,
                dram = 375.0,
                shared = 19.0,
                con = 27.0,
                dram_loaded = 375.0,
            ),  # loaded ASM (= unloaded)
            (
                l1 = 108.3,
                shared = 12080GiB / 80 / 1.380,
                l2 = 2155 / 1.380,
                dram = 750GiB / 1.380,
            ), 131072, ICACHE_70),
    # A100-SXM4-80GB (CliMA CI): lat L25 T3 (L1 33, shared 29, DRAM 566), T4 (L2 near hit
    # 208; far hit 357), const ASM (V100); bw L25 T5 (L1 99.5, shared 127.9, L2 2008 B/clk),
    # DRAM 2039 GB/s (WP) × 0.905 (L25 T5 measured/peak on A100 PCIe). dram_loaded = 800:
    # MEASURED CALIBRATION (A100 Nsight Compute profile of a latency-bound fused
    # linear-solver kernel at 47% DRAM / 60% L2 throughput, 24 warps/SM: warp cycles per
    # instruction 15.3 with long_scoreboard 56%; the model gives 11.6 at the unloaded 566
    # and 13.3 at 800 from the kernel's SASS; 800 also keeps the DRAM-bound unfused kernels
    # within 25%).
    :sm_80 =>
        device(3, "NVIDIA A100-SXM4-80GB", 80, 108, 1.410, 4, 167936, 166912, 128, 1024,
            (
                l1 = 33.0,
                l2 = 208.0,
                dram = 566.0,
                shared = 29.0,
                con = 27.0,
                dram_loaded = 800.0,
            ),
            (l1 = 99.5, shared = 127.9, l2 = 2008.0, dram = 2039 * 0.905 / 1.410),
            196608, ICACHE_70),  # L1 192 KB: WP
)

"Copy of `dev` with fields replaced, e.g. `with(dev; lat = (; dev.lat..., l2 = 357.0))`."
with(dev::Device; kw...) =
    Device((get(kw, f, getfield(dev, f)) for f in fieldnames(Device))...)

"""
    occupancy(device, threads, regs = 0, shmem = 0) -> Int

Resident blocks per SM of `device` (`:sm_80` or a `Device`): the minimum of the warp,
block, register and shared-memory limits.
"""
occupancy(device::Symbol, args...) = occupancy(DEVICES[device], args...)
function occupancy(dev::Device, threads, regs = 0, shmem = 0)
    W = cld(threads, 32)
    n = min(dev.max_blocks, dev.max_warps ÷ W)
    rpw = cld(regs * 32, 256) * 256   # registers are allocated per warp, 256 at a time
    regs > 0 && (n = min(n, ((dev.regs ÷ dev.n_sched) ÷ rpw) * dev.n_sched ÷ W))
    used = cld(shmem + dev.smem_reserved, dev.smem_unit) * dev.smem_unit
    return used > 0 ? min(n, dev.smem ÷ used) : n
end

# ─── Kernel representation ───────────────────────────────────────────────────

const CLASSES =
    (:FREE, :FP32, :IADD, :ALU, :SETP, :IMAD, :IMADW, :FP64, :XU, :CONV, :SHFL, :BAR, :ADU,
        :BRA, :S2R, :UNI, :LDG, :STG, :LDS, :STS, :LDL, :STL, :LDC, :ATOM, :PREF)

const CID = Dict(c => UInt8(i) for (i, c) in enumerate(CLASSES))

const FIRST_MEM = CID[:LDG]

"Basic blocks of SASS-like ops over value ids (SSA values or registers)."
mutable struct Kernel
    cls::Vector{UInt8}
    dst::Vector{Int32}              # value written (0: none); dst2: high register of a pair
    dst2::Vector{Int32}
    src_ptr::Vector{Int32}          # srcs of op k: srcs[src_ptr[k]:src_ptr[k+1]-1]
    srcs::Vector{Int32}
    bytes::Vector{Int16}            # bytes per thread of memory ops
    # 32-byte sectors one warp access touches (0: coalesced; < 0: through a loaded index,
    # -Inf: unknown)
    sectors::Vector{Float32}
    partial::Vector{Float32}        # of which a warp store writes partially
    waves::Vector{Float32}          # L1 bank passes of one warp access (0: coalesced)
    lines::Vector{Float32}          # 128-byte lines it touches (0: coalesced)
    blocks::Vector{UnitRange{Int}}  # op ranges
    labels::Vector{String}
    succs::Vector{Vector{Int}}
    probs::Vector{Vector{Float64}}  # NaN: unspecified
    conds::Vector{String}           # predicate of each two-way branch ("" if none)
    trips::Dict{Int, Int}           # inferred trip counts by loop header
    # headers in `trips` whose loop can also leave through another exit (a `break` or `return`
    # on computed data, e.g. a convergence test): `trips` is its bound
    early::Set{Int}
    nvalues::Int

    # registers per value (0: predicate, kernel argument, folded value)
    width::Vector{UInt8}
    phipred::Dict{Int, Vector{Int}} # phi op => predecessor block of each source
    notes::Vector{String}
    frame::Int                      # local bytes per thread: stack frame (SASS), allocas (IR)
    clines::Dict{Int, Int}          # parsed op => constant-bank line (64 B) of its operand (SASS)
    orig::Vector{Int32}             # parsed op each op is (0: inserted by `spill`)
    tables::Set{Int}                # parsed loads from a module constant table (L1 hits)
    splits::Set{Int}                # blocks that branch on thread-range splits (see `walk`)
    merges::Dict{Int, Int}          # block that splits every warp's lanes => reconvergence
    # block whose branch sends the warps of a block different ways => (side of warp w
    # (mod 16): 1 true, 2 false, 3 both, as SIMT; reconvergence block of side 3, or 0)
    wsplit::Dict{Int, Tuple{Vector{UInt8}, Int}}
    # block that ends in a call of a device function => (its entry block, its return blocks)
    calls::Dict{Int, Tuple{Int, Vector{Int}}}
    # block of a callee copy => the same block of the callee's first copy (one code copy:
    # they share instruction-cache lines)
    alias::Dict{Int, Int}
    # header of a loop with a run-time bound => (induction step, largest trip count)
    steps::Dict{Int, Tuple{Int, Int}}
    # block of a divergent lane test (`merges`, `wsplit`) => lanes that take its true side,
    # by warp (mod 16)
    lanes::Dict{Int, Vector{UInt8}}
    # parsed global load in a loop whose lanes' addresses advance by fewer bytes than a line
    # per iteration (0: the same address in every iteration) => (those bytes, loop header)
    walks::Dict{Int, Tuple{Int, Int}}
    # parsed global access => (sectors, partial, waves, lines, L1 fraction of a load: NaN
    # if unknown) when the run-time divisors of the thread index are 1 (`inner_extent = 1`)
    extent1::Dict{Int, NTuple{5, Float32}}
    # parsed ops of shuffle member-mask checks (dropped on sm_6x, see `simulate`)
    syncs::Set{Int}
end
Kernel() =
    Kernel(UInt8[], Int32[], Int32[], Int32[1], Int32[], Int16[], Float32[], Float32[],
        Float32[], Float32[], UnitRange{Int}[],
        String[], Vector{Int}[],
        Vector{Float64}[], String[], Dict{Int, Int}(), Set{Int}(), 0, UInt8[],
        Dict{Int, Vector{Int}}(),
        String[], 0, Dict{Int, Int}(), Int32[], Set{Int}(), Set{Int}(), Dict{Int, Int}(),
        Dict{Int, Tuple{Vector{UInt8}, Int}}(), Dict{Int, Tuple{Int, Vector{Int}}}(),
        Dict{Int, Int}(), Dict{Int, Tuple{Int, Int}}(), Dict{Int, Vector{UInt8}}(),
        Dict{Int, Tuple{Int, Int}}(), Dict{Int, NTuple{5, Float32}}(), Set{Int}())

# `Kernel.extent1` of a parsed kernel, evaluated at its first `simulate` with
# `inner_extent = 1`
const EXTENT1 = WeakKeyDict{Kernel, Function}()

newval!(K, width = 0) = (K.nvalues += 1; push!(K.width, width); Int32(K.nvalues))

srcs(K, k) = view(K.srcs, K.src_ptr[k]:(K.src_ptr[k + 1] - 1))

function emit!(
    K,
    c::Symbol,
    dst,
    srcs;
    bytes = 0,
    dst2 = 0,
    sectors = 0,
    partial = 0,
    waves = 0,
    lines = 0,
    orig = length(K.cls) + 1,
)
    push!(K.cls, CID[c])
    push!(K.orig, orig)
    push!(K.dst, dst)
    push!(K.dst2, dst2)
    push!(K.bytes, bytes)
    push!(K.sectors, sectors)
    push!(K.partial, partial)
    push!(K.waves, waves)
    push!(K.lines, lines)
    append!(K.srcs, srcs)
    push!(K.src_ptr, length(K.srcs) + 1)
    return dst
end

"""
Emit a dependent `chain` of classes (the last writes `dst`) plus `side` ops reading `srcs`.
"""
function emit_chain!(K, chain, side, srcs, dst)
    prev = srcs
    for (i, c) in enumerate(chain)
        prev = [emit!(K, c, i == length(chain) ? dst : newval!(K), prev)]
    end
    foreach(c -> emit!(K, c, 0, srcs), side)
end

"""
    op_counts(K; dynamic = false, trips, conds, probs, loops, default_trip) -> Dict

Op counts by class: static, or per thread over the walked trace (`dynamic = true`, under
the control-flow keywords of [`simulate`](@ref)).
"""
function op_counts(K::Kernel; dynamic = false, loops = nothing, default_trip = 1, kw...)
    dynamic || return counts(K, eachindex(K.blocks))
    early = loops_trip(loops)
    return counts(K, walk(K; kw..., default_trip = something(early, default_trip), early))
end

# op counts by class over a block sequence
function counts(K, blocks)
    visits = zeros(Int, length(K.blocks))
    foreach(b -> visits[b] += 1, blocks)
    n = Dict{Symbol, Int}()
    for b in findall(>(0), visits), k in K.blocks[b]
        K.cls[k] == 1 || (n[CLASSES[K.cls[k]]] = get(n, CLASSES[K.cls[k]], 0) + visits[b])
    end
    return n
end

# ─── LLVM IR front end ───────────────────────────────────────────────────────

# IR → SASS-like classes (typical ptxas sm_70/80 output; chain = dependent
# sequence, side = independent checks and branches; validate with SASS):
#   fadd/fsub/fmul/fma → FP32|FP64; a `contract` fmul whose only user is a
#     `contract` fadd/fsub fuses into it (FFMA); fneg/fabs/casts/phi → free
#   fdiv f32 → MUFU.RCP + 5 FFMA (+FCHK, BSSY, BRA, BSYNC); 1/x → MUFU + 2 FFMA;
#     `fast`/`arcp`/`afn` → MUFU + FMUL; f64 → MUFU.RCP64H + 8 DFMA; sqrt alike
#   add/sub/icmp i64 → 2 ops (carry chain; icmp of 32-bit-range values: 1);
#     mul i32 → IMAD; mul i64 → IMAD.WIDE (+2 IMAD unless both fit 32 bits);
#     and/or/xor/shifts/select/sext → ALU (×2 for 64-bit values); a 64-bit value split
#     into or assembled from 32-bit halves (`lshr`/`shl` by 32, `or` of the halves) → free
#   udiv/urem by constant → IMAD.HI + SHF (+IMAD + IADD); by a power of 2 → SHF;
#     by a variable → 16-op I2F/MUFU.RCP/F2I/IMAD sequence
#   getelementptr → per distinct (block, base, leaves): one IMAD.WIDE per
#     32-bit leaf (+IMAD per 64-bit leaf) of the index tree; zext/sext and
#     add/mul/shl by constants that only feed addresses fold (a constant add
#     folds through zext/sext only with nuw/nsw)
#   load/store addrspace 1|0 → LDG/STG, 3 → LDS/STS, 5 → LDL/STL, 4 → LDC;
#     generic pointers into an `alloca` (argument structs forced into memory) → LDL/STL;
#     runs of adjacent, aligned accesses through one base and index prefix
#     merge into one 8/16-byte access (the LoadStoreVectorizer runs in codegen);
#     each global access records the sectors its warp touches (see `footprint`)
#   calls: see CALLS; libdevice calls with exact one-instruction semantics map
#     like their LLVM intrinsics (fabs: free operand modifier, copysign: ALU,
#     frcp_rn: the IEEE reciprocal sequence), others to their SASS sequences
#     ("libm.<name>"), else to a placeholder chain (noted); __nv_* calls with
#     constant arguments fold, repeats of a dominating call are merged
# "2FP32 XU | ALU" → ((:FP32, :FP32, :XU), (:ALU,)): chain | side ops, a count repeats a class
seq(s) = Tuple(
    Tuple(
        Symbol(m[2]) for w in split(h) for m in (match(r"^(\d*)(\w+)$", w),)
        for _ in 1:something(tryparse(Int, m[1]), 1)
    ) for h in split(s, '|'))

const EXPANSIONS = Dict(
    "fdiv.f32" => ((:XU, :FP32, :FP32, :FP32, :FP32, :FP32), (:ALU, :BRA, :BRA, :BRA)),
    "fast.f32" => ((:XU, :FP32), ()),
    "rcp.f32" => ((:XU, :FP32, :FP32), (:IADD, :SETP, :BRA, :BRA, :BRA)),
    "fdiv.f64" =>
        ((:XU, ntuple(_ -> :FP64, 8)...), (:FP32, :SETP, :SETP, :BRA, :BRA, :BRA)),
    "rcp.f64" => ((:XU, :FP64, :FP64, :FP64, :FP64), (:FP32, :SETP, :BRA, :BRA, :BRA)),
    "sqrt.f32" => ((:XU, :FP32, :FP32, :FP32), (:IADD, :SETP, :BRA, :BRA, :BRA)),
    "sqrt.f64" =>
        ((:XU, ntuple(_ -> :FP64, 7)...), (:IADD, :IADD, :SETP, :BRA, :BRA, :BRA)),
    "div.i32" => (
        (
            :CONV,
            :XU,
            :IADD,
            :CONV,
            :IMAD,
            :IMAD,
            :IMAD,
            :IMAD,
            :IMAD,
            :SETP,
            :IADD,
            :SETP,
            :IADD,
        ),
        (:IADD, :SETP, :ALU)),
    "div.i64" => (
        (
            :CONV,
            :XU,
            :IADD,
            :CONV,
            :IMAD,
            :IMAD,
            :IMAD,
            :IMAD,
            :IMAD,
            :SETP,
            :IADD,
            :SETP,
            :IADD,
        ),
        (:IADD, :SETP, :ALU, :SETP, :BRA, :BRA)),
    "divc.i32" => ((:IMAD, :ALU), ()),
    "remc.i32" => ((:IMAD, :ALU, :IMAD, :IADD), ()),
    "divc.i64" => ((:IMADW, :IMAD, :IMAD, :ALU), (:ALU,)),
    "remc.i64" => ((:IMADW, :IMAD, :IMAD, :ALU, :IMADW, :IADD, :IADD), (:ALU,)),
    "minmax.i64" => ((:SETP, :SETP, :ALU), (:ALU,)),
    "minmax.f64" => ((:FP64, :ALU), (:IMAD,)),
    "sat" => ((:IADD, :ALU), ()),
    "libm.f32" => (
        (:FP32, :FP32, :FP32, :FP32, :XU, :FP32, :FP32, :FP32, :FP32),
        (:ALU, :ALU, :SETP),
    ),
    "libm.f64" =>
        (ntuple(_ -> :FP64, 16), (:ALU, :ALU, :ALU, :ALU, :SETP, :SETP, :CONV)),
    # Libdevice functions as ptxas emits them (CUDA 13.0, sm_80; used for every device): the
    # instructions between the load and the store of a kernel `a[i] = f(a[i])` up to its EXIT
    # (the fast path; special values are mostly predicated), chain = the longest dependency
    # path from the argument to the result; `powf` with a constant exponent (a variable one
    # adds ~20 ops). Other `__nv_*` functions use the placeholders above.
    "libm.expf" => seq("6FP32 XU FP32 | 2ALU"),
    "libm.logf" => seq("SETP FP32 IADD ALU IADD 13FP32 ALU | 2ALU CONV 2FP32 2SETP"),
    "libm.log2f" => seq("SETP FP32 IADD ALU IADD 15FP32 ALU | 2ALU CONV 2FP32 2SETP"),
    "libm.powf" => seq(
        "FP32 ALU IADD ALU IADD FP32 XU 11FP32 2ALU FP32 CONV ALU CONV 10FP32 2ALU | 5ALU 5BRA 2CONV 20FP32 IADD 11SETP",
    ),
    "libm.tanhf" => seq("FP32 XU FP32 XU FP32 2ALU FP32 | 2ALU 6FP32 3SETP"),
    "libm.atanhf" => seq(
        "FP32 XU 2FP32 ALU FP32 IADD ALU IADD 14FP32 ALU FP32 | 4ALU CONV 3FP32 IADD 5SETP",
    ),
    "libm.sinpif" =>
        seq("FP32 2ALU FP32 CONV ALU CONV 9FP32 | 6ALU 2CONV 3FP32 2IMAD 3SETP"),
    "libm.cbrtf" => seq("FP32 ALU XU 2FP32 XU 5FP32 ALU | FP32 3SETP"),
    "libm.asinf" => seq("2FP32 XU 3FP32 2ALU 9FP32 SETP ALU | 2ALU 2FP32 3SETP"),
    "libm.acosf" => seq("2FP32 XU 3FP32 3ALU 8FP32 ALU 2FP32 | 2ALU 2FP32 4SETP"),
    "libm.atan2f" => seq("ALU IMAD 11FP32 3ALU | 4ALU 4BRA 6FP32 2IMAD 5SETP XU"),
    "libm.log1pf" => seq("FP32 IADD ALU IADD 14FP32 ALU | 2ALU CONV 2FP32 IADD 3SETP"),
    "libm.expm1f" => seq("ALU SETP ALU 11FP32 ALU FP32 | 3ALU 3FP32 IADD 3SETP"),
    "libm.exp" => seq("IMAD 16FP64 | 8ALU 3BRA FP32 2FP64 5IMAD 2SETP"),
    "libm.log" =>
        seq("IMAD FP64 2IMAD 20FP64 ALU | 9ALU 4BRA 9FP64 3IADD 8IMAD 3SETP XU"),
    "libm.tanh" => seq(
        "IMAD FP64 CONV FP32 2ALU FP32 CONV ALU 2CONV 18FP64 ALU FP64 ALU 13FP64 | 5ALU 4BRA CONV 3FP64 7IMAD 5SETP 2XU",
    ),
    # `sinf`, `cosf` (|x| < 105615; beyond, a Payne-Hanek reduction loop): the Cody-Waite
    # reduction by π/2, then a degree-4 polynomial chosen by the quadrant
    "libm.sinf" =>
        seq("FP32 2CONV 3FP32 IMAD 2FP32 ALU 4FP32 | FP32 4IMAD 5ALU 2SETP 4BRA"),
    "libm.cosf" =>
        seq("FP32 2CONV 3FP32 IMAD 2FP32 ALU 4FP32 | FP32 IADD 4IMAD 5ALU 2SETP 4BRA"),
    # `pow`: the call of `__internal_accurate_pow` (a double-double log, then exp) and the
    # caller's special-value tests
    "libm.pow" => seq(
        "IMAD FP64 IMAD 2ALU SETP IADD IMAD FP64 XU 54FP64 IMAD FP64 ALU | 25FP64 3IADD 15IMAD 8ALU 4SETP 6BRA",
    ),
)

# Callee regex => class (scalar op, ×2 for 64-bit where marked) or expansion stem.
const CALLS = [
    # ntid/nctaid: constant bank
    r"^llvm\.(assume|lifetime|dbg|experimental|fabs|trap)|sreg\.n|^__nv_fabs" => :FREE,
    r"sreg\." => :S2R, r"nvvm\.(barrier|bar\.sync)" => :BAR, r"nvvm\.shfl" => :SHFL,
    r"llvm\.(fma|fmuladd)\.|nvvm\.(fma|add|mul)\.r" => :FMA,
    r"llvm\.sqrt|__nv_sqrt|nvvm\.sqrt\.rn" => "sqrt", r"__nv_[fd]rcp_rn" => "rcp",
    r"nvvm\.(ex2|lg2|sin|cos|rcp\.approx|rsqrt|sqrt\.approx|tanh)|__nv_(rsqrt|fast_)" =>
        :XU,
    r"llvm\.(minnum|maxnum|minimum|maximum|[su]min|[su]max|abs)\.|__nv_(f(min|max)|l?l?abs$)" => "minmax",
    # value tests: one compare (FSETP; DSETP on the FP64 pipe)
    r"__nv_(isnan|isinf|isfinite)d$" => :FP64,
    r"__nv_(isnanf|isinff|finitef)$" => :SETP,
    r"llvm\.(floor|ceil|trunc|rint|round|nearbyint)|__nv_(floor|ceil|trunc|rint|round)" =>
        :CONV,
    r"llvm\.[su](add|sub)\.sat" => "sat",
    r"llvm\.(copysign|bswap|fshl|fshr)|nvvm\.vote|__nv_copysign" => :ALU,
    r"llvm\.(ctpop|ctlz|cttz)" => :XU, r"nvvm\.(bar\.warp|membar)" => :BRA,
    r"nvvm\.(ldg|ld)\.global" => :LDG,
    r"^llvm\.prefetch|^prefetch" => :PREF, r"^__nv_" => "libm",
]

const FREE_OPS = Set(
    split(
        "bitcast addrspacecast ptrtoint inttoptr freeze trunc zext extractvalue " *
        "insertvalue " *
        "extractelement insertelement shufflevector phi alloca fneg unreachable",
    ),
)

# same registers as the operand
const RENAMES = Set(split("bitcast addrspacecast ptrtoint inttoptr freeze"))

const CASTS = Set(split("zext sext trunc fpext fptrunc sitofp uitofp fptosi fptoui"))

const REGS = Dict("i1" => 0, "i8" => 1, "i16" => 1, "half" => 1, "i32" => 1, "float" => 1,
    "i64" => 2, "double" => 2,
    "ptr" => 2, "i128" => 4)

regs(t) = (
    m = match(r"^<(\d+) x (.+)>$", t);
    m === nothing ? get(REGS, t, endswith(t, "*") ? 2 : cld(typesize(t), 4)) :
    parse(Int, m[1]) * regs(m[2])
)

const FLAGS = Set(
    split(
        "contract fast nnan ninf nsz arcp afn reassoc nuw nsw exact inbounds disjoint " *
        "volatile eq ne ugt uge ult ule sgt sge slt sle oeq ogt oge olt ole one ord ueq " *
        "une uno true false noundef",
    ),
)

const VALUE = r"%(\"[^\"]+\"|[-\w.$]+)"

const SIZES =
    Dict("half" => 2, "float" => 4, "double" => 8, "i1" => 1, "i8" => 1, "i16" => 2,
        "i32" => 4,
        "i64" => 8, "i128" => 16, "ptr" => 8)

typesize(t) = (
    m = match(r"^[<\[](\d+) x (.+)[>\]]$", t);
    m === nothing ? get(SIZES, t, endswith(t, "*") ? 8 : 4) :
    parse(Int, m[1]) * typesize(m[2])
)

lanes(t) = (m = match(r"^<(\d+) x", t); m === nothing ? 1 : parse(Int, m[1]))

# bytes of an `alloca`'s type: arrays and (nested) structs with naturally aligned members
function allocsize(rest)
    t = strip(splitargs(rest)[1])
    startswith(t, "{") || return typesize(firsttype(rest))
    size_align(t) =
        if startswith(t, "{")
            n = a = 0
            for x in splitargs(strip(t)[2:(end - 1)])
                (m, b) = size_align(strip(x))
                n = cld(n, b) * b + m
                a = max(a, b)
            end
            (cld(n, max(a, 1)) * max(a, 1), max(a, 1))
        elseif (m = match(r"^\[(\d+) x (.+)\]$", t)) !== nothing
            (e, b) = size_align(m[2])
            (parse(Int, m[1]) * e, b)
        else
            (z = typesize(t); (z, min(z, 8)))
        end
    return size_align(t)[1]
end

function firsttype(rest)  # first type token of an operand list (skips flags and predicates)
    for m in eachmatch(
        r"<\d+ x [^>]+>|\[[^\]]+\]|\{[^}]*\}|[\w.]+\**(?: addrspace\(\d+\)\**)?",
        rest,
    )
        m.match in FLAGS || return m.match
    end
    return "i32"
end

function splitargs(s)  # split at top-level commas
    out = String[]
    depth = 0
    i0 = 1
    for (i, ch) in pairs(s)
        if ch == '(' || ch == '[' || ch == '{' || ch == '<'
            depth += 1
        elseif ch == ')' || ch == ']' || ch == '}' || ch == '>'
            depth -= 1
        elseif ch == ',' && depth == 0
            push!(out, strip(s[i0:prevind(s, i)]))
            i0 = nextind(s, i)
        end
    end
    return push!(out, strip(s[i0:end]))
end

lastword(s) = split(s)[end]

valname(s) = strip(strip(s)[2:end], '"')

wide(t) = occursin(r"i64|\*|ptr|double", t)

addrspace(t) = (m = match(r"addrspace\((\d+)\)", t); m === nothing ? 0 : parse(Int, m[1]))

# address space of a pointer operand ("<type> <value>"): the qualifier of the type's outermost
# `*` (`float addrspace(1)**` is a generic pointer to a global pointer)
function ptrspace(s)
    m = match(
        r"^[\w.]+(?:(?: addrspace\(\d+\))?\*)*(?: addrspace\((\d+)\))?\*(?=[\s,]|$)",
        s,
    )
    return m === nothing ? addrspace(s) : m[1] === nothing ? 0 : parse(Int, m[1])
end

# lane-program node kinds of the address evaluation in `parse_llvm`
const K_UNK, K_NONE, K_LANEID, K_TID, K_ZERO, K_SMIN, K_SMAX, K_UMIN, K_UMAX = Int8.(1:9)
const K_PHI, K_CAST, K_GEP, K_SELECT, K_SEL2, K_SEL3 = Int8.(10:15)
const K_ADD, K_SUB, K_MUL, K_SHL, K_AND, K_XOR, K_LSHR, K_ASHR = Int8.(16:23)
const K_UDIV, K_SDIV, K_UREM, K_SREM, K_EQ = Int8.(24:28)
const K_LOAD = Int8(38)  # a loaded value (after the 10 icmp kinds from K_EQ)
const BINOPS = Dict("add" => K_ADD, "or" => K_ADD, "sub" => K_SUB, "mul" => K_MUL,
    "shl" => K_SHL, "and" => K_AND, "xor" => K_XOR, "lshr" => K_LSHR, "ashr" => K_ASHR,
    "udiv" => K_UDIV, "sdiv" => K_SDIV, "urem" => K_UREM, "srem" => K_SREM,
    "icmp" => K_EQ)

# icmp predicates: kind => (comparison, unsigned)
const CMPS = ((==, false), (!=, false), (<, true), (<, false), (<=, true), (<=, false),
    (>, true), (>, false), (>=, true), (>=, false))
const ICMPS = Dict(
    p => Int8(K_EQ + i - 1) for (i, p) in
    enumerate(("eq", "ne", "ult", "slt", "ule", "sle", "ugt", "sgt", "uge", "sge"))
)

const LaneVal = Union{Nothing, Tuple{Vector{Int128}, Bool}}

struct IRInst
    dst::String
    op::String
    rest::String  # operands, metadata stripped
    meta::String  # !prof id
    block::Int
    id::Int       # position in the kernel
end

# ─── Calls of device functions ───────────────────────────────────────────────

# A call of a function defined in the module (a device function that LLVM did not inline) is
# stepped into: the call site gets a copy of the callee's body whose values and labels carry
# the prefix "<callee>.<n>." and whose parameters are the call's arguments (so addresses, loop
# bounds and local-memory pointers resolve through the call), entered from the calling block
# (`Kernel.calls`) and left at its `ret`. Recursive calls, calls nested deeper than
# `CALL_DEPTH` and calls of functions that never return stay one branch op.
const CALL_DEPTH = 4
const CALL_LINES = 10^6  # copied callee lines per kernel
const ASM_STR =
    r"asm (?:sideeffect )?(?:alignstack )?(?:inteldialect )?\"[^\"]*\", \"[^\"]*\""
const LABEL = r"^(\"[^\"]+\"|[-\w.$]+):"
const CALL_SITE =
    r"^\s*(?:%(\"[^\"]+\"|[-\w.$]+) = )?(?:(?:tail|notail|musttail) )?call [^@]*@(\"[^\"]+\"|[-\w.$]+)\("

# the text between the parenthesis that ends `m` and its partner
function parenthesized(l, m)
    i = j = m.offset + ncodeunits(m.match)
    depth = 1
    while j <= ncodeunits(l)
        c = codeunit(l, j)
        c == UInt8('(') ? (depth += 1) : c == UInt8(')') && (depth -= 1) == 0 && break
        j += 1
    end
    return l[i:(j - 1)]
end

# defined functions: name => (parameter names, body lines, :step, or :noreturn / :recursive)
function definitions(lines)
    defs = Dict{String, Tuple{Vector{String}, UnitRange{Int}, Bool}}()
    i = 1
    while i <= length(lines)
        if startswith(lines[i], "define")
            j = something(
                findnext(l -> startswith(l, "}"), lines, i + 1),
                length(lines) + 1,
            )
            m = match(r"@(\"[^\"]+\"|[-\w.$]+)\(", lines[i])
            if m !== nothing && !occursin("ptx_kernel", lines[i])
                ps = [
                    something(
                        match(r"%(\"[^\"]+\"|[-\w.$]+)$", strip(a)),
                        (; captures = [""]),
                    ).captures[1]
                    for a in splitargs(parenthesized(lines[i], m)) if !isempty(strip(a))
                ]
                defs[strip(m[1], '"')] = (
                    String.(strip.(ps, '"')),
                    (i + 1):(j - 1),
                    any(l -> occursin(r"^\s*ret\b", l), view(lines, (i + 1):(j - 1))),
                )
            end
            i = j
        end
        i += 1
    end
    callees = Dict(
        f => Set(
            strip(m[2], '"') for l in view(lines, r)
            for m in (match(CALL_SITE, l),) if
            m !== nothing && haskey(defs, strip(m[2], '"'))
        )
        for (f, (_, r, _)) in defs
    )
    function reaches(f, g, seen = Set{String}())
        g in callees[f] && return true
        push!(seen, f)
        return any(h -> !(h in seen) && reaches(h, g, seen), callees[f])
    end
    return Dict(
        f => (p, r, !ret ? :noreturn : reaches(f, f) ? :recursive : :step) for
        (f, (p, r, ret)) in defs
    )
end

# a callee line with its values and labels renamed and its parameters bound
function rename(l, prefix, bind, types)
    q(name) = (n = prefix * name; occursin(r"^[-\w.$]+$", n) ? n : "\"" * n * "\"")
    sub(x) = (
        name = strip(x[2:end], '"');
        haskey(bind, name) ? bind[name] : name in types ? x : "%" * q(name)
    )
    m = match(LABEL, l)
    m === nothing || return q(strip(m[1], '"')) * l[(ncodeunits(m.match)):end]
    a = findfirst(ASM_STR, l)
    a === nothing && return replace(l, VALUE => sub)
    return replace(l[1:(first(a) - 1)], VALUE => sub) * l[a] *
           replace(l[(last(a) + 1):end], VALUE => sub)
end

# "@name(" of the module's device functions that return (calls of them are stepped into)
function returning_functions(lines)
    fs = String[]
    for (i, l) in enumerate(lines)
        (startswith(l, "define") && !occursin("ptx_kernel", l)) || continue
        j = something(findnext(x -> startswith(x, "}"), lines, i + 1), length(lines))
        any(x -> startswith(lstrip(x), "ret "), view(lines, (i + 1):j)) || continue
        k = findfirst('@', l)
        k === nothing || push!(fs, l[k:something(findnext('(', l, k), lastindex(l))])
    end
    return fs
end

# the kernel body with its calls expanded and the callee copies appended
function expand_kernel(lines, body, links, notes)
    types = Set(
        strip(m[1], '"') for l in lines if startswith(l, '%') &&
            (m = match(r"^%(\"[^\"]+\"|[-\w.$]+) = type ", l)) !== nothing
    )
    own, extra =
        expand_calls(lines, body, definitions(lines), types, String[], links, notes)
    return [own; extra]
end

# The lines of one function body with its calls expanded, and the callee copies (appended
# after the kernel); `links`: (calling block, callee entry, return blocks) labels, callee and
# the copy's prefix.
function expand_calls(lines, body, defs, types, stack, links, notes, n = Ref(0))
    own, extra = String[], String[]
    label = orig = "entry"
    last_part = Dict{String, String}()  # split block => label of its last part
    for l in body
        m = match(LABEL, l)
        m === nothing || (label = orig = strip(m[1], '"'))
        c = m === nothing ? match(CALL_SITE, l) : nothing
        d = c === nothing ? nothing : get(defs, strip(c[2], '"'), nothing)
        name = c === nothing ? "" : strip(c[2], '"')
        why =
            d === nothing || d[3] === :noreturn ? "" :
            d[3] === :recursive ? "recursive" :
            length(stack) >= CALL_DEPTH || n[] >= CALL_LINES ? "depth or size limit" :
            "step"
        if why != "step"
            msg = "call @$name not stepped into ($why): one branch op"
            isempty(why) || msg in notes || push!(notes, msg)
            push!(own, l)
            continue
        end
        params, rng, _ = d
        n[] += length(rng)
        prefix = replace(name, r"[^-\w.$]" => "_") * ".$(length(links) + 1)."
        vals = map(splitargs(parenthesized(l, c))) do a
            v = isempty(strip(a)) ? "" : split(strip(a))[end]
            occursin(r"^(%|@|-?\d|true$|false$|null$|undef$|poison$|zeroinitializer$)", v) ? v :
            "undef"
        end
        bind = Dict(p => v for (p, v) in zip(params, vals) if !isempty(p))
        copy = [rename(x, prefix, bind, types) for x in view(lines, rng)]
        i1 = findfirst(x -> !isempty(strip(x)), copy)
        (i1 === nothing || match(LABEL, copy[i1]) === nothing) &&  # unnamed entry block
            pushfirst!(copy, "$(prefix)$(count(p -> occursin(r"^\d+$", p), params)):")
        entry = strip(
            match(LABEL, copy[findfirst(x -> match(LABEL, x) !== nothing, copy)])[1],
            '"',
        )
        push!(stack, name)
        cown, cextra = expand_calls(lines, copy, defs, types, stack, links, notes, n)
        pop!(stack)
        rets = Tuple{String, String, String}[]  # (block, type, value)
        blk = entry
        for x in cown
            (mm = match(LABEL, x)) === nothing || (blk = strip(mm[1], '"'))
            (mm = match(r"^\s*ret (.*)$", replace(x, r",\s*!.*$" => ""))) === nothing &&
                continue
            tv = rsplit(strip(mm[1]), ' '; limit = 2)
            v =
                length(tv) == 2 &&
                occursin(r"^(%|-?\d|true$|false$|null$|undef$|poison$)", tv[2]) ? tv[2] :
                length(tv) == 2 ? "undef" : ""
            push!(rets, (blk, length(tv) == 2 ? tv[1] : "", v))
        end
        cont = prefix * "ret"
        push!(own, "  br label %\"$cont\"", "\"$cont\":")
        dst = c[1]
        if dst !== nothing && !isempty(rets) && !isempty(rets[1][3])
            t = rets[1][2]
            push!(own,
                length(rets) == 1 ? "  %$dst = bitcast $t $(rets[1][3]) to $t" :
                "  %$dst = phi $t " * join(("[ $v, %\"$b\" ]" for (b, _, v) in rets), ", "))
        end
        push!(links, (label, entry, first.(rets), name, prefix))
        last_part[orig] = label = cont
        append!(extra, cown, cextra)
    end
    # phis of the successors of a split block name its last part as their predecessor
    isempty(last_part) || for (i, x) in enumerate(own)
        occursin(r"= phi ", x) || continue
        own[i] = replace(
            x,
            r"(\[\s*[^\[\],]+,\s*)%(\"[^\"]+\"|[-\w.$]+)(\s*\])" =>
                s -> begin
                    mm = match(r"(\[\s*[^\[\],]+,\s*)%(\"[^\"]+\"|[-\w.$]+)(\s*\])", s)
                    b = strip(mm[2], '"')
                    haskey(last_part, b) ? mm[1] * "%\"" * last_part[b] * "\"" * mm[3] : s
                end,
        )
    end
    return own, extra
end

"""
    parse_llvm(ir; fn = nothing) -> Kernel

Parse the kernel (first `ptx_kernel`, or the `define` line containing `fn`)
of an LLVM IR module (typed or opaque pointers) into SASS-like op classes and
infer loop trip counts from constant-bounded induction variables (constants
may come from `llvm.assume(icmp eq %v, C)`).
"""
function parse_llvm(ir::AbstractString; fn = nothing)
    lines = split(ir, '\n')
    weight_re = r"^(!\d+) = !\{!\"branch_weights\",((?: i32 \d+,?)+)\}"
    weight_lines = (match(weight_re, l) for l in lines)
    weights = Dict(
        m[1] => [parse(Float64, x[1]) for x in eachmatch(r"i32 (\d+)", m[2])] for
        m in weight_lines if m !== nothing
    )
    i0 = findfirst(
        l -> startswith(l, "define") && occursin(something(fn, "ptx_kernel"), l),
        lines,
    )
    i0 === nothing && error(
        "no kernel definition found: expected a `define` line with " *
        (fn === nothing ? "`ptx_kernel` (or pass `fn`, part of its name)" : "\"$fn\""),
    )
    K = Kernel()
    ids = Dict{String, Int32}()
    vid(name) = get!(() -> newval!(K), ids, name)
    foreach(m -> vid(m[1]), eachmatch(VALUE, lines[i0]))  # arguments: ready at start
    insts = IRInst[]
    label_of = Dict{String, Int}()
    pending = ""  # joins multi-line `switch` statements
    body = view(
        lines,
        (i0 + 1):(something(
            findnext(l -> startswith(l, "}"), lines, i0 + 1),
            length(lines) + 1,
        ) - 1),
    )
    links = Tuple{String, String, Vector{String}, String, String}[]  # see `expand_calls`
    # (compiled at the first kernel that calls a device function that returns)
    fs = returning_functions(lines)
    isempty(fs) || !any(l -> any(f -> occursin(f, l), fs), body) ||
        (body = Base.invokelatest(expand_kernel, lines, body, links, K.notes))
    for l in body
        if !isempty(pending) || startswith(strip(l), "switch") && !occursin(']', l)
            pending *= " " * strip(l)
            occursin(']', l) || continue
            l = pending
            pending = ""
        end
        (s = strip(l); isempty(s) || startswith(s, ";")) && continue
        m = match(r"^(\"[^\"]+\"|[-\w.$]+):", l)
        if m !== nothing || isempty(K.labels)
            push!(K.labels, m === nothing ? "entry" : strip(m[1], '"'))
            label_of[K.labels[end]] = length(K.labels)
            m === nothing || continue
        end
        meta = something(match(r"!prof (!\d+)", s), (; captures = [""])).captures[1]
        s = replace(
            s,
            r",\s*!(?:[\w.]+) !\d+" => "",
            r",\s*!\w+ !\{[^}]*\}" => "",
            r"\s*;.*$" => "",
        )
        m = match(
            r"^(?:%(\"[^\"]+\"|[-\w.$]+) = )?(?:(?:tail|notail|musttail) )?(\w+)\s*(.*)$",
            s,
        )
        id = length(insts) + 1
        push!(insts, IRInst(something(m[1], ""), m[2], m[3], meta, length(K.labels), id))
    end
    def = Dict(r.dst => r for r in insts if !isempty(r.dst))
    opv = Vector{Union{Nothing, Vector{SubString{String}}}}(nothing, length(insts))
    function operands(r)  # names of the values it reads (memoized)
        o = opv[r.id]
        o === nothing || return o
        return opv[r.id] =
            r.op == "phi" ?
            [x[1] for x in eachmatch(r"\[\s*%(\"[^\"]+\"|[-\w.$]+)\s*,", r.rest)] :
            unique(
                x[1] for x in
                eachmatch(VALUE, replace(r.rest, r"label %(\"[^\"]+\"|[-\w.$]+)" => ""))
            )
    end
    users = Dict{String, Vector{IRInst}}()
    foreach(r -> foreach(v -> push!(get!(users, v, IRInst[]), r), operands(r)), insts)
    known = Dict{String, Int128}()  # values fixed by llvm.assume(icmp eq %v, C)
    for r in insts
        (r.op == "call" && occursin("@llvm.assume", r.rest)) || continue
        c = get(
            def,
            something(match(VALUE, r.rest), (; captures = [""])).captures[1],
            nothing,
        )
        c !== nothing && c.op == "icmp" &&
            (m = match(r"^eq \S+ %([-\w.$]+), (-?\d+)$", c.rest)) !== nothing &&
            (known[m[1]] = parse(Int128, m[2]))
    end
    cval(s) = (
        s = strip(s);
        startswith(s, "%") ? get(known, valname(s), nothing) :
        (m = match(r"^-?\d+$", s); m === nothing ? nothing : parse(Int128, s))
    )
    argv = Vector{Union{Nothing, Vector{String}}}(nothing, length(insts))
    tyv = Vector{Union{Nothing, String}}(nothing, length(insts))
    function args(r)  # operands without leading keywords (memoized)
        a = argv[r.id]
        return a === nothing ?
               (argv[r.id] = splitargs(replace(r.rest, r"^(\w+ )*" => ""))) : a
    end
    function rtype(r)  # first operand type (memoized)
        t = tyv[r.id]
        return t === nothing ? (tyv[r.id] = String(firsttype(r.rest))) : t
    end
    castsrc(r) = (m = match(VALUE, r.rest); m === nothing ? "" : "%" * m[1])  # "": a global
    narrow(s) = (
        c = cval(s);
        c !== nothing ? abs(c) < 2^31 :
        startswith(strip(s), "%") &&
        (
            d = get(def, valname(s), nothing);
            d !== nothing && !wide(rtype(d)) && d.op != "bitcast"
        )
    )

    # Index arithmetic that only feeds addresses (fixed point over users), including
    # instcombine's 64-bit form of sext(trunc(x) ± c): `ashr|lshr exact (±(shl x, 32) +
    # (c << 32)), 32`, which codegen folds into the address offset like a constant add.
    exact32(r) =
        r.op in ("ashr", "lshr") && occursin("exact", r.rest) && cval(args(r)[2]) == 32
    foldable = Set(
        r.dst for r in insts if r.op in ("zext", "sext") ||
            r.op in ("add", "sub", "mul", "shl", "or") &&
            wide(rtype(r)) ||
            r.op == "add" && occursin(r"nuw|nsw", r.rest) && cval(args(r)[2]) !== nothing ||
            exact32(r)
    )
    changed = true
    while changed
        changed = false
        for v in collect(foldable)
            r = def[v]
            us = get(users, v, IRInst[])
            ok =
                all(
                    u ->
                        u.op == "getelementptr" && lastword(args(u)[2]) != "%$v" ||
                        u.dst in foldable,
                    us,
                ) &&
                !(
                    r.op in ("mul", "shl", "sub") && cval(args(r)[2]) === nothing &&
                    !(r.op == "sub" && cval(args(r)[1]) !== nothing && all(exact32, us))
                ) &&
                (
                    r.op != "add" || wide(rtype(r)) ||
                    all(
                        u ->
                            u.op == "zext" && occursin("nuw", r.rest) ||
                            u.op == "sext" && occursin("nsw", r.rest), us)
                )
            ok || (delete!(foldable, v); changed = true)
        end
    end
    function leaves!(acc, s)  # variable leaves (name, fits 32 bits) of an address index
        (cval(s) !== nothing || !startswith(strip(s), "%")) && return acc
        r = get(def, valname(s), nothing)
        (r === nothing || !(r.dst in foldable)) && return push!(acc, (s, narrow(s)))
        if r.op in ("zext", "sext")
            d = get(def, valname(castsrc(r)), nothing)
            d !== nothing && d.op == "add" && d.dst in foldable &&
                return push!(acc, (lastword(args(d)[1]), true))
            return push!(acc, (castsrc(r), true))
        end
        a = args(r)
        leaves!(acc, lastword(a[1]))
        return r.op in ("mul", "shl") ? acc : leaves!(acc, a[2])
    end

    # Address space a generic pointer points into: local for an `alloca` (kernel-argument
    # structs that a run-time index forces into memory), the source space of an
    # `addrspacecast` (shared memory), else global (0).
    function origin_space(s, d = 0)
        r = get(def, valname(s), nothing)
        (r === nothing || d > 32) && return 0
        r.op == "alloca" && return 5
        r.op == "addrspacecast" && return addrspace(r.rest)
        r.op == "getelementptr" && return origin_space(lastword(args(r)[2]), d + 1)
        return r.op in RENAMES ? origin_space(castsrc(r), d + 1) : 0
    end
    local_ptr(s) = origin_space(s) == 5

    # A pointer into a module global: a constant table (differentiation matrix, quadrature
    # weights) that every warp of an SM re-reads, so its loads hit L1.
    function table_ptr(s)
        for _ in 1:16
            startswith(strip(s), "@") && return true
            r = get(def, valname(s), nothing)
            r === nothing && return false
            if r.op == "getelementptr"
                s = lastword(args(r)[2])
            elseif r.op in RENAMES
                c = castsrc(r)
                isempty(c) && return occursin('@', r.rest)
                s = c
            else
                return false
            end
        end
        return false
    end

    # Warp access footprint: the address of a global access is evaluated for the 32 lanes of
    # one warp (lane l has tid.x = l in block 0; a loop-carried value takes its first
    # iteration's incoming value). Kernel-argument fields (run-time extents and strides) and
    # their alloca copies are lane-invariant unknowns taken as 2^20, i.e. at least a warp
    # wide: an index component multiplied by one keeps its lanes in distinct sectors, and a
    # division by one has a lane-invariant quotient and a remainder with the dividend's lane
    # pattern (a mixed-radix index such as (level, node, element) recomposes to the same
    # address either way). Values
    # that pass through an unknown are exact only in their lane-to-lane differences; the
    # other nonlinear ops (shifts right, masks, comparisons) need exact operands; `or` is
    # the disjoint add LLVM emits for addresses. The access is charged the 32-byte sectors
    # the warp's start addresses fall in, the lowest one aligned (`Kernel.sectors`; 0 where
    # the address is not a function of the lane: the coalesced `bytes` × 32). One thread per
    # column of a column-major array touches 32 sectors per 4-byte access: 8× the coalesced
    # traffic. Warps 17-32 are warps 1-16 with `inner_extent = 1`, where the run-time values
    # that divide an index (`divisors`: the size of a degenerate innermost dimension, e.g. a
    # level count of 1) are 1.
    ZERO = zeros(Int128, 32)
    UNK = fill(Int128(2)^20, 32)
    ONE = ones(Int128, 32)
    LANES = [Int128.(32w .+ (0:31)) for w in 0:15]  # tid.x of warp w + 1
    invariant(x) = all(==(x[1]), x)
    divisors = Set{String}()
    for r in insts
        r.op in ("sdiv", "udiv", "srem", "urem") || continue
        local dv = lastword(args(r)[2])
        local dr
        while (dr = get(def, valname(dv), nothing)) !== nothing &&
              (dr.op in RENAMES || dr.op in ("zext", "sext", "trunc"))
            dv = castsrc(dr)
        end
        startswith(dv, "%") && push!(divisors, valname(dv))
    end

    # Each value an address uses is decoded once into a node: its op kind, operand refs (> 0:
    # node, < 0: constant -ref, 0: neither, i.e. not a function of the lane), the unsigned
    # range of its type, and the step of each GEP index (`gfail`: the index whose element
    # type is not an array). Evaluated per warp with per-warp caches.
    names = String[]
    kind = Int8[]  # 0: not decoded yet
    nrefs = Vector{Vector{Int}}()
    nbits = Int128[]
    steps = Vector{Vector{Int128}}()
    gfail = Int[]
    consts = Int128[]
    node = Dict{String, Int}()
    philab = Dict{Int, Tuple{Int, Vector{String}}}()  # phi => (block, incoming labels)
    dep1 = Int8[]  # see `divdep`
    lseen = [Bool[] for _ in 1:32]
    vals = [LaneVal[] for _ in 1:32]
    function ref(s)
        s = strip(s)
        c = cval(s)
        c !== nothing && (push!(consts, c); return -length(consts))
        startswith(s, "%") || startswith(s, "@") || return 0
        key = startswith(s, "@") ? s : valname(s)  # a module global (constant table): invariant
        return get!(node, key) do
            push!(names, key)
            push!(kind, 0)
            push!(nrefs, Int[])
            push!(nbits, 0)
            push!(steps, Int128[])
            push!(gfail, 0)
            foreach(v -> push!(v, false), lseen)
            foreach(v -> push!(v, nothing), vals)
            push!(dep1, 0)
            length(names)
        end
    end
    function decode!(n)
        r = get(def, names[n], nothing)
        r === nothing && return (kind[n] = K_UNK)  # kernel argument or module global
        a = args(r)
        op = r.op
        nbits[n] = Int128(2)^(wide(rtype(r)) ? 64 : 32)
        ops(is) = (nrefs[n] = [ref(lastword(a[i])) for i in is])
        src = op == "extractvalue" ? get(def, valname(lastword(a[1])), nothing) : r
        global_load(x) = x !== nothing && x.op == "load" && !local_ptr(lastword(args(x)[2]))
        kind[n] = if global_load(src)  # a loaded index, or a field of a loaded tuple
            t = rtype(src)
            endswith(t, "*") || t == "ptr" ? K_NONE :
            (nrefs[n] = [ref(lastword(args(src)[2]))]; K_LOAD)
        elseif op in ("extractvalue", "load")
            K_UNK  # a kernel-argument field or its local copy
        elseif op == "inttoptr" && global_load(get(def, valname(castsrc(r)), nothing))
            K_NONE  # a loaded pointer
        elseif op == "call"
            callee =
                something(match(r"@([\w.$]+)", r.rest), (; captures = [""])).captures[1]
            m = match(r"asm[^\"]*\"mov\.u32 \$0, %(\w+)", r.rest)
            m === nothing || (callee = "sreg." * replace(m[1], "tid" => "tid.x"))
            if occursin("sreg.laneid", callee)
                K_LANEID
            elseif occursin("sreg.tid.x", callee)
                K_TID
            elseif occursin(r"sreg\.(tid|ctaid)", callee)
                K_ZERO
            elseif occursin("sreg.", callee)
                K_UNK
            elseif (mm = match(r"llvm\.([su])(min|max)\.", callee)) !== nothing
                inner = replace(r.rest, r"^.*?\(" => "", r"\)[^)]*$" => "")
                nrefs[n] = [ref(lastword(q)) for q in splitargs(inner)]
                mm[2] == "min" ? (mm[1] == "u" ? K_UMIN : K_SMIN) :
                (mm[1] == "u" ? K_UMAX : K_SMAX)
            else
                K_NONE
            end
        elseif op == "phi"
            nrefs[n] = [ref(m[1]) for m in eachmatch(r"\[\s*([^,]+),", r.rest)]
            labs = [
                String(strip(m[1], '"')) for
                m in eachmatch(r"\[\s*[^,]+,\s*%(\"[^\"]+\"|[-\w.$]+)\s*\]", r.rest)
            ]
            # a merge of forward paths (no incoming block at or after its own)
            all(l -> label_of[l] < r.block, labs) && (philab[n] = (r.block, labs))
            K_PHI
        elseif op in RENAMES || op in ("zext", "sext", "trunc")
            nrefs[n] = [ref(castsrc(r))]
            K_CAST
        elseif op == "getelementptr"
            ops(2:length(a))
            t = a[1]
            gfail[n] = typemax(Int)
            for i in 3:length(a)  # later indices step into array element types
                if i > 3
                    m = match(r"^\[\d+ x (.+)\]$", t)
                    m === nothing ? (gfail[n] = min(gfail[n], i)) : (t = m[1])
                end
                push!(steps[n], typesize(t))
            end
            K_GEP
        elseif op == "select"
            ops(1:3)
            lastword(a[2]) in ("undef", "poison") ? K_SEL3 :
            lastword(a[3]) in ("undef", "poison") ? K_SEL2 : K_SELECT
        elseif haskey(BINOPS, op)
            ops(1:2)
            pr = split(r.rest)[1]  # (other predicates compare as uge/sge)
            op != "icmp" ? BINOPS[op] :
            get(ICMPS, pr, startswith(pr, "u") ? ICMPS["uge"] : ICMPS["sge"])
        else
            K_NONE
        end
    end

    # A loaded index (`gather = :local`) is its 1-based position in an identity map (e.g. a
    # structured mesh's element, vertex or face list): lanes that read one entry share it (a
    # broadcast), and
    # consecutive entries give consecutive indices. The position is the first varying index
    # of the load's address.
    function loadpos(i, w, d)
        while i > 0 && (kind[i] == 0 && decode!(i); kind[i] == K_CAST)
            i = nrefs[i][1]
        end
        if !(i > 0 && kind[i] == K_GEP)
            x = lanev(i, w, d)
            return x === nothing || !invariant(x[1]) ? nothing : UNK
        end
        y = nothing
        for j in 2:length(nrefs[i])
            y = lanev(nrefs[i][j], w, d)
            (y === nothing || !invariant(y[1])) && break
        end
        return y === nothing ? nothing : y[1] .+ 1
    end

    # Lanes' paths into block `b`: each lane follows the branches from the kernel's entry,
    # evaluated with its own values, to its first arrival at `b`; the label of the block it
    # arrives from, by lane, or nothing (an unknown or inexact branch, a revisited block, an
    # exit, more than 64 blocks). A phi that merges forward paths takes each lane's value.
    tparse = Dict{Int, Any}()  # block => (condition or switch value ref, labels, cases)
    labels = K.labels
    pmemo = Dict{Tuple{Int, Int}, Union{Nothing, Vector{String}}}()
    function lanepreds(b, w, d)
        haskey(pmemo, (b, w)) && return pmemo[(b, w)]
        if isempty(tparse)
            for r in insts
                r.op in ("br", "switch") || continue
                labs = [
                    strip(m[1], '"') for
                    m in eachmatch(r"label %(\"[^\"]+\"|[-\w.$]+)", r.rest)
                ]
                c = length(labs) == 1 ? 0 : ref(lastword(splitargs(r.rest)[1]))
                o = something(findfirst('[', r.rest), lastindex(r.rest) + 1)
                cases = [
                    parse(Int128, m[1]) for
                    m in eachmatch(r"i\d+ (-?\d+),\s*label", r.rest[o:end])
                ]
                tparse[r.block] = (c, labs, cases)
            end
        end
        pmemo[(b, w)] = nothing
        out = fill("", 32)
        for l in 1:32
            x = 1
            been = Set{Int}()
            for _ in 1:64
                push!(been, x)
                t = get(tparse, x, nothing)
                t === nothing && return nothing
                (c, labs, cases) = t
                nx = labs[1]
                if c != 0
                    v = lanev(c, w, d + 1)
                    (v === nothing || !v[2]) && return nothing
                    m = c > 0 && nbits[c] > 0 ? nbits[c] : Int128(2)^64
                    j = findfirst(q -> mod(q, m) == mod(v[1][l], m), cases)
                    nx =
                        isempty(cases) ? labs[v[1][l] != 0 ? 1 : 2] :
                        j === nothing ? labs[1] : labs[j + 1]
                end
                y = label_of[nx]
                y == b && (out[l] = labels[x]; break)
                y in been && return nothing
                x = y
            end
            isempty(out[l]) && return nothing
        end
        return pmemo[(b, w)] = out
    end

    # A phi that merges forward paths, with `inner_extent = 1`: each lane's incoming value
    function phi_lanes(i, w, d)::LaneVal
        (b, labs) = philab[i]
        rs = nrefs[i]
        allequal(rs) && return nothing
        pl = lanepreds(b, w, d + 1)
        js = pl === nothing ? nothing : indexin(pl, labs)
        (js === nothing || any(isnothing, js)) && return nothing
        xs = Dict(j => lanev(rs[j], w, d + 1) for j in unique(js))
        any(isnothing, values(xs)) && return nothing
        return ([xs[js[l]][1][l] for l in 1:32], all(y -> y[2], values(xs)))
    end

    # Whether a node's value can differ with `inner_extent = 1`: it depends on a run-time
    # divisor or on a merge of forward paths (or lies on a cycle); the others take their
    # values from warps 1-16.
    function divdep(i)
        i <= 0 && return false
        dep1[i] in (1, 3) && return true
        dep1[i] == 2 && return false
        dep1[i] = 3
        kind[i] == 0 && decode!(i)
        y =
            kind[i] == K_UNK ? names[i] in divisors :
            haskey(philab, i) || any(divdep, nrefs[i])
        dep1[i] = y ? 1 : 2
        return y
    end

    # (value per lane, exact) of operand ref `i` in warp w, or nothing; a value seen again
    # while it is evaluated (a loop-carried value) is nothing, so a phi takes its first
    # incoming value that does not depend on itself
    function lanev(i, w, d = 0)::LaneVal
        i < 0 && return (fill(consts[-i], 32), true)
        i == 0 && return nothing
        w > 16 && !Base.inferencebarrier(divdep)(i)::Bool && return lanev(i, w - 16, d)
        lseen[w][i] && return vals[w][i]
        d > 48 && return nothing
        lseen[w][i] = true  # vals[w][i] = nothing while it is evaluated
        kind[i] == 0 && decode!(i)
        k = kind[i]
        rs = nrefs[i]
        bits = nbits[i]
        ev(j) = lanev(rs[j], w, d + 1)::LaneVal  # (a recursive closure is boxed: assert its type)
        u(x) = mod.(x, bits)  # unsigned view
        function both(f, exact_only = false)
            p = ev(1)
            q = ev(2)
            (p === nothing || q === nothing || exact_only && !(p[2] && q[2])) &&
                return nothing
            return (f(p[1], q[1]), p[2] && q[2])
        end
        v = if k == K_UNK
            w > 16 && names[i] in divisors ? (ONE, true) : (UNK, false)
        elseif k == K_NONE
            nothing
        elseif k == K_LOAD
            x = loadpos(rs[1], w, d + 1)
            x === nothing ? nothing : (x, false)
        elseif k == K_LANEID
            (Int128.(0:31), true)
        elseif k == K_TID
            (LANES[mod1(w, 16)], true)
        elseif k == K_ZERO
            (ZERO, true)
        elseif k in (K_SMIN, K_SMAX, K_UMIN, K_UMAX)
            xs = [ev(j) for j in eachindex(rs)]
            f = k in (K_SMIN, K_UMIN) ? min : max
            uns = k in (K_UMIN, K_UMAX)
            any(isnothing, xs) ? nothing :
            all(x -> x[2], xs) ? (f.((uns ? u(x[1]) : x[1] for x in xs)...), true) :
            all(x -> invariant(x[1]), xs) ? (UNK, false) : nothing
        elseif k == K_PHI
            x =
                w > 16 && haskey(philab, i) ?
                Base.inferencebarrier(phi_lanes)(i, w, d)::LaneVal : nothing
            for j in eachindex(rs)
                x === nothing || break
                x = ev(j)
            end
            x
        elseif k == K_CAST
            ev(1)
        elseif k == K_GEP
            x = ev(1)
            for j in 2:length(rs)
                x === nothing && break
                y = ev(j)
                j + 1 >= gfail[i] && (y = nothing)
                x =
                    y === nothing ? nothing :
                    (x[1] .+ y[1] .* steps[i][j - 1], x[2] && y[2])
            end
            x
        elseif k == K_ADD
            both(+)
        elseif k == K_SUB
            both(-)
        elseif k == K_MUL
            x = ev(1)
            y = ev(2)
            if any(z -> z !== nothing && z[2] && all(==(0), z[1]), (x, y))
                (ZERO, true)
            elseif x === nothing || y === nothing
                nothing
            elseif invariant(x[1]) || invariant(y[1]) || x[2] && y[2]
                (x[1] .* y[1], x[2] && y[2])
            end
        elseif k == K_SHL
            x = ev(1)
            y = ev(2)
            x === nothing || y === nothing || !(invariant(y[1]) || x[2] && y[2]) ?
            nothing : (x[1] .<< y[1], x[2] && y[2])
        elseif k == K_AND
            x = ev(1)
            y = ev(2)
            m = y !== nothing && y[2] && invariant(y[1]) ? y[1][1] : nothing

            # identity masks (all low bits of the type) pass; other masks need exact
            # operands
            m !== nothing && m >= 2^32 - 1 && ispow2(m + 1) ? x :
            both((x, y) -> u(x) .& u(y), true)
        elseif k == K_XOR
            both((x, y) -> xor.(u(x), u(y)), true)
        elseif k == K_LSHR
            both((x, y) -> u(x) .>> y, true)
        elseif k == K_ASHR
            both((x, y) -> x .>> y, true)
        elseif k in (K_UDIV, K_SDIV, K_UREM, K_SREM)
            x = ev(1)
            y = ev(2)
            quot = k in (K_UDIV, K_SDIV)
            if x === nothing || y === nothing
                nothing
            elseif !y[2] && invariant(y[1])  # by a run-time extent: at least a warp wide
                (quot ? UNK : x[1], false)
            elseif x[2] && y[2] && !any(==(0), y[1])
                signed = k in (K_SDIV, K_SREM)
                (
                    (quot ? div : rem).(signed ? x[1] : u(x[1]), signed ? y[1] : u(y[1])),
                    true,
                )
            end
        elseif k >= K_EQ  # icmp; a run-time extent compares as its 2^20 with small constants
            f, uns = CMPS[k - K_EQ + 1]
            x = ev(1)
            y = ev(2)
            ext(a, b) =
                !a[2] && a[1] == UNK && b[2] && all(v -> abs(v) < UNK[1] ÷ 2, b[1])
            x === nothing || y === nothing ||
            !(x[2] && y[2] || ext(x, y) || ext(y, x)) ? nothing :
            (Int128.(uns ? f.(u(x[1]), u(y[1])) : f.(x[1], y[1])), true)
        elseif k == K_SEL3
            ev(3)
        elseif k == K_SEL2
            ev(2)
        else  # select
            c = ev(1)
            x = ev(2)
            y = ev(3)
            if x === nothing || y === nothing
                nothing
            elseif c !== nothing && c[2]
                (ifelse.(c[1] .== 1, x[1], y[1]), x[2] && y[2])
            elseif x[1] == y[1]
                (x[1], x[2] && y[2])
            elseif invariant(x[1]) && invariant(y[1])
                (UNK, false)
            end
        end
        return vals[w][i] = v
    end
    function kinds(i)  # node kinds an operand ref depends on
        found = Set{Int8}()
        visited = Set{Int}()
        todo = [i]
        while !isempty(todo)
            j = pop!(todo)
            (j <= 0 || j in visited) && continue
            push!(visited, j)
            kind[j] == 0 && decode!(j)
            push!(found, kind[j])
            append!(todo, nrefs[j])
        end
        return found
    end

    # Distinct values of a shuffle's member mask over the lanes of warp 1 (run-time extents
    # at 2^20: a mask of the lanes below an extent covers the warp); 0 if the mask is not an
    # evaluable function of the lane (e.g. loaded)
    function maskpasses(i)
        memo = Dict{Int, Union{Nothing, Vector{Int128}}}()
        function mv(i, d)::Union{Nothing, Vector{Int128}}
            i < 0 && return fill(consts[-i], 32)
            (i == 0 || d > 32) && return nothing
            haskey(memo, i) && return memo[i]
            kind[i] == 0 && decode!(i)
            k = kind[i]
            e(j) = mv(nrefs[i][j], d + 1)
            u(x) = mod.(x, nbits[i])
            x = k in (K_SELECT, K_LOAD, K_PHI, K_GEP) || k < K_ADD ? nothing : e(1)
            y = k in (K_SELECT, K_LOAD, K_PHI, K_GEP) || k < K_ADD ? nothing : e(2)
            v = if k == K_UNK
                fill(Int128(2)^20, 32)
            elseif k in (K_LANEID, K_TID)
                Int128.(0:31)
            elseif k == K_ZERO
                zeros(Int128, 32)
            elseif k in (K_CAST, K_SEL2, K_SEL3)
                e(k == K_CAST ? 1 : k == K_SEL2 ? 2 : 3)
            elseif k == K_SELECT
                c, a, b = e(1), e(2), e(3)
                c === nothing || a === nothing || b === nothing ? nothing :
                ifelse.(c .!= 0, a, b)
            elseif k in (K_SMIN, K_SMAX, K_UMIN, K_UMAX)
                xs = [mv(j, d + 1) for j in nrefs[i]]
                any(isnothing, xs) ? nothing :
                (k in (K_SMIN, K_UMIN) ? min : max).(
                    (k >= K_UMIN ? u(z) : z for z in xs)...,
                )
            elseif x === nothing || y === nothing || k == K_LOAD
                nothing
            elseif k <= K_ASHR
                s = clamp.(y, 0, 127)
                k == K_ADD ? x .+ y :
                k == K_SUB ? x .- y :
                k == K_MUL ? x .* y :
                k == K_SHL ? u(x .<< s) :
                k == K_AND ? u(x) .& u(y) :
                k == K_XOR ? xor.(u(x), u(y)) : k == K_LSHR ? u(x) .>> s : x .>> s
            elseif k <= K_SREM
                sg = k in (K_SDIV, K_SREM)
                any(==(0), y) ? nothing :
                (k in (K_UDIV, K_SDIV) ? div : rem).(sg ? x : u(x), sg ? y : u(y))
            else
                (f, uns) = CMPS[k - K_EQ + 1]
                Int128.(uns ? f.(u(x), u(y)) : f.(x, y))
            end
            return memo[i] = v
        end
        v = mv(i, 0)
        return v === nothing ? 0 : length(unique(mod.(v, Int128(2)^32)))
    end
    unknown_footprints = Ref(0)
    pend = Tuple{Int, String, Int, Bool, NTuple{4, Float32}}[]  # global accesses

    # `Kernel.extent1`: footprints with `inner_extent = 1` (warps 17-32) and L1 shares
    function fill_extent1!(ext1)
        wo = isempty(divisors) ? 0 : 16
        for (k, ptr, nby, load, f) in pend
            f1 = wo == 0 ? f : footprint(ptr, nby, wo)
            f1 = Float32.((f1[1], load ? 0 : f1[2], f1[3], f1[4]))
            (load || f1 != f) && (ext1[k] = (f1..., load ? l1share(ptr, nby, wo) : NaN32))
        end
        return ext1
    end

    # L1 fraction of a load with `inner_extent = 1`: the share of its sectors that earlier
    # loads (in program order) through the same base pointer, or the same load in earlier
    # warps, touched among the first 16 warps of a block; NaN if its lanes' addresses are
    # unknown.
    touched = Dict{Int, Set{Int128}}()  # base pointer node => sectors
    function l1share(ptr, bytes, wo)
        i = r = ref(ptr)
        while r > 0 && (kind[r] == 0 && decode!(r); kind[r] in (K_GEP, K_CAST))
            r = nrefs[r][1]
        end
        S = get!(Set{Int128}, touched, r)
        hit = n = 0
        ss = Int128[]
        for w in eachindex(LANES)
            x = lanev(i, w + wo)
            x === nothing && return NaN32
            empty!(ss)
            foreach(a -> append!(ss, (a >> 5):((a + bytes - 1) >> 5)), x[1])
            for s in unique!(sort!(ss))
                n += 1
                s in S ? (hit += 1) : push!(S, s)
            end
        end
        return Float32(hit / n)
    end

    # (sectors, partially written sectors, L1 bank passes, 128-byte lines) of one warp access,
    # zeros if the lane pattern is unknown (sectors < 0: through a loaded index, see
    # `loadpos` and `gather`; -Inf: unknown): the mean over the first 16 warps of block 0
    # (tid.x = 32w + lane), whose start addresses take the offsets mod 32 B of consecutive
    # rows (a contiguous 252-byte run, e.g. a 63-level Float32 column: most warp accesses
    # touch 5 sectors, two of them partially). The L1
    # data array has the 32 4-byte banks of shared memory: words of different lines in one
    # bank take separate passes, and a pass serves at most 128 B, so 8- and 16-byte accesses
    # of more than 128 B run in phases of 16 and 8 lanes: 2 rows of 16 lanes 256 B apart
    # (e.g. columns of 64 Float32 levels) take 16 passes, 512 B apart (64 Float64 levels) 32
    # (16 per half-warp), 504 B apart 2.
    function footprint(ptr, bytes, wo = 0)
        n = p = l = 0
        nw = 0.0
        i = ref(ptr)
        sgn = K_LOAD in kinds(i) ? -1 : 1
        perbank = zeros(Int, 32)
        words = Int128[]
        phase = clamp(128 ÷ bytes, 1, 32)  # lanes per pass
        function passes(x, r)  # the most distinct words in one bank among lanes r
            fill!(perbank, 0)
            empty!(words)
            for a in view(x, r), q in fld(a, 4):fld(a + bytes - 1, 4)
                push!(words, q)
            end
            foreach(q -> perbank[mod(q, 32) + 1] += 1, unique!(sort!(words)))
            return maximum(perbank)
        end
        for w in eachindex(LANES)
            x = lanev(i, w + wo)::LaneVal
            x === nothing && (
                wo == 0 && (unknown_footprints[] += 1);
                return (sgn < 0 ? -Inf : 0, 0, 0, 0)
            )
            hi = typemin(Int128)  # last byte of the lanes' byte ranges so far (in order)
            sec = line = typemin(Int128)
            m = Int128(0)  # bytes of sector `sec` touched
            # distinct words in one bank are in distinct lines; more than 32 words (128 B) run
            # in phases
            p1 = passes(x[1], 1:32)
            nw +=
                length(words) <= 32 ? p1 :
                sum(j -> passes(x[1], j:min(j + phase - 1, 32)), 1:phase:32)
            for a in sort(x[1])
                lo = max(a, hi + 1)
                e = a + bytes - 1
                lo > e && continue
                for s in (lo >> 5):(e >> 5)  # each sector, and whether it is touched in full
                    s == sec ||
                        (
                            sec > typemin(Int128) && (n += 1; p += m < 32);
                            (sec, m) = (s, Int128(0))
                        )
                    s >> 2 == line || (l += 1; line = s >> 2)
                    m += min(e, (s << 5) + 31) - max(lo, s << 5) + 1
                end
                hi = e
            end
            sec > typemin(Int128) && (n += 1; p += m < 32)
        end
        return (sgn * n / 16, p / 16, nw / 16, l / 16)
    end

    # Reuse across loop iterations: the bytes by which every lane's address of an access
    # in a loop advances per iteration (a lane that walks an array, e.g. a column level by
    # level), 0 if it
    # is the same in every iteration, with the loop's header, or nothing. The address may
    # depend on one phi of the innermost loop that contains the access, the loop counter
    # (a phi that adds a constant to itself), and on values defined outside that loop,
    # which do not change in it.
    function counter_step(n)
        kind[n] == 0 && decode!(n)
        kind[n] == K_PHI || return nothing
        for j in nrefs[n]
            j > 0 || continue
            kind[j] == 0 && decode!(j)
            kind[j] in (K_ADD, K_GEP) && length(nrefs[j]) == 2 || continue
            a, b = nrefs[j]
            b < 0 && a == n && return consts[-b] * (kind[j] == K_GEP ? steps[j][1] : 1)
        end
        return nothing
    end
    function dstep(i, iv, s, memo)::Union{Nothing, Int128}
        i <= 0 && return Int128(0)
        i == iv && return s
        haskey(memo, i) && return memo[i]
        memo[i] = Int128(0)  # a value on a cycle through itself: invariant unless shown not
        kind[i] == 0 && decode!(i)
        k = kind[i]
        rs = nrefs[i]
        d(j) = dstep(rs[j], iv, s, memo)::Union{Nothing, Int128}
        v = if k == K_CAST
            d(1)
        elseif k == K_GEP
            x = d(1)
            for j in 2:length(rs)
                y = d(j)
                (x === nothing || y === nothing || y != 0 && j + 1 >= gfail[i]) &&
                    (x = nothing; break)
                x += y * steps[i][j - 1]
            end
            x
        elseif k == K_ADD || k == K_SUB
            x = d(1)
            y = d(2)
            x === nothing || y === nothing ? nothing : k == K_ADD ? x + y : x - y
        elseif k == K_MUL || k == K_SHL
            x = d(1)
            y = d(2)
            c =
                rs[2] < 0 ? consts[-rs[2]] :
                rs[1] < 0 && k == K_MUL ? consts[-rs[1]] : nothing
            x === nothing || y === nothing ? nothing :
            x == y == 0 ? Int128(0) :
            c === nothing ? nothing :
            k == K_SHL ? x << c : (rs[2] < 0 ? x : y) * c
        elseif (k == K_ASHR || k == K_LSHR) && rs[2] < 0  # exact for a multiple of 2^c
            x = d(1)
            c = consts[-rs[2]]
            x === nothing || x % (Int128(1) << c) != 0 ? nothing : x >> c
        elseif k == K_AND && rs[2] < 0 && consts[-rs[2]] >= 2^32 - 1 &&
               ispow2(consts[-rs[2]] + 1)  # an identity mask
            d(1)
        else  # nonlinear ops change only if an operand does
            all(j -> d(j) == 0, eachindex(rs)) ? Int128(0) : nothing
        end
        return memo[i] = v
    end
    loopbody = Dict{Int, BitSet}()  # natural loops: header => blocks (filled below)
    function reuse(ptr, blk)
        hs = [h for (h, body) in loopbody if blk in body]
        isempty(hs) && return nothing
        h = argmin(h -> length(loopbody[h]), hs)
        i = ref(ptr)
        phis = Int[]
        visited = Set{Int}()
        todo = [i]
        while !isempty(todo)
            j = pop!(todo)
            (j <= 0 || j in visited) && continue
            push!(visited, j)
            kind[j] == 0 && decode!(j)
            kind[j] == K_PHI && def[names[j]].block in loopbody[h] && push!(phis, j)
            append!(todo, nrefs[j])
        end
        isempty(phis) && return (0, h)
        p = phis[1]
        (length(phis) == 1 && def[names[p]].block == h) || return nothing
        s = counter_step(p)
        δ = s === nothing ? nothing : dstep(i, p, s, Dict{Int, Union{Nothing, Int128}}())
        return δ === nothing ? nothing : (Int(δ), h)
    end

    # Load/store vectorization (LLVM's LoadStoreVectorizer runs in NVPTX codegen, after this
    # IR): loads, or stores, of one block through the same base pointer and index prefix at
    # consecutive constant offsets become one 8/16-byte access (LDS.64/128, STG.128, ...)
    # when the run is aligned to its size and nothing that may alias (a store between loads,
    # a load between stores, a call, fence or barrier) intervenes. Members of a run cost no
    # issue slot. instruction => (carrier: first load / last store of the run, bytes of the
    # run)
    vec = Dict{Int, Tuple{Int, Int}}()
    let runs =
        # key => [(inst, offset, align)]
            Dict{Tuple{Int, String, String, String, String}, Vector{Tuple{Int, Int, Int}}}()
        function flush!(pred)
            for (k, run) in collect(runs)
                pred(k) || continue
                length(run) > 1 && foreach(
                    x ->
                        vec[x[1]] = (
                            run[k[2] == "load" ? 1 : end][1],
                            length(run) * typesize(k[3]),
                        ),
                    run,
                )
                delete!(runs, k)
            end
        end
        for (i, r) in enumerate(insts)
            (i > 1 && insts[i - 1].block != r.block) && flush!(_ -> true)
            if r.op in ("load", "store")
                a = splitargs(r.rest)
                t = r.op == "load" ? rtype(r) : firsttype(a[1])
                ptr = lastword(a[r.op == "load" ? 2 : 2])
                g = get(def, valname(ptr), nothing)
                sz = typesize(t)
                al = (
                    m = match(r"align (\d+)", r.rest);
                    m === nothing ? sz : parse(Int, m[1])
                )
                key = nothing
                off = 0
                if g !== nothing && g.op == "getelementptr" && lanes(t) == 1
                    ga = args(g)
                    c = cval(lastword(ga[end]))
                    elem = (
                        m = match(r"^\[\d+ x (.+)\]$", ga[1]);
                        m === nothing ? ga[1] : m[1]
                    )
                    c !== nothing && elem == t &&
                        (
                            key = (
                                r.block,
                                r.op,
                                t,
                                lastword(ga[2]),
                                join(ga[3:(end - 1)], ","),
                            );
                            off = Int(c) * sz
                        )
                end
                flush!(k -> k[2] != r.op)  # a store breaks load runs and vice versa
                key === nothing && continue
                run = get(runs, key, nothing)
                if run !== nothing && off == run[end][2] + sz &&
                   (length(run) + 1) * sz <= 16 && run[1][3] >= (length(run) + 1) * sz
                    push!(run, (i, off, al))
                else
                    run === nothing || flush!(==(key))
                    runs[key] = [(i, off, al)]
                end
            elseif r.op in ("fence", "atomicrmw", "cmpxchg") ||
                   r.op == "call" &&
                   !occursin(
                r"@llvm\.(assume|lifetime|dbg|fabs|fma|fmuladd|sqrt|floor|ceil|trunc|rint|round|minnum|maxnum|[su]min|[su]max|abs|copysign|ctpop|ctlz|cttz)|@__nv_",
                r.rest,
            )
                flush!(_ -> true)
            end
        end
        flush!(_ -> true)
    end
    fused = Set(
        v for
        r in insts if r.op in ("fadd", "fsub") && occursin(r"contract|fast", r.rest)
        for v in operands(r) if (
            d = get(def, v, nothing);
            d !== nothing && d.op == "fmul" &&
                occursin(r"contract|fast", d.rest) && length(get(users, v, [])) == 1
        )
    )
    seen = Set{Tuple{Int, String, Vector{String}}}()

    note!(msg) = msg in K.notes || push!(K.notes, msg)
    callnotes = Pair{Int, String}[]  # block => note, kept unless the block is cold

    # Registers of the result of r (first operand type t, sources src): predicates, folded
    # values and free ops of register-free sources (kernel-argument fields: constant bank)
    # take none.
    rwidth(r, t, src) =
        r.op in ("icmp", "fcmp") || r.dst in fused || r.dst in foldable ||
        r.op in FREE_OPS && r.op != "phi" && all(v -> K.width[v] == 0, src) ? 0 :
        r.op in CASTS ? regs(lastword(r.rest)) :
        r.op == "select" ? regs(firsttype(args(r)[2])) :
        r.op == "getelementptr" ? 2 : regs(t)

    # Libdevice calls (`__nv_*`) are pure and linked before optimization in the real
    # compile: one with constant arguments folds, and one that repeats a call of a
    # dominating block with the same arguments reuses its result (GVN). Dominator sets by
    # iterative data flow over the CFG.
    preds = [Int[] for _ in K.labels]
    for r in insts, x in eachmatch(r"label %(\"[^\"]+\"|[-\w.$]+)", r.rest)
        r.op in ("br", "switch") && push!(preds[label_of[strip(x[1], '"')]], r.block)
    end
    first_copy = Dict{String, String}()  # callee => prefix of its first copy
    for (b, e, rets, f, pre) in links
        K.calls[label_of[b]] = (label_of[e], [label_of[x] for x in rets])
        push!(preds[label_of[e]], label_of[b])
        p0 = get!(first_copy, f, pre)
        p0 == pre && continue
        for (l, i) in label_of
            startswith(l, pre) &&
                (j = get(label_of, p0 * l[(ncodeunits(pre) + 1):end], 0)) > 0 &&
                (K.alias[i] = j)
        end
    end
    dom = [BitSet(b == 1 ? (1:1) : eachindex(K.labels)) for b in eachindex(K.labels)]
    changed = true
    while changed
        changed = false
        for b in 2:length(dom)
            d = push!(reduce(intersect, (dom[p] for p in preds[b]); init = dom[b]), b)
            d == dom[b] || (dom[b] = d; changed = true)
        end
    end
    # Loop strength reduction (LLVM's LSR in codegen, ptxas address folding): an address
    # in a loop that advances by a constant per iteration of its loop counter (see
    # `reuse`) is a pointer that the loop increments, one 64-bit add (one 32-bit add
    # for shared memory), shared by the addresses of a block and base that differ from it
    # by a constant (immediate offsets); the index arithmetic that only feeds such addresses
    # leaves the loop.
    strided = Dict{String, Union{Nothing, Vector{Int128}}}()  # GEP => offset from its base
    pointers = Dict{Tuple{Int, String}, Vector{Union{Nothing, Vector{Int128}}}}()  # (block, base) => offsets
    function loops!()  # natural loops (`loopbody`: back edges b => h, h dominating b), then LSR
        for h in eachindex(preds), q in preds[h]
            h in dom[q] || continue
            local lb = push!(get!(BitSet, loopbody, h), h)
            local todo = [q]
            while !isempty(todo)
                local x = pop!(todo)
                x in lb || (push!(lb, x); append!(todo, preds[x]))
            end
        end
        for r in insts
            r.op == "getelementptr" && length(args(r)) > 2 &&
            !all(x -> cval(lastword(x)) !== nothing, args(r)[3:end]) &&
            !local_ptr("%" * r.dst) &&
            something(reuse("%" * r.dst, r.block), (0,))[1] != 0 ||
                continue
            local i = ref("%" * r.dst)
            local off = ZERO
            for j in 2:length(nrefs[i])
                local y = j + 1 < gfail[i] ? lanev(nrefs[i][j], 1) : nothing
                (y === nothing || !y[2]) && (off = nothing; break)
                off = off .+ y[1] .* steps[i][j - 1]
            end
            strided[r.dst] = off
        end
        local grow = !isempty(strided)
        while grow
            grow = false
            for r in insts
                r.op in
                ("add", "sub", "mul", "shl", "or", "sext", "zext", "trunc", "ashr",
                    "lshr", "and") && !(r.dst in foldable) || continue
                local us = get(users, r.dst, IRInst[])
                !isempty(us) &&
                all(
                    u ->
                        u.op == "getelementptr" && haskey(strided, u.dst) &&
                        lastword(args(u)[2]) != "%" * r.dst || u.dst in foldable,
                    us,
                ) && something(reuse("%" * r.dst, r.block), (0,))[1] != 0 || continue
                push!(foldable, r.dst)
                grow = true
            end
        end
        return nothing
    end
    loops!()

    # a strength-reduced address: an immediate offset from a pointer of its block and base, or
    # a new pointer
    function lsr_gep!(r, base, dst, src)
        local off = strided[r.dst]
        local ps = get!(pointers, (r.block, base), Union{Nothing, Vector{Int128}}[])
        off !== nothing && any(y -> y !== nothing && allequal(off .- y), ps) &&
            return emit!(K, :FREE, dst, src)
        push!(ps, off)
        addrspace(r.rest) == 3 && return emit!(K, :IADD, dst, Int32[])
        return emit!(K, :IADD, dst, [emit!(K, :IADD, newval!(K), Int32[])])
    end

    pure = Dict{String, Tuple{Int, Int32}}()  # call => (block, result)
    function repeat_call(r, src, dst)
        occursin("@__nv_", r.rest) || return nothing
        isempty(src) && return Int32[]
        p = get(pure, r.rest, nothing)
        p !== nothing && p[1] in dom[r.block] && return Int32[p[2]]
        pure[r.rest] = (r.block, dst)
        return nothing
    end

    # A 64-bit value split into or assembled from 32-bit halves (a shuffled double) is a
    # register pair in SASS: `lshr`/`shl` by 32 and the `or` of the halves take no instruction.
    defof(v) = get(def, valname(v), nothing)
    zext32(v) = (d = defof(v); d !== nothing && d.op == "zext" && rtype(d) == "i32")
    hi32(v) = (
        d = defof(v);
        d !== nothing && d.op == "shl" && cval(args(d)[2]) == 32 &&
            zext32(lastword(args(d)[1]))
    )
    halfmove(r) =
        rtype(r) == "i64" && (
            r.op == "lshr" && cval(args(r)[2]) == 32 ||
            r.op == "shl" && hi32("%" * r.dst) ||
            r.op == "or" &&
            (
                a = lastword.(args(r));
                hi32(a[1]) && zext32(a[2]) || hi32(a[2]) && zext32(a[1])
            )
        )
    start = 1
    checked = Set{Tuple{Int, String}}()  # (block, mask) of shuffle mask checks
    for (i, r) in enumerate(insts)
        op = r.op
        m = op in RENAMES && !isempty(r.dst) ? match(VALUE, r.rest) : nothing
        if m !== nothing  # alias of its operand
            ids[r.dst] = vid(m[1])
            dst = Int32(0)
            src = Int32[]
        else
            dst = isempty(r.dst) ? Int32(0) : vid(r.dst)
            src = Int32[vid(v) for v in operands(r)]
        end
        t = rtype(r)
        n = lanes(t)
        w2 = wide(t) ? 2 : 1
        dst > 0 && (K.width[dst] = rwidth(r, t, src))
        op == "phi" && (
            K.phipred[length(K.cls) + 1] = [
                label_of[strip(x[2], '"')] for x in
                eachmatch(
                    r"\[\s*(%(?:\"[^\"]+\"|[-\w.$]+))\s*,\s*%(\"[^\"]+\"|[-\w.$]+)\s*\]",
                    r.rest,
                )
            ]
        )
        chain(key) = emit_chain!(K, EXPANSIONS[key]..., src, dst)
        simple(c, m = 1) =
            for j in 1:m
                emit!(K, c, j == m ? dst : newval!(K), src)
            end
        op == "alloca" && (K.frame += allocsize(r.rest))  # a local array or struct
        if op in FREE_OPS || r.dst in fused || r.dst in foldable || halfmove(r) ||
           op == "br" && !startswith(r.rest, "i1")
            simple(:FREE)
        elseif op in ("fadd", "fsub", "fmul", "frem")
            simple(w2 == 2 ? :FP64 : :FP32, n)
        elseif op == "fdiv"
            one = occursin(r"^1\.0+e\+00$|^1\.0$", args(r)[1])
            key =
                w2 == 2 ? (one ? "rcp.f64" : "fdiv.f64") :
                occursin(r"\b(fast|arcp|afn)\b", r.rest) ? "fast.f32" :
                one ? "rcp.f32" : "fdiv.f32"
            foreach(_ -> chain(key), 1:n)
        elseif op in ("fcmp", "icmp")
            a = args(r)
            op == "fcmp" ? simple(w2 == 2 ? :FP64 : :SETP, n) :
            w2 == 2 && !(narrow(lastword(a[1])) && narrow(a[2])) ?
            emit_chain!(K, (:SETP, :SETP), (), src, dst) : simple(:SETP)
        elseif op in ("add", "sub")
            w2 == 2 ? emit_chain!(K, (:IADD, :IADD), (), src, dst) : simple(:IADD, n)
        elseif op == "mul"
            a = args(r)
            w2 == 1 ? simple(:IMAD, n) :
            narrow(lastword(a[1])) && narrow(a[2]) ? simple(:IMADW) :
            emit_chain!(K, (:IMADW, :IMAD), (:IMAD,), src, dst)
        elseif op in ("and", "or", "xor", "shl", "lshr", "ashr", "select", "sext")
            simple(:ALU, (wide(op == "select" ? firsttype(args(r)[2]) : t) ? 2 : 1) * n)
        elseif op in ("udiv", "sdiv", "urem", "srem")
            c = cval(args(r)[2])
            s = w2 == 2 ? "i64" : "i32"
            c !== nothing && ispow2(abs(c)) ? simple(:ALU, w2) :
            chain(c === nothing ? "div.$s" : (op[2:end] == "rem" ? "remc." : "divc.") * s)
        elseif op in ("sitofp", "uitofp", "fptosi", "fptoui", "fpext", "fptrunc")
            simple(:CONV, n)
        elseif op == "getelementptr"
            a = args(r)
            base = lastword(a[2])
            acc = Tuple{String, Bool}[]
            foreach(x -> leaves!(acc, lastword(x)), a[3:end])
            key = (r.block, base, sort(first.(acc)))
            if haskey(strided, r.dst)
                lsr_gep!(r, base, dst, src)
            elseif isempty(acc) || key in seen
                simple(:FREE)
            else
                push!(seen, key)
                prev = startswith(base, "%") ? Int32[vid(valname(base))] : Int32[]
                for (j, (l, nar)) in enumerate(acc)
                    lv = vid(valname(l))
                    fin = j == length(acc)
                    d = emit!(K, :IMADW, fin && nar ? dst : newval!(K), [prev; lv])
                    nar || (d = emit!(K, :IMAD, fin ? dst : newval!(K), [d, lv]))
                    prev = [d]
                end
            end
        elseif op in ("load", "store", "atomicrmw", "cmpxchg")
            mem = op in ("load", "store")
            ptr = mem ? lastword(splitargs(r.rest)[2]) : ""
            as = mem ? ptrspace(strip(splitargs(r.rest)[2])) : addrspace(r.rest)
            as == 0 && mem && (as = origin_space(ptr))
            c =
                !mem ? (as == 3 ? :LDS : :ATOM) :
                op == "load" ? (as == 3 ? :LDS : as == 5 ? :LDL : as == 4 ? :LDC : :LDG) :
                (as == 3 ? :STS : as == 5 ? :STL : :STG)
            v = get(vec, i, nothing)
            c === :LDG && op == "load" && table_ptr(ptr) &&
                push!(K.tables, length(K.cls) + 1)
            if c === :LDG && op == "load" && (v === nothing || v[1] == i)
                ru = reuse(ptr, r.block)
                ru !== nothing && abs(ru[1]) < 128 && (K.walks[length(K.cls) + 1] = ru)
            end
            fp(bytes) = c in (:LDG, :STG) ? footprint(ptr, bytes) : (0, 0, 0, 0)
            f =
                v === nothing ? fp(mem ? typesize(t) : 4) :
                v[1] == i ? fp(v[2]) : (0, 0, 0, 0)
            c in (:LDG, :STG) && (v === nothing || v[1] == i) &&
                push!(
                    pend,  # see `fill_extent1!`
                    (length(K.cls) + 1, ptr, v === nothing ? (mem ? typesize(t) : 4) : v[2],
                        c === :LDG, Float32.(f)),
                )

            # member of a vectorized run: its value comes with the carrier access
            if v !== nothing && v[1] != i
                emit!(K, :FREE, dst, op == "load" ? Int32[vid(insts[v[1]].dst)] : Int32[])

                # carrier: one access of the run's bytes (a store waits for every stored
                # value)
            elseif v !== nothing
                members =
                    op == "store" ? [j for (j, (l, _)) in vec if l == i && j != i] : Int[]
                emit!(
                    K,
                    c,
                    dst,
                    unique(
                        Int32[
                            src
                            (vid(x) for j in members for x in operands(insts[j]))...
                        ],
                    );
                    bytes = v[2],
                    sectors = f[1],
                    partial = c == :STG ? f[2] : 0,
                    waves = f[3],
                    lines = f[4],
                )
            else
                emit!(
                    K,
                    c,
                    dst,
                    src;
                    bytes = mem ? typesize(t) : 4,
                    sectors = mem ? f[1] : 0,
                    partial = c == :STG ? f[2] : 0,
                    waves = mem ? f[3] : 0,
                    lines = mem ? f[4] : 0,
                )
            end
        elseif op == "call" && (p = repeat_call(r, src, dst)) !== nothing
            emit!(K, :FREE, dst, p)
        elseif op == "call"
            callee =
                something(match(r"@([\w.$]+)", r.rest), (; captures = [""])).captures[1]
            # Inline PTX prefetches (NVPTX drops the generic intrinsic in instruction
            # selection before LLVM 20).
            (m = match(r"asm sideeffect \"(prefetch\.global\.L[12])", r.rest)) ===
            nothing || (callee = m[1])
            j = findfirst(p -> occursin(p[1], callee), CALLS)
            j === nothing && push!(
                callnotes,
                r.block =>
                    (isempty(callee) ? "inline asm" : "call @$callee") *
                    " modeled as one branch-class op",
            )
            h = j === nothing ? :BRA : CALLS[j][2]
            f64 = w2 == 2 || occursin("f64", callee)
            if h == "minmax"
                f64 ?
                chain(
                    occursin(r"[su](min|max)|abs", callee) ? "minmax.i64" : "minmax.f64",
                ) : simple(:ALU, n)
            elseif h == "libm" && haskey(EXPANSIONS, "libm." * callee[6:end])
                chain("libm." * callee[6:end])
            elseif h in ("sqrt", "rcp", "libm")
                h == "libm" && note!(
                    "approximate expansion for libdevice call $callee (validate with SASS)",
                )
                chain(h * (f64 || endswith(callee, "sqrt") ? ".f64" : ".f32"))
            elseif h isa String
                chain(h)

                # a prefetch moves one line per warp: 4 bytes per thread
            elseif h === :FMA || h === :LDG || h === :PREF
                h === :FMA ? simple(f64 ? :FP64 : :FP32, n) :
                emit!(K, h, dst, src; bytes = h === :PREF ? 4 : typesize(t))
            elseif h === :SHFL && (
                mask = lastword(
                    splitargs(replace(r.rest, r"^.*?\(" => "", r"\)[^)]*$" => ""))[1],
                );
                startswith(mask, "%")
            )
                # A run-time member mask: ptxas checks that its lanes are converged (MATCH.ANY
                # and REDUX.OR of the mask, VOTEU.ANY, LOP3, then BRA.DIV to a slow path);
                # shuffles with one mask in one block share the check, and a shuffle whose
                # result is unused keeps only its check.
                dead = !haskey(users, r.dst)
                if dead || !((r.block, mask) in checked)
                    dead || push!(checked, (r.block, mask))
                    n0 = length(K.cls)
                    mid = Int32[vid(valname(mask))]
                    np = maskpasses(ref(mask))
                    np == 0 && note!(
                        "a shuffle member mask that is not a function of " *
                        "the lane: one MATCH.ANY pass",
                    )
                    mt = emit!(K, :ADU, newval!(K, 1), mid; waves = max(np, 1))
                    rx = emit!(K, :ADU, newval!(K), mid; waves = 1)
                    vt = emit!(K, :UNI, newval!(K), Int32[])
                    emit!(K, :BRA, 0, [emit!(K, :ALU, newval!(K), [mt, vt]), rx])
                    union!(K.syncs, (n0 + 1):length(K.cls))
                end
                dead ? simple(:FREE) : simple(:SHFL, w2)
            else
                simple(
                    h,
                    h in (:SHFL, :XU) && w2 == 2 || occursin("copysign", callee) ? 2 :
                    h in (:ALU, :CONV) ? n : 1,
                )
            end
        elseif op in ("br", "switch", "ret", "fence")
            simple(:BRA, op == "switch" ? count("label", r.rest) - 1 : 1)
        else
            note!("unknown IR op `$op` modeled as ALU")
            simple(:ALU)
        end
        if i == length(insts) || insts[i + 1].block != r.block
            push!(K.blocks, start:length(K.cls))
            start = length(K.cls) + 1
        end
    end

    unknown_footprints[] > 0 && note!(
        "$(unknown_footprints[]) global accesses with an address that is not a function of " *
        "the lane (indirect indexing): charged as coalesced",
    )

    # CFG, branch probabilities (`!prof`; 0 into `unreachable` paths), trip counts.
    term =
        Dict(r.block => r for r in insts if r.op in ("br", "switch", "ret", "unreachable"))
    for b in eachindex(K.labels)
        r = get(term, b, nothing)
        targets =
            r === nothing ? K.labels[(b + 1):min(b + 1, end)] :
            [strip(x[1], '"') for x in eachmatch(r"label %(\"[^\"]+\"|[-\w.$]+)", r.rest)]
        push!(K.succs, [label_of[x] for x in targets])
        w = r === nothing ? nothing : get(weights, r.meta, nothing)
        push!(
            K.probs,
            w !== nothing && length(w) == length(targets) ? w ./ sum(w) :
            fill(NaN, length(targets)),
        )
        cm = r === nothing ? nothing : match(r"^i1 %(\"[^\"]+\"|[-\w.$]+), label", r.rest)
        push!(K.conds, cm === nothing ? "" : strip(cm[1], '"'))
    end
    cold = [get(term, b, (; op = "")).op == "unreachable" for b in eachindex(K.labels)]
    propagate_cold!(cold, K.succs)
    for b in eachindex(K.labels), (i, s) in enumerate(K.succs[b])
        isnan(K.probs[b][i]) && cold[s] && !all(cold[K.succs[b]]) && (K.probs[b][i] = 0.0)
    end
    foreach(((b, msg),) -> cold[b] || note!(msg), callnotes)

    # A predicate of the thread index and run-time sizes only (no loaded value) splits the
    # thread range at run-time boundaries (e.g. vertex threads before face threads): whole
    # blocks take each side. A lane test that splits the lanes of every warp (e.g. the idle
    # lanes of 9 points in a 16-lane sub-block) makes each warp run both sides
    # until they reconverge (SIMT divergence, `reconverge!`).
    # A lane test that sends the warps of a block different ways (`tid % 64 == 0`: lane 0 of
    # warp 0 only) gives each warp its own side (`Kernel.wsplit`), both sides (SIMT) where its
    # lanes split.
    forks = Int[]
    side = zeros(UInt8, length(LANES))
    ntrue = zeros(UInt8, length(LANES))
    for b in eachindex(K.labels)
        isempty(K.conds[b]) && continue
        c = ref("%" * K.conds[b])
        ks = kinds(c)
        K_TID in ks && K_UNK in ks && !(K_NONE in ks || K_LOAD in ks) && push!(K.splits, b)
        length(K.succs[b]) == 2 || continue
        for w in eachindex(LANES)
            x = lanev(c, w)
            side[w] =
                x === nothing || !x[2] ? 0 : all(!=(0), x[1]) ? 1 : all(==(0), x[1]) ? 2 : 3
            ntrue[w] = x === nothing ? 0 : count(!=(0), x[1])
        end
        all(==(3), side) ? push!(forks, b) :
        all(>(0), side) && !all(==(side[1]), side) ? (K.wsplit[b] = (copy(side), 0)) :
        continue
        K.lanes[b] = copy(ntrue)
    end
    isempty(forks) && isempty(K.wsplit) || reconverge!(K, forks)
    kpreds = [Int[] for _ in K.labels]
    foreach(b -> foreach(x -> push!(kpreds[x], b), K.succs[b]), eachindex(K.labels))
    for (u, h) in back_edges(K, roots(K))
        haskey(term, u) || continue
        lbody = Set([h])
        work = [u]
        while !isempty(work)
            (x = pop!(work)) in lbody || (push!(lbody, x); append!(work, kpreds[x]))
        end
        # the latch's test, or, when the latch branches back always, the first test on the
        # straight path from the header (a header split at calls)
        r = term[u]
        b = h
        for _ in 1:16
            startswith(r.rest, "i1 ") && break
            r = get(term, b, r)
            s = K.succs[b]
            startswith(r.rest, "i1 ") || length(s) != 1 || !(s[1] in lbody) || s[1] == h ||
                (b = s[1]; continue)
            break
        end
        n = infer_trip(r, h, u, lbody, def, cval, label_of)
        tb = r.block  # the block of the counted test
        # else a counted test at another exit (e.g. the header's, with a data test at the
        # latch) bounds the loop
        for x in (n === nothing ? sort!(collect(lbody)) : Int[])
            x != r.block && haskey(term, x) &&
            any(s -> !(s in lbody) && !cold[s], K.succs[x]) || continue
            m = infer_trip(term[x], h, u, lbody, def, cval, label_of)
            m === nothing || n !== nothing && n <= m || ((n, tb) = (m, x))
        end
        n === nothing ?
        ((x = loop_shape(r, h, def, cval)) === nothing || (K.steps[h] = x)) :
        (K.trips[h] = n)
        n === nothing ||
            any(
                x -> x != tb && any(s -> !(s in lbody) && !cold[s], K.succs[x]),
                lbody,
            ) && push!(K.early, h)
    end
    if !isempty(links)
        n = Dict{String, Int}()
        foreach(x -> n[x[4]] = get(n, x[4], 0) + 1, links)
        note!(
            "stepped into the calls of device functions: " *
            join(("@$f ×$c" for (f, c) in sort(collect(n))), ", "),
        )
    end
    ext1 = K.extent1
    isempty(pend) || (EXTENT1[K] = () -> fill_extent1!(ext1))
    return K
end

# Reconvergence block of each divergent branch in `forks`: its immediate post-dominator
# (post-dominator sets by data flow over the reversed CFG; none: the warps exit apart).
function reconverge!(K::Kernel, forks)
    nb = length(K.labels)
    ws = collect(keys(K.wsplit))
    pdom = [isempty(K.succs[b]) ? BitSet([b]) : BitSet(1:nb) for b in 1:nb]
    changed = true
    while changed
        changed = false
        for b in nb:-1:1
            isempty(K.succs[b]) && continue
            d = push!(
                reduce(intersect, (pdom[x] for x in K.succs[b]); init = BitSet(1:nb)),
                b,
            )
            d == pdom[b] || (pdom[b] = d; changed = true)
        end
    end
    for b in [forks; ws]
        m = findfirst(
            d -> d != b && length(pdom[d]) == length(pdom[b]) - 1,
            collect(pdom[b]),
        )
        m === nothing && continue
        haskey(K.wsplit, b) ? (K.wsplit[b] = (K.wsplit[b][1], collect(pdom[b])[m])) :
        (K.merges[b] = collect(pdom[b])[m])
    end
    return K
end

# Trip count (header runs) of the loop with header `h`, latch `u` and blocks `body` that exits
# at the test `r` (the latch's or the header's): run its induction variable (phi = start; next = phi op
# C, op ∈ add/sub/mul/shl/or) until the `icmp` of phi or `phi op C` against a constant exits.
const STEP = Dict("add" => +, "sub" => -, "mul" => *, "shl" => (v, c) -> v << c, "or" => |)

function infer_trip(r, h, u, body, def, cval, label_of)
    m = match(r"^i1 %([-\w.$]+), label %([-\w.$]+), label %([-\w.$]+)", r.rest)
    c = m === nothing ? nothing : get(def, m[1], nothing)
    (c === nothing || c.op != "icmp") && return nothing
    cm = match(r"^(\w+) (\S+) ([^,]+), (.+)$", c.rest)
    cm === nothing && return nothing
    M = Int128(2)^(cm[2] == "i32" ? 32 : 64)  # unsigned predicates compare modulo M
    # the latch's merge of the next value with `undef` from the exiting path (a rotated
    # loop whose exit test precedes the increment) is that value
    defined(d) =
        d === nothing || d.op != "phi" ? d :
        (vs = [strip(x[1]) for x in eachmatch(r"\[\s*([^,]+),", d.rest)];
            vs = filter(v -> !(v in ("undef", "poison")), vs);
            length(vs) == 1 ? get(def, valname(vs[1]), nothing) : d)
    affine(d) =
        (d === nothing || !haskey(STEP, d.op)) ? nothing :
        (a = splitargs(replace(d.rest, r"^(\w+ )*" => "")); C = cval(a[2]);
            C === nothing ? nothing :
            (v -> STEP[d.op](v, C), get(def, valname(lastword(a[1])), nothing)))
    for (iv, bound, swapped) in ((cm[3], cm[4], false), (cm[4], cm[3], true))
        B = cval(bound)
        (B === nothing || !startswith(strip(iv), "%")) && continue
        f, phi = identity, get(def, valname(iv), nothing)
        phi !== nothing && phi.op != "phi" &&
            ((x = affine(phi)) === nothing ? continue : ((f, phi) = x))
        (phi === nothing || phi.op != "phi" || phi.block != h) && continue
        inc = [
            (strip(p[1]), strip(p[2], '"')) for
            p in eachmatch(r"\[\s*([^,]+),\s*%(\"[^\"]+\"|[-\w.$]+)\s*\]", phi.rest)
        ]
        init = [cval(v) for (v, l) in inc if label_of[l] != u]
        nxt = [
            affine(defined(get(def, valname(v), nothing))) for
            (v, l) in inc if label_of[l] == u
        ]
        (
            length(inc) == 2 && length(init) == 1 && init[1] !== nothing &&
            length(nxt) == 1 && nxt[1] !== nothing &&
            nxt[1][2] === phi
        ) || continue
        v = init[1]
        for n in 1:(10 ^ 6)
            l, rr = swapped ? (B, f(v)) : (f(v), B)
            p = cm[1]
            startswith(p, "u") && (l = mod(l, M); rr = mod(rr, M))
            cond =
                p == "eq" ? l == rr :
                p == "ne" ? l != rr :
                p in ("slt", "ult") ? l < rr :
                p in ("sle", "ule") ? l <= rr : p in ("sgt", "ugt") ? l > rr : l >= rr
            cond == (label_of[m[2]] in body) || return n
            v = nxt[1][1](v)
        end
    end
    return nothing
end

# (induction step, largest trip count) of a loop whose latch compares its induction variable
# with a run-time value: `trips` and `loops` count the iterations of the source loop, which an
# unrolled loop's variable advances by the unroll factor at a time; a remainder loop's bound
# `n & (f - 1)` keeps it below f trips.
function loop_shape(r, h, def, cval)
    m = match(r"^i1 %([-\w.$]+), label", r.rest)
    c = m === nothing ? nothing : get(def, m[1], nothing)
    (c === nothing || c.op != "icmp") && return nothing
    cm = match(r"^\w+ \S+ ([^,]+), (.+)$", c.rest)
    cm === nothing && return nothing
    for (iv, bound) in ((cm[1], cm[2]), (cm[2], cm[1]))
        d = get(def, valname(iv), nothing)
        (d === nothing || d.op != "add") && continue
        a = splitargs(replace(d.rest, r"^(\w+ )*" => ""))
        step = cval(a[2])
        phi = get(def, valname(lastword(a[1])), nothing)
        (step === nothing || phi === nothing || phi.op != "phi" || phi.block != h) &&
            continue
        b = get(def, valname(bound), nothing)
        cap =
            b !== nothing && b.op == "and" ?
            cval(splitargs(replace(b.rest, r"^(\w+ )*" => ""))[2]) : nothing
        return (Int(abs(step)),
            cap !== nothing && 0 < cap < 2^16 && ispow2(cap + 1) ? Int(cap) : typemax(Int))
    end
    return nothing
end

# blocks whose successors are all cold are cold (to the fixed point)
function propagate_cold!(cold, succs)
    changed = true
    while changed
        changed = false
        for b in eachindex(cold)
            !cold[b] && !isempty(succs[b]) && all(s -> cold[s], succs[b]) &&
                (cold[b] = changed = true)
        end
    end
end

# (latch, header) pairs from a DFS of the CFG from `roots` (the kernel's entry: its own loops)
function back_edges(K, roots = 1:1)
    edges = Tuple{Int, Int}[]
    state = zeros(Int8, length(K.blocks))
    for r in roots
        state[r] == 0 || continue
        state[r] = 1
        stack = [(r, 1)]
        while !isempty(stack)
            b, i = stack[end]
            i > length(K.succs[b]) && (state[b] = 2; pop!(stack); continue)
            stack[end] = (b, i + 1)
            s = K.succs[b][i]
            state[s] == 1 ? push!(edges, (b, s)) :
            state[s] == 0 && (state[s] = 1; push!(stack, (s, 1)))
        end
    end
    return edges
end

# entry blocks of the kernel and of the device functions it calls
roots(K) = [1; [e for (e, _) in values(K.calls)]]

# ─── SASS front end ──────────────────────────────────────────────────────────

const SASS_CID = Dict(
    op => c for
    (c, ops) in (:FP32 => "FADD FMUL FFMA FADD32I FMUL32I FFMA32I HADD2 HMUL2 HFMA2",
        :IADD => "IADD3 IADD IADD32I IABS", :SETP => "ISETP FSETP FSET ISET HSETP2",
        :ALU =>
            "LOP3 LOP LOP32I SHF SHL SHR SEL FSEL MOV MOV32I PRMT IMNMX FMNMX LEA SGXT " *
            "BMSK P2R R2P PLOP3 FCHK VOTE",
        :IMAD => "IMAD IMUL IMAD32I XMAD IDP", :FP64 => "DADD DMUL DFMA DSETP DMNMX",
        :XU => "MUFU POPC FLO BREV",
        :CONV => "I2F F2I F2F I2I FRND F2FP I2FP", :LDG => "LDG LD LDGSTS",
        :STG => "STG ST RED", :LDS => "LDS LDSM ATOMS",
        :STS => "STS", :LDL => "LDL", :STL => "STL", :LDC => "LDC",
        :ATOM => "ATOM ATOMG", :SHFL => "SHFL", :BAR => "BAR", :ADU => "MATCH REDUX",
        :S2R => "S2R CS2R S2UR",
        :BRA =>
            "BRA BRX JMP JMX CALL RET EXIT BSSY BSYNC BREAK WARPSYNC NOP YIELD BPT KILL " *
            "MEMBAR ERRBAR DEPBAR CCTL") for op in split(ops)
)

const NO_DST = Set(
    split(
        "STG ST RED STS STL ATOMS BAR BRA BRX JMP JMX CALL RET EXIT BSSY BSYNC BREAK " *
        "WARPSYNC NOP YIELD BPT KILL MEMBAR ERRBAR DEPBAR CCTL",
    ),
)

"""
    parse_sass(text) -> Kernel

Parse SASS (`cuobjdump -sass`, `nvdisasm`, `CUDA.@device_code_sass`; first
function). Registers carry dependencies (64/128-bit results write register
pairs/quads); `@P` guards are sources. Loop trips go to `simulate(; trips)`
keyed by block label (".L_x_3" or address "0x0120"); `conds` keys are guards
("!P0") and give the probability of taking the branch. A guarded `EXIT`/`RET`
branches to a final empty "EXIT" block (so trips can end a loop there); an
unguarded `CALL` of a subroutine or of another function in the text steps into
it (`Kernel.calls`) and its unguarded `RET` returns; `\$__internal_*` /
`*_slowpath` subroutines and blocks that only call them are cold (edge
probability 0). Control words (stall counts, barrier masks) are not read.
`K.frame` is the highest stack-pointer (`R1`) offset of the local accesses plus
their size, `K.clines` the constant-bank line each op reads.
"""
function parse_sass(text::AbstractString)
    K = Kernel()
    ids = Dict{String, Int32}()
    reg(r) = get!(() -> newval!(K, occursin(r"^U?P", r) ? 0 : 1), ids, r)
    hex(x) = parse(Int, x; base = 16)
    rows = Tuple{String, Vector{String}, SubString{String}, SubString{String}, String}[]
    pending = String[]  # (address, labels, opcode, operands, guard)
    # the first function, then the functions it calls (cuobjdump headers or nvdisasm sections;
    # their addresses restart, so their branches resolve by label)
    lines = split(text, '\n')
    heads = Dict{String, Int}()
    for (i, l) in enumerate(lines)
        m = match(r"^\s*(?:Function\s*:\s*(\S+)|\.text\.(\S+):)", l)
        m === nothing || get!(heads, something(m[1], m[2]), i)
    end
    todo = [1]
    done = Set{Int}()
    while !isempty(todo)
        i0 = popfirst!(todo)
        i0 in done && continue
        push!(done, i0)
        first_row = length(rows) + 1
        for l in view(lines, i0:length(lines))
            length(rows) >= first_row && occursin(r"^\s*(Function\s*:|\.text\.\S+:)", l) &&
                break
            occursin(';', l) || occursin(':', l) || continue  # (only labels and ops are read)
            s = replace(l, r"/\* 0x[0-9a-f]{16} \*/" => "")
            (m = match(r"^\s*Function\s*:\s*(\S+)", s)) === nothing ||
                (first_row > 1 && push!(pending, m[1]); continue)
            (m = match(r"^\s*(\.L_[\w$]+|\$[\w$.]+|\.text\.\S+|[A-Za-z_][\w$]*):", s)) !==
            nothing &&
                (
                    push!(
                        pending,
                        startswith(m[1], ".text.") && first_row > 1 ? m[1][7:end] : m[1],
                    );
                    continue)
            m = match(
                r"^\s*(?:/\*([0-9a-f]+)\*/)?\s*(@!?U?P[T0-9])?\s*([A-Z][A-Z0-9_.]*)\b\s*([^;]*);",
                s,
            )
            m === nothing && continue
            push!(
                rows,
                (
                    m[1] === nothing || first_row > 1 ? "" : "0x" * lpad(m[1], 4, '0'),
                    copy(pending),
                    m[3],
                    m[4],
                    m[2] === nothing ? "" : String(m[2]),
                ),
            )
            empty!(pending)
            startswith(m[3], "CALL") &&
                (c = match(r"`\(([^)]+)\)", m[4])) !== nothing && haskey(heads, c[1]) &&
                push!(todo, heads[c[1]])
        end
    end
    isempty(rows) && error("no SASS instructions found")
    addr_of = Dict(r[1] => i for (i, r) in enumerate(rows) if !isempty(r[1]))
    label_row = Dict(l => i for (i, r) in enumerate(rows) for l in r[2])
    target(ops) = (
        m = match(r"`\(([^)]+)\)|0x([0-9a-f]+)", ops);
        m === nothing ? nothing :
        m[1] !== nothing ? get(label_row, m[1], nothing) :
        get(addr_of, "0x" * lpad(m[2], 4, '0'), nothing)
    )

    # CALL: an edge to the callee (RET ends the walk)
    isbranch(r) = split(r[3], '.')[1] in ("BRA", "BRX", "JMP", "JMX", "EXIT", "RET", "CALL")
    leaders = Set([1])
    for (i, r) in enumerate(rows)
        isempty(r[2]) || push!(leaders, i)
        isbranch(r) &&
            (push!(leaders, i + 1); (t = target(r[4])) === nothing || push!(leaders, t))
    end
    starts = sort([i for i in leaders if i <= length(rows)])
    block_of = Dict(s => b for (b, s) in enumerate(starts))
    regs_in(x) = [m.match for m in eachmatch(r"\bU?R\d+\b|\bU?P\d\b", x)]
    calls = Dict{Int, Int}()  # calling block => callee entry
    rets = Set{Int}()
    for (b, s) in enumerate(starts)
        e = b < length(starts) ? starts[b + 1] - 1 : length(rows)
        first_op = length(K.cls) + 1
        for (_, _, opc, opnds, guard) in rows[s:e]
            base = split(opc, '.')[1]
            c =
                startswith(opc, "IMAD.WIDE") ? :IMADW :
                get(SASS_CID, base, startswith(base, "U") ? :UNI : :ALU)
            msg = "unknown SASS op $opc modeled as ALU"
            haskey(SASS_CID, base) || startswith(base, "U") || msg in K.notes ||
                push!(K.notes, msg)
            ops = filter(!isempty, strip.(split(opnds, r",(?![^\[]*\])")))
            hasdst = !(base in NO_DST) && !isempty(ops)

            # SHFL/ATOMG predicate sink
            hasdst && ops[1] == "PT" && length(ops) > 1 && (ops = ops[2:end])
            d = hasdst ? regs_in(ops[1]) : String[]
            srcs = Int32[
                reg(x) for x in unique(
                    [regs_in(join(ops[(hasdst ? 2 : 1):end], ","))
                        isempty(guard) ? String[] : [replace(guard, r"[@!]" => "")]],
                )
            ]
            w =
                occursin(".128", opc) ? 4 :
                occursin(r"\.(64|WIDE)\b|^[FI]2F\.F64", opc) || c === :FP64 ? 2 : 1
            m = isempty(d) ? nothing : match(r"^R(\d+)$", d[1])
            hi(j) = reg("R$(parse(Int, m[1]) + j)")
            bytes =
                Int(CID[c]) >= FIRST_MEM ?
                (
                    w == 4 ? 16 :
                    w == 2 ? 8 :
                    occursin(r"\.[US]8", opc) ? 1 :
                    occursin(r"\.[US]16", opc) ? 2 : 4
                ) : 0
            f = match(r"\[R1(?:\+0x(\w+))?\]", opnds)
            f === nothing || (K.frame = max(K.frame, hex(something(f[1], "0")) + bytes))
            f = match(r"c\[0x(\w+)\]\[0x(\w+)\]", opnds)
            f === nothing || (K.clines[length(K.cls) + 1] = 1024hex(f[1]) + hex(f[2]) ÷ 64)
            dst = emit!(
                K,
                c,
                isempty(d) ? Int32(0) : reg(d[1]),
                srcs;
                bytes,
                dst2 = m === nothing || w == 1 ? 0 : hi(1),
            )
            w == 4 && m !== nothing && foreach(j -> emit!(K, :FREE, hi(j), [dst]), 2:3)
        end
        push!(K.blocks, first_op:length(K.cls))
        push!(
            K.labels,
            isempty(rows[s][2]) ? (isempty(rows[s][1]) ? "B$b" : rows[s][1]) :
            rows[s][2][1],
        )
        lr = rows[e]
        t = isbranch(lr) ? target(lr[4]) : nothing
        fall = e < length(rows) ? [block_of[e + 1]] : Int[]
        op = split(lr[3], '.')[1]
        op == "RET" && isempty(lr[5]) && push!(rets, b)

        # an unguarded call of a subroutine (not a slow path) returns to the next block
        callee = t === nothing || isempty(rows[t][2]) ? "" : rows[t][2][1]
        if op == "CALL" && t !== nothing && isempty(lr[5]) && !isempty(fall) &&
           !(startswith(callee, "\$") || occursin("slowpath", callee))
            calls[b] = block_of[t]
            t = nothing
        end

        # synthetic empty block after the code: the target of guarded EXIT/RET
        term = length(starts) + 1
        succ = unique(
            op in ("EXIT", "RET") ? (isempty(lr[5]) ? Int[] : [term; fall]) :
            t === nothing ? fall : isempty(lr[5]) ? [block_of[t]] : [block_of[t]; fall],
        )
        push!(K.succs, succ)
        push!(K.probs, fill(NaN, length(succ)))
        push!(K.conds, length(succ) == 2 ? replace(lr[5], "@" => "") : "")
    end

    # guarded exits: a kernel-end block (no ops) so that the walk can leave a loop there
    if any(s -> length(starts) + 1 in s, K.succs)
        push!(K.blocks, (length(K.cls) + 1):length(K.cls))
        push!(K.labels, "EXIT")
        push!(K.succs, Int[])
        push!(K.probs, Float64[])
        push!(K.conds, "")
    end

    # Slow paths (ptxas subroutines for rcp/sqrt/div special inputs,
    # `$__internal_*`/`*_slowpath`) and the blocks that only lead there are cold: edges into
    # them get probability 0 where the branch has another way (as the IR front end does for
    # `unreachable` paths; the fast path's FCHK/BSSY/BRA ops are still charged).
    cold = [startswith(l, "\$") || occursin("slowpath", l) for l in K.labels]
    propagate_cold!(cold, K.succs)
    for b in eachindex(K.labels), (i, s) in enumerate(K.succs[b])
        cold[s] && !all(cold[K.succs[b]]) && (K.probs[b][i] = 0.0)
    end
    for (b, e) in calls  # a subroutine returns from its RET blocks
        seen = Set([e])
        work = [e]
        while !isempty(work)
            foreach(x -> x in seen || (push!(seen, x); push!(work, x)), K.succs[pop!(work)])
        end
        K.calls[b] = (e, sort!(collect(intersect(seen, rets))))
    end
    return K
end

# ─── Trace walk ──────────────────────────────────────────────────────────────

# Header runs of loop `h`: `trips` by label, else its inferred count (at most `early`, the
# `loops` count, when it can leave early on data), else `default_trip` source iterations of
# its run-time bound
function tripcount(K, h, trips, default_trip, early)
    haskey(trips, K.labels[h]) && return trips[K.labels[h]]
    n = get(K.trips, h, 0)
    n > 0 && return early !== nothing && h in K.early ? min(n, early) : n
    (s, cap) = get(K.steps, h, (1, typemax(Int)))
    return min(cap, cld(default_trip, s))
end

"""
Dynamic block trace under trip counts and branch probabilities (see `simulate`); `phase` in
[0, 1) places the block among the blocks that thread-range splits share out (`mixed[]` is
set when it decided one).
"""
function walk(K::Kernel; trips = Dict{String, Int}(),
    probs = Dict{Tuple{String, String}, Float64}(),
    conds = Dict{String, Float64}(), default_trip = 1, early = nothing, max_len = 10^7,
    notes = nothing, phase = nothing, mixed = nothing, warp = nothing, active = nothing)
    nb = length(K.blocks)
    preds = [Int[] for _ in 1:nb]
    foreach(b -> foreach(s -> push!(preds[s], b), K.succs[b]), 1:nb)
    bodies = Dict{Int, Set{Int}}()  # natural loops by header
    for (u, h) in back_edges(K, roots(K))
        body = get!(bodies, h, Set([h]))
        work = [u]
        while !isempty(work)
            (x = pop!(work)) in body || (push!(body, x); append!(work, preds[x]))
        end
    end
    trip = Dict(h => tripcount(K, h, trips, default_trip, early) for h in keys(bodies))
    if notes !== nothing
        # the kernel's outermost loops (not in a device function, not nested): their count is
        # usually the work per thread, which `loops` does not describe
        kh = Set(h for (_, h) in back_edges(K))
        outer(h) = h in kh && !any(g -> g != h && h in bodies[g], kh)
        for h in keys(bodies)
            l, n = K.labels[h], trip[h]
            (haskey(trips, l) || !(h in K.early) && haskey(K.trips, h)) && continue
            msg =
                !(h in K.early) ? "unknown trip count, using $n" :
                early === nothing ?
                "leaves early on data, using its bound $n (set `loops`)" :
                "leaves early on data, using $n"
            outer(h) &&
                (msg *= "; outermost loop: set `trips = Dict(\"$l\" => n)`, n per thread")
            push!(notes, "loop $l: $msg")
        end
        early === nothing ||
            any(h -> !haskey(trips, K.labels[h]) && (h in K.early || !haskey(K.trips, h)),
                keys(bodies)) ||
            push!(
                notes,
                "`loops` changes nothing: no loop without a `trips` entry has a " *
                "run-time count or an early exit",
            )
    end
    p = [
        haskey(conds, K.conds[b]) ? [conds[K.conds[b]], 1 - conds[K.conds[b]]] :
        [
            get(probs, (K.labels[b], K.labels[s]), K.probs[b][i]) for
            (i, s) in enumerate(K.succs[b])
        ] for b in 1:nb
    ]
    guessed = [count(isnan, q) > 1 for q in p]
    for (c, x) in conds
        0 <= x <= 1 || error("conds[\"$c\"] = $x is not a probability")
    end
    if notes !== nothing
        unused = [
            collect(setdiff(keys(trips), (K.labels[h] for h in keys(bodies))))
            collect(setdiff(keys(conds), K.conds))
            collect(
                setdiff(
                    keys(probs),
                    ((K.labels[b], K.labels[s]) for b in 1:nb for s in K.succs[b]),
                ),
            )
        ]
        isempty(unused) ||
            push!(notes, "unused trips/conds/probs keys: $(join(unused, ", "))")
    end
    # Entry and extent guards: an unset two-way branch outside the loops enters the side
    # that does work when the other side only leaves the kernel (through branches), else
    # the side that leads into a loop when the other does not, as a bounds check does for
    # all threads but the surplus of the last block.
    inloop = falses(nb)
    foreach(body -> inloop[collect(body)] .= true, values(bodies))
    trivial = [all(k -> K.cls[k] in (CID[:FREE], CID[:BRA]), r) for r in K.blocks]
    idle = falses(nb)  # leaves the kernel through branches only
    loopy = [haskey(bodies, b) for b in 1:nb]  # reaches a loop header
    changed = true
    while changed
        changed = false
        for b in 1:nb
            idle[b] || !trivial[b] || !all(s -> idle[s], K.succs[b]) ||
                (idle[b] = changed = true)
            loopy[b] || !any(s -> loopy[s], K.succs[b]) || (loopy[b] = changed = true)
        end
    end
    # A loop's zero-trip guard (in a loop or not): an unset two-way branch whose one side runs
    # straight into the header of a loop that the other side does not reach without passing
    # the branch again enters the loop (LLVM tests a rotated loop's run-time bound before it).
    function straight_header(s)
        for _ in 1:4
            haskey(bodies, s) && return s
            length(K.succs[s]) == 1 || return 0
            s = K.succs[s][1]
        end
        return 0
    end
    function reaches(s, h, b)
        seen = falses(nb)
        work = [s]
        while !isempty(work)
            x = pop!(work)
            x == h && return true
            (x == b || seen[x]) && continue
            seen[x] = true
            append!(work, K.succs[x])
        end
        return false
    end
    entered = Int[]
    for b in 1:nb
        guessed[b] && length(p[b]) == 2 || continue
        s1, s2 = K.succs[b]
        if !inloop[b] && idle[s1] != idle[s2]
            p[b] .= (idle[s2], idle[s1])
        elseif !inloop[b] && loopy[s1] != loopy[s2]
            p[b] .= (loopy[s1], loopy[s2])
        else
            h1, h2 = straight_header(s1), straight_header(s2)
            (h1 > 0) == (h2 > 0) && continue
            h, o = h1 > 0 ? (h1, s2) : (h2, s1)
            (b in bodies[h] || trip[h] < 1 || reaches(o, h, b)) && continue
            p[b] .= (h1 > 0, h2 > 0)
        end
        guessed[b] = false
        push!(entered, b)
    end
    notes === nothing || isempty(entered) ||
        push!(
            notes,
            "guards assumed to enter the kernel body (set `conds`/`probs` to override): " *
            join(
                (
                    isempty(K.conds[b]) ? K.labels[b] : K.conds[b] for
                    b in entered[1:min(end, 5)]
                ),
                ", ",
            ),
        )
    for q in p
        free = isnan.(q)
        any(free) && (q[free] .= max(0.0, 1 - sum(q[.!free])) / count(free))
    end
    credit = [zeros(length(q)) for q in p]
    loops_of = [
        sort([h for (h, body) in bodies if b in body]; by = h -> length(bodies[h])) for
        b in 1:nb
    ]
    exits(h, succ) = any(s -> !(s in bodies[h]), succ) && any(s -> s in bodies[h], succ)
    # Does an exit leave loop h with a positive probability? One without (e.g. `conds` that
    # never take its exit) is left through any exit after its trips.
    warm = Dict{Int, Bool}()
    haswarm(h) = get!(warm, h) do
        any(
            b ->
                any(i -> !(K.succs[b][i] in bodies[h]) && p[b][i] > 0, eachindex(p[b])),
            bodies[h])
    end
    forced = Set{Int}()

    # per block: the innermost loop its branch exits (0: none) and the successors that stay
    # in it; per edge: into a loop header from inside its loop (1), from outside (0), else -1
    leaves = zeros(Int, nb)
    stays = [Bool[] for _ in 1:nb]
    for b in 1:nb
        j = findfirst(h -> exits(h, K.succs[b]), loops_of[b])
        j === nothing && continue
        leaves[b] = loops_of[b][j]
        stays[b] = [s in bodies[leaves[b]] for s in K.succs[b]]
    end
    into =
        [[haskey(bodies, s) ? Int(b in bodies[s]) : -1 for s in K.succs[b]] for b in 1:nb]
    last_trip = [haskey(trip, h) ? trip[h] - 1 : 0 for h in 1:nb]
    iter = zeros(Int, nb)
    guesses = zeros(Int, nb)  # visits of guessed branches, first visit order
    first_guess = Int[]
    ok = Bool[]
    trace = Int32[]
    # divergent branches: (reconvergence block, side still to run, active lanes outside the
    # branch and on that side)
    pending = Tuple{Int, Int, Float32, Float32}[]
    act = 1.0f0  # fraction of the warp's lanes active
    rejoin = zeros(Int, nb)      # divergent branches that reconverge within their loop
    foreach(((b, m),) -> loops_of[m] == loops_of[b] && (rejoin[b] = m), K.merges)
    # this warp's side of the branches that send the warps of a block different ways
    wside = zeros(UInt8, nb)
    warp === nothing || for (b, (side, m)) in K.wsplit
        wside[b] = side[mod1(warp, length(side))]
        wside[b] == 3 &&
            (m > 0 && loops_of[m] == loops_of[b] ? (rejoin[b] = m) : (wside[b] = 0))
    end
    returns = Int[]  # call stack: the blocks after the calls
    callee = zeros(Int, nb)  # entry of the device function a block calls
    foreach(((b, (e, _)),) -> callee[b] = e, K.calls)
    b = 1
    while true
        # at a reconvergence block, run the divergent branch's other side first
        while !isempty(pending) && b == pending[end][1]
            m, s2, a0, a2 = pending[end]
            s2 > 0 && s2 != m ? (pending[end] = (m, 0, a0, a2); b = s2; act = a2) :
            (pop!(pending); act = a0)
        end
        push!(trace, b)
        active === nothing || push!(active, act)
        length(trace) > max_len &&
            error("trace longer than $max_len blocks; check trips/probs")
        succ = K.succs[b]
        if callee[b] > 0  # into the device function; its return blocks come back
            push!(returns, succ[1])
            b = callee[b]
            continue
        end
        if isempty(succ)
            isempty(returns) && break
            b = pop!(returns)
            continue
        end
        h = leaves[b]
        if h == 0 && rejoin[b] > 0  # lanes split: the warp runs both sides
            f = get(K.lanes, b, UInt8[32])
            f = act * Float32(f[mod1(something(warp, 1), length(f))] / 32)
            push!(pending, (rejoin[b], succ[2], act, act - f))
            act = f
            b = succ[1]
            continue
        end
        resize!(ok, length(succ))
        if h == 0
            guessed[b] && wside[b] == 0 &&
                (guesses[b] == 0 && push!(first_guess, b); guesses[b] += 1)
            fill!(ok, true)
        else
            stay = iter[h] < last_trip[h]
            for (i, s) in enumerate(stays[b])  # never leave through a cold edge
                ok[i] = s == stay && (stay || p[b][i] > 0)
            end
            any(ok) ||
                (haswarm(h) ? (ok .= stays[b]) : (push!(forced, h); ok .= .!stays[b]))
        end

        # A thread-range split outside loops (`Kernel.splits`) runs once per thread: with
        # `phase` (one block's point in [0, 1)), its side follows the probabilities across
        # blocks. Otherwise error diffusion over the allowed successors.
        i = 1
        if h == 0 && wside[b] in (1, 2)  # this warp's side
            i = Int(wside[b])
        elseif h == 0 && phase !== nothing && b in K.splits && count(>(0), p[b]) > 1
            u = mod(phase + 0.6180339887498949 * b, 1.0) * sum(p[b])
            i = something(findfirst(>(u), cumsum(p[b])), length(succ))
            mixed === nothing || (mixed[] = true)
        elseif h != 0 || length(succ) > 1
            cr = credit[b]
            cr .+= p[b]
            i = argmax(i -> ok[i] ? cr[i] : -Inf, eachindex(ok))
            cr[i] -= 1
        end
        into[b][i] >= 0 && (iter[succ[i]] = into[b][i] == 1 ? iter[succ[i]] + 1 : 0)
        b = succ[i]
    end
    notes === nothing || for h in forced
        push!(
            notes,
            "loop $(K.labels[h]): no exit with a positive probability (check " *
            "`conds`/`probs`), left after $(trip[h]) trips",
        )
    end
    if notes !== nothing && !isempty(first_guess)
        n = Dict{String, Int}()
        for b in first_guess
            c = isempty(K.conds[b]) ? "switch in $(K.labels[b])" : K.conds[b]
            n[c] = get(n, c, 0) + guesses[b]
        end
        push!(
            notes,
            "uniform branch probabilities (set `conds`/`probs`) on " *
            join(
                (
                    "$c ×$v" for
                    (c, v) in sort(collect(n); by = last, rev = true)[1:min(end, 5)]
                ),
                ", ",
            ),
        )
    end
    return trace
end

"""
    reorder(K, dev, glat) -> Kernel

List-schedule each block's ops (one issue per cycle, latency-aware, longest
path to the block end first), as ptxas does (IR order is close to source
order), keeping data dependencies, the order of memory ops, and barriers and
terminators in place. Ignores register pressure, so it bounds the ILP that
ptxas can expose from above, unless `live` is given: at most `live` values
defined in a segment are then kept live at once (an op whose result would
exceed it waits until a ready op frees a value, as ptxas's scheduler batches
loads only up to the register budget), which brackets the order from below.
"""
function reorder(K::Kernel, dev::Device, glat, live::Integer = typemax(Int))
    lat(k) = (
        c = CLASSES[K.cls[k]];
        c === :FREE ? 0.0 :
        c in (:LDG, :ATOM) ? glat :
        c in (:LDS, :STS) ?
        dev.lat.shared :
        c in (:LDL, :LDC) ? dev.lat.l1 : Int(K.cls[k]) >= FIRST_MEM ? 1.0 : dev.ops[c][1]
    )
    order = Int[]
    for r in K.blocks
        seg = Int[]

        # segments end at barriers and at the block's last op, which stay in place
        for k in r
            fence = K.cls[k] == CID[:BAR] || k == last(r)
            fence || push!(seg, k)
            fence || continue

            # preds: (op, delay) with delay = latency for data, 0 for memory order
            prod = Dict{Int32, Int}()
            preds = [Tuple{Int, Float64}[] for _ in seg]
            lastmem = 0
            for (i, k2) in enumerate(seg)
                foreach(
                    v -> haskey(prod, v) && push!(preds[i], (prod[v], lat(seg[prod[v]]))),
                    srcs(K, k2),
                )
                Int(K.cls[k2]) >= FIRST_MEM &&
                    (lastmem > 0 && push!(preds[i], (lastmem, 0.0)); lastmem = i)
                foreach(d -> d > 0 && (prod[d] = i), (K.dst[k2], K.dst2[k2]))
            end
            height = [lat(k2) for k2 in seg]
            for i in length(seg):-1:1, (j, delay) in preds[i]
                height[j] = max(height[j], delay + height[i])
            end
            npred = length.(preds)
            succs = [Tuple{Int, Float64}[] for _ in seg]
            avail = zeros(length(seg))
            t = 0.0
            foreach(
                i -> foreach(((j, delay),) -> push!(succs[j], (i, delay)), preds[i]),
                eachindex(seg),
            )
            ready = Set(i for i in eachindex(seg) if npred[i] == 0)

            # uses left in the segment of each value defined in it (0: live out), and the
            # count of values defined and still live; an op's pressure delta = values
            # defined - values freed
            uses = Dict{Int32, Int}()
            nlive = 0
            for (i, k2) in enumerate(seg), v in srcs(K, k2)
                haskey(prod, v) && (uses[v] = get(uses, v, 0) + 1)
            end
            defines(i) =
                count(d -> d > 0 && K.width[d] > 0, (K.dst[seg[i]], K.dst2[seg[i]]))
            frees(i) = count(
                v -> haskey(prod, v) && K.width[v] > 0 && get(uses, v, 0) == 1,
                unique(srcs(K, seg[i])),
            )

            # issue the highest ready op each cycle; idle until one is ready
            while !isempty(ready)
                t = max(t, minimum(i -> avail[i], ready))
                cands = collect(ready)

                # over budget: only ops that do not add pressure, if any are ready
                if nlive >= live
                    fits = filter(i -> defines(i) <= frees(i), cands)
                    isempty(fits) || (cands = fits)
                end
                i = argmax(i -> (avail[i] <= t, height[i], -i), cands)
                nlive += defines(i) - frees(i)
                for v in unique(srcs(K, seg[i]))
                    haskey(uses, v) && (uses[v] -= 1)
                end
                delete!(ready, i)
                push!(order, seg[i])
                t += 1
                for (j, delay) in succs[i]
                    avail[j] = max(avail[j], t - 1 + delay)
                    (npred[j] -= 1) == 0 && push!(ready, j)
                end
            end
            push!(order, k)
            empty!(seg)
        end
    end
    R = deepcopy(K)
    R.cls = K.cls[order]
    R.dst = K.dst[order]
    R.dst2 = K.dst2[order]
    R.bytes = K.bytes[order]
    R.sectors = K.sectors[order]
    R.partial = K.partial[order]
    R.waves = K.waves[order]
    R.lines = K.lines[order]
    R.srcs = reduce(vcat, (srcs(K, k) for k in order); init = Int32[])
    R.src_ptr = cumsum([Int32(1); Int32[K.src_ptr[k + 1] - K.src_ptr[k] for k in order]])
    R.orig = K.orig[order]
    R.phipred = Dict(findfirst(==(k), order) => v for (k, v) in K.phipred)
    return R
end

# ─── Register pressure and spills ────────────────────────────────────────────

"""
Values (ids of `width` > 0) live into and out of each block; phi sources are live out of
their predecessors.
"""
function liveness(K::Kernel)
    nb = length(K.blocks)
    use, def, pout = ([BitSet() for _ in 1:nb] for _ in 1:3)
    for b in 1:nb, k in K.blocks[b]
        preds = get(K.phipred, k, nothing)
        for (j, v) in enumerate(srcs(K, k))
            K.width[v] == 0 || (
                preds === nothing ? (v in def[b] || push!(use[b], v)) :
                push!(pout[preds[j]], v)
            )
        end
        foreach(d -> d > 0 && K.width[d] > 0 && push!(def[b], d), (K.dst[k], K.dst2[k]))
    end
    lin, lout = ([BitSet() for _ in 1:nb] for _ in 1:2)
    # (a device function is allocated apart from its caller: values live across the call
    # stay out of its pressure, as the calling convention saves them)
    succs = K.succs
    changed = true
    while changed
        changed = false
        for b in nb:-1:1
            out = copy(pout[b])
            foreach(s -> union!(out, lin[s]), succs[b])
            inn = union(use[b], setdiff(out, def[b]))
            (out == lout[b] && inn == lin[b]) ||
                (lout[b] = out; lin[b] = inn; changed = true)
        end
    end
    return lin, lout
end

"""
    spill(K, cap; remat = false) -> (K′, stats)

Register allocation with `cap` registers per thread by the furthest-next-use
rule (what ptxas approximates): each block is walked in order with its live-in
values in registers; when the live registers exceed `cap`, the value used
furthest ahead (block-exit values first) is evicted to local memory (one `STL`
after its definition) and reloaded (`LDL`) before each use while evicted.
Kernel arguments and their fields (constant bank) and predicates take no
registers. `remat = true` instead recomputes an evicted value of one cheap
integer op (IMAD/IADD/ALU/S2R/free) whose sources are available at the use
(ptxas with `-O3` on simple addresses; measured A100 spill frames of large
fused kernels match `remat = false`). `stats`: cap, peak
live registers and the op reaching it, values and 32-bit words spilled,
static `STL`/`LDL` counts, recomputed values. Dynamic counts:
`simulate(; spills = true)`.
"""
function spill(K::Kernel, cap::Integer; remat = false)
    lin, lout = liveness(K)
    w = K.width
    defop = zeros(Int, K.nvalues)
    foreach(k -> K.dst[k] > 0 && (defop[K.dst[k]] = k), eachindex(K.cls))
    cheap =
        (CID[:FREE], CID[:IMADW], CID[:IMAD], CID[:IADD], CID[:ALU], CID[:S2R], CID[:UNI])
    rem = falses(K.nvalues)  # rematerializable: defined by one cheap op (not a phi)
    for v in 1:K.nvalues
        k = defop[v]
        remat && k > 0 && K.cls[k] in cheap && !haskey(K.phipred, k) && (rem[v] = true)
    end
    stores = Set{Int32}()
    evicted = Set{Int32}()  # spilled to local memory; dropped to be recomputed
    reloads = Dict{Int, Vector{Int32}}()
    remats = Dict{Int, Vector{Int32}}()
    nremat = 0
    peak = 0
    peak_at = 0
    scan!(cap) =
        for (b, r) in enumerate(K.blocks)
            uses = Dict{Int32, Vector{Int}}()
            for (i, k) in enumerate(r), v in srcs(K, k)
                w[v] > 0 && !haskey(K.phipred, k) && push!(get!(uses, v, Int[]), i)
            end
            far = length(r) + 1  # next use of block-exit values; 0: dead
            nextuse(v, i) = (
                u = get(uses, v, Int[]);
                j = findfirst(>=(i), u);
                j === nothing ? (v in lout[b] ? far : 0) : u[j]
            )
            inreg = Dict{Int32, Int}()
            used = 0
            avail(u) = w[u] == 0 || haskey(inreg, u) || u in stores || u in evicted
            function evict!(i, keep)
                used > peak && (peak = used; peak_at = r[min(i, length(r))])
                while used > cap
                    cands = [v for v in keys(inreg) if !(v in keep)]
                    isempty(cands) && break
                    v = argmax(v -> (nextuse(v, i), w[v]), cands)
                    used -= pop!(inreg, v)

                    # recomputable if sources available
                    push!(rem[v] && all(avail, srcs(K, defop[v])) ? evicted : stores, v)
                end
            end

            # evicted value v for op k: recompute it from its sources, else reload it
            function bring!(v, k, depth)
                haskey(inreg, v) && return
                if v in evicted && depth < 4  # recomputed in this block only
                    foreach(u -> w[u] > 0 && bring!(u, k, depth + 1), srcs(K, defop[v]))
                    push!(get!(remats, k, Int32[]), v)
                    nremat += 1
                else
                    push!(get!(reloads, k, Int32[]), v)
                end
                inreg[v] = w[v]
                used += w[v]
            end
            for v in lin[b]
                v in stores || v in evicted || (inreg[v] = w[v]; used += w[v])
            end
            evict!(1, ())
            for (i, k) in enumerate(r)
                s = srcs(K, k)
                phi = haskey(K.phipred, k)
                if !phi
                    foreach(v -> w[v] > 0 && bring!(v, k, 0), s)
                    evict!(i, s)
                    for v in s
                        haskey(inreg, v) && nextuse(v, i + 1) == 0 &&
                            (used -= pop!(inreg, v))
                    end
                end
                for d in (K.dst[k], K.dst2[k])
                    d > 0 && w[d] > 0 && nextuse(d, i + 1) > 0 &&
                        (inreg[d] = w[d]; used += w[d])
                end
                evict!(i + 1, (K.dst[k], K.dst2[k]))
            end
        end
    scan!(typemax(Int))  # uncapped: the peak pressure (within the cap: nothing to evict)
    nremat = 0
    peak > cap && scan!(cap)
    R = Kernel()
    R.labels = K.labels
    R.succs = K.succs
    R.probs = K.probs
    R.conds = K.conds
    R.trips = K.trips
    R.early = K.early
    R.nvalues = K.nvalues
    R.width = K.width
    R.notes = copy(K.notes)
    R.frame = K.frame
    R.clines = K.clines
    R.tables = K.tables
    R.splits = K.splits
    R.merges = K.merges
    R.wsplit = K.wsplit
    R.calls = K.calls
    R.alias = K.alias
    R.steps = K.steps
    R.lanes = K.lanes
    R.walks = K.walks
    R.extent1 = K.extent1
    R.syncs = K.syncs
    for r in K.blocks
        first_op = length(R.cls) + 1
        for k in r
            foreach(
                v -> (
                    d = defop[v];
                    emit!(
                        R,
                        CLASSES[K.cls[d]],
                        v,
                        srcs(K, d);
                        bytes = K.bytes[d],
                        dst2 = K.dst2[d],
                        sectors = K.sectors[d],
                        partial = K.partial[d],
                        waves = K.waves[d],
                        lines = K.lines[d],
                        orig = K.orig[d],
                    )
                ),
                get(remats, k, Int32[]))
            foreach(
                v -> emit!(R, :LDL, v, [v]; bytes = 4 * w[v], orig = 0),
                get(reloads, k, Int32[]),
            )
            haskey(K.phipred, k) && (R.phipred[length(R.cls) + 1] = K.phipred[k])
            emit!(
                R,
                CLASSES[K.cls[k]],
                K.dst[k],
                srcs(K, k);
                bytes = K.bytes[k],
                dst2 = K.dst2[k],
                sectors = K.sectors[k],
                partial = K.partial[k],
                waves = K.waves[k],
                lines = K.lines[k],
                orig = K.orig[k],
            )
            foreach(
                d ->
                    d > 0 && d in stores &&
                    emit!(R, :STL, 0, [d]; bytes = 4 * w[d], orig = 0),
                (K.dst[k], K.dst2[k]),
            )
        end
        push!(R.blocks, first_op:length(R.cls))
    end
    stats = (; cap, peak, peak_at, spilled = length(stores),
        words = sum(Int(w[v]) for v in stores; init = 0),
        stl = length(stores), ldl = sum(length, values(reloads); init = 0),
        remat = nremat)
    return R, stats
end

# ─── Simulation ──────────────────────────────────────────────────────────────

const PIPES = (:fma, :alu, :fp64, :xu, :cbu, :uni)

const SMUNITS = (:lsu, :bar)

# Stall reasons: issued; at a barrier; no warp; waiting on a value of class c;
# pipe busy; SM unit busy; load delayed by the L2/DRAM queue.
const REASONS = [:issue; :barrier; :idle; [Symbol(:wait_, c) for c in CLASSES]
    [Symbol(:pipe_, p) for p in PIPES]
    [Symbol(:unit_, u) for u in SMUNITS]; :l2_bandwidth; :dram_bandwidth
    :no_instruction]

const R_ISSUE, R_BAR, R_IDLE, R_WAIT = 1, 2, 3, 3

const R_PIPE = R_WAIT + length(CLASSES)

const R_SMU = R_PIPE + length(PIPES)

const R_L2, R_DRAM, R_NOINST = R_SMU + length(SMUNITS) .+ (1, 2, 3)

const NPHASE = 8  # traces of a kernel whose blocks take different sides of a branch

# `loops` flag: source iterations of loops whose trip count is neither given nor inferred
# (or whose inferred bound has an early exit on data), the geometric middle of each value's
# range (so at most 2x off inside it): `:few` 2-8
# (substeps, Newton iterations, the points of a one-dimensional rule), `:short` 8-32 (fixed
# counts in the tens, two-dimensional quadrature rules), `:long` 32-128 (the levels or
# layers of a column)
const LOOPS = Dict(:few => 4, :short => 16, :long => 64)

loops_trip(loops) =
    loops === nothing ? nothing :
    get(() -> error("loops must be :few, :short, :long or nothing"), LOOPS, loops)

"Result of [`simulate`](@ref); `display` it for a report."
struct Prediction
    device::String
    time_us::Float64
    cycles_sm::Float64                      # busiest SM, whole grid
    cycles_block::Float64                   # mean block lifetime
    cycles_warp::Float64                    # mean warp lifetime
    blocks_per_sm::Int                      # resident
    blocks_total_per_sm::Int

    # a unit (≥ 85% busy), :latency, or the top stall
    binding::Symbol

    # fractions of scheduler cycles (schedulers with warps)
    stalls::Vector{Pair{Symbol, Float64}}
    util::Vector{Pair{Symbol, Float64}}     # busy fractions of pipes, SM units, queues

    # contention-free cycles per SM (busy time, 1-warp latency)
    bounds::Vector{Pair{Symbol, Float64}}
    counts::Dict{Symbol, Int}               # dynamic ops per thread

    # mean cycles per block between barriers ("label#op")
    phases::Vector{Tuple{String, Float64}}

    # misses per warp
    icache::NamedTuple{(:code_bytes, :l0_misses, :l1_misses), Tuple{Int, Float64, Float64}}
    notes::Vector{String}
end

"""
    simulate(K; threads, blocks = 0, device = :sm_80, regs = 0, shmem = 0,
             blocks_per_sm = nothing, hits = (l1 = 0.0, l2 = 0.0, dram = 1.0),
             store_dram = hits.dram, coalescing = 1.0, shared_ways = 1.0, spills = false,
             remat = false, local_hits = nothing, icache = true, stagger = 0.0,
             expansion = 1.0, trips = Dict(), conds = Dict(), probs = Dict(),
             default_trip = 1, sim_waves = 3, reorder = false, hit_levels = true,
             scoreboards = 6, loaded = true, load_hits = Dict(), clock_ghz = 0,
             launch_us = 2.0, cache = :cold, gather = :local, loops = nothing,
             inner_extent = nothing) -> Prediction

Simulate one SM running `cld(blocks, n_sm)` blocks of `threads` threads (`blocks = 0`:
one wave; pass the blocks that have work); residency is `occupancy(device, threads, regs,
shmem)` unless `blocks_per_sm` is given. Take `regs`/`shmem` from ptxas or the kernel's
launch statistics (`spill(K, 255)[2].peak` bounds `regs` from above).

Control flow (each guess is listed in `notes`): `trips` (loop header label => runs of the
header; a loop tested at its top runs its body one time fewer), `conds` (branch predicate
=> P(true)), `probs` ((from, to) labels => probability). A loop that `trips` does not name
runs its count inferred from constants, else `default_trip` (1). Set `trips` for the
kernel's outermost loop whenever its count is run-time data (a grid-stride loop, a matmul's
inner dimension): unset, it runs once.

Memory: `hits` (fractions of global loads served by L1/L2/DRAM; from Nsight Compute: L1 =
1 - L2 sectors read from L1 / L1 load sectors, DRAM = DRAM read bytes / L1 load bytes; not
`lts__t_sector_hit_rate`, which counts stores), `store_dram` (fraction of stored bytes
written to DRAM, and of partially written sectors read first), `coalescing` (multiplies the
bytes of every global access beyond the sectors its warp touches, `Kernel.sectors`),
`shared_ways` (mean bank-conflict degree), `load_hits` (static op index of a load => hits,
e.g. `(l1 = 1, l2 = 0, dram = 0)` for a value a prefetch made ready).

Four flags describe what the kernel alone cannot show; decide them from the algorithm and
the kernel's place in the program (the skill's decision tables), never from measured times:
- `cache = :warm`: the inputs were just touched by preceding kernels and the kernel's
  data fits in half the L2: DRAM loads hit L2 and stores stay there (`store_dram = 0`).
- `gather = :scattered`: consecutive threads reach unrelated memory through a loaded index
  (connectivity, neighbor lists, sparse columns): a sector and a line per lane.
- `loops = :few | :short | :long`: loops whose count is run-time data, or a constant bound
  they can leave early on data (a convergence test), run 4 | 16 | 64 source iterations
  (2-8, 8-32, 32-128 expected) unless `trips` names them; unset, run-time counts run
  `default_trip` and early exits are never taken. It also applies to the outermost loop.
- `inner_extent = 1`: the innermost dimension of the index space has size 1 in this
  launch: the run-time sizes that the thread index is divided by are 1 in the footprints,
  and loads take their L1 hits from the sectors the block already touched instead of
  `hits.l1`.

Mechanism switches (report each): `reorder = true | n` (list-schedule IR blocks, with at
most `n` values live; SASS order is exact), `hit_levels` (false: hits-weighted mean
latency per load), `scoreboards` (dependency barriers per warp; 0: unlimited), `loaded`
(DRAM latency rising with utilization up to `lat.dram_loaded`), `icache` (instruction
fetch misses), `stagger` (cycles between resident blocks' starts), `spills` (allocate
`regs` registers and insert `STL`/`LDL`, see [`spill`](@ref); `remat` recomputes,
`local_hits` places the local traffic, default L1 if it fits beside the shared memory),
`expansion` (issue slots per modeled op, e.g. SASS count / `sum(values(op_counts(K)))`).
`clock_ghz` is the SM clock of the run (Nsight Compute: `gpc__cycles_elapsed.avg.per_second`;
0: the device's); `launch_us` is added per kernel; `sim_waves` waves of resident blocks are
simulated before the steady state is extrapolated.
"""
function simulate(K::Kernel; threads::Integer, blocks::Integer = 0, device = :sm_80,
    regs = 0, shmem = 0,
    blocks_per_sm = nothing, hits = (l1 = 0.0, l2 = 0.0, dram = 1.0),
    store_dram = hits.dram, coalescing = 1.0,
    shared_ways = 1.0, spills = false, remat = false, local_hits = nothing, icache = true,
    stagger = 0.0,
    expansion = 1.0, trips = Dict{String, Int}(), conds = Dict{String, Float64}(),
    probs = Dict{Tuple{String, String}, Float64}(),
    default_trip = 1, sim_waves = 3, reorder = false, hit_levels = true, scoreboards = 6,
    loaded = true,
    load_hits = Dict{Int, NamedTuple}(), clock_ghz = 0, launch_us = 2.0, cache = :cold,
    gather = :local, loops = nothing, inner_extent = nothing)
    args = (; threads, blocks, device, regs, shmem, blocks_per_sm, hits, store_dram,
        coalescing, shared_ways, spills, remat, local_hits, icache, stagger, expansion,
        conds, probs, sim_waves, reorder, hit_levels, scoreboards, loaded, load_hits,
        clock_ghz, launch_us, cache, gather, loops, inner_extent)
    dev =
        device isa Device ? device :
        get(() -> error("device must be :sm_60, :sm_70, :sm_80 or a Device"), DEVICES,
            device)
    early = loops_trip(loops)
    default_trip = something(early, default_trip)

    # DRAM latency under load: at most the device calibration (see below)
    d0 = dev.lat.dram
    loaded && (dev = with(dev; lat = (; dev.lat..., dram = dev.lat.dram_loaded)))

    # the run's SM clock (the DRAM bandwidth is fixed in bytes/s)
    f = clock_ghz > 0 ? dev.clock_ghz / clock_ghz : 1.0
    dev =
        with(dev; clock_ghz = dev.clock_ghz / f, bw = (; dev.bw..., dram = f * dev.bw.dram))
    expansion >= 1 || error("expansion must be ≥ 1")
    notes = String[]  # the walk's guesses first: they change predictions
    # Blocks take the sides of thread-range splits in proportion (`phase`), e.g. the vertex
    # and face blocks of a summation kernel: block j runs traces[mod1(j, end)].
    # Warps that take different sides of `Kernel.wsplit` branches run their own traces: warp
    # i of block j runs traces[tmap[i, mod1(j, end)]].
    mixed = Ref(false)
    wkw = (; trips, probs, conds, default_trip, early)
    W = cld(threads, 32)
    sig(w) = [side[mod1(w, length(side))] for (side, _) in values(K.wsplit)]
    reps = Int[]  # a warp of each distinct signature
    wsig = [
        something(findfirst(r -> sig(r) == sig(w), reps), (push!(reps, w); length(reps)))
        for w in 1:W
    ]
    rep(j) = isempty(K.wsplit) ? nothing : reps[j]
    acts = [Float32[]]  # by trace: the fraction of lanes active in each block it visits
    traces = [
        walk(
            K;
            wkw...,
            notes,
            phase = 0.5 / NPHASE,
            mixed,
            warp = rep(1),
            active = acts[1],
        ),
    ]
    nph = mixed[] ? NPHASE : 1
    for k in 1:nph, j in eachindex(reps)
        k == j == 1 && continue
        push!(acts, Float32[])
        push!(
            traces,
            walk(K; wkw..., phase = (k - 0.5) / NPHASE, warp = rep(j), active = acts[end]),
        )
    end
    tmap = [(k - 1) * length(reps) + wsig[w] for w in 1:W, k in 1:nph]
    all(a -> all(>(0.5), a), acts) && empty!(acts)  # no side with at most half the lanes
    # A simulation beyond SIM_OPS ops (its cost: the resident blocks' warp traces) runs 2
    # and 4 trips of the kernel's outermost loop with the most trips, extrapolated linearly
    # to its trips (two trips apart: a branch guessed at 50/50 alternates per trip).
    work(tr) = sum(b -> length(K.blocks[b]), tr; init = 0)
    R1 = something(blocks_per_sm, occupancy(dev, threads, regs, shmem))
    nsim1 = min(cld(blocks == 0 ? R1 * dev.n_sm : blocks, dev.n_sm), sim_waves * R1)
    wk = work.(traces)
    if nsim1 * sum(i -> wk[tmap[i, 1]], 1:W) > SIM_OPS[]
        heads = [h for (_, h) in back_edges(K)]
        outer = filter(h -> !any(g -> g != h && h in loop_body(K, g), heads), unique(heads))
        n(h) = tripcount(K, h, trips, default_trip, early)
        h = isempty(outer) ? 0 : argmax(n, outer)
        if h > 0 && n(h) > 4
            p1, p2 = (
                simulate(K; args..., default_trip,
                    trips = merge(trips, Dict(K.labels[h] => j))) for j in (2, 4)
            )
            return extrapolate(p1, p2, (n(h) - 2) / 2,
                "simulated 2 and 4 trips of loop $(K.labels[h]), extrapolated to $(n(h))")
        end
    end
    seen = sum(b -> length(K.blocks[b]), reduce(union!, traces; init = BitSet()); init = 0)
    seen < length(K.cls) / 4 && pushfirst!(
        notes,
        "the walk ran $(round(Int, 100seen / length(K.cls)))% of the kernel's ops: " *
        "check the guessed branches (an unset bounds check skips the body)",
    )
    append!(notes, K.notes)

    # Per-load hit overrides by parsed op (`Kernel.orig` survives spills and reorder).
    vhits = Dict{Int32, NamedTuple{(:l1, :l2, :dram), NTuple{3, Float64}}}()
    for (k, hh) in load_hits
        CLASSES[K.cls[k]] in (:LDG, :LDL, :ATOM) && K.dst[k] > 0 ||
            error("load_hits key $k is not a load with a result")
        abs(sum(hh) - 1) < 1e-6 || error("hits must sum to 1")
        vhits[k] = map(Float64, (; hh.l1, hh.l2, hh.dram))
    end
    foreach(k -> get!(vhits, k, (l1 = 1.0, l2 = 0.0, dram = 0.0)), K.tables)
    R0 = something(blocks_per_sm, occupancy(dev, threads, regs, shmem))
    R0 < 1 && error("no block fits on an SM (threads $threads, regs $regs, shmem $shmem)")
    nblk = blocks == 0 ? R0 * dev.n_sm : blocks
    per_sm = cld(nblk, dev.n_sm)
    R = min(R0, per_sm)
    nsim = min(per_sm, sim_waves * R)
    cache in (:cold, :warm) || error("cache must be :cold or :warm")
    gather in (:local, :scattered) || error("gather must be :local or :scattered")
    gather === :scattered && !any(<(0), K.sectors) &&
        push!(
            notes,
            "`gather = :scattered` changes nothing: no access through a loaded index",
        )
    h = map(Float64, (; hits.l1, hits.l2, hits.dram))

    # inputs just written by earlier kernels and still in L2: DRAM loads hit L2, and the
    # stores stay there (their write-back is later traffic)
    cache === :warm && (h = (; h.l1, l2 = h.l2 + h.dram, dram = 0.0); store_dram = 0.0)

    # A degenerate innermost dimension (`inner_extent = 1`): accesses whose lanes'
    # addresses are known (not through a loaded index with `gather = :scattered`) take their
    # footprints with the run-time divisors of the thread index 1, and such a load hits L1
    # for the sectors that earlier loads of the block touched (`Kernel.extent1`), its other
    # sectors going to L2 and DRAM in the proportion of `hits`: the lanes span the outer
    # dimensions (e.g. the nodes of several elements, not the levels of a column), so
    # `hits`, which describes lanes along a long innermost dimension, does not apply.
    inner_extent === nothing || inner_extent == 1 ||
        error("inner_extent must be 1 or unset")
    inner_extent == 1 &&
        (K = Base.inferencebarrier(extent1_kernel)(K, h, vhits, gather)::Kernel)

    # sm_6x (no independent thread scheduling) needs no shuffle member-mask checks
    if dev.sm < 70 && !isempty(K.syncs)
        K = Kernel((getfield(K, f) for f in fieldnames(Kernel))...)
        K.cls = [K.orig[k] in K.syncs ? CID[:FREE] : c for (k, c) in enumerate(K.cls)]
    end

    # Loads that reuse lines across the iterations of their loop (`Kernel.walks`): a lane
    # that walks an array re-reads the line of its previous iteration, so only the loads that
    # cross into a new line (|step| / 128 of them) miss, and a load of the same address in
    # every iteration misses in the first of the loop's trips only. The re-reads hit if the
    # lines that all resident warps reuse fit in the L1 beside the shared memory (else LRU
    # evicts each before its next use); misses go to L2 and DRAM in the proportion of `hits`.
    # `hits` stays the fraction over all global loads where it can: the other loads get the
    # L1 hits that these loads do not take (none if they take more).
    if !isempty(K.walks) && h.l1 < 1
        visits = zeros(Int, length(K.blocks))  # by the warps' traces
        for (t, tr) in enumerate(traces)
            nw = count(==(t), tmap)
            foreach(b -> visits[b] += nw, tr)
        end
        n = Dict{Int, Float64}()  # executions of each global load
        for b in findall(>(0), visits), k in K.blocks[b]
            K.cls[k] in (C_LDG, C_ATOM) && !haskey(vhits, k) &&
                (n[k] = get(n, k, 0.0) + visits[b])
        end
        reused =
            sum(k -> haskey(n, k) ? (K.lines[k] > 0 ? K.lines[k] : K.bytes[k] / 4) : 0.0,
                keys(K.walks))
        miss = h.l2 + h.dram
        wl = wn = 0.0
        if 128 * reused * R * W <= dev.l1_bytes - R * shmem
            for (k, (δ, hd)) in K.walks
                haskey(n, k) || continue
                m =
                    δ != 0 ? abs(δ) / 128 :
                    1 / max(1, tripcount(K, hd, trips, default_trip, early))
                l1 = max(h.l1, 1 - m)
                vhits[k] =
                    (; l1, l2 = (1 - l1) * h.l2 / miss, dram = (1 - l1) * h.dram / miss)
                wl += n[k] * l1
                wn += n[k]
            end
        end
        rest = sum(values(n); init = 0.0) - wn
        l1 = rest > 0 ? clamp((h.l1 * (rest + wn) - wl) / rest, 0.0, h.l1) : h.l1
        h = (; l1, l2 = (1 - l1) * h.l2 / miss, dram = (1 - l1) * h.dram / miss)
        wn > 0 && push!(notes,
            "$(count(k -> haskey(n, k), keys(K.walks))) loads reuse lines across loop " *
            "iterations: L1 hits after the first touch")
    end
    reorder isa Bool && !reorder ||
        (
            K = GPUPerfModel.reorder(K, dev,
                h.l1 * dev.lat.l1 + h.l2 * dev.lat.l2 + h.dram * dev.lat.dram,
                reorder isa Bool ? typemax(Int) : reorder)
        )
    words = 0
    if spills
        regs > 0 || error("`spills` needs the register cap `regs`")
        K, st = spill(K, regs; remat)
        words = st.words
        push!(
            notes,
            "spills: peak $(st.peak) live registers (op $(st.peak_at)), cap $regs: " *
            "$(st.spilled) values ($(st.words) words) spilled, $(st.stl) STL + $(st.ldl) " *
            "LDL, $(st.remat) recomputed values per thread (static)",
        )
    end
    if local_hits === nothing
        lb = 4words + K.frame
        fits = lb * threads * R <= dev.l1_bytes - R * shmem
        local_hits =
            fits ? (l1 = 1.0, l2 = 0.0, dram = 0.0) : (l1 = 0.0, l2 = 1.0, dram = 0.0)
        lb > 0 && push!(
            notes,
            "local memory $lb B/thread × $(threads * R) resident threads " *
            (fits ? "fits L1" : "exceeds L1: L2 hits"),
        )
    end
    hl = map(Float64, (; local_hits.l1, local_hits.l2, local_hits.dram))
    abs(sum(h) - 1) + abs(sum(hl) - 1) < 1e-6 || error("hits must sum to 1")
    opts = (;
        h,
        hl,
        store_dram,
        coalescing,
        shared_ways,
        icache,
        stagger,
        expansion,
        hit_levels,
        scoreboards,
        vhits,
        gather,
    )
    go(dev) =
        run_sm(K, dev, traces, acts, tmap, W, R, nsim; opts...,
            l2_bw = dev.bw.l2 / min(dev.n_sm, nblk),
            dram_bw = dev.bw.dram / min(dev.n_sm, nblk))
    sim = go(dev)

    # ... and 1/(1 - u) times the unloaded latency beyond L2 at a DRAM utilization u
    u = get(sim.busy, :dram, 0.0) / sim.t_end
    l = dev.lat.l2 + (d0 - dev.lat.l2) / (1 - min(u, 0.99))
    if loaded && l < dev.lat.dram
        dev = with(dev; lat = (; dev.lat..., dram = l))
        push!(notes, "DRAM latency $(round(Int, l)) cycles at $(round(Int, 100u))% use")
        sim = go(dev)
    end
    lat1 = run_sm(
        K,
        dev,
        traces,
        acts,
        fill(argmax(t -> wk[t], eachindex(traces)), 1, 1),  # the busiest warp
        1,
        1,
        1;
        (; opts..., stagger = 0.0)...,
        l2_bw = Inf,
        dram_bw = Inf,
    ).t_end
    bounds = sort(
        [
            :issue => sim.stall[R_ISSUE] / dev.n_sched
            collect(sim.busy)
            :latency => lat1 * cld(nsim, R)
        ];
        by = last, rev = true)
    tot = sum(sim.stall) - sim.stall[R_IDLE]
    stalls = sort(
        [REASONS[i] => x / tot for (i, x) in enumerate(sim.stall) if x > 0 && i != R_IDLE];
        by = last,
        rev = true,
    )
    waits = filter(x -> first(x) != :issue, stalls)
    binding =
        last(bounds[1]) >= 0.85 * sim.t_end || isempty(waits) ? first(bounds[1]) :
        first(waits[1])
    bounds = [k => v * per_sm / nsim for (k, v) in bounds]

    # cycles per extra block
    steady = nsim > R ? (sim.t_end - sim.t_first) / (nsim - R) : sim.t_end / R
    T = sim.t_end + max(per_sm - nsim, 0) * steady

    # busy time is a hard floor
    T = max(T, maximum(last(b) for b in bounds if first(b) != :latency))
    return Prediction(dev.name, T / (dev.clock_ghz * 1e3) + launch_us, T,
        sim.block_time / sim.blocks_done,
        sim.warp_time / (sim.blocks_done * W), R, per_sm, binding, stalls,
        sort([k => v / sim.t_end for (k, v) in sim.busy]; by = last, rev = true), bounds,
        op_mean(K, traces, tmap),
        sort([(l, x / sim.blocks_done) for (l, x) in sim.phases]; by = last, rev = true),
        (
            code_bytes = sim.code_bytes,
            l0_misses = sim.miss0 / (sim.blocks_done * W),
            l1_misses = sim.miss1 / (sim.blocks_done * W),
        ),
        notes)
end

# The kernel with `inner_extent = 1` (see `simulate`): its accesses take their footprints
# with unit divisors (`Kernel.extent1`) and its loads' L1 shares go into `vhits`.
function extent1_kernel(K::Kernel, h, vhits, gather)
    th = pop!(EXTENT1, K, nothing)
    th === nothing || th()
    isempty(K.extent1) && return K
    K = Kernel((getfield(K, f) for f in fieldnames(Kernel))...)
    (K.sectors, K.partial, K.waves, K.lines) =
        copy.((K.sectors, K.partial, K.waves, K.lines))
    miss = h.l2 + h.dram
    for k in eachindex(K.cls)
        o = get(K.extent1, K.orig[k], nothing)
        (o === nothing || gather === :scattered && o[1] < 0) && continue
        (K.sectors[k], K.partial[k], K.waves[k], K.lines[k], l1) = o
        isnan(l1) || haskey(vhits, K.orig[k]) ||
            (
                vhits[K.orig[k]] =
                    miss > 0 ?
                    (; l1 = Float64(l1), l2 = (1 - l1) * h.l2 / miss,
                        dram = (1 - l1) * h.dram / miss) :
                    (; l1 = Float64(l1), l2 = 1.0 - l1, dram = 0.0)
            )
    end
    return K
end

const Hits = NamedTuple{(:l1, :l2, :dram), NTuple{3, Float64}}

# simulated ops (resident blocks × their warps' trace ops) beyond which `simulate`
# extrapolates over the outermost loop (≤ 4% off on serial solves, ≤ 10% on fused solvers)
const SIM_OPS = Ref(2 * 10^6)

# natural loop body of header h (kernel loops)
function loop_body(K, h)
    preds = [Int[] for _ in K.blocks]
    foreach(b -> foreach(s -> push!(preds[s], b), K.succs[b]), eachindex(K.blocks))
    body = Set([h])
    work = [u for (u, g) in back_edges(K) if g == h]
    while !isempty(work)
        (x = pop!(work)) in body || (push!(body, x); append!(work, preds[x]))
    end
    return body
end

# p1 + m (p2 - p1) for the additive parts of two predictions; p2's shares and notes
lin(a, b, m) = a + m * (b - a)
function extrapolate(p1, p2, m, note)
    lin_pairs(x, y) = [k => lin(get(Dict(x), k, 0.0), v, m) for (k, v) in y]
    return Prediction(p2.device, lin(p1.time_us, p2.time_us, m),
        lin(p1.cycles_sm, p2.cycles_sm, m), lin(p1.cycles_block, p2.cycles_block, m),
        lin(p1.cycles_warp, p2.cycles_warp, m), p2.blocks_per_sm, p2.blocks_total_per_sm,
        p2.binding, p2.stalls, p2.util, lin_pairs(p1.bounds, p2.bounds),
        Dict(k => round(Int, lin(get(p1.counts, k, 0), v, m)) for (k, v) in p2.counts),
        p2.phases, p2.icache, [p2.notes; note])
end

# dynamic ops per thread, the mean over the warps of `tmap`
function op_mean(K, traces, tmap)
    n = Dict{Symbol, Float64}()
    for t in eachindex(traces), (c, v) in counts(K, traces[t])
        n[c] = get(n, c, 0.0) + v * count(==(t), tmap)
    end
    return Dict(c => round(Int, v / length(tmap)) for (c, v) in n)
end

const C_XU, C_SHFL, C_BAR, C_ADU, C_S2R = map(c -> CID[c], (:XU, :SHFL, :BAR, :ADU, :S2R))

const C_LDG, C_STG, C_LDS, C_STS = map(c -> CID[c], (:LDG, :STG, :LDS, :STS))

const C_LDL, C_STL, C_LDC, C_ATOM, C_PREF =
    map(c -> CID[c], (:LDL, :STL, :LDC, :ATOM, :PREF))

# busy-time accounts: pipes, SM units, the L2 and DRAM queues
const UNITS = (PIPES..., SMUNITS..., :l2, :dram)

const U_LSU, U_L2, U_DRAM = length(PIPES) + 1, length(UNITS) - 1, length(UNITS)

# LRU cache of instruction (key ≥ 0: line number, index key + 1) and constant-bank lines
# (key < 0, index from `index`): fill time and last use (-1: not resident) by index; a miss
# in a full cache evicts the first line of `resident` (in its iteration order) with the
# oldest last use.
struct Lines
    resident::Set{Int}
    fill::Vector{Float64}
    last::Vector{Float64}
    index::Dict{Int, Int}
    cap::Int
end
Lines(n, cap, index = Dict{Int, Int}()) =
    Lines(Set{Int}(), zeros(n), fill(-1.0, n), index, cap)

# (delay until line `key` (index i) is filled, miss)
@inline function touch!(c::Lines, key, i, t, fill)
    @inbounds c.last[i] < 0 && return (fill_line!(c, key, i, t, fill), true)
    @inbounds c.last[i] = max(c.last[i], t)
    return (max(@inbounds(c.fill[i]) - t, 0.0), false)
end

function fill_line!(c::Lines, key, i, t, fill)
    if length(c.resident) >= c.cap
        lru, m = 0, Inf
        for l in c.resident
            x = c.last[l >= 0 ? l + 1 : c.index[l]]
            isless(x, m) && (lru = l; m = x)
        end
        delete!(c.resident, lru)
        c.last[lru >= 0 ? lru + 1 : c.index[lru]] = -1.0
    end
    push!(c.resident, key)
    c.fill[i] = t + fill
    c.last[i] = t
    return fill
end

# What the simulation reads of an op, in one record: class, whether a dependency barrier
# tracks its result, the values it writes, its sources (range of `Kernel.srcs`), the keys of
# its instruction-cache lines, and the key and cache index of its constant-bank line (index
# 0: none); for memory ops the `load_hits` override (index into `SM.hvs`, 0: none), the
# partially written sectors, the bytes moved per warp access and the LSU cycles.
struct Op
    cls::UInt8
    track::Bool
    dst::Int32
    dst2::Int32
    srcs::UnitRange{Int32}
    l0::Int32
    l1::Int32
    ckey::Int32
    cidx::Int32
    hv::Int32
    partial::Float32
    b::Float64
    o::Float64
    ln::Float64
end

# ready time, stall reason, dependency barrier (0: none) and that barrier's generation when it
# took the value, of a value in one warp
struct VState
    ready::Float64
    why::UInt8
    sb::UInt8
    gen::UInt32
end

# earliest issue time (and its reason) of a warp's next op before its scheduler's pipes and
# the SM units, the row of what it then waits for in `SM.con`, and its SM unit (0: a pipe)
struct Ready
    e::Float64
    r::Int32
    c::Int8
    u::Int8
end

# State of the event-driven simulation of `nsim` blocks (of W warps, R resident) on one SM.
mutable struct SM
    const K::Kernel
    const traces::Vector{Vector{Int32}}
    const acts::Vector{Vector{Float32}} # by trace position: lanes active (empty: all)
    const tr::Vector{Int}               # by resident warp: its trace
    const tmap::Matrix{Int}             # trace of warp i of block j: tmap[i, mod1(j, end)]
    const blocks::Vector{UnitRange{Int}}
    const ops::Vector{Op}
    const srcs::Vector{Int32}
    const W::Int
    const R::Int
    const nsim::Int
    const ns::Int
    const lat_l1::Float64
    const lat_l2::Float64
    const lat_dram::Float64
    const lat_shared::Float64
    const lat_con::Float64
    const h::Hits
    const hl::Hits
    const hvs::Vector{Hits}
    const store_dram::Float64
    const coalescing::Float64
    const icache::Bool
    const hit_levels::Bool
    const scoreboards::Int
    const l2_bw::Float64
    const dram_bw::Float64

    # by class: dependent latency, pipe occupancy (and its share of the pipe's busy account),
    # first pipe, SM unit (memory: the LSU), row in `con`
    const lat::Vector{Float64}
    const occ::Vector{Float64}
    const occ_ns::Vector{Float64}
    const p1::Vector{Int}
    const smu::Vector{Int}
    const unit::Vector{Int}
    const crow::Vector{Int}

    # by scheduler: the free time (and stall reason) of what an op waits for before it issues,
    # by row: each pipe, each SM unit, each pipe pair (the pipe that is free first), and a row
    # that is always free; by pipe: the rows of its pairs and their pipes
    const con::Matrix{Float64}
    const conr::Matrix{Int32}
    const pairs::Vector{Vector{Tuple{Int, Int, Int}}}

    # instruction caches (per scheduler L0, per SM L1) and their line-fill cycles
    const L0::Vector{Lines}
    const L1::Lines
    const miss0c::Float64
    const miss1c::Float64

    # by value and warp: state; by warp: hit-level credits (global, local loads), dependency
    # barriers; by resident warp (scheduler (w - 1) % ns + 1): trace position, next op and the
    # last op of its block, issue constraints (block start or barrier release, last result,
    # start, instruction fetch), at a barrier, done, and the readiness of its next op
    const vs::Matrix{VState}
    const lvl_credit::Matrix{Float64}
    const lvl_credit_l::Matrix{Float64}
    const sb_ready::Matrix{Float64}
    const sb_why::Matrix{UInt8}
    const sb_gen::Matrix{UInt32}
    const pos::Vector{Int}
    const pc::Vector{Int}
    const stop::Vector{Int}
    const blocked::Vector{Float64}
    const wmax::Vector{Float64}
    const wstart::Vector{Float64}
    const fready::Vector{Float64}
    const atbar::Vector{Bool}
    const done::Vector{Bool}
    const ready::Vector{Ready}

    # by scheduler: rescan its warps (else its last scan's result stands until a warp is
    # ready); the earliest ready time of its warps waiting for each SM unit
    const dirty::Vector{Bool}
    const thr::Matrix{Float64}

    # SM-wide: SM-unit free times, L2/DRAM queue tails; per block slot: warps at the barrier,
    # warps alive, start and last barrier release; busy cycles per account; barrier phases;
    # blocks started and finished, times, cache misses
    const sm_free::Vector{Float64}
    q_l2::Float64
    q_dram::Float64
    const arrived::Vector{Int}
    const alive::Vector{Int}
    const bstart::Vector{Float64}
    const last_rel::Vector{Float64}
    const busy::Vector{Float64}
    const used::Vector{Bool}
    const busy_order::Vector{Int}       # UNITS in order of first use
    const phase::Vector{Float64}        # by barrier op k (index k + 1; 1: "exit")
    const phase_blk::Vector{Int}
    const phase_order::Vector{Int}      # barrier ops (0: exit) in order of first release
    started::Int
    finished::Int
    t_first::Float64
    block_time::Float64
    warp_time::Float64
    miss0::Int
    miss1::Int
end

warps(S::SM, slot) = ((slot - 1) * S.W + 1):(slot * S.W)

mlat(S::SM, h) = h.l1 * S.lat_l1 + h.l2 * S.lat_l2 + h.dram * S.lat_dram

function addbusy!(S::SM, u, x)
    S.used[u] || (S.used[u] = true; push!(S.busy_order, u))
    S.busy[u] += x
end

# an SM unit is busy longer: rescan the schedulers whose warps it may now delay
function unit_busy!(S::SM, u)
    for s in 1:(S.ns)
        S.con[length(PIPES) + u, s] = S.sm_free[u]
        S.sm_free[u] > S.thr[u, s] && (S.dirty[s] = true)
    end
end

# pipe p of scheduler s is busy until f (and so is a pair where it is the first free)
@inline function pipe_busy!(S::SM, s, p, f)
    @inbounds begin
        S.con[p, s] = f
        for (j, p1, p2) in S.pairs[p]
            pp = S.con[p2, s] < S.con[p1, s] ? p2 : p1
            S.con[j, s] = S.con[pp, s]
            S.conr[j, s] = R_PIPE + pp
        end
    end
end

# Dependency barriers: a warp tracks its variable-latency results (memory, MUFU, shuffles,
# S2R) with `scoreboards` barriers (6 per warp on sm_70+: J18 §3.2); when all are in use, a
# new result joins the barrier that clears last, so that its consumer waits for the whole
# group (how ptxas shares barriers between batched loads).
function track!(S::SM, w, t, x, y, op)
    S.scoreboards > 0 || return
    @inbounds begin
        sb = S.sb_ready
        i = argmin(j -> sb[j, w], 1:(S.scoreboards))

        # none free: join the group that clears last (its members now wait for this result
        # too)
        if sb[i, w] > t
            i = argmax(j -> sb[j, w], 1:(S.scoreboards))
            x > sb[i, w] && (sb[i, w] = x; S.sb_why[i, w] = y)
        else
            sb[i, w] = x
            S.sb_why[i, w] = y
            S.sb_gen[i, w] += 0x1
        end
        g = S.sb_gen[i, w]
        for d in (op.dst, op.dst2)
            d > 0 || continue
            v = S.vs[d, w]
            S.vs[d, w] = VState(v.ready, v.why, i % UInt8, g)
        end
    end
end

# ready time and stall reason of value v, including its dependency-barrier group
@inline function vready(S::SM, v, w)
    @inbounds begin
        x = S.vs[v, w]
        i = Int(x.sb)
        return i > 0 && x.gen == S.sb_gen[i, w] && S.sb_ready[i, w] > x.ready ?
               (S.sb_ready[i, w], S.sb_why[i, w]) : (x.ready, x.why)
    end
end

# Each load is served by one level (L1, L2 or DRAM) chosen per warp by error diffusion over
# the `hits` fractions, so that a batch of independent loads waits for its slowest member (a
# DRAM miss) rather than for the mean latency; `hit_levels = false` gives every load the
# hits-weighted mean latency and fractional queue traffic. Returns (latency, L2 traffic
# fraction, DRAM traffic fraction).
function level!(S::SM, cr, w, h)
    S.hit_levels || return (mlat(S, h), h.l2 + h.dram, h.dram)
    @inbounds begin
        a = cr[1, w] += h.l1
        b = cr[2, w] += h.l2
        c = cr[3, w] += h.dram
        i = b > a ? (c > b ? 3 : 2) : (c > a ? 3 : 1)  # the first largest credit
        cr[i, w] -= 1
    end
    return i == 1 ? (S.lat_l1, 0.0, 0.0) :
           i == 2 ? (S.lat_l2, 1.0, 0.0) : (S.lat_dram, 1.0, 1.0)
end

# Instruction supply: a warp fetches its next op when the previous one issues, through the
# per-scheduler L0 and the per-SM L1 instruction caches (LRU over lines); a miss delays the
# issue by the line-fill cost. A constant-bank operand's line is filled into the same per-SM
# cache on first use.
@inline function fetch!(S::SM, w, t, op)
    S.icache || return
    @inbounds begin
        d, m = touch!(S.L1, op.l1, op.l1 + 1, t, S.miss1c)
        S.miss1 += m
        d0, m = touch!(S.L0[(w - 1) % S.ns + 1], op.l0, op.l0 + 1, t + d, S.miss0c)
        S.miss0 += m
        f = t + d + d0
        op.cidx > 0 && (f = max(f, t + touch!(S.L1, op.ckey, op.cidx, t, S.miss1c)[1]))
        S.fready[w] = f
    end
    return
end

# Skip free ops (ready = max of sources) and handle block and warp ends; fetch the next op and
# compute its readiness.
function settle!(S::SM, w, t)
    @inbounds while true
        pc = S.pc[w]
        if pc > S.stop[w]
            tr = S.traces[S.tr[w]]
            if (S.pos[w] += 1) > length(tr)
                warp_done!(S, w, t)
                break
            end
            r = S.blocks[tr[S.pos[w]]]
            S.pc[w] = first(r)
            S.stop[w] = last(r)
            continue
        end
        op = S.ops[pc]
        if op.cls != 1
            fetch!(S, w, t, op)
            S.dirty[(w - 1) % S.ns + 1] = true  # see `warp_ready!`
            S.atbar[w] ? (S.ready[w] = Ready(Inf, R_BAR, size(S.con, 1), 0)) :
            ready!(S, w, op)
            return
        end
        x = 0.0
        y = UInt8(R_IDLE)
        for j in op.srcs
            rv, ry = vready(S, S.srcs[j], w)
            rv > x && (x = rv; y = ry)
        end
        op.dst > 0 && (S.vs[op.dst, w] = VState(x, y, 0, 0))
        S.pc[w] = pc + 1
    end
    warp_ready!(S, w)
end

# warp w ran its trace; the last warp of a block ends it and starts the next block in its slot
function warp_done!(S::SM, w, t)
    @inbounds begin
        S.done[w] = true
        S.wmax[w] = max(t + 1, S.wmax[w])
        S.warp_time += S.wmax[w] - S.wstart[w]
        slot = (w - 1) ÷ S.W + 1
        if (S.alive[slot] -= 1) == 0
            fin = maximum(v -> S.wmax[v], warps(S, slot))
            S.finished += 1
            S.block_time += fin - S.bstart[slot]
            S.finished == 1 && push!(S.phase_order, 0)
            S.phase[1] = S.phase[1] + fin - S.last_rel[slot]
            S.finished == S.R && (S.t_first = fin)
            S.started < S.nsim && (S.started += 1; start_block!(S, slot, fin, S.started))
        end
    end
end

function start_block!(S::SM, slot, t, j)
    S.bstart[slot] = S.last_rel[slot] = t
    S.alive[slot] = S.W
    S.arrived[slot] = 0
    k = mod1(j, size(S.tmap, 2))
    for (i, w) in enumerate(warps(S, slot))
        S.tr[w] = S.tmap[i, k]
        r = S.blocks[S.traces[S.tr[w]][1]]
        S.sb_ready[:, w] .= -Inf
        S.sb_gen[:, w] .+= 0x1
        S.done[w] = S.atbar[w] = false
        S.blocked[w] = S.wstart[w] = S.wmax[w] = t
        S.pos[w] = 1
        S.pc[w] = first(r)
        S.stop[w] = last(r)
        settle!(S, w, t)
    end
end

# Earliest issue time of warp w's next op from its own state (block start or barrier
# release, sources, instruction fetch), and what it waits for; the warp's scheduler adds
# its pipes and the SM units (`earliest`).
@inline function warp_ready!(S::SM, w)
    @inbounds begin
        S.dirty[(w - 1) % S.ns + 1] = true
        S.done[w] || S.atbar[w] ?
        (S.ready[w] = Ready(Inf, S.done[w] ? R_IDLE : R_BAR, size(S.con, 1), 0)) :
        ready!(S, w, S.ops[S.pc[w]])
    end
    return
end

# readiness of warp w's next op `op` (see `warp_ready!`)
@inline function ready!(S::SM, w, op)
    @inbounds begin
        e = S.blocked[w]
        r = e == S.wstart[w] ? R_IDLE : R_BAR
        for j in op.srcs
            rv, ry = vready(S, S.srcs[j], w)
            rv > e && (e = rv; r = Int(ry))
        end
        S.fready[w] > e && (e = S.fready[w]; r = R_NOINST)
        c = op.cls
        S.ready[w] = Ready(e, r, S.crow[c], S.unit[c])
    end
    return
end

# earliest issue time and its reason of ready state x on scheduler s
@inline function earliest(S::SM, x::Ready, s)
    @inbounds f = S.con[x.c, s]
    return f > x.e ? (f, Int(@inbounds S.conr[x.c, s])) : (x.e, Int(x.r))
end

# Scheduler s at time t: the warp that issues (0: none: then the earliest time one can, its
# reason, and the first warp in scan order that reaches it) in scan order, greedy-then-oldest:
# the last issued warp li, then warps s, s + ns, ... (n of them). After li, without branches
# over the warps: a bitmask of those that can issue, and the minimum. Also the earliest ready
# time of the warps that wait for each SM unit (`unit_busy!` rescans when the unit becomes
# busy later).
function scan(S::SM, s, t, li, n)
    ns = S.ns
    x = @inbounds S.ready[ifelse(li == 0, length(S.ready), li)]
    best = earliest(S, x, s)[1]
    best <= t && return (li, best, R_IDLE, 0)
    bi = 0
    th1 = ifelse(x.u == 1, x.e, Inf)
    th2 = ifelse(x.u == 2, x.e, Inf)
    mask = UInt64(0)
    @inbounds for i in 1:n
        x = S.ready[s + (i - 1) * ns]
        f = S.con[x.c, s]
        e = ifelse(f > x.e, f, x.e)
        mask |= UInt64(e <= t) << i
        th1 = ifelse(x.u == 1 && x.e < th1, x.e, th1)
        th2 = ifelse(x.u == 2 && x.e < th2, x.e, th2)
        best, bi = ifelse(e < best, (e, i), (best, bi))
    end
    @inbounds S.thr[1, s] = th1
    @inbounds S.thr[2, s] = th2
    pos(i) = i == 0 ? li : s + (i - 1) * ns
    mask != 0 && return (pos(trailing_zeros(mask)), best, R_IDLE, 0)
    best < Inf && return (0, best, earliest(S, S.ready[pos(bi)], s)[2], pos(bi))

    # every warp waits for a barrier or is done: the first other reason than idle
    for i in 1:n
        r = Int(S.ready[s + (i - 1) * ns].r)
        r == R_IDLE || return (0, best, r, 0)
    end
    return (0, best, R_IDLE, 0)
end

function issue!(S::SM, w, s, t)
    @inbounds begin
        k = S.pc[w]
        op = S.ops[k]
        c = op.cls
        if S.p1[c] > 0
            pp = Int(S.conr[S.crow[c], s]) - R_PIPE  # the second pipe if it is free first
            pipe_busy!(S, s, pp, t + S.occ[c])
            addbusy!(S, pp, S.occ_ns[c])
        elseif S.smu[c] > 0
            u = S.smu[c]
            o = c == C_ADU ? op.o : S.occ[c]  # ADU: a unit slot per pass
            S.sm_free[u] = max(S.sm_free[u], t) + o
            unit_busy!(S, u)
            addbusy!(S, length(PIPES) + u, o)
        end
        x = t + S.lat[c] + (c == C_ADU ? op.o - S.occ[c] : 0.0)
        y = UInt8(R_WAIT + c)
        if c >= FIRST_MEM
            # On a divergent side with at most half the lanes active (e.g. lane 0 of a
            # group, 4 lanes of 36), only those lanes move data (at least a sector, a line
            # and an L1 pass); more active lanes, contiguous runs, touch most of the warp's
            # sectors.
            a = isempty(S.acts) ? 1.0f0 : S.acts[S.tr[w]][S.pos[w]]
            a > 0.5 && (a = 1.0f0)
            b = a < 1 ? max(32.0, a * op.b) : op.b
            shared = c == C_LDS || c == C_STS
            loc = c == C_LDL || c == C_STL

            # An overridden load (see `load_hits`) takes its own mean latency; a prefetch
            # fills one line of L2 per warp from DRAM with the DRAM fraction of the global
            # loads, and waits for nothing.
            ml, f2, fd =
                op.hv > 0 ? (hv = S.hvs[op.hv]; (mlat(S, hv), hv.l2 + hv.dram, hv.dram)) :
                c == C_LDG || c == C_LDL || c == C_ATOM ?
                level!(S, loc ? S.lvl_credit_l : S.lvl_credit, w, loc ? S.hl : S.h) :
                c == C_PREF ? (0.0, 1.0, S.h.dram) :
                c == C_STL ? (0.0, S.hl.l2 + S.hl.dram, S.hl.dram) :
                (0.0, 1.0, S.store_dram)

            # the L1 sends a missing or stored line to L2 per cycle: a scattered access that
            # misses (or a store) takes one LSU cycle per line it touches
            o = op.o + f2 * max(0.0, op.ln - op.o)
            a < 1 && (o = max(1.0, a * o))
            S.sm_free[1] = max(S.sm_free[1], t) + o
            unit_busy!(S, 1)
            addbusy!(S, U_LSU, o)
            x =
                t + (
                    shared ? S.lat_shared + o - 1 :
                    c == C_LDC ? S.lat_con :
                    c == C_STG ? S.lat_l2 :
                    c == C_STL ? S.lat_l1 : c == C_ATOM ? max(ml, S.lat_l2) : ml
                )
            # bandwidth queues; loads wait behind earlier traffic
            if !shared && c != C_LDC
                delay = 0.0
                if f2 > 0
                    st = max(S.q_l2, t)
                    S.q_l2 = st + f2 * b * S.coalescing / S.l2_bw
                    addbusy!(S, U_L2, S.q_l2 - st)
                    st - t > delay && (delay = st - t; y = UInt8(R_L2))
                end
                # a partially written sector is first read from DRAM (read-modify-write)
                if fd > 0
                    st = max(S.q_dram, t)
                    S.q_dram =
                        st + fd * (b * S.coalescing + 32 * a * op.partial) / S.dram_bw
                    addbusy!(S, U_DRAM, S.q_dram - st)
                    st - t > delay && (delay = st - t; y = UInt8(R_DRAM))
                end
                x += delay
            end
        end
        if c == C_BAR
            slot = (w - 1) ÷ S.W + 1
            S.atbar[w] = true
            if (S.arrived[slot] += 1) == S.alive[slot]  # last arrival releases the block
                rel = t + S.lat[c]
                for v in warps(S, slot)
                    S.atbar[v] = false
                    S.blocked[v] = max(S.blocked[v], rel)
                    warp_ready!(S, v)
                end
                S.arrived[slot] = 0
                S.phase_blk[k] == 0 &&
                    (
                        S.phase_blk[k] = S.traces[S.tr[w]][S.pos[w]];
                        push!(S.phase_order, k)
                    )
                S.phase[k + 1] = S.phase[k + 1] + rel - S.last_rel[slot]
                S.last_rel[slot] = rel
            end
        end
        for d in (op.dst, op.dst2)
            d > 0 && (S.vs[d, w] = VState(x, y, 0, 0))
        end
        op.track && track!(S, w, t, x, y, op)
        S.wmax[w] = max(S.wmax[w], x, t + 1)
        S.pc[w] = k + 1
    end
    settle!(S, w, t)
end

# Event-driven simulation of `nsim` blocks (of W warps, R resident) on one SM.
function run_sm(K, dev, traces, acts, tmap, W, R, nsim; h, hl, store_dram, coalescing,
    shared_ways,
    icache, stagger, expansion, hit_levels, scoreboards, l2_bw, dram_bw,
    vhits = Dict{Int32, Hits}(), gather = :local)
    nc = length(CLASSES)
    ns = dev.n_sched
    nw = R * W
    nk = length(K.cls)
    lat, occ = zeros(nc), zeros(nc)
    p1, p2, smu = zeros(Int, nc), zeros(Int, nc), zeros(Int, nc)
    for (c, (l, rate, pipes...)) in dev.ops
        i = CID[c]
        lat[i] = l
        u = findfirst(==(pipes[1]), SMUNITS)
        if u !== nothing
            smu[i] = u
            occ[i] = 32 / rate
        else
            p1[i] = findfirst(==(pipes[1]), PIPES)
            occ[i] = 32 * ns / rate * length(pipes)
            length(pipes) > 1 && (p2[i] = findfirst(==(pipes[2]), PIPES))
        end
    end
    unit = [c >= FIRST_MEM ? 1 : smu[c] for c in 1:nc]
    pairs = unique((p1[c], p2[c]) for c in 1:nc if p2[c] > 0)
    npu = length(PIPES) + length(SMUNITS)
    crow = [
        unit[c] > 0 ? length(PIPES) + unit[c] :
        p2[c] > 0 ? npu + findfirst(==((p1[c], p2[c])), pairs) : p1[c] for c in 1:nc
    ]
    con = zeros(npu + length(pairs) + 1, ns)
    con[end, :] .= -Inf
    conr = Int32[
        j <= length(PIPES) ? R_PIPE + j :
        j <= npu ? R_SMU + j - length(PIPES) :
        j < size(con, 1) ? R_PIPE + pairs[j - npu][1] : R_IDLE for
        j in axes(con, 1), _ in 1:ns
    ]

    # ops; code offsets in the instruction caches
    ic = dev.icache
    off = zeros(nk)
    code = 0.0
    for (b, r) in enumerate(K.blocks), k in r
        off[k] = code
        K.cls[k] == 1 || haskey(K.alias, b) || (code += ic.inst)
    end
    for (b, a) in K.alias  # a callee copy runs the first copy's code
        r = K.blocks[a]
        isempty(r) ||
            foreach(((i, k),) -> off[k] = off[r[min(i, end)]], enumerate(K.blocks[b]))
    end
    nl0, nl1 = Int(code ÷ ic.line0) + 1, Int(code ÷ ic.line1) + 1  # lines of code
    index = Dict{Int, Int}()  # constant-bank line key => L1 cache index
    hvs = Hits[]
    ops = Vector{Op}(undef, nk)
    for k in 1:nk
        c = K.cls[k]
        mem = c >= FIRST_MEM
        # an access through a loaded index with `gather = :scattered`: a sector per lane
        ind = K.sectors[k] < 0 && gather === :scattered
        s = abs(K.sectors[k])
        b = !mem ? 0.0 : ind ? 1024.0 : isfinite(s) && s > 0 ? 32.0 * s : K.bytes[k] * 32.0

        # a global access takes the L1 data path for its bytes, or one cycle per bank pass
        o =
            c == C_ADU ? occ[c] * max(1.0, K.waves[k]) :
            !mem ? 0.0 :
            c == C_LDS || c == C_STS ? b * shared_ways / dev.bw.shared :
            c == C_LDC ? 1.0 : max(b * coalescing / dev.bw.l1, Float64(K.waves[k]))
        hv = 0
        (c == C_LDG || c == C_LDL || c == C_ATOM) && haskey(vhits, K.orig[k]) &&
            (push!(hvs, vhits[K.orig[k]]); hv = length(hvs))
        l = get(K.clines, K.orig[k], -1)
        ckey = l >= 0 ? -1 - l : 0
        cidx = l >= 0 ? get!(index, ckey, nl1 + length(index) + 1) : 0
        track =
            (
                mem && c != C_STG && c != C_STS && c != C_STL ||
                c in (C_XU, C_SHFL, C_ADU, C_S2R)
            ) &&
            K.dst[k] > 0
        ops[k] = Op(c, track, K.dst[k], K.dst2[k], K.src_ptr[k]:(K.src_ptr[k + 1] - 1),
            Int(off[k] ÷ ic.line0), Int(off[k] ÷ ic.line1), ckey, cidx, hv,
            ind && c == C_STG && K.bytes[k] < 32 ? 32.0f0 : K.partial[k], b, o,
            ind ? 32.0 : Float64(K.lines[k]))
    end
    nsb = max(scoreboards, 1)
    cap0, cap1 = Int(ic.l0 ÷ ic.line0), Int(ic.l1 ÷ ic.line1)
    S = SM(K, traces, acts, zeros(Int, nw), tmap, K.blocks, ops, K.srcs, W, R, nsim, ns,
        dev.lat.l1, dev.lat.l2, dev.lat.dram, dev.lat.shared, dev.lat.con, h, hl, hvs,
        store_dram, coalescing,
        icache, hit_levels, scoreboards, l2_bw, dram_bw, lat, occ, occ ./ ns, p1, smu,
        unit, crow, con, conr,
        [
            [(npu + j, q...) for (j, q) in enumerate(pairs) if p in q] for
            p in eachindex(PIPES)
        ],
        [Lines(nl0, cap0) for _ in 1:ns],
        Lines(nl1 + length(index), cap1, index), ic.miss0, ic.miss1,
        fill(VState(0.0, 0, 0, 0), K.nvalues, nw), zeros(3, nw),
        zeros(3, nw), fill(-Inf, nsb, nw), zeros(UInt8, nsb, nw),
        zeros(UInt32, nsb, nw),
        ones(Int, nw), zeros(Int, nw), zeros(Int, nw), zeros(nw), zeros(nw), zeros(nw),
        zeros(nw), fill(false, nw), fill(true, nw),
        fill(Ready(Inf, R_IDLE, size(con, 1), 0), nw + 1),
        fill(true, ns), fill(Inf, length(SMUNITS), ns), zeros(length(SMUNITS)), 0.0,
        0.0, zeros(Int, R), zeros(Int, R), zeros(R),
        zeros(R), zeros(length(UNITS)), fill(false, length(UNITS)), Int[],
        zeros(nk + 1), zeros(Int, nk), Int[], min(R, nsim), 0, 0.0, 0.0, 0.0, 0, 0)

    # first-wave start times (NaN once started)
    starts = [slot <= nsim ? (slot - 1) * float(stagger) : NaN for slot in 1:R]
    next_start = minimum(x -> isnan(x) ? Inf : x, starts)
    t = 0.0
    stall = zeros(length(REASONS))
    last_issued = zeros(Int, ns)
    reason = fill(R_IDLE, ns)
    next_e = fill(Inf, ns)
    next_w = zeros(Int, ns)  # the first warp in scan order that reaches next_e
    did = fill(false, ns)
    sched_free = zeros(ns)
    debt = zeros(ns)
    nws = [length(s:ns:nw) for s in 1:ns]  # warps of scheduler s: s, s + ns, ...
    @inbounds while S.finished < nsim
        if t >= next_start
            for slot in 1:R
                starts[slot] <= t && (start_block!(S, slot, t, slot); starts[slot] = NaN)
            end
            next_start = minimum(x -> isnan(x) ? Inf : x, starts)
        end
        anydid = false
        for s in 1:ns
            did[s] = false
            # expansion slots
            if sched_free[s] > t
                next_e[s] = sched_free[s]
                reason[s] = R_ISSUE
                S.dirty[s] = true
                continue
            end
            # Unchanged since the last scan (not dirty): nothing issues before next_e, and at
            # next_e the warp that reached it first in scan order is the one a scan would pick.
            iw = 0
            if !S.dirty[s]
                next_e[s] > t && continue
                next_e[s] == t && (iw = next_w[s])
            end
            if iw == 0
                S.dirty[s] = false
                iw, next_e[s], reason[s], next_w[s] = scan(S, s, t, last_issued[s], nws[s])
            end
            if iw > 0
                issue!(S, iw, s, t)
                last_issued[s] = iw
                did[s] = anydid = true
                debt[s] += expansion - 1
                d = floor(debt[s])
                debt[s] -= d
                sched_free[s] = t + 1 + d  # extra slots
            end
        end
        tn = anydid ? t + 1 : min(minimum(next_e), next_start)
        isfinite(tn) || error("deadlock at cycle $t")
        for s in 1:ns
            did[s] ? (stall[R_ISSUE] += 1) : (stall[reason[s]] += tn - t)
        end
        t = tn
    end
    t_end = max(t, maximum(S.wmax), S.q_l2, S.q_dram)  # queues drain before the kernel ends
    busy = Dict{Symbol, Float64}()
    foreach(u -> busy[UNITS[u]] = S.busy[u], S.busy_order)
    phases = Dict{String, Float64}()
    for k in S.phase_order
        phases[k == 0 ? "exit" : K.labels[S.phase_blk[k]] * "#" * string(k)] =
            S.phase[k + 1]
    end
    return (; t_end, t_first = R == nsim ? t_end : S.t_first, stall, busy, phases,
        blocks_done = S.finished, S.block_time, S.warp_time, code_bytes = round(Int, code),
        S.miss0, S.miss1)
end

function Base.show(io::IO, ::MIME"text/plain", p::Prediction)
    pct(x) = string(round(100x; digits = 1), "%")
    top(v, f, n = 8) = join((f(x) for x in v[1:min(end, n)]), ", ")
    println(io, "GPUPerfModel prediction on ", p.device, ": ",
        round(p.time_us; sigdigits = 4), " µs, binding: ",
        p.binding)
    println(io, "  cycles: SM ", round(Int, p.cycles_sm), ", block ",
        round(Int, p.cycles_block), ", warp ",
        round(Int, p.cycles_warp), "; blocks/SM ", p.blocks_per_sm, " resident, ",
        p.blocks_total_per_sm, " total")
    println(io, "  code: ", round(p.icache.code_bytes / 1024; digits = 1),
        " KB; instruction-cache misses per warp: L0 ",
        round(Int, p.icache.l0_misses), ", L1 ", round(Int, p.icache.l1_misses))
    println(io, "  scheduler cycles: ", top(p.stalls, ((k, v),) -> "$k $(pct(v))"))
    println(io, "  utilization: ", top(p.util, ((k, v),) -> "$k $(pct(v))"))
    println(
        io,
        "  bounds (cycles/SM; units busy, latency = 1 warp alone): ",
        top(p.bounds, ((k, v),) -> "$k $(round(Int, v))", 6),
    )
    println(
        io,
        "  ops/thread: ",
        top(sort(collect(p.counts); by = last, rev = true), ((k, v),) -> "$k $v", 20),
    )
    isempty(p.phases) || println(
        io,
        "  phases (cycles/block): ",
        top(p.phases, ((l, x),) -> "$l $(round(Int, x))", 6),
    )
    foreach(n -> println(io, "  note: ", n), p.notes)
end

# ─── Device-free LLVM IR (CUDA.jl must be loaded by the caller) ──────────────

"""
    device_ir(f, tt::Type{<:Tuple}; sm = 80) -> String

Optimized LLVM IR of kernel `f` for device argument types `tt`, as `@cuda`
compiles it, without a GPU (IR validation and libdevice linking disabled, so
`__nv_*` calls remain declarations; kernels that allocate still compile).
"""
function device_ir(f, tt::Type{<:Tuple}; sm = 80)
    id = Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")
    haskey(Base.loaded_modules, id) || error("device_ir needs CUDA.jl: `import CUDA` first")
    CUDA = Base.root_module(id)
    C = isdefined(CUDA, :CUDACompilerParams) ? CUDA : CUDA.CUDACore  # CUDA.jl 5 / 6
    G = C.GPUCompiler
    cap = VersionNumber(sm ÷ 10, sm % 10)
    params =
        hasfield(C.CUDACompilerParams, :cap) ? C.CUDACompilerParams(; cap, ptx = v"7.8") :
        C.CUDACompilerParams(; sm = C.SMVersion(cap), ptx = v"7.8")
    config =
        G.CompilerConfig(G.PTXCompilerTarget(; cap, ptx = v"7.8"), params; kernel = true,
            libraries = false,
            validate = false, always_inline = true)
    job = G.CompilerJob(G.methodinstance(typeof(f), tt), config)
    return G.JuliaContext(_ -> string(G.compile(:llvm, job)[1]))
end

end # module
