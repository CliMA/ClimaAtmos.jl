# Self-contained tests for gpu_perf_model.jl. Run with an environment that has
# CUDA.jl (no GPU needed): julia --project=<env> gpu_perf_model_tests.jl
using Test

import CUDA

using CUDA: CuDeviceVector, threadIdx, blockIdx, blockDim, sync_threads

using Base.Cartesian: @nexprs, @ntuple

include(joinpath(@__DIR__, "gpu_perf_model.jl"))

using .GPUPerfModel

using .GPUPerfModel: DEVICES, walk, with

const A100 = DEVICES[:sm_80]

const V32 = CuDeviceVector{Float32, 1}

const L2 = (l1 = 0.0, l2 = 1.0, dram = 0.0)

gid() = (blockIdx().x - 1) * blockDim().x + threadIdx().x

fp32_lat(dev) = dev.ops[:FP32][1]

bound(p, k) = last(p.bounds[findfirst(b -> first(b) == k, p.bounds)])

stall(p, k) =
    (i = findfirst(s -> first(s) == k, p.stalls); i === nothing ? 0.0 : last(p.stalls[i]))

kernel(f, tt) = parse_llvm(device_ir(f, tt))

# pure latency/throughput answers: no compulsory fetch misses
sim(K; kw...) = simulate(K; icache = false, kw...)

# ─── Known-answer kernels (straight-line code: the IR keeps loops rolled) ────

function fma_chain(a, x, ::Val{N}) where {N}
    i = threadIdx().x
    v = @inbounds a[i]
    @nexprs 8 k -> (@nexprs 16 l -> v = muladd(v, x, 1.5f0))
    N == 256 && @nexprs 8 k -> (@nexprs 16 l -> v = muladd(v, x, 2.5f0))
    @inbounds a[i] = v
    return nothing
end

function fma_independent(a, x)
    i = gid()
    v = @inbounds a[i]
    @nexprs 8 j -> v_j = v + Float32(j)
    @nexprs 32 k -> (@nexprs 8 j -> v_j = muladd(v_j, x, 1.5f0))
    @inbounds a[i] = +(@ntuple(8, v)...)
    return nothing
end

function copy_kernel(b, a)
    i = gid()
    @inbounds b[i] = a[i]
    return nothing
end

ex2(x) = ccall("llvm.nvvm.ex2.approx.ftz.f", llvmcall, Float32, (Float32,), x)

function mufu_kernel(a)
    i = gid()
    v = @inbounds a[i]
    @nexprs 8 j -> v_j = v + Float32(j)
    @nexprs 16 k -> (@nexprs 8 j -> v_j = ex2(v_j))
    @inbounds a[i] = +(@ntuple(8, v)...)
    return nothing
end

function barrier_phases(a, x)
    i = threadIdx().x
    v = @inbounds a[i]
    @nexprs 8 j -> v_j = v + Float32(j)
    @nexprs 16 k -> (@nexprs 8 j -> v_j = muladd(v_j, x, 1.5f0))
    sync_threads()
    @nexprs 16 k -> (@nexprs 8 j -> v_j = muladd(v_j, x, 2.5f0))
    sync_threads()
    @inbounds a[i] = +(@ntuple(8, v)...)
    return nothing
end

function chase(a)
    j = @inbounds a[threadIdx().x]
    @nexprs 16 k -> j = @inbounds a[j]
    @inbounds a[threadIdx().x] = j
    return nothing
end

function rolled_loop(a, x)
    i = threadIdx().x
    v = @inbounds a[i]
    for _ in 1:1000  # not fully unrolled by LLVM 15/16
        v = muladd(v, x, 1.5f0)
    end
    @inbounds a[i] = v
    return nothing
end

@testset "GPUPerfModel" begin
    @testset "device table and occupancy" begin
        for (name, dev) in DEVICES
            @test all(
                c -> haskey(dev.ops, c),
                (:FP32, :IADD, :ALU, :SETP, :IMAD, :IMADW, :FP64, :XU, :CONV,
                    :SHFL, :BAR, :BRA, :S2R, :UNI),
            )
            @test all(row -> row[1] > 0 && row[2] > 0, values(dev.ops))
            @test dev.lat.l1 < dev.lat.l2 < dev.lat.dram && dev.bw.dram < dev.bw.l2
            @test 0 < dev.icache.l0 <= dev.icache.l1 &&
                  dev.icache.miss0 < dev.icache.miss1 && dev.l1_bytes > 0
        end

        # A100 residency of 64-thread blocks (as reported by ptxas and CUDA):
        @test occupancy(A100, 64, 255) == 4    # 255 regs: 2 warps/scheduler
        @test occupancy(A100, 64, 168) == 6    # 168 regs
        @test occupancy(A100, 64, 32, 42224) == 3  # static shared memory binds
        @test occupancy(A100, 1024, 32) == 2 && occupancy(A100, 32) == 32
        @test occupancy(DEVICES[:sm_60], 256, 64) == 4
    end

    @testset "IR → op class mapping" begin
        wrap(
            body;
            args = "float %a, float %b, float %c, i32 %n, i64 %m, float addrspace(1)* %p",
        ) = """
define ptx_kernel void @k($args) {
entry:
$body
  ret void
}"""
        ops(body; kw...) =
            (c = op_counts(parse_llvm(wrap(body; kw...))); delete!(c, :BRA); c)
        @test ops("%1 = fmul contract float %a, %b\n%2 = fadd contract float %1, %c") ==
              Dict(:FP32 => 1)
        @test ops("%1 = fmul float %a, %b\n%2 = fadd float %1, %c") == Dict(:FP32 => 2)
        @test ops(
            "%1 = fmul contract float %a, %b\n%2 = fadd contract float %1, %c\n%3 = fadd float %1, %2",
        ) ==
              Dict(:FP32 => 3)  # two users: no fusion
        @test ops("%1 = fdiv float %a, %b") == Dict(:XU => 1, :FP32 => 5, :ALU => 1)
        @test ops("%1 = fdiv fast float %a, %b") == Dict(:XU => 1, :FP32 => 1)
        @test ops("%1 = fdiv float 1.000000e+00, %a") ==
              Dict(:XU => 1, :FP32 => 2, :IADD => 1, :SETP => 1)
        @test ops("%1 = fpext float %a to double\n%2 = fdiv double %1, %1")[:FP64] == 8
        @test ops("%1 = call float @llvm.sqrt.f32(float %a)") ==
              Dict(:XU => 1, :FP32 => 3, :IADD => 1, :SETP => 1)
        @test ops("%1 = call float @llvm.fma.f32(float %a, float %b, float %c)") ==
              Dict(:FP32 => 1)
        @test ops("%1 = fneg float %a\n%2 = call float @llvm.fabs.f32(float %1)") == Dict()
        @test ops("%1 = fcmp olt float %a, %b\n%2 = select i1 %1, float %a, float %b") ==
              Dict(:SETP => 1, :ALU => 1)
        # Address arithmetic: one IMAD.WIDE per variable leaf; constant offsets fold.
        gep(
            idx,
        ) = "%x = zext i32 %n to i64\n$idx\n%v = load float, float addrspace(1)* %q, align 4"
        @test ops(
            gep("%q = getelementptr inbounds float, float addrspace(1)* %p, i64 %x"),
        ) ==
              Dict(:IMADW => 1, :LDG => 1)
        @test ops(
            gep("%q = getelementptr inbounds float, float addrspace(1)* %p, i64 7"),
        ) == Dict(:LDG => 1)
        two(add) = """
            %s = $add i32 %n, 1
            %x1 = zext i32 %s to i64
            %x0 = zext i32 %n to i64
            %q0 = getelementptr inbounds float, float addrspace(1)* %p, i64 %x0
            %q1 = getelementptr inbounds float, float addrspace(1)* %p, i64 %x1
            %v0 = load float, float addrspace(1)* %q0, align 4
            %v1 = load float, float addrspace(1)* %q1, align 4"""
        @test ops(two("add nuw"))[:IMADW] == 1   # shared address + immediate offset
        @test ops(two("add"))[:IMADW] == 2       # zext(add) without nuw is a new leaf
        @test ops(two("add"))[:IADD] == 1
        @test ops("""
            %y = add nsw i64 %m, 5
            %q = getelementptr inbounds float, float addrspace(1)* %p, i64 %y
            %v = load float, float addrspace(1)* %q, align 4""") ==
              Dict(:IMADW => 1, :IMAD => 1, :LDG => 1)
        @test ops("%y = add i64 %m, 5\n%c1 = icmp slt i64 %y, 9") ==
              Dict(:IADD => 2, :SETP => 2)
        @test ops("%x = zext i32 %n to i64\n%c1 = icmp ult i64 %x, 63") == Dict(:SETP => 1)
        @test ops("%x = mul i32 %n, %n\n%y = mul i64 %m, %m") ==
              Dict(:IMAD => 3, :IMADW => 1)
        @test ops("%x = udiv i32 %n, 1008") == Dict(:IMAD => 1, :ALU => 1)
        @test ops("%x = urem i32 %n, 64") == Dict(:ALU => 1)
        @test sum(values(ops("%x = sdiv i32 %n, %n"))) == 16
        @test ops("%x = sext i32 %n to i64\n%y = trunc i64 %x to i32") == Dict(:ALU => 1)

        # Memory spaces and sizes; special registers; barriers; shuffles.
        K = parse_llvm(wrap("""
            %q = addrspacecast float addrspace(1)* %p to float addrspace(3)*
            %v = load float, float addrspace(3)* %q, align 4
            store float %v, float addrspace(1)* %p, align 4
            %r = bitcast float addrspace(1)* %p to <4 x float> addrspace(1)*
            %w = load <4 x float>, <4 x float> addrspace(1)* %r, align 16
            %t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
            %u = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
            call void @llvm.nvvm.barrier0()
            %s = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %t, i32 1, i32 31)"""))
        c = op_counts(K)
        @test c[:LDS] == 1 && c[:STG] == 1 && c[:LDG] == 1 && c[:S2R] == 1 &&
              c[:BAR] == 1 && c[:SHFL] == 1
        @test sort(K.bytes[K.cls .>= GPUPerfModel.FIRST_MEM]) == [4, 4, 16]
        K = parse_llvm(wrap("%e = call float @__nv_atanf(float %a)"))
        @test op_counts(K)[:XU] == 1 && any(contains("__nv_atanf"), K.notes)
    end

    @testset "control flow: trip counts and branch probabilities" begin
        loop(cmp, init, next, extra = "", step = "add i64 %i, 1") = parse_llvm("""
            define ptx_kernel void @k(float %x, i64 %nn) {
            entry:
              $extra
              br label %L
            L:
              %i = phi i64 [ $init, %entry ], [ %j, %L ]
              %v = phi float [ 0.0, %entry ], [ %w, %L ]
              %w = fadd float %v, %x
              %j = $step
              %c = $cmp
              br i1 %c, label %E, label %L
            E:
              ret void
            }""")
        tripcount(K) = op_counts(K; dynamic = true)[:FP32]
        @test tripcount(loop("icmp eq i64 %i, 64", 1, "%j")) == 64
        @test tripcount(loop("icmp eq i64 %j, 10", 0, "%j")) == 10
        @test tripcount(loop("icmp sge i64 %j, 7", 0, "%j")) == 7

        # PCR stride doubling
        @test tripcount(loop("icmp sgt i64 %j, 62", 1, "%j", "", "shl i64 %i, 1")) == 6
        assumed = "%a = icmp eq i64 %nn, 63\n  call void @llvm.assume(i1 %a)"
        @test tripcount(loop("icmp eq i64 %j, %nn", 0, "%j", assumed)) == 63
        K = loop("icmp eq i64 %j, %nn", 0, "%j")  # unknown: default, override by label
        @test tripcount(K) == 1 &&
              op_counts(K; dynamic = true, trips = Dict("L" => 5))[:FP32] == 5
        @test any(contains("unknown trip"), simulate(K; threads = 32, blocks = 1).notes)

        # the kernel's outermost loop is noted for `trips`; `loops` applies to it too, also
        # in `op_counts`; `conds` that never take a loop's exit leave it after its trips
        @test any(contains("outermost loop: set `trips = Dict(\"L\" => n)`"),
            simulate(K; threads = 32, blocks = 1).notes)
        @test op_counts(K; dynamic = true, loops = :short)[:FP32] == 16
        nexit = String[]
        @test op_counts(K; dynamic = true, trips = Dict("L" => 5), conds = Dict("c" => 0.0),
            notes = nexit)[:FP32] == 5 && any(contains("no exit with a positive"), nexit)
        @test_throws ErrorException simulate(K; threads = 32, device = :sm_90)
        K = parse_llvm("""
            define ptx_kernel void @k(float %x, i1 %c) {
            entry:
              br i1 %c, label %T, label %F, !prof !1
            T:
              %a = fadd float %x, %x
              br label %F
            F:
              br i1 %c, label %X, label %R
            X:
              call void @gpu_report_exception()
              unreachable
            R:
              ret void
            }
            !1 = !{!"branch_weights", i32 1, i32 3}""")
        @test K.probs[1] ≈ [0.25, 0.75] && isequal(K.probs[3], [0.0, NaN])

        # T skipped (credit 0.25), X cold
        @test op_counts(K; dynamic = true) == Dict(:BRA => 3)
        @test op_counts(K; dynamic = true, probs = Dict(("entry", "T") => 1.0))[:FP32] == 1
    end

    @testset "list scheduling recovers ILP hidden by IR order" begin
        chain(c, x) = join(
            ("%$c$i = fadd float %$(i == 1 ? x : "$c$(i - 1)"), %$x" for i in 1:16),
            "\n  ",
        )
        K = parse_llvm("""
            define ptx_kernel void @k(float %x, float %y) {
            entry:
              $(chain("a", "x"))
              $(chain("b", "y"))
              ret void
            }""")
        p0 = sim(K; threads = 32, blocks = 1)
        p1 = sim(K; threads = 32, blocks = 1, reorder = true)
        @test p0.cycles_warp ≈ 32 * fp32_lat(A100) rtol = 0.05    # two chains back to back
        @test p1.cycles_warp ≈ 16 * fp32_lat(A100) rtol = 0.1     # interleaved
        @test op_counts(GPUPerfModel.reorder(K, A100, 0.0)) == op_counts(K)
    end

    @testset "list scheduling under a live-value budget" begin
        # 8 independent loads summed by a chain: unbounded scheduling issues every load
        # first (one latency); a budget of 2 live values issues a load only after the chain
        # frees one (a latency per load).
        K = parse_llvm(
            """
 define ptx_kernel void @k(float addrspace(1)* %p, float addrspace(1)* %q) {
 entry:
   $(join(("%a$i = load float, float addrspace(1)* %p, align 4" for i in 1:8), "\n  "))
   $(join(("%s$i = fadd float %$(i == 1 ? "a1" : "s$(i - 1)"), %a$(i + 1)" for i in 1:7), "\n  "))
   store float %s7, float addrspace(1)* %q, align 4
   ret void
 }""",
        )
        pos(R) = [i for (i, c) in enumerate(R.cls) if GPUPerfModel.CLASSES[c] === :LDG]
        @test pos(GPUPerfModel.reorder(K, A100, 500.0)) == 1:8
        R2 = GPUPerfModel.reorder(K, A100, 500.0, 2)
        @test pos(R2) == [1, 2, 4, 6, 8, 10, 12, 14] && op_counts(R2) == op_counts(K)
        p8 = sim(K; threads = 32, blocks = 1, reorder = true)
        p2 = sim(K; threads = 32, blocks = 1, reorder = 2)
        @test p2.cycles_warp > 2 * p8.cycles_warp
    end

    @testset "SASS front end" begin
        sass = """
                Function : _Z4testPf
            .headerflags    @"EF_CUDA_SM80 EF_CUDA_PTX_SM(EF_CUDA_SM80)"
                /*0000*/                   MOV R1, c[0x0][0x28] ;                     /* 0x00000a0000017a02 */
                /*0010*/                   S2R R0, SR_TID.X ;
                /*0020*/                   IMAD.WIDE R2, R0, 0x4, c[0x0][0x160] ;
                /*0030*/                   LDG.E.64 R4, [R2.64] ;
                /*0040*/                   DFMA R4, R4, R4, 1 ;
                /*0050*/                   MUFU.RCP R6, R5 ;
                /*0060*/                   ISETP.GE.AND P0, PT, R6, 0x10, PT ;
                /*0070*/              @!P0 BRA 0x40 ;
                /*0080*/                   STG.E.64 [R2.64], R4 ;
                /*0090*/                   EXIT ;
                /*00a0*/                   BRA 0xa0;"""
        K = parse_sass(sass)
        @test length(K.blocks) == 4 && K.succs[2] == [2, 3] && isempty(K.succs[3])
        @test op_counts(K) ==
              Dict(:ALU => 1, :S2R => 1, :IMADW => 1, :LDG => 1, :FP64 => 1, :XU => 1,
            :SETP => 1,
            :BRA => 3, :STG => 1)
        @test op_counts(K; dynamic = true, trips = Dict("0x0040" => 10))[:FP64] == 10
        @test K.bytes[findfirst(==(GPUPerfModel.CID[:LDG]), K.cls)] == 8

        # MUFU reads R5, the high half of the 64-bit DFMA result: a dependency.
        p = sim(K; threads = 32, blocks = 1, hits = L2, trips = Dict("0x0040" => 10))
        @test p.binding == :latency
        @test stall(p, :wait_FP64) > 0.05 && stall(p, :wait_XU) > 0.2 &&
              stall(p, :wait_LDG) > 0.2
        dfma, mufu = A100.ops[:FP64][1], A100.ops[:XU][1]
        @test p.cycles_warp > 10 * (dfma + mufu + 4) + A100.lat.l2

        # Predicate-sink destinations, F64 conversion pairs, spills.
        K = parse_sass("""
                /*0000*/                   LDL.64 R6, [R1+0x10] ;
                /*0010*/                   SHFL.BFLY PT, R3, R6, 0x1, 0x1f ;
                /*0020*/                   F2F.F64.F32 R4, R3 ;
                /*0030*/                   DADD R8, R4, R4 ;
                /*0040*/                   STL [R1+0x20], R5 ;
                /*0050*/                   EXIT ;""")
        srcs(k) = K.srcs[K.src_ptr[k]:(K.src_ptr[k + 1] - 1)]

        # R3 from SHFL; R5 from F2F
        @test srcs(3) == [K.dst[2]] && K.dst2[3] != 0 && K.dst2[3] in srcs(5)
        @test K.bytes[1] == 8 && K.bytes[5] == 4
        l1 = sim(K; threads = 32, blocks = 1)
        l2 = sim(K; threads = 32, blocks = 1, local_hits = L2)
        @test l2.cycles_warp - l1.cycles_warp ≈ A100.lat.l2 - A100.lat.l1 atol = 1
        @test stall(l1, :wait_SHFL) > 0 && stall(l1, :wait_LDL) > 0

        # `expansion`: issue slots per op (ptxas adds SASS beyond the op classes); fractions
        # accumulate.
        U = parse_sass(
            join(
                (
                    "        /*0000*/   UIADD3 UR$(i % 8), UR$((i + 1) % 8), 0x1, URZ ;"
                    for i in 1:1000
                ), "\n") * "\n EXIT ;",
        )
        c1 = sim(U; threads = 32, blocks = 1).cycles_warp
        @test sim(U; threads = 32, blocks = 1, expansion = 1.5).cycles_warp ≈ 1.5c1 rtol =
            0.02
        @test sim(U; threads = 32, blocks = 1, expansion = 3).cycles_warp ≈ 3c1 rtol = 0.02
    end

    @testset "known answers (device-free compiled kernels)" begin
        # Dependent FMA chain: latency-bound, 128 extra FFMAs cost 128 × latency.
        k128 = kernel(fma_chain, Tuple{V32, Float32, Val{128}})
        k256 = kernel(fma_chain, Tuple{V32, Float32, Val{256}})
        @test op_counts(k256)[:FP32] - op_counts(k128)[:FP32] == 128
        p128 = sim(k128; threads = 32, blocks = 1, hits = L2)
        p256 = sim(k256; threads = 32, blocks = 1, hits = L2)
        @test p256.cycles_warp - p128.cycles_warp ≈ 128 * fp32_lat(A100) rtol = 0.01
        @test p256.binding == :latency && first(p256.stalls[1]) == :wait_FP32
        for sm in (:sm_60, :sm_70)
            d =
                sim(k256; device = sm, threads = 32, blocks = 1, hits = L2).cycles_warp -
                sim(k128; device = sm, threads = 32, blocks = 1, hits = L2).cycles_warp
            @test d ≈ 128 * fp32_lat(DEVICES[sm]) rtol = 0.01
        end

        # Independent FMAs at full occupancy: FMA-pipe-bound (64 results/clk/SM).
        k = kernel(fma_independent, Tuple{V32, Float32})
        nfp = op_counts(k)[:FP32]
        @test nfp in (8 + 256 + 7, 8 + 256 + 8)  # LLVM 16 SLP: one extra undef-lane op
        p = sim(k; threads = 256, blocks = 108 * 8 * 4, regs = 32, hits = L2)
        @test p.blocks_per_sm == 8 && p.binding == :fma
        @test p.cycles_sm ≈ 4 * 2048 * nfp / A100.ops[:FP32][2] rtol = 0.08
        @test stall(p, :pipe_fma) > 0.3 && stall(p, :issue) > 0.45

        # Streaming copy: DRAM-bound at the device bandwidth.
        k = kernel(copy_kernel, Tuple{V32, V32})
        n = 108 * 2048 * 64
        p = sim(k; threads = 256, blocks = n ÷ 256, regs = 16)
        @test p.binding == :dram
        @test p.time_us - 2 ≈ 8n / (A100.bw.dram * A100.clock_ghz * 1e3) rtol = 0.08
        @test stall(p, :dram_bandwidth) > 0.4

        # MUFU-heavy: XU-bound (16 results/clk/SM).
        k = kernel(mufu_kernel, Tuple{V32})
        @test op_counts(k)[:XU] == 128
        p = sim(k; threads = 256, blocks = 108 * 8 * 4, regs = 32, hits = L2)
        @test p.binding == :xu
        @test p.cycles_sm ≈ 4 * 2048 * 128 / A100.ops[:XU][2] rtol = 0.08

        # Barrier-separated phases: barriers wait for the slowest warp. With 5
        # warps, scheduler 1 runs two of them, so a phase of n FFMAs takes
        # 2 × n × 2 pipe cycles, and the other warps wait at the barrier. (n is
        # counted from the IR: LLVM 16 sinks some phase-1 FMAs past the barrier.)
        k = kernel(barrier_phases, Tuple{V32, Float32})
        @test op_counts(k)[:BAR] == 2
        bars = findall(==(GPUPerfModel.CID[:BAR]), k.cls)
        n2 = count(==(GPUPerfModel.CID[:FP32]), k.cls[bars[1]:bars[2]])
        bar, occ = A100.ops[:BAR][1], 32 * 4 / A100.ops[:FP32][2]

        # IR order differs across LLVMs
        p5 = sim(k; threads = 160, blocks = 1, hits = L2, reorder = true)
        p4 = sim(k; threads = 128, blocks = 1, hits = L2, reorder = true)
        phase2(p) = sort(
            filter(x -> first(x) != "exit", p.phases);
            by = x -> parse(Int, split(first(x), '#')[2]),
        )[2][2]
        @test phase2(p5) ≈ 2 * n2 * occ + bar rtol = 0.03
        @test phase2(p4) ≈ n2 * occ + bar rtol = 0.03
        @test stall(p5, :barrier) > 0.25 > stall(p4, :barrier)

        # Occupancy sweep: more warps hide the dependent-load latency until the
        # L2 bandwidth (per-SM share) binds.
        k = kernel(chase, Tuple{CuDeviceVector{Int32, 1}})
        @test op_counts(k)[:LDG] == 17
        thr(R) = (p = sim(k; threads = 64, blocks = 108 * R, blocks_per_sm = R, hits = L2);
            (R * 2 * 17 / p.cycles_sm, p))
        (t2, p2), (t4, p4), (t16, p16), (t32, p32) = thr(2), thr(4), thr(16), thr(32)
        @test t4 / t2 ≈ 2 rtol = 0.05            # latency-bound: throughput ∝ warps
        @test p2.binding == :latency
        @test t32 / t16 < 1.15 && p32.binding == :l2   # bandwidth-bound plateau

        # one 128-B warp load per (bytes / BW) cycles
        @test t32 ≈ A100.bw.l2 / 108 / 128 rtol = 0.1

        # Loops may stay rolled (or partially unrolled) in IR: trips from the induction
        # variable.
        k = kernel(rolled_loop, Tuple{V32, Float32})
        @test !isempty(k.trips) && op_counts(k; dynamic = true)[:FP32] == 1000
    end

    @testset "instruction supply" begin
        # One warp streaming N independent FADDs (16 B each): every L0 (256 B) and L1 (512
        # B) line misses once, and each miss delays the next issue by its fill cost (J18
        # Fig. 3.4-3.5).
        straight(N) = parse_llvm(
            "define ptx_kernel void @k(float %x, float %y) {\nentry:\n" *
            join(("  %a$i = fadd float %x, %y" for i in 1:N), "\n") * "\n  ret void\n}",
        )
        ic = A100.icache
        K = straight(4096)
        code = 16 * 4097
        n0, n1 = cld(code, ic.line0), cld(code, ic.line1)
        p0 = sim(K; threads = 32, blocks = 108)
        p1 = simulate(K; threads = 32, blocks = 108)
        @test p1.icache == (code_bytes = code, l0_misses = n0, l1_misses = n1)
        @test p1.cycles_warp - p0.cycles_warp ≈ n0 * ic.miss0 + n1 * ic.miss1 rtol = 0.02
        @test first(p1.stalls[1]) == :no_instruction && stall(p1, :no_instruction) > 0.7

        # Lockstep warps wait for the same fills (each line is fetched once per SM, once per
        # scheduler for L0); a block starting after the first one finished hits L1 (64 KB <
        # 128 KB) and misses only its scheduler's L0.
        p2 = simulate(K; threads = 32, blocks = 216, blocks_per_sm = 2)
        p3 = simulate(
            K;
            threads = 32,
            blocks = 216,
            blocks_per_sm = 2,
            stagger = 2 * p1.cycles_warp,
        )
        @test p2.cycles_sm ≈ p1.cycles_warp && p2.icache.l1_misses ≈ n1 / 2 &&
              p2.icache.l0_misses == n0
        @test p3.cycles_sm - 2 * p1.cycles_warp ≈ p0.cycles_warp + n0 * ic.miss0 rtol = 0.05
        @test p3.icache.l1_misses ≈ n1 / 2

        # Later blocks in a slot hit L1 unless the code exceeds it (LRU streaming); loop
        # bodies that fit miss once.
        p4 = simulate(K; threads = 32, blocks = 324, blocks_per_sm = 1)

        # 192 KB > 128 KB
        p5 = simulate(straight(12288); threads = 32, blocks = 324, blocks_per_sm = 1)
        @test p4.icache.l1_misses ≈ n1 / 3 &&
              p5.icache.l1_misses == cld(16 * 12289, ic.line1)
        k = kernel(rolled_loop, Tuple{V32, Float32})
        p6 = simulate(k; threads = 32, blocks = 1)
        @test p6.icache.l1_misses == cld(p6.icache.code_bytes, ic.line1) &&
              p6.icache.code_bytes < ic.l0
    end

    @testset "register spills" begin
        # V values defined up front and consumed in order: with `cap` registers, V - cap of
        # them are evicted (furthest next use first), stored once and reloaded once; the
        # reloads wait on L1.
        V = 100
        S = parse_llvm(
            "define ptx_kernel void @k(float %x, float addrspace(1)* %p) {\nentry:\n" *
            join(("  %v$i = fadd float %x, $(float(i))" for i in 1:V), "\n") *
            "\n  %s1 = fadd float %v1, %v2\n" *
            join(("  %s$i = fadd float %s$(i - 1), %v$(i + 1)" for i in 2:(V - 1)), "\n") *
            "\n  store float %s$(V - 1), float addrspace(1)* %p\n  ret void\n}",
        )
        K, st = spill(S, 40)
        @test st.peak == V && st.spilled == st.words == st.stl == st.ldl == V - 40
        @test op_counts(K)[:LDL] == op_counts(K)[:STL] == V - 40 &&
              spill(S, V)[2].spilled == 0
        p = sim(S; threads = 32, blocks = 1, regs = 40, spills = true)
        @test p.counts[:LDL] == V - 40 && stall(p, :wait_LDL) > 0.3 &&
              any(contains("fits L1"), p.notes)
        # Local memory beyond the L1 capacity hits L2 (unless `local_hits` says otherwise).
        q = sim(S; threads = 1024, blocks = 108, regs = 40, spills = true)
        @test any(contains("exceeds L1"), q.notes) &&
              q.cycles_warp > p.cycles_warp + (A100.lat.l2 - A100.lat.l1)
        @test sim(
            S;
            threads = 1024,
            blocks = 108,
            regs = 40,
            spills = true,
            local_hits = L2,
        ).cycles_warp == q.cycles_warp

        # 64-bit values take two registers; kernel arguments and predicates none.
        D = parse_llvm(
            "define ptx_kernel void @k(double %x, i64 %n, double addrspace(1)* %p) {\nentry:\n" *
            join(("  %v$i = fadd double %x, $(float(i))" for i in 1:10), "\n") *
            "\n  %c = fcmp olt double %v1, %v2\n  %s1 = select i1 %c, double %v1, double %v2\n" *
            join(("  %s$i = fadd double %s$(i - 1), %v$i" for i in 2:10), "\n") *
            "\n  store double %s10, double addrspace(1)* %p\n  ret void\n}",
        )
        @test spill(D, 20)[2] == (
            cap = 20,
            peak = 20,
            peak_at = 11,
            spilled = 0,
            words = 0,
            stl = 0,
            ldl = 0,
            remat = 0,
        )
        @test spill(D, 19)[2].words == 2 && spill(D, 19)[2].spilled == 1

        # Values live across a loop and used after it are stored once and reloaded once,
        # whatever the trip count.
        W = 60
        L = parse_llvm(
            "define ptx_kernel void @k(float %x, float addrspace(1)* %p) {\nentry:\n" *
            join(("  %v$i = fadd float %x, $(float(i))" for i in 1:W), "\n") *
            "\n  br label %L\nL:\n  %i = phi i64 [ 0, %entry ], [ %j, %L ]\n  %a = phi float [ 0.0, %entry ], [ %b, %L ]\n" *
            "  %b = fadd float %a, %x\n  %j = add i64 %i, 1\n  %c = icmp eq i64 %j, 100\n  br i1 %c, label %E, label %L\n" *
            "E:\n  %s1 = fadd float %v1, %b\n" *
            join(("  %s$i = fadd float %s$(i - 1), %v$i" for i in 2:W), "\n") *
            "\n  store float %s$W, float addrspace(1)* %p\n  ret void\n}",
        )
        lin, lout = GPUPerfModel.liveness(L)

        # the loop's phis are defined in its header
        @test length(lin[2]) == W && length(lout[1]) == W
        K, st = spill(L, 32)
        cls(b, c) = count(==(GPUPerfModel.CID[c]), K.cls[K.blocks[b]])
        n = W + 3 - 32  # the loop keeps its i64 counter and float accumulator in registers
        @test st.peak == W + 3 && st.stl == st.ldl == n && cls(1, :STL) == n &&
              cls(3, :LDL) == n && cls(2, :LDL) == 0
        p = sim(L; threads = 32, blocks = 1, regs = 32, spills = true)
        @test p.counts[:FP32] == 2W + 100 && p.counts[:LDL] == p.counts[:STL] == n

        # Fields of kernel-argument aggregates stay in the constant bank: no registers,
        # nothing to spill.
        A = parse_llvm(
            "define ptx_kernel void @k({ float, [40 x float] } %s, float addrspace(1)* %p) {\nentry:\n" *
            join(
                (
                    "  %f$i = extractvalue { float, [40 x float] } %s, 1, $(i - 1)" for
                    i in 1:40
                ),
                "\n",
            ) *
            "\n  %s1 = fadd float %f1, %f2\n" *
            join(("  %s$i = fadd float %s$(i - 1), %f$(i + 1)" for i in 2:39), "\n") *
            "\n  store float %s39, float addrspace(1)* %p\n  ret void\n}",
        )
        @test spill(A, 8)[2].peak == 1 && spill(A, 8)[2].spilled == 0

        # Rematerialization (`remat = true`): values of one cheap integer op whose sources
        # are available (special registers here) are recomputed at their next use instead of
        # being stored and reloaded.
        M = parse_llvm(
            "define ptx_kernel void @k(float %x, float addrspace(1)* %p) {\nentry:\n" *
            "  %t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n  %q = shl i32 %t, 2\n" *
            join(("  %v$i = fadd float %x, $(float(i))" for i in 1:V), "\n") *
            "\n  %s1 = fadd float %v1, %v2\n" *
            join(("  %s$i = fadd float %s$(i - 1), %v$(i + 1)" for i in 2:(V - 1)), "\n") *
            "\n  %u = add i32 %q, %t\n  %z = zext i32 %u to i64\n  %a = getelementptr float, float addrspace(1)* %p, i64 %z\n" *
            "  store float %s$(V - 1), float addrspace(1)* %a\n  ret void\n}",
        )
        Kn, sn = spill(M, 40)
        Kr, sr = spill(M, 40; remat = true)
        @test sn.remat == 0 && sn.spilled == V - 40 + 2 && sr.remat == 2 &&
              sr.spilled == V - 40
        @test op_counts(Kr)[:S2R] == 2 && op_counts(Kr)[:ALU] == 2 &&
              op_counts(Kn)[:S2R] == 1
        @test sim(M; threads = 32, blocks = 1, regs = 40, spills = true, remat = true).counts[:LDL] ==
              V - 40
    end

    @testset "load/store vectorization" begin
        # Four adjacent, 16-byte-aligned float accesses through one base and index become
        # one LDS.128 / STS.128; a run whose first element is only 4-byte aligned, or that
        # mixes loads and stores, stays scalar.
        vec(al1, al2) = parse_llvm("""
            @shmem = external addrspace(3) global [0 x i8], align 32
            define ptx_kernel void @k(float %x, i64 %i, float addrspace(1)* %p) {
            entry:
              %b = bitcast [0 x i8] addrspace(3)* @shmem to [4 x float] addrspace(3)*
              %p0 = getelementptr inbounds [4 x float], [4 x float] addrspace(3)* %b, i64 %i, i64 0
              %p1 = getelementptr inbounds [4 x float], [4 x float] addrspace(3)* %b, i64 %i, i64 1
              %p2 = getelementptr inbounds [4 x float], [4 x float] addrspace(3)* %b, i64 %i, i64 2
              %p3 = getelementptr inbounds [4 x float], [4 x float] addrspace(3)* %b, i64 %i, i64 3
              store float %x, float addrspace(3)* %p0, align $al1
              store float %x, float addrspace(3)* %p1, align 4
              store float %x, float addrspace(3)* %p2, align 8
              store float %x, float addrspace(3)* %p3, align 4
              call void @llvm.nvvm.barrier0()
              %v0 = load float, float addrspace(3)* %p0, align $al2
              %v1 = load float, float addrspace(3)* %p1, align 4
              %v2 = load float, float addrspace(3)* %p2, align 8
              %v3 = load float, float addrspace(3)* %p3, align 4
              %s = fadd float %v0, %v3
              store float %s, float addrspace(1)* %p
              ret void
            }
            declare void @llvm.nvvm.barrier0()""")
        K = vec(16, 16)
        @test op_counts(K)[:STS] == 1 && op_counts(K)[:LDS] == 1
        @test sort(Int.(K.bytes[K.cls .== GPUPerfModel.CID[:LDS]])) == [16] &&
              Int.(K.bytes[K.cls .== GPUPerfModel.CID[:STS]]) == [16]
        # elements 2-3 pair up as 8-byte accesses
        @test op_counts(vec(4, 4)) == merge(op_counts(K), Dict(:STS => 3, :LDS => 3))

        # The scalar values of a vectorized load are ready with the lead access; the store
        # waits for every value.
        p = sim(K; threads = 32, blocks = 1, hits = L2)
        @test p.counts[:LDS] == 1 && p.counts[:STS] == 1 && stall(p, :wait_LDS) > 0
        @test spill(K, 255)[2].peak <= spill(vec(4, 4), 255)[2].peak
    end

    @testset "hit levels and dependency barriers" begin
        # N independent loads, then one use of all of them: with per-load levels at hits
        # 0.5/0/0.5 the batch waits for its DRAM member (the slowest); the hits-weighted
        # mean latency is shorter.
        batch(N) = parse_llvm(
            "define ptx_kernel void @k(float addrspace(1)* %p) {\nentry:\n" *
            join(
                (
                    "  %a$i = getelementptr float, float addrspace(1)* %p, i64 $(32i)\n  %v$i = load float, float addrspace(1)* %a$i"
                    for i in 1:N
                ),
                "\n",
            ) *
            "\n  %s1 = fadd float %v1, %v2\n" *
            join(("  %s$i = fadd float %s$(i - 1), %v$(i + 1)" for i in 2:(N - 1)), "\n") *
            "\n  store float %s$(N - 1), float addrspace(1)* %p\n  ret void\n}",
        )
        half = (l1 = 0.5, l2 = 0.0, dram = 0.5)
        B = batch(4)
        lv = sim(B; threads = 32, blocks = 1, hits = half)
        mean = sim(B; threads = 32, blocks = 1, hits = half, hit_levels = false)
        @test lv.cycles_warp - mean.cycles_warp ≈
              (A100.lat.dram - (A100.lat.l1 + A100.lat.dram) / 2) atol = 4
        @test stall(lv, :wait_LDG) > stall(mean, :wait_LDG)

        # 6 dependency barriers per warp: a 7th load issued while six are outstanding joins
        # the group that clears last, so the consumer of that group's earlier member waits
        # for the 7th load too; unlimited barriers do not.
        C = parse_llvm(
            "define ptx_kernel void @k(float addrspace(1)* %p, float %x) {\nentry:\n" *
            join(
                (
                    "  %a$i = getelementptr float, float addrspace(1)* %p, i64 $(32i)\n  %v$i = load float, float addrspace(1)* %a$i"
                    for i in 1:6
                ),
                "\n",
            ) *
            "\n  %c1 = fadd float %x, 1.0\n" *
            join(("  %c$i = fadd float %c$(i - 1), $(float(i))" for i in 2:20), "\n") *
            "\n  %a7 = getelementptr float, float addrspace(1)* %p, i64 224\n  %v7 = load float, float addrspace(1)* %a7\n" *
            "  %d1 = fadd float %v6, %c20\n" *
            join(("  %d$i = fadd float %d$(i - 1), $(float(i))" for i in 2:20), "\n") *
            "\n  %t = fadd float %d20, %v7\n  %u = fadd float %t, %v1\n  store float %u, float addrspace(1)* %p\n  ret void\n}",
        )
        s6 = sim(C; threads = 32, blocks = 1, hits = L2)
        s0 = sim(C; threads = 32, blocks = 1, hits = L2, scoreboards = 0)

        # v6's chain starts when v7 lands
        @test s6.cycles_warp - s0.cycles_warp ≈ 20 * fp32_lat(A100) atol = 12
        @test sim(batch(6); threads = 32, blocks = 1, hits = L2).cycles_warp ==
              sim(batch(6); threads = 32, blocks = 1, hits = L2, scoreboards = 0).cycles_warp
    end

    @testset "prefetches and per-load hit overrides" begin
        # Prefetch ops (the generic intrinsic and inline PTX) are LSU accesses of one line
        # per warp without a result.
        K = parse_llvm(
            """
 define ptx_kernel void @k(float addrspace(1)* %p, i8 addrspace(1)* %q, i64 %m) {
 entry:
   call void @llvm.prefetch.p1i8(i8 addrspace(1)* %q, i32 0, i32 3, i32 1)
   call void asm sideeffect "prefetch.global.L2 [\$0];", "l"(i64 %m)
   %a = getelementptr float, float addrspace(1)* %p, i64 32
   %v = load float, float addrspace(1)* %a
   %w = fadd float %v, 1.0
   store float %w, float addrspace(1)* %p
   ret void
 }
 declare void @llvm.prefetch.p1i8(i8 addrspace(1)*, i32, i32, i32)""",
        )
        c = op_counts(K)
        @test c[:PREF] == 2 && c[:LDG] == 1 && !any(contains("branch-class"), K.notes)
        pref = findall(==(GPUPerfModel.CID[:PREF]), K.cls)
        ld = findfirst(==(GPUPerfModel.CID[:LDG]), K.cls)
        @test K.bytes[pref] == [4, 4] && K.dst[pref] == [0, 0]

        # The fadd waits for the (loaded) DRAM latency of its load; overriding that load to
        # L2 shortens the wait by the latency difference, and the prefetches wait for
        # nothing (they only take LSU slots and queue traffic).
        d = sim(K; threads = 32, blocks = 1)
        l2 = sim(K; threads = 32, blocks = 1, load_hits = Dict(ld => L2))
        @test d.cycles_warp - l2.cycles_warp ≈ A100.lat.dram - A100.lat.l2 atol = 2
        @test stall(l2, :wait_PREF) == 0 && l2.counts[:PREF] == 2 &&
              last(l2.util[findfirst(u -> first(u) == :dram, l2.util)]) > 0
        @test_throws ErrorException sim(
            K;
            threads = 32,
            blocks = 1,
            load_hits = Dict(pref[1] => L2),
        )

        # Without prefetches the kernel keeps the same dependent latency: a prefetch costs
        # only its LSU access.
        K0 = parse_llvm("""
            define ptx_kernel void @k(float addrspace(1)* %p) {
            entry:
              %a = getelementptr float, float addrspace(1)* %p, i64 32
              %v = load float, float addrspace(1)* %a
              %w = fadd float %v, 1.0
              store float %w, float addrspace(1)* %p
              ret void
            }""")
        d0 = sim(K0; threads = 32, blocks = 1)
        @test d.cycles_warp - d0.cycles_warp ≈ 2 atol = 1

        # The override is keyed by the loaded value, so it survives list scheduling and
        # spilling (which renumber ops).
        B = parse_llvm(
            "define ptx_kernel void @k(float addrspace(1)* %p) {\nentry:\n" *
            join(
                (
                    "  %a$i = getelementptr float, float addrspace(1)* %p, i64 $(32i)\n  %v$i = load float, float addrspace(1)* %a$i"
                    for i in 1:24
                ),
                "\n",
            ) *
            "\n  %s1 = fadd float %v1, %v2\n" *
            join(("  %s$i = fadd float %s$(i - 1), %v$(i + 1)" for i in 2:23), "\n") *
            "\n  store float %s23, float addrspace(1)* %p\n  ret void\n}",
        )
        loads = findall(==(GPUPerfModel.CID[:LDG]), B.cls)
        ready = Dict(k => (l1 = 1.0, l2 = 0.0, dram = 0.0) for k in loads)
        for kw in ((; reorder = true), (; spills = true, regs = 8))
            q = sim(B; threads = 32, blocks = 1, kw...)
            r = sim(B; threads = 32, blocks = 1, load_hits = ready, kw...)
            @test r.cycles_warp < q.cycles_warp && stall(r, :wait_LDG) < stall(q, :wait_LDG)
        end
    end

    @testset "SASS control flow: guarded exits, calls, slow paths" begin
        # A guarded EXIT gets a kernel-end successor (so a loop can leave there); CALL is an
        # edge to the callee; the `$__internal` slow-path subroutine and the block that only
        # calls it are cold (probability 0).
        sass = """
                /*0000*/                   S2R R0, SR_TID.X ;
                /*0010*/                   ISETP.GE.AND P0, PT, R0, 0x40, PT ;
                /*0020*/              @P0  EXIT ;
            .L_x_0:
                /*0030*/                   FFMA R2, R0, R0, 1 ;
                /*0040*/                   ISETP.GT.U32.AND P1, PT, R2, 0x1ffffff, PT ;
                /*0050*/              @P1  BRA `(.L_x_1) ;
                /*0060*/                   CALL.REL.NOINC `(\$__internal_0_\$__cuda_sm20_rcp_rn_f32_slowpath) ;
                /*0070*/                   BRA `(.L_x_2) ;
            .L_x_1:
                /*0080*/                   MUFU.RCP R3, R2 ;
            .L_x_2:
                /*0090*/                   IADD3 R0, R0, 0x1, RZ ;
                /*00a0*/                   ISETP.GE.AND P2, PT, R0, 0x80, PT ;
                /*00b0*/              @P2  EXIT ;
                /*00c0*/                   BRA `(.L_x_0) ;
            \$__internal_0_\$__cuda_sm20_rcp_rn_f32_slowpath:
                /*00d0*/                   FADD R3, R2, R2 ;
                /*00e0*/                   RET.REL.NODEC R4 `(_Z4test) ;"""
        K = parse_sass(sass)
        @test K.labels[end] == "EXIT" && isempty(K.succs[end]) &&
              K.succs[1][1] == length(K.blocks)
        slow = findfirst(startswith("\$__internal"), K.labels)
        callb = findfirst(b -> slow in K.succs[b], eachindex(K.blocks))
        chk = findfirst(
            b -> callb in K.succs[b] && length(K.succs[b]) == 2,
            eachindex(K.blocks),
        )  # the range check

        # the MUFU path, not the call, is taken
        @test K.probs[chk][findfirst(==(callb), K.succs[chk])] == 0.0
        trace = walk(K; trips = Dict(".L_x_0" => 5), conds = Dict("P0" => 0.0))
        @test count(==(2), trace) == 5 && !(slow in trace) && trace[end] == length(K.blocks)
        @test op_counts(
            K;
            dynamic = true,
            trips = Dict(".L_x_0" => 5),
            conds = Dict("P0" => 0.0),
        )[:XU] == 5
    end

    @testset "loops never exit through cold edges" begin
        # A bounds check inside the loop leads to `unreachable`: the walk leaves the loop at
        # its latch, not there.
        K = parse_llvm("""
            define ptx_kernel void @k(float addrspace(1)* %p, i64 %n) {
            entry:
              br label %L
            L:
              %i = phi i64 [ 0, %entry ], [ %j, %ok ]
              %c = icmp ult i64 %i, %n
              br i1 %c, label %ok, label %err
            err:
              call void @throw()
              unreachable
            ok:
              %a = getelementptr float, float addrspace(1)* %p, i64 %i
              %v = load float, float addrspace(1)* %a
              %w = fadd float %v, 1.0
              store float %w, float addrspace(1)* %a
              %j = add i64 %i, 1
              %d = icmp eq i64 %j, 10
              br i1 %d, label %E, label %L
            E:
              ret void
            }
            declare void @throw()""")
        n = op_counts(K; dynamic = true)
        @test n[:LDG] == 10 && n[:STG] == 10 && walk(K)[end] == length(K.blocks)
    end

    @testset "serial loop-carried chains" begin
        # A recurrence through a phi is ready when the previous trip produced it: n
        # dependent multiplies per trip cost n FP32 latencies per trip, whatever the trip
        # count; with the multiplier loaded inside the loop (128 B further each trip: no line
        # reuse), the next trip's load is issued in order behind the previous multiply, so its
        # latency (L1 33 or L2 208 cycles) is exposed once per trip.
        loop(N, n, inside) = parse_llvm("""
            define ptx_kernel void @k(float addrspace(1)* %x, float addrspace(1)* %y) {
            top:
              %x0 = load float, float addrspace(1)* %x, align 4
              br label %loop
            loop:
              %acc = phi float [ 1.0, %top ], [ %m$n, %loop ]
              %i = phi i64 [ 0, %top ], [ %inext, %loop ]
              %k = mul i64 %i, 32
              %p = getelementptr inbounds float, float addrspace(1)* %x, i64 %k
              %v = load float, float addrspace(1)* %p, align 4
              %m0 = fadd float %acc, 0.0
            $(join(("  %m$j = fmul float %m$(j - 1), $(inside ? "%v" : "%x0")" for j in 1:n), '\n'))
              %inext = add nuw nsw i64 %i, 1
              %c = icmp eq i64 %inext, $N
              br i1 %c, label %exit, label %loop
            exit:
              store float %m$n, float addrspace(1)* %y, align 4
              ret void
            }""")
        L1 = (l1 = 1.0, l2 = 0.0, dram = 0.0)
        trip(n, inside, hits) =
            (
                sim(loop(128, n, inside); threads = 32, blocks = 108, hits).cycles_warp -
                sim(loop(64, n, inside); threads = 32, blocks = 108, hits).cycles_warp
            ) / 64
        # register recurrence per trip
        @test trip(12, false, L1) - trip(8, false, L1) ≈ 4 * fp32_lat(A100) rtol = 0.02

        # + the loop control issued behind it
        @test 9 * fp32_lat(A100) < trip(8, false, L1) < 9 * fp32_lat(A100) + 30

        # the in-order load path is exposed
        @test trip(8, true, L1) > trip(8, false, L1)

        # ... at its full latency
        @test trip(8, true, L2) - trip(8, true, L1) ≈ A100.lat.l2 - A100.lat.l1 rtol = 0.02
        p = sim(loop(64, 8, true); threads = 32, blocks = 108, hits = L2)
        @test p.binding == :latency && first(p.stalls[1]) == :wait_LDG
    end

    @testset "warp access footprint and alloca reloads" begin
        # Each global access is charged the 32-byte sectors its 32 lanes touch, evaluated
        # from the IR address.
        access(idx...) = parse_llvm("""
            define ptx_kernel void @k(float addrspace(1)* %x, i64 %n, { i64 } %s) {
            top:
              %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
              %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
              %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
              %b = mul i32 %ntid, %bid
              %t32 = add i32 %tid, %b
              %t = zext i32 %t32 to i64
              $(join(idx, "\n  "))
              %p = getelementptr inbounds float, float addrspace(1)* %x, i64 %idx
              %v = load float, float addrspace(1)* %p, align 4
              %d = bitcast float addrspace(1)* %x to double addrspace(1)*
              %q = getelementptr inbounds double, double addrspace(1)* %d, i64 %t
              %w = load double, double addrspace(1)* %q, align 8
              %f = fptrunc double %w to float
              %r = fadd float %v, %f
              store float %r, float addrspace(1)* %p, align 4
              ret void
            }
            declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
            declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
            declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()""")
        sectors(K) = [
            Int(K.sectors[k]) for
            k in eachindex(K.cls) if GPUPerfModel.CLASSES[K.cls[k]] in (:LDG, :STG)
        ]

        # coalesced: bytes × 32 / 32
        @test sectors(access("%idx = add i64 %t, 0")) == [4, 8, 4]

        # one thread per 64-level column
        @test sectors(access("%idx = mul i64 %t, 64")) == [32, 8, 32]
        @test sectors(access("%idx = mul i64 %t, 2")) == [8, 8, 8]       # every other float

        # warp-uniform (kernel argument)
        @test sectors(access("%idx = add i64 %n, 3")) == [1, 8, 1]
        @test sectors(
            access("%s0 = extractvalue { i64 } %s, 0", "%idx = add i64 %s0, %t"),
        ) == [4, 8, 4]

        # Mixed-radix column indices with run-time extents recompose to the column stride
        # (extents ≥ a warp assumed).
        ij = ("%i = urem i64 %t, %n", "%j = udiv i64 %t, %n")
        @test sectors(
            access(
                ij...,
                "%c = mul i64 %j, %n",
                "%k = add i64 %c, %i",
                "%idx = mul i64 %k, 64",
            ),
        ) == [32, 8, 32]
        @test sectors(
            access(
                ij...,
                "%c = mul i64 %j, 7",
                "%k = add i64 %c, %i",
                "%idx = shl i64 %k, 6",
            ),
        ) == [32, 8, 32]
        @test sectors(
            access("%i = and i64 %t, 3", "%j = lshr i64 %t, 2", "%c = shl i64 %j, 2",
                "%k = or i64 %c, %i",
                "%idx = mul i64 %k, 64"),
        ) == [32, 8, 32]

        # Indirect indexing (an index loaded from memory) is unknown: coalesced, with a
        # note.
        K = access("%ip = getelementptr inbounds float, float addrspace(1)* %x, i64 %t",
            "%iv = load float, float addrspace(1)* %ip",
            "%idx = fptosi float %iv to i64")
        @test sectors(K) == [4, 0, 8, 0] &&
              any(n -> occursin("not a function of the lane", n), K.notes)
        # The footprint drives the L2/DRAM traffic and the LSU time; `coalescing` multiplies
        # it further.
        l2(K; kw...) = bound(sim(K; threads = 32, blocks = 108, hits = L2, kw...), :l2)
        Kc, Ks = access("%idx = add i64 %t, 0"), access("%idx = mul i64 %t, 64")
        @test l2(Ks) / l2(Kc) ≈ (32 + 8 + 32) / (4 + 8 + 4) rtol = 0.01
        @test l2(Ks; coalescing = 2.0) ≈ 2 * l2(Ks) rtol = 0.01
        @test sectors(spill(Ks, 255)[1]) == sectors(Ks) &&
              sectors(GPUPerfModel.reorder(Ks, A100, 208.0)) == sectors(Ks)
        # An argument struct forced into an alloca is local memory: its reloads are LDL, not
        # global loads.
        K = parse_llvm("""
            define ptx_kernel void @k({ float, i64 } %s, float addrspace(1)* %x) {
            top:
              %a = alloca { float, i64 }, align 8
              %f = extractvalue { float, i64 } %s, 0
              %fp = getelementptr inbounds { float, i64 }, { float, i64 }* %a, i64 0, i32 0
              store float %f, float* %fp, align 8
              %v = load float, float* %fp, align 8
              store float %v, float addrspace(1)* %x, align 4
              ret void
            }""")
        @test op_counts(K) == Dict(:STL => 1, :LDL => 1, :STG => 1, :BRA => 1)
        # So is a global pointer kept in such a struct: its store and reload through
        # `float addrspace(1)**` (a generic pointer to a global pointer) are STL/LDL, the
        # access through the reloaded pointer LDG/STG.
        K = parse_llvm("""
            define ptx_kernel void @k(float addrspace(1)* %x) {
            top:
              %a = alloca float addrspace(1)*, align 8
              store float addrspace(1)* %x, float addrspace(1)** %a, align 8
              %p = load float addrspace(1)*, float addrspace(1)** %a, align 8
              %v = load float, float addrspace(1)* %p, align 4
              store float %v, float addrspace(1)* %p, align 4
              ret void
            }""")
        @test op_counts(K) == Dict(:STL => 1, :LDL => 1, :LDG => 1, :STG => 1, :BRA => 1)
    end

    @testset "latency under load, run clock, launch floor, constant lines, SASS frame" begin
        # A chain of 4 dependent DRAM loads (pointer chase).
        chase = parse_llvm(
            "define ptx_kernel void @k(i64 addrspace(1)* %p) {\nentry:\n" *
            "  %v0 = ptrtoint i64 addrspace(1)* %p to i64\n" *
            join(
                (
                    "  %a$i = inttoptr i64 %v$(i - 1) to i64 addrspace(1)*\n  %v$i = load i64, i64 addrspace(1)* %a$i"
                    for i in 1:4
                ),
                "\n",
            ) * "\n  store i64 %v4, i64 addrspace(1)* %p\n  ret void\n}",
        )

        # `loaded`: the DRAM latency beyond L2 is 1/(1 - u) times its unloaded value at the
        # DRAM utilization u of a run at `lat.dram_loaded`, at most that value; the chain's
        # 1-warp latency bound shows it. A lone warp: the unloaded latency; 2048 threads per
        # SM: the measured loaded latency; 256: in between.
        lat(p) = bound(p, :latency)
        q0 = sim(chase; threads = 32, blocks = 1, loaded = false)
        @test lat(sim(chase; threads = 32, blocks = 1)) ≈ lat(q0) atol = 2
        q = sim(chase; threads = 1024, blocks = 216)
        @test lat(q) - lat(q0) ≈ 4 * (A100.lat.dram_loaded - A100.lat.dram) atol = 2
        p = sim(chase; threads = 256, blocks = 108)
        m = match(r"DRAM latency (\d+) cycles at (\d+)% use", join(p.notes, "\n"))
        l, u = parse(Int, m[1]), parse(Int, m[2]) / 100
        @test A100.lat.dram < l < A100.lat.dram_loaded
        @test lat(p) - lat(q0) ≈ 4 * (l - A100.lat.dram) atol = 6
        @test l ≈ A100.lat.l2 + (A100.lat.dram - A100.lat.l2) / (1 - u) rtol = 0.02

        # `clock_ghz`: µs of the run's cycles at that SM clock, plus `launch_us` (2 µs) per
        # kernel; the DRAM bandwidth is fixed in bytes/s, so at 1.14 GHz the DRAM moves
        # 1.41/1.14 more bytes per cycle.
        a = sim(chase; threads = 32, blocks = 1, loaded = false)
        b = sim(chase; threads = 32, blocks = 1, loaded = false, clock_ghz = 1.14)
        @test a.cycles_sm == b.cycles_sm && b.time_us - 2 ≈ b.cycles_sm / 1140 &&
              a.time_us - 2 ≈ a.cycles_sm / 1410
        @test sim(chase; threads = 32, blocks = 1, loaded = false, launch_us = 0).time_us ≈
              a.time_us - 2
        a = sim(chase; threads = 1024, blocks = 216)
        b = sim(chase; threads = 1024, blocks = 216, clock_ghz = 1.14)
        @test bound(b, :dram) / bound(a, :dram) ≈ 1.14 / 1.41

        # A constant-bank operand's 64-byte line is filled on its first use per SM through the
        # L1 instruction cache (Nsight `imc_miss`): the first fill overlaps the cold
        # instruction fetch, two more lines cost a fill each, a repeated line hits.
        cops(offs) = parse_sass(
            join(
                (
                    "        /*$(lpad(string(16i; base = 16), 4, '0'))*/   IMAD.MOV.U32 R$i, RZ, RZ, $o ;"
                    for (i, o) in enumerate(offs)
                ),
                "\n",
            ) * "\n        EXIT ;",
        )
        imm = sim(cops(fill("0x1", 4)); threads = 32, blocks = 1, icache = true).cycles_warp
        offs = ["c[0x0][0x160]", "c[0x0][0x1a0]", "c[0x0][0x164]", "c[0x0][0x1e0]"]
        @test sim(cops(offs); threads = 32, blocks = 1, icache = true).cycles_warp - imm ≈
              2 * A100.icache.miss1 atol = 4

        # SASS local memory: the stack frame is the highest offset off R1 plus the access
        # size; when the resident threads' frames exceed the L1 beside shared memory, local
        # loads hit L2.
        F(off) = parse_sass("""
                /*0000*/                   STL [R1+$off], R2 ;
                /*0010*/                   LDL R3, [R1+$off] ;
                /*0020*/                   FADD R4, R3, R3 ;
                /*0030*/                   STG.E [R6.64], R4 ;
                /*0040*/                   EXIT ;""")
        @test F("0x10").frame == 0x14 && F("0x2000").frame == 0x2004
        s = sim(F("0x10"); threads = 32, blocks = 1)
        l = sim(F("0x2000"); threads = 32, blocks = 1)
        @test any(contains("fits L1"), s.notes) && any(contains("exceeds L1"), l.notes)
        @test l.cycles_warp - s.cycles_warp ≈ A100.lat.l2 - A100.lat.l1 atol = 2
    end

    @testset "misaligned columns, partial-sector stores, pure libdevice calls" begin
        # 64-thread blocks over nv-level Float32 columns (level = tid % 64, column = tid ÷ 64
        # of the global thread index): a 63-level column is 252 bytes, so the warps' 128-byte
        # accesses start at all 8 offsets mod 32 and touch (4 + 7 × 5) / 8 sectors on
        # average; a store writes the first and last of them partially (2 × 7/8).
        column(nv) = parse_llvm("""
            define ptx_kernel void @k(float addrspace(1)* %x) {
            top:
              %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
              %t = zext i32 %tid to i64
              %lev = and i64 %t, 63
              %col = lshr i64 %t, 6
              %c = mul i64 %col, $nv
              %idx = add i64 %c, %lev
              %p = getelementptr inbounds float, float addrspace(1)* %x, i64 %idx
              %v = load float, float addrspace(1)* %p, align 4
              %r = fadd float %v, %v
              store float %r, float addrspace(1)* %p, align 4
              ret void
            }
            declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()""")
        mem(K) =
            [k for k in eachindex(K.cls) if GPUPerfModel.CLASSES[K.cls[k]] in (:LDG, :STG)]
        K63, K64 = column(63), column(64)
        @test K63.sectors[mem(K63)] ≈ [4.875, 4.875] && K63.partial[mem(K63)] ≈ [0, 1.75]
        @test K64.sectors[mem(K64)] ≈ [4, 4] && K64.partial[mem(K64)] ≈ [0, 0]

        # A partially written sector is first read from DRAM (read-modify-write): DRAM
        # traffic per warp (4.875 + 4.875 + 1.75) vs (4 + 4) sectors.
        dram(K) = bound(sim(K; threads = 64, blocks = 3456), :dram)
        @test dram(K63) / dram(K64) ≈ 11.5 / 8 rtol = 0.01

        # Libdevice calls are pure: a repeated call with the same arguments in the same or a
        # dominating block reuses the result, a call with constant arguments folds; the
        # call in the branch that the first block does not dominate stays.
        K = parse_llvm("""
            define ptx_kernel void @k(float addrspace(1)* %x, i1 %c) {
            top:
              %v = load float, float addrspace(1)* %x, align 4
              %a = call float @__nv_expf(float %v)
              %b = call float @__nv_expf(float %v)
              %k = call float @__nv_cbrtf(float 2.0)
              br i1 %c, label %l, label %r
            l:
              %d = call float @__nv_expf(float %v)
              %w = call float @__nv_expf(float %a)
              br label %r
            r:
              %e = call float @__nv_expf(float %a)
              ret void
            }
            declare float @__nv_expf(float)
            declare float @__nv_cbrtf(float)""")
        @test op_counts(K)[:XU] == 3  # %a, %w and %e: one MUFU per libm placeholder

        # Libdevice calls with exact one-instruction semantics map like the LLVM intrinsics:
        # fabs is a free operand modifier, copysign two logic ops (as llvm.copysign), and
        # frcp_rn the IEEE reciprocal sequence of `fdiv 1.0, x` (MUFU + 2 FFMA + range check).
        K = parse_llvm("""
            define ptx_kernel void @k(float addrspace(1)* %x) {
            top:
              %v = load float, float addrspace(1)* %x, align 4
              %a = call float @__nv_fabsf(float %v)
              %b = call float @__nv_copysignf(float %a, float %v)
              %c = call float @__nv_frcp_rn(float %b)
              store float %c, float addrspace(1)* %x, align 4
              ret void
            }
            declare float @__nv_fabsf(float)
            declare float @__nv_copysignf(float, float)
            declare float @__nv_frcp_rn(float)""")
        R = parse_llvm("""
            define ptx_kernel void @k(float addrspace(1)* %x) {
            top:
              %v = load float, float addrspace(1)* %x, align 4
              %c = fdiv float 1.0, %v
              store float %c, float addrspace(1)* %x, align 4
              ret void
            }""")
        n = op_counts(K)
        @test n[:ALU] == 2 && !any(contains("libdevice"), K.notes)
        @test n == merge(+, op_counts(R), Dict(:ALU => 2))

        # Other libdevice functions expand to the instructions ptxas emits for them (sm_80
        # SASS): expf = FFMA.SAT, FFMA.RM, FADD, 2 FFMA, MUFU.EX2, FMUL, SHF and 2 constant
        # moves; logf = 26 instructions with a 13-FFMA polynomial; functions without a
        # measured sequence keep the noted placeholder.
        lib(f) = parse_llvm("""
            define ptx_kernel void @k(float addrspace(1)* %x) {
            top:
              %v = load float, float addrspace(1)* %x, align 4
              %c = call float @__nv_$f(float %v)
              store float %c, float addrspace(1)* %x, align 4
              ret void
            }
            declare float @__nv_$f(float)""")
        n = op_counts(lib("expf"))
        @test n == Dict(:LDG => 1, :STG => 1, :BRA => 1, :FP32 => 7, :ALU => 2, :XU => 1)
        @test sum(values(op_counts(lib("logf")))) == 26 + 3
        @test any(contains("libdevice call __nv_fmodf"), lib("fmodf").notes)
        # sinf/cosf: Cody-Waite reduction and a polynomial, 30/31 instructions, a 14-op chain
        @test sum(values(op_counts(lib("cosf")))) == 31 + 3 &&
              sum(values(op_counts(lib("sinf")))) == 30 + 3
        # double pow: the call of `__internal_accurate_pow` (a double-double log, then exp)
        # with the caller's special-value tests, 128 instructions, 82 of them on the FP64 pipe
        pw = parse_llvm("""
            define ptx_kernel void @k(double addrspace(1)* %x) {
            top:
              %v = load double, double addrspace(1)* %x, align 8
              %c = call double @__nv_pow(double %v, double 8.000000e-01)
              store double %c, double addrspace(1)* %x, align 8
              ret void
            }
            declare double @__nv_pow(double, double)""")
        n = op_counts(pw)
        @test n[:FP64] == 82 && n[:XU] == 1 && sum(values(n)) == 128 + 3
        @test !any(contains("libdevice"), pw.notes)
        # value tests are one compare (DSETP on the FP64 pipe), llabs the 64-bit abs of
        # llvm.abs.i64
        vt = parse_llvm("""
            define ptx_kernel void @k(double addrspace(1)* %x, float addrspace(1)* %y,
                                      i64 addrspace(1)* %z) {
            top:
              %v = load double, double addrspace(1)* %x, align 8
              %w = load float, float addrspace(1)* %y, align 4
              %n = load i64, i64 addrspace(1)* %z, align 8
              %a = call i32 @__nv_isnand(double %v)
              %b = call i32 @__nv_finitef(float %w)
              %m = call i64 @__nv_llabs(i64 %n)
              %ab = or i32 %a, %b
              %e = sext i32 %ab to i64
              %r = add i64 %m, %e
              store i64 %r, i64 addrspace(1)* %z, align 8
              ret void
            }
            declare i32 @__nv_isnand(double)
            declare i32 @__nv_finitef(float)
            declare i64 @__nv_llabs(i64)""")
        n = op_counts(vt)
        @test n[:FP64] == 1 && n[:SETP] == 3 && !haskey(n, :XU)
        @test !any(contains("libdevice"), vt.notes)
    end

    @testset "guards, notes, SASS keys and local stores" begin
        # An unset bounds check enters the kernel body; `conds` still overrides it.
        K = parse_llvm("""
            define ptx_kernel void @k(float addrspace(1)* %x, i64 %n) {
            entry:
              %t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
              %i = zext i32 %t to i64
              %out = icmp sge i64 %i, %n
              br i1 %out, label %done, label %body
            body:
              %p = getelementptr inbounds float, float addrspace(1)* %x, i64 %i
              %v = load float, float addrspace(1)* %p, align 4
              %w = fadd float %v, 1.0
              store float %w, float addrspace(1)* %p, align 4
              br label %done
            done:
              ret void
            }""")
        p = sim(K; threads = 256, blocks = 108)
        @test p.counts[:LDG] == 1 && any(contains("guards assumed"), p.notes)
        @test !haskey(
            sim(K; threads = 256, blocks = 108, conds = Dict("out" => 1.0)).counts,
            :LDG,
        )
        @test_throws ErrorException sim(
            K;
            threads = 32,
            blocks = 1,
            conds = Dict("out" => 2.0),
        )
        @test any(
            contains("unused"),
            sim(K; threads = 32, blocks = 1, trips = Dict("nope" => 3)).notes,
        )

        # SASS: notes deduplicated and all printed; `load_hits` overrides one load, not every
        # load into the same register; `spills` keeps the constant-bank lines; only the first
        # function of a multi-function dump is read.
        S = parse_sass("""
                /*0000*/                   LDG.E R4, [R2.64] ;
                /*0010*/                   VOTEU.ANY UR4, UPT, PT ;
                /*0020*/                   FADD R6, R4, R4 ;
                /*0030*/                   VOTEU.ANY UR5, UPT, PT ;
                /*0040*/                   LDG.E R4, [R8.64] ;
                /*0050*/                   FADD R7, R4, R6 ;
                /*0060*/                   IMAD.MOV.U32 R9, RZ, RZ, c[0x0][0x160] ;
                /*0070*/                   STG.E [R2.64], R7 ;
                /*0080*/                   EXIT ;""")
        @test count(contains("VOTEU"), S.notes) <= 1
        ld = findall(==(GPUPerfModel.CID[:LDG]), S.cls)
        L1 = (l1 = 1.0, l2 = 0.0, dram = 0.0)
        c0 = sim(S; threads = 32, blocks = 1, loaded = false).cycles_warp
        c1 = sim(S; threads = 32, blocks = 1, loaded = false, load_hits = Dict(ld[1] => L1))
        @test c0 - c1.cycles_warp ≈ A100.lat.dram - A100.lat.l1 atol = 4
        @test simulate(S; threads = 32, blocks = 1).cycles_warp ==
              simulate(S; threads = 32, blocks = 1, spills = true, regs = 255).cycles_warp
        two =
            "        Function : f\n        /*0000*/  FADD R2, R0, R0 ;\n        /*0010*/  EXIT ;\n" *
            "        Function : g\n        /*0000*/  MUFU.RCP R2, R0 ;\n        /*0010*/  EXIT ;"
        @test !haskey(op_counts(parse_sass(two)), :XU)

        # Spill stores follow `local_hits`: an L1-resident frame moves nothing to DRAM.
        F = parse_sass(
            join(
                (
                    "        /*$(lpad(string(16i; base = 16), 4, '0'))*/   STL [R1+0x$(string(4i; base = 16))], R2 ;"
                    for i in 0:31
                ),
                "\n",
            ) *
            "\n        EXIT ;",
        )
        q = sim(F; threads = 1024, blocks = 216, local_hits = L1)
        @test !any(b -> first(b) == :dram && last(b) > 0, q.bounds)

        # Generic pointers into shared memory are shared accesses.
        G = parse_llvm("""
            @sh = internal addrspace(3) global [256 x float] undef, align 4
            define ptx_kernel void @k(float addrspace(1)* %x) {
            entry:
              %t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
              %z = zext i32 %t to i64
              %g = addrspacecast [256 x float] addrspace(3)* @sh to [256 x float]*
              %p = getelementptr inbounds [256 x float], [256 x float]* %g, i64 0, i64 %z
              %v = load float, float* %p, align 4
              %q = getelementptr inbounds float, float addrspace(1)* %x, i64 %z
              store float %v, float addrspace(1)* %q, align 4
              ret void
            }""")
        @test op_counts(G)[:LDS] == 1 && !haskey(op_counts(G), :LDG)
        @test occupancy(:sm_80, 256, 32) == occupancy(A100, 256, 32)
    end

    @testset "constant tables, extents, register pairs, L1 banks and lines, splits, flags" begin
        # A kernel from its argument list (first line) and body lines (module globals `@...`
        # are hoisted), after the global thread index %t (i64) and %tid.
        function kern(s)
            args, body = split(s, '\n'; limit = 2)
            lines = split(body, '\n')
            glob = filter(l -> startswith(lstrip(l), '@'), lines)
            code = filter(l -> !startswith(lstrip(l), '@'), lines)
            return parse_llvm("""
                $(join(glob, '\n'))
                define ptx_kernel void @k($args) {
                top:
                  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
                  %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
                  %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
                  %b = mul i32 %ntid, %bid
                  %t32 = add i32 %tid, %b
                  %t = zext i32 %t32 to i64
                $(join(code, '\n'))
                }
                declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
                declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
                declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
                declare i32 @llvm.nvvm.shfl.sync.idx.i32(i32, i32, i32, i32)""")
        end
        mem(K) = findall(c -> GPUPerfModel.CLASSES[c] in (:LDG, :STG), K.cls)
        chain(op, n) = join(("%f$j = $op float %f$(j - 1), 1.5" for j in 1:n), '\n')
        DRAM = (l1 = 0.0, l2 = 0.0, dram = 1.0)
        L1 = (l1 = 1.0, l2 = 0.0, dram = 0.0)

        # A module constant table (a differentiation matrix) read by lane: its footprint is
        # the 2 distinct doubles (1 sector), and it hits L1 whatever `hits` says.
        K = kern("""
            double addrspace(1)* %x
            @tab = private unnamed_addr constant [2 x double] [double 1.0, double 2.0]
              %l = and i32 %tid, 1
              %li = zext i32 %l to i64
              %tp = getelementptr inbounds [2 x double], [2 x double]* @tab, i64 0, i64 %li
              %c = load double, double* %tp, align 8
              %p = getelementptr inbounds double, double addrspace(1)* %x, i64 %t
              %v = load double, double addrspace(1)* %p, align 8
              %r = fmul double %v, %c
              store double %r, double addrspace(1)* %p, align 8
              ret void""")
        tab, data, st = mem(K)
        @test K.sectors[[tab, data, st]] == [1, 8, 8] && K.tables == Set([tab])
        kw = (; threads = 256, blocks = 1080, hits = DRAM)
        @test sim(K; kw...).time_us == sim(K; kw..., load_hits = Dict(tab => L1)).time_us

        # `size == 1` tests of broadcasts compare a run-time extent (taken as 2^20): the
        # address select resolves to the lane index, so the footprint is known (coalesced).
        K = kern("""
            float addrspace(1)* %x, i64 %n
              %one = icmp eq i64 %n, 1
              %i = select i1 %one, i64 0, i64 %t
              %p = getelementptr inbounds float, float addrspace(1)* %x, i64 %i
              %v = load float, float addrspace(1)* %p, align 4
              store float %v, float addrspace(1)* %p, align 4
              ret void""")
        @test K.sectors[mem(K)] == [4, 4]
        @test !any(n -> occursin("not a function", n), K.notes)

        # A shuffled double is split into and reassembled from 32-bit halves: register pairs
        # in SASS, so only the two shuffles cost issue slots.
        K = kern("""
            double addrspace(1)* %x
              %p = getelementptr inbounds double, double addrspace(1)* %x, i64 %t
              %d = load double, double addrspace(1)* %p, align 8
              %bi = bitcast double %d to i64
              %hi = lshr i64 %bi, 32
              %h32 = trunc i64 %hi to i32
              %l32 = trunc i64 %bi to i32
              %sh = call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %h32, i32 0, i32 31)
              %sl = call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %l32, i32 0, i32 31)
              %zh = zext i32 %sh to i64
              %hs = shl nuw i64 %zh, 32
              %zl = zext i32 %sl to i64
              %o = or i64 %hs, %zl
              %r = bitcast i64 %o to double
              store double %r, double addrspace(1)* %p, align 8
              ret void""")
        @test op_counts(K)[:SHFL] == 2 && !haskey(op_counts(K), :ALU)

        # L1 bank passes: 16 node columns of 64 doubles (512 B apart) put each level's
        # words in the same banks, 16 passes per half-warp; 63 levels (504 B) spread them, 2
        # passes.
        cols(nv) = kern("""
            double addrspace(1)* %x
              %c = and i64 %t, 15
              %lv = lshr i64 %t, 4
              %cc = mul i64 %c, $nv
              %i = add i64 %cc, %lv
              %p = getelementptr inbounds double, double addrspace(1)* %x, i64 %i
              %v = load double, double addrspace(1)* %p, align 8
              %q = getelementptr inbounds double, double addrspace(1)* %x, i64 %t
              store double %v, double addrspace(1)* %q, align 8
              ret void""")
        K64, K63 = cols(64), cols(63)
        @test K64.waves[mem(K64)] == [32, 2] && K63.waves[mem(K63)] == [2, 2]
        @test K64.lines[mem(K64)] == [16, 2]
        lsu(K; kw...) = bound(sim(K; threads = 256, blocks = 108, kw...), :lsu)
        @test lsu(K64; hits = L1) > 2 * lsu(K63; hits = L1)

        # A warp access that misses L1 (or a store) sends each of its lines to L2: one LSU
        # cycle per line, beyond the bandwidth time of L1 hits. One thread per 63-float
        # column: 32 lines (in distinct banks) per access.
        K = kern("""
            float addrspace(1)* %x
              %i = mul i64 %t, 63
              %p = getelementptr inbounds float, float addrspace(1)* %x, i64 %i
              %v = load float, float addrspace(1)* %p, align 4
              %w = fadd float %v, 1.0
              store float %w, float addrspace(1)* %p, align 4
              ret void""")
        ld, st = mem(K)
        @test K.lines[ld] == 32 && K.sectors[ld] == 32 && K.waves[ld] == 1
        bw = 32 * 32 / A100.bw.l1  # one access's L1 bandwidth time
        @test lsu(K; hits = L1) / 256 * 32 ≈ bw + 32 rtol = 0.01  # hit + store
        @test lsu(K; hits = L2) / 256 * 32 ≈ 32 + 32 rtol = 0.01  # miss + store

        # A thread-range split at a run-time boundary (vertex threads, then face threads):
        # whole blocks take each side in proportion, so a quarter of the blocks run the long
        # side.
        long = chain("fadd", 64)
        K = kern("""
            float addrspace(1)* %x, i64 %n
              %v = icmp ult i64 %t, %n
              %p = getelementptr inbounds float, float addrspace(1)* %x, i64 %t
              %f0 = load float, float addrspace(1)* %p, align 4
              br i1 %v, label %long, label %short
            long:
            $long
              br label %done
            short:
              %g = fadd float %f0, 2.0
              br label %done
            done:
              %r = phi float [ %f64, %long ], [ %g, %short ]
              store float %r, float addrspace(1)* %p, align 4
              ret void""")
        @test !isempty(K.splits)
        q = sim(K; threads = 256, blocks = 108 * 8, conds = Dict("v" => 0.25), hits = L2)
        @test q.counts[:FP32] == round(Int, (2 * 64 + 6 * 1) / 8)  # 2 of 8 block traces

        # SIMT divergence: a lane test that splits every warp (idle lanes 9-15 of a 9-point
        # slab) runs both sides, whatever `conds` says.
        work = chain("fmul", 16)
        K = kern("""
            float addrspace(1)* %x
              %lane = and i32 %tid, 15
              %idle = icmp ugt i32 %lane, 8
              %p = getelementptr inbounds float, float addrspace(1)* %x, i64 %t
              %f0 = load float, float addrspace(1)* %p, align 4
              br i1 %idle, label %done, label %work
            work:
            $work
              store float %f16, float addrspace(1)* %p, align 4
              br label %done
            done:
              ret void""")
        @test !isempty(K.merges)
        @test op_counts(K; dynamic = true, conds = Dict("idle" => 1.0))[:FP32] == 16

        # `gather = :scattered`: an access through a loaded index (a connectivity map) is
        # charged a sector per lane instead of the footprint of an identity map (1-based).
        K = kern("""
            float addrspace(1)* %x, i64 addrspace(1)* %map
              %mp = getelementptr inbounds i64, i64 addrspace(1)* %map, i64 %t
              %e = load i64, i64 addrspace(1)* %mp, align 8
              %e0 = add i64 %e, -1
              %p = getelementptr inbounds float, float addrspace(1)* %x, i64 %e0
              %v = load float, float addrspace(1)* %p, align 4
              store float %v, float addrspace(1)* %p, align 4
              ret void""")
        @test K.sectors[mem(K)] == [8, -4, -4]
        l2(; kw...) = bound(sim(K; threads = 256, blocks = 1080, hits = L2, kw...), :l2)
        @test l2(gather = :scattered) / l2() ≈ (8 + 32 + 32) / (8 + 4 + 4) rtol = 0.01

        # `cache = :warm`: inputs left in L2 by the preceding kernels serve DRAM loads from
        # L2, and the stores stay there.
        kw = (; threads = 256, blocks = 1080)
        @test sim(K; kw..., hits = DRAM, cache = :warm).time_us ==
              sim(K; kw..., hits = L2, store_dram = 0.0).time_us
        @test_throws ErrorException sim(K; kw..., cache = :hot)

        # A local array (an `alloca` the optimizer kept) is the IR kernel's local frame.
        K = parse_llvm("""
            define ptx_kernel void @k(i64 %i) {
            top:
              %a = alloca [36 x double], align 8
              %p = getelementptr inbounds [36 x double], [36 x double]* %a, i64 0, i64 %i
              store double 1.0, double* %p, align 8
              ret void
            }""")
        @test K.frame == 288
        # A struct takes its members' bytes with natural alignment: 4 doubles, an i64 and a
        # float are 44 bytes, padded to 48.
        K = parse_llvm("""
            define ptx_kernel void @k(i64 %i) {
            top:
              %a = alloca { [4 x double], i64, float }, align 8
              %p = bitcast { [4 x double], i64, float }* %a to double*
              store double 1.0, double* %p, align 8
              ret void
            }""")
        @test K.frame == 48
    end
    @testset "device-function calls, warp-dependent branches, the loops flag" begin
        # A non-inlined device function is stepped into at each call site, its parameters
        # bound to the call's arguments: a constant bound gives its loop a trip count (16),
        # a run-time one takes `default_trip` (or `loops`), and the kernel's own loops (the
        # runner's) exclude the callee's.
        f = """
            define internal fastcc float @f(float %0, i64 %1) {
            top:
              %2 = icmp slt i64 %1, 1
              br i1 %2, label %done, label %loop
            loop:
              %i = phi i64 [ 0, %top ], [ %i.next, %loop ]
              %x = phi float [ %0, %top ], [ %y, %loop ]
              %y = fdiv float 1.000000e+00, %x
              %i.next = add i64 %i, 1
              %c = icmp eq i64 %i.next, %1
              br i1 %c, label %done, label %loop
            done:
              %r = phi float [ %0, %top ], [ %y, %loop ]
              ret float %r
            }
            """
        kernel(calls) = parse_llvm(f * """
            define ptx_kernel void @k(float addrspace(1)* %a, i64 %n) {
            entry:
              %t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
              %t64 = zext i32 %t to i64
              %p = getelementptr inbounds float, float addrspace(1)* %a, i64 %t64
              %v = load float, float addrspace(1)* %p, align 4
            $calls
              store float %w2, float addrspace(1)* %p, align 4
              ret void
            }
            declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()""")
        K = kernel("""
              %w = call fastcc float @f(float %v, i64 %n)
              %w2 = call fastcc float @f(float %w, i64 16)""")
        @test length(K.calls) == 2 && isempty(GPUPerfModel.back_edges(K))
        @test sort(collect(values(K.trips))) == [16]
        @test any(contains("@f ×2"), K.notes)
        xu(; kw...) = op_counts(K; dynamic = true, kw...)[:XU]
        @test xu() == 1 + 16 && xu(default_trip = 5) == 5 + 16
        kw = (; threads = 256, blocks = 1080)
        @test sim(K; kw..., loops = :short).counts[:XU] == 16 + 16
        @test sim(K; kw..., loops = :short).time_us ==
              sim(K; kw..., default_trip = 16).time_us
        @test sim(K; kw..., loops = :few).counts[:XU] == 4 + 16
        @test sim(K; kw..., loops = :long).counts[:XU] == 64 + 16
        @test_throws ErrorException sim(K; kw..., loops = :medium)

        # the callee's loads see the caller's addresses: a lane-indexed load is 4 sectors
        g = replace(f,
            "%y = fdiv float 1.000000e+00, %x" =>
                "%q = getelementptr inbounds float, float addrspace(1)* %3, i64 %4\n" *
                "  %y = load float, float addrspace(1)* %q, align 4",
            "(float %0, i64 %1)" => "(float %0, i64 %1, float addrspace(1)* %3, i64 %4)",
        )
        K = parse_llvm(
            g * """
 define ptx_kernel void @k(float addrspace(1)* %a, i64 %n) {
 entry:
   %t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
   %t64 = zext i32 %t to i64
   %w2 = call fastcc float @f(float 1.0, i64 %n, float addrspace(1)* %a, i64 %t64)
   ret void
 }
 declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()""",
        )
        @test K.sectors[findfirst(==(GPUPerfModel.CID[:LDG]), K.cls)] == 4

        # a recursive function stays one branch op (noted)
        r = replace(
            f,
            "%y = fdiv float 1.000000e+00, %x" => "%y = call fastcc float @f(float %x, i64 %i)",
        )
        K = parse_llvm(r * """
            define ptx_kernel void @k(float %v, i64 %n) {
            entry:
              %w = call fastcc float @f(float %v, i64 %n)
              ret void
            }""")
        @test isempty(K.calls) && any(contains("(recursive)"), K.notes)

        # An unrolled loop's induction variable steps by the unroll factor and its remainder
        # loop runs below it: `default_trip` = 16 source iterations run the unrolled header
        # 16 / 4 times and the remainder loop at most 3 times.
        u = replace(f, "%i.next = add i64 %i, 1" => "%i.next = add i64 %i, 4",
            "br i1 %2, label %done, label %loop" => "br label %loop",
            "%c = icmp eq i64 %i.next, %1" => "%m = and i64 %1, -4\n  %c = icmp eq i64 %i.next, %m",
        )
        u = replace(u, "ret float %r" => """
              %rem = and i64 %1, 3
              br label %epil
            epil:
              %j = phi i64 [ 0, %done ], [ %j.next, %epil ]
              %z = phi float [ %r, %done ], [ %z2, %epil ]
              %z2 = fdiv float 1.000000e+00, %z
              %j.next = add i64 %j, 1
              %e = icmp eq i64 %j.next, %rem
              br i1 %e, label %out, label %epil
            out:
              ret float %z2""")
        K = parse_llvm(u * """
            define ptx_kernel void @k(float %v, i64 %n) {
            entry:
              %w = call fastcc float @f(float %v, i64 %n)
              ret void
            }""")
        @test sort(collect(values(K.steps))) == [(1, 3), (4, typemax(Int))]
        @test op_counts(K; dynamic = true, default_trip = 16)[:XU] == 4 + 3
        @test op_counts(K; dynamic = true)[:XU] == 1 + 1

        # the copies of one callee share its code: a second call site adds its own ops only
        code(K) = sim(K; kw...).icache.code_bytes
        K1 = kernel("""
              %w2 = call fastcc float @f(float %v, i64 %n)""")
        K2 = kernel("""
              %w = call fastcc float @f(float %v, i64 %n)
              %w2 = call fastcc float @f(float %w, i64 %n)""")
        @test !isempty(K2.alias) && code(K2) - code(K1) <= 4 * 16

        # A simulation too long to run in full (beyond SIM_OPS ops) is run at 2 and 4 trips of
        # the kernel's outermost loop and extrapolated: close to the full run.
        K = parse_llvm("""
            define ptx_kernel void @k(float addrspace(1)* %a, i64 %n) {
            entry:
              %t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
              %t64 = zext i32 %t to i64
              br label %L
            L:
              %i = phi i64 [ 0, %entry ], [ %i.next, %L ]
              %j = add i64 %t64, %i
              %p = getelementptr inbounds float, float addrspace(1)* %a, i64 %j
              %v = load float, float addrspace(1)* %p, align 4
              %w = fmul float %v, 2.0
              store float %w, float addrspace(1)* %p, align 4
              %i.next = add i64 %i, 256
              %c = icmp slt i64 %i.next, %n
              br i1 %c, label %L, label %done
            done:
              ret void
            }
            declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()""")
        run(n) = sim(K; threads = 256, blocks = 216, trips = Dict("L" => n))
        p = GPUPerfModel.extrapolate(run(2), run(4), 19, "")
        @test p.time_us ≈ run(40).time_us rtol = 0.05
        @test p.counts[:LDG] == 40
        # Two trips apart: a branch guessed at 50/50 alternates per trip (here between 64
        # dependent FMULs and none), which runs of 2 and 3 trips would sample once.
        heavy = join(("  %h$k = fmul float %h$(k - 1), 1.5" for k in 1:64), "\n")
        A = parse_llvm("""
            define ptx_kernel void @k(float addrspace(1)* %a, i64 %n) {
            entry:
              %t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
              %t64 = zext i32 %t to i64
              br label %L
            L:
              %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
              %j = add i64 %t64, %i
              %p = getelementptr inbounds float, float addrspace(1)* %a, i64 %j
              %h0 = load float, float addrspace(1)* %p, align 4
              %c = fcmp ogt float %h0, 0.0
              br i1 %c, label %work, label %latch
            work:
            $heavy
              store float %h64, float addrspace(1)* %p, align 4
              br label %latch
            latch:
              %i.next = add i64 %i, 256
              %e = icmp slt i64 %i.next, %n
              br i1 %e, label %L, label %done
            done:
              ret void
            }
            declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()""")
        akw = (; threads = 256, blocks = 216, trips = Dict("L" => 24))
        full = sim(A; akw...)
        ops = GPUPerfModel.SIM_OPS[]
        GPUPerfModel.SIM_OPS[] = 1000
        ext = sim(A; akw...)
        GPUPerfModel.SIM_OPS[] = ops
        @test any(contains("simulated 2 and 4 trips of loop L"), ext.notes)
        @test ext.time_us ≈ full.time_us rtol = 0.05

        # A loop unrolled by 2 that tests its induction variable in the header (the latch
        # branches back always) runs its header 16 times; `or` adds the odd offset.
        K = parse_llvm("""
            define ptx_kernel void @k(float addrspace(1)* %a) {
            entry:
              br label %L
            L:
              %i = phi i64 [ 2, %entry ], [ %i.next, %L.1 ]
              %e = icmp eq i64 %i, 32
              br i1 %e, label %done, label %L.1
            L.1:
              %j = or i64 %i, 1
              %p = getelementptr inbounds float, float addrspace(1)* %a, i64 %j
              store float 1.0, float addrspace(1)* %p, align 4
              %i.next = add i64 %i, 2
              br label %L
            done:
              ret void
            }""")
        @test only(values(K.trips)) == 16 && op_counts(K; dynamic = true)[:STG] == 15

        # ... and an exit test on `or` of the variable (an add of its known-zero bits)
        K = parse_llvm("""
            define ptx_kernel void @k(float addrspace(1)* %a) {
            entry:
              br label %L
            L:
              %i = phi i64 [ 2, %entry ], [ %i.next, %L ]
              %j = or i64 %i, 1
              %i.next = add i64 %i, 2
              %e = icmp eq i64 %j, 63
              br i1 %e, label %done, label %L
            done:
              ret void
            }""")
        @test only(values(K.trips)) == 31

        # ... also when a call splits the header before its test
        K = parse_llvm(f * """
            define ptx_kernel void @k(float addrspace(1)* %a, float %v) {
            entry:
              br label %L
            L:
              %i = phi i64 [ 2, %entry ], [ %i.next, %L.1 ]
              %w = call fastcc float @f(float %v, i64 1)
              %e = icmp eq i64 %i, 32
              br i1 %e, label %done, label %L.1
            L.1:
              %i.next = add i64 %i, 2
              br label %L
            done:
              ret void
            }""")
        @test sort(collect(values(K.trips))) == [1, 16]  # the loop and the callee's

        # A SASS subroutine call steps into the subroutine and returns after the call.
        K = parse_sass("""
                /*0000*/                   S2R R0, SR_TID.X ;
                /*0010*/                   CALL.REL.NOINC `(sub) ;
                /*0020*/                   FADD R1, R2, R2 ;
                /*0030*/                   EXIT ;
            sub:
                /*0040*/                   MUFU.RCP R2, R0 ;
                /*0050*/                   RET.REL.NODEC R4 `(_Z4test) ;""")
        @test length(K.calls) == 1
        @test op_counts(K; dynamic = true)[:XU] == 1 &&
              op_counts(K; dynamic = true)[:FP32] == 1

        # A lane test that sends the warps of a block different ways: lane 0 of each 64-thread
        # column runs a 62-step serial loop (warp 1 of 2, its other lanes idle), warp 2 goes
        # straight to the barrier. Each warp runs its own trace, so the block takes the loop's
        # serial latency every time, and the mean op count per thread is half the loop's.
        K = parse_llvm("""
            define ptx_kernel void @k(float addrspace(1)* %a) {
            top:
              %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
              %l = and i32 %tid, 63
              %c = icmp eq i32 %l, 0
              br i1 %c, label %loop, label %join
            loop:
              %i = phi i32 [ 0, %top ], [ %i.next, %loop ]
              %x = phi float [ 0.0, %top ], [ %y, %loop ]
              %y = fadd float %x, 1.000000e+00
              %i.next = add i32 %i, 1
              %e = icmp eq i32 %i.next, 62
              br i1 %e, label %done, label %loop
            done:
              store float %y, float addrspace(1)* %a, align 4
              br label %join
            join:
              call void @llvm.nvvm.barrier0()
              ret void
            }
            declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()""")
        b = only(keys(K.wsplit))
        @test K.wsplit[b][1][1:2] == [3, 2] && K.labels[K.wsplit[b][2]] == "join"
        loop = findfirst(==("loop"), K.labels)
        @test count(==(loop), walk(K; warp = 1)) == 62 &&
              count(==(loop), walk(K; warp = 2)) == 0
        p = sim(K; threads = 64, blocks = 108, conds = Dict("c" => 1 / 64))
        @test p.counts[:FP32] == 31
        @test p.cycles_block >= 62 * fp32_lat(A100)

        # On a divergent side only the active lanes move data: a coalesced load that lane 0
        # of each warp runs moves one sector (32 B) instead of the warp's 128 B.
        lane0(test) = parse_llvm("""
            define ptx_kernel void @k(float addrspace(1)* %a) {
            top:
              %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
              %t = zext i32 %tid to i64
              %l = and i32 %tid, 31
              %c = icmp $test i32 %l, 0
              br i1 %c, label %work, label %join
            work:
              %p = getelementptr inbounds float, float addrspace(1)* %a, i64 %t
              %v = load float, float addrspace(1)* %p, align 4
              store float %v, float addrspace(1)* %p, align 4
              br label %join
            join:
              ret void
            }
            declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()""")
        dram(K) = bound(sim(K; threads = 256, blocks = 4320), :dram)
        @test !isempty(lane0("eq").merges)
        @test dram(lane0("eq")) / dram(lane0("sge")) ≈ 0.25 rtol = 0.01
    end

    @testset "zero-trip guards enter their loops, also inside loops" begin
        # A loop with a run-time bound sits behind a test of that bound (rotated loops):
        # unset, the walk enters it, also when the guard is inside an outer loop, so the
        # inner loop runs on every outer iteration (noted); `conds` overrides it.
        K = parse_llvm("""
            define ptx_kernel void @k(float addrspace(1)* %a, i64 %n, i64 %m) {
            entry:
              br label %outer
            outer:
              %j = phi i64 [ 0, %entry ], [ %j.next, %olatch ]
              %g = icmp slt i64 %m, 1
              br i1 %g, label %olatch, label %inner
            inner:
              %i = phi i64 [ 0, %outer ], [ %i.next, %inner ]
              %p = getelementptr inbounds float, float addrspace(1)* %a, i64 %i
              %v = load float, float addrspace(1)* %p, align 4
              %i.next = add i64 %i, 1
              %c = icmp eq i64 %i.next, %m
              br i1 %c, label %olatch, label %inner
            olatch:
              %j.next = add i64 %j, 1
              %e = icmp eq i64 %j.next, %n
              br i1 %e, label %done, label %outer
            done:
              ret void
            }""")
        t = Dict("outer" => 10, "inner" => 5)
        @test op_counts(K; dynamic = true, trips = t)[:LDG] == 50
        @test !haskey(
            op_counts(K; dynamic = true, trips = t, conds = Dict("g" => 1.0)),
            :LDG,
        )
        p = sim(K; threads = 256, blocks = 108, trips = t)
        @test any(n -> contains(n, "guards assumed to enter") && contains(n, "g"), p.notes)
    end

    @testset "early exits: a constant bound that data can cut short, and `loops`" begin
        # A loop whose count follows from constants but that can also leave on computed data
        # (a Newton iteration's convergence test, bound `maxiters` = 1000) runs its bound
        # unless `loops` is set (noted); `trips` names it over `loops`. A loop without such an
        # exit keeps its inferred count whatever `loops` says.
        newton(test) = parse_llvm("""
            define ptx_kernel void @k(float addrspace(1)* %a) {
            entry:
              %t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
              %t64 = zext i32 %t to i64
              %p = getelementptr inbounds float, float addrspace(1)* %a, i64 %t64
              %v = load float, float addrspace(1)* %p, align 4
              br label %L
            L:
              %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
              %x = phi float [ %v, %entry ], [ %y, %latch ]
              %y = call float @llvm.nvvm.ex2.approx.ftz.f(float %x)
              %d = fsub float %y, %x
              %ok = fcmp olt float %d, 1.000000e-03
              $test
            latch:
              %i.next = add i64 %i, 1
              %c = icmp eq i64 %i.next, 1000
              br i1 %c, label %done, label %L
            done:
              store float %y, float addrspace(1)* %p, align 4
              ret void
            }
            declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
            declare float @llvm.nvvm.ex2.approx.ftz.f(float)""")
        K = newton("br i1 %ok, label %done, label %latch")
        C = newton("br label %latch")
        L = findfirst(==("L"), K.labels)
        @test K.trips[L] == 1000 && L in K.early
        @test C.trips[L] == 1000 && isempty(C.early)
        kw = (; threads = 256, blocks = 1080)
        p = sim(K; kw...)
        @test p.counts[:XU] == 1000 &&
              any(contains("loop L: leaves early on data, using its bound 1000"), p.notes)
        @test sim(K; kw..., loops = :few).counts[:XU] == 4
        @test sim(K; kw..., loops = :long).counts[:XU] == 64
        @test sim(K; kw..., loops = :few, trips = Dict("L" => 7)).counts[:XU] == 7
        @test sim(C; kw..., loops = :few).counts[:XU] == 1000 &&
              any(contains("`loops` changes nothing"), sim(C; kw..., loops = :few).notes)
        @test sim(K; kw..., loops = :few).cycles_warp < p.cycles_warp / 10
    end

    @testset "L1 bank passes of 8- and 16-byte accesses run per 128 B" begin
        # lanes 0-15: node column c (c = lane) at level 0, lanes 16-31: level 1 (a spectral slab
        # of 16 nodes, two levels per warp); columns `nv` elements apart
        function slab(T, nv; broadcast = false)
            t = T == "double" ? 8 : 4
            idx =
                broadcast ? "%z = add i64 0, 0\n  %i = add i64 %z, 7" :
                "%c = and i64 %t, 15\n  %lv = lshr i64 %t, 4\n  %cc = mul i64 %c, $nv\n  %i = add i64 %cc, %lv"
            return parse_llvm("""
                define ptx_kernel void @k($T addrspace(1)* %x, $T addrspace(1)* %y) {
                top:
                  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
                  %t = zext i32 %tid to i64
                  $idx
                  %p = getelementptr inbounds $T, $T addrspace(1)* %x, i64 %i
                  %v = load $T, $T addrspace(1)* %p, align $t
                  %q = getelementptr inbounds $T, $T addrspace(1)* %y, i64 %t
                  store $T %v, $T addrspace(1)* %q, align $t
                  ret void
                }
                declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()""")
        end
        ldst(K) = findall(c -> GPUPerfModel.CLASSES[c] in (:LDG, :STG), K.cls)

        # A pass reads one word from each of the 32 4-byte banks (128 B). Float32 columns 256 B
        # apart (64 levels) put the 16 columns' words of a level in one bank: 16 passes. Float64
        # columns 512 B apart do the same with 256 B per warp, which run as two half-warps of 16
        # passes each (Nsight, Float64 operators on 64-level boxes: L1 data-pipe wavefronts 1.5-2.1x
        # the 16-pass count, Float32 twins 0.8-1.1x). Columns 504 B apart (63 levels) spread the
        # words over all banks: 1 pass per half-warp. A coalesced 256-B store: 2 passes; all
        # lanes reading one double: 1.
        K32, K64, K63 = slab("float", 64), slab("double", 64), slab("double", 63)
        @test K32.waves[ldst(K32)] == [16, 1] && K32.lines[ldst(K32)][1] == 16
        @test K64.waves[ldst(K64)] == [32, 2] && K64.lines[ldst(K64)][1] == 16
        @test K63.waves[ldst(K63)] == [2, 2]
        Kb = slab("double", 64; broadcast = true)
        @test Kb.waves[ldst(Kb)][1] == 1

        # L1 hits: the load takes its passes on the LSU (16 more for the Float64 slab), the store
        # its bytes at the L1 bandwidth (128 B more), for each of the 16 warps of an SM
        L1 = (l1 = 1.0, l2 = 0.0, dram = 0.0)
        lsu(K) = bound(sim(K; threads = 512, blocks = 108, hits = L1), :lsu)
        @test lsu(K64) - lsu(K32) ≈ 16 * (16 + 128 / A100.bw.l1) rtol = 0.02
    end

    @testset "loaded indices: broadcasts, index maps, loaded aggregates" begin
        function kern(s)
            args, body = split(s, '\n'; limit = 2)
            return parse_llvm("""
                define ptx_kernel void @k($args) {
                top:
                  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
                  %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
                  %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
                  %b = mul i32 %ntid, %bid
                  %t32 = add i32 %tid, %b
                  %t = zext i32 %t32 to i64
                $body
                }
                declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
                declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
                declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()""")
        end
        mem(K) = findall(c -> GPUPerfModel.CLASSES[c] in (:LDG, :STG), K.cls)
        l2(K; kw...) = bound(sim(K; threads = 256, blocks = 1080, hits = L2, kw...), :l2)

        # An index every lane loads from one address is a broadcast (the levels of one DSS
        # vertex read its `local_vertices` entry): the access through it runs over contiguous
        # levels (4 sectors), or is one sector when no lane term is added; sectors < 0 mark
        # accesses through a loaded index, which `gather = :scattered` charges per lane.
        K = kern("""
            float addrspace(1)* %x, i64 addrspace(1)* %map, i64 %o
              %mp = getelementptr inbounds i64, i64 addrspace(1)* %map, i64 %o
              %e = load i64, i64 addrspace(1)* %mp, align 8
              %eb = mul i64 %e, 64
              %i = add i64 %eb, %t
              %p = getelementptr inbounds float, float addrspace(1)* %x, i64 %i
              %v = load float, float addrspace(1)* %p, align 4
              %q = getelementptr inbounds float, float addrspace(1)* %x, i64 %e
              store float %v, float addrspace(1)* %q, align 4
              ret void""")
        @test K.sectors[mem(K)] == [1, -4, -1]
        @test l2(K) / l2(K; gather = :scattered) ≈ (1 + 4 + 1) / (1 + 32 + 32) rtol = 0.01

        # Lanes that read consecutive entries of an index map get consecutive (1-based) indices
        # (an element list): the first varying index of the load's address, also for
        # a field of a loaded tuple (the elements of an interior face) and a [2 x i64] entry
        # (a vertex's (element, corner) pair).
        K = kern(
            """
       float addrspace(1)* %x, { i64, i64 } addrspace(1)* %faces, [2 x i64] addrspace(1)* %verts
         %fp = getelementptr inbounds { i64, i64 }, { i64, i64 } addrspace(1)* %faces, i64 %t
         %f = load { i64, i64 }, { i64, i64 } addrspace(1)* %fp, align 8
         %h = extractvalue { i64, i64 } %f, 0
         %h0 = add i64 %h, -1
         %p = getelementptr inbounds float, float addrspace(1)* %x, i64 %h0
         %v = load float, float addrspace(1)* %p, align 4
         %vp = getelementptr inbounds [2 x i64], [2 x i64] addrspace(1)* %verts, i64 %t, i64 0
         %g = load i64, i64 addrspace(1)* %vp, align 8
         %g0 = add i64 %g, -1
         %q = getelementptr inbounds float, float addrspace(1)* %x, i64 %g0
         store float %v, float addrspace(1)* %q, align 4
         ret void""",
        )
        @test K.sectors[mem(K)[[2, 4]]] == [-4, -4]
    end

    @testset "loop reuse: walks, re-reads, strength reduction, the 64-bit index idiom" begin
        # A load whose lanes' addresses advance by `step` bytes per iteration of the loop
        # counter (a lane walking a column) re-reads the line of the previous iteration:
        # 4-byte steps miss once per 32 iterations; a load of the same address in every
        # iteration misses on the first of the loop's trips only. Both are per-load hit
        # overrides, so they predict exactly what the matching `load_hits` predict.
        walker(step, extra = ""; lanestride = 1) = parse_llvm("""
            define ptx_kernel void @k(float addrspace(1)* %x, float addrspace(1)* %y) {
            top:
              %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
              %t = zext i32 %tid to i64
              %tt = mul i64 %t, $lanestride
              br label %loop
            loop:
              %acc = phi float [ 0.0, %top ], [ %s, %loop ]
              %i = phi i64 [ 0, %top ], [ %inext, %loop ]
              %j = mul i64 %i, $step
              %a = add i64 %j, %tt
              %p = getelementptr inbounds float, float addrspace(1)* %x, i64 %a
              %v = load float, float addrspace(1)* %p, align 4
              %s = fadd float %acc, %v
            $extra
              %inext = add nuw nsw i64 %i, 1
              %c = icmp eq i64 %inext, 64
              br i1 %c, label %exit, label %loop
            exit:
              %q = getelementptr inbounds float, float addrspace(1)* %y, i64 %t
              store float %s, float addrspace(1)* %q, align 4
              ret void
            }
            declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()""")
        lds(K) = findall(==(GPUPerfModel.CID[:LDG]), K.cls)
        over(K, hs...) = Dict(k => h for (k, h) in zip(lds(K), hs))
        DR = (l1 = 0.0, l2 = 0.0, dram = 1.0)
        cfg = (; threads = 32, blocks = 108)
        same(K, kw, ov) =
            (p = sim(K; cfg..., kw...); q = sim(K; cfg..., kw..., load_hits = ov);
                p.time_us == q.time_us && p.stalls == q.stalls)
        K = walker(1)
        @test K.walks == Dict(lds(K)[1] => (4, 2))
        @test same(K, (; hits = DR), over(K, (l1 = 31 / 32, l2 = 0.0, dram = 1 / 32)))
        @test any(contains("reuse lines"), sim(K; cfg..., hits = DR).notes)

        # its misses go to L2 and DRAM as `hits` splits them
        @test same(K, (; hits = (l1 = 0.0, l2 = 0.5, dram = 0.5)),
            over(K, (l1 = 31 / 32, l2 = 1 / 64, dram = 1 / 64)))

        # 128-byte steps touch a new line every iteration: no reuse
        @test isempty(walker(32).walks)

        # the same address in every iteration: one miss per loop entry (64 trips)
        K0 = walker(0)
        @test K0.walks == Dict(lds(K0)[1] => (0, 2))
        @test same(K0, (; hits = DR), over(K0, (l1 = 63 / 64, l2 = 0.0, dram = 1 / 64)))

        # the lines that all resident warps walk must fit in the L1 beside the shared memory:
        # lanes walking their own columns touch 32 lines per access; 2 blocks of 1024 threads
        # walk 256 KB
        KT = walker(1; lanestride = 64)
        @test same(KT, (; hits = DR), over(KT, (l1 = 31 / 32, l2 = 0.0, dram = 1 / 32)))
        @test sim(KT; threads = 1024, blocks = 216, hits = DR).time_us ==
              sim(KT; threads = 1024, blocks = 216, hits = DR, load_hits = over(KT, DR)).time_us

        # `hits` stays the fraction over all loads where it can: with a second, non-walking
        # load and hits.l1 = 0.75, the walk takes 31/32 and the other loads the other 17/32
        K2 = walker(
            1,
            """
   %r = mul i64 %i, 4096
   %u = getelementptr inbounds float, float addrspace(1)* %y, i64 %r
   %w = load float, float addrspace(1)* %u, align 4""",
        )
        @test collect(keys(K2.walks)) == lds(K2)[1:1]
        @test sim(K2; cfg..., hits = (l1 = 0.75, l2 = 0.0, dram = 0.25)).time_us ==
              sim(K2; cfg..., hits = (l1 = 17 / 32, l2 = 0.0, dram = 15 / 32),
            store_dram = 0.25,
            load_hits = over(K2, (l1 = 31 / 32, l2 = 0.0, dram = 1 / 32))).time_us

        # Loop strength reduction: the walking address is a pointer that the loop increments
        # (2 IADD, no IMAD.WIDE, no index arithmetic in the loop); an address at a constant
        # offset from it shares the pointer (an immediate offset).
        function cnt(K, lab)
            c = Dict{Symbol, Int}()
            for k in K.blocks[findfirst(==(lab), K.labels)]
                x = GPUPerfModel.CLASSES[K.cls[k]]
                c[x] = get(c, x, 0) + 1
            end
            return c
        end
        c = cnt(K, "loop")
        @test get(c, :IMADW, 0) == 0 && get(c, :IMAD, 0) == 0
        @test c[:IADD] == 2 + 2  # pointer + 64-bit counter
        K3 = walker(
            1,
            """
   %a7 = add i64 %a, 7
   %p7 = getelementptr inbounds float, float addrspace(1)* %x, i64 %a7
   %v7 = load float, float addrspace(1)* %p7, align 4""",
        )
        @test cnt(K3, "loop")[:IADD] == 4

        # instcombine's 64-bit form of sext(trunc(i) + 7), `ashr exact ((shl i, 32) +
        # (7 << 32)), 32`, folds into the address like `add nsw i64 %i, 7`
        idiom(idx) = parse_llvm("""
            define ptx_kernel void @k(float addrspace(1)* %x, i64 %i) {
            top:
            $idx
              %p = getelementptr inbounds float, float addrspace(1)* %x, i64 %j
              %v = load float, float addrspace(1)* %p, align 4
              store float %v, float addrspace(1)* %x, align 4
              ret void
            }""")
        a = idiom(
            "  %s = shl i64 %i, 32\n  %e = add nsw i64 %s, 30064771072\n  %j = ashr exact i64 %e, 32",
        )
        b = idiom("  %j = add nsw i64 %i, 7")
        @test op_counts(a) == op_counts(b)
    end

    @testset "inner_extent = 1: run-time divisors, per-lane paths, L1 shares" begin
        function kern(s)
            args, body = split(s, '\n'; limit = 2)
            return parse_llvm("""
                define ptx_kernel void @k($args) {
                top:
                  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
                  %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
                  %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
                  %b = mul i32 %ntid, %bid
                  %t32 = add i32 %tid, %b
                  %t = zext i32 %t32 to i64
                $body
                }
                declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
                declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
                declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()""")
        end
        loads(K) = findall(==(GPUPerfModel.CID[:LDG]), K.cls)
        l2(K; kw...) = bound(sim(K; threads = 256, blocks = 1080, hits = L2, kw...), :l2)

        # The thread index split by a run-time inner extent `%nv` (v = t % nv, node = t ÷
        # nv): a run-time extent is at least a warp wide, so the node is lane-invariant and
        # the lanes read consecutive inner entries (4 sectors, 5 one float further); with
        # `inner_extent = 1` it is 1, the lanes read every other float (8 sectors), and the
        # second load reads only sectors that the first one touched (an L1 share of 1).
        K = kern("""
            float addrspace(1)* %x, float addrspace(1)* %y, i64 %nv
              %node = udiv i64 %t, %nv
              %v = urem i64 %t, %nv
              %i0 = shl i64 %node, 1
              %i = add i64 %i0, %v
              %p = getelementptr inbounds float, float addrspace(1)* %x, i64 %i
              %a = load float, float addrspace(1)* %p, align 4
              %i1 = add i64 %i, 1
              %p1 = getelementptr inbounds float, float addrspace(1)* %x, i64 %i1
              %c = load float, float addrspace(1)* %p1, align 4
              %s = fadd float %a, %c
              %q = getelementptr inbounds float, float addrspace(1)* %y, i64 %t
              store float %s, float addrspace(1)* %q, align 4
              ret void""")
        k1, k2 = loads(K)
        @test K.sectors[[k1, k2]] == [4, 5]
        @test l2(K) ≈ l2(K; inner_extent = nothing)
        one = l2(K; inner_extent = 1)
        @test K.extent1[k1][[1, 5]] == (8, 0) && K.extent1[k2][[1, 5]] == (8, 1)
        # default: 4 + 5 load sectors and a 4-sector store; inner extent 1: 8 (the second
        # load hits L1) and the store
        @test one / l2(K) ≈ (8 + 4) / (4 + 5 + 4) rtol = 0.01
        @test_throws ErrorException sim(K; threads = 256, inner_extent = 2)

        # With `inner_extent = 1` a phi that merges forward paths takes each lane's incoming
        # value: even lanes read floats 0-30, odd lanes floats 1025-1055 (8 sectors).
        K = kern("""
            float addrspace(1)* %x, i64 %nv
              %node = udiv i64 %t, %nv
              %odd = and i64 %t, 1
              %c = icmp eq i64 %odd, 0
              br i1 %c, label %even, label %oddl
            even:
              br label %m
            oddl:
              br label %m
            m:
              %o = phi i64 [ 0, %even ], [ 1024, %oddl ]
              %i = add i64 %node, %o
              %p = getelementptr inbounds float, float addrspace(1)* %x, i64 %i
              %a = load float, float addrspace(1)* %p, align 4
              store float %a, float addrspace(1)* %p, align 4
              ret void""")
        sim(K; threads = 256, inner_extent = 1)
        @test K.extent1[loads(K)[1]][1] == 8
    end

    @testset "a kernel outside ClimaCore: RRTMGP's column amounts" begin
        # RRTMGP's `compute_col_gas_CUDA!`: one thread per layer and column of (nlay, ncol)
        # arrays, the thread index split by the run-time layer count
        function col_gas!(col_dry, p_lev, g)
            glx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            nlay, ncol = size(col_dry)
            if glx ≤ nlay * ncol
                glay = (glx % nlay == 0) ? nlay : (glx % nlay)
                gcol = cld(glx, nlay)
                @inbounds col_dry[glay, gcol] =
                    (p_lev[glay, gcol] - p_lev[glay + 1, gcol]) / g
            end
            return nothing
        end
        M = CUDA.CuDeviceMatrix{Float32, 1}
        K = kernel(col_gas!, Tuple{M, M, Float32})
        mem = findall(c -> c in (GPUPerfModel.CID[:LDG], GPUPerfModel.CID[:STG]), K.cls)
        # many layers: the lanes walk consecutive layers, and the select of `glay` (a
        # comparison of a run-time remainder) leaves the addresses unknown: coalesced
        p = sim(K; threads = 256, blocks = cld(63 * 24576, 256))
        @test K.sectors[mem] == [0, 0, 0] && p.counts[:LDG] == 2 && p.counts[:STG] == 1
        # one layer (`inner_extent = 1`): `glay` = 1 and the lanes are columns; `col_dry`'s
        # column stride is the divisor (contiguous: 4 sectors), `p_lev`'s a run-time extent
        # (32 sectors), and the second `p_lev` load reads the first one's sectors (L1)
        sim(K; threads = 256, blocks = 96, inner_extent = 1)
        @test [K.extent1[k][1] for k in mem] == [32, 32, 4] && K.extent1[mem[2]][5] == 1
    end

    @testset "shuffle member-mask checks: MATCH.ANY passes on the ADU" begin
        # `body` after a full-warp mask %m (the lanes below the run-time block size) and a
        # lane-pair mask %m2 (16 distinct values), as CUDA.jl's segmented scans build them
        k(body) = parse_llvm("""
            define ptx_kernel void @k(float addrspace(1)* %x) {
            top:
              %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
              %lane = call i32 @llvm.nvvm.read.ptx.sreg.laneid()
              %n = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
              %w = and i32 %tid, 992
              %a = sub i32 %n, %w
              %c = call i32 @llvm.umin.i32(i32 %a, i32 32)
              %s = sub i32 32, %c
              %m = lshr i32 -1, %s
              %pm = and i32 %lane, -2
              %p3 = shl i32 3, %pm
              %m2 = and i32 %m, %p3
              %t = zext i32 %tid to i64
              %p = getelementptr inbounds float, float addrspace(1)* %x, i64 %t
              %v = load float, float addrspace(1)* %p, align 4
            $body
              ret void
            }
            declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
            declare i32 @llvm.nvvm.read.ptx.sreg.laneid()
            declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
            declare i32 @llvm.umin.i32(i32, i32)
            declare float @llvm.nvvm.shfl.sync.bfly.f32(i32, float, i32, i32)
            declare float @llvm.nvvm.shfl.sync.idx.f32(i32, float, i32, i32)""")
        shfl = "call float @llvm.nvvm.shfl.sync.bfly.f32"
        bfly(d, m, x, j) = "%$d = $shfl(i32 $m, float $x, i32 $j, i32 31)"
        # Shuffles with one run-time mask in one block share a check (MATCH.ANY + REDUX.OR on
        # the ADU, VOTEU, LOP3, BRA.DIV); a dead shuffle keeps its check only; a constant
        # mask needs none. MATCH.ANY takes a pass per distinct mask value.
        K = k(
            join(
                [bfly("s1", "%m", "%v", 1), "%v1 = fadd float %v, %s1",
                    bfly("s2", "%m", "%v1", 2), "%v2 = fadd float %v1, %s2",
                    bfly("s3", "%m2", "%v2", 1),
                    "%d = call float @llvm.nvvm.shfl.sync.idx.f32(i32 %m2, float %v2, i32 0, i32 31)",
                    bfly("s4", "-1", "%s3", 4),
                    "store float %s4, float addrspace(1)* %p, align 4"],
                '\n'),
        )
        c = op_counts(K)
        adu = findall(==(GPUPerfModel.CID[:ADU]), K.cls)
        @test c[:SHFL] == 4 && c[:ADU] == 6 && c[:UNI] == 3 &&
              K.waves[adu] == [1, 1, 16, 1, 16, 1]
        # the ADU serves 0.5 passes per clock and SM: 64 warps × 36 passes × 2 cycles
        kw = (; threads = 64, blocks = 32 * 108)
        @test bound(sim(K; kw...), :bar) == 64 * 36 * 2
        # sm_6x has no independent thread scheduling: no checks
        p = sim(K; kw..., device = :sm_60)
        @test p.counts[:SHFL] == 4 && !haskey(p.counts, :ADU) && !haskey(p.counts, :UNI)
        # 16 checks of lane-pair masks bind the ADU: 64 warps × 16 × 17 passes × 2 cycles
        K = k(
            join(
                [
                    "%q$j = or i32 %m2, 0\n" *
                    bfly("r$j", "%q$j", j == 1 ? "%v" : "%r$(j - 1)", 1)
                    for j in 1:16
                ], '\n') * "\nstore float %r16, float addrspace(1)* %p, align 4",
        )
        p = sim(K; kw...)
        @test bound(p, :bar) == 64 * 16 * 17 * 2 && 1 <= p.cycles_sm / bound(p, :bar) < 1.1
    end

    @testset "trip counts: a rotated loop with a range-check exit" begin
        # `for j in 2:5` over a table, as LLVM rotates it: the exit test (j - 6) <u -5
        # precedes the increment, which reaches the latch through a phi with `undef`
        K = parse_llvm("""
            define ptx_kernel void @k() {
            top:
              br label %h
            h:
              %i = phi i64 [ 2, %top ], [ %in, %latch ]
              %d = add i64 %i, -6
              %ex = icmp ult i64 %d, -5
              br i1 %ex, label %latch, label %step
            step:
              %i1 = add nuw nsw i64 %i, 1
              br label %latch
            latch:
              %in = phi i64 [ undef, %h ], [ %i1, %step ]
              br i1 %ex, label %done, label %h
            done:
              ret void
            }""")
        @test K.trips[findfirst(==("h"), K.labels)] == 5
    end
end
