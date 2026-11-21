#!/usr/bin/env python3
"""Temporary (comparison branch, not for merging): attributes the device time of the
CUDA.jl profiles written by profile_amip_steps.jl (host.tsv, device.tsv, nvtx.tsv) to the
NVTX ranges that enclose each launch, and prints per-step tables that compare the trees.
With an ncu CSV of one coupling step (the step after the two compiling steps, which is the
first traced step), it also sums the counters of each kernel into its components, matching
kernels by their order in that step.
usage: attribute_trace.py <label>=<dir>[,<ncu csv>] [<label>=<dir>[,<ncu csv>] ...]"""
import collections
import csv
import re
import sys

from annotate_components import COMPONENTS

STEP = "coupling step"
ATMOS_STEP = "atmos step"
SURFACE_STEP = "surface model steps"


def read_tsv(path):
    with open(path, newline="") as io:
        return list(csv.DictReader(io, delimiter="\t", quoting=csv.QUOTE_NONE))


def short(name):
    """The function name of an NVTX range message (`f!(args) file.jl:1` or `f!`)."""
    m = re.match(r"([^\s(]+)\(", name.strip())
    return m.group(1) if m else name.strip()


def kernel_source(name):
    m = re.search(r"_FILE_(Clima[A-Za-z]+)", name)
    if m:
        return m.group(1)
    for key in ("rrtmgp", "RRTMGP", "rte_lw", "rte_sw"):
        if key in name:
            return "RRTMGP"
    return "other"


def load(directory, atmos_names):
    host = {int(r["id"]): (float(r["start"]), r["tid"]) for r in read_tsv(f"{directory}/host.tsv")}
    device = read_tsv(f"{directory}/device.tsv")
    starts, ends = {}, {}
    for seq, r in enumerate(read_tsv(f"{directory}/nvtx.tsv")):
        if r["type"] == "start":
            starts[int(r["id"])] = (float(r["start"]), r["tid"], short(r["name"]), seq)
        elif r["type"] == "end":
            ends[int(r["id"])] = (float(r["start"]), seq)
    # events per thread: (time, order, kind, payload); range events at equal times keep the
    # order in which they were recorded (a range can start and end within one clock tick)
    events = collections.defaultdict(list)
    for rid, (t0, tid, name, seq) in starts.items():
        if rid in ends:
            events[0].append((t0, seq, "start", (rid, name)))
            events[0].append((ends[rid][0], ends[rid][1], "end", rid))
    kernels = []
    for i, r in enumerate(device):
        rid = int(r["id"])
        if rid not in host:
            continue
        t, tid = host[rid]
        events[0].append((t, float("inf"), "launch", i))
    stack_of = {}
    ranges = {}  # range id -> [name, parent ids, is atmos component step]
    step_ranges = []
    for tid, evs in events.items():
        evs.sort(key=lambda e: (e[0], e[1]))
        stack = []
        for t, _, kind, payload in evs:
            if kind == "start":
                rid, name = payload
                ranges[rid] = [name, False]
                if name == STEP:
                    step_ranges.append(rid)
                if name in atmos_names:
                    for sid in stack:
                        if ranges[sid][0] == "step!":
                            ranges[sid][1] = True
                stack.append(rid)
            elif kind == "end":
                if payload in stack:
                    stack.remove(payload)
            else:
                stack_of[payload] = tuple(stack)
    step_wall = [ends[rid][0] - starts[rid][0] for rid in step_ranges]
    for i, r in enumerate(device):
        if i not in stack_of:
            continue
        # Within the atmos step, only the marked components are kept (ranges of other
        # packages, e.g., ClimaCore's solvers, are attributed to the enclosing component)
        path, ids = [], []
        in_atmos = False
        for rid in stack_of[i]:
            name, is_atmos = ranges[rid]
            if name == "step!":
                name = ATMOS_STEP if is_atmos else SURFACE_STEP
                in_atmos = is_atmos
            elif in_atmos and name not in atmos_names:
                continue
            if not path or path[-1] != name:
                path.append(name)
                ids.append(rid)
        if STEP not in path:
            continue
        first = path.index(STEP) + 1
        path, ids = path[first:], ids[first:]
        if not path or path[-1] in (ATMOS_STEP, SURFACE_STEP):
            path.append(f"other ({kernel_source(r['name'])} kernels)")
            ids.append(None)
        kernels.append({
            "start": float(r["start"]),
            "path": tuple(path),
            "ids": tuple(ids),
            "time": float(r["stop"]) - float(r["start"]),
            "name": r["name"],
            "threads": (int(r["grid"]) * int(r["block"])) if r["grid"] and r["block"] else 0,
            "is_kernel": bool(r["grid"]),
        })
    kernels.sort(key=lambda k: k["start"])
    return kernels, step_wall


def read_ncu(path):
    lines = open(path, errors="replace").read().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
    rows = list(csv.DictReader(lines[start:]))
    rows = [r for r in rows if r.get("ID", "").isdigit()]
    return rows


def number(s):
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return 0.0


# (metric, column, scale), with the warp-level instruction counts times 32 as the
# thread-level counts when ncu does not provide the latter
COUNTERS = (
    ("smsp__sass_thread_inst_executed_op_global_ld_pred_on.sum", "global loads (M)", 1e-6),
    ("smsp__sass_thread_inst_executed_op_global_st_pred_on.sum", "global stores (M)", 1e-6),
    ("dram__bytes_read.sum", "DRAM read (MB)", 1e-6),
    ("dram__bytes_write.sum", "DRAM write (MB)", 1e-6),
)
FALLBACKS = {
    "smsp__sass_thread_inst_executed_op_global_ld_pred_on.sum": "smsp__inst_executed_op_global_ld.sum",
    "smsp__sass_thread_inst_executed_op_global_st_pred_on.sum": "smsp__inst_executed_op_global_st.sum",
}


def counter(row, metric):
    if metric in row and row[metric] not in ("", None):
        return number(row[metric])
    return 32 * number(row.get(FALLBACKS.get(metric, ""), ""))


def main():
    trees = []
    for arg in sys.argv[1:]:
        label, spec = arg.split("=", 1)
        directory, _, ncu = spec.partition(",")
        trees.append((label, directory, ncu))
    has_ncu = any(ncu for _, _, ncu in trees)
    data = {}
    for label, directory, ncu in trees:
        kernels, step_wall = load(directory, set(COMPONENTS))
        n_steps = len(step_wall)
        node_time = collections.defaultdict(float)
        node_count = collections.defaultdict(float)
        node_calls = collections.defaultdict(set)
        for k in kernels:
            for depth in range(1, len(k["path"]) + 1):
                node_time[k["path"][:depth]] += k["time"] / n_steps
                node_count[k["path"][:depth]] += k["is_kernel"] / n_steps
                node_calls[k["path"][:depth]].add(k["ids"][depth - 1])
        flat = collections.defaultdict(lambda: {"incl": 0.0, "excl": 0.0, "kernels": 0.0, "calls": set(),
                                                "counters": [0.0] * len(COUNTERS)})
        for k in kernels:
            marked = [(name, rid) for name, rid in zip(k["path"], k["ids"]) if name in COMPONENTS]
            for name in {name for name, _ in marked}:
                flat[name]["incl"] += k["time"] / n_steps
                flat[name]["kernels"] += k["is_kernel"] / n_steps
            for name, rid in marked:
                flat[name]["calls"].add(rid)
            if marked:
                flat[marked[-1][0]]["excl"] += k["time"] / n_steps
        counters = collections.defaultdict(lambda: [0.0] * len(COUNTERS))
        if ncu:
            rows = read_ncu(ncu)
            step_kernels = [k for k in kernels if k["is_kernel"]][: len(rows)]
            per_step = sum(k["is_kernel"] for k in kernels) / n_steps
            mismatches = sum(
                int(number(r.get("launch__grid_size", "0")) * number(r.get("launch__block_size", "0")))
                != k["threads"]
                for r, k in zip(rows, step_kernels)
            )
            print(f"{label}: {len(rows)} kernels under ncu, {per_step:.1f} traced kernels per step, "
                  f"{mismatches} of the matched kernels with different thread counts")
            for r, k in zip(rows, step_kernels):
                values = [counter(r, metric) * scale for metric, _, scale in COUNTERS]
                for depth in range(1, len(k["path"]) + 1):
                    c = counters[k["path"][:depth]]
                    for j, v in enumerate(values):
                        c[j] += v
                for name in {name for name in k["path"] if name in COMPONENTS}:
                    for j, v in enumerate(values):
                        flat[name]["counters"][j] += v
        total = sum(t for p, t in node_time.items() if len(p) == 1)
        wall = sum(step_wall) / max(n_steps, 1)
        calls = {p: (len(ids) / n_steps if None not in ids else float("nan")) for p, ids in node_calls.items()}
        data[label] = (node_time, node_count, counters, total, calls, flat, n_steps)
        print(f"{label}: {n_steps} steps, {1e3 * wall:.1f} ms/step of wall time (profiled, "
              f"synchronized at the end of each step), {1e3 * total:.1f} ms/step of device time")
    labels = [label for label, _, _ in trees]

    # Depth-first order, with the children of each component sorted by their largest time
    children = collections.defaultdict(set)
    for label in labels:
        for p in data[label][0]:
            children[p[:-1]].add(p)
    def largest_time(p):
        return max(data[label][0].get(p, 0.0) for label in labels)
    order = []
    def visit(p):
        for c in sorted(children[p], key=lambda c: -largest_time(c)):
            order.append(c)
            visit(c)
    visit(())

    header = ["Component"] + [f"{l} ms/step (share)" for l in labels]
    if len(labels) == 2:
        header.append(f"{labels[0]} vs {labels[1]}")
    header += [f"{l} kernels/step" for l in labels] + [f"{l} calls/step" for l in labels]
    if has_ncu:
        header += [f"{l} {name}" for _, name, _ in COUNTERS for l in labels]
    print()
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for p in order:
        times = [data[l][0].get(p, 0.0) for l in labels]
        if max(times) < 1e-5:
            continue
        cells = [f"{1e3 * t:.2f} ({100 * t / data[l][3]:.1f}%)" for l, t in zip(labels, times)]
        if len(labels) == 2:
            cells.append(f"{100 * (times[0] / times[1] - 1):+.0f}%" if times[1] > 0 else "new")
        cells += [f"{data[l][1].get(p, 0.0):.1f}" for l in labels]
        cells += [f"{data[l][4].get(p, float('nan')):.1f}" for l in labels]
        if has_ncu:
            for j in range(len(COUNTERS)):
                cells += [f"{data[l][2][p][j]:.1f}" if p in data[l][2] else "" for l in labels]
        print("| " + "&nbsp;&nbsp;" * (len(p) - 1) + p[-1] + " | " + " | ".join(cells) + " |")

    # Every marked component, with the time of all its calls (inclusive) and of the kernels
    # that it launches outside other marked components (exclusive)
    names = sorted({n for l in labels for n in data[l][5]}, key=lambda n: -max(data[l][5][n]["incl"] if n in data[l][5] else 0 for l in labels))
    header = ["Component (all calls)"] + [f"{l} ms/step (share)" for l in labels]
    if len(labels) == 2:
        header.append(f"{labels[0]} vs {labels[1]}")
    header += [f"{l} exclusive ms/step" for l in labels] + [f"{l} calls/step" for l in labels]
    header += [f"{l} kernels/step" for l in labels]
    if has_ncu:
        header += [f"{l} {name}" for _, name, _ in COUNTERS for l in labels]
    print()
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    empty = {"incl": 0.0, "excl": 0.0, "kernels": 0.0, "calls": set(), "counters": [0.0] * len(COUNTERS)}
    for n in names:
        rows = [data[l][5].get(n, empty) for l in labels]
        cells = [f"{1e3 * r['incl']:.2f} ({100 * r['incl'] / data[l][3]:.1f}%)" for l, r in zip(labels, rows)]
        if len(labels) == 2:
            cells.append(f"{100 * (rows[0]['incl'] / rows[1]['incl'] - 1):+.0f}%" if rows[1]["incl"] > 0 else "new")
        cells += [f"{1e3 * r['excl']:.2f}" for r in rows]
        cells += [f"{len(r['calls']) / data[l][6]:.1f}" for l, r in zip(labels, rows)]
        cells += [f"{r['kernels']:.1f}" for r in rows]
        if has_ncu:
            for j in range(len(COUNTERS)):
                cells += [f"{r['counters'][j]:.1f}" for r in rows]
        print("| " + n + " | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
