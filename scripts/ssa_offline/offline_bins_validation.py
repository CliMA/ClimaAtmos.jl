#!/usr/bin/env python3
"""Validation figures for the offline many-bin sea-salt column.

usage: offline_bins_validation.py --online <online_diag_dir> --off2 <offline_2bin.nc>
                                  --off50 <offline_50bin.nc> --edges2 <2bin.toml>
                                  --edges50 <50bin.toml> --out <fig_dir>

online_diag_dir: ClimaAtmos output_active dir of the ONLINE 2-bin (+SSLTH) run
                 (NetCDF diagnostics <short>_1h_inst.nc / <short>_10m_inst.nc).
offline nc:      written by offline_bins.jl (mmrSSLTxx, upSSLTxx, mmrSSH*, rho ...).
"""
import argparse, glob, os, re, tomllib
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ap = argparse.ArgumentParser()
ap.add_argument("--online", required=True); ap.add_argument("--off2", required=True)
ap.add_argument("--off50", required=True); ap.add_argument("--edges2", required=True)
ap.add_argument("--edges50", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--online_bins_only", default=None, help="output_active dir of the bins-only online run (optional)")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
HIST = ["SSHAGE", "SSHUEM", "SSHSET", "SSHSET2", "SSHWET", "SSHINC"]


# --------------------------------------------------------------------------- loaders
def load_online(d, short, period="10m"):
    fs = sorted(glob.glob(os.path.join(d, f"{short}_{period}_*.nc")))
    if not fs:
        fs = sorted(glob.glob(os.path.join(d, f"{short}_1h_*.nc")))
    if not fs:
        return None, None, None
    ds = nc.Dataset(fs[0]); v = ds.variables[short]
    a = np.squeeze(np.array(v[:], dtype=float)); t = np.array(ds.variables["time"][:], dtype=float)
    z = np.array(ds.variables["z"][:], dtype=float) if "z" in ds.variables else None
    if a.ndim == 2 and a.shape[0] != len(t):
        a = a.T
    return a, t, z


class Offline:
    def __init__(self, fn):
        self.ds = nc.Dataset(fn)
        self.t = np.array(self.ds.variables["time"][:], dtype=float)
        self.z = np.array(self.ds.variables["z"][:], dtype=float)
        self.rho = self.var("rho")
        self.bins = sorted(k[3:] for k in self.ds.variables if re.match(r"mmrSSLT\d+$", k))
        self.n = len(self.bins)

    def var(self, name):  # -> (time, z): NCDatasets wrote ("z","time"), netCDF4 reads it as (time, z)
        return np.array(self.ds.variables[name][:], dtype=float)

    def mmr(self, b): return self.var("mmr" + b)
    def up(self, b): return self.var("up" + b)
    def mmr_all(self):  # (time, z, nbins)
        return np.stack([self.mmr(b) for b in self.bins], axis=-1)


def edges_um(toml_file):
    with open(toml_file, "rb") as f:
        return np.array(tomllib.load(f)["ssa_size_bin_divisions"]["value"]) * 1e6


def dz_of(z):
    e = np.concatenate([[0.0], 0.5 * (z[1:] + z[:-1]), [z[-1] + 0.5 * (z[-1] - z[-2])]])
    return np.diff(e)


def burden(mmr, rho, z):  # kg m-2 from (time,z) mmr and rho
    return np.sum(mmr * rho * dz_of(z)[None, :], axis=1)


off2, off50 = Offline(A.off2), Offline(A.off50)
e2, e50 = edges_um(A.edges2), edges_um(A.edges50)
r50 = np.sqrt(e50[1:] * e50[:-1]); dlogr50 = np.diff(np.log10(e50))
split = int(np.argmin(np.abs(e50 - e2[1])))  # index of the 1 um edge in the 50-bin edges
print(f"offline 2-bin: {off2.n} bins, {len(off2.t)} outputs; offline 50-bin: {off50.n} bins; split index {split} (edge {e50[split]:.3g} um)")
nan_report = []
for name, o in (("off2", off2), ("off50", off50)):
    M = o.mmr_all()
    nan_report.append(f"{name}: NaN={int(np.isnan(M).sum())} neg={int((M < 0).sum())} min={np.nanmin(M):.3e} max={np.nanmax(M):.3e}")
print("\n".join(nan_report))

on_b1, t_on, z_on = load_online(A.online, "mmrsslt01")
on_b2, _, _ = load_online(A.online, "mmrsslt02")
on_rho = None
ta, _, _ = load_online(A.online, "ta", "1h")  # only for a reference profile panel
# online density is not output; use the offline-copied rho at matching times (identical by construction)

# --------------------------------------------------------------------------- Fig 1: offline-2bin vs online-2bin
def nearest(tt, t):
    return np.array([np.argmin(np.abs(tt - x)) for x in t])


fig, axs = plt.subplots(2, 3, figsize=(15, 8))
for k, (b, on) in enumerate((("SSLT01", on_b1), ("SSLT02", on_b2))):
    ax = axs[0, k]
    mo = off2.mmr(b)
    io = nearest(off2.t, t_on) if on is not None else None
    if on is not None:
        rho_on = off2.rho[io]
        ax.plot(t_on / 3600, burden(on, rho_on, z_on) * 1e6, "k-", lw=3, alpha=0.4, label="online (implicit, in-Jacobian)")
    ax.plot(off2.t / 3600, burden(mo, off2.rho, off2.z) * 1e6, "r--", label="offline 2-bin (driven, explicit)")
    ax.set_xlabel("t [h]"); ax.set_ylabel("column burden [mg m$^{-2}$]"); ax.set_title(f"{b} ({e2[k]:.2g}-{e2[k+1]:.2g} µm)"); ax.legend(fontsize=8)
    ax = axs[1, k]
    for th, c in ((1, "C0"), (3, "C1"), (6, "C2")):
        j = np.argmin(np.abs(off2.t / 3600 - th))
        ax.plot(mo[j] * 1e9, off2.z, "--", color=c, label=f"offline {th} h")
        if on is not None:
            jo = np.argmin(np.abs(t_on / 3600 - th)); ax.plot(on[jo] * 1e9, z_on, "-", color=c, lw=3, alpha=0.35, label=f"online {th} h")
    ax.set_xlabel("mmr [µg kg$^{-1}$]"); ax.set_ylabel("z [m]"); ax.set_ylim(0, 3000); ax.legend(fontsize=7)
ax = axs[0, 2]
if on_b1 is not None:
    for b, on in (("SSLT01", on_b1), ("SSLT02", on_b2)):
        io = nearest(off2.t, t_on); mo = off2.mmr(b)[io]
        bo, bn = burden(on, off2.rho[io], z_on), burden(mo, off2.rho[io], off2.z)
        ax.plot(t_on / 3600, 100 * (bn - bo) / np.maximum(bo, 1e-300), label=b)
    ax.axhline(0, color="k", lw=0.5); ax.set_ylim(-20, 20)
ax.set_xlabel("t [h]"); ax.set_ylabel("burden error offline-online [%]"); ax.set_title("splitting + explicit sub-step error"); ax.legend()
ax = axs[1, 2]
M50 = off50.mmr_all()
s1, s2 = M50[..., :split].sum(-1), M50[..., split:].sum(-1)
ax.plot(off50.t / 3600, burden(s1, off50.rho, off50.z) * 1e6, "b-", label=f"{off50.n}-bin, sub-bins < 1 µm")
ax.plot(off50.t / 3600, burden(s2, off50.rho, off50.z) * 1e6, "g-", label=f"{off50.n}-bin, sub-bins > 1 µm")
ax.plot(off2.t / 3600, burden(off2.mmr("SSLT01"), off2.rho, off2.z) * 1e6, "b--", label="2-bin SSLT01")
ax.plot(off2.t / 3600, burden(off2.mmr("SSLT02"), off2.rho, off2.z) * 1e6, "g--", label="2-bin SSLT02")
ax.set_xlabel("t [h]"); ax.set_ylabel("column burden [mg m$^{-2}$]"); ax.set_title("bin-resolution dependence of the two-bin burdens"); ax.legend(fontsize=8)
fig.suptitle("Offline driven column vs online run (Bomex, 1M, 2 bins)")
fig.tight_layout(); fig.savefig(os.path.join(A.out, "fig1_offline_vs_online_2bin.png"), dpi=140); plt.close(fig)

# --------------------------------------------------------------------------- Fig 2: 50-bin spectra
fig, axs = plt.subplots(1, 3, figsize=(16, 5))
levels = [(0, "surface layer"), (np.argmin(np.abs(off50.z - 500)), "z≈500 m (sub-cloud)"), (np.argmin(np.abs(off50.z - 1200)), "z≈1200 m (cloud layer)")]
for ax, (k, lab) in zip(axs, levels):
    for th, c in ((0.5, "C0"), (1, "C1"), (2, "C2"), (4, "C3"), (6, "C4")):
        j = np.argmin(np.abs(off50.t / 3600 - th))
        ax.plot(r50, M50[j, k, :] / dlogr50 * 1e9, "-o", ms=2, color=c, label=f"{th} h")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("dry radius [µm]"); ax.set_ylabel("dM/dlog$_{10}$r [µg kg$^{-1}$]")
    ax.axvline(e2[1], color="k", lw=0.5, ls=":"); ax.set_title(lab); ax.legend(fontsize=8)
    ax.set_ylim(bottom=max(1e-6, np.nanmax(M50 / dlogr50 * 1e9) * 1e-6))
fig.suptitle(f"Offline {off50.n}-bin mass spectrum evolution (Bomex)")
fig.tight_layout(); fig.savefig(os.path.join(A.out, f"fig2_spectra_{off50.n}bin.png"), dpi=140); plt.close(fig)

# --------------------------------------------------------------------------- Fig 3: normalized spectra shape (size memory)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
ax = axs[0]
for th, c in ((0.5, "C0"), (1, "C1"), (2, "C2"), (4, "C3"), (6, "C4")):
    j = np.argmin(np.abs(off50.t / 3600 - th))
    for k, ls in ((0, "-"), (levels[2][0], "--")):
        s = M50[j, k, :]; s = s / max(s.sum(), 1e-300) / dlogr50
        ax.plot(r50, s, ls, color=c, label=f"{th} h, {'sfc' if k == 0 else 'cloud'}")
ax.set_xscale("log"); ax.set_xlabel("dry radius [µm]"); ax.set_ylabel("normalized dM/dlog r"); ax.set_title("spectrum shape: surface vs cloud layer"); ax.legend(fontsize=7, ncol=2)
ax = axs[1]
j = np.argmin(np.abs(off50.t / 3600 - 6))
Z = M50[j] / dlogr50[None, :] * 1e9
im = ax.pcolormesh(r50, off50.z, np.maximum(Z, 1e-12), norm=LogNorm(vmin=max(1e-6, Z.max() * 1e-5), vmax=Z.max()), shading="nearest", cmap="viridis")
ax.set_xscale("log"); ax.set_ylim(0, 3000); ax.set_xlabel("dry radius [µm]"); ax.set_ylabel("z [m]"); ax.set_title("dM/dlog r at 6 h [µg kg$^{-1}$]")
fig.colorbar(im, ax=ax)
fig.tight_layout(); fig.savefig(os.path.join(A.out, "fig3_spectrum_shape_and_height.png"), dpi=140); plt.close(fig)

# --------------------------------------------------------------------------- Fig 4: history tracers, offline vs online
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
M2 = off2.mmr("SSLT01") + off2.mmr("SSLT02")
Mon = (on_b1 + on_b2) if on_b1 is not None else None
for ax, h in zip(axs.flat, HIST):
    def ratio(hh, MM):  # tracer per unit mass, masked where the mass is < 0.1 % of the column max
        return np.where(MM > 1e-3 * np.nanmax(MM, axis=1, keepdims=True), hh / np.maximum(MM, 1e-300), np.nan)
    ho = off2.mmr(h); r_off = ratio(ho, M2)
    h50 = off50.mmr(h); M50s = M50.sum(-1); r_50 = ratio(h50, M50s)
    scale = 1 / 3600 if h == "SSHAGE" else 1.0
    unit = "[h]" if h == "SSHAGE" else ("[m s$^{-1}$]" if h == "SSHUEM" else "[-]")
    for th, c in ((2, "C1"), (6, "C2")):
        j = np.argmin(np.abs(off2.t / 3600 - th)); ax.plot(r_off[j] * scale, off2.z, "--", color=c, label=f"offline 2-bin {th} h")
        j = np.argmin(np.abs(off50.t / 3600 - th)); ax.plot(r_50[j] * scale, off50.z, ":", color=c, label=f"offline {off50.n}-bin {th} h")
        hon, ton, zon = load_online(A.online, "mmr" + h.lower())
        if hon is not None and Mon is not None:
            jo = np.argmin(np.abs(ton / 3600 - th)); r_on = ratio(hon[jo][None, :], Mon[jo][None, :])[0]
            ax.plot(r_on * scale, zon, "-", color=c, label=f"online {th} h")
    ax.set_ylim(0, 3000); ax.set_xlabel(f"{h}/M {unit}"); ax.set_ylabel("z [m]"); ax.set_title(h); ax.legend(fontsize=7)
fig.suptitle(f"History tracers per unit sea-salt mass: online vs offline (2 and {off50.n} bins)")
fig.tight_layout(); fig.savefig(os.path.join(A.out, "fig4_history_tracers.png"), dpi=140); plt.close(fig)

# --------------------------------------------------------------------------- Fig 5: effective radius / moments from the 50 bins
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# mass-weighted mean log radius and its spread, per height and time
lr = np.log10(r50)
w = np.maximum(M50, 0); W = w.sum(-1)
W = np.where(W > 1e-3 * np.nanmax(W, axis=1, keepdims=True), W, 0.0)   # mask cells with < 0.1 % of the column max
mean_lr = (w * lr).sum(-1) / np.maximum(W, 1e-300); sd_lr = np.sqrt((w * (lr[None, None, :] - mean_lr[..., None]) ** 2).sum(-1) / np.maximum(W, 1e-300))
ax = axs[0]
for th, c in ((0.5, "C0"), (1, "C1"), (2, "C2"), (4, "C3"), (6, "C4")):
    j = np.argmin(np.abs(off50.t / 3600 - th)); ok = W[j] > 1e-30
    ax.plot(10 ** mean_lr[j][ok], off50.z[ok], color=c, label=f"{th} h")
ax.set_xscale("log"); ax.set_ylim(0, 3000); ax.set_xlabel("mass-weighted geometric mean dry radius [µm]"); ax.set_ylabel("z [m]"); ax.legend(); ax.set_title("mean size vs height")
ax = axs[1]
for th, c in ((0.5, "C0"), (1, "C1"), (2, "C2"), (4, "C3"), (6, "C4")):
    j = np.argmin(np.abs(off50.t / 3600 - th)); ok = W[j] > 1e-30
    ax.plot(sd_lr[j][ok], off50.z[ok], color=c, label=f"{th} h")
ax.set_ylim(0, 3000); ax.set_xlabel("std of log$_{10}$ r (mass-weighted)"); ax.set_ylabel("z [m]"); ax.legend(); ax.set_title("spectral width vs height")
fig.tight_layout(); fig.savefig(os.path.join(A.out, "fig5_size_moments.png"), dpi=140); plt.close(fig)

with open(os.path.join(A.out, "summary.txt"), "w") as f:
    f.write("\n".join(nan_report) + "\n")
    if on_b1 is not None:
        io = nearest(off2.t, t_on)
        for b, on in (("SSLT01", on_b1), ("SSLT02", on_b2)):
            bo = burden(on, off2.rho[io], z_on); bn = burden(off2.mmr(b)[io], off2.rho[io], off2.z)
            f.write(f"{b}: burden at 6 h online {bo[-1]:.4e} offline {bn[-1]:.4e} kg/m2, rel err {(bn[-1]-bo[-1])/bo[-1]*100:.2f}%\n")
    b1 = burden(s1, off50.rho, off50.z)[-1]; b2 = burden(s2, off50.rho, off50.z)[-1]
    c1 = burden(off2.mmr("SSLT01"), off2.rho, off2.z)[-1]; c2 = burden(off2.mmr("SSLT02"), off2.rho, off2.z)[-1]
    f.write(f"50-bin vs 2-bin burdens at end: sub-um {b1:.4e} vs {c1:.4e} ({(b1-c1)/c1*100:+.2f}%), super-um {b2:.4e} vs {c2:.4e} ({(b2-c2)/c2*100:+.2f}%)\n")
print(open(os.path.join(A.out, "summary.txt")).read())
print("figures in", A.out)
