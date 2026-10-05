import numpy as np, warnings, sys; warnings.filterwarnings("ignore")
sys.path.insert(0,"src")
import batman
from lazuli_exo import tables
from lazuli_exo.lightcurve import transit_flux
from lazuli_exo.targets import TARGETS
worst=[]
for n,t in sorted(TARGETS.items()):
    wl,u1,u2,m=tables._read(str(tables.resolve_tables_dir(None) / "ld" / f"{t['slug']}.npz"),"wavelength_um","u1","u2")
    k=np.sqrt(t["depth"]); e=t["ecc"]; w=t["w_deg"]
    p=batman.TransitParams(); p.t0=0; p.per=t["period_d"]; p.rp=k; p.a=t["a_over_rs"]; p.inc=t["inc_deg"]; p.ecc=e; p.w=w; p.limb_dark="quadratic"
    tt=np.linspace(-t["t14_s"],t["t14_s"],801)/86400
    # z(t) as batman computes (true Keplerian), then feed our solver the same z
    worstd=0; lddiag=[]
    for i in range(0,len(wl),6):
        p.u=[u1[i],u2[i]]
        mb=batman.TransitModel(p,tt); fb=mb.light_curve(p)
        z=mb.ds
        ft=transit_flux(z,k,u1[i],u2[i])
        worstd=max(worstd,np.max(np.abs(ft-fb)))
    # geometry check: terminator's z(t) vs batman Keplerian z
    from lazuli_exo.lightcurve import light_curves
    import terminator
    o=terminator.observe(n,"flat",R=30); lc=o.light_curves(1,seed=0,max_points=801)
    p.u=[float(lc.u1[0]),float(lc.u2[0])]; p.rp=float(np.sqrt(lc.depth[0]))
    mb=batman.TransitModel(p,lc.time_hr/24); fb=mb.light_curve(p)
    gd=np.max(np.abs(lc.flux_model[0]-fb))
    # duration from batman model vs registry
    intr=lc.time_hr[fb<1-1e-9]; dur=(intr.max()-intr.min())*3600 if intr.size else 0
    # LD physicality
    mu=np.linspace(0,1,201); I=1-u1[:,None]*(1-mu)-u2[:,None]*(1-mu)**2
    bad_pos=int(np.sum((I<0).any(1))); bad_mono=int(np.sum((np.diff(I,axis=1)<0).any(1)))
    print(f"{n:14s} solver|Δ|max={worstd*1e6:6.2f} ppm  geometry|Δ|max={gd*1e6:7.1f} ppm (depth {lc.depth[0]*1e6:.0f})  batman t14/registry={dur/t['t14_s']:.3f}  LD chunks I<0:{bad_pos} nonmono:{bad_mono} u1+u2 max {np.max(u1+u2):.3f}")
