# Vega->AB corrected magnitudes AND saturation judged on charge at last read
import warnings; warnings.filterwarnings("ignore")
import numpy as np, sys
sys.path.insert(0,"precompute")
import make_noise_tables as M
from lazuli_exo import tables
from lazuli_exo.targets import PLANET_NAMES, get_target
from slicersim.lazuli import LazuliTarget
DAB=0.899
for name in PLANET_NAMES:
    t=get_target(name); wl,f,_=tables.load_star(M.TABLES,name)
    tg=LazuliTarget(wl*1e4,f,mag=t["J_mag"]+DAB,band="2massj",instrument="lazuli_cbe")
    sim=tg.simulation; det=sim.detector; sat0=det.meta["saturation"]
    n,m,d=det.nmd
    def run(nmd):
        tg.change_detector(nmd=nmd)
        det.meta["saturation"]=sat0*det.integration_time/det.exposure_time
        s=np.asarray(sim.get_snr(),float); ns=det.nsaturated_detpx or 0
        det.meta["saturation"]=sat0; return s,ns
    s,ns=run((n,m,d))
    while ns and n>2: n=max(2,n//2); s,ns=run((n,m,d))
    while ns and m>1: m=max(1,m//2); s,ns=run((n,m,d))
    err=np.sqrt(2)/(s*np.sqrt(t["t14_s"]/sim.observing_time))
    _,e0,mt=tables.load_noise(M.TABLES,name)
    print(f"{name:14s} nmd={(n,m,d)} nsat(total charge)={ns:5d}  err={np.nanmedian(err)*1e6:6.0f} ppm  vs table {np.median(e0)*1e6:5.0f} ppm  ratio {np.nanmedian(err)/np.median(e0):.2f}")
