import warnings; warnings.filterwarnings("ignore")
import numpy as np, sys, sncosmo
sys.path.insert(0,"precompute")
import make_noise_tables as M
from lazuli_exo import tables
from lazuli_exo.targets import PLANET_NAMES, get_target
from slicersim.lazuli import LazuliTarget
from lazuli_exo.slicer import _desaturate
dAB=-2.5*np.log10(sncosmo.get_magsystem('vega').zpbandflux('2massj')/sncosmo.get_magsystem('ab').zpbandflux('2massj'))
print(f"J_AB - J_Vega = {dAB:.3f}")
for name in PLANET_NAMES:
    t=get_target(name); wl,f,_=tables.load_star(M.TABLES,name)
    out=[]
    for mag in (t["J_mag"], t["J_mag"]+dAB):
        tg=LazuliTarget(wl*1e4,f,mag=mag,band="2massj",instrument="lazuli_cbe")
        snr,nmd,ns=_desaturate(tg); sim=tg.simulation
        err=np.sqrt(2)/(snr*np.sqrt(t["t14_s"]/sim.observing_time))
        out.append((nmd,ns,np.nanmedian(err)))
    (n0,s0,e0),(n1,s1,e1)=out
    print(f"{name:14s} as-built(J as AB): nmd={n0} nsat={s0:5d} err={e0*1e6:6.0f} ppm | corrected(Vega->AB): nmd={n1} nsat={s1:5d} err={e1*1e6:6.0f} ppm  ratio {e1/e0:.2f}")
