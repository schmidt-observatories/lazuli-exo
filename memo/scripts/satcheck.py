import warnings, sys, time; warnings.filterwarnings("ignore")
import numpy as np
from slicersim.lazuli import LazuliTarget
from lazuli_exo import tables
from lazuli_exo.targets import PLANET_NAMES, get_target
T=str(tables.resolve_tables_dir(None))
names = sys.argv[1:] or PLANET_NAMES
for name in names:
    t0=time.time()
    t=get_target(name); wl,f,_=tables.load_star(T,name)
    _,_,mn=tables.load_noise(T,name)
    tg=LazuliTarget(wl*1e4,f,mag=t["J_mag"],band="2massj",instrument="lazuli_cbe")
    sim=tg.simulation
    tg.change_detector(nmd=tuple(mn["nmd_used"]))
    snr=np.asarray(sim.get_snr(),float); ns=sim.detector.nsaturated_detpx
    n,m,d=sim.detector.nmd
    frac=sim.detector.integration_time/sim.detector.exposure_time
    sat0=sim.detector.meta["saturation"]
    sim.detector.meta["saturation"]=sat0*frac   # charge at last read ~ exposure_time, not integration_time
    snr2=np.asarray(sim.get_snr(),float); ns2=sim.detector.nsaturated_detpx
    sim.detector.meta["saturation"]=sat0
    # optimal-efficiency alternative with same total frames: (nframes,1,0)
    nf=sim.detector.nframes
    alt=None
    if nf>=2 and (n,m)!=(nf,1):
        tg.change_detector(nmd=(nf,1,0))
        sim.detector.meta["saturation"]=sat0*sim.detector.integration_time/sim.detector.exposure_time
        s3=np.asarray(sim.get_snr(),float); ns3=sim.detector.nsaturated_detpx
        sim.detector.meta["saturation"]=sat0
        alt=(nf, ns3, np.nanmedian(s3/np.sqrt(sim.observing_time))/np.nanmedian(snr/np.sqrt(nf*sim.detector.tframe)))
    print(f"{name:14s} nmd={mn['nmd_used']} nsat(slicersim)={ns} nsat(total-charge)={ns2} "
          f"int/exp={frac:.2f} alt(N,1,0)={alt}  [{time.time()-t0:.0f}s]", flush=True)
