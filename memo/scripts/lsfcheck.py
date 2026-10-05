import numpy as np, warnings; warnings.filterwarnings("ignore")
import lazuli_exo as terminator
from lazuli_exo import tables, binning
from lazuli_exo.detect import detect_features
# Lazuli IFS R_instr(lambda) from slicersim lazuli_cbe (sampled at 7 pts)
wR=np.array([0.4,0.5,0.6,0.8,1.0,1.3,1.6,1.7]); RR=np.array([369,232,166,112,101,115,150,160])
def lsf(wl,d):
    lnw=np.log(wl); dl=lnw[1]-lnw[0]
    sig=1/np.interp(wl,wR,RR)/2.3548/dl   # sigma in native pixels (log grid)
    out=np.empty_like(d)
    for i in range(len(d)):
        s=sig[i]; h=int(5*s)+1; j=np.arange(max(0,i-h),min(len(d),i+h+1))
        k=np.exp(-0.5*((j-i)/s)**2); out[i]=np.sum(k*d[j])/k.sum()
    return out
def overlap_bin(wl,d,edges,wt=None):
    # flux-conserving (per unit wavelength) with optional photon weights
    e=np.concatenate([[wl[0]],0.5*(wl[1:]+wl[:-1]),[wl[-1]]]); dw=np.diff(e)
    w=dw*(1 if wt is None else wt)
    c=np.concatenate([[0],np.cumsum(w*d)]); cw=np.concatenate([[0],np.cumsum(w)])
    # interpolate cumulative at edges
    return np.diff(np.interp(edges,e,c))/np.diff(np.interp(edges,e,cw))
for planet,sc in [("WASP-39 b","clear"),("WASP-107 b","clear"),("GJ 1214 b","clear"),("K2-18 b","clear")]:
    wl,d,_=tables.load_model(str(tables.resolve_tables_dir(None)),planet,sc)
    ws,fs,_=tables.load_star(str(tables.resolve_tables_dir(None)),planet)
    phot=np.interp(wl,ws,fs)*wl   # photon counts ~ lambda F_lambda (no throughput)
    dl=lsf(wl,d)
    for R in [30,100,200]:
        o=terminator.observe(planet,sc,n_transits=10,R=R)
        edges=binning.constant_R_edges(o.wavelength[0]**2/ o.wavelength[0], 1, R) if False else None
        wn,en,_=tables.load_noise(str(tables.resolve_tables_dir(None)),planet); edges=binning.constant_R_edges(wn.min(),wn.max(),R)
        b0=o.depth; b1=overlap_bin(wl,d,edges,phot); b2=overlap_bin(wl,dl,edges,phot)
        s0=detect_features(o.wavelength,b0,o.depth_err)["template_snr_diagonal"]
        s2=detect_features(o.wavelength,b2,o.depth_err)["template_snr_diagonal"]
        r1=np.abs(b1-b0)/o.depth_err; r2=np.abs(b2-b0)/o.depth_err
        print(f"{planet:11s} R={R:3d} max|photon-wt overlap - mean| = {np.max(np.abs(b1-b0))*1e6:5.0f} ppm ({r1.max():.2f} sig@10tr) | +LSF: {np.max(np.abs(b2-b0))*1e6:5.0f} ppm ({r2.max():.2f} sig), SNR {s0:.1f} -> {s2:.1f}")
