import numpy as np, math, warnings; warnings.filterwarnings("ignore")
from lazuli_exo import tables, binning
from lazuli_exo.targets import TARGETS
T=str(tables.resolve_tables_dir(None))
RJ,RS,MJ,G,kB,mH=7.1492e7,6.957e8,1.89813e27,6.674e-11,1.380649e-23,1.6735e-27
print("== Registry self-consistency ==")
print(f"{'planet':14s} {'k_rad/k_arch':>12s} {'aRs_Kep/aRs':>11s} {'t14calc/t14':>11s} {'b':>6s} {'Teq_calc/Teq':>12s}")
for n,t in sorted(TARGETS.items()):
    k_arch=math.sqrt(t["depth"]); k_rad=t["R_p_RJ"]*RJ/(t["R_s_RSun"]*RS)
    Ms=10**t["log_g_star"]/100*(t["R_s_RSun"]*RS)**2/G
    a=(G*Ms*(t["period_d"]*86400)**2/(4*math.pi**2))**(1/3); aRs_kep=a/(t["R_s_RSun"]*RS)
    e,w,i,ar=t["ecc"],math.radians(t["w_deg"]),math.radians(t["inc_deg"]),t["a_over_rs"]
    b=ar*math.cos(i)*(1-e*e)/(1+e*math.sin(w))
    arg=math.sqrt(max((1+k_arch)**2-b*b,0))/(ar*math.sin(i))
    t14=t["period_d"]*86400/math.pi*math.asin(min(arg,1))*math.sqrt(1-e*e)/(1+e*math.sin(w))
    teq=t["T_eff_K"]*math.sqrt(1/(2*ar))
    print(f"{n:14s} {k_rad/k_arch:12.3f} {aRs_kep/ar:11.3f} {t14/t['t14_s']:11.3f} {b:6.3f} {teq/t['T_eq_K']:12.3f}")
print("\n== Model tables vs flat/radii; feature amplitude in scale heights ==")
print(f"{'planet':14s} {'scen':16s} {'med/flat':>8s} {'min/kRad2':>9s} {'dR_min/H':>8s} {'ampl/H':>7s} {'ampl ppm':>8s}")
for n,t in sorted(TARGETS.items()):
    Rp=t["R_p_RJ"]*RJ; Rs=t["R_s_RSun"]*RS; g=G*t["M_p_MJ"]*MJ/Rp**2
    Tiso=max(t["T_eq_K"],300.0)
    for sc in ["clear","cloudy","hazy","high_metallicity"]:
        wl,d,m=tables.load_model(T,n,sc)
        sel=(wl>=0.4)&(wl<=1.7); d=d[sel]
        mu=2.3 if sc!="high_metallicity" else None
        H=kB*Tiso/(2.3*mH*g)
        Reff=np.sqrt(d)*Rs
        print(f"{n:14s} {sc:16s} {np.median(d)/t['depth']:8.3f} {d.min()/(Rp/Rs)**2:9.3f} {(Reff.min()-Rp)/H:8.2f} {(Reff.max()-Reff.min())/H:7.2f} {(d.max()-d.min())*1e6:8.0f}")
