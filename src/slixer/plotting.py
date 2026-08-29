"""Matplotlib plotting for SimulatedSpectrum (requires the [plot] extra)."""


def plot_spectrum(obs, ax=None, seed=0, show_model=True):
    import matplotlib.pyplot as plt
    try:
        plt.style.use("gks")
    except OSError:
        pass

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))

    if show_model:
        ax.plot(obs.model_wavelength, obs.model_depth * 100.0,
                lw=0.8, alpha=0.7, label=f"model ({obs.scenario})")
    points = obs.simulate_points(seed=seed)
    ax.errorbar(obs.wavelength, points * 100.0, yerr=obs.depth_err * 100.0,
                fmt="o", ms=3.5, lw=1, capsize=2,
                label=f"simulated (R={obs.R})")
    ax.set_xscale("log")
    ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.7])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Transit depth (%)")
    ax.set_title(f"{obs.planet} — {obs.scenario}, "
                 f"{obs.n_transits} transit{'s' if obs.n_transits > 1 else ''}")
    ax.legend(frameon=False)
    return ax
