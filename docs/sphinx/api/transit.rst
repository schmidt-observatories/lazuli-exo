Transit models (``lazuli_exo.transit``)
=======================================

Instrument-agnostic transit light-curve models and the NASA Exoplanet Archive
cache, formerly the standalone ``lazuli-transit`` package. An instrument
simulator (the WCC ETC, the IFS) consumes a :class:`~lazuli_exo.transit.FluxModel`
through ``relative_flux(time)``; nothing here imports an instrument package.

Needs the ``transit`` extra (astropy, pandas, jaxoplanet); the archive download
needs ``archive`` (astroquery); :class:`~lazuli_exo.transit.MoonTransitModel`
needs ``pandora`` (pandoramoon).

.. automodule:: lazuli_exo.transit
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: lazuli_exo.transit.models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: lazuli_exo.transit.archive
   :members:
   :undoc-members:

.. automodule:: lazuli_exo.transit.plotting
   :members:
   :undoc-members:
