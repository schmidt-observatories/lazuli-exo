Installation
============

Requirements
------------

- **Python 3.10 or newer.**
- **numpy 2.0 or newer**, the only hard runtime dependency.

Optional extras:

============  =====================================================
Extra         Pulls in / needed for
============  =====================================================
``plot``      ``matplotlib`` — the ``.plot()`` methods (:mod:`lazuli_exo.plotting`)
``test``      ``pytest`` — the test suite
``web``       ``flask`` — the local Plotly demo UI under ``webapp/``
``docs``      Sphinx, nbsphinx, the RTD theme — building these docs
``transit``   ``astropy``, ``pandas``, ``jaxoplanet`` — :mod:`lazuli_exo.transit` models
``archive``   ``astroquery`` — :func:`~lazuli_exo.transit.download_exoplanet_archive`
============  =====================================================

The parametric model (:func:`lazuli_exo.simulate`) additionally needs
``slicersim`` and ``sncosmo`` *at call time* (the noise is computed live).
They are not declared as dependencies because slicersim is not on PyPI;
install it from its repository into the same environment.

Nothing heavier is ever imported at runtime. POSEIDON, slicersim, jax, and
astropy are needed only to *regenerate* the lookup tables (see
:doc:`user_guide/tables`), never to use them.

Install from PyPI
-----------------

.. code-block:: bash

   pip install lazuli-exo                      # numpy + the lookup tables (~35 MB)
   pip install "lazuli-exo[plot]"              # matplotlib for the .plot() methods
   pip install "lazuli-exo[transit,archive]"   # transit models, Exoplanet Archive cache

Install from source
-------------------

Clone the repository and install in *editable* mode so local changes to
``src/`` are picked up immediately:

.. code-block:: bash

   git clone git@github.com:schmidt-observatories/lazuli-exo.git
   cd lazuli-exo
   pip install -e ".[plot,test]"

Verify the installation
-----------------------

.. code-block:: python

   import lazuli_exo

   print(lazuli_exo.list_planets())          # 14 showcase targets

   obs = lazuli_exo.observe("WASP-39 b", scenario="clear", n_transits=1)
   print(obs.detect()["template_snr_diagonal"])

If that prints a list of 14 planets and a number around 69, you are ready to
go. Continue with :doc:`quickstart`.

Where the tables come from
--------------------------

The runtime reads a directory of ``.npz`` lookup tables (about 50 MB,
including the parametric :math:`\zeta` and PHOENIX grids), committed at
``src/lazuli_exo/data/tables/`` and declared as package data, so editable
installs, VCS pins and wheels all ship them. ``observe()`` resolves it in this
order:

1. an explicit ``tables_dir=`` argument,
2. the ``LAZULI_EXO_TABLES`` environment variable,
3. the ``data/tables/`` directory inside the installed package.

Running the test suite
----------------------

.. code-block:: bash

   pip install -e ".[plot,test]"
   pytest

Building the documentation locally
----------------------------------

The docs are built with `Sphinx <https://www.sphinx-doc.org>`_ and the *Read
the Docs* theme. Tutorial pages are rendered from the Jupyter notebooks in
``notebooks/`` via `nbsphinx <https://nbsphinx.readthedocs.io>`_, which
requires a `pandoc <https://pandoc.org>`_ binary.

.. code-block:: bash

   # from the repository root
   pip install -e ".[docs]"     # Sphinx + RTD theme + nbsphinx
   # plus a pandoc binary, e.g.  conda install pandoc   or   brew install pandoc

   cd docs/sphinx
   make html

Open ``docs/sphinx/_build/html/index.html``. Use ``make clean html`` to force
a full rebuild.

.. note::

   The notebooks are **not re-executed** at build time
   (``nbsphinx_execute = "never"``) — the committed outputs are what gets
   rendered. ``conf.py`` copies ``notebooks/0*.ipynb`` into
   ``docs/sphinx/notebooks/`` on each build, so ``notebooks/`` stays the
   single source of truth. To refresh the rendered outputs, re-run the
   notebooks and commit them:

   .. code-block:: bash

      jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
